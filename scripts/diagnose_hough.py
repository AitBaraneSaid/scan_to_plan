"""Diagnostique pourquoi Hough ne détecte pas de segments.

Relit la density map sauvegardée, teste plusieurs seuillages et paramètres Hough,
et sauvegarde des images de chaque étape pour comprendre où ça bloque.

Usage :
    python scripts/diagnose_hough.py data/T92123_31_density_map_mid.png
    python scripts/diagnose_hough.py data/T92123_31_density_map_mid.npy  (si sauvegardé)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def diagnose(image_path: str) -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(image_path)

    # --- Chargement ---
    if path.suffix == ".npy":
        dmap_raw = np.load(path)
        print(f"Density map chargée depuis .npy : {dmap_raw.shape}  max={dmap_raw.max()}")
    else:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            print(f"Impossible de charger : {path}")
            return
        dmap_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        print(f"Image chargée : {dmap_raw.shape}  max={dmap_raw.max():.1f}")

    # Normaliser en uint8
    max_val = float(dmap_raw.max())
    if max_val == 0:
        print("Image entièrement noire — density map vide.")
        return
    img8 = np.clip((dmap_raw / max_val * 255), 0, 255).astype(np.uint8)

    occupied_pct = float((img8 > 0).sum()) / img8.size * 100
    print(f"Pixels occupés (img8 > 0) : {(img8 > 0).sum():,} / {img8.size:,}  ({occupied_pct:.2f}%)")
    print(f"Percentiles img8 : p50={np.percentile(img8, 50):.1f}  p90={np.percentile(img8, 90):.1f}  p99={np.percentile(img8, 99):.1f}")

    out_dir = path.parent
    stem = path.stem

    # --- Figure 1 : comparaison seuillages ---
    methods = [
        ("Otsu", None),
        ("Percentile 90", 90),
        ("Percentile 95", 95),
        ("Percentile 99", 99),
        ("Fixe 10", "fixed_10"),
        ("Fixe 30", "fixed_30"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flat

    for (label, param), ax in zip(methods, axes_flat):
        if param is None:
            thresh, binary = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            title = f"Otsu (seuil={thresh:.0f})"
        elif isinstance(param, int):
            thresh = float(np.percentile(img8[img8 > 0], param)) if (img8 > 0).any() else 1.0
            _, binary = cv2.threshold(img8, thresh, 255, cv2.THRESH_BINARY)
            title = f"Percentile {param} (seuil={thresh:.0f})"
        else:
            fixed = int(param.split("_")[1])
            _, binary = cv2.threshold(img8, fixed, 255, cv2.THRESH_BINARY)
            title = f"Seuil fixe {fixed}"
            thresh = fixed

        occ = int((binary > 0).sum())
        print(f"  {title} → {occ:,} px occupés ({occ/binary.size*100:.2f}%)")
        ax.imshow(binary, cmap="gray", origin="upper")
        ax.set_title(f"{title}\n{occ:,} px ({occ/binary.size*100:.1f}%)")
        ax.axis("off")

    plt.suptitle(f"Comparaison seuillages — {stem}", fontsize=14)
    plt.tight_layout()
    out1 = out_dir / f"{stem}_binarization_compare.png"
    fig.savefig(str(out1), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure 1 sauvegardée : {out1}")

    # --- Figure 2 : Hough avec différents paramètres sur Otsu ---
    _, binary_otsu = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Fermeture morphologique légère avant Hough
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_morph = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel, iterations=1)

    hough_configs = [
        {"threshold": 10, "min_line_length": 20, "max_line_gap": 5, "label": "th=10 len=20"},
        {"threshold": 20, "min_line_length": 30, "max_line_gap": 10, "label": "th=20 len=30"},
        {"threshold": 30, "min_line_length": 50, "max_line_gap": 15, "label": "th=30 len=50"},
        {"threshold": 50, "min_line_length": 50, "max_line_gap": 20, "label": "th=50 len=50 (défaut)"},
        {"threshold": 10, "min_line_length": 10, "max_line_gap": 5, "label": "th=10 len=10 (agressif)"},
        {"threshold": 5, "min_line_length": 5, "max_line_gap": 3, "label": "th=5 len=5 (max)"},
    ]

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))

    for cfg, ax in zip(hough_configs, axes2.flat):
        overlay = cv2.cvtColor(binary_morph, cv2.COLOR_GRAY2BGR)
        lines = cv2.HoughLinesP(
            binary_morph,
            rho=1,
            theta=np.deg2rad(0.5),
            threshold=cfg["threshold"],
            minLineLength=cfg["min_line_length"],
            maxLineGap=cfg["max_line_gap"],
        )
        n_lines = 0 if lines is None else len(lines)
        print(f"  Hough {cfg['label']} → {n_lines} segments")
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), origin="upper")
        ax.set_title(f"{cfg['label']}\n{n_lines} segments")
        ax.axis("off")

    plt.suptitle(f"Hough probabiliste — {stem}", fontsize=14)
    plt.tight_layout()
    out2 = out_dir / f"{stem}_hough_compare.png"
    fig2.savefig(str(out2), dpi=100, bbox_inches="tight")
    plt.close(fig2)
    print(f"Figure 2 sauvegardée : {out2}")

    # --- Figure 3 : Canny + Hough ---
    edges = cv2.Canny(img8, 30, 100)
    edges_occ = int((edges > 0).sum())
    print(f"\nCanny (30,100) : {edges_occ:,} px d'arêtes ({edges_occ/edges.size*100:.2f}%)")

    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))

    axes3[0].imshow(edges, cmap="gray", origin="upper")
    axes3[0].set_title(f"Canny (30,100) — {edges_occ:,} px")
    axes3[0].axis("off")

    for ax, (thresh, min_len) in zip(axes3[1:], [(20, 20), (10, 10)]):
        overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.deg2rad(0.5),
                                 threshold=thresh, minLineLength=min_len, maxLineGap=10)
        n = 0 if lines is None else len(lines)
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), origin="upper")
        ax.set_title(f"Canny+Hough th={thresh} len={min_len} — {n} seg.")
        ax.axis("off")

    plt.suptitle(f"Canny + Hough — {stem}", fontsize=14)
    plt.tight_layout()
    out3 = out_dir / f"{stem}_canny_hough.png"
    fig3.savefig(str(out3), dpi=100, bbox_inches="tight")
    plt.close(fig3)
    print(f"Figure 3 sauvegardée : {out3}")

    print("\nDiagnostic terminé.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python scripts/diagnose_hough.py <density_map.png|.npy>")
        sys.exit(1)
    diagnose(sys.argv[1])
