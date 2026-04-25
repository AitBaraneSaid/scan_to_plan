"""Visualise une slice XY directement depuis les points du nuage.

Reconstruit la density map à différentes résolutions et teste la binarisation
et Hough directement sur les données brutes, sans passer par une PNG intermédiaire.

Usage :
    python scripts/visualize_slice.py data/T92123_31.e57 \
        --floor-z -0.75 --ceiling-z 2.35 \
        --x-min 2.86 --x-max 17.36 --y-min 1.09 --y-max 16.59
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise une slice et teste Hough.")
    parser.add_argument("input", help="Fichier E57/LAS")
    parser.add_argument("--floor-z", type=float, default=None)
    parser.add_argument("--ceiling-z", type=float, default=None)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--slice-height", type=float, default=1.10,
                        help="Hauteur de slice relative au sol (m)")
    parser.add_argument("--thickness", type=float, default=0.10)
    args = parser.parse_args()

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scan2plan.io.readers import read_point_cloud
    from scan2plan.preprocessing.floor_ceiling import detect_floor_rdc
    from scan2plan.slicing.density_map import create_density_map

    path = Path(args.input)
    print(f"Lecture : {path.name}")
    points = read_point_cloud(path)
    print(f"  {len(points):,} points")

    # Crop XY
    mask = np.ones(len(points), dtype=bool)
    if args.x_min is not None:
        mask &= points[:, 0] >= args.x_min
    if args.x_max is not None:
        mask &= points[:, 0] <= args.x_max
    if args.y_min is not None:
        mask &= points[:, 1] >= args.y_min
    if args.y_max is not None:
        mask &= points[:, 1] <= args.y_max
    points = points[mask]
    print(f"  Après crop XY : {len(points):,} points")

    # Sol/plafond
    if args.floor_z is not None:
        z_floor = args.floor_z
        z_ceiling = args.ceiling_z if args.ceiling_z is not None else z_floor + 3.0
    else:
        z_floor, z_ceiling = detect_floor_rdc(points)
    print(f"  Sol={z_floor:.3f} m  Plafond={z_ceiling:.3f} m")

    # Slice
    z_lo = z_floor + args.slice_height - args.thickness / 2
    z_hi = z_floor + args.slice_height + args.thickness / 2
    mask_z = (points[:, 2] >= z_lo) & (points[:, 2] <= z_hi)
    slice_xy = points[mask_z, :2]
    print(f"  Slice h={args.slice_height}m [{z_lo:.3f}, {z_hi:.3f}] : {len(slice_xy):,} points")

    if len(slice_xy) == 0:
        print("Slice vide.")
        return

    out_dir = path.parent
    stem = path.stem

    # --- Test à plusieurs résolutions ---
    resolutions = [0.02, 0.01, 0.005]  # 2cm, 1cm, 5mm
    fig, axes = plt.subplots(len(resolutions), 4, figsize=(20, 5 * len(resolutions)))

    for row, res in enumerate(resolutions):
        dmap = create_density_map(slice_xy, resolution=res, margin=0.1)
        img = dmap.image
        print(f"\n  Résolution {res*100:.0f}mm : image {img.shape[1]}×{img.shape[0]} px")
        print(f"    Pixels occupés : {(img > 0).sum():,} / {img.size:,} ({(img>0).sum()/img.size*100:.1f}%)")
        print(f"    Densité max : {img.max()}  p50 : {np.percentile(img[img>0], 50):.1f}  p95 : {np.percentile(img[img>0], 95):.1f}")

        # Normaliser pour affichage
        vmax = float(np.percentile(img[img > 0], 95)) if (img > 0).any() else 1.0
        axes[row, 0].imshow(img, cmap="hot", origin="upper", vmin=0, vmax=vmax)
        axes[row, 0].set_title(f"Density map {res*100:.0f}mm/px\n{img.shape[1]}×{img.shape[0]} px")
        axes[row, 0].axis("off")

        # Normaliser en uint8 pour OpenCV
        max_val = float(img.max())
        img8 = np.clip((img.astype(np.float32) / max_val * 255), 0, 255).astype(np.uint8)

        # Otsu
        otsu_t, binary = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        occ_pct = (binary > 0).sum() / binary.size * 100
        print(f"    Otsu seuil={otsu_t:.0f} → {occ_pct:.1f}% occupé")
        axes[row, 1].imshow(binary, cmap="gray", origin="upper")
        axes[row, 1].set_title(f"Otsu (seuil={otsu_t:.0f})\n{occ_pct:.1f}% occupé")
        axes[row, 1].axis("off")

        # Morphologie
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_m = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary_m = cv2.morphologyEx(binary_m, cv2.MORPH_OPEN, kernel, iterations=1)
        axes[row, 2].imshow(binary_m, cmap="gray", origin="upper")
        axes[row, 2].set_title(f"Après morphologie\n{(binary_m>0).sum()/binary_m.size*100:.1f}% occupé")
        axes[row, 2].axis("off")

        # Hough
        overlay = cv2.cvtColor(binary_m, cv2.COLOR_GRAY2BGR)
        min_len = max(5, int(0.20 / res))  # 20cm minimum
        lines = cv2.HoughLinesP(
            binary_m, rho=1, theta=np.deg2rad(0.5),
            threshold=20, minLineLength=min_len, maxLineGap=int(0.10 / res)
        )
        n_lines = 0 if lines is None else len(lines)
        print(f"    Hough (th=20, len={min_len}px={min_len*res*100:.0f}cm) : {n_lines} segments")
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        axes[row, 3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), origin="upper")
        axes[row, 3].set_title(f"Hough th=20 len={min_len}px\n{n_lines} segments")
        axes[row, 3].axis("off")

    plt.suptitle(
        f"Diagnostic slice h={args.slice_height}m — {stem}\n"
        f"Sol={z_floor:.2f}m  XY crop=[{args.x_min},{args.x_max}]×[{args.y_min},{args.y_max}]",
        fontsize=13
    )
    plt.tight_layout()
    out = out_dir / f"{stem}_slice_diagnostic.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure sauvegardée : {out}")


if __name__ == "__main__":
    main()
