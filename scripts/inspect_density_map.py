"""Inspecte une density map et génère une version visible avec zoom."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def inspect(e57_path: str) -> None:
    from scan2plan.config import ScanConfig
    from scan2plan.io.readers import read_point_cloud
    from scan2plan.slicing.density_map import create_density_map
    from scan2plan.slicing.slicer import extract_multi_slices
    from scan2plan.preprocessing.floor_ceiling import detect_floor_rdc

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(e57_path)
    print(f"Lecture : {path.name}")
    points = read_point_cloud(path)
    print(f"  {len(points):,} points  BBox Z=[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")

    # Détection sol/plafond
    z_floor, z_ceiling = detect_floor_rdc(points)
    print(f"  Sol={z_floor:.3f} m  Plafond={z_ceiling:.3f} m")

    # Slice mid à résolution grossière pour visualisation rapide
    resolution = 0.05  # 5 cm/pixel pour voir la structure globale
    cfg = ScanConfig()
    slices = extract_multi_slices(points, heights=cfg.slicing.heights,
                                  thickness=cfg.slicing.thickness, floor_z=z_floor)
    mid = slices.get("mid", np.empty((0, 2)))
    print(f"  Slice mid : {len(mid):,} points")

    if len(mid) == 0:
        print("Slice vide — vérifier les hauteurs de sol/plafond.")
        return

    dmap = create_density_map(mid, resolution)
    img = dmap.image
    print(f"  Density map : {img.shape[1]}×{img.shape[0]} px")
    print(f"  Densité max : {img.max():.1f}  moy : {img[img>0].mean():.2f}  pixels occupés : {(img>0).sum():,}")

    # Sauvegarde avec normalisation correcte
    out = path.parent / f"{path.stem}_density_5cm.png"
    fig, ax = plt.subplots(figsize=(12, 8))
    vmax = float(np.percentile(img[img > 0], 95)) if (img > 0).any() else 1.0
    ax.imshow(img, cmap="hot", origin="upper", vmin=0, vmax=vmax)
    ax.set_title(f"Density map 5 cm/px — {path.stem}\nSol={z_floor:.2f}m Plafond={z_ceiling:.2f}m  max={img.max():.0f} pts/px")
    plt.colorbar(ax.images[0], ax=ax, label="pts/pixel")
    plt.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sauvegardé : {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python scripts/inspect_density_map.py <fichier.e57>")
        sys.exit(1)
    inspect(sys.argv[1])
