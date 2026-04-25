"""Trouve la bounding box XY réelle du bâtiment dans la slice mid."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def find_bbox(e57_path: str) -> None:
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

    z_floor, z_ceiling = detect_floor_rdc(points)
    print(f"Sol={z_floor:.3f} m  Plafond={z_ceiling:.3f} m")

    # Slice mid
    slices = extract_multi_slices(
        points, heights=[2.10, 1.10, 0.20], thickness=0.10, floor_z=z_floor
    )
    mid = slices.get("mid", np.empty((0, 2)))
    if len(mid) == 0:
        print("Slice vide.")
        return

    # BBox XY réelle des points de la slice
    x_min, y_min = mid.min(axis=0)
    x_max, y_max = mid.max(axis=0)
    print(f"\nBBox XY de la slice mid (coordonnées métriques) :")
    print(f"  X : [{x_min:.3f}, {x_max:.3f}] m  (largeur {x_max - x_min:.2f} m)")
    print(f"  Y : [{y_min:.3f}, {y_max:.3f}] m  (profondeur {y_max - y_min:.2f} m)")

    # Density map à 50 cm pour trouver la zone dense (image petite, clusters visibles)
    resolution = 0.50
    dmap = create_density_map(mid, resolution)
    img = dmap.image

    # Seuil : garder uniquement les pixels au-dessus du percentile 90
    # pour isoler le bâtiment (zones denses) vs terrain épars
    occupied = img > 0
    if not occupied.any():
        print("Aucun pixel occupé.")
        return

    threshold = float(np.percentile(img[occupied], 75))
    dense = img >= threshold

    # Trouver le plus grand composant connexe (= le bâtiment principal)
    from scipy.ndimage import label, find_objects
    labeled, n_labels = label(dense)
    if n_labels == 0:
        print("Aucun cluster dense trouvé.")
        return

    # Cluster le plus grand
    sizes = [(labeled == i).sum() for i in range(1, n_labels + 1)]
    biggest = int(np.argmax(sizes)) + 1
    cluster_mask = labeled == biggest

    rows_nz, cols_nz = np.where(cluster_mask)
    r_min, r_max = int(rows_nz.min()), int(rows_nz.max())
    c_min, c_max = int(cols_nz.min()), int(cols_nz.max())

    bx_min = dmap.x_min + c_min * resolution
    bx_max = dmap.x_min + (c_max + 1) * resolution
    # y est inversé (row 0 = y_max)
    by_max = dmap.y_min + dmap.height * resolution - r_min * resolution
    by_min = dmap.y_min + dmap.height * resolution - (r_max + 1) * resolution

    print(f"\nZone occupée (bâtiment) :")
    print(f"  X : [{bx_min:.2f}, {bx_max:.2f}] m  (largeur {bx_max - bx_min:.2f} m)")
    print(f"  Y : [{by_min:.2f}, {by_max:.2f}] m  (profondeur {by_max - by_min:.2f} m)")
    margin = 2.0
    print(f"\nCommande suggérée (marge +{margin} m) :")
    print(
        f"  scan2plan process {path.name} "
        f"--floor-z {z_floor:.2f} --ceiling-z {z_ceiling:.2f} "
        f"--x-min {bx_min - margin:.2f} --x-max {bx_max + margin:.2f} "
        f"--y-min {by_min - margin:.2f} --y-max {by_max + margin:.2f} "
        f"--save-intermediates"
    )

    # Zoom sur la zone du bâtiment
    pad = 20  # pixels de marge
    r1 = max(0, r_min - pad)
    r2 = min(img.shape[0], r_max + pad)
    c1 = max(0, c_min - pad)
    c2 = min(img.shape[1], c_max + pad)
    crop = img[r1:r2, c1:c2]

    out = path.parent / f"{path.stem}_building_zoom.png"
    fig, ax = plt.subplots(figsize=(10, 10))
    vmax = float(np.percentile(crop[crop > 0], 95)) if (crop > 0).any() else 1.0
    ax.imshow(crop, cmap="hot", origin="upper", vmin=0, vmax=vmax)
    ax.set_title(
        f"Zoom bâtiment — {path.stem}\n"
        f"X=[{bx_min:.1f}, {bx_max:.1f}] m  Y=[{by_min:.1f}, {by_max:.1f}] m"
    )
    plt.colorbar(ax.images[0], ax=ax, label="pts/pixel")
    plt.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nZoom sauvegardé : {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python scripts/find_building_bbox.py <fichier.e57>")
        sys.exit(1)
    find_bbox(sys.argv[1])
