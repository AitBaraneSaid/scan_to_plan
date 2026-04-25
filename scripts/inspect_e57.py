"""Inspection d'un fichier E57 : métadonnées, scans, position des stations."""

from __future__ import annotations

import sys
from pathlib import Path

import pye57


def inspect_e57(filepath: str) -> None:
    path = Path(filepath)
    if not path.exists():
        print(f"Fichier introuvable : {path}")
        sys.exit(1)

    e57 = pye57.E57(str(path))
    scan_count = e57.scan_count

    print(f"\n{'='*60}")
    print(f"Fichier : {path.name}")
    print(f"Nombre de scans (stations) : {scan_count}")
    print(f"{'='*60}")

    for i in range(scan_count):
        header = e57.get_header(i)
        print(f"\n--- Scan #{i} ---")

        # Nom / GUID du scan
        if hasattr(header, "name") and header.name:
            print(f"  Nom          : {header.name}")
        if hasattr(header, "guid") and header.guid:
            print(f"  GUID         : {header.guid}")

        # Nombre de points
        print(f"  Points       : {header.point_count:,}")

        # Pose de la station (matrice de transformation)
        if hasattr(header, "rotation") and header.rotation is not None:
            r = header.rotation
            print(f"  Rotation     : [{r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f}, {r[3]:.4f}]  (quaternion w,x,y,z)")
        if hasattr(header, "translation") and header.translation is not None:
            t = header.translation
            print(f"  Translation  : X={t[0]:.3f} m  Y={t[1]:.3f} m  Z={t[2]:.3f} m")
            print(f"  → Hauteur station (Z) : {t[2]:.3f} m")

        # Bounding box du scan
        if hasattr(header, "cartesian_bounds") and header.cartesian_bounds is not None:
            b = header.cartesian_bounds
            print(f"  BBox X       : [{b.x_minimum:.3f}, {b.x_maximum:.3f}] m")
            print(f"  BBox Y       : [{b.y_minimum:.3f}, {b.y_maximum:.3f}] m")
            print(f"  BBox Z       : [{b.z_minimum:.3f}, {b.z_maximum:.3f}] m")

        # Champs disponibles
        if hasattr(header, "point_fields"):
            print(f"  Champs       : {', '.join(header.point_fields)}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python scripts/inspect_e57.py <chemin_vers_fichier.e57>")
        sys.exit(1)
    inspect_e57(sys.argv[1])
