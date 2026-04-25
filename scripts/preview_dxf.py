"""Prévisualise un fichier DXF en PNG — calques colorés, segments annotés.

Usage :
    python scripts/preview_dxf.py data/T92123_31.dxf
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def preview(dxf_path: str) -> None:
    import ezdxf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    path = Path(dxf_path)
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()

    # Collecte tous les segments par calque
    layer_colors = {
        "MURS": "steelblue",
        "CLOISONS": "royalblue",
        "PORTES": "green",
        "FENETRES": "orange",
        "INCERTAIN": "red",
    }
    default_color = "gray"

    segments_by_layer: dict[str, list[tuple]] = {}
    for entity in msp:
        if entity.dxftype() in ("LINE", "LWPOLYLINE"):
            layer = entity.dxf.layer.upper()
            if layer not in segments_by_layer:
                segments_by_layer[layer] = []
            if entity.dxftype() == "LINE":
                s = entity.dxf.start
                e = entity.dxf.end
                segments_by_layer[layer].append((s.x, s.y, e.x, e.y))
            elif entity.dxftype() == "LWPOLYLINE":
                pts = list(entity.get_points())
                for i in range(len(pts) - 1):
                    segments_by_layer[layer].append((pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1]))

    total = sum(len(v) for v in segments_by_layer.values())
    print(f"DXF : {path.name}")
    print(f"  Total : {total} segments sur {len(segments_by_layer)} calques")
    for layer, segs in sorted(segments_by_layer.items()):
        lengths = [np.hypot(x2-x1, y2-y1) for x1, y1, x2, y2 in segs]
        print(f"  {layer:15s} : {len(segs):4d} segments  (total {sum(lengths):.1f} m, moy {np.mean(lengths):.2f} m)")

    # Figure
    fig, ax = plt.subplots(figsize=(14, 14))
    legend_patches = []

    for layer, segs in segments_by_layer.items():
        color = layer_colors.get(layer, default_color)
        lw = 2.0 if layer == "MURS" else 1.5
        for x1, y1, x2, y2 in segs:
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, solid_capstyle="round")
        legend_patches.append(mpatches.Patch(color=color, label=f"{layer} ({len(segs)})"))

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(handles=legend_patches, loc="upper right", fontsize=10)
    ax.set_title(f"{path.name} — {total} segments", fontsize=13)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    out = path.with_suffix(".preview.png")
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPrévisualisation : {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python scripts/preview_dxf.py <fichier.dxf>")
        sys.exit(1)
    preview(sys.argv[1])
