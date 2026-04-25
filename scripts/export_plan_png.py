"""Exporte le plan final en PNG haute qualité, style plan architectural.

Usage :
    python scripts/export_plan_png.py data/T92123_31_v12.dxf
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def export(dxf_path: str) -> None:
    import ezdxf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    path = Path(dxf_path)
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()

    layer_style = {
        "MURS":     {"color": "#1a1a1a", "lw": 3.0, "zorder": 3},
        "CLOISONS": {"color": "#333333", "lw": 2.0, "zorder": 3},
        "PORTES":   {"color": "#2e7d32", "lw": 2.0, "zorder": 4},
        "FENETRES": {"color": "#1565c0", "lw": 2.0, "zorder": 4},
        "INCERTAIN":{"color": "#c62828", "lw": 1.5, "zorder": 2, "ls": "--"},
    }

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
                    segments_by_layer[layer].append(
                        (pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
                    )

    # Calcul de l'emprise
    all_x, all_y = [], []
    for segs in segments_by_layer.values():
        for x1, y1, x2, y2 in segs:
            all_x += [x1, x2]
            all_y += [y1, y2]

    if not all_x:
        print("Aucun segment trouvé.")
        return

    margin = 0.5
    xmin, xmax = min(all_x) - margin, max(all_x) + margin
    ymin, ymax = min(all_y) - margin, max(all_y) + margin
    width_m  = xmax - xmin
    height_m = ymax - ymin

    # Figure proportionnelle au bâtiment
    scale = 14.0 / max(width_m, height_m)
    fig_w = width_m * scale
    fig_h = height_m * scale

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("white")

    for layer, segs in segments_by_layer.items():
        style = layer_style.get(layer, {"color": "gray", "lw": 1.5, "zorder": 2})
        color  = style["color"]
        lw     = style["lw"]
        zorder = style.get("zorder", 2)
        ls     = style.get("ls", "-")
        for x1, y1, x2, y2 in segs:
            ax.plot([x1, x2], [y1, y2],
                    color=color, linewidth=lw, linestyle=ls,
                    solid_capstyle="round", zorder=zorder)

    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)   # Y inversé : convention plan archi (nord en haut)

    # Grille légère au mètre
    ax.set_xticks(np.arange(int(xmin), int(xmax) + 1, 1))
    ax.set_yticks(np.arange(int(ymin), int(ymax) + 1, 1))
    ax.grid(True, color="#dddddd", linewidth=0.4, zorder=0)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)

    # Légende
    legend_items = []
    for layer, segs in sorted(segments_by_layer.items()):
        style = layer_style.get(layer, {"color": "gray", "lw": 1.5})
        total_len = sum(np.hypot(x2-x1, y2-y1) for x1, y1, x2, y2 in segs)
        legend_items.append(Line2D(
            [0], [0],
            color=style["color"],
            linewidth=style["lw"],
            linestyle=style.get("ls", "-"),
            label=f"{layer} ({len(segs)} seg. — {total_len:.1f} m)",
        ))
    ax.legend(handles=legend_items, loc="lower right", fontsize=7,
              framealpha=0.9, edgecolor="#aaaaaa")

    total_segs = sum(len(v) for v in segments_by_layer.values())
    total_len  = sum(
        np.hypot(x2-x1, y2-y1)
        for segs in segments_by_layer.values()
        for x1, y1, x2, y2 in segs
    )
    ax.set_title(
        f"Plan 2D — {path.stem}\n"
        f"{total_segs} segments · {total_len:.1f} m de murs · "
        f"{width_m:.1f} × {height_m:.1f} m",
        fontsize=9, pad=8,
    )

    out = path.with_stem(path.stem + "_plan").with_suffix(".png")
    fig.savefig(str(out), dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Plan exporté : {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python scripts/export_plan_png.py <fichier.dxf>")
        sys.exit(1)
    export(sys.argv[1])
