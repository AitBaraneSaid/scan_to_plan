"""Visualise la density map binaire avec les segments Hough superposés.

Permet de voir exactement ce que Hough détecte avant tout filtrage.

Usage :
    python scripts/show_binary_and_segments.py data/T92123_31_points_preprocessed.npy \
        --floor-z -0.75 --ceiling-z 2.35
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_file")
    parser.add_argument("--floor-z", type=float, required=True)
    parser.add_argument("--ceiling-z", type=float, required=True)
    args = parser.parse_args()

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scan2plan.config import ScanConfig
    from scan2plan.detection.line_detection import detect_lines_hough
    from scan2plan.detection.morphology import binarize_density_map, morphological_cleanup
    from scan2plan.detection.multi_slice_filter import (
        classify_segments, match_segments_across_slices,
    )
    from scan2plan.detection.segment_fusion import fuse_collinear_segments
    from scan2plan.preprocessing.floor_ceiling import filter_vertical_range
    from scan2plan.slicing.density_map import create_density_map
    from scan2plan.slicing.slicer import extract_multi_slices
    from scan2plan.vectorization.regularization import align_parallel_segments, regularize_segments
    from scan2plan.detection.orientation import detect_dominant_orientations

    npy_path = Path(args.npy_file)
    config = ScanConfig()

    points = np.load(npy_path)
    points = filter_vertical_range(points, args.floor_z, args.ceiling_z)

    cfg_sl = config.slicing
    slices_xy = extract_multi_slices(
        points, heights=cfg_sl.heights, thickness=cfg_sl.thickness, floor_z=args.floor_z
    )

    cfg_h = config.hough
    cfg_morph = config.morphology
    res = config.density_map.resolution

    segments_by_slice = {}
    dmaps = {}
    binaries = {}

    for label, slice_xy in slices_xy.items():
        if len(slice_xy) == 0:
            segments_by_slice[label] = []
            continue
        dmap = create_density_map(slice_xy, res)
        dmaps[label] = dmap
        binary_raw = binarize_density_map(dmap.image)
        binary = morphological_cleanup(binary_raw, cfg_morph.kernel_size,
                                        cfg_morph.close_iterations, cfg_morph.open_iterations)
        binaries[label] = binary
        segs = detect_lines_hough(binary, dmap, rho=cfg_h.rho, theta_deg=cfg_h.theta_deg,
                                   threshold=cfg_h.threshold, min_line_length=cfg_h.min_line_length,
                                   max_line_gap=cfg_h.max_line_gap, source_slice=label)
        segments_by_slice[label] = segs

    cfg_msf = config._data.get("multi_slice_filter", {})
    matches = match_segments_across_slices(
        segments_by_slice,
        angle_tolerance_deg=float(cfg_msf.get("angle_tolerance_deg", 5.0)),
        distance_tolerance=float(cfg_msf.get("distance_tolerance", 0.10)),
    )
    wall_segments = classify_segments(matches)

    cfg_sf = config.segment_fusion
    fused = fuse_collinear_segments(wall_segments, cfg_sf.angle_tolerance_deg,
                                     cfg_sf.perpendicular_dist, cfg_sf.max_gap)
    dominant = detect_dominant_orientations(fused)
    cfg_reg = config._data.get("regularization", {})
    regularized = regularize_segments(fused, dominant, float(cfg_reg.get("snap_tolerance_deg", 5.0)))
    regularized = align_parallel_segments(regularized, dominant)

    print(f"Segments après régularisation : {len(regularized)}")
    lengths = sorted([s.length for s in regularized], reverse=True)
    print(f"Longueurs : {[f'{l:.2f}m' for l in lengths[:20]]}")

    # Figure : density map mid + segments à chaque étape
    label = "mid"
    dmap = dmaps[label]
    binary = binaries[label]
    res_m = dmap.resolution

    def seg_to_px(x1, y1, x2, y2):
        """Convertit des coordonnées métriques en pixels image."""
        px1 = int((x1 - dmap.x_min) / res_m)
        py1 = int(dmap.height - (y1 - dmap.y_min) / res_m)
        px2 = int((x2 - dmap.x_min) / res_m)
        py2 = int(dmap.height - (y2 - dmap.y_min) / res_m)
        return px1, py1, px2, py2

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))

    # Panel 1 : Hough brut (tous les segments high)
    overlay1 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for s in segments_by_slice.get("high", []):
        px1, py1, px2, py2 = seg_to_px(s.x1, s.y1, s.x2, s.y2)
        cv2.line(overlay1, (px1, py1), (px2, py2), (0, 255, 0), 2)
    axes[0].imshow(cv2.cvtColor(overlay1, cv2.COLOR_BGR2RGB), origin="upper")
    axes[0].set_title(f"Hough brut [high] — {len(segments_by_slice.get('high',[]))} seg.")
    axes[0].axis("off")

    # Panel 2 : après classification multi-slice
    overlay2 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for s in wall_segments:
        px1, py1, px2, py2 = seg_to_px(s.x1, s.y1, s.x2, s.y2)
        cv2.line(overlay2, (px1, py1), (px2, py2), (0, 200, 255), 2)
    axes[1].imshow(cv2.cvtColor(overlay2, cv2.COLOR_BGR2RGB), origin="upper")
    axes[1].set_title(f"Murs classifiés — {len(wall_segments)} seg.")
    axes[1].axis("off")

    # Panel 3 : après fusion + régularisation
    overlay3 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for s in regularized:
        px1, py1, px2, py2 = seg_to_px(s.x1, s.y1, s.x2, s.y2)
        color = (0, 0, 255) if s.length > 1.0 else (128, 128, 255)
        cv2.line(overlay3, (px1, py1), (px2, py2), color, 2)
    axes[2].imshow(cv2.cvtColor(overlay3, cv2.COLOR_BGR2RGB), origin="upper")
    axes[2].set_title(f"Après fusion+régularisation — {len(regularized)} seg.")
    axes[2].axis("off")

    plt.suptitle(f"Segments sur density map mid — {npy_path.stem}", fontsize=13)
    plt.tight_layout()
    out = npy_path.parent / f"{npy_path.stem}_segments_overlay.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure : {out}")


if __name__ == "__main__":
    main()
