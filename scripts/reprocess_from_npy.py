"""Repart du nuage prétraité (.npy) pour relancer slicing → DXF rapidement.

Évite de relire le E57 (2 min) et de refaire downsampling+SOR (5 min).
Utile pour itérer sur les paramètres de Hough, fusion, topologie.

Usage :
    python scripts/reprocess_from_npy.py data/T92123_31_points_preprocessed.npy \
        --floor-z -0.75 --ceiling-z 2.35 --output data/T92123_31_v2.dxf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Retraite depuis un .npy prétraité.")
    parser.add_argument("npy_file", help="Fichier .npy (nuage prétraité)")
    parser.add_argument("--floor-z", type=float, required=True)
    parser.add_argument("--ceiling-z", type=float, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Fichier DXF de sortie (défaut : même nom que .npy)")
    parser.add_argument("--config", type=str, default=None,
                        help="Fichier de configuration YAML")
    parser.add_argument("--x-min", type=float, default=None, help="Clip XY : X minimum (m)")
    parser.add_argument("--x-max", type=float, default=None, help="Clip XY : X maximum (m)")
    parser.add_argument("--y-min", type=float, default=None, help="Clip XY : Y minimum (m)")
    parser.add_argument("--y-max", type=float, default=None, help="Clip XY : Y maximum (m)")
    parser.add_argument("--debug", action="store_true",
                        help="Afficher les logs DEBUG")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    log = logging.getLogger("reprocess")

    from scan2plan.config import ScanConfig
    from scan2plan.detection.line_detection import detect_lines_hough
    from scan2plan.detection.micro_fusion import micro_fuse_segments
    from scan2plan.detection.segment_cleanup import clean_parasites
    from scan2plan.detection.morphology import binarize_density_map, morphological_cleanup
    from scan2plan.detection.multi_slice_filter import (
        classify_segments, get_door_candidates, get_window_candidates,
        match_segments_across_slices,
    )
    from scan2plan.detection.openings import detect_all_openings
    from scan2plan.detection.segment_fusion import fuse_collinear_segments
    from scan2plan.io.writers import export_dxf_v1
    from scan2plan.preprocessing.floor_ceiling import filter_vertical_range
    from scan2plan.qa.validator import validate_plan
    from scan2plan.slicing.density_map import create_density_map
    from scan2plan.slicing.slicer import extract_multi_slices
    from scan2plan.vectorization.angular_regularization import (
        detect_dominant_orientations, snap_angles,
    )
    from scan2plan.vectorization.topology import build_wall_graph, clean_topology

    npy_path = Path(args.npy_file)
    output_path = Path(args.output) if args.output else npy_path.with_suffix(".dxf")
    config = ScanConfig(user_config_path=Path(args.config) if args.config else None)

    log.info("Chargement : %s", npy_path)
    points = np.load(npy_path)
    log.info("  %d points chargés", len(points))

    # Filtrage vertical
    points = filter_vertical_range(points, args.floor_z, args.ceiling_z)
    log.info("  Apres filtrage vertical : %d points", len(points))

    # Clip XY optionnel
    mask = np.ones(len(points), dtype=bool)
    if args.x_min is not None:
        mask &= points[:, 0] >= args.x_min
    if args.x_max is not None:
        mask &= points[:, 0] <= args.x_max
    if args.y_min is not None:
        mask &= points[:, 1] >= args.y_min
    if args.y_max is not None:
        mask &= points[:, 1] <= args.y_max
    if not mask.all():
        points = points[mask]
        log.info("  Apres clip XY : %d points", len(points))

    # Slices
    cfg_sl = config.slicing
    slices_xy = extract_multi_slices(
        points, heights=cfg_sl.heights, thickness=cfg_sl.thickness, floor_z=args.floor_z
    )
    for lbl, pts in slices_xy.items():
        log.info("  Slice %s : %d points", lbl, len(pts))

    # Density map + Hough
    cfg_h = config.hough
    cfg_morph = config.morphology
    cfg_dm = config.density_map

    segments_by_slice: dict = {}
    density_maps: dict = {}
    binary_images: dict = {}
    all_detected = []

    for label, slice_xy in slices_xy.items():
        if len(slice_xy) == 0:
            segments_by_slice[label] = []
            continue
        dmap = create_density_map(slice_xy, cfg_dm.resolution)
        density_maps[label] = dmap
        binary_raw = binarize_density_map(dmap.image)
        binary = morphological_cleanup(
            binary_raw, cfg_morph.kernel_size, cfg_morph.close_iterations, cfg_morph.open_iterations
        )
        binary_images[label] = binary
        segs = detect_lines_hough(
            binary, dmap,
            rho=cfg_h.rho, theta_deg=cfg_h.theta_deg,
            threshold=cfg_h.threshold, min_line_length=cfg_h.min_line_length,
            max_line_gap=cfg_h.max_line_gap, source_slice=label,
        )
        segments_by_slice[label] = segs
        all_detected.extend(segs)
        log.info("  Hough [%s] : %d segments", label, len(segs))

    log.info("Total brut : %d segments", len(all_detected))

    # Multi-slice filter
    cfg_msf = config._data.get("multi_slice_filter", {})
    matches = match_segments_across_slices(
        segments_by_slice,
        angle_tolerance_deg=float(cfg_msf.get("angle_tolerance_deg", 5.0)),
        distance_tolerance=float(cfg_msf.get("distance_tolerance", 0.10)),
    )
    wall_segments = classify_segments(matches)
    door_cands = get_door_candidates(matches)
    window_cands = get_window_candidates(matches)
    log.info("Multi-slice : %d murs, %d portes cand., %d fenêtres cand.",
             len(wall_segments), len(door_cands), len(window_cands))

    # Force confidence=0.9 : tous ces segments ont passé le filtre multi-slice.
    # La confidence basée sur la longueur (length/1.0m) n'est pas pertinente ici
    # et relègue à tort les cloisons courtes (< 50 cm) sur le calque INCERTAIN.
    from scan2plan.detection.line_detection import DetectedSegment as _DS
    wall_segments = [
        _DS(x1=s.x1, y1=s.y1, x2=s.x2, y2=s.y2,
            source_slice=s.source_slice, confidence=0.9)
        for s in wall_segments
    ]

    if not wall_segments:
        log.error("Aucun segment de mur — arrêt.")
        return

    # Micro-fusion : recoller uniquement les fragments Hough (gap < 5 cm)
    # Ne jamais franchir une ouverture (porte ≥ 70 cm, fenêtre ≥ 30 cm)
    fused = micro_fuse_segments(
        wall_segments,
        max_gap=0.05,
        angle_tolerance_deg=3.0,
        perpendicular_tolerance=0.02,
    )
    log.info("Micro-fusion : %d -> %d segments", len(wall_segments), len(fused))

    # Nettoyage des parasites : supprimer les courts isolés, garder les cloisons légitimes
    fused = clean_parasites(
        fused,
        min_length=0.15,
        parallel_search_distance=0.30,
        perpendicular_search_distance=0.10,
    )
    log.info("Clean parasites : %d segments apres nettoyage", len(fused))

    # Détection des orientations dominantes (histogramme pondéré par longueur)
    dominant_angles = detect_dominant_orientations(fused)
    log.info("Orientations : %s", [f"{np.degrees(a):.1f}°" for a in dominant_angles])

    # Snap angulaire pur : rotation autour du centre, sans fusion ni déplacement
    cfg_reg = config._data.get("regularization", {})
    snap_tol = float(cfg_reg.get("snap_tolerance_deg", 5.0))
    regularized = snap_angles(fused, dominant_angles, tolerance_deg=snap_tol)
    log.info("Apres regularisation : %d segments (entree=%d)", len(regularized), len(fused))

    # Wall pairing AVANT align_parallel : les faces opposées sont encore distinctes
    from scan2plan.vectorization.wall_pairing import PairingConfig, find_wall_pairs, Segment as PairSeg
    from scan2plan.detection.line_detection import DetectedSegment

    wall_pairs_for_export: list = []
    cfg_wp = config._data.get("wall_pairing", {})
    if cfg_wp.get("enabled", True):
        pairing_config = PairingConfig(
            angle_tolerance_deg=float(cfg_wp.get("angle_tolerance_deg", 3.0)),
            min_distance=float(cfg_wp.get("min_distance", 0.04)),
            max_distance=float(cfg_wp.get("max_distance", 0.30)),
            min_overlap_abs=float(cfg_wp.get("min_overlap_abs", 0.10)),
            min_overlap_ratio=float(cfg_wp.get("min_overlap_ratio", 0.20)),
            corridor_margin=float(cfg_wp.get("corridor_margin", 0.02)),
            typical_wall_thickness=float(cfg_wp.get("typical_wall_thickness", 0.15)),
            min_segment_length=float(cfg_wp.get("min_segment_length", 0.10)),
            corridor_intersection_threshold=float(cfg_wp.get("corridor_intersection_threshold", 0.05)),
        )
        pairing_segs = [
            PairSeg(x1=s.x1, y1=s.y1, x2=s.x2, y2=s.y2,
                    label="wall", confidence=s.confidence, source_slice=s.source_slice)
            for s in regularized
        ]
        pairing_result = find_wall_pairs(pairing_segs, pairing_config)
        log.info("Wall pairing : %d -> %d segments (%d paires, %d rejet corridor, %d conflits)",
                 len(regularized), len(pairing_result.pairs) + len(pairing_result.unpaired_segments),
                 pairing_result.num_pairs_confirmed,
                 pairing_result.num_pairs_rejected_corridor,
                 pairing_result.num_pairs_rejected_conflict)
        if pairing_result.pairs:
            thicknesses = sorted([p.thickness for p in pairing_result.pairs])
            log.info("  Epaisseurs paires (m) : min=%.3f med=%.3f max=%.3f",
                     thicknesses[0], thicknesses[len(thicknesses)//2], thicknesses[-1])
        wall_pairs_for_export = pairing_result.pairs
        # Pour la topologie : on garde les médianes des paires + les non-appariés
        # Les faces brutes (face_a, face_b) sont stockées dans wall_pairs_for_export pour l'export DXF
        paired_out = [wp.median_segment for wp in pairing_result.pairs]
        paired_out.extend(pairing_result.unpaired_segments)
        regularized = [
            DetectedSegment(x1=s.x1, y1=s.y1, x2=s.x2, y2=s.y2,
                            source_slice=s.source_slice, confidence=s.confidence)
            for s in paired_out
        ]
        _dump_segments_dxf(regularized, output_path.with_stem(output_path.stem + "_post_pair"))
        log.info("Dump post-pairing : %d segments (medianes + non apparies)", len(regularized))
    else:
        log.info("Wall pairing desactive.")

    # Ouvertures
    cfg_op = config._data.get("openings", {})
    openings = detect_all_openings(door_cands + window_cands, density_maps, binary_images, cfg_op)
    log.info("Ouvertures : %d", len(openings))

    # Dump pre-topologie
    _dump_segments_dxf(regularized, output_path.with_stem(output_path.stem + "_pre_topo"))
    log.info("Dump pre-topologie : %d segments", len(regularized))

    # Topologie : on connecte les intersections sans supprimer les cloisons courtes
    # min_seg réduit à 0.05m pour conserver les embrasures et cloisons courtes
    cfg_top = config._data.get("topology", {})
    inter_dist = float(cfg_top.get("intersection_distance", 0.50))
    min_seg = float(cfg_top.get("min_segment_length", 0.05))
    lengths_pre = sorted([s.length for s in regularized], reverse=True)
    n_above = sum(1 for l in lengths_pre if l >= min_seg)
    n_below = len(lengths_pre) - n_above
    log.info("Longueurs pre-topo (>= %.2fm: %d, < %.2fm: %d) top-10: %s",
             min_seg, n_above, min_seg, n_below,
             [f"{l:.2f}" for l in lengths_pre[:10]])

    log.info("Avant topologie : %d segments, inter_dist=%.2fm, min_seg=%.2fm",
             len(regularized), inter_dist, min_seg)
    wall_graph = build_wall_graph(regularized, openings, inter_dist, min_seg)
    log.info("Topologie : %d segments, %d noeuds", len(wall_graph.segments), len(wall_graph.nodes))

    # Export DXF (double ligne si wall pairing activé)
    export_dxf_v1(wall_graph, openings, output_path, config=config,
                  wall_pairs=wall_pairs_for_export or None)
    log.info("DXF exporté : %s", output_path)

    # QA
    qa = validate_plan(wall_graph, openings)
    print(f"\n{qa.summary()}")
    print(f"Segments : {len(wall_graph.segments)} (murs={sum(1 for s in wall_graph.segments if s.source_slice != 'uncertain')})")

    # Diagnostique des gaps : distances entre extrémités non connectées
    _diagnose_gaps(wall_graph)


def _diagnose_gaps(wall_graph) -> None:
    """Affiche les distances minimales entre extrémités non connectées."""
    import numpy as np
    nodes = wall_graph.nodes
    edges = wall_graph.edges

    degree: dict[int, int] = {}
    for a, b in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1

    gap_nodes = [i for i, n in enumerate(nodes) if degree.get(i, 0) == 1]
    connected_nodes = [i for i, n in enumerate(nodes) if degree.get(i, 0) > 1]

    if not gap_nodes:
        print("Pas de gaps détectés.")
        return

    print(f"\n=== Diagnostique gaps ({len(gap_nodes)} extrémités libres) ===")
    gap_distances = []
    gap_nearest = []
    for gi in gap_nodes:
        gx, gy = nodes[gi]
        min_dist = float("inf")
        min_ni = -1
        for ni in connected_nodes + gap_nodes:
            if ni == gi:
                continue
            nx, ny = nodes[ni]
            d = float(np.hypot(gx - nx, gy - ny))
            if d < min_dist:
                min_dist = d
                min_ni = ni
        gap_distances.append(min_dist)
        gap_nearest.append((gi, gx, gy, min_ni, min_dist,
                            nodes[min_ni] if min_ni >= 0 else None))

    gap_distances_sorted = sorted(gap_distances)
    print(f"  Distance au plus proche voisin (min/med/max) : "
          f"{gap_distances_sorted[0]:.2f}m / {gap_distances_sorted[len(gap_distances_sorted)//2]:.2f}m / {gap_distances_sorted[-1]:.2f}m")
    print(f"  Distribution : <0.10m: {sum(1 for d in gap_distances_sorted if d < 0.10)}, "
          f"0.10-0.30m: {sum(1 for d in gap_distances_sorted if 0.10 <= d < 0.30)}, "
          f"0.30-1.0m: {sum(1 for d in gap_distances_sorted if 0.30 <= d < 1.0)}, "
          f">1.0m: {sum(1 for d in gap_distances_sorted if d >= 1.0)}")

    # Afficher les proches (< 0.50m) pour diagnostique
    close_gaps = [(gi, gx, gy, ni, d, nn) for gi, gx, gy, ni, d, nn in gap_nearest if d < 0.50]
    close_gaps.sort(key=lambda x: x[4])
    print("\n  Gaps proches (< 0.50m) :")
    for gi, gx, gy, ni, d, nn in close_gaps:
        deg_type = "libre" if degree.get(ni, 0) == 1 else "connecté"
        print(f"    ({gx:.2f},{gy:.2f}) -> ({nn[0]:.2f},{nn[1]:.2f}) d={d:.3f}m  voisin={deg_type}")


def _dump_segments_dxf(
    segments: list,
    output_path: Path,
) -> None:
    """Exporte une liste de segments directement en DXF sans topologie."""
    import ezdxf
    doc = ezdxf.new(dxfversion="R2013")
    msp = doc.modelspace()
    doc.layers.add("MURS", color=5)
    doc.layers.add("INCERTAIN", color=1)
    for seg in segments:
        layer = "INCERTAIN" if seg.confidence < 0.5 else "MURS"
        msp.add_line((seg.x1, seg.y1), (seg.x2, seg.y2), dxfattribs={"layer": layer})
    doc.saveas(str(output_path))


if __name__ == "__main__":
    main()
