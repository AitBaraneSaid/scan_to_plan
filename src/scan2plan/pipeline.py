"""Orchestrateur du pipeline Scan2Plan complet."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from scan2plan.config import ScanConfig
from scan2plan.detection.line_detection import detect_segments_hough
from scan2plan.detection.morphology import binarize_density_map, morphological_cleanup
from scan2plan.detection.segment_fusion import fuse_collinear_segments
from scan2plan.io.readers import read_point_cloud
from scan2plan.io.writers import write_segments_to_dxf
from scan2plan.preprocessing.downsampling import voxel_downsample
from scan2plan.preprocessing.floor_ceiling import (
    detect_floor,
    detect_ceiling,
    filter_vertical_range,
)
from scan2plan.preprocessing.outlier_removal import remove_statistical_outliers
from scan2plan.qa.metrics import compute_basic_metrics
from scan2plan.qa.validator import validate_plan
from scan2plan.slicing.density_map import create_density_map
from scan2plan.slicing.slicer import extract_slice
from scan2plan.utils.coordinate import segments_pixel_to_metric
from scan2plan.vectorization.topology import remove_short_segments
from scan2plan.vectorization.wall_builder import build_wall_entities

logger = logging.getLogger(__name__)


def run_pipeline(
    input_path: Path,
    output_path: Path,
    config: ScanConfig,
) -> None:
    """Exécute le pipeline MVP complet de Scan2Plan.

    Étapes :
    1. Lecture du nuage de points.
    2. Voxel downsampling.
    3. Statistical Outlier Removal.
    4. Détection du sol par RANSAC.
    5. Filtrage vertical.
    6. Extraction de la slice médiane (~1.10 m).
    7. Density map.
    8. Binarisation + morphologie.
    9. Détection de segments par Hough.
    10. Fusion des segments colinéaires.
    11. Suppression des micro-segments.
    12. Construction des entités mur.
    13. Export DXF.
    14. Contrôle qualité.

    Args:
        input_path: Chemin vers le fichier E57 ou LAS/LAZ d'entrée.
        output_path: Chemin vers le fichier DXF de sortie.
        config: Configuration du pipeline.

    Raises:
        FileNotFoundError: Si le fichier d'entrée est introuvable.
        RuntimeError: Si le pipeline échoue à une étape critique.
    """
    logger.info("=== Démarrage du pipeline Scan2Plan MVP ===")
    logger.info("Entrée : %s", input_path)
    logger.info("Sortie : %s", output_path)

    # Étape 1 — Lecture
    points = read_point_cloud(input_path)

    # Étape 2 — Downsampling
    points = voxel_downsample(points, config.preprocessing.voxel_size)

    # Étape 3 — SOR
    points = remove_statistical_outliers(
        points,
        config.preprocessing.sor_k_neighbors,
        config.preprocessing.sor_std_ratio,
    )

    # Étape 4 — Détection sol/plafond
    z_floor, _ = detect_floor(
        points,
        distance_threshold=config.floor_ceiling.ransac_distance,
        num_iterations=config.floor_ceiling.ransac_iterations,
        normal_tolerance_deg=config.floor_ceiling.normal_tolerance_deg,
    )
    z_ceiling, _ = detect_ceiling(
        points,
        floor_z=z_floor,
        distance_threshold=config.floor_ceiling.ransac_distance,
        num_iterations=config.floor_ceiling.ransac_iterations,
        normal_tolerance_deg=config.floor_ceiling.normal_tolerance_deg,
    )

    # Étape 5 — Filtrage vertical
    points = filter_vertical_range(points, z_floor, z_ceiling)

    # Étape 6 — Slice médiane (MVP : une seule slice)
    median_height = config.slicing.heights[1] if len(config.slicing.heights) > 1 else 1.10
    slice_xy = extract_slice(
        points,
        height=median_height,
        thickness=config.slicing.thickness,
        floor_z=z_floor,
    )

    # Étape 7 — Density map
    dmap = create_density_map(slice_xy, config.density_map.resolution)

    # Étape 8 — Binarisation + morphologie
    binary_raw = binarize_density_map(dmap.image)
    binary = morphological_cleanup(
        binary_raw,
        config.morphology.kernel_size,
        config.morphology.close_iterations,
        config.morphology.open_iterations,
    )

    # Étape 9 — Hough
    segments_px = detect_segments_hough(
        binary,
        rho=config.hough.rho,
        theta_deg=config.hough.theta_deg,
        threshold=config.hough.threshold,
        min_line_length=config.hough.min_line_length,
        max_line_gap=config.hough.max_line_gap,
    )

    if len(segments_px) == 0:
        logger.error("Aucun segment détecté par Hough. Le DXF ne sera pas produit.")
        return

    # Conversion pixel → mètres
    segments_m = segments_pixel_to_metric(
        segments_px.astype(np.float64),
        dmap.x_min,
        dmap.y_min,
        dmap.resolution,
        dmap.height,
    )

    # Étape 10 — Fusion
    segments_m = fuse_collinear_segments(
        segments_m,
        config.segment_fusion.angle_tolerance_deg,
        config.segment_fusion.perpendicular_dist,
        config.segment_fusion.max_gap,
    )

    # Étape 11 — Micro-segments
    segments_m = remove_short_segments(segments_m, config.topology.min_segment_length)

    # Étape 12 — Entités mur
    wall_entities = build_wall_entities(segments_m)

    # Étape 13 — Export DXF
    write_segments_to_dxf(
        wall_entities,
        output_path,
        layer=config.dxf.layers["walls"],
        dxf_version=config.dxf.version,
    )

    # Étape 14 — QA
    report = compute_basic_metrics(wall_entities, config.topology.min_segment_length)
    report = validate_plan(wall_entities, report)

    if report.warnings:
        for w in report.warnings:
            logger.warning("QA: %s", w)
    if report.errors:
        for e in report.errors:
            logger.error("QA: %s", e)

    logger.info(
        "=== Pipeline terminé. %d murs exportés dans %s ===",
        len(wall_entities),
        output_path,
    )
