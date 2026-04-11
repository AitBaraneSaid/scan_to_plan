"""Fusion de segments colinéaires fragmentés issus de la détection Hough."""

from __future__ import annotations

import logging

import numpy as np

from scan2plan.detection.line_detection import DetectedSegment
from scan2plan.utils.geometry import (
    angle_between_segments,
    perpendicular_distance_segment_to_segment,
    segments_overlap_or_gap,
)

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 10


def fuse_collinear_segments(
    segments: list[DetectedSegment],
    angle_tolerance_deg: float = 3.0,
    perpendicular_dist: float = 0.03,
    max_gap: float = 0.20,
) -> list[DetectedSegment]:
    """Fusionne les segments quasi-colinéaires en segments continus.

    Deux segments sont fusionnés si :
    1. Leur angle est < ``angle_tolerance_deg``.
    2. La distance perpendiculaire entre leurs droites est < ``perpendicular_dist``.
    3. Le gap ou chevauchement le long de la direction commune est < ``max_gap``.

    L'algorithme itère jusqu'à convergence (aucune fusion) ou jusqu'à
    ``_MAX_ITERATIONS`` passes.

    Args:
        segments: Liste de ``DetectedSegment`` en coordonnées métriques.
        angle_tolerance_deg: Tolérance angulaire pour la colinéarité (degrés).
        perpendicular_dist: Distance perpendiculaire maximale (mètres).
        max_gap: Gap maximal entre segments à fusionner (mètres).

    Returns:
        Liste de ``DetectedSegment`` fusionnés (longueur ≤ entrée).

    Example:
        >>> result = fuse_collinear_segments(segs, angle_tolerance_deg=3.0)
        >>> len(result) <= len(segs)
        True
    """
    if not segments:
        return []

    angle_tol_rad = float(np.deg2rad(angle_tolerance_deg))
    result = list(segments)
    n_initial = len(result)

    for iteration in range(_MAX_ITERATIONS):
        result, n_fused = _fusion_pass(result, angle_tol_rad, perpendicular_dist, max_gap)
        logger.debug("Fusion passe %d : %d fusions.", iteration + 1, n_fused)
        if n_fused == 0:
            break

    logger.info(
        "Fusion segments : %d → %d segments en %d passes (max %d).",
        n_initial,
        len(result),
        iteration + 1,
        _MAX_ITERATIONS,
    )
    return result


def _fusion_pass(
    segments: list[DetectedSegment],
    angle_tol_rad: float,
    perp_dist: float,
    max_gap: float,
) -> tuple[list[DetectedSegment], int]:
    """Effectue une passe de fusion sur la liste de segments.

    Args:
        segments: Liste de ``DetectedSegment``.
        angle_tol_rad: Tolérance angulaire en RADIANS.
        perp_dist: Distance perpendiculaire max (mètres).
        max_gap: Gap max (mètres).

    Returns:
        (liste fusionnée, nombre de fusions effectuées).
    """
    used = [False] * len(segments)
    fused: list[DetectedSegment] = []
    n_fused = 0

    for i, seg_i in enumerate(segments):
        if used[i]:
            continue
        current = seg_i
        for j in range(i + 1, len(segments)):
            if used[j]:
                continue
            if _are_fusible(current, segments[j], angle_tol_rad, perp_dist, max_gap):
                current = _merge_two(current, segments[j])
                used[j] = True
                n_fused += 1
        fused.append(current)
        used[i] = True

    return fused, n_fused


def _are_fusible(
    s1: DetectedSegment,
    s2: DetectedSegment,
    angle_tol_rad: float,
    perp_dist: float,
    max_gap: float,
) -> bool:
    """Vérifie si deux segments satisfont les 3 critères de fusion.

    Args:
        s1: Premier segment.
        s2: Second segment.
        angle_tol_rad: Tolérance angulaire en radians.
        perp_dist: Distance perpendiculaire max (mètres).
        max_gap: Gap max (mètres).

    Returns:
        True si les segments doivent être fusionnés.
    """
    angle = angle_between_segments(s1.as_tuple(), s2.as_tuple())
    if angle > angle_tol_rad:
        return False

    d = perpendicular_distance_segment_to_segment(s1.as_tuple(), s2.as_tuple())
    if d > perp_dist:
        return False

    gap = segments_overlap_or_gap(s1.as_tuple(), s2.as_tuple())
    return gap <= max_gap


def _merge_two(s1: DetectedSegment, s2: DetectedSegment) -> DetectedSegment:
    """Fusionne deux segments colinéaires en un seul.

    Calcule la droite de régression sur les 4 extrémités par SVD, puis
    projette les extrémités sur cette droite et conserve les projections
    extrêmes.

    Args:
        s1: Premier segment.
        s2: Second segment.

    Returns:
        Segment fusionné avec la confiance max des deux.
    """
    pts = np.array([
        [s1.x1, s1.y1],
        [s1.x2, s1.y2],
        [s2.x1, s2.y1],
        [s2.x2, s2.y2],
    ], dtype=np.float64)

    center = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - center)
    direction = vt[0]

    projections = (pts - center) @ direction
    i_min, i_max = int(projections.argmin()), int(projections.argmax())
    p_start = center + projections[i_min] * direction
    p_end = center + projections[i_max] * direction

    return DetectedSegment(
        x1=float(p_start[0]),
        y1=float(p_start[1]),
        x2=float(p_end[0]),
        y2=float(p_end[1]),
        source_slice=s1.source_slice,
        confidence=max(s1.confidence, s2.confidence),
    )
