"""Fusion de segments colinéaires fragmentés issus de la détection Hough."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def fuse_collinear_segments(
    segments: np.ndarray,
    angle_tolerance_deg: float,
    perpendicular_dist: float,
    max_gap: float,
) -> np.ndarray:
    """Fusionne les segments colinéaires fragmentés en segments continus.

    Deux segments sont candidats à la fusion si :
    1. Leur différence d'angle est < ``angle_tolerance_deg``.
    2. La distance perpendiculaire entre leurs droites support est < ``perpendicular_dist``.
    3. Le gap ou chevauchement le long de la direction commune est < ``max_gap``.

    Itère jusqu'à convergence (aucune fusion possible).

    Args:
        segments: Array (M, 4) float64 — segments [x1, y1, x2, y2] en mètres.
        angle_tolerance_deg: Tolérance angulaire pour la colinéarité (degrés).
        perpendicular_dist: Distance perpendiculaire maximale (mètres).
        max_gap: Gap maximal entre segments à fusionner (mètres).

    Returns:
        Array (K, 4) float64 — segments fusionnés (K <= M).
    """
    if len(segments) == 0:
        return segments

    result = segments.astype(np.float64).copy()
    changed = True
    passes = 0

    while changed:
        result, changed = _fusion_pass(result, angle_tolerance_deg, perpendicular_dist, max_gap)
        passes += 1

    logger.info(
        "Fusion segments : %d → %d segments en %d passes.",
        len(segments),
        len(result),
        passes,
    )
    return result


def _fusion_pass(
    segments: np.ndarray,
    angle_tol: float,
    perp_dist: float,
    max_gap: float,
) -> tuple[np.ndarray, bool]:
    """Effectue une passe de fusion sur le tableau de segments.

    Args:
        segments: Array (M, 4) float64.
        angle_tol: Tolérance angulaire (degrés).
        perp_dist: Distance perpendiculaire max (mètres).
        max_gap: Gap max (mètres).

    Returns:
        (segments_fusionnés, changed) — changed=True si au moins une fusion a eu lieu.
    """
    used = np.zeros(len(segments), dtype=bool)
    fused: list[np.ndarray] = []
    changed = False

    for i in range(len(segments)):
        if used[i]:
            continue
        current = segments[i]
        for j in range(i + 1, len(segments)):
            if used[j]:
                continue
            if _are_fusible(current, segments[j], angle_tol, perp_dist, max_gap):
                current = _merge_two(current, segments[j])
                used[j] = True
                changed = True
        fused.append(current)
        used[i] = True

    return np.array(fused, dtype=np.float64), changed


def _are_fusible(
    s1: np.ndarray,
    s2: np.ndarray,
    angle_tol: float,
    perp_dist: float,
    max_gap: float,
) -> bool:
    """Vérifie si deux segments satisfont les 3 critères de fusion.

    Args:
        s1: Segment [x1, y1, x2, y2].
        s2: Segment [x1, y1, x2, y2].
        angle_tol: Tolérance angulaire (degrés).
        perp_dist: Distance perpendiculaire max (mètres).
        max_gap: Gap max (mètres).

    Returns:
        True si les segments doivent être fusionnés.
    """
    from scan2plan.utils.geometry import (
        angle_between_segments_deg,
        perpendicular_distance_point_to_line,
    )

    angle = angle_between_segments_deg(tuple(s1), tuple(s2))  # type: ignore[arg-type]
    if angle > angle_tol:
        return False

    d = perpendicular_distance_point_to_line((s2[0], s2[1]), tuple(s1))  # type: ignore[arg-type]
    if d > perp_dist:
        return False

    gap = _longitudinal_gap(s1, s2)
    return gap <= max_gap


def _longitudinal_gap(s1: np.ndarray, s2: np.ndarray) -> float:
    """Calcule le gap longitudinal entre deux segments quasi-colinéaires.

    Args:
        s1: Segment [x1, y1, x2, y2].
        s2: Segment [x1, y1, x2, y2].

    Returns:
        Gap en mètres (négatif si chevauchement).
    """
    dx = s1[2] - s1[0]
    dy = s1[3] - s1[1]
    length = np.hypot(dx, dy)
    if length < 1e-9:
        return 0.0
    ux, uy = dx / length, dy / length

    # Projections des 4 extrémités sur l'axe de s1
    p1 = s1[0] * ux + s1[1] * uy
    p2 = s1[2] * ux + s1[3] * uy
    p3 = s2[0] * ux + s2[1] * uy
    p4 = s2[2] * ux + s2[3] * uy

    seg1_min, seg1_max = min(p1, p2), max(p1, p2)
    seg2_min, seg2_max = min(p3, p4), max(p3, p4)

    # Gap positif = espace entre les deux, négatif = chevauchement
    return max(seg2_min - seg1_max, seg1_min - seg2_max)


def _merge_two(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Fusionne deux segments colinéaires en un seul couvrant leurs extrêmes.

    Args:
        s1: Segment [x1, y1, x2, y2].
        s2: Segment [x1, y1, x2, y2].

    Returns:
        Segment fusionné [x1, y1, x2, y2].
    """
    pts = np.array([[s1[0], s1[1]], [s1[2], s1[3]], [s2[0], s2[1]], [s2[2], s2[3]]])
    # Régresser une droite sur les 4 points et prendre les projections extrêmes
    center = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - center)
    direction = vt[0]

    projections = (pts - center) @ direction
    i_min, i_max = projections.argmin(), projections.argmax()
    p_start = center + projections[i_min] * direction
    p_end = center + projections[i_max] * direction
    return np.array([p_start[0], p_start[1], p_end[0], p_end[1]])
