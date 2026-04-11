"""Régularisation géométrique : snapping angulaire sur les orientations dominantes."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def regularize_segments(
    segments: np.ndarray,
    dominant_orientations: list[float],
    snap_tolerance_deg: float,
) -> np.ndarray:
    """Snappe les segments sur les orientations dominantes si l'écart est faible.

    Pour chaque segment, calcule son angle et cherche l'orientation dominante
    la plus proche. Si l'écart < ``snap_tolerance_deg``, pivote le segment
    autour de son centre pour l'aligner exactement.

    Args:
        segments: Array (M, 4) float64 — segments [x1, y1, x2, y2] en mètres.
        dominant_orientations: Liste d'angles dominants en radians.
        snap_tolerance_deg: Tolérance angulaire pour le snapping (degrés).

    Returns:
        Array (M, 4) float64 — segments régularisés.
    """
    if not dominant_orientations or len(segments) == 0:
        logger.debug("Régularisation ignorée : pas d'orientations dominantes ou segments vides.")
        return segments

    result = segments.astype(np.float64).copy()
    snap_tol_rad = np.deg2rad(snap_tolerance_deg)
    snapped_count = 0

    for i, seg in enumerate(result):
        angle = _segment_angle(seg)
        target, diff = _nearest_dominant(angle, dominant_orientations)
        if diff <= snap_tol_rad:
            result[i] = _snap_segment(seg, target)
            snapped_count += 1

    logger.info(
        "Régularisation (tolérance=%.1f°) : %d / %d segments snappés.",
        snap_tolerance_deg,
        snapped_count,
        len(segments),
    )
    return result


def _segment_angle(seg: np.ndarray) -> float:
    """Retourne l'angle du segment en radians dans [0, π).

    Args:
        seg: [x1, y1, x2, y2].

    Returns:
        Angle en radians.
    """
    return float(np.arctan2(seg[3] - seg[1], seg[2] - seg[0]) % np.pi)


def _nearest_dominant(
    angle: float,
    orientations: list[float],
) -> tuple[float, float]:
    """Trouve l'orientation dominante la plus proche d'un angle.

    Args:
        angle: Angle du segment en radians.
        orientations: Orientations dominantes en radians.

    Returns:
        (target_angle, diff_rad) — orientation cible et écart minimal.
    """
    diffs = [min(abs(angle - o) % np.pi, np.pi - abs(angle - o) % np.pi) for o in orientations]
    idx = int(np.argmin(diffs))
    return orientations[idx], diffs[idx]


def _snap_segment(seg: np.ndarray, target_angle: float) -> np.ndarray:
    """Pivote un segment autour de son centre pour l'aligner sur target_angle.

    Args:
        seg: [x1, y1, x2, y2] en mètres.
        target_angle: Angle cible en radians.

    Returns:
        Segment réaligné [x1, y1, x2, y2].
    """
    cx = (seg[0] + seg[2]) / 2
    cy = (seg[1] + seg[3]) / 2
    half_len = np.hypot(seg[2] - seg[0], seg[3] - seg[1]) / 2
    dx = np.cos(target_angle) * half_len
    dy = np.sin(target_angle) * half_len
    return np.array([cx - dx, cy - dy, cx + dx, cy + dy])
