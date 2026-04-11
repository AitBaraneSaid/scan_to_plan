"""Reconstruction topologique : connexion des murs aux intersections."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def resolve_intersections(
    segments: np.ndarray,
    intersection_distance: float,
    min_segment_length: float,
) -> np.ndarray:
    """Connecte les extrémités proches de segments quasi-perpendiculaires.

    Pour chaque paire de segments dont les extrémités sont à moins de
    ``intersection_distance`` et dont l'angle est proche de 90°, prolonge
    les deux segments jusqu'à leur point d'intersection.

    Note: Implémentation complète en V1. Ce stub retourne les segments inchangés.

    Args:
        segments: Array (M, 4) float64 — segments [x1, y1, x2, y2] en mètres.
        intersection_distance: Distance max pour chercher une intersection (mètres).
        min_segment_length: Longueur minimale d'un segment conservé (mètres).

    Returns:
        Array (K, 4) float64 — segments avec intersections résolues.
    """
    logger.info(
        "Reconstruction topologique (V1 — stub) : %d segments en entrée.", len(segments)
    )
    # TODO(V1): implémenter la résolution des intersections avec NetworkX + Shapely
    return segments


def remove_short_segments(
    segments: np.ndarray,
    min_length: float,
) -> np.ndarray:
    """Supprime les segments plus courts que la longueur minimale.

    Args:
        segments: Array (M, 4) float64 — segments [x1, y1, x2, y2].
        min_length: Longueur minimale en mètres.

    Returns:
        Array (K, 4) float64 — segments filtrés (K <= M).
    """
    if len(segments) == 0:
        return segments
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    mask = lengths >= min_length
    result = segments[mask]
    removed = len(segments) - len(result)
    if removed > 0:
        logger.info(
            "Suppression de %d micro-segments (longueur < %.2f m).", removed, min_length
        )
    return result
