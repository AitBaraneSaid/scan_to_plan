"""Fonctions géométriques 2D : intersections, distances, angles."""

from __future__ import annotations

import numpy as np


def angle_between_segments_deg(
    seg1: tuple[float, float, float, float],
    seg2: tuple[float, float, float, float],
) -> float:
    """Retourne l'angle en degrés entre deux segments (dans [0, 90]).

    Args:
        seg1: (x1, y1, x2, y2) du premier segment.
        seg2: (x1, y1, x2, y2) du second segment.

    Returns:
        Angle en degrés entre 0 et 90.
    """
    dx1, dy1 = seg1[2] - seg1[0], seg1[3] - seg1[1]
    dx2, dy2 = seg2[2] - seg2[0], seg2[3] - seg2[1]
    theta1 = np.arctan2(dy1, dx1)
    theta2 = np.arctan2(dy2, dx2)
    diff = abs(np.degrees(theta1 - theta2)) % 180.0
    return min(diff, 180.0 - diff)


def perpendicular_distance_point_to_line(
    point: tuple[float, float],
    seg: tuple[float, float, float, float],
) -> float:
    """Distance perpendiculaire d'un point à la droite supportant un segment.

    Args:
        point: (x, y) du point.
        seg: (x1, y1, x2, y2) du segment.

    Returns:
        Distance en unités métriques (mètres si les coordonnées sont en mètres).
    """
    x0, y0 = point
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length == 0:
        return np.hypot(x0 - x1, y0 - y1)
    return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / length


def segment_length(seg: tuple[float, float, float, float]) -> float:
    """Longueur euclidienne d'un segment.

    Args:
        seg: (x1, y1, x2, y2).

    Returns:
        Longueur en unités métriques.
    """
    return np.hypot(seg[2] - seg[0], seg[3] - seg[1])


def line_intersection(
    seg1: tuple[float, float, float, float],
    seg2: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    """Calcule le point d'intersection de deux droites (support de segments).

    Args:
        seg1: (x1, y1, x2, y2) premier segment.
        seg2: (x1, y1, x2, y2) second segment.

    Returns:
        (x, y) du point d'intersection, ou None si les droites sont parallèles.
    """
    x1, y1, x2, y2 = seg1
    x3, y3, x4, y4 = seg2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return (x, y)
