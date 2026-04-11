"""Fonctions géométriques 2D : intersections, distances, angles.

Chaque segment est représenté comme un tuple (x1, y1, x2, y2) en mètres.
"""

from __future__ import annotations

import numpy as np

# Seuil en dessous duquel un segment est considéré de longueur nulle
_EPSILON = 1e-9


def segment_length(seg: tuple[float, float, float, float]) -> float:
    """Longueur euclidienne d'un segment.

    Args:
        seg: (x1, y1, x2, y2) en mètres.

    Returns:
        Longueur en mètres. Retourne 0.0 pour un segment dégénéré.

    Example:
        >>> segment_length((0.0, 0.0, 3.0, 4.0))
        5.0
    """
    return float(np.hypot(seg[2] - seg[0], seg[3] - seg[1]))


def segment_angle(seg: tuple[float, float, float, float]) -> float:
    """Angle du segment par rapport à l'axe X, dans [0, π).

    Pour un segment dégénéré (longueur nulle), retourne 0.0.

    Args:
        seg: (x1, y1, x2, y2) en mètres.

    Returns:
        Angle en radians dans [0, π).

    Example:
        >>> import math
        >>> segment_angle((0.0, 0.0, 0.0, 1.0))  # vertical
        1.5707963...
    """
    dx = seg[2] - seg[0]
    dy = seg[3] - seg[1]
    if abs(dx) < _EPSILON and abs(dy) < _EPSILON:
        return 0.0
    angle = float(np.arctan2(dy, dx))
    # Ramener dans [0, π) — deux directions opposées représentent la même droite
    if angle < 0:
        angle += np.pi
    return angle % np.pi


def angle_between_segments(
    seg1: tuple[float, float, float, float],
    seg2: tuple[float, float, float, float],
) -> float:
    """Angle entre deux segments en radians, dans [0, π/2].

    Retourne toujours un angle dans [0, π/2] : les directions opposées
    sont identiques pour des droites non orientées.

    Args:
        seg1: (x1, y1, x2, y2) premier segment.
        seg2: (x1, y1, x2, y2) second segment.

    Returns:
        Angle en radians dans [0, π/2].

    Example:
        >>> import math
        >>> angle_between_segments((0,0,1,0), (0,0,0,1))
        1.5707963...
    """
    a1 = segment_angle(seg1)
    a2 = segment_angle(seg2)
    diff = abs(a1 - a2) % np.pi
    return float(min(diff, np.pi - diff))


def perpendicular_distance_segment_to_segment(
    seg1: tuple[float, float, float, float],
    seg2: tuple[float, float, float, float],
) -> float:
    """Distance perpendiculaire moyenne entre deux segments quasi-parallèles.

    Calcule la moyenne des distances perpendiculaires des deux extrémités
    de seg2 à la droite support de seg1.

    Args:
        seg1: Segment de référence (x1, y1, x2, y2).
        seg2: Segment à mesurer (x1, y1, x2, y2).

    Returns:
        Distance en mètres. Retourne la distance point-à-point si seg1 est dégénéré.

    Example:
        >>> perpendicular_distance_segment_to_segment((0,0,1,0), (0,0.05,1,0.05))
        0.05
    """
    d1 = perpendicular_distance_point_to_line((seg2[0], seg2[1]), seg1)
    d2 = perpendicular_distance_point_to_line((seg2[2], seg2[3]), seg1)
    return float((d1 + d2) / 2.0)


def perpendicular_distance_point_to_line(
    point: tuple[float, float],
    seg: tuple[float, float, float, float],
) -> float:
    """Distance perpendiculaire d'un point à la droite supportant un segment.

    Args:
        point: (x, y) du point.
        seg: (x1, y1, x2, y2) du segment.

    Returns:
        Distance en mètres. Si le segment est dégénéré, retourne la distance
        euclidienne au point de départ du segment.
    """
    x0, y0 = point
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    if length < _EPSILON:
        return float(np.hypot(x0 - x1, y0 - y1))
    return float(abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / length)


def segments_overlap_or_gap(
    seg1: tuple[float, float, float, float],
    seg2: tuple[float, float, float, float],
) -> float:
    """Chevauchement (négatif) ou gap (positif) entre deux segments projetés.

    Projette les quatre extrémités sur la direction de seg1, puis calcule
    la distance signée entre les bornes des deux intervalles.

    Args:
        seg1: Premier segment (x1, y1, x2, y2).
        seg2: Second segment (x1, y1, x2, y2).

    Returns:
        Valeur positive = gap en mètres entre les segments.
        Valeur négative = longueur de chevauchement en mètres.
        0.0 si seg1 est dégénéré.

    Example:
        >>> segments_overlap_or_gap((0,0,1,0), (1.1,0,2,0))
        0.1
    """
    dx = seg1[2] - seg1[0]
    dy = seg1[3] - seg1[1]
    length = float(np.hypot(dx, dy))
    if length < _EPSILON:
        return 0.0
    ux, uy = dx / length, dy / length

    p1 = seg1[0] * ux + seg1[1] * uy
    p2 = seg1[2] * ux + seg1[3] * uy
    p3 = seg2[0] * ux + seg2[1] * uy
    p4 = seg2[2] * ux + seg2[3] * uy

    seg1_min, seg1_max = min(p1, p2), max(p1, p2)
    seg2_min, seg2_max = min(p3, p4), max(p3, p4)

    # gap > 0 si espace entre les deux, < 0 si chevauchement
    return float(max(seg2_min - seg1_max, seg1_min - seg2_max))


def line_intersection(
    seg1: tuple[float, float, float, float],
    seg2: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    """Point d'intersection des droites support de deux segments.

    Args:
        seg1: (x1, y1, x2, y2) premier segment.
        seg2: (x1, y1, x2, y2) second segment.

    Returns:
        (x, y) du point d'intersection, ou None si les droites sont parallèles
        (|déterminant| < 1e-10).

    Example:
        >>> line_intersection((0,0,1,0), (0.5,-1,0.5,1))
        (0.5, 0.0)
    """
    x1, y1, x2, y2 = seg1
    x3, y3, x4, y4 = seg2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return (float(x), float(y))


# ---------------------------------------------------------------------------
# Alias conservé pour compatibilité interne (segment_fusion.py)
# ---------------------------------------------------------------------------

def angle_between_segments_deg(
    seg1: tuple[float, float, float, float],
    seg2: tuple[float, float, float, float],
) -> float:
    """Angle entre deux segments en DEGRÉS, dans [0, 90].

    Wrapper autour de ``angle_between_segments`` pour la compatibilité
    avec le code existant.

    Args:
        seg1: (x1, y1, x2, y2) premier segment.
        seg2: (x1, y1, x2, y2) second segment.

    Returns:
        Angle en degrés dans [0, 90].
    """
    return float(np.degrees(angle_between_segments(seg1, seg2)))
