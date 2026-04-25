"""Suppression des segments parasites courts et isolés.

Un parasite est court ET isolé : aucun voisin parallèle proche (face de mur)
ni voisin perpendiculaire connecté (bout de mur). Une cloison courte légitime
est toujours proche d'un autre segment qui la justifie.
"""

from __future__ import annotations

import logging

import numpy as np

from scan2plan.detection.line_detection import DetectedSegment

logger = logging.getLogger(__name__)

Segment = DetectedSegment


def clean_parasites(
    segments: list[Segment],
    min_length: float = 0.15,
    parallel_search_distance: float = 0.30,
    perpendicular_search_distance: float = 0.10,
) -> list[Segment]:
    """Supprime les segments parasites courts et isolés.

    Un segment court (< min_length) est conservé s'il a :
    - un voisin parallèle proche avec chevauchement longitudinal (face de cloison), ou
    - un voisin perpendiculaire avec une extrémité à moins de
      perpendicular_search_distance (bout de mur connecté).

    Sinon il est supprimé (parasite isolé).

    Args:
        segments: Segments après micro-fusion.
        min_length: Longueur en dessous de laquelle un segment est suspect (mètres).
        parallel_search_distance: Distance perpendiculaire maximale pour chercher
            une face parallèle (mètres).
        perpendicular_search_distance: Distance maximale entre extrémités pour
            chercher un mur perpendiculaire connecté (mètres).

    Returns:
        Segments nettoyés — seuls les parasites isolés sont supprimés.

    Example:
        >>> short_isolated = DetectedSegment(0.0, 0.0, 0.08, 0.0, "high", 0.3)
        >>> long_wall = DetectedSegment(5.0, 0.0, 8.0, 0.0, "high", 0.9)
        >>> result = clean_parasites([short_isolated, long_wall])
        >>> len(result)
        1
    """
    if not segments:
        return []

    kept: list[Segment] = []
    n_removed = 0

    for i, seg in enumerate(segments):
        if seg.length >= min_length:
            kept.append(seg)
            continue

        # Segment court : chercher une justification parmi les autres
        others = [s for j, s in enumerate(segments) if j != i]

        if _has_parallel_neighbor(seg, others, parallel_search_distance):
            kept.append(seg)
        elif _has_connected_perpendicular(seg, others, perpendicular_search_distance):
            kept.append(seg)
        else:
            n_removed += 1
            logger.debug(
                "clean_parasites : suppression segment (%.3fm) @ (%.2f,%.2f)-(%.2f,%.2f)",
                seg.length, seg.x1, seg.y1, seg.x2, seg.y2,
            )

    logger.info(
        "clean_parasites : %d -> %d segments (%d parasites supprimes).",
        len(segments),
        len(kept),
        n_removed,
    )
    return kept


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------

def _has_parallel_neighbor(
    seg: Segment,
    others: list[Segment],
    max_perp_dist: float,
    angle_tolerance_deg: float = 10.0,
) -> bool:
    """Vérifie qu'il existe un voisin parallèle avec chevauchement longitudinal.

    Critères :
    - Angle < angle_tolerance_deg (quasi-parallèle).
    - Distance perpendiculaire < max_perp_dist.
    - Chevauchement longitudinal > 0 (les segments se font face sur l'axe commun).

    Args:
        seg: Segment à tester.
        others: Autres segments.
        max_perp_dist: Distance perpendiculaire maximale (mètres).
        angle_tolerance_deg: Tolérance angulaire (degrés).

    Returns:
        True si un voisin parallèle avec chevauchement existe.
    """
    angle_tol_rad = float(np.deg2rad(angle_tolerance_deg))
    a_seg = _angle(seg)
    ux, uy = _unit_direction(seg)

    for other in others:
        # Critère 1 : parallélisme
        if _angle_diff(a_seg, _angle(other)) > angle_tol_rad:
            continue

        # Critère 2 : proximité perpendiculaire
        if _perp_dist(seg, other) > max_perp_dist:
            continue

        # Critère 3 : chevauchement longitudinal
        # Projeter les 4 extrémités sur l'axe du segment référence
        ref_pt = np.array([seg.x1, seg.y1])
        u = np.array([ux, uy])

        t_seg_min = 0.0
        t_seg_max = seg.length

        pts_other = np.array([
            [other.x1, other.y1],
            [other.x2, other.y2],
        ])
        t_other = (pts_other - ref_pt) @ u
        t_other_min = float(t_other.min())
        t_other_max = float(t_other.max())

        # Chevauchement = intersection des intervalles
        overlap = min(t_seg_max, t_other_max) - max(t_seg_min, t_other_min)
        if overlap > 0:
            return True

    return False


def _has_connected_perpendicular(
    seg: Segment,
    others: list[Segment],
    max_endpoint_dist: float,
    angle_min_deg: float = 60.0,
) -> bool:
    """Vérifie qu'il existe un voisin quasi-perpendiculaire connecté par une extrémité.

    Un segment est considéré connecté si une de ses extrémités est à moins de
    max_endpoint_dist d'une extrémité de seg, et que l'angle entre les deux
    segments est supérieur à angle_min_deg (non-parallèles).

    Args:
        seg: Segment à tester.
        others: Autres segments.
        max_endpoint_dist: Distance maximale entre extrémités (mètres).
        angle_min_deg: Angle minimal pour qu'un segment soit considéré non-parallèle.

    Returns:
        True si un voisin connecté perpendiculairement existe.
    """
    angle_min_rad = float(np.deg2rad(angle_min_deg))
    a_seg = _angle(seg)

    endpoints_seg = [
        np.array([seg.x1, seg.y1]),
        np.array([seg.x2, seg.y2]),
    ]

    for other in others:
        # Doit être suffisamment non-parallèle
        if _angle_diff(a_seg, _angle(other)) < angle_min_rad:
            continue

        endpoints_other = [
            np.array([other.x1, other.y1]),
            np.array([other.x2, other.y2]),
        ]

        for ep_seg in endpoints_seg:
            for ep_other in endpoints_other:
                dist = float(np.hypot(ep_seg[0] - ep_other[0], ep_seg[1] - ep_other[1]))
                if dist <= max_endpoint_dist:
                    return True

    return False


def _angle(seg: Segment) -> float:
    """Angle du segment en radians dans [0, π).

    Args:
        seg: Segment.

    Returns:
        Angle en radians.
    """
    return float(np.arctan2(seg.y2 - seg.y1, seg.x2 - seg.x1) % np.pi)


def _angle_diff(a1: float, a2: float) -> float:
    """Différence angulaire minimale dans [0, π/2].

    Args:
        a1: Premier angle (radians).
        a2: Deuxième angle (radians).

    Returns:
        Différence en radians.
    """
    diff = abs(a1 - a2) % np.pi
    return float(min(diff, np.pi - diff))


def _unit_direction(seg: Segment) -> tuple[float, float]:
    """Vecteur unitaire dans la direction du segment.

    Args:
        seg: Segment.

    Returns:
        ``(ux, uy)`` normalisé.
    """
    dx = seg.x2 - seg.x1
    dy = seg.y2 - seg.y1
    length = float(np.hypot(dx, dy))
    if length < 1e-9:
        return 1.0, 0.0
    return dx / length, dy / length


def _perp_dist(seg_a: Segment, seg_b: Segment) -> float:
    """Distance perpendiculaire du centre de seg_b à la droite portant seg_a.

    Args:
        seg_a: Segment de référence.
        seg_b: Segment dont on mesure l'écart.

    Returns:
        Distance perpendiculaire en mètres.
    """
    ux, uy = _unit_direction(seg_a)
    nx, ny = -uy, ux
    cx = (seg_b.x1 + seg_b.x2) / 2.0
    cy = (seg_b.y1 + seg_b.y2) / 2.0
    dx = cx - seg_a.x1
    dy = cy - seg_a.y1
    return float(abs(dx * nx + dy * ny))
