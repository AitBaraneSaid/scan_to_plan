"""Fusion ultra-conservative des fragments colinéaires issus de Hough.

Principe : ne recoller que les fragments évidents d'un même mur (gap < 5 cm).
Ne jamais fusionner des segments séparés par une ouverture (porte ≥ 70 cm,
fenêtre ≥ 30 cm). Avec max_gap=5 cm, aucun risque de franchir une ouverture.

Différence critique avec fuse_collinear_segments :
- max_gap : 5 cm (vs 20 cm) → les portes et fenêtres sont préservées
- perpendicular_tolerance : 2 cm (vs 3 cm) → plus strict
- Pas de require_overlap : le seul critère est le gap < max_gap
"""

from __future__ import annotations

import logging

import numpy as np

from scan2plan.detection.line_detection import DetectedSegment

logger = logging.getLogger(__name__)

# Alias de type pour la clarté
Segment = DetectedSegment


def micro_fuse_segments(
    segments: list[Segment],
    max_gap: float = 0.05,
    angle_tolerance_deg: float = 3.0,
    perpendicular_tolerance: float = 0.02,
    max_iterations: int = 10,
) -> list[Segment]:
    """Fusionne les fragments colinéaires séparés par des micro-gaps.

    Ne fusionne que les segments sur la même droite porteuse avec un gap
    longitudinal <= max_gap. Les gaps plus larges (ouvertures architecturales)
    sont intégralement préservés.

    Args:
        segments: Segments Hough bruts ou filtrés multi-slice.
        max_gap: Gap longitudinal maximum pour fusionner (mètres). Défaut 5 cm.
            Aucune ouverture architecturale (porte ≥ 70 cm, fenêtre ≥ 30 cm)
            ne sera franchie avec cette valeur.
        angle_tolerance_deg: Tolérance angulaire pour la colinéarité (degrés).
        perpendicular_tolerance: Distance perpendiculaire maximale entre les
            droites porteuses des deux segments (mètres).
        max_iterations: Nombre maximum d'itérations de fusion.

    Returns:
        Liste de segments avec les fragments recollés. Les ouvertures et
        cloisons courtes isolées sont préservées intactes.

    Example:
        >>> s1 = DetectedSegment(0.0, 0.0, 1.0, 0.0, "high", 0.9)
        >>> s2 = DetectedSegment(1.03, 0.0, 2.0, 0.0, "high", 0.9)
        >>> result = micro_fuse_segments([s1, s2], max_gap=0.05)
        >>> len(result)
        1
    """
    if len(segments) <= 1:
        return list(segments)

    angle_tol_rad = float(np.deg2rad(angle_tolerance_deg))
    current = list(segments)
    n_in = len(current)

    for iteration in range(max_iterations):
        current, fused_any = _one_fusion_pass(
            current, max_gap, angle_tol_rad, perpendicular_tolerance
        )
        if not fused_any:
            logger.debug(
                "micro_fuse_segments : convergence en %d iteration(s).", iteration + 1
            )
            break
    else:
        logger.debug(
            "micro_fuse_segments : max_iterations=%d atteint.", max_iterations
        )

    logger.info(
        "micro_fuse_segments : %d -> %d segments (max_gap=%.3fm).",
        n_in,
        len(current),
        max_gap,
    )
    return current


def _one_fusion_pass(
    segments: list[Segment],
    max_gap: float,
    angle_tol_rad: float,
    perp_tol: float,
) -> tuple[list[Segment], bool]:
    """Effectue une passe de fusion : parcourt toutes les paires éligibles.

    Args:
        segments: Segments courants.
        max_gap: Gap longitudinal maximal (mètres).
        angle_tol_rad: Tolérance angulaire (radians).
        perp_tol: Tolérance perpendiculaire (mètres).

    Returns:
        ``(nouvelle_liste, au_moins_une_fusion_effectuee)``.
    """
    used = [False] * len(segments)
    result: list[Segment] = []
    fused_any = False

    for i in range(len(segments)):
        if used[i]:
            continue
        partner = _find_fusable_partner(
            segments, i, used, max_gap, angle_tol_rad, perp_tol
        )
        if partner is not None:
            result.append(_merge_two_segments(segments[i], segments[partner]))
            used[i] = True
            used[partner] = True
            fused_any = True
        else:
            result.append(segments[i])
            used[i] = True

    return result, fused_any


def _find_fusable_partner(
    segments: list[Segment],
    i: int,
    used: list[bool],
    max_gap: float,
    angle_tol_rad: float,
    perp_tol: float,
) -> int | None:
    """Cherche le premier segment j > i fusable avec segments[i].

    Critères (les 3 doivent être satisfaits) :
    1. Angle entre les deux segments < angle_tol_rad.
    2. Distance perpendiculaire entre les droites porteuses < perp_tol.
    3. Gap longitudinal <= max_gap (chevauchement autorisé : gap < 0).

    Args:
        segments: Liste des segments.
        i: Indice du segment de référence.
        used: Masque des segments déjà appariés.
        max_gap: Gap longitudinal maximal (mètres).
        angle_tol_rad: Tolérance angulaire (radians).
        perp_tol: Tolérance perpendiculaire (mètres).

    Returns:
        Indice du partenaire trouvé, ou ``None``.
    """
    si = segments[i]
    ai = _angle(si)

    for j in range(i + 1, len(segments)):
        if used[j]:
            continue
        sj = segments[j]
        aj = _angle(sj)

        # Critère 1 : colinéarité angulaire
        if _angle_diff(ai, aj) > angle_tol_rad:
            continue

        # Critère 2 : proximité perpendiculaire
        if _perpendicular_distance(si, sj) > perp_tol:
            continue

        # Critère 3 : gap longitudinal <= max_gap (tolérance numérique 0.1 mm)
        gap = _compute_gap(si, sj)
        if gap <= max_gap + 1e-4:
            return j

    return None


def _compute_gap(seg_a: Segment, seg_b: Segment) -> float:
    """Calcule le gap longitudinal entre deux segments quasi-colinéaires.

    Projette les 4 extrémités sur la direction commune (moyenne des deux
    vecteurs unitaires). Le gap est la distance entre la fin du premier
    segment et le début du second sur cet axe.

    Args:
        seg_a: Premier segment.
        seg_b: Deuxième segment.

    Returns:
        Gap en mètres. Négatif si les segments se chevauchent.
    """
    # Direction commune : moyenne normalisée des deux vecteurs unitaires
    ux, uy = _unit_direction(seg_a)
    vx, vy = _unit_direction(seg_b)
    # Aligner les vecteurs (s'assurer qu'ils pointent dans le même sens)
    if ux * vx + uy * vy < 0:
        vx, vy = -vx, -vy
    mx = (ux + vx) / 2.0
    my = (uy + vy) / 2.0
    norm = float(np.hypot(mx, my))
    if norm < 1e-9:
        mx, my = ux, uy
    else:
        mx /= norm
        my /= norm

    # Projeter les 4 extrémités sur l'axe commun
    pts = [
        (seg_a.x1, seg_a.y1),
        (seg_a.x2, seg_a.y2),
        (seg_b.x1, seg_b.y1),
        (seg_b.x2, seg_b.y2),
    ]
    proj = [px * mx + py * my for px, py in pts]

    # Intervalle de seg_a et seg_b sur cet axe
    a_min = min(proj[0], proj[1])
    a_max = max(proj[0], proj[1])
    b_min = min(proj[2], proj[3])
    b_max = max(proj[2], proj[3])

    # Gap = max(0, séparation) avec signe : négatif = chevauchement
    if b_min >= a_min:
        return float(b_min - a_max)   # seg_b après seg_a
    else:
        return float(a_min - b_max)   # seg_a après seg_b


def _merge_two_segments(seg_a: Segment, seg_b: Segment) -> Segment:
    """Fusionne deux segments colinéaires en un seul.

    Calcule la droite porteuse commune par SVD sur les 4 extrémités.
    Le segment résultant va de la projection extrême min à max sur cet axe.

    Args:
        seg_a: Premier segment.
        seg_b: Deuxième segment.

    Returns:
        Segment fusionné sur la droite commune.
    """
    pts = np.array([
        [seg_a.x1, seg_a.y1],
        [seg_a.x2, seg_a.y2],
        [seg_b.x1, seg_b.y1],
        [seg_b.x2, seg_b.y2],
    ])

    # Pondération par longueur : les points des segments longs comptent plus
    la = max(seg_a.length, 1e-9)
    lb = max(seg_b.length, 1e-9)
    weights = np.array([la, la, lb, lb])
    weights /= weights.sum()

    # Centre de masse pondéré
    centroid = (pts * weights[:, None]).sum(axis=0)

    # SVD sur les points centrés → direction principale
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]  # vecteur unitaire de la droite porteuse

    # Projeter les 4 points sur la droite
    projections = centered @ direction

    t_min = float(projections.min())
    t_max = float(projections.max())

    x1 = float(centroid[0] + t_min * direction[0])
    y1 = float(centroid[1] + t_min * direction[1])
    x2 = float(centroid[0] + t_max * direction[0])
    y2 = float(centroid[1] + t_max * direction[1])

    return DetectedSegment(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        source_slice=seg_a.source_slice,
        confidence=max(seg_a.confidence, seg_b.confidence),
    )


# ---------------------------------------------------------------------------
# Helpers géométriques privés
# ---------------------------------------------------------------------------


def _angle(seg: Segment) -> float:
    """Angle du segment en radians dans [0, π).

    Args:
        seg: Segment.

    Returns:
        Angle en radians.
    """
    return float(np.arctan2(seg.y2 - seg.y1, seg.x2 - seg.x1) % np.pi)


def _angle_diff(a1: float, a2: float) -> float:
    """Différence angulaire minimale entre deux angles dans [0, π).

    Args:
        a1: Premier angle (radians).
        a2: Deuxième angle (radians).

    Returns:
        Différence en radians dans [0, π/2].
    """
    diff = abs(a1 - a2) % np.pi
    return float(min(diff, np.pi - diff))


def _unit_direction(seg: Segment) -> tuple[float, float]:
    """Vecteur unitaire dans la direction du segment.

    Args:
        seg: Segment.

    Returns:
        ``(ux, uy)`` normalisé. Retourne ``(1, 0)`` si longueur nulle.
    """
    dx = seg.x2 - seg.x1
    dy = seg.y2 - seg.y1
    length = float(np.hypot(dx, dy))
    if length < 1e-9:
        return 1.0, 0.0
    return dx / length, dy / length


def _perpendicular_distance(seg_a: Segment, seg_b: Segment) -> float:
    """Distance perpendiculaire entre les droites porteuses de deux segments.

    Utilise le centre de seg_b et sa distance à la droite infinie portant seg_a.

    Args:
        seg_a: Segment de référence.
        seg_b: Segment dont on mesure l'écart.

    Returns:
        Distance perpendiculaire en mètres.
    """
    ux, uy = _unit_direction(seg_a)
    # Normale à seg_a
    nx, ny = -uy, ux

    # Centre de seg_b
    cx = (seg_b.x1 + seg_b.x2) / 2.0
    cy = (seg_b.y1 + seg_b.y2) / 2.0

    # Distance signée du centre de seg_b à la droite porteuse de seg_a
    dx = cx - seg_a.x1
    dy = cy - seg_a.y1
    return float(abs(dx * nx + dy * ny))
