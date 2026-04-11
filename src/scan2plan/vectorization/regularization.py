"""Régularisation géométrique : snapping angulaire sur les orientations dominantes."""

from __future__ import annotations

import logging

import numpy as np

from scan2plan.detection.line_detection import DetectedSegment

logger = logging.getLogger(__name__)


def regularize_segments(
    segments: list[DetectedSegment],
    dominant_angles: list[float],
    snap_tolerance_deg: float = 5.0,
) -> list[DetectedSegment]:
    """Snappe les segments sur les directions dominantes.

    Pour chaque segment :
    1. Calculer son angle θ (modulo π).
    2. Trouver l'angle dominant le plus proche (en tenant compte de la
       symétrie à π).
    3. Si |θ - θ_dominant| < snap_tolerance_deg :
       a. Pivoter le segment autour de son centre pour l'aligner exactement.
       b. La longueur est conservée.
    4. Sinon, laisser le segment inchangé (mur oblique).

    Args:
        segments: Liste de segments de murs à régulariser.
        dominant_angles: Angles dominants en radians (issus de
            ``detect_dominant_orientations``).
        snap_tolerance_deg: Tolérance angulaire pour le snapping (degrés).
            Un segment dont l'écart avec le dominant le plus proche est
            supérieur à cette valeur n'est pas modifié.

    Returns:
        Nouvelle liste de ``DetectedSegment`` régularisés. Les segments
        snappés ont leur angle modifié ; les autres sont retournés tels quels.
        Les attributs ``source_slice`` et ``confidence`` sont préservés.

    Example:
        >>> segs = [DetectedSegment(0, 0, 1, 0.02, "high", 1.0)]  # ≈1° off
        >>> regularize_segments(segs, [0.0], snap_tolerance_deg=5.0)
        [DetectedSegment(0.0, 0.01, 1.0, 0.01, ...)]
    """
    if not dominant_angles or not segments:
        logger.debug("Régularisation ignorée : pas d'angles dominants ou liste vide.")
        return list(segments)

    snap_tol_rad = float(np.deg2rad(snap_tolerance_deg))
    result: list[DetectedSegment] = []
    snapped_count = 0

    for seg in segments:
        angle = _segment_angle_rad(seg)
        target, diff = _nearest_dominant(angle, dominant_angles)

        if diff <= snap_tol_rad:
            result.append(_snap_segment(seg, target))
            snapped_count += 1
        else:
            result.append(seg)

    logger.info(
        "Régularisation (tolérance=%.1f°) : %d / %d segments snappés.",
        snap_tolerance_deg,
        snapped_count,
        len(segments),
    )
    return result


def align_parallel_segments(
    segments: list[DetectedSegment],
    dominant_angles: list[float],
    alignment_tolerance: float = 0.02,
) -> list[DetectedSegment]:
    """Aligne les segments parallèles proches sur une même droite porteuse.

    Deux segments sont candidats à l'alignement si :
    - Ils ont la même direction dominante (angle le plus proche identique).
    - La distance perpendiculaire entre leurs droites support est inférieure
      à ``alignment_tolerance``.

    Dans ce cas, les segments sont projetés sur la droite médiane pondérée
    par la longueur (un mur long déplace moins que la moyenne).

    Args:
        segments: Liste de segments (idéalement après régularisation angulaire).
        dominant_angles: Angles dominants en radians.
        alignment_tolerance: Distance perpendiculaire maximale entre deux
            droites support pour les aligner (mètres).

    Returns:
        Nouvelle liste de ``DetectedSegment`` avec les segments parallèles
        proches alignés sur la même droite porteuse.
    """
    if not dominant_angles or not segments:
        return list(segments)

    from scan2plan.utils.geometry import perpendicular_distance_segment_to_segment

    # Grouper les segments par direction dominante
    groups: dict[int, list[DetectedSegment]] = {}
    for seg in segments:
        angle = _segment_angle_rad(seg)
        _, _ = _nearest_dominant(angle, dominant_angles)
        dom_idx = _nearest_dominant_index(angle, dominant_angles)
        groups.setdefault(dom_idx, []).append(seg)

    result: list[DetectedSegment] = []

    for dom_idx, group in groups.items():
        target_angle = dominant_angles[dom_idx]
        aligned = _align_group(group, target_angle, alignment_tolerance,
                               perpendicular_distance_segment_to_segment)
        result.extend(aligned)

    logger.info(
        "align_parallel_segments : %d segments → %d après alignement.",
        len(segments),
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------

def _segment_angle_rad(seg: DetectedSegment) -> float:
    """Retourne l'angle du segment en radians dans [0, π).

    Args:
        seg: Segment de mur.

    Returns:
        Angle en radians.
    """
    return float(np.arctan2(seg.y2 - seg.y1, seg.x2 - seg.x1) % np.pi)


def _nearest_dominant(
    angle: float,
    orientations: list[float],
) -> tuple[float, float]:
    """Trouve l'orientation dominante la plus proche d'un angle.

    Tient compte de la symétrie à π : un segment à 179° est proche de 0°.

    Args:
        angle: Angle du segment en radians (dans [0, π)).
        orientations: Orientations dominantes en radians.

    Returns:
        ``(target_angle, diff_rad)`` — orientation cible et écart minimal
        en radians (dans [0, π/2]).
    """
    best_target = orientations[0]
    best_diff = float("inf")

    for o in orientations:
        raw_diff = abs(angle - o) % np.pi
        diff = min(raw_diff, np.pi - raw_diff)
        if diff < best_diff:
            best_diff = diff
            best_target = o

    return best_target, best_diff


def _nearest_dominant_index(angle: float, orientations: list[float]) -> int:
    """Retourne l'index de l'orientation dominante la plus proche.

    Args:
        angle: Angle en radians.
        orientations: Liste d'orientations dominantes.

    Returns:
        Index dans ``orientations``.
    """
    best_idx = 0
    best_diff = float("inf")
    for i, o in enumerate(orientations):
        raw_diff = abs(angle - o) % np.pi
        diff = min(raw_diff, np.pi - raw_diff)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return best_idx


def _snap_segment(seg: DetectedSegment, target_angle: float) -> DetectedSegment:
    """Pivote un segment autour de son centre pour l'aligner sur target_angle.

    La longueur du segment est conservée. Les attributs ``source_slice`` et
    ``confidence`` sont préservés.

    Args:
        seg: Segment original.
        target_angle: Angle cible en radians.

    Returns:
        Nouveau ``DetectedSegment`` avec l'angle ajusté.
    """
    cx = (seg.x1 + seg.x2) / 2.0
    cy = (seg.y1 + seg.y2) / 2.0
    half_len = seg.length / 2.0
    dx = float(np.cos(target_angle)) * half_len
    dy = float(np.sin(target_angle)) * half_len
    return DetectedSegment(
        x1=cx - dx,
        y1=cy - dy,
        x2=cx + dx,
        y2=cy + dy,
        source_slice=seg.source_slice,
        confidence=seg.confidence,
    )


def _align_group(
    group: list[DetectedSegment],
    target_angle: float,
    tolerance: float,
    dist_fn: "callable",
) -> list[DetectedSegment]:
    """Aligne les segments d'un groupe de même direction dominante.

    Itère jusqu'à convergence : à chaque passe, fusionne la première paire
    dont la distance perpendiculaire est < ``tolerance``.

    Args:
        group: Segments de même direction dominante.
        target_angle: Angle dominant du groupe (radians).
        tolerance: Seuil de distance perpendiculaire (mètres).
        dist_fn: Fonction ``perpendicular_distance_segment_to_segment``.

    Returns:
        Groupe aligné (même nombre de segments ou moins si fusion).
    """
    if len(group) <= 1:
        return list(group)

    current = list(group)
    merged = True

    while merged and len(current) > 1:
        current, merged = _one_merge_pass(current, target_angle, tolerance, dist_fn)

    return current


def _one_merge_pass(
    segs: list[DetectedSegment],
    target_angle: float,
    tolerance: float,
    dist_fn: "callable",
) -> tuple[list[DetectedSegment], bool]:
    """Effectue une passe de fusion : fusionne toutes les paires éligibles.

    Args:
        segs: Segments courants.
        target_angle: Direction de la droite porteuse (radians).
        tolerance: Seuil de distance perpendiculaire (mètres).
        dist_fn: Fonction de distance perpendiculaire.

    Returns:
        ``(nouvelle_liste, au_moins_une_fusion_effectuée)``.
    """
    used = [False] * len(segs)
    result: list[DetectedSegment] = []
    merged_any = False

    for i in range(len(segs)):
        if used[i]:
            continue
        partner = _find_close_partner(segs, i, used, tolerance, dist_fn)
        if partner is not None:
            result.append(_merge_on_common_line(segs[i], segs[partner], target_angle))
            used[i] = True
            used[partner] = True
            merged_any = True
        else:
            result.append(segs[i])
            used[i] = True

    return result, merged_any


def _find_close_partner(
    segs: list[DetectedSegment],
    i: int,
    used: list[bool],
    tolerance: float,
    dist_fn: "callable",
) -> int | None:
    """Cherche le premier segment j > i non utilisé à distance < tolerance de segs[i].

    Args:
        segs: Liste de segments.
        i: Indice du segment de référence.
        used: Masque des segments déjà appariés.
        tolerance: Seuil de distance (mètres).
        dist_fn: Fonction de distance perpendiculaire.

    Returns:
        Indice du partenaire trouvé, ou ``None``.
    """
    for j in range(i + 1, len(segs)):
        if used[j]:
            continue
        if dist_fn(segs[i].as_tuple(), segs[j].as_tuple()) < tolerance:
            return j
    return None


def _merge_on_common_line(
    s1: DetectedSegment,
    s2: DetectedSegment,
    target_angle: float,
) -> DetectedSegment:
    """Fusionne deux segments sur une droite porteuse commune.

    La droite porteuse est déterminée par le centre de masse pondéré par la
    longueur des deux segments. La direction est ``target_angle``.
    Les extrémités sont les projections les plus éloignées des 4 points.

    Args:
        s1: Premier segment.
        s2: Deuxième segment.
        target_angle: Direction de la droite porteuse (radians).

    Returns:
        Segment fusionné sur la droite commune.
    """
    ux = float(np.cos(target_angle))
    uy = float(np.sin(target_angle))
    # Vecteur perpendiculaire
    nx, ny = -uy, ux

    # Centre de masse pondéré par longueur → décalage perpendiculaire moyen
    l1, l2 = s1.length, s2.length
    total_len = l1 + l2 if (l1 + l2) > 1e-9 else 1.0

    # Centre de chaque segment
    c1x, c1y = (s1.x1 + s1.x2) / 2.0, (s1.y1 + s1.y2) / 2.0
    c2x, c2y = (s2.x1 + s2.x2) / 2.0, (s2.y1 + s2.y2) / 2.0

    # Décalage perpendiculaire moyen pondéré
    d1 = c1x * nx + c1y * ny
    d2 = c2x * nx + c2y * ny
    d_mean = (d1 * l1 + d2 * l2) / total_len

    # Projeter les 4 extrémités sur l'axe parallèle
    pts = [(s1.x1, s1.y1), (s1.x2, s1.y2), (s2.x1, s2.y1), (s2.x2, s2.y2)]
    projections = [px * ux + py * uy for px, py in pts]

    t_min = min(projections)
    t_max = max(projections)

    # Reconstruire les extrémités sur la droite porteuse
    x1 = t_min * ux + d_mean * nx
    y1 = t_min * uy + d_mean * ny
    x2 = t_max * ux + d_mean * nx
    y2 = t_max * uy + d_mean * ny

    return DetectedSegment(
        x1=x1, y1=y1, x2=x2, y2=y2,
        source_slice=s1.source_slice,
        confidence=max(s1.confidence, s2.confidence),
    )
