"""Filtrage des segments de murs par comparaison multi-slice.

Principe : un mur est présent à toutes les hauteurs, un meuble seulement
à certaines. La slice HAUTE est la référence car elle est au-dessus du
mobilier et des ouvertures.

Classifications :
- "wall"             : segment présent en HIGH (avec ou sans correspondance en MID/LOW)
- "furniture"        : segment absent en HIGH, présent uniquement en MID et/ou LOW
- "window_candidate" : présent en HIGH et LOW, absent en MID
- "door_candidate"   : segment HIGH sans correspondance en LOW (portée ouverte)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from scan2plan.detection.line_detection import DetectedSegment
from scan2plan.utils.geometry import (
    angle_between_segments,
    perpendicular_distance_segment_to_segment,
)

logger = logging.getLogger(__name__)


@dataclass
class SegmentMatch:
    """Correspondance d'un segment haut avec ses équivalents dans les autres slices.

    Attributes:
        segment_high: Segment de référence (slice haute).
        segment_mid: Segment correspondant en slice médiane, ou None.
        segment_low: Segment correspondant en slice basse, ou None.
        classification: Résultat de la classification ("wall", "furniture",
            "window_candidate", "door_candidate").
    """

    segment_high: DetectedSegment
    segment_mid: DetectedSegment | None
    segment_low: DetectedSegment | None
    classification: str = "wall"


def match_segments_across_slices(
    segments_by_slice: dict[str, list[DetectedSegment]],
    angle_tolerance_deg: float = 5.0,
    distance_tolerance: float = 0.10,
) -> list[SegmentMatch]:
    """Cherche les correspondances entre segments de différentes slices.

    Pour chaque segment de la slice haute (référence), cherche le meilleur
    segment correspondant dans les slices médiane et basse selon deux critères :
    - angle entre les segments < ``angle_tolerance_deg``
    - distance perpendiculaire entre les droites support < ``distance_tolerance``

    Les segments présents uniquement en slice médiane ou basse (sans
    correspondance en haute) sont collectés séparément et classifiés
    "furniture".

    Args:
        segments_by_slice: Dictionnaire ``{"high": [...], "mid": [...], "low": [...]}``.
            Les clés "mid" et "low" sont optionnelles.
        angle_tolerance_deg: Tolérance angulaire pour la correspondance (degrés).
        distance_tolerance: Distance perpendiculaire maximale pour la correspondance (mètres).

    Returns:
        Liste de ``SegmentMatch`` — un par segment haut, plus un par segment
        sans correspondance haut (furniture).
    """
    segs_high = segments_by_slice.get("high", [])
    segs_mid = segments_by_slice.get("mid", [])
    segs_low = segments_by_slice.get("low", [])

    angle_tol_rad = float(np.deg2rad(angle_tolerance_deg))
    matches: list[SegmentMatch] = []

    # Suivre quels segments mid/low ont été utilisés (pour détecter le mobilier)
    used_mid = [False] * len(segs_mid)
    used_low = [False] * len(segs_low)

    for seg_h in segs_high:
        best_mid, idx_mid = _find_best_match(seg_h, segs_mid, angle_tol_rad, distance_tolerance)
        best_low, idx_low = _find_best_match(seg_h, segs_low, angle_tol_rad, distance_tolerance)

        if idx_mid is not None:
            used_mid[idx_mid] = True
        if idx_low is not None:
            used_low[idx_low] = True

        match = SegmentMatch(segment_high=seg_h, segment_mid=best_mid, segment_low=best_low)
        matches.append(match)

    # Segments mid non appariés à un haut → mobilier (ou fenêtre basse)
    for i, seg_m in enumerate(segs_mid):
        if used_mid[i]:
            continue
        # Chercher si ce segment mid a un correspondant low
        best_low, idx_low = _find_best_match(seg_m, segs_low, angle_tol_rad, distance_tolerance)
        if idx_low is not None:
            used_low[idx_low] = True
        # Pas de segment high → mobilier quelle que soit la présence en low
        dummy = SegmentMatch(
            segment_high=seg_m,  # on met le seg mid comme "référence" pour la classification
            segment_mid=None,
            segment_low=best_low,
            classification="furniture",
        )
        matches.append(dummy)

    # Segments low non appariés → mobilier bas
    for i, seg_l in enumerate(segs_low):
        if used_low[i]:
            continue
        dummy = SegmentMatch(
            segment_high=seg_l,
            segment_mid=None,
            segment_low=None,
            classification="furniture",
        )
        matches.append(dummy)

    logger.info(
        "match_segments_across_slices : %d high, %d mid, %d low → %d matches.",
        len(segs_high),
        len(segs_mid),
        len(segs_low),
        len(matches),
    )
    return matches


def classify_segments(matches: list[SegmentMatch]) -> list[DetectedSegment]:
    """Classifie les correspondances et retourne les segments de murs confirmés.

    Règles (la slice haute est la RÉFÉRENCE) :
    - HIGH présent, quel que soit MID/LOW → "wall"
    - HIGH + LOW, MID absent → "window_candidate" (porte-fenêtre ou fenêtre basse)
    - HIGH présent, LOW absent → "door_candidate" (ouverture en bas possible)
    - HIGH absent (segment orphelin MID ou LOW) → "furniture"

    Les segments "wall", "window_candidate", "door_candidate" sont retournés
    (le segment haut est conservé comme représentant). Les "furniture" sont filtrés.

    Args:
        matches: Liste de ``SegmentMatch`` issue de ``match_segments_across_slices``.

    Returns:
        Liste de ``DetectedSegment`` classifiés comme murs (ou candidats ouverture).
        Les meubles sont exclus.
    """
    walls: list[DetectedSegment] = []
    n_wall = n_furniture = n_window = n_door = 0

    for match in matches:
        # Segments déjà pré-classifiés "furniture" (orphelins mid/low sans high)
        if match.classification == "furniture":
            n_furniture += 1
            continue

        has_high = True  # segment_high est toujours présent ici
        has_mid = match.segment_mid is not None
        has_low = match.segment_low is not None

        if has_high and has_low and not has_mid:
            match.classification = "window_candidate"
            n_window += 1
        elif has_high and not has_low:
            match.classification = "door_candidate"
            n_door += 1
        else:
            match.classification = "wall"
            n_wall += 1

        walls.append(match.segment_high)

    logger.info(
        "classify_segments : %d murs, %d mobilier, %d fenêtres candidates, %d portes candidates.",
        n_wall,
        n_furniture,
        n_window,
        n_door,
    )
    return walls


def get_door_candidates(matches: list[SegmentMatch]) -> list[DetectedSegment]:
    """Retourne les segments classifiés comme candidats portes.

    Args:
        matches: Liste de ``SegmentMatch`` après classification (``classify_segments``
            doit avoir été appelé avant).

    Returns:
        Liste de ``DetectedSegment`` de type "door_candidate".
    """
    return [m.segment_high for m in matches if m.classification == "door_candidate"]


def get_window_candidates(matches: list[SegmentMatch]) -> list[DetectedSegment]:
    """Retourne les segments classifiés comme candidats fenêtres.

    Args:
        matches: Liste de ``SegmentMatch`` après classification.

    Returns:
        Liste de ``DetectedSegment`` de type "window_candidate".
    """
    return [m.segment_high for m in matches if m.classification == "window_candidate"]


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------

def _find_best_match(
    reference: DetectedSegment,
    candidates: list[DetectedSegment],
    angle_tol_rad: float,
    distance_tol: float,
) -> tuple[DetectedSegment | None, int | None]:
    """Cherche le meilleur segment correspondant parmi les candidats.

    Le meilleur candidat est celui qui minimise la distance perpendiculaire,
    sous réserve de satisfaire les deux tolérances.

    Args:
        reference: Segment de référence.
        candidates: Liste de segments candidats.
        angle_tol_rad: Tolérance angulaire en radians.
        distance_tol: Tolérance de distance perpendiculaire en mètres.

    Returns:
        ``(best_candidate, index)`` ou ``(None, None)`` si aucune correspondance.
    """
    best: DetectedSegment | None = None
    best_idx: int | None = None
    best_dist = float("inf")

    for idx, cand in enumerate(candidates):
        angle = angle_between_segments(reference.as_tuple(), cand.as_tuple())
        if angle > angle_tol_rad:
            continue
        dist = perpendicular_distance_segment_to_segment(reference.as_tuple(), cand.as_tuple())
        if dist < distance_tol and dist < best_dist:
            best_dist = dist
            best = cand
            best_idx = idx

    return best, best_idx
