"""Régularisation angulaire pure : snap sur les directions dominantes.

Ce module fait UNIQUEMENT de la rotation autour du centre de chaque segment.
Pas de fusion transversale, pas de déplacement, pas de suppression.
Le nombre de segments en sortie est identique au nombre en entrée.

Différence critique avec regularization.py :
- Pas d'``align_parallel_segments`` (cause de perte de segments face-de-mur).
- Détection intégrée des orientations dominantes (histogramme pondéré + lissage).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from scan2plan.detection.line_detection import DetectedSegment

logger = logging.getLogger(__name__)

Segment = DetectedSegment


def detect_dominant_orientations(
    segments: list[Segment],
    bin_size_deg: float = 1.0,
    min_peak_ratio: float = 0.1,
) -> list[float]:
    """Détecte les orientations dominantes des murs par histogramme angulaire.

    Algorithme :
    1. Pour chaque segment, calculer l'angle (modulo π).
    2. Pondérer par la longueur du segment.
    3. Construire un histogramme angulaire pondéré (bins de ``bin_size_deg``).
    4. Lisser (filtre gaussien σ=2 bins).
    5. Détecter les pics (prominence > ``min_peak_ratio`` × max).
    6. Retourner les angles des pics significatifs.

    Args:
        segments: Segments de murs (issus du multi-slice filter).
        bin_size_deg: Taille d'un bin angulaire (degrés). Défaut 1°.
        min_peak_ratio: Hauteur minimale d'un pic relative au maximum pour
            être retenu. Défaut 0.1 (10 % du pic dominant).

    Returns:
        Liste d'angles dominants en radians dans [0, π).
        Typiquement 2 angles quasi-perpendiculaires pour un bâtiment orthogonal.
        Retourne ``[0.0]`` si aucun pic n'est trouvé.

    Example:
        >>> segs_h = [DetectedSegment(0, 0, 3, 0, "high", 0.9)] * 10
        >>> segs_v = [DetectedSegment(0, 0, 0, 3, "high", 0.9)] * 10
        >>> angles = detect_dominant_orientations(segs_h + segs_v)
        >>> len(angles)
        2
    """
    if not segments:
        return [0.0]

    n_bins = round(180.0 / bin_size_deg)
    histogram = np.zeros(n_bins, dtype=float)

    for seg in segments:
        angle_rad = float(np.arctan2(seg.y2 - seg.y1, seg.x2 - seg.x1) % np.pi)
        bin_idx = int(np.degrees(angle_rad) / bin_size_deg) % n_bins
        histogram[bin_idx] += seg.length

    # Lissage gaussien (σ = 2 bins) avec wrap-around (angulaire périodique)
    smoothed = gaussian_filter1d(histogram, sigma=2.0, mode="wrap")

    # find_peaks ne détecte pas les pics aux bords du tableau (indices 0 et n-1).
    # On travaille sur une copie doublée pour gérer la périodicité (modulo π),
    # puis on ramène les indices dans [0, n_bins).
    doubled = np.concatenate([smoothed, smoothed])
    peak_threshold = float(smoothed.max()) * min_peak_ratio
    raw_peaks, _ = find_peaks(doubled, height=peak_threshold, distance=5)

    # Ne garder que les pics dans la première période, dédupliqués
    seen: set[int] = set()
    peaks_list: list[int] = []
    for p in raw_peaks:
        canonical = int(p) % n_bins
        if canonical not in seen:
            seen.add(canonical)
            peaks_list.append(canonical)
    peaks = np.array(peaks_list) if peaks_list else np.array([], dtype=int)

    if len(peaks) == 0:
        # Fallback : pic global
        peaks = np.array([int(np.argmax(smoothed))])

    dominant_angles = [float(np.deg2rad(p * bin_size_deg)) for p in peaks]

    logger.info(
        "detect_dominant_orientations : %d pic(s) détecté(s) sur %d segments : %s",
        len(dominant_angles),
        len(segments),
        [f"{np.degrees(a):.1f}°" for a in dominant_angles],
    )
    return dominant_angles


def snap_angles(
    segments: list[Segment],
    dominant_angles: list[float],
    tolerance_deg: float = 5.0,
) -> list[Segment]:
    """Snappe chaque segment sur la direction dominante la plus proche.

    Pour chaque segment :
    1. Calculer son angle θ (modulo π).
    2. Trouver la direction dominante la plus proche θ_dom.
    3. Si |θ - θ_dom| < tolerance_deg :
       - Pivoter autour du CENTRE du segment.
       - Conserver la longueur exacte.
       - Le segment ne se déplace PAS transversalement.
    4. Sinon : laisser le segment inchangé (mur oblique réel).

    Cette fonction ne fait QUE de la rotation.
    Pas de déplacement, pas de fusion, pas de suppression.

    Args:
        segments: Segments à régulariser.
        dominant_angles: Angles dominants en radians (issus de
            ``detect_dominant_orientations`` ou fournis manuellement).
        tolerance_deg: Tolérance angulaire pour le snapping (degrés).
            Au-delà, le segment est considéré oblique et laissé intact.

    Returns:
        Nouvelle liste de ``Segment`` avec les angles corrigés.
        Même nombre de segments qu'en entrée.

    Example:
        >>> s = DetectedSegment(0, 0, 1, 0.03, "high", 0.9)  # ≈1.7° off
        >>> result = snap_angles([s], [0.0], tolerance_deg=5.0)
        >>> abs(result[0].y2 - result[0].y1) < 0.001  # horizontal après snap
        True
    """
    if not segments:
        return []
    if not dominant_angles:
        logger.debug("snap_angles : pas d'angles dominants, retour sans modification.")
        return list(segments)

    tol_rad = float(np.deg2rad(tolerance_deg))
    result: list[Segment] = []
    snapped_count = 0

    for seg in segments:
        angle = float(np.arctan2(seg.y2 - seg.y1, seg.x2 - seg.x1) % np.pi)
        target, diff = _nearest_dominant(angle, dominant_angles)

        if diff <= tol_rad:
            result.append(_rotate_to_angle(seg, target))
            snapped_count += 1
        else:
            result.append(seg)

    logger.info(
        "snap_angles (tolérance=%.1f°) : %d / %d segments snappés.",
        tolerance_deg,
        snapped_count,
        len(segments),
    )
    return result


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------


def _nearest_dominant(
    angle: float,
    dominant_angles: list[float],
) -> tuple[float, float]:
    """Trouve l'orientation dominante la plus proche d'un angle dans [0, π).

    Tient compte de la symétrie à π.

    Args:
        angle: Angle du segment en radians.
        dominant_angles: Orientations dominantes en radians.

    Returns:
        ``(target_angle, diff_rad)`` — orientation cible et écart minimal.
    """
    best_target = dominant_angles[0]
    best_diff = float("inf")

    for dom in dominant_angles:
        raw_diff = abs(angle - dom) % np.pi
        diff = float(min(raw_diff, np.pi - raw_diff))
        if diff < best_diff:
            best_diff = diff
            best_target = dom

    return best_target, best_diff


def _rotate_to_angle(seg: Segment, target_angle: float) -> Segment:
    """Pivote un segment autour de son centre vers ``target_angle``.

    La longueur est exactement conservée. Le centre ne bouge pas.

    Args:
        seg: Segment original.
        target_angle: Angle cible en radians.

    Returns:
        Nouveau ``Segment`` avec l'angle ajusté, même centre et même longueur.
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
