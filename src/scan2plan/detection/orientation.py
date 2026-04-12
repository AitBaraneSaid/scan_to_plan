"""Détection des orientations dominantes du bâtiment par histogramme angulaire.

Principe : calculer l'angle de chaque segment de mur détecté, pondérer par la
longueur (un mur long compte plus qu'un court), construire un histogramme angulaire
lissé et détecter les pics.

Pas de dépendance à Open3D : l'histogramme est calculé directement sur les
segments détectés plutôt que sur les normales du nuage de points.
"""

from __future__ import annotations

import logging

import numpy as np

from scan2plan.detection.line_detection import DetectedSegment

logger = logging.getLogger(__name__)


def detect_dominant_orientations(
    segments: list[DetectedSegment],
    bin_size_deg: float = 1.0,
    min_peak_ratio: float = 0.1,
) -> list[float]:
    """Détecte les orientations dominantes des murs par histogramme angulaire.

    Algorithme :
    1. Calculer l'angle de chaque segment (modulo 180°).
    2. Pondérer chaque contribution par la longueur du segment.
    3. Construire un histogramme angulaire pondéré sur [0°, 180°].
    4. Lisser avec un filtre gaussien (σ = 2 bins).
    5. Détecter les pics via scipy.signal.find_peaks.
    6. Retourner les angles des pics dont la hauteur dépasse
       ``min_peak_ratio × max_peak``.

    Args:
        segments: Liste de segments de murs détectés.
        bin_size_deg: Largeur de chaque bin en degrés. Défaut : 1°.
        min_peak_ratio: Fraction du pic maximal en dessous de laquelle un pic
            est ignoré. Défaut : 0.1 (pics > 10 % du max).

    Returns:
        Liste d'angles dominants en radians, triés par importance décroissante.
        Liste vide si aucun segment ou aucun pic significatif.

    Example:
        >>> segs = [DetectedSegment(0, 0, 4, 0, "high", 1.0),  # horizontal
        ...         DetectedSegment(0, 0, 0, 3, "high", 1.0)]  # vertical
        >>> detect_dominant_orientations(segs)
        [0.0, 1.5707...]
    """
    if not segments:
        logger.warning("detect_dominant_orientations : aucun segment fourni.")
        return []

    n_bins = max(1, int(round(180.0 / bin_size_deg)))
    hist = np.zeros(n_bins, dtype=np.float64)

    for seg in segments:
        angle_rad = float(np.arctan2(seg.y2 - seg.y1, seg.x2 - seg.x1)) % np.pi
        angle_deg = float(np.degrees(angle_rad))
        # Indice du bin (clamp au dernier si exactement 180°)
        bin_idx = min(int(angle_deg / bin_size_deg), n_bins - 1)
        hist[bin_idx] += seg.length

    if hist.sum() < 1e-9:
        logger.warning("detect_dominant_orientations : histogramme vide.")
        return []

    # Lissage gaussien circulaire (σ ≈ 2 bins).
    # On triple l'histogramme pour que le lissage traite correctement les bords
    # 0°/180° comme une frontière cyclique, puis on ne garde que la copie centrale.
    smoothed_full = _gaussian_smooth(np.tile(hist, 3), sigma=2.0)
    smoothed = smoothed_full[n_bins : 2 * n_bins]

    # Détection des pics
    dominant_angles = _find_significant_peaks(smoothed, bin_size_deg, min_peak_ratio)

    logger.info(
        "Orientations dominantes détectées : %s.",
        [f"{np.degrees(a):.1f}°" for a in dominant_angles],
    )
    return dominant_angles


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------


def _gaussian_smooth(histogram: np.ndarray, sigma: float) -> np.ndarray:
    """Lisse un histogramme angulaire circulaire avec un filtre gaussien.

    Utilise scipy.ndimage.gaussian_filter1d avec mode='wrap' pour que les
    bords de l'histogramme [0°, 180°] soient traités de façon circulaire.

    Args:
        histogram: Tableau 1D de valeurs non négatives.
        sigma: Écart-type du filtre gaussien en bins.

    Returns:
        Histogramme lissé de même forme.
    """
    from scipy.ndimage import gaussian_filter1d

    return np.asarray(gaussian_filter1d(histogram.astype(np.float64), sigma=sigma, mode="wrap"))


def _find_significant_peaks(
    smoothed: np.ndarray,
    bin_size_deg: float,
    min_peak_ratio: float,
) -> list[float]:
    """Détecte les pics significatifs dans l'histogramme lissé.

    Double l'histogramme pour traiter correctement les pics aux bords (wrap
    angulaire). Utilise ``scipy.signal.find_peaks`` ; si aucun pic n'est
    détecté (histogramme en plateau), retourne l'indice du maximum global.

    Args:
        smoothed: Histogramme lissé (longueur N, couvrant [0°, 180°]).
        bin_size_deg: Largeur d'un bin en degrés.
        min_peak_ratio: Fraction du pic maximal en dessous de laquelle un pic
            est ignoré.

    Returns:
        Liste d'angles en radians des pics significatifs, triés par hauteur
        décroissante.
    """
    from scipy.signal import find_peaks

    max_val = float(smoothed.max())
    if max_val < 1e-9:
        return []

    n = len(smoothed)
    # Doubler pour que find_peaks voie les pics aux bords comme des maxima locaux
    doubled = np.concatenate([smoothed, smoothed])

    min_height = min_peak_ratio * max_val
    min_distance = max(1, int(5.0 / bin_size_deg))

    peak_indices, properties = find_peaks(
        doubled,
        height=min_height,
        distance=min_distance,
    )

    # Ramener tous les pics dans [0, N) en appliquant modulo N.
    # Un pic à l'indice N+k dans le tableau doublé correspond au bin k.
    folded_indices = peak_indices % n
    folded_heights = properties["peak_heights"]

    # Dédupliquer : si deux indices doublés tombent sur le même bin, garder le plus haut
    seen: dict[int, float] = {}
    for idx, h in zip(folded_indices, folded_heights):
        if idx not in seen or h > seen[idx]:
            seen[idx] = float(h)

    if not seen:
        # Fallback : retourner le maximum global
        best = int(np.argmax(smoothed))
        seen = {best: float(smoothed[best])}

    # Trier par hauteur décroissante
    sorted_bins = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)

    return [float(np.deg2rad((idx + 0.5) * bin_size_deg)) for idx, _ in sorted_bins]
