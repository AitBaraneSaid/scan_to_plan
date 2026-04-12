"""Détection des ouvertures (portes et fenêtres) le long des murs.

Principe : pour chaque segment de mur confirmé, on extrait le profil de
densité le long du mur dans chaque slice (HIGH, MID, LOW), puis on cherche
les zones de faible densité qui caractérisent une ouverture :

- Densité nulle en MID et LOW, présente en HIGH → PORTE
- Densité nulle en MID, présente en HIGH et LOW → FENÊTRE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from scan2plan.detection.line_detection import DetectedSegment
from scan2plan.slicing.density_map import DensityMapResult
from scan2plan.utils.coordinate import metric_to_pixel

logger = logging.getLogger(__name__)

# Épaisseur de la bande d'échantillonnage de chaque côté du mur (mètres)
_BAND_HALF_WIDTH = 0.10

# Seuil de densité normalisée en-dessous duquel un pixel est considéré "vide"
_DENSITY_THRESHOLD_RATIO = 0.15

# Nombre minimal de pixels vides consécutifs pour former une ouverture
_MIN_GAP_PIXELS = 3


@dataclass
class Opening:
    """Ouverture (porte ou fenêtre) détectée le long d'un mur.

    Attributes:
        type: ``"door"`` ou ``"window"``.
        wall_segment: Le segment de mur qui contient l'ouverture.
        position_start: Position du début de l'ouverture le long du mur
            (mètres depuis la première extrémité du segment).
        position_end: Position de la fin de l'ouverture le long du mur.
        width: Largeur de l'ouverture (mètres).
        confidence: Score de confiance 0–1, basé sur la netteté des
            transitions dans le profil de densité.
    """

    type: str
    wall_segment: DetectedSegment
    position_start: float
    position_end: float
    width: float
    confidence: float


def detect_openings_along_wall(
    wall_segment: DetectedSegment,
    density_maps: dict[str, DensityMapResult],
    binary_images: dict[str, np.ndarray],
    min_door_width: float = 0.60,
    max_door_width: float = 1.40,
    min_window_width: float = 0.30,
    max_window_width: float = 2.50,
) -> list[Opening]:
    """Détecte les ouvertures le long d'un segment de mur.

    Algorithme :
    1. Échantillonner le profil de densité le long du mur pour chaque slice,
       en intégrant les pixels dans une bande de ±10 cm perpendiculairement
       au mur.
    2. Identifier les zones de faible densité dans les slices MID et/ou LOW.
    3. Classifier chaque zone selon la présence en HIGH vs MID vs LOW.
    4. Mesurer la largeur et filtrer par dimensions plausibles.

    Args:
        wall_segment: Segment de mur de référence.
        density_maps: Density maps par label de slice (``"high"``, ``"mid"``,
            ``"low"``). Les clés absentes sont tolérées.
        binary_images: Images binaires par label. Disponibles pour extension
            future ; non utilisées directement dans l'algorithme actuel.
        min_door_width: Largeur minimale d'une porte (mètres).
        max_door_width: Largeur maximale d'une porte (mètres).
        min_window_width: Largeur minimale d'une fenêtre (mètres).
        max_window_width: Largeur maximale d'une fenêtre (mètres).

    Returns:
        Liste d'``Opening`` détectées le long du mur, triées par position
        de début.
    """
    if "high" not in density_maps:
        return []

    # Extraire les profils pour chaque slice disponible
    profiles: dict[str, np.ndarray] = {}
    for label, dmap in density_maps.items():
        profiles[label] = _extract_wall_profile(wall_segment, dmap)

    high_profile = profiles.get("high", np.array([]))
    if len(high_profile) == 0:
        return []

    mid_profile = profiles.get("mid", np.array([]))
    low_profile = profiles.get("low", np.array([]))

    # Normaliser les profils pour le seuillage
    high_norm = _normalize_profile(high_profile)
    n = len(high_norm)
    mid_norm = _resize_profile(_normalize_profile(mid_profile), n) if len(mid_profile) else np.zeros(n)
    low_norm = _resize_profile(_normalize_profile(low_profile), n) if len(low_profile) else np.zeros(n)

    # Masques de présence/absence à chaque position
    high_present = high_norm > _DENSITY_THRESHOLD_RATIO
    mid_absent = mid_norm <= _DENSITY_THRESHOLD_RATIO
    low_absent = low_norm <= _DENSITY_THRESHOLD_RATIO

    resolution = _get_profile_resolution(wall_segment, density_maps)

    openings: list[Opening] = []

    # Porte : HIGH présent, MID absent, LOW absent
    door_gaps = _find_gaps(high_present & mid_absent & low_absent)
    for start_px, end_px in door_gaps:
        opening = _make_opening(
            "door", wall_segment, start_px, end_px, n, resolution,
            high_norm, min_door_width, max_door_width,
        )
        if opening is not None:
            openings.append(opening)

    # Fenêtre : HIGH présent, MID absent, LOW présent
    window_gaps = _find_gaps(high_present & mid_absent & ~low_absent)
    for start_px, end_px in window_gaps:
        opening = _make_opening(
            "window", wall_segment, start_px, end_px, n, resolution,
            high_norm, min_window_width, max_window_width,
        )
        if opening is not None:
            openings.append(opening)

    openings.sort(key=lambda o: o.position_start)

    logger.debug(
        "Mur (%.2f, %.2f)→(%.2f, %.2f) : %d ouvertures détectées.",
        wall_segment.x1, wall_segment.y1, wall_segment.x2, wall_segment.y2,
        len(openings),
    )
    return openings


def detect_all_openings(
    wall_segments: list[DetectedSegment],
    density_maps: dict[str, DensityMapResult],
    binary_images: dict[str, np.ndarray],
    config: dict,
) -> list[Opening]:
    """Détecte les ouvertures sur tous les segments de murs.

    Args:
        wall_segments: Liste des segments de murs confirmés.
        density_maps: Density maps par label de slice.
        binary_images: Images binaires par label de slice.
        config: Dictionnaire de configuration avec les clés optionnelles :
            ``min_door_width``, ``max_door_width``,
            ``min_window_width``, ``max_window_width``.

    Returns:
        Liste de toutes les ``Opening`` détectées sur l'ensemble des murs.
    """
    min_door = float(config.get("min_door_width", 0.60))
    max_door = float(config.get("max_door_width", 1.40))
    min_win = float(config.get("min_window_width", 0.30))
    max_win = float(config.get("max_window_width", 2.50))

    all_openings: list[Opening] = []
    for seg in wall_segments:
        found = detect_openings_along_wall(
            seg, density_maps, binary_images,
            min_door_width=min_door,
            max_door_width=max_door,
            min_window_width=min_win,
            max_window_width=max_win,
        )
        all_openings.extend(found)

    logger.info(
        "detect_all_openings : %d murs analysés → %d ouvertures "
        "(%d portes, %d fenêtres).",
        len(wall_segments),
        len(all_openings),
        sum(1 for o in all_openings if o.type == "door"),
        sum(1 for o in all_openings if o.type == "window"),
    )
    return all_openings


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------

def _extract_wall_profile(
    wall_segment: DetectedSegment,
    dmap: DensityMapResult,
) -> np.ndarray:
    """Extrait le profil de densité intégrée le long d'un segment de mur.

    Échantillonne la density map en n points régulièrement espacés le long
    du segment. Pour chaque point, intègre les pixels dans une bande de
    ±``_BAND_HALF_WIDTH`` perpendiculairement au mur.

    Args:
        wall_segment: Segment de mur.
        dmap: Density map géoréférencée.

    Returns:
        Array 1D float64 de longueur n ≈ longueur_mur / résolution.
        Chaque valeur est la somme des pixels de densité dans la bande.
    """
    length = wall_segment.length
    if length < dmap.resolution:
        return np.array([])

    n_samples = max(1, int(round(length / dmap.resolution)))

    dx = wall_segment.x2 - wall_segment.x1
    dy = wall_segment.y2 - wall_segment.y1
    ux, uy = dx / length, dy / length
    # Direction perpendiculaire (rotation 90°)
    px, py = -uy, ux

    # Nombre de pixels dans la demi-bande perpendiculaire
    band_px = max(1, int(round(_BAND_HALF_WIDTH / dmap.resolution)))

    profile = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        t = i / max(1, n_samples - 1)
        mx = wall_segment.x1 + t * dx
        my = wall_segment.y1 + t * dy

        total = 0.0
        for k in range(-band_px, band_px + 1):
            bx = mx + k * dmap.resolution * px
            by = my + k * dmap.resolution * py
            col, row = metric_to_pixel(
                bx, by,
                dmap.x_min, dmap.y_min,
                dmap.resolution, dmap.height,
            )
            if 0 <= row < dmap.height and 0 <= col < dmap.width:
                total += float(dmap.image[row, col])
        profile[i] = total

    return profile


def _normalize_profile(profile: np.ndarray) -> np.ndarray:
    """Normalise un profil dans [0, 1] par rapport à son maximum.

    Args:
        profile: Profil de densité brut.

    Returns:
        Profil normalisé float64. Retourne un tableau de zéros si le profil
        est entièrement nul.
    """
    if len(profile) == 0:
        return np.array([], dtype=np.float64)
    max_val = float(profile.max())
    if max_val < 1e-9:
        return np.zeros_like(profile, dtype=np.float64)
    return profile.astype(np.float64) / max_val


def _resize_profile(profile: np.ndarray, target_len: int) -> np.ndarray:
    """Redimensionne un profil 1D par interpolation linéaire.

    Args:
        profile: Profil source.
        target_len: Longueur cible.

    Returns:
        Profil interpolé de longueur ``target_len``.
    """
    if len(profile) == 0:
        return np.zeros(target_len, dtype=np.float64)
    if len(profile) == target_len:
        return profile.astype(np.float64)
    src_x = np.linspace(0.0, 1.0, len(profile))
    dst_x = np.linspace(0.0, 1.0, target_len)
    return np.interp(dst_x, src_x, profile.astype(np.float64))


def _get_profile_resolution(
    wall_segment: DetectedSegment,
    density_maps: dict[str, DensityMapResult],
) -> float:
    """Retourne la résolution métrique du profil (mètres/échantillon).

    Args:
        wall_segment: Segment de mur.
        density_maps: Density maps disponibles.

    Returns:
        Mètres par échantillon du profil.
    """
    dmap = next(iter(density_maps.values()), None)
    if dmap is None:
        return 0.005
    length = wall_segment.length
    n_samples = max(1, int(round(length / dmap.resolution)))
    return length / n_samples


def _find_gaps(mask: np.ndarray) -> list[tuple[int, int]]:
    """Identifie les plages de True consécutifs dans un masque booléen.

    Args:
        mask: Tableau booléen 1D.

    Returns:
        Liste de ``(start_idx, end_idx)`` (inclus/inclus) pour chaque plage
        de True d'au moins ``_MIN_GAP_PIXELS`` pixels consécutifs.
    """
    gaps: list[tuple[int, int]] = []
    in_gap = False
    start = 0

    for i, val in enumerate(mask):
        if val and not in_gap:
            in_gap = True
            start = i
        elif not val and in_gap:
            in_gap = False
            if i - start >= _MIN_GAP_PIXELS:
                gaps.append((start, i - 1))

    if in_gap and len(mask) - start >= _MIN_GAP_PIXELS:
        gaps.append((start, len(mask) - 1))

    return gaps


def _make_opening(
    opening_type: str,
    wall_segment: DetectedSegment,
    start_px: int,
    end_px: int,
    profile_len: int,
    resolution: float,
    high_norm: np.ndarray,
    min_width: float,
    max_width: float,
) -> Opening | None:
    """Construit un objet Opening depuis des indices de profil.

    Convertit les indices pixel en positions métriques, vérifie les
    dimensions, et calcule le score de confiance basé sur la netteté
    des transitions de densité aux bords du gap.

    Args:
        opening_type: ``"door"`` ou ``"window"``.
        wall_segment: Segment de mur de référence.
        start_px: Indice de début du gap dans le profil (inclus).
        end_px: Indice de fin du gap dans le profil (inclus).
        profile_len: Longueur totale du profil en pixels.
        resolution: Mètres par échantillon du profil.
        high_norm: Profil normalisé de la slice haute.
        min_width: Largeur minimale acceptable (mètres).
        max_width: Largeur maximale acceptable (mètres).

    Returns:
        ``Opening`` si les dimensions sont valides, ``None`` sinon.
    """
    pos_start = start_px * resolution
    pos_end = (end_px + 1) * resolution
    width = pos_end - pos_start

    if width < min_width or width > max_width:
        return None

    # Confiance : amplitude des transitions de densité aux bords du gap
    left_idx = max(0, start_px - 1)
    right_idx = min(profile_len - 1, end_px + 1)

    left_val = float(high_norm[left_idx]) if left_idx < len(high_norm) else 0.0
    gap_left_val = float(high_norm[start_px]) if start_px < len(high_norm) else 0.0
    right_val = float(high_norm[right_idx]) if right_idx < len(high_norm) else 0.0
    gap_right_val = float(high_norm[end_px]) if end_px < len(high_norm) else 0.0

    left_drop = left_val - gap_left_val
    right_drop = right_val - gap_right_val
    confidence = float(np.clip((left_drop + right_drop) / 2.0, 0.0, 1.0))

    return Opening(
        type=opening_type,
        wall_segment=wall_segment,
        position_start=pos_start,
        position_end=pos_end,
        width=width,
        confidence=confidence,
    )
