"""Profils de paramètres et auto-calibrage du pipeline Scan2Plan.

Trois profils prédéfinis :
- ``recent``  : logements neufs, cloisons fines (7-10 cm), angles droits.
- ``ancien``  : bâtiments anciens, murs épais (20-50 cm), angles variables.
- ``bureau``  : locaux professionnels, grands espaces, faux plafond.

L'auto-calibrage analyse la density map et la hauteur sous plafond pour
proposer le profil le plus adapté.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from scan2plan.runtime_paths import get_config_dir

logger = logging.getLogger(__name__)

# Profils disponibles
AVAILABLE_PROFILES = ("recent", "ancien", "bureau")

# Seuils d'auto-calibrage (en mètres)
_THIN_WALL_THRESHOLD_M = 0.12  # < 12 cm → cloison fine (récent)
_THICK_WALL_THRESHOLD_M = 0.20  # > 20 cm → mur épais (ancien)

# Seuils de hauteur sous plafond
_LOW_CEILING_M = 2.55  # < 2.55 m → logement récent standard
_HIGH_CEILING_M = 2.85  # > 2.85 m → logement ancien (hauts plafonds) ou bureau

# Seuil de densité pour les grands espaces ouverts (bureau)
_OPEN_SPACE_RATIO = 0.30  # < 30 % de pixels occupés → espace ouvert


@dataclass
class CalibrationResult:
    """Résultat de l'auto-calibrage.

    Attributes:
        suggested_profile: Nom du profil suggéré (``"recent"``, ``"ancien"``
            ou ``"bureau"``).
        ceiling_height_m: Hauteur sous plafond mesurée (mètres).
        median_wall_thickness_m: Épaisseur médiane de mur estimée (mètres).
        open_space_ratio: Fraction de pixels non occupés dans la density map.
        confidence: Confiance dans la suggestion (0–1).
        reasoning: Explication textuelle du choix.
    """

    suggested_profile: str
    ceiling_height_m: float
    median_wall_thickness_m: float
    open_space_ratio: float
    confidence: float
    reasoning: str


def load_profile(profile_name: str) -> dict[str, Any]:
    """Charge un profil de paramètres par son nom.

    Les valeurs du profil sont des surcharges partielles à fusionner
    avec ``default_params.yaml`` via ``ScanConfig``.

    Args:
        profile_name: Nom du profil (``"recent"``, ``"ancien"``, ``"bureau"``).

    Returns:
        Dictionnaire de paramètres du profil (surcharges uniquement).

    Raises:
        ValueError: Si le profil est inconnu.
        FileNotFoundError: Si le fichier du profil est introuvable.

    Example:
        ```python
        params = load_profile("recent")
        snap_tol = params["regularization"]["snap_tolerance_deg"]  # 3.0
        ```
    """
    if profile_name not in AVAILABLE_PROFILES:
        raise ValueError(
            f"Profil inconnu : '{profile_name}'. Profils disponibles : {AVAILABLE_PROFILES}."
        )

    profile_path = get_config_dir(__file__) / "profiles" / f"{profile_name}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Fichier de profil introuvable : {profile_path}")

    import yaml

    with profile_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    logger.info("Profil '%s' chargé depuis : %s", profile_name, profile_path)
    return data


def apply_profile(config: "Any", profile_name: str) -> "Any":
    """Applique un profil à une configuration existante (fusion deep).

    Modifie ``config._data`` en place en fusionnant les paramètres du profil.

    Args:
        config: Instance de ``ScanConfig`` à modifier.
        profile_name: Nom du profil à appliquer.

    Returns:
        La même instance ``config`` modifiée (pour le chaining).

    Example:
        >>> cfg = ScanConfig()
        >>> apply_profile(cfg, "recent")
        >>> cfg.regularization.snap_tolerance_deg
        3.0
    """
    from scan2plan.config import _deep_merge

    profile_data = load_profile(profile_name)
    config._data = _deep_merge(config._data, profile_data)
    logger.info("Profil '%s' appliqué à la configuration.", profile_name)
    return config


def auto_calibrate(
    density_map_image: np.ndarray,
    ceiling_height_m: float,
    resolution_m: float,
) -> CalibrationResult:
    """Suggère automatiquement un profil adapté au bâtiment analysé.

    Analyse :
    1. La hauteur sous plafond (``ceiling_height_m``) pour distinguer ancien/récent.
    2. La distribution des épaisseurs locales dans la density map pour estimer
       l'épaisseur médiane des murs.
    3. La fraction de pixels occupés (ratio espaces ouverts/fermés) pour détecter
       les grands plateaux de bureaux.

    Règles de décision (par priorité décroissante) :
    - Si ``open_space_ratio < _OPEN_SPACE_RATIO`` → ``bureau``
    - Si ``median_wall_thickness_m > _THICK_WALL_THRESHOLD_M`` → ``ancien``
    - Si ``ceiling_height_m > _HIGH_CEILING_M`` → ``ancien``
    - Sinon → ``recent``

    Args:
        density_map_image: Image de densité (float ou uint16), shape (H, W).
        ceiling_height_m: Hauteur sous plafond mesurée par RANSAC (mètres).
        resolution_m: Résolution de la density map (mètres/pixel).

    Returns:
        ``CalibrationResult`` avec le profil suggéré et les métriques.
    """
    open_ratio = _compute_open_space_ratio(density_map_image)
    median_thickness = _estimate_median_wall_thickness(density_map_image, resolution_m)

    profile, confidence, reasoning = _decide_profile(
        open_ratio, median_thickness, ceiling_height_m
    )

    result = CalibrationResult(
        suggested_profile=profile,
        ceiling_height_m=round(ceiling_height_m, 3),
        median_wall_thickness_m=round(median_thickness, 3),
        open_space_ratio=round(open_ratio, 3),
        confidence=round(confidence, 2),
        reasoning=reasoning,
    )

    logger.info(
        "Auto-calibrage → profil '%s' (confiance %.0f%%). %s",
        profile,
        confidence * 100,
        reasoning,
    )
    return result


def calibrate_slice_heights(ceiling_height_m: float) -> list[float]:
    """Calcule des hauteurs de slices adaptées à la hauteur sous plafond.

    Les trois slices sont positionnées à :
    - Haute   : 85 % de la hauteur (au-dessus des ouvertures).
    - Médiane : 45 % (hauteur de coupe standard).
    - Basse   : 8 % (bas de portes).

    Les hauteurs sont arrondies à 5 cm et bornées à des valeurs plausibles.

    Args:
        ceiling_height_m: Hauteur sous plafond mesurée (mètres).

    Returns:
        Liste de 3 hauteurs en mètres [haute, médiane, basse].

    Example:
        >>> calibrate_slice_heights(2.50)
        [2.12, 1.12, 0.20]
    """
    h_high = _clamp(round(ceiling_height_m * 0.85 / 0.05) * 0.05, 1.80, 3.50)
    h_mid = _clamp(round(ceiling_height_m * 0.45 / 0.05) * 0.05, 0.90, 1.50)
    h_low = _clamp(round(ceiling_height_m * 0.08 / 0.05) * 0.05, 0.10, 0.40)
    return [h_high, h_mid, h_low]


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------


def _compute_open_space_ratio(image: np.ndarray) -> float:
    """Calcule la fraction de pixels non occupés dans la density map.

    Un pixel est considéré « occupé » si sa valeur > 0.

    Args:
        image: Density map (float ou uint).

    Returns:
        Fraction de pixels occupés dans [0, 1].
    """
    if image.size == 0:
        return 0.0
    occupied = float((image > 0).sum())
    return occupied / float(image.size)


def _estimate_median_wall_thickness(
    image: np.ndarray,
    resolution_m: float,
) -> float:
    """Estime l'épaisseur médiane des murs par analyse des runs horizontaux.

    Pour chaque ligne de l'image, mesure la largeur (en pixels) des séquences
    continues de pixels occupés. La médiane de ces largeurs, convertie en mètres,
    est une approximation de l'épaisseur médiane des murs.

    Filtre les runs de longueur extrême (< 2 px = bruit, > 200 px = espace ouvert)
    pour ne garder que les runs correspondant à des murs.

    Args:
        image: Density map (float ou uint).
        resolution_m: Résolution en mètres/pixel.

    Returns:
        Épaisseur médiane estimée en mètres. Retourne 0.15 si aucun run valide.
    """
    binary = (image > 0).astype(np.uint8)
    run_lengths: list[int] = []

    for row in binary:
        runs = _measure_runs(row)
        run_lengths.extend(r for r in runs if 2 <= r <= 200)

    if not run_lengths:
        return 0.15  # valeur par défaut

    median_px = float(np.median(run_lengths))
    return float(median_px * resolution_m)


def _measure_runs(row: np.ndarray) -> list[int]:
    """Mesure les longueurs des séquences de 1 consécutifs dans un tableau 1D.

    Args:
        row: Tableau 1D binaire (0 ou 1).

    Returns:
        Liste des longueurs de runs de valeur 1.
    """
    if row.max() == 0:
        return []

    # Détecter les transitions
    padded = np.concatenate([[0], row, [0]])
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return [int(e - s) for s, e in zip(starts, ends)]


def _decide_profile(
    open_ratio: float,
    median_thickness_m: float,
    ceiling_height_m: float,
) -> tuple[str, float, str]:
    """Applique les règles de décision pour choisir un profil.

    Args:
        open_ratio: Fraction de pixels occupés.
        median_thickness_m: Épaisseur médiane des murs (mètres).
        ceiling_height_m: Hauteur sous plafond (mètres).

    Returns:
        ``(profil, confiance, raisonnement)``
    """
    reasons: list[str] = []
    votes: dict[str, float] = {"recent": 0.0, "ancien": 0.0, "bureau": 0.0}

    # Vote : espaces ouverts → bureau
    if open_ratio < _OPEN_SPACE_RATIO:
        votes["bureau"] += 2.0
        reasons.append(
            f"faible densité de murs ({open_ratio:.0%} pixels occupés → grands espaces)"
        )
    else:
        votes["recent"] += 0.5

    # Vote : épaisseur des murs
    if median_thickness_m > _THICK_WALL_THRESHOLD_M:
        votes["ancien"] += 2.0
        reasons.append(f"murs épais (épaisseur médiane {median_thickness_m * 100:.0f} cm)")
    elif median_thickness_m < _THIN_WALL_THRESHOLD_M:
        votes["recent"] += 1.5
        reasons.append(f"murs fins (épaisseur médiane {median_thickness_m * 100:.0f} cm)")

    # Vote : hauteur sous plafond
    if ceiling_height_m > _HIGH_CEILING_M:
        votes["ancien"] += 1.5
        reasons.append(f"hauts plafonds ({ceiling_height_m:.2f} m)")
    elif ceiling_height_m < _LOW_CEILING_M:
        votes["recent"] += 1.0
        reasons.append(f"hauteur standard ({ceiling_height_m:.2f} m)")

    # Choisir le profil avec le plus de votes
    best_profile = max(votes, key=lambda k: votes[k])
    total_votes = sum(votes.values())
    confidence = votes[best_profile] / total_votes if total_votes > 0 else 0.5
    confidence = min(1.0, confidence)

    reasoning = "Critères : " + " ; ".join(reasons) if reasons else "Aucun critère décisif."
    return best_profile, confidence, reasoning


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Borne une valeur dans [min_val, max_val].

    Args:
        value: Valeur à borner.
        min_val: Borne inférieure.
        max_val: Borne supérieure.

    Returns:
        Valeur bornée.
    """
    return max(min_val, min(max_val, value))
