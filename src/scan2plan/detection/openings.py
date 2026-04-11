"""Détection des ouvertures (portes et fenêtres) par comparaison multi-slice."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class OpeningType(str, Enum):
    """Type d'ouverture détectée."""

    DOOR = "door"
    WINDOW = "window"
    UNKNOWN = "unknown"


@dataclass
class Opening:
    """Représente une ouverture détectée dans un mur.

    Attributes:
        opening_type: Type de l'ouverture (porte, fenêtre, inconnu).
        position: Centre de l'ouverture en coordonnées métriques (x, y).
        width: Largeur de l'ouverture en mètres.
        wall_segment: Segment de mur porteur [x1, y1, x2, y2] en mètres.
    """

    opening_type: OpeningType
    position: tuple[float, float]
    width: float
    wall_segment: np.ndarray


def detect_openings(
    slices: dict[float, np.ndarray],
    confirmed_walls: np.ndarray,
    heights: list[float],
    resolution: float,
) -> list[Opening]:
    """Détecte les ouvertures par comparaison de la densité multi-slice.

    Logique (V1) :
    - Mur présent slice haute + absente slice médiane/basse → porte.
    - Mur présent slice haute + basse, absent slice médiane → fenêtre.

    Note: Implémentation complète en V1. Ce stub retourne une liste vide.

    Args:
        slices: Dictionnaire {hauteur: array (M, 3)} des slices extraites.
        confirmed_walls: Array (K, 4) — segments de murs confirmés.
        heights: Liste des hauteurs de slice dans l'ordre [haute, médiane, basse].
        resolution: Résolution de la density map (m/pixel).

    Returns:
        Liste d'ouvertures détectées.
    """
    logger.info(
        "Détection des ouvertures (V1 — stub) : %d murs analysés.", len(confirmed_walls)
    )
    # TODO(V1): implémenter la détection par comparaison multi-slice
    return []
