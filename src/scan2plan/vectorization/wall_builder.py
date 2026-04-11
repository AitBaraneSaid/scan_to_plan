"""Construction des entités mur (simple ligne — V1, double ligne en V2+)."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def build_wall_entities(segments: np.ndarray) -> np.ndarray:
    """Retourne les segments de murs prêts pour l'export DXF.

    En V1 : les murs sont représentés en ligne simple (axe du mur).
    En V2 : on ajoutera la double ligne avec épaisseur réelle.

    Args:
        segments: Array (M, 4) float64 — segments [x1, y1, x2, y2] en mètres.

    Returns:
        Array (M, 4) float64 — entités mur prêtes pour l'export.
    """
    logger.debug("Construction de %d entités mur (ligne simple).", len(segments))
    return segments.astype(np.float64)
