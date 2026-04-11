"""Extraction de tranches horizontales du nuage de points."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class InsufficientPointsError(RuntimeError):
    """Levée quand une slice contient trop peu de points pour être traitée."""


_MIN_POINTS_PER_SLICE = 100


def extract_slice(
    points: np.ndarray,
    height: float,
    thickness: float = 0.10,
    floor_z: float = 0.0,
) -> np.ndarray:
    """Extrait une tranche horizontale du nuage à une hauteur relative au sol.

    Args:
        points: Array (N, 3) float64 — nuage complet.
        height: Hauteur relative au sol de la tranche (mètres).
        thickness: Épaisseur de la tranche (mètres). Défaut : 0.10 m.
        floor_z: Altitude du sol en mètres. Défaut : 0.0.

    Returns:
        Array (M, 2) float64 — coordonnées XY des points dans la tranche.
        Un avertissement est loggé si M < ``_MIN_POINTS_PER_SLICE``.

    Example:
        >>> slice_xy = extract_slice(points, height=1.10, thickness=0.10, floor_z=0.05)
        >>> slice_xy.shape  # (M, 2)
    """
    z_low = floor_z + height - thickness / 2.0
    z_high = floor_z + height + thickness / 2.0
    mask = (points[:, 2] >= z_low) & (points[:, 2] <= z_high)
    result_xy = points[mask, :2]

    logger.debug(
        "Slice h=%.2f m ±%.2f m → Z=[%.3f, %.3f] : %d points.",
        height,
        thickness / 2.0,
        z_low,
        z_high,
        len(result_xy),
    )

    if len(result_xy) < _MIN_POINTS_PER_SLICE:
        logger.warning(
            "Slice à h=%.2f m : seulement %d points (minimum recommandé : %d). "
            "La tranche est peut-être vide ou le scan manque de densité à cette hauteur.",
            height,
            len(result_xy),
            _MIN_POINTS_PER_SLICE,
        )
    return result_xy


def extract_all_slices(
    points: np.ndarray,
    heights: list[float],
    thickness: float = 0.10,
    floor_z: float = 0.0,
) -> dict[float, np.ndarray]:
    """Extrait toutes les tranches définies dans la configuration.

    Args:
        points: Array (N, 3) float64 — nuage complet.
        heights: Liste des hauteurs relatives au sol (mètres).
        thickness: Épaisseur commune des tranches (mètres). Défaut : 0.10 m.
        floor_z: Altitude du sol en mètres. Défaut : 0.0.

    Returns:
        Dictionnaire {hauteur: array (M, 2)} — une entrée par slice.
        Les slices avec moins de ``_MIN_POINTS_PER_SLICE`` points sont
        incluses mais génèrent un warning.
    """
    slices: dict[float, np.ndarray] = {}
    for h in heights:
        slices[h] = extract_slice(points, h, thickness, floor_z)
    logger.info(
        "%d slices extraites (floor_z=%.3f m, épaisseur=%.2f m).",
        len(slices),
        floor_z,
        thickness,
    )
    return slices
