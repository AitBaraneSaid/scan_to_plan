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
    z_floor: float,
    height: float,
    thickness: float,
) -> np.ndarray:
    """Extrait une tranche horizontale du nuage à une hauteur relative au sol.

    Args:
        points: Array (N, 3) float64 — nuage complet.
        z_floor: Altitude du sol en mètres.
        height: Hauteur relative au sol de la tranche (mètres).
        thickness: Épaisseur de la tranche (mètres).

    Returns:
        Array (M, 3) float64 — points dans la tranche.

    Raises:
        InsufficientPointsError: Si moins de ``_MIN_POINTS_PER_SLICE`` points
            se trouvent dans la tranche.

    Example:
        >>> slice_pts = extract_slice(points, z_floor=0.1, height=1.10, thickness=0.10)
    """
    z_low = z_floor + height - thickness / 2.0
    z_high = z_floor + height + thickness / 2.0
    mask = (points[:, 2] >= z_low) & (points[:, 2] <= z_high)
    result = points[mask]

    logger.debug(
        "Slice h=%.2f m ±%.2f m → Z=[%.3f, %.3f] : %d points.",
        height,
        thickness / 2.0,
        z_low,
        z_high,
        len(result),
    )

    if len(result) < _MIN_POINTS_PER_SLICE:
        raise InsufficientPointsError(
            f"Slice à h={height:.2f} m : seulement {len(result)} points "
            f"(minimum requis : {_MIN_POINTS_PER_SLICE}). "
            "La tranche est peut-être vide ou le scan manque de densité à cette hauteur."
        )
    return result


def extract_all_slices(
    points: np.ndarray,
    z_floor: float,
    heights: list[float],
    thickness: float,
) -> dict[float, np.ndarray]:
    """Extrait toutes les tranches définies dans la configuration.

    Args:
        points: Array (N, 3) float64 — nuage complet.
        z_floor: Altitude du sol en mètres.
        heights: Liste des hauteurs relatives au sol (mètres).
        thickness: Épaisseur commune des tranches (mètres).

    Returns:
        Dictionnaire {hauteur: array (M, 3)} — une entrée par slice réussie.
        Les slices en échec (InsufficientPointsError) sont loggées en WARNING
        et absentes du dictionnaire.
    """
    slices: dict[float, np.ndarray] = {}
    for h in heights:
        try:
            slices[h] = extract_slice(points, z_floor, h, thickness)
        except InsufficientPointsError as exc:
            logger.warning("Slice ignorée : %s", exc)
    logger.info(
        "%d / %d slices extraites avec succès.",
        len(slices),
        len(heights),
    )
    return slices
