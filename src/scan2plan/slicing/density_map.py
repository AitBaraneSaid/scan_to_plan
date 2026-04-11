"""Projection 2D et génération de density maps par histogram2d."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DensityMap:
    """Résultat d'une projection 2D d'une slice.

    Attributes:
        image: Array 2D (H, W) uint16 — densité de points par pixel.
        x_min: Coordonnée X minimale du raster (mètres).
        y_min: Coordonnée Y minimale du raster (mètres).
        resolution: Taille d'un pixel en mètres/pixel.
    """

    image: np.ndarray
    x_min: float
    y_min: float
    resolution: float

    @property
    def shape(self) -> tuple[int, int]:
        """Forme du raster (hauteur, largeur) en pixels."""
        return self.image.shape  # type: ignore[return-value]


def compute_density_map(
    slice_points: np.ndarray,
    resolution: float,
    margin: float = 0.5,
) -> DensityMap:
    """Projette une slice de points sur le plan XY et calcule la density map.

    Args:
        slice_points: Array (N, 3) float64 — points de la tranche.
        resolution: Taille d'un pixel en mètres/pixel.
        margin: Marge ajoutée autour de la bounding box (mètres).

    Returns:
        DensityMap avec l'image de densité et les paramètres de géoréférencement.

    Raises:
        ValueError: Si ``resolution`` <= 0 ou slice vide.

    Example:
        >>> dmap = compute_density_map(slice_pts, resolution=0.005)
        >>> dmap.image.shape
        (800, 1000)
    """
    if resolution <= 0:
        raise ValueError(f"resolution doit être > 0, reçu : {resolution}")
    if len(slice_points) == 0:
        raise ValueError("La slice est vide.")

    x = slice_points[:, 0]
    y = slice_points[:, 1]

    x_min = float(x.min()) - margin
    x_max = float(x.max()) + margin
    y_min = float(y.min()) - margin
    y_max = float(y.max()) + margin

    n_cols = max(1, int(np.ceil((x_max - x_min) / resolution)))
    n_rows = max(1, int(np.ceil((y_max - y_min) / resolution)))

    hist, _, _ = np.histogram2d(
        x,
        y,
        bins=[n_cols, n_rows],
        range=[[x_min, x_min + n_cols * resolution], [y_min, y_min + n_rows * resolution]],
    )
    # histogram2d retourne (n_cols, n_rows) ; on transpose pour (rows, cols) = (H, W)
    image = np.clip(hist.T, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    logger.debug(
        "Density map : %d × %d px (%.1f × %.1f m) — résolution %.1f mm/px — max densité : %d.",
        n_cols,
        n_rows,
        n_cols * resolution,
        n_rows * resolution,
        resolution * 1000,
        int(image.max()),
    )
    return DensityMap(image=image, x_min=x_min, y_min=y_min, resolution=resolution)
