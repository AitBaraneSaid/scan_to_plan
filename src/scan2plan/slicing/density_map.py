"""Projection 2D et génération de density maps par histogram2d."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DensityMapResult:
    """Résultat d'une projection 2D d'une slice.

    Convention image : row 0 est en haut (Y métrique maximal).
    L'axe Y image est inversé par rapport à l'axe Y métrique.

    Attributes:
        image: Array 2D (H, W) uint16 — densité de points par pixel.
        x_min: Coordonnée X minimale du raster (mètres).
        y_min: Coordonnée Y minimale du raster (mètres).
        resolution: Taille d'un pixel en mètres/pixel.
        width: Largeur du raster en pixels (axe X).
        height: Hauteur du raster en pixels (axe Y image).
    """

    image: np.ndarray
    x_min: float
    y_min: float
    resolution: float
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        """Forme du raster (hauteur, largeur) en pixels."""
        return (self.height, self.width)


def create_density_map(
    points_2d: np.ndarray,
    resolution: float = 0.005,
    margin: float = 0.5,
) -> DensityMapResult:
    """Projette des points XY sur un raster et calcule la density map.

    L'axe Y de l'image est inversé : row 0 correspond au Y métrique maximal
    (convention image standard, compatible OpenCV et matplotlib avec origin='upper').

    Args:
        points_2d: Array (M, 2) float64 — coordonnées XY des points de la tranche.
        resolution: Taille d'un pixel en mètres/pixel. Défaut : 0.005 m.
        margin: Marge ajoutée autour de la bounding box (mètres). Défaut : 0.5 m.

    Returns:
        DensityMapResult avec l'image de densité et les paramètres de géoréférencement.

    Raises:
        ValueError: Si ``resolution`` <= 0 ou ``points_2d`` vide.

    Example:
        >>> dmap = create_density_map(slice_xy, resolution=0.005)
        >>> dmap.image.shape  # (H, W)
    """
    if resolution <= 0:
        raise ValueError(f"resolution doit être > 0, reçu : {resolution}")
    if len(points_2d) == 0:
        raise ValueError("Le tableau de points est vide.")

    x = points_2d[:, 0]
    y = points_2d[:, 1]

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
    # histogram2d retourne (n_cols, n_rows) ; on transpose pour (rows=H, cols=W)
    # Puis on inverse l'axe Y pour que row 0 = Y maximal (convention image)
    image = np.clip(hist.T[::-1, :], 0, np.iinfo(np.uint16).max).astype(np.uint16)

    logger.debug(
        "Density map : %d × %d px (%.1f × %.1f m) — résolution %.1f mm/px — max densité : %d.",
        n_cols,
        n_rows,
        n_cols * resolution,
        n_rows * resolution,
        resolution * 1000,
        int(image.max()),
    )
    return DensityMapResult(
        image=image,
        x_min=x_min,
        y_min=y_min,
        resolution=resolution,
        width=n_cols,
        height=n_rows,
    )
