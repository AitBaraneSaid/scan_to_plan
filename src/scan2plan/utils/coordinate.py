"""Conversion coordonnées pixel ↔ métrique pour les density maps.

Convention Y :
    - Axe Y métrique : croît vers le haut.
    - Axe Y image : row 0 est en haut (Y maximal), croît vers le bas.
    - Conversion : py_image = image_height - 1 - py_geo
                   py_geo   = image_height - 1 - py_image
"""

from __future__ import annotations

import numpy as np


def pixel_to_metric(
    px: float,
    py: float,
    x_min: float,
    y_min: float,
    resolution: float,
    image_height: int,
) -> tuple[float, float]:
    """Convertit des coordonnées pixel en coordonnées métriques.

    L'axe Y image est inversé par rapport à l'axe Y métrique :
    row 0 correspond au Y métrique maximal.

    Args:
        px: Coordonnée colonne (pixel, axe X).
        py: Coordonnée ligne (pixel, axe Y image — 0 = haut).
        x_min: Coordonnée X minimale du raster (mètres).
        y_min: Coordonnée Y minimale du raster (mètres).
        resolution: Taille d'un pixel en mètres/pixel.
        image_height: Nombre de lignes de l'image (H).

    Returns:
        (x, y) en mètres dans le repère du relevé.
    """
    x = x_min + px * resolution
    # Inversion Y : py=0 → y_max, py=H-1 → y_min
    py_geo = (image_height - 1) - py
    y = y_min + py_geo * resolution
    return x, y


def metric_to_pixel(
    x: float,
    y: float,
    x_min: float,
    y_min: float,
    resolution: float,
    image_height: int,
) -> tuple[int, int]:
    """Convertit des coordonnées métriques en coordonnées pixel.

    L'axe Y image est inversé par rapport à l'axe Y métrique.

    Args:
        x: Coordonnée X en mètres.
        y: Coordonnée Y en mètres.
        x_min: Coordonnée X minimale du raster (mètres).
        y_min: Coordonnée Y minimale du raster (mètres).
        resolution: Taille d'un pixel en mètres/pixel.
        image_height: Nombre de lignes de l'image (H).

    Returns:
        (col, row) en pixels (entiers), avec row=0 en haut de l'image.
    """
    col = int(round((x - x_min) / resolution))
    py_geo = int(round((y - y_min) / resolution))
    # Inversion Y : y_geo → row image
    row = (image_height - 1) - py_geo
    return col, row


def segments_pixel_to_metric(
    segments: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    image_height: int,
) -> np.ndarray:
    """Convertit un tableau de segments pixel en coordonnées métriques.

    Args:
        segments: Array (M, 4) avec colonnes [x1_px, y1_px, x2_px, y2_px].
            Les coordonnées y sont en convention image (0 = haut).
        x_min: Coordonnée X minimale du raster (mètres).
        y_min: Coordonnée Y minimale du raster (mètres).
        resolution: Taille d'un pixel en mètres/pixel.
        image_height: Nombre de lignes de l'image (H).

    Returns:
        Array (M, 4) float64 avec colonnes [x1_m, y1_m, x2_m, y2_m].
    """
    metric = segments.astype(np.float64).copy()
    metric[:, 0] = x_min + segments[:, 0] * resolution  # x1
    metric[:, 1] = y_min + ((image_height - 1) - segments[:, 1]) * resolution  # y1
    metric[:, 2] = x_min + segments[:, 2] * resolution  # x2
    metric[:, 3] = y_min + ((image_height - 1) - segments[:, 3]) * resolution  # y2
    return metric
