"""Conversion coordonnées pixel ↔ métrique pour les density maps."""

from __future__ import annotations

import numpy as np


def pixel_to_metric(
    px: float,
    py: float,
    x_min: float,
    y_min: float,
    resolution: float,
) -> tuple[float, float]:
    """Convertit des coordonnées pixel en coordonnées métriques.

    Args:
        px: Coordonnée colonne (pixel).
        py: Coordonnée ligne (pixel).
        x_min: Coordonnée X minimale du raster (mètres).
        y_min: Coordonnée Y minimale du raster (mètres).
        resolution: Taille d'un pixel en mètres/pixel.

    Returns:
        (x, y) en mètres dans le repère du relevé.
    """
    return x_min + px * resolution, y_min + py * resolution


def metric_to_pixel(
    x: float,
    y: float,
    x_min: float,
    y_min: float,
    resolution: float,
) -> tuple[int, int]:
    """Convertit des coordonnées métriques en coordonnées pixel.

    Args:
        x: Coordonnée X en mètres.
        y: Coordonnée Y en mètres.
        x_min: Coordonnée X minimale du raster (mètres).
        y_min: Coordonnée Y minimale du raster (mètres).
        resolution: Taille d'un pixel en mètres/pixel.

    Returns:
        (col, row) en pixels (entiers).
    """
    col = int(round((x - x_min) / resolution))
    row = int(round((y - y_min) / resolution))
    return col, row


def segments_pixel_to_metric(
    segments: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
) -> np.ndarray:
    """Convertit un tableau de segments pixel en coordonnées métriques.

    Args:
        segments: Array (M, 4) avec colonnes [x1_px, y1_px, x2_px, y2_px].
        x_min: Coordonnée X minimale du raster (mètres).
        y_min: Coordonnée Y minimale du raster (mètres).
        resolution: Taille d'un pixel en mètres/pixel.

    Returns:
        Array (M, 4) float64 avec colonnes [x1_m, y1_m, x2_m, y2_m].
    """
    metric = segments.astype(np.float64).copy()
    metric[:, 0] = x_min + segments[:, 0] * resolution  # x1
    metric[:, 1] = y_min + segments[:, 1] * resolution  # y1
    metric[:, 2] = x_min + segments[:, 2] * resolution  # x2
    metric[:, 3] = y_min + segments[:, 3] * resolution  # y2
    return metric
