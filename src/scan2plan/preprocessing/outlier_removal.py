"""Statistical Outlier Removal (SOR) via Open3D."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def remove_statistical_outliers(
    points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """Supprime les points aberrants par Statistical Outlier Removal.

    Pour chaque point, calcule la distance moyenne à ses ``nb_neighbors``
    plus proches voisins. Supprime les points dont la distance dépasse
    μ + ``std_ratio`` × σ sur l'ensemble du nuage.

    Args:
        points: Array (N, 3) float64.
        nb_neighbors: Nombre de voisins pour le calcul de distance. Défaut : 20.
        std_ratio: Multiplicateur de l'écart-type pour le seuil de rejet. Défaut : 2.0.

    Returns:
        Array (M, 3) float64 — nuage nettoyé (M <= N).

    Raises:
        ValueError: Si ``nb_neighbors`` < 1 ou ``std_ratio`` <= 0.

    Example:
        >>> cleaned = remove_statistical_outliers(points, nb_neighbors=20, std_ratio=2.0)
    """
    import open3d as o3d

    if nb_neighbors < 1:
        raise ValueError(f"nb_neighbors doit être >= 1, reçu : {nb_neighbors}")
    if std_ratio <= 0:
        raise ValueError(f"std_ratio doit être > 0, reçu : {std_ratio}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cleaned, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    result = np.asarray(cleaned.points, dtype=np.float64)

    removed = len(points) - len(result)
    logger.info(
        "SOR (k=%d, σ×%.1f) : %d points supprimés (%d → %d).",
        nb_neighbors,
        std_ratio,
        removed,
        len(points),
        len(result),
    )
    return result
