"""Statistical Outlier Removal (SOR) via Open3D."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def remove_statistical_outliers(
    points: np.ndarray,
    k_neighbors: int,
    std_ratio: float,
) -> np.ndarray:
    """Supprime les points aberrants par Statistical Outlier Removal.

    Pour chaque point, calcule la distance moyenne à ses ``k_neighbors``
    plus proches voisins. Supprime les points dont la distance dépasse
    μ + ``std_ratio`` × σ sur l'ensemble du nuage.

    Args:
        points: Array (N, 3) float64.
        k_neighbors: Nombre de voisins pour le calcul de distance.
        std_ratio: Multiplicateur de l'écart-type pour le seuil de rejet.

    Returns:
        Array (M, 3) float64 — nuage nettoyé (M <= N).

    Raises:
        ValueError: Si ``k_neighbors`` < 1 ou ``std_ratio`` <= 0.
    """
    import open3d as o3d

    if k_neighbors < 1:
        raise ValueError(f"k_neighbors doit être >= 1, reçu : {k_neighbors}")
    if std_ratio <= 0:
        raise ValueError(f"std_ratio doit être > 0, reçu : {std_ratio}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cleaned, _ = pcd.remove_statistical_outlier(
        nb_neighbors=k_neighbors,
        std_ratio=std_ratio,
    )
    result = np.asarray(cleaned.points, dtype=np.float64)

    removed = len(points) - len(result)
    logger.info(
        "SOR (k=%d, σ×%.1f) : %d points supprimés (%d → %d).",
        k_neighbors,
        std_ratio,
        removed,
        len(points),
        len(result),
    )
    return result
