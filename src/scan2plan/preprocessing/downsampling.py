"""Voxel grid downsampling du nuage de points via Open3D."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.005) -> np.ndarray:
    """Sous-échantillonne le nuage par grille de voxels.

    Conserve exactement un point par voxel cubique de côté ``voxel_size``.
    Élimine le biais de densité lié à la proximité au scanner et réduit
    le volume de données d'un facteur 10 à 100×.

    Args:
        points: Tableau (N, 3) des coordonnées XYZ en mètres.
        voxel_size: Taille du voxel en mètres. Défaut : 0.005 m (5 mm).

    Returns:
        Tableau (M, 3) sous-échantillonné avec M <= N, dtype float64.

    Raises:
        ValueError: Si ``voxel_size`` <= 0 ou si ``points`` est vide.

    Example:
        >>> result = voxel_downsample(points, voxel_size=0.005)
        >>> result.shape[1]
        3
    """
    import open3d as o3d

    if voxel_size <= 0:
        raise ValueError(f"voxel_size doit être > 0, reçu : {voxel_size}")
    if len(points) == 0:
        raise ValueError("Le nuage de points est vide.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    result = np.asarray(downsampled.points, dtype=np.float64)

    reduction = len(points) / max(len(result), 1)
    logger.info(
        "Voxel downsampling (voxel=%.4f m) : %d → %d points (réduction %.1f×).",
        voxel_size,
        len(points),
        len(result),
        reduction,
    )
    return result
