"""Voxel grid downsampling du nuage de points via Open3D."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Sous-échantillonne le nuage de points par voxel grid.

    Conserve exactement un point par voxel cubique de côté ``voxel_size``.
    Élimine le biais de densité lié à la proximité au scanner.

    Args:
        points: Array (N, 3) float64 — nuage de points d'entrée.
        voxel_size: Côté du voxel en mètres. Doit être > 0.

    Returns:
        Array (M, 3) float64 — nuage sous-échantillonné (M <= N).

    Raises:
        ValueError: Si ``voxel_size`` <= 0 ou si ``points`` est vide.
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

    logger.info(
        "Voxel downsampling (voxel=%.4f m) : %d → %d points (réduction %.1f×).",
        voxel_size,
        len(points),
        len(result),
        len(points) / max(len(result), 1),
    )
    return result
