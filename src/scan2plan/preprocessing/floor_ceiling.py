"""Détection du sol et du plafond par RANSAC avec contrainte d'horizontalité."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_MAX_RANSAC_ATTEMPTS = 5


class NoFloorDetectedError(Exception):
    """Levée quand aucun plan horizontal ne peut être détecté comme sol."""


class NoCeilingDetectedError(Exception):
    """Levée quand aucun plan horizontal ne peut être détecté comme plafond."""


def detect_floor(
    points: np.ndarray,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    normal_tolerance_deg: float = 10.0,
) -> tuple[float, np.ndarray]:
    """Détecte le plan du sol par RANSAC avec contrainte d'horizontalité.

    Cherche le plan horizontal (normale ≈ [0, 0, 1]) avec le plus d'inliers.
    Si le premier plan RANSAC n'est pas horizontal, ses inliers sont retirés
    et la recherche recommence (jusqu'à ``_MAX_RANSAC_ATTEMPTS`` tentatives).

    Args:
        points: Array (N, 3) float64 — nuage de points complet.
        distance_threshold: Distance seuil pour les inliers RANSAC (mètres).
        ransac_n: Nombre de points minimal pour ajuster le modèle.
        num_iterations: Nombre d'itérations RANSAC.
        normal_tolerance_deg: Tolérance angulaire pour qualifier un plan
            d'horizontal (degrés entre la normale et [0, 0, 1]).

    Returns:
        Tuple (z_floor, inlier_mask) :

        - ``z_floor`` : altitude Z moyenne du sol en mètres.
        - ``inlier_mask`` : masque booléen (N,) — True pour les points du sol.

    Raises:
        NoFloorDetectedError: Si aucun plan horizontal n'est trouvé après
            ``_MAX_RANSAC_ATTEMPTS`` tentatives.

    Example:
        >>> z_floor, mask = detect_floor(points)
        >>> abs(z_floor) < 0.05
        True
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    remaining_indices = np.arange(len(points))

    for attempt in range(_MAX_RANSAC_ATTEMPTS):
        if len(remaining_indices) < ransac_n:
            break

        sub_pcd = pcd.select_by_index(remaining_indices.tolist())
        plane_model, local_inliers = sub_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        a, b, c, _ = plane_model
        normal = np.array([a, b, c], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        angle_deg = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))

        global_inliers = remaining_indices[local_inliers]

        if angle_deg <= normal_tolerance_deg:
            z_floor = float(points[global_inliers, 2].mean())
            inlier_mask = np.zeros(len(points), dtype=bool)
            inlier_mask[global_inliers] = True
            logger.info(
                "Sol détecté (tentative %d) : Z=%.3f m, angle=%.1f°, %d inliers.",
                attempt + 1,
                z_floor,
                angle_deg,
                len(global_inliers),
            )
            return z_floor, inlier_mask

        logger.debug(
            "Tentative %d : plan rejeté (angle=%.1f° > %.1f°), %d inliers retirés.",
            attempt + 1,
            angle_deg,
            normal_tolerance_deg,
            len(local_inliers),
        )
        remaining_indices = np.setdiff1d(remaining_indices, global_inliers)

    raise NoFloorDetectedError(
        f"Aucun plan horizontal détecté après {_MAX_RANSAC_ATTEMPTS} tentatives. "
        "Vérifiez la qualité du scan ou augmentez normal_tolerance_deg."
    )


def detect_ceiling(
    points: np.ndarray,
    floor_z: float,
    min_height: float = 2.0,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    normal_tolerance_deg: float = 10.0,
) -> tuple[float, np.ndarray]:
    """Détecte le plan du plafond par RANSAC au-dessus d'une hauteur minimale.

    Filtre les points avec Z > floor_z + min_height avant de lancer RANSAC,
    puis applique la même logique d'horizontalité que ``detect_floor``.

    Args:
        points: Array (N, 3) float64 — nuage de points complet.
        floor_z: Altitude du sol détecté (mètres).
        min_height: Hauteur minimale du plafond au-dessus du sol (mètres).
        distance_threshold: Distance seuil pour les inliers RANSAC (mètres).
        ransac_n: Nombre de points minimal pour ajuster le modèle.
        num_iterations: Nombre d'itérations RANSAC.
        normal_tolerance_deg: Tolérance angulaire pour qualifier un plan
            d'horizontal (degrés entre la normale et [0, 0, 1]).

    Returns:
        Tuple (z_ceiling, inlier_mask) :

        - ``z_ceiling`` : altitude Z moyenne du plafond en mètres.
        - ``inlier_mask`` : masque booléen (N,) sur le nuage **original**.

    Raises:
        NoCeilingDetectedError: Si aucun plan horizontal n'est trouvé dans la
            zone haute après ``_MAX_RANSAC_ATTEMPTS`` tentatives.

    Example:
        >>> z_ceil, mask = detect_ceiling(points, floor_z=0.05)
        >>> abs(z_ceil - 2.5) < 0.05
        True
    """
    import open3d as o3d

    z_min_ceiling = floor_z + min_height
    high_mask = points[:, 2] > z_min_ceiling
    high_indices = np.nonzero(high_mask)[0]

    if len(high_indices) < ransac_n:
        raise NoCeilingDetectedError(
            f"Pas assez de points au-dessus de {z_min_ceiling:.2f} m "
            f"({len(high_indices)} points, minimum requis : {ransac_n})."
        )

    high_points = points[high_indices]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(high_points.astype(np.float64))
    remaining_indices = np.arange(len(high_points))

    for attempt in range(_MAX_RANSAC_ATTEMPTS):
        if len(remaining_indices) < ransac_n:
            break

        sub_pcd = pcd.select_by_index(remaining_indices.tolist())
        plane_model, local_inliers = sub_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        a, b, c, _ = plane_model
        normal = np.array([a, b, c], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        angle_deg = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))

        local_global = remaining_indices[local_inliers]
        original_inliers = high_indices[local_global]

        if angle_deg <= normal_tolerance_deg:
            z_ceiling = float(points[original_inliers, 2].mean())
            inlier_mask = np.zeros(len(points), dtype=bool)
            inlier_mask[original_inliers] = True
            logger.info(
                "Plafond détecté (tentative %d) : Z=%.3f m, angle=%.1f°, %d inliers.",
                attempt + 1,
                z_ceiling,
                angle_deg,
                len(original_inliers),
            )
            return z_ceiling, inlier_mask

        logger.debug(
            "Tentative %d plafond : plan rejeté (angle=%.1f° > %.1f°).",
            attempt + 1,
            angle_deg,
            normal_tolerance_deg,
        )
        remaining_indices = np.setdiff1d(remaining_indices, local_global)

    raise NoCeilingDetectedError(
        f"Aucun plan horizontal détecté comme plafond après {_MAX_RANSAC_ATTEMPTS} tentatives."
    )


def filter_vertical_range(
    points: np.ndarray,
    z_min: float,
    z_max: float,
    margin: float = 0.05,
) -> np.ndarray:
    """Conserve uniquement les points dans la tranche verticale [z_min - margin, z_max + margin].

    Args:
        points: Array (N, 3) float64.
        z_min: Borne basse (altitude du sol, mètres).
        z_max: Borne haute (altitude du plafond, mètres).
        margin: Marge ajoutée de chaque côté pour ne pas tronquer les plans
            sol/plafond eux-mêmes (mètres). Défaut : 0.05 m.

    Returns:
        Array (M, 3) float64 — points dans la tranche.

    Example:
        >>> filtered = filter_vertical_range(points, z_min=0.0, z_max=2.5)
    """
    z_low = z_min - margin
    z_high = z_max + margin
    mask = (points[:, 2] >= z_low) & (points[:, 2] <= z_high)
    filtered = points[mask]
    logger.info(
        "Filtrage vertical [%.3f m, %.3f m] (marge=±%.3f m) : %d → %d points.",
        z_min,
        z_max,
        margin,
        len(points),
        len(filtered),
    )
    return filtered
