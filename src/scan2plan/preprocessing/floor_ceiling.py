"""Détection du sol et du plafond par RANSAC avec contrainte d'horizontalité."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class NoFloorDetectedError(RuntimeError):
    """Levée quand aucun plan horizontal ne peut être détecté comme sol."""


class NoCeilingDetectedError(RuntimeError):
    """Levée quand aucun plan horizontal ne peut être détecté comme plafond."""


def detect_floor_and_ceiling(
    points: np.ndarray,
    ransac_distance: float,
    ransac_iterations: int,
    normal_tolerance_deg: float,
) -> tuple[float, float]:
    """Détecte l'altitude du sol et du plafond par RANSAC.

    Cherche les deux plans horizontaux les plus représentés dans le nuage :
    le plan avec la plus faible altitude Z = sol, le plus élevé = plafond.

    Args:
        points: Array (N, 3) float64 — nuage de points complet.
        ransac_distance: Distance seuil pour les inliers RANSAC (mètres).
        ransac_iterations: Nombre d'itérations RANSAC.
        normal_tolerance_deg: Tolérance angulaire pour qualifier un plan
            d'horizontal (degrés entre la normale et l'axe Z).

    Returns:
        (z_floor, z_ceiling) — altitudes en mètres.

    Raises:
        NoFloorDetectedError: Si aucun plan horizontal n'est trouvé.
        NoCeilingDetectedError: Si un seul plan horizontal est trouvé.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    horizontal_planes: list[float] = []
    remaining = pcd

    for iteration in range(10):  # 10 tentatives max pour trouver 2 plans horizontaux
        if len(remaining.points) < 3:
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=ransac_distance,
            ransac_n=3,
            num_iterations=ransac_iterations,
        )
        a, b, c, _ = plane_model
        normal = np.array([a, b, c])
        normal_unit = normal / np.linalg.norm(normal)

        # Angle entre la normale du plan et l'axe Z
        angle_deg = np.degrees(np.arccos(np.clip(abs(normal_unit[2]), 0, 1)))

        if angle_deg <= normal_tolerance_deg:
            z_inliers = np.asarray(remaining.points)[inliers, 2]
            z_median = float(np.median(z_inliers))
            horizontal_planes.append(z_median)
            logger.debug(
                "Plan horizontal trouvé (itération %d) : Z=%.3f m, %d inliers.",
                iteration,
                z_median,
                len(inliers),
            )

        # Retirer les inliers pour chercher le prochain plan
        remaining = remaining.select_by_index(inliers, invert=True)

        if len(horizontal_planes) >= 2:
            break

    if not horizontal_planes:
        raise NoFloorDetectedError(
            "Aucun plan horizontal détecté dans le nuage. "
            "Vérifiez la qualité du scan ou élargissez normal_tolerance_deg."
        )

    horizontal_planes.sort()
    z_floor = horizontal_planes[0]

    if len(horizontal_planes) < 2:
        raise NoCeilingDetectedError(
            f"Un seul plan horizontal détecté (sol à Z={z_floor:.3f} m). "
            "Impossible de détecter le plafond."
        )

    z_ceiling = horizontal_planes[-1]
    height = z_ceiling - z_floor
    logger.info(
        "Sol détecté : Z=%.3f m | Plafond détecté : Z=%.3f m | Hauteur sous plafond : %.2f m.",
        z_floor,
        z_ceiling,
        height,
    )
    return z_floor, z_ceiling


def filter_vertical_range(
    points: np.ndarray,
    z_floor: float,
    z_ceiling: float,
) -> np.ndarray:
    """Conserve uniquement les points dans l'espace intérieur utile.

    Args:
        points: Array (N, 3) float64.
        z_floor: Altitude du sol en mètres.
        z_ceiling: Altitude du plafond en mètres.

    Returns:
        Array (M, 3) float64 — points avec z_floor <= Z <= z_ceiling.
    """
    mask = (points[:, 2] >= z_floor) & (points[:, 2] <= z_ceiling)
    filtered = points[mask]
    logger.info(
        "Filtrage vertical [%.3f m, %.3f m] : %d → %d points.",
        z_floor,
        z_ceiling,
        len(points),
        len(filtered),
    )
    return filtered
