"""Détection des orientations dominantes du bâtiment par histogramme angulaire."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def detect_dominant_orientations(
    points: np.ndarray,
    n_orientations: int = 2,
    histogram_bins: int = 180,
) -> list[float]:
    """Détecte les orientations dominantes des murs par histogramme angulaire des normales.

    Estime les normales des points de surfaces verticales, projette sur XY,
    construit un histogramme angulaire modulo 180° et retourne les pics.

    Args:
        points: Array (N, 3) float64 — nuage de points (filtré verticalement).
        n_orientations: Nombre d'orientations dominantes à retourner.
        histogram_bins: Résolution de l'histogramme angulaire (bins sur [0°, 180°]).

    Returns:
        Liste d'angles en radians (taille <= n_orientations), triés par importance.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    normals = np.asarray(pcd.normals)
    # Garder seulement les normales quasi-horizontales (surfaces verticales)
    vertical_mask = np.abs(normals[:, 2]) < 0.3
    horiz_normals = normals[vertical_mask, :2]

    if len(horiz_normals) < 10:
        logger.warning(
            "Peu de surfaces verticales détectées (%d points), "
            "orientations dominantes non fiables.",
            len(horiz_normals),
        )
        return []

    angles = np.arctan2(horiz_normals[:, 1], horiz_normals[:, 0]) % np.pi
    hist, bin_edges = np.histogram(angles, bins=histogram_bins, range=(0, np.pi))

    peaks = _find_peaks(hist, n_peaks=n_orientations)
    dominant = [float(bin_edges[p] + (bin_edges[1] - bin_edges[0]) / 2) for p in peaks]

    logger.info(
        "Orientations dominantes détectées : %s (degrés).",
        [f"{np.degrees(a):.1f}°" for a in dominant],
    )
    return dominant


def _find_peaks(histogram: np.ndarray, n_peaks: int) -> list[int]:
    """Retourne les indices des N pics les plus importants d'un histogramme.

    Args:
        histogram: Array 1D de comptages.
        n_peaks: Nombre de pics à retourner.

    Returns:
        Liste d'indices des pics, triés par valeur décroissante.
    """
    sorted_idx = np.argsort(histogram)[::-1]
    peaks: list[int] = []
    min_separation = max(1, len(histogram) // (n_peaks * 2 + 1))

    for idx in sorted_idx:
        if all(abs(idx - p) >= min_separation for p in peaks):
            peaks.append(int(idx))
        if len(peaks) >= n_peaks:
            break
    return peaks
