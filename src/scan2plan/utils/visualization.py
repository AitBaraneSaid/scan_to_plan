"""Visualisation matplotlib pour le debug et la validation du pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_SIZE = 50_000
_FIGURE_DPI = 150


def plot_point_cloud_3d(
    points: np.ndarray,
    title: str,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
) -> None:
    """Affiche un nuage de points en 3D avec matplotlib.

    Le nuage est sous-échantillonné aléatoirement à ``sample_size`` points
    pour garantir des performances d'affichage acceptables.

    Args:
        points: Array (N, 3) float64 — coordonnées XYZ.
        title: Titre de la figure.
        sample_size: Nombre maximal de points à afficher (défaut : 50 000).

    Example:
        >>> plot_point_cloud_3d(points, "Nuage brut", sample_size=20000)
    """
    import matplotlib.pyplot as plt

    displayed = _subsample(points, sample_size)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        displayed[:, 0],
        displayed[:, 1],
        displayed[:, 2],
        s=0.5,
        c=displayed[:, 2],
        cmap="viridis",
        linewidths=0,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    logger.debug("Affichage 3D : %d / %d points.", len(displayed), len(points))
    plt.tight_layout()
    plt.show()


def plot_point_cloud_2d(points: np.ndarray, title: str) -> None:
    """Affiche la projection XY d'un nuage de points.

    Tous les points sont projetés sur le plan horizontal. Utile pour vérifier
    le résultat d'une slice ou d'une density map brute.

    Args:
        points: Array (N, 3) float64 — coordonnées XYZ.
        title: Titre de la figure.

    Example:
        >>> plot_point_cloud_2d(slice_points, "Slice 1.10 m")
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(points[:, 0], points[:, 1], s=0.3, c="steelblue", linewidths=0)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    logger.debug("Affichage 2D : %d points.", len(points))
    plt.tight_layout()
    plt.show()


def plot_density_map(
    density_map: np.ndarray,
    title: str,
    resolution: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Affiche une density map avec les axes en mètres.

    Args:
        density_map: Array 2D (H, W) — nombre de points par pixel.
        title: Titre de la figure.
        resolution: Résolution en mètres/pixel.
        origin: Coordonnées métriques du coin bas-gauche (x_min, y_min).

    Example:
        >>> plot_density_map(dmap, "Density map slice 1.10 m", resolution=0.005)
    """
    import matplotlib.pyplot as plt

    h, w = density_map.shape
    extent = [
        origin[0],
        origin[0] + w * resolution,
        origin[1],
        origin[1] + h * resolution,
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(
        density_map,
        origin="lower",
        extent=extent,
        cmap="hot",
        interpolation="nearest",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    plt.colorbar(ax.images[0], ax=ax, label="Densité (pts/pixel)")
    plt.tight_layout()
    plt.show()


def save_figure(fig: "matplotlib.figure.Figure", path: Path) -> None:
    """Sauvegarde une figure matplotlib en PNG.

    Args:
        fig: Figure matplotlib à sauvegarder.
        path: Chemin de sortie (l'extension .png est ajoutée si absente).

    Example:
        >>> fig, ax = plt.subplots()
        >>> save_figure(fig, Path("output/density_map.png"))
    """
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=_FIGURE_DPI, bbox_inches="tight")
    logger.info("Figure sauvegardée : %s", path)


# ------------------------------------------------------------------
# Utilitaire interne
# ------------------------------------------------------------------

def _subsample(points: np.ndarray, max_count: int) -> np.ndarray:
    """Sous-échantillonne aléatoirement un nuage si nécessaire.

    Args:
        points: Array (N, 3).
        max_count: Nombre maximal de points à conserver.

    Returns:
        Sous-ensemble de points (max_count, 3) ou points original si N <= max_count.
    """
    if len(points) <= max_count:
        return points
    idx = np.random.choice(len(points), size=max_count, replace=False)
    return points[idx]
