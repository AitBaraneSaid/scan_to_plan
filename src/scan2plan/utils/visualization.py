"""Visualisation matplotlib pour le debug et la validation du pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_SIZE = 50_000
_FIGURE_DPI = 150
_LABEL_X = "X (m)"
_LABEL_Y = "Y (m)"
_LABEL_Z = "Z (m)"

# Entropie système explicite : la visualisation n'a pas besoin de reproductibilité,
# chaque affichage peut montrer un sous-ensemble différent.
_RNG = np.random.default_rng(seed=int.from_bytes(os.urandom(8), "little"))


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
    ax.set_xlabel(_LABEL_X)
    ax.set_ylabel(_LABEL_Y)
    ax.set_zlabel(_LABEL_Z)
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

    _, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(points[:, 0], points[:, 1], s=0.3, c="steelblue", linewidths=0)
    ax.set_xlabel(_LABEL_X)
    ax.set_ylabel(_LABEL_Y)
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

    L'image est supposée en convention standard (row 0 = Y maximal).
    ``origin`` désigne le coin bas-gauche (x_min, y_min) en mètres.

    Args:
        density_map: Array 2D (H, W) — nombre de points par pixel.
        title: Titre de la figure.
        resolution: Résolution en mètres/pixel.
        origin: Coordonnées métriques du coin bas-gauche (x_min, y_min).

    Example:
        >>> plot_density_map(dmap.image, "Density map slice 1.10 m",
        ...                  resolution=dmap.resolution, origin=(dmap.x_min, dmap.y_min))
    """
    import matplotlib.pyplot as plt

    h, w = density_map.shape
    x_min, y_min = origin
    # extent : [left, right, bottom, top] en coordonnées métriques
    extent = [x_min, x_min + w * resolution, y_min, y_min + h * resolution]
    _, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(
        density_map,
        origin="upper",
        extent=extent,
        cmap="hot",
        interpolation="nearest",
    )
    ax.set_xlabel(_LABEL_X)
    ax.set_ylabel(_LABEL_Y)
    ax.set_title(title)
    plt.colorbar(ax.images[0], ax=ax, label="Densité (pts/pixel)")
    plt.tight_layout()
    plt.show()


def plot_preprocessing_results(
    original: np.ndarray,
    downsampled: np.ndarray,
    filtered: np.ndarray,
    floor_z: float,
    ceiling_z: float,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
) -> None:
    """Affiche les résultats des étapes de prétraitement en une figure 2×2.

    Les quatre vues montrent : nuage brut, après downsampling, après filtrage
    vertical, et un profil XZ avec les niveaux sol/plafond annotés.

    Args:
        original: Array (N, 3) — nuage brut d'entrée.
        downsampled: Array (M, 3) — après voxel downsampling.
        filtered: Array (K, 3) — après filtrage vertical.
        floor_z: Altitude du sol détecté (mètres).
        ceiling_z: Altitude du plafond détecté (mètres).
        sample_size: Nombre maximal de points à afficher par vue.

    Example:
        >>> plot_preprocessing_results(raw, down, filt, floor_z=0.02, ceiling_z=2.48)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Résultats du prétraitement", fontsize=14, fontweight="bold")

    # -- Vue 1 : nuage brut (projection XY) --
    ax = axes[0, 0]
    pts = _subsample(original, sample_size)
    ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c="gray", linewidths=0)
    ax.set_title(f"Nuage brut ({len(original):,} pts)")
    ax.set_xlabel(_LABEL_X)
    ax.set_ylabel(_LABEL_Y)
    ax.set_aspect("equal")

    # -- Vue 2 : après downsampling (projection XY) --
    ax = axes[0, 1]
    pts = _subsample(downsampled, sample_size)
    ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c="steelblue", linewidths=0)
    ratio = len(original) / max(len(downsampled), 1)
    ax.set_title(f"Après downsampling ({len(downsampled):,} pts, ×{ratio:.1f})")
    ax.set_xlabel(_LABEL_X)
    ax.set_ylabel(_LABEL_Y)
    ax.set_aspect("equal")

    # -- Vue 3 : après filtrage vertical (projection XY) --
    ax = axes[1, 0]
    pts = _subsample(filtered, sample_size)
    ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c="seagreen", linewidths=0)
    ax.set_title(f"Après filtrage vertical ({len(filtered):,} pts)")
    ax.set_xlabel(_LABEL_X)
    ax.set_ylabel(_LABEL_Y)
    ax.set_aspect("equal")

    # -- Vue 4 : profil XZ avec sol/plafond annotés --
    ax = axes[1, 1]
    pts = _subsample(downsampled, sample_size)
    ax.scatter(pts[:, 0], pts[:, 2], s=0.3, c="steelblue", alpha=0.4, linewidths=0)
    ax.axhline(floor_z, color="orangered", linewidth=1.5, label=f"Sol Z={floor_z:.3f} m")
    ax.axhline(ceiling_z, color="purple", linewidth=1.5, label=f"Plafond Z={ceiling_z:.3f} m")
    ax.set_title("Profil XZ avec sol/plafond")
    ax.set_xlabel(_LABEL_X)
    ax.set_ylabel(_LABEL_Z)
    ax.legend(fontsize=8)

    plt.tight_layout()
    logger.debug(
        "plot_preprocessing_results : brut=%d, down=%d, filtré=%d, sol=%.3f m, plafond=%.3f m.",
        len(original), len(downsampled), len(filtered), floor_z, ceiling_z,
    )
    plt.show()


def plot_binary_image(image: np.ndarray, title: str) -> None:
    """Affiche une image binaire (résultat de binarisation ou nettoyage morphologique).

    Args:
        image: Array 2D (H, W) uint8 — image binaire (0=vide, 255=occupé).
        title: Titre de la figure.

    Example:
        >>> plot_binary_image(binary, "Image binaire Otsu")
    """
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap="gray", origin="upper", interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    logger.debug("Affichage image binaire : %d × %d px.", image.shape[1], image.shape[0])
    plt.tight_layout()
    plt.show()


def plot_slice_pipeline(
    density: np.ndarray,
    binary: np.ndarray,
    cleaned: np.ndarray,
    titles: tuple[str, str, str] = ("Density map", "Binaire (Otsu)", "Nettoyé"),
) -> None:
    """Affiche les trois étapes du traitement d'une slice en une figure 1×3.

    Args:
        density: Array 2D (H, W) — density map brute.
        binary: Array 2D (H, W) uint8 — après binarisation Otsu.
        cleaned: Array 2D (H, W) uint8 — après nettoyage morphologique.
        titles: Tuple de 3 titres pour les sous-figures.

    Example:
        >>> plot_slice_pipeline(dmap.image, binary_raw, binary_cleaned)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Traitement de la slice", fontsize=13, fontweight="bold")

    axes[0].imshow(density, cmap="hot", origin="upper", interpolation="nearest")
    axes[0].set_title(titles[0])
    axes[0].axis("off")

    axes[1].imshow(binary, cmap="gray", origin="upper", interpolation="nearest")
    axes[1].set_title(titles[1])
    axes[1].axis("off")

    axes[2].imshow(cleaned, cmap="gray", origin="upper", interpolation="nearest")
    axes[2].set_title(titles[2])
    axes[2].axis("off")

    plt.tight_layout()
    logger.debug("plot_slice_pipeline : %s", " | ".join(titles))
    plt.show()


def plot_detected_segments(
    density_map_result: "scan2plan.slicing.density_map.DensityMapResult",
    segments: "list[scan2plan.detection.line_detection.DetectedSegment]",
    title: str,
) -> None:
    """Overlay les segments détectés sur la density map avec une couleur par confiance.

    Les segments sont colorés selon leur score de confiance (jet : bleu=faible,
    rouge=élevé). L'épaisseur du trait est proportionnelle à la confiance.

    Args:
        density_map_result: Résultat de la density map (image + géoréférencement).
        segments: Liste de ``DetectedSegment`` en coordonnées métriques.
        title: Titre de la figure.

    Example:
        >>> plot_detected_segments(dmap, segments, "Segments détectés slice 1.10 m")
    """
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    dmap = density_map_result
    h, w = dmap.height, dmap.width
    extent = [dmap.x_min, dmap.x_min + w * dmap.resolution,
              dmap.y_min, dmap.y_min + h * dmap.resolution]

    _, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(dmap.image, origin="upper", extent=extent, cmap="gray", interpolation="nearest")

    cmap = cm.get_cmap("jet")
    for seg in segments:
        color = cmap(seg.confidence)
        lw = 1.0 + seg.confidence * 2.0
        ax.plot([seg.x1, seg.x2], [seg.y1, seg.y2], color=color, linewidth=lw)

    ax.set_xlabel(_LABEL_X)
    ax.set_ylabel(_LABEL_Y)
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Confiance")

    logger.debug("plot_detected_segments : %d segments.", len(segments))
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
    idx = _RNG.choice(len(points), size=max_count, replace=False)
    return points[idx]
