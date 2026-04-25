"""Binarisation et nettoyage morphologique des density maps."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def binarize_density_map(
    density_map: np.ndarray,
    method: str = "otsu",
) -> np.ndarray:
    """Binarise une density map par seuillage.

    Seule la méthode ``"otsu"`` est supportée (seuillage adaptatif).
    Le seuil d'Otsu calculé est loggé en DEBUG pour faciliter le réglage.

    Args:
        density_map: Array 2D (H, W) numérique — densité de points par pixel.
        method: Méthode de seuillage. Seul ``"otsu"`` est accepté.

    Returns:
        Array 2D (H, W) uint8 — image binaire (0=vide, 255=occupé).

    Raises:
        ValueError: Si ``method`` est inconnu.

    Example:
        >>> binary = binarize_density_map(dmap.image)
    """
    import cv2

    if method != "otsu":
        raise ValueError(f"Méthode de seuillage inconnue : '{method}'. Seul 'otsu' est supporté.")

    img_8bit = _to_uint8(density_map)
    otsu_threshold, binary = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    occupied = int((binary > 0).sum())
    logger.debug(
        "Binarisation Otsu : seuil=%.1f — %d px occupés / %d total.",
        otsu_threshold,
        occupied,
        binary.size,
    )
    return np.asarray(binary, dtype=np.uint8)


def morphological_cleanup(
    binary_image: np.ndarray,
    kernel_size: int = 5,
    close_iterations: int = 2,
    open_iterations: int = 1,
) -> np.ndarray:
    """Nettoie une image binaire par opérations morphologiques.

    Applique une fermeture (comble les trous dans les murs) puis une ouverture
    (supprime les artefacts isolés).

    Args:
        binary_image: Array 2D (H, W) uint8 — image binaire (0=vide, 255=occupé).
        kernel_size: Taille de l'élément structurant rectangulaire (pixels). Défaut : 5.
        close_iterations: Nombre d'itérations de fermeture. Défaut : 2.
        open_iterations: Nombre d'itérations d'ouverture. Défaut : 1.

    Returns:
        Array 2D (H, W) uint8 — image binaire nettoyée.

    Example:
        >>> cleaned = morphological_cleanup(binary, kernel_size=5, close_iterations=2)
    """
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if close_iterations > 0:
        closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    else:
        closed = binary_image
    if open_iterations > 0:
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    else:
        cleaned = closed

    occupied = int((cleaned > 0).sum())
    logger.debug(
        "Nettoyage morphologique (kernel=%d, close=%d, open=%d) : %d px occupés.",
        kernel_size,
        close_iterations,
        open_iterations,
        occupied,
    )
    return np.asarray(cleaned, dtype=np.uint8)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convertit une image en uint8 en normalisant sur [0, 255].

    Args:
        image: Array 2D numérique.

    Returns:
        Array 2D uint8.
    """
    max_val = image.max()
    if max_val == 0:
        return np.zeros_like(image, dtype=np.uint8)
    return np.asarray(((image.astype(np.float32) / max_val) * 255), dtype=np.uint8)
