"""Binarisation et nettoyage morphologique des density maps."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def binarize_and_clean(
    density_map: np.ndarray,
    kernel_size: int,
    close_iterations: int,
    open_iterations: int,
) -> np.ndarray:
    """Binarise une density map et applique un nettoyage morphologique.

    Utilise le seuillage d'Otsu pour la binarisation, puis une fermeture
    morphologique pour combler les trous dans les murs, et une ouverture
    pour supprimer les artefacts isolés.

    Args:
        density_map: Array 2D (H, W) uint16 — densité de points par pixel.
        kernel_size: Taille de l'élément structurant (pixels, impair recommandé).
        close_iterations: Nombre d'itérations de fermeture.
        open_iterations: Nombre d'itérations d'ouverture.

    Returns:
        Array 2D (H, W) uint8 — image binaire (0=vide, 255=occupé).
    """
    import cv2

    img_8bit = _to_uint8(density_map)
    _, binary = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=open_iterations)

    occupied = int((cleaned > 0).sum())
    logger.debug(
        "Binarisation Otsu + morphologie (kernel=%d, close=%d, open=%d) : %d px occupés.",
        kernel_size,
        close_iterations,
        open_iterations,
        occupied,
    )
    return cleaned


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
    return ((image.astype(np.float32) / max_val) * 255).astype(np.uint8)
