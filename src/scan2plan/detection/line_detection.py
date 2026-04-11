"""Détection de segments de murs par transformée de Hough probabiliste."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def detect_segments_hough(
    binary_image: np.ndarray,
    rho: int,
    theta_deg: float,
    threshold: int,
    min_line_length: int,
    max_line_gap: int,
) -> np.ndarray:
    """Détecte des segments de droite dans une image binaire par Hough probabiliste.

    Args:
        binary_image: Array 2D uint8 — image binaire (0=vide, 255=occupé).
        rho: Résolution spatiale de l'accumulation (pixels).
        theta_deg: Résolution angulaire (degrés).
        threshold: Seuil d'accumulation pour valider une ligne.
        min_line_length: Longueur minimale d'un segment valide (pixels).
        max_line_gap: Gap maximal autorisé dans un segment (pixels).

    Returns:
        Array (M, 4) int32 — segments détectés [x1, y1, x2, y2] en pixels.
        Array vide de forme (0, 4) si aucun segment n'est détecté.
    """
    import cv2

    theta_rad = np.deg2rad(theta_deg)
    lines = cv2.HoughLinesP(
        binary_image,
        rho=rho,
        theta=theta_rad,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        logger.warning("Hough : aucun segment détecté.")
        return np.zeros((0, 4), dtype=np.int32)

    segments = lines.reshape(-1, 4)
    logger.info("Hough : %d segments détectés.", len(segments))
    return segments
