"""Détection de segments de murs par transformée de Hough probabiliste."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from scan2plan.slicing.density_map import DensityMapResult
from scan2plan.utils.coordinate import pixel_to_metric
from scan2plan.utils.geometry import segment_length

logger = logging.getLogger(__name__)

# Longueur de référence pour la confiance : segments ≥ 1 m → confiance max
_CONFIDENCE_REF_LENGTH = 1.0


@dataclass
class DetectedSegment:
    """Segment de mur détecté, en coordonnées métriques.

    Attributes:
        x1: Coordonnée X du premier point (mètres).
        y1: Coordonnée Y du premier point (mètres).
        x2: Coordonnée X du second point (mètres).
        y2: Coordonnée Y du second point (mètres).
        source_slice: Identifiant de la slice d'origine ("high", "mid", "low").
        confidence: Score de confiance dans [0, 1] basé sur la longueur du segment.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    source_slice: str
    confidence: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Retourne le segment sous forme (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def length(self) -> float:
        """Longueur du segment en mètres."""
        return segment_length(self.as_tuple())


def detect_lines_hough(
    binary_image: np.ndarray,
    density_map_result: DensityMapResult,
    rho: int = 1,
    theta_deg: float = 0.5,
    threshold: int = 50,
    min_line_length: int = 50,
    max_line_gap: int = 20,
    source_slice: str = "mid",
) -> list[DetectedSegment]:
    """Détecte des segments de murs dans une image binaire par Hough probabiliste.

    Applique ``cv2.HoughLinesP`` sur l'image binaire, puis convertit chaque
    segment pixel → métrique via les métadonnées du ``DensityMapResult``.
    La conversion pixel ↔ métrique est effectuée UNE SEULE FOIS ici.

    Args:
        binary_image: Array 2D uint8 — image binaire (0=vide, 255=occupé).
        density_map_result: Métadonnées du raster (x_min, y_min, résolution, hauteur).
        rho: Résolution spatiale de l'accumulation (pixels). Défaut : 1.
        theta_deg: Résolution angulaire (degrés). Défaut : 0.5.
        threshold: Seuil d'accumulation pour valider une ligne. Défaut : 50.
        min_line_length: Longueur minimale d'un segment valide (pixels). Défaut : 50.
        max_line_gap: Gap maximal autorisé dans un segment (pixels). Défaut : 20.
        source_slice: Identifiant de la slice ("high", "mid", "low"). Défaut : "mid".

    Returns:
        Liste de ``DetectedSegment`` en coordonnées métriques.
        Liste vide si aucun segment n'est détecté.

    Example:
        >>> segments = detect_lines_hough(binary, dmap, threshold=30)
        >>> len(segments)
        4
    """
    import cv2

    theta_rad = np.deg2rad(theta_deg)
    lines = cv2.HoughLinesP(
        binary_image,
        rho=rho,
        theta=float(theta_rad),
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        logger.warning("Hough [%s] : aucun segment détecté.", source_slice)
        return []

    raw = lines.reshape(-1, 4)
    dmap = density_map_result
    result: list[DetectedSegment] = []

    for px1, py1, px2, py2 in raw:
        x1, y1 = pixel_to_metric(float(px1), float(py1), dmap.x_min, dmap.y_min,
                                  dmap.resolution, dmap.height)
        x2, y2 = pixel_to_metric(float(px2), float(py2), dmap.x_min, dmap.y_min,
                                  dmap.resolution, dmap.height)
        seg = DetectedSegment(x1=x1, y1=y1, x2=x2, y2=y2, source_slice=source_slice,
                               confidence=0.0)
        seg.confidence = min(1.0, seg.length / _CONFIDENCE_REF_LENGTH)
        result.append(seg)

    logger.info("Hough [%s] : %d segments détectés.", source_slice, len(result))
    return result
