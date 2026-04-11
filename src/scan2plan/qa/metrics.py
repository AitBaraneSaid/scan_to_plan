"""Métriques de qualité sur le plan produit."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QAReport:
    """Rapport de contrôle qualité du plan produit.

    Attributes:
        total_segments: Nombre total de segments dans le plan.
        short_segments: Nombre de segments < 10 cm (parasites).
        total_length_m: Longueur totale des murs en mètres.
        warnings: Liste de messages d'avertissement.
        errors: Liste d'erreurs critiques.
    """

    total_segments: int = 0
    short_segments: int = 0
    total_length_m: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True si le rapport ne contient aucune erreur critique."""
        return len(self.errors) == 0


def compute_basic_metrics(
    segments: np.ndarray,
    min_segment_length: float,
) -> QAReport:
    """Calcule les métriques de qualité de base sur les segments produits.

    Args:
        segments: Array (M, 4) float64 — segments [x1, y1, x2, y2] en mètres.
        min_segment_length: Seuil en dessous duquel un segment est considéré parasite.

    Returns:
        Rapport QA avec les métriques calculées.
    """
    report = QAReport()

    if len(segments) == 0:
        report.errors.append("Aucun segment produit par le pipeline.")
        return report

    lengths = np.hypot(
        segments[:, 2] - segments[:, 0],
        segments[:, 3] - segments[:, 1],
    )

    report.total_segments = len(segments)
    report.total_length_m = float(lengths.sum())
    report.short_segments = int((lengths < min_segment_length).sum())

    if report.short_segments > 0:
        report.warnings.append(
            f"{report.short_segments} micro-segments détectés (< {min_segment_length:.2f} m). "
            "Appliquer remove_short_segments() ou réduire max_line_gap."
        )

    logger.info(
        "QA — %d segments, longueur totale %.1f m, %d micro-segments parasites.",
        report.total_segments,
        report.total_length_m,
        report.short_segments,
    )
    return report
