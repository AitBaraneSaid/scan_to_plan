"""Validations de cohérence géométrique du plan produit."""

from __future__ import annotations

import logging

import numpy as np

from scan2plan.qa.metrics import QAReport

logger = logging.getLogger(__name__)

# Dimensions plausibles pour une pièce
_MIN_ROOM_DIM_M = 0.5
_MAX_ROOM_DIM_M = 50.0
_MIN_WALL_LENGTH_M = 0.1


def validate_plan(segments: np.ndarray, report: QAReport) -> QAReport:
    """Effectue des vérifications de cohérence sur le plan et enrichit le rapport QA.

    Vérifications effectuées :
    - Dimensions de la bounding box dans des valeurs plausibles.
    - Absence de segments dégénérés (longueur nulle).

    Args:
        segments: Array (M, 4) float64 — segments en mètres.
        report: Rapport QA à enrichir (modifié en place).

    Returns:
        Rapport QA enrichi.
    """
    if len(segments) == 0:
        return report

    all_x = np.concatenate([segments[:, 0], segments[:, 2]])
    all_y = np.concatenate([segments[:, 1], segments[:, 3]])
    width = float(all_x.max() - all_x.min())
    height_plan = float(all_y.max() - all_y.min())

    if width < _MIN_ROOM_DIM_M or height_plan < _MIN_ROOM_DIM_M:
        report.warnings.append(
            f"Plan très petit : {width:.2f} m × {height_plan:.2f} m. "
            "Vérifiez les paramètres de slicing et la qualité du scan."
        )
    if width > _MAX_ROOM_DIM_M or height_plan > _MAX_ROOM_DIM_M:
        report.warnings.append(
            f"Plan très grand : {width:.2f} m × {height_plan:.2f} m. "
            "Possible présence de segments parasites ou de points hors-bâtiment."
        )

    lengths = np.hypot(
        segments[:, 2] - segments[:, 0],
        segments[:, 3] - segments[:, 1],
    )
    degenerate = int((lengths < 1e-6).sum())
    if degenerate > 0:
        report.errors.append(f"{degenerate} segments dégénérés (longueur nulle) détectés.")

    logger.info(
        "Validation plan : %d warnings, %d erreurs.",
        len(report.warnings),
        len(report.errors),
    )
    return report
