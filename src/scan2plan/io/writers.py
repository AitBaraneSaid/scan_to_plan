"""Export DXF : écriture des entités vectorielles dans un fichier AutoCAD."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Nom du calque par défaut si aucune config n'est passée
_DEFAULT_LAYER = "MURS"


def write_segments_to_dxf(
    segments: np.ndarray,
    path: Path,
    layer: str = _DEFAULT_LAYER,
    dxf_version: str = "R2013",
) -> None:
    """Exporte une liste de segments 2D dans un fichier DXF.

    Chaque segment est écrit comme une entité LINE sur le calque spécifié.
    Les coordonnées sont en mètres, Z = 0.

    Args:
        segments: Array (M, 4) float64 avec colonnes [x1, y1, x2, y2] en mètres.
        path: Chemin de sortie du fichier .dxf.
        layer: Nom du calque DXF cible.
        dxf_version: Version DXF (ex: "R2013", "R2010").

    Raises:
        ValueError: Si segments n'a pas la forme (M, 4).

    Example:
        >>> segs = np.array([[0, 0, 4, 0], [4, 0, 4, 3]])
        >>> write_segments_to_dxf(segs, Path("plan.dxf"))
    """
    import ezdxf

    if segments.ndim != 2 or segments.shape[1] != 4:
        raise ValueError(
            f"segments doit être de forme (M, 4), reçu : {segments.shape}"
        )

    doc = ezdxf.new(dxfversion=dxf_version)
    msp = doc.modelspace()

    if layer not in doc.layers:
        doc.layers.add(layer)

    for x1, y1, x2, y2 in segments:
        msp.add_line(
            start=(float(x1), float(y1), 0.0),
            end=(float(x2), float(y2), 0.0),
            dxfattribs={"layer": layer},
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(path))
    logger.info(
        "DXF exporté : %s — %d segments sur le calque '%s' (version %s).",
        path,
        len(segments),
        layer,
        dxf_version,
    )
