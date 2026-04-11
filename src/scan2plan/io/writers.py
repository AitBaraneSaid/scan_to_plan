"""Export DXF : écriture des entités vectorielles dans un fichier AutoCAD."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import ezdxf.document
    from scan2plan.detection.line_detection import DetectedSegment

logger = logging.getLogger(__name__)

# Nom du calque par défaut si aucune config n'est passée
_DEFAULT_LAYER = "MURS"

# Calque unique pour le MVP
_MVP_LAYER = "MURS_DETECTES"

# Couleurs ACI (AutoCAD Color Index) par calque métier
_LAYER_COLORS: dict[str, int] = {
    "MURS": 7,          # blanc
    "CLOISONS": 3,      # vert
    "PORTES": 1,        # rouge
    "FENETRES": 5,      # bleu
    "INCERTAIN": 8,     # gris
    _MVP_LAYER: 7,      # blanc (MVP)
}


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
        doc.layers.new(layer)

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


def export_dxf(
    segments: "list[DetectedSegment]",
    output_path: Path,
    version: str = "R2013",
    layer_config: dict | None = None,
) -> Path:
    """Exporte les segments détectés dans un fichier DXF.

    Pour le MVP, tous les segments vont sur le calque ``"MURS_DETECTES"``.
    Si ``layer_config`` est fourni, les calques sont créés via ``setup_dxf_layers``
    et les segments peuvent être répartis selon leur ``source_slice`` (V1+).

    Args:
        segments: Liste de ``DetectedSegment`` en coordonnées métriques.
        output_path: Chemin du fichier DXF de sortie (.dxf).
        version: Version DXF (``"R2010"`` ou ``"R2013"``). Défaut : ``"R2013"``.
        layer_config: Configuration des calques (optionnel). Si fourni, les calques
            définis dans la config sont créés. Non utilisé pour l'affectation des
            entités en MVP.

    Returns:
        Le chemin du fichier créé (= ``output_path`` normalisé avec extension .dxf).

    Raises:
        ValueError: Si ``version`` n'est pas une version DXF valide reconnue par ezdxf.

    Example:
        >>> path = export_dxf(segments, Path("output/plan.dxf"))
        >>> path.exists()
        True
    """
    import ezdxf

    doc = ezdxf.new(dxfversion=version)
    msp = doc.modelspace()

    # Créer le calque MVP
    _add_layer(doc, _MVP_LAYER, _LAYER_COLORS[_MVP_LAYER])

    # Créer les calques métier si une config est fournie (V1+)
    if layer_config is not None:
        setup_dxf_layers(doc, layer_config)

    for seg in segments:
        msp.add_line(
            start=(seg.x1, seg.y1, 0.0),
            end=(seg.x2, seg.y2, 0.0),
            dxfattribs={"layer": _MVP_LAYER},
        )

    # Assurer que l'extension est .dxf
    output_path = output_path.with_suffix(".dxf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(output_path))

    logger.info(
        "DXF exporté : %s — %d entités LINE sur '%s' (version %s).",
        output_path,
        len(segments),
        _MVP_LAYER,
        version,
    )
    return output_path


def setup_dxf_layers(
    doc: "ezdxf.document.Drawing",
    layer_config: dict,
) -> None:
    """Configure les calques DXF métier selon la configuration (V1+).

    Crée les calques MURS, CLOISONS, PORTES, FENETRES, INCERTAIN avec
    une couleur ACI distincte pour chacun. Les calques déjà existants
    dans le document ne sont pas modifiés.

    Args:
        doc: Document ezdxf en cours de construction.
        layer_config: Dictionnaire ``{clé: nom_calque}`` issu de la config YAML,
            par exemple ``{"walls": "MURS", "doors": "PORTES", ...}``.

    Example:
        >>> setup_dxf_layers(doc, {"walls": "MURS", "doors": "PORTES"})
    """
    for layer_name in layer_config.values():
        color = _LAYER_COLORS.get(layer_name, 7)
        _add_layer(doc, layer_name, color)
    logger.debug(
        "Calques DXF configurés : %s.",
        ", ".join(str(v) for v in layer_config.values()),
    )


def _add_layer(doc: "ezdxf.document.Drawing", name: str, color: int) -> None:
    """Ajoute un calque au document s'il n'existe pas déjà.

    Args:
        doc: Document ezdxf.
        name: Nom du calque.
        color: Couleur ACI (entier 1-255).
    """
    if name not in doc.layers:
        doc.layers.new(name, dxfattribs={"color": color})
