"""Export DXF : écriture des entités vectorielles dans un fichier AutoCAD."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import ezdxf.document

    from scan2plan.detection.line_detection import DetectedSegment
    from scan2plan.detection.openings import Opening

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


def export_openings_to_dxf(
    openings: "list[Opening]",
    doc: "ezdxf.document.Drawing",
    layer_config: dict | None = None,
) -> int:
    """Ajoute les ouvertures dans un document DXF existant.

    Chaque ouverture est représentée par deux LINE perpendiculaires au mur,
    aux extrémités de l'ouverture, sur le calque PORTES ou FENETRES.

    Args:
        openings: Liste d'``Opening`` à exporter.
        doc: Document ezdxf ouvert (modifié en place).
        layer_config: Configuration des calques. Si fourni, utilise les noms
            ``layer_config["doors"]`` et ``layer_config["windows"]``. Sinon,
            utilise les noms par défaut ``"PORTES"`` et ``"FENETRES"``.

    Returns:
        Nombre d'entités LINE ajoutées.
    """
    door_layer = "PORTES"
    window_layer = "FENETRES"
    if layer_config:
        door_layer = layer_config.get("doors", door_layer)
        window_layer = layer_config.get("windows", window_layer)

    _add_layer(doc, door_layer, _LAYER_COLORS.get(door_layer, 1))
    _add_layer(doc, window_layer, _LAYER_COLORS.get(window_layer, 5))

    msp = doc.modelspace()
    n_entities = 0

    for op in openings:
        seg = op.wall_segment
        length = seg.length
        if length < 1e-9:
            continue

        # Direction unitaire le long du mur et perpendiculaire
        dx = (seg.x2 - seg.x1) / length
        dy = (seg.y2 - seg.y1) / length
        px, py = -dy, dx   # perpendiculaire, longueur = 1 m

        # Demi-épaisseur du trait perpendiculaire = 10 cm
        half = 0.10
        layer = door_layer if op.type == "door" else window_layer

        # Trait au début de l'ouverture
        sx1 = seg.x1 + op.position_start * dx
        sy1 = seg.y1 + op.position_start * dy
        msp.add_line(
            start=(sx1 - half * px, sy1 - half * py, 0.0),
            end=(sx1 + half * px, sy1 + half * py, 0.0),
            dxfattribs={"layer": layer},
        )
        n_entities += 1

        # Trait à la fin de l'ouverture
        ex1 = seg.x1 + op.position_end * dx
        ey1 = seg.y1 + op.position_end * dy
        msp.add_line(
            start=(ex1 - half * px, ey1 - half * py, 0.0),
            end=(ex1 + half * px, ey1 + half * py, 0.0),
            dxfattribs={"layer": layer},
        )
        n_entities += 1

    logger.info(
        "export_openings_to_dxf : %d ouvertures → %d entités LINE.",
        len(openings),
        n_entities,
    )
    return n_entities


def export_dxf_v1(
    wall_graph: "Any",
    openings: "list[Opening]",
    output_path: Path,
    config: "Any | None" = None,
) -> Path:
    """Exporte le plan V1 : murs structurés par calques métier + ouvertures.

    Calques créés :
    - MURS (blanc ACI 7, lineweight 35 = 0.35 mm) : segments de haute confiance.
    - CLOISONS (cyan ACI 4) : segments avec source_slice="low" uniquement.
    - INCERTAIN (jaune ACI 2, tirets) : segments à confiance < seuil.
    - PORTES (rouge ACI 1) : ouvertures de type porte.
    - FENETRES (bleu ACI 5) : ouvertures de type fenêtre.

    Args:
        wall_graph: WallGraph avec les segments régularisés et nettoyés.
        openings: Liste d'ouvertures détectées.
        output_path: Chemin du fichier DXF de sortie.
        config: ScanConfig optionnel pour les noms de calques et la version DXF.

    Returns:
        Chemin du fichier DXF créé.
    """
    import ezdxf

    version = "R2013"
    layer_names = {
        "walls": "MURS",
        "partitions": "CLOISONS",
        "doors": "PORTES",
        "windows": "FENETRES",
        "uncertain": "INCERTAIN",
    }
    confidence_threshold = 0.50

    if config is not None:
        version = getattr(config.dxf, "version", version)
        cfg_layers = getattr(config.dxf, "layers", {})
        layer_names.update(cfg_layers)

    doc = ezdxf.new(dxfversion=version)

    # Calques métier avec couleurs et attributs
    _add_layer(doc, layer_names["walls"], 7)       # blanc
    _add_layer(doc, layer_names["partitions"], 4)  # cyan
    _add_layer(doc, layer_names["uncertain"], 2)   # jaune
    _add_layer(doc, layer_names["doors"], 1)       # rouge
    _add_layer(doc, layer_names["windows"], 5)     # bleu

    # Linetype tirets pour INCERTAIN
    _ensure_dashed_linetype(doc)
    doc.layers.get(layer_names["uncertain"]).dxf.linetype = "DASHED"

    # Lineweight 35 (= 0.35 mm) pour MURS
    doc.layers.get(layer_names["walls"]).dxf.lineweight = 35

    msp = doc.modelspace()

    # ---- Segments de murs -------------------------------------------------
    for seg in wall_graph.segments:
        layer = _classify_segment_layer(seg, layer_names, confidence_threshold)
        msp.add_line(
            start=(seg.x1, seg.y1, 0.0),
            end=(seg.x2, seg.y2, 0.0),
            dxfattribs={"layer": layer},
        )

    # ---- Ouvertures -------------------------------------------------------
    export_openings_to_dxf(openings, doc, layer_names)

    output_path = output_path.with_suffix(".dxf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(output_path))

    logger.info(
        "DXF V1 exporté : %s — %d segments, %d ouvertures (version %s).",
        output_path,
        len(wall_graph.segments),
        len(openings),
        version,
    )
    return output_path


def _classify_segment_layer(
    seg: "DetectedSegment",
    layer_names: dict[str, str],
    confidence_threshold: float,
) -> str:
    """Détermine le calque d'un segment selon sa confiance et sa source.

    Args:
        seg: Segment de mur.
        layer_names: Mapping {clé: nom_calque}.
        confidence_threshold: Seuil de confiance pour le calque INCERTAIN.

    Returns:
        Nom du calque DXF.
    """
    if seg.confidence < confidence_threshold:
        return layer_names["uncertain"]
    if seg.source_slice == "low":
        return layer_names["partitions"]
    return layer_names["walls"]


def _ensure_dashed_linetype(doc: "ezdxf.document.Drawing") -> None:
    """Ajoute le linetype DASHED au document s'il n'existe pas.

    Args:
        doc: Document ezdxf.
    """
    if "DASHED" not in doc.linetypes:
        doc.linetypes.new("DASHED", dxfattribs={"description": "Dashed"})


def _add_layer(doc: "ezdxf.document.Drawing", name: str, color: int) -> None:
    """Ajoute un calque au document s'il n'existe pas déjà.

    Args:
        doc: Document ezdxf.
        name: Nom du calque.
        color: Couleur ACI (entier 1-255).
    """
    if name not in doc.layers:
        doc.layers.new(name, dxfattribs={"color": color})
