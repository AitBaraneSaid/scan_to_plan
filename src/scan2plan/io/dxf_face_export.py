"""Export DXF direct des faces de murs — plan de travail pour le technicien.

Exporte les segments tels qu'ils ont été scannés (deux faces parallèles par mur
physique) sans passer par des médianes. Organisation en calques métier avec
hachures, annotations d'épaisseur et ouvertures détectées.

Ce DXF est un plan à ~80 % prêt — le technicien finalise dans AutoCAD :
ajout des symboles de portes/fenêtres, cotations finales, noms de pièces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calques et couleurs ACI
# ---------------------------------------------------------------------------

_LAYERS: dict[str, int] = {
    "MURS_PORTEURS": 7,    # blanc  — faces pairées, épaisseur > 12 cm
    "CLOISONS":      3,    # vert   — faces pairées, épaisseur ≤ 12 cm
    "MURS_SIMPLE":   4,    # cyan   — faces non pairées (mur vu d'un seul côté)
    "OUVERTURES":    1,    # rouge  — ouvertures détectées
    "INCERTAIN":     8,    # gris   — segments à faible confiance
    "HACHURES":      31,   # orange pâle — hachures entre les faces
    "EPAISSEURS":    2,    # jaune  — textes d'épaisseur
    "ANNOTATIONS":   6,    # magenta — cartouche et annotations générales
}

# Épaisseur de cloison maximale (m) — au-dessus = mur porteur
_PARTITION_THRESHOLD = 0.12  # 12 cm

# Hauteur du texte d'épaisseur (m) — lisible à l'échelle 1:100
_TEXT_HEIGHT = 0.05

# Longueur des traits d'ouverture perpendiculaires (m)
_OPENING_TICK_LENGTH = 0.10

# Scale du pattern ANSI31 adapté à des plans en mètres 1:100
_HATCH_SCALE = 0.01


# ---------------------------------------------------------------------------
# Structures de données locales
# ---------------------------------------------------------------------------


@dataclass
class _Segment:
    """Segment minimal pour les besoins de cet export."""
    x1: float
    y1: float
    x2: float
    y2: float
    source_slice: str = "high"
    confidence: float = 0.9

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    @property
    def midpoint(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def direction(self) -> tuple[float, float]:
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        lg = float(np.hypot(dx, dy))
        if lg < 1e-9:
            return 1.0, 0.0
        return dx / lg, dy / lg

    @property
    def normal(self) -> tuple[float, float]:
        ux, uy = self.direction
        return -uy, ux


@dataclass
class _FacePair:
    """Paire de faces avec épaisseur pour l'export."""
    face_a: _Segment
    face_b: _Segment
    thickness: float
    overlap_length: float = 0.0
    score: float = 0.0


@dataclass
class _Opening:
    """Ouverture détectée dans un mur pairé."""
    x1: float          # début de l'ouverture (point sur face_a)
    y1: float
    x2: float          # fin de l'ouverture
    y2: float
    width: float       # largeur de l'ouverture (m)
    wall_normal: tuple[float, float]  # normale au mur pour les traits


# ---------------------------------------------------------------------------
# Helpers géométriques
# ---------------------------------------------------------------------------


def _setup_layers(doc: object) -> None:
    """Crée les 8 calques avec leurs couleurs ACI dans le document.

    Args:
        doc: Document ezdxf.
    """
    layers = doc.layers  # type: ignore[attr-defined]
    for name, color in _LAYERS.items():
        if name not in layers:
            layer = layers.add(name)
            layer.color = color


def _wall_layer(thickness: float) -> str:
    """Retourne le calque selon l'épaisseur du mur.

    Args:
        thickness: Épaisseur du mur en mètres.

    Returns:
        ``"CLOISONS"`` si ≤ 12 cm, ``"MURS_PORTEURS"`` sinon.
    """
    return "CLOISONS" if thickness <= _PARTITION_THRESHOLD else "MURS_PORTEURS"


def _project_on_axis(
    pt: tuple[float, float],
    origin: tuple[float, float],
    direction: tuple[float, float],
) -> float:
    """Projette un point sur un axe paramétrique.

    Args:
        pt: Point à projeter.
        origin: Origine de l'axe.
        direction: Vecteur unitaire de l'axe.

    Returns:
        Paramètre scalaire de la projection (mètres).
    """
    return (pt[0] - origin[0]) * direction[0] + (pt[1] - origin[1]) * direction[1]


# ---------------------------------------------------------------------------
# Export des murs pairés (double ligne + hachure + annotation)
# ---------------------------------------------------------------------------


def _export_paired_wall(msp: object, pair: _FacePair) -> None:
    """Exporte une paire de faces : deux LINE + HATCH + TEXT d'épaisseur.

    Args:
        msp: Model space ezdxf.
        pair: Paire de faces avec épaisseur.
    """
    layer = _wall_layer(pair.thickness)
    fa, fb = pair.face_a, pair.face_b

    # Les deux faces
    msp.add_line(  # type: ignore[attr-defined]
        (fa.x1, fa.y1), (fa.x2, fa.y2),
        dxfattribs={"layer": layer},
    )
    msp.add_line(  # type: ignore[attr-defined]
        (fb.x1, fb.y1), (fb.x2, fb.y2),
        dxfattribs={"layer": layer},
    )

    # Hachure entre les deux faces
    _add_wall_hatch(msp, pair)

    # Annotation de l'épaisseur au centre
    _add_thickness_text(msp, pair)


def _add_wall_hatch(msp: object, pair: _FacePair) -> None:
    """Ajoute une hachure ANSI31 entre les deux faces du mur.

    Construit un quadrilatère avec les 4 extrémités des deux faces
    (fa.p1, fa.p2, fb.p2, fb.p1) et le remplit avec ANSI31.

    Args:
        msp: Model space ezdxf.
        pair: Paire de faces.
    """
    fa, fb = pair.face_a, pair.face_b
    # Quadrilatère orienté dans le sens trigonométrique
    pts = [
        (fa.x1, fa.y1),
        (fa.x2, fa.y2),
        (fb.x2, fb.y2),
        (fb.x1, fb.y1),
        (fa.x1, fa.y1),  # fermeture
    ]
    try:
        hatch = msp.add_hatch(color=256, dxfattribs={"layer": "HACHURES"})  # type: ignore[attr-defined]
        hatch.set_pattern_fill("ANSI31", scale=_HATCH_SCALE)
        with hatch.edit_boundary() as boundary:
            boundary.add_polyline_path(pts)
    except Exception as exc:
        logger.debug("_add_wall_hatch : hachure ignorée (%s).", exc)


def _add_thickness_text(msp: object, pair: _FacePair) -> None:
    """Ajoute un TEXT d'épaisseur perpendiculaire au centre du mur.

    Args:
        msp: Model space ezdxf.
        pair: Paire de faces.
    """
    fa = pair.face_a
    cx, cy = fa.midpoint
    # Décaler légèrement dans la direction perpendiculaire pour lisibilité
    nx, ny = fa.normal
    tx = cx + nx * pair.thickness * 0.5
    ty = cy + ny * pair.thickness * 0.5

    thickness_cm = round(pair.thickness * 100)
    label = f"{thickness_cm}cm"

    # Angle du texte aligné sur le mur
    angle_deg = float(np.degrees(np.arctan2(fa.y2 - fa.y1, fa.x2 - fa.x1)))

    msp.add_text(  # type: ignore[attr-defined]
        label,
        height=_TEXT_HEIGHT,
        dxfattribs={
            "layer": "EPAISSEURS",
            "insert": (tx, ty),
            "rotation": angle_deg,
        },
    )


# ---------------------------------------------------------------------------
# Export des murs non pairés
# ---------------------------------------------------------------------------


def _export_unpaired_wall(msp: object, segment: _Segment) -> None:
    """Exporte un segment non pairé comme LINE simple sur MURS_SIMPLE.

    Ces murs sont vus d'un seul côté : murs extérieurs, mitoyens, ou zones
    occultées par du mobilier. Le technicien ajoutera la deuxième face.

    Args:
        msp: Model space ezdxf.
        segment: Segment non pairé.
    """
    msp.add_line(  # type: ignore[attr-defined]
        (segment.x1, segment.y1),
        (segment.x2, segment.y2),
        dxfattribs={"layer": "MURS_SIMPLE"},
    )


# ---------------------------------------------------------------------------
# Détection et export des ouvertures
# ---------------------------------------------------------------------------


def _detect_openings_from_gaps(
    face_pairs: list[_FacePair],
    min_opening_width: float = 0.50,
    max_opening_width: float = 2.50,
) -> list[_Opening]:
    """Détecte les ouvertures comme des gaps entre faces pairées colinéaires.

    Pour chaque paire de paires qui partagent approximativement la même droite
    porteuse, le gap entre elles (si > min_opening_width) est une ouverture.

    Cette méthode simple repose sur le fait que les murs Hough se fragmentent
    naturellement aux portes : le scanner ne voit pas le dessus de la porte
    (la surface de la porte fermée ou l'absence de mur) et Hough produit deux
    segments distincts.

    Args:
        face_pairs: Paires de faces identifiées.
        min_opening_width: Largeur minimale pour être une ouverture (m).
        max_opening_width: Largeur maximale (m). Au-delà, probable erreur.

    Returns:
        Liste d'ouvertures détectées.
    """
    openings: list[_Opening] = []
    angle_tol = float(np.deg2rad(5.0))
    perp_tol = 0.05  # 5 cm

    for i, pi in enumerate(face_pairs):
        for j, pj in enumerate(face_pairs):
            if j <= i:
                continue
            # Même direction ?
            ai = float(np.arctan2(pi.face_a.y2 - pi.face_a.y1,
                                  pi.face_a.x2 - pi.face_a.x1) % np.pi)
            aj = float(np.arctan2(pj.face_a.y2 - pj.face_a.y1,
                                  pj.face_a.x2 - pj.face_a.x1) % np.pi)
            diff = abs(ai - aj) % np.pi
            if min(diff, np.pi - diff) > angle_tol:
                continue

            # Même droite porteuse ?
            origin = (pi.face_a.x1, pi.face_a.y1)
            direction = pi.face_a.direction
            nx, ny = pi.face_a.normal
            mid_j = pj.face_a.midpoint
            perp = abs((mid_j[0] - origin[0]) * nx + (mid_j[1] - origin[1]) * ny)
            if perp > perp_tol:
                continue

            # Projeter les 4 extrémités sur l'axe
            ti1 = _project_on_axis((pi.face_a.x1, pi.face_a.y1), origin, direction)
            ti2 = _project_on_axis((pi.face_a.x2, pi.face_a.y2), origin, direction)
            tj1 = _project_on_axis((pj.face_a.x1, pj.face_a.y1), origin, direction)
            tj2 = _project_on_axis((pj.face_a.x2, pj.face_a.y2), origin, direction)

            ti_min, ti_max = min(ti1, ti2), max(ti1, ti2)
            tj_min, tj_max = min(tj1, tj2), max(tj1, tj2)

            # Gap entre les deux paires
            if tj_min >= ti_max:
                gap = tj_min - ti_max
                gap_start, gap_end = ti_max, tj_min
            elif ti_min >= tj_max:
                gap = ti_min - tj_max
                gap_start, gap_end = tj_max, ti_min
            else:
                continue  # chevauchement → pas de gap

            if min_opening_width <= gap <= max_opening_width:
                # Point du début et fin de l'ouverture sur face_a
                ox1 = origin[0] + gap_start * direction[0]
                oy1 = origin[1] + gap_start * direction[1]
                ox2 = origin[0] + gap_end * direction[0]
                oy2 = origin[1] + gap_end * direction[1]
                openings.append(_Opening(
                    x1=ox1, y1=oy1,
                    x2=ox2, y2=oy2,
                    width=gap,
                    wall_normal=(nx, ny),
                ))

    return openings


def _export_opening(msp: object, opening: _Opening) -> None:
    """Exporte une ouverture : deux traits perpendiculaires + TEXT de largeur.

    Dessine deux tirets perpendiculaires au mur (10 cm vers l'intérieur)
    aux bords de l'ouverture, signalant la position au technicien.

    Args:
        msp: Model space ezdxf.
        opening: Ouverture à exporter.
    """
    nx, ny = opening.wall_normal
    tick = _OPENING_TICK_LENGTH

    # Trait au début de l'ouverture
    msp.add_line(  # type: ignore[attr-defined]
        (opening.x1, opening.y1),
        (opening.x1 + nx * tick, opening.y1 + ny * tick),
        dxfattribs={"layer": "OUVERTURES"},
    )
    # Trait à la fin de l'ouverture
    msp.add_line(  # type: ignore[attr-defined]
        (opening.x2, opening.y2),
        (opening.x2 + nx * tick, opening.y2 + ny * tick),
        dxfattribs={"layer": "OUVERTURES"},
    )
    # Texte de largeur au centre de l'ouverture
    cx = (opening.x1 + opening.x2) / 2.0
    cy = (opening.y1 + opening.y2) / 2.0
    label = f"{opening.width:.2f}"
    msp.add_text(  # type: ignore[attr-defined]
        label,
        height=_TEXT_HEIGHT,
        dxfattribs={
            "layer": "OUVERTURES",
            "insert": (cx + nx * tick * 0.5, cy + ny * tick * 0.5),
        },
    )


# ---------------------------------------------------------------------------
# Cartouche et annotations générales
# ---------------------------------------------------------------------------


def _add_helper_annotations(
    msp: object,
    face_pairs: list[_FacePair],
    unpaired: list[_Segment],
    openings: list[_Opening],
) -> None:
    """Ajoute un cartouche récapitulatif en bas à gauche du plan.

    Args:
        msp: Model space ezdxf.
        face_pairs: Paires de faces.
        unpaired: Segments non pairés.
        openings: Ouvertures détectées.
    """
    n_porteurs = sum(1 for p in face_pairs if p.thickness > _PARTITION_THRESHOLD)
    n_cloisons = len(face_pairs) - n_porteurs
    lines = [
        "Plan genere automatiquement -- A completer",
        f"Murs porteurs : {n_porteurs}  Cloisons : {n_cloisons}",
        f"Murs simples : {len(unpaired)}",
        f"Ouvertures detectees : {len(openings)}",
        "Completer : portes, fenetres, cotations, noms de pieces",
    ]

    # Position : calculer le bord inférieur-gauche du plan
    all_x = []
    all_y = []
    for p in face_pairs:
        all_x += [p.face_a.x1, p.face_a.x2, p.face_b.x1, p.face_b.x2]
        all_y += [p.face_a.y1, p.face_a.y2, p.face_b.y1, p.face_b.y2]
    for s in unpaired:
        all_x += [s.x1, s.x2]
        all_y += [s.y1, s.y2]

    if not all_x:
        return

    x0 = min(all_x)
    y0 = min(all_y) - 0.8  # 80 cm sous le plan

    line_height = _TEXT_HEIGHT * 1.5
    for k, line in enumerate(lines):
        msp.add_text(  # type: ignore[attr-defined]
            line,
            height=_TEXT_HEIGHT,
            dxfattribs={
                "layer": "ANNOTATIONS",
                "insert": (x0, y0 - k * line_height),
            },
        )


# ---------------------------------------------------------------------------
# Conversion des types externes vers types locaux
# ---------------------------------------------------------------------------


def _to_local_segment(seg: object) -> _Segment:
    """Convertit un DetectedSegment ou Segment quelconque vers _Segment local.

    Args:
        seg: Objet segment avec attributs x1, y1, x2, y2.

    Returns:
        Instance ``_Segment``.
    """
    return _Segment(
        x1=float(getattr(seg, "x1", 0.0)),
        y1=float(getattr(seg, "y1", 0.0)),
        x2=float(getattr(seg, "x2", 0.0)),
        y2=float(getattr(seg, "y2", 0.0)),
        source_slice=str(getattr(seg, "source_slice", "high")),
        confidence=float(getattr(seg, "confidence", 0.9)),
    )


def _to_local_pair(pair: object) -> _FacePair:
    """Convertit un FacePair (wall_pairing) vers _FacePair local.

    Args:
        pair: Objet avec attributs face_a, face_b, thickness.

    Returns:
        Instance ``_FacePair``.
    """
    return _FacePair(
        face_a=_to_local_segment(getattr(pair, "face_a")),
        face_b=_to_local_segment(getattr(pair, "face_b")),
        thickness=float(getattr(pair, "thickness", 0.15)),
        overlap_length=float(getattr(pair, "overlap_length", 0.0)),
        score=float(getattr(pair, "score", 0.0)),
    )


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------


def export_dxf_faces(
    segments: list[object],
    face_pairs: list[object],
    output_path: Path,
    config: dict | None = None,
) -> Path:
    """Exporte le plan en DXF avec faces directes et calques métier.

    Remplace l'ancien export qui utilisait des médianes. Chaque segment est
    exporté tel qu'il a été détecté (face réelle scannée). Les paires reçoivent
    une hachure et une annotation d'épaisseur. Les segments non pairés sont
    sur MURS_SIMPLE.

    Args:
        segments: Tous les segments du pipeline (pairés + non pairés).
        face_pairs: Paires de faces issues de ``pair_wall_faces``.
        output_path: Chemin du fichier DXF de sortie.
        config: Configuration optionnelle (non utilisée pour l'instant).

    Returns:
        Chemin absolu du fichier créé.

    Example:
        >>> path = export_dxf_faces(segs, pairs, Path("plan.dxf"))
        >>> path.exists()
        True
    """
    import ezdxf

    output_path = Path(output_path)

    doc = ezdxf.new(dxfversion="R2013")
    doc.header["$INSUNITS"] = 6  # mètres
    msp = doc.modelspace()

    _setup_layers(doc)

    # Convertir vers les types locaux
    local_pairs = [_to_local_pair(p) for p in face_pairs]
    local_segments = [_to_local_segment(s) for s in segments]

    # Identifier les segments déjà pairés (pour ne pas les doubler sur MURS_SIMPLE)
    paired_ids: set[int] = set()
    for lp in local_pairs:
        for seg in local_segments:
            if (abs(seg.x1 - lp.face_a.x1) < 1e-6 and
                    abs(seg.y1 - lp.face_a.y1) < 1e-6):
                paired_ids.add(id(seg))
            if (abs(seg.x1 - lp.face_b.x1) < 1e-6 and
                    abs(seg.y1 - lp.face_b.y1) < 1e-6):
                paired_ids.add(id(seg))

    unpaired = [s for s in local_segments if id(s) not in paired_ids]

    # Export des murs pairés
    for lp in local_pairs:
        _export_paired_wall(msp, lp)

    # Export des murs non pairés
    for seg in unpaired:
        _export_unpaired_wall(msp, seg)

    # Détection des ouvertures
    openings = _detect_openings_from_gaps(local_pairs)
    for op in openings:
        _export_opening(msp, op)

    # Cartouche
    _add_helper_annotations(msp, local_pairs, unpaired, openings)

    doc.saveas(output_path)

    logger.info(
        "export_dxf_faces : %s — %d paires, %d simples, %d ouvertures.",
        output_path.name,
        len(local_pairs),
        len(unpaired),
        len(openings),
    )
    return output_path.resolve()
