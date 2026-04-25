"""Tests unitaires pour dxf_face_export.

Vérifie que l'export DXF des faces directes produit un fichier valide
avec les bons calques, les bonnes entités et les bonnes coordonnées.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from scan2plan.io.dxf_face_export import export_dxf_faces


# ---------------------------------------------------------------------------
# Fixtures et helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeSeg:
    """Segment factice imitant DetectedSegment."""
    x1: float
    y1: float
    x2: float
    y2: float
    source_slice: str = "high"
    confidence: float = 0.9


@dataclass
class _FakePair:
    """Paire de faces factice imitant FacePair de wall_pairing."""
    face_a: _FakeSeg
    face_b: _FakeSeg
    thickness: float
    overlap_length: float = 2.0
    score: float = 1.0


def _make_porteur_pair() -> _FakePair:
    """Mur porteur horizontal de 3 m, épaisseur 20 cm."""
    face_a = _FakeSeg(0.0, 0.0, 3.0, 0.0)
    face_b = _FakeSeg(0.0, 0.20, 3.0, 0.20)
    return _FakePair(face_a=face_a, face_b=face_b, thickness=0.20)


def _make_cloison_pair() -> _FakePair:
    """Cloison verticale de 2 m, épaisseur 8 cm."""
    face_a = _FakeSeg(0.0, 0.0, 0.0, 2.0)
    face_b = _FakeSeg(0.08, 0.0, 0.08, 2.0)
    return _FakePair(face_a=face_a, face_b=face_b, thickness=0.08)


def _make_unpaired_seg() -> _FakeSeg:
    """Segment non pairé — mur extérieur visible d'un seul côté."""
    return _FakeSeg(5.0, 0.0, 8.0, 0.0)


def _export_and_read(
    tmp_path: Path,
    segments: list,
    pairs: list,
    filename: str = "test_plan.dxf",
) -> object:
    """Exporte le DXF et le relit avec ezdxf — retourne le document."""
    import ezdxf

    out = tmp_path / filename
    export_dxf_faces(segments, pairs, out)
    return ezdxf.readfile(str(out))


# ---------------------------------------------------------------------------
# test_dxf_opens_without_error
# ---------------------------------------------------------------------------


def test_dxf_opens_without_error(tmp_path: Path) -> None:
    """Le DXF produit peut être relu par ezdxf.readfile() sans erreur."""
    pair = _make_porteur_pair()
    seg = pair.face_a
    doc = _export_and_read(tmp_path, [seg], [pair])
    assert doc is not None


# ---------------------------------------------------------------------------
# test_layers_created
# ---------------------------------------------------------------------------


_EXPECTED_LAYERS = {
    "MURS_PORTEURS",
    "CLOISONS",
    "MURS_SIMPLE",
    "OUVERTURES",
    "INCERTAIN",
    "HACHURES",
    "EPAISSEURS",
    "ANNOTATIONS",
}


def test_layers_created(tmp_path: Path) -> None:
    """Les 8 calques métier sont présents dans le DXF."""
    pair = _make_porteur_pair()
    doc = _export_and_read(tmp_path, [pair.face_a], [pair])
    layer_names = {layer.dxf.name for layer in doc.layers}
    assert _EXPECTED_LAYERS.issubset(layer_names)


# ---------------------------------------------------------------------------
# test_paired_walls_on_correct_layer
# ---------------------------------------------------------------------------


def test_paired_walls_on_correct_layer(tmp_path: Path) -> None:
    """Mur de 20 cm → entités LINE sur le calque MURS_PORTEURS."""
    pair = _make_porteur_pair()
    doc = _export_and_read(tmp_path, [pair.face_a, pair.face_b], [pair])
    msp = doc.modelspace()
    lines_on_porteurs = [
        e for e in msp
        if e.dxftype() == "LINE" and e.dxf.layer == "MURS_PORTEURS"
    ]
    assert len(lines_on_porteurs) >= 2  # au moins les deux faces


# ---------------------------------------------------------------------------
# test_thin_walls_on_correct_layer
# ---------------------------------------------------------------------------


def test_thin_walls_on_correct_layer(tmp_path: Path) -> None:
    """Mur de 8 cm (≤ 12 cm) → entités LINE sur le calque CLOISONS."""
    pair = _make_cloison_pair()
    doc = _export_and_read(tmp_path, [pair.face_a, pair.face_b], [pair])
    msp = doc.modelspace()
    lines_on_cloisons = [
        e for e in msp
        if e.dxftype() == "LINE" and e.dxf.layer == "CLOISONS"
    ]
    assert len(lines_on_cloisons) >= 2


# ---------------------------------------------------------------------------
# test_unpaired_on_murs_simple
# ---------------------------------------------------------------------------


def test_unpaired_on_murs_simple(tmp_path: Path) -> None:
    """Segments non pairés → LINE sur calque MURS_SIMPLE."""
    pair = _make_porteur_pair()
    unpaired = _make_unpaired_seg()
    # Fournir les deux faces + le segment non pairé comme segments
    all_segs = [pair.face_a, pair.face_b, unpaired]
    doc = _export_and_read(tmp_path, all_segs, [pair])
    msp = doc.modelspace()
    lines_simple = [
        e for e in msp
        if e.dxftype() == "LINE" and e.dxf.layer == "MURS_SIMPLE"
    ]
    assert len(lines_simple) >= 1


# ---------------------------------------------------------------------------
# test_hatch_between_faces
# ---------------------------------------------------------------------------


def test_hatch_between_faces(tmp_path: Path) -> None:
    """Chaque paire de faces produit au moins une entité HATCH dans le DXF."""
    pair = _make_porteur_pair()
    doc = _export_and_read(tmp_path, [pair.face_a, pair.face_b], [pair])
    msp = doc.modelspace()
    hatches = [e for e in msp if e.dxftype() == "HATCH"]
    assert len(hatches) >= 1


# ---------------------------------------------------------------------------
# test_thickness_annotations
# ---------------------------------------------------------------------------


def test_thickness_annotations(tmp_path: Path) -> None:
    """Chaque paire de faces produit un TEXT d'épaisseur sur calque EPAISSEURS."""
    pair = _make_porteur_pair()  # épaisseur 20 cm
    doc = _export_and_read(tmp_path, [pair.face_a, pair.face_b], [pair])
    msp = doc.modelspace()
    texts_epaisseurs = [
        e for e in msp
        if e.dxftype() == "TEXT" and e.dxf.layer == "EPAISSEURS"
    ]
    assert len(texts_epaisseurs) >= 1
    # Vérifier que le contenu contient "20cm"
    labels = [e.dxf.text for e in texts_epaisseurs]
    assert any("20" in lbl for lbl in labels)


# ---------------------------------------------------------------------------
# test_openings_detected
# ---------------------------------------------------------------------------


def test_openings_detected(tmp_path: Path) -> None:
    """Un gap de 80 cm entre deux paires colinéaires → ouverture sur OUVERTURES."""
    # Mur gauche : x=[0, 1.5], y=0 (face_a) et y=0.20 (face_b)
    left_pair = _FakePair(
        face_a=_FakeSeg(0.0, 0.0, 1.5, 0.0),
        face_b=_FakeSeg(0.0, 0.20, 1.5, 0.20),
        thickness=0.20,
    )
    # Mur droit : x=[2.3, 4.0], y=0 (face_a) et y=0.20 (face_b) — gap = 0.8 m
    right_pair = _FakePair(
        face_a=_FakeSeg(2.3, 0.0, 4.0, 0.0),
        face_b=_FakeSeg(2.3, 0.20, 4.0, 0.20),
        thickness=0.20,
    )
    all_segs = [
        left_pair.face_a, left_pair.face_b,
        right_pair.face_a, right_pair.face_b,
    ]
    doc = _export_and_read(tmp_path, all_segs, [left_pair, right_pair])
    msp = doc.modelspace()
    opening_entities = [
        e for e in msp if e.dxf.layer == "OUVERTURES"
    ]
    assert len(opening_entities) >= 1


# ---------------------------------------------------------------------------
# test_coordinates_in_meters
# ---------------------------------------------------------------------------


def test_coordinates_in_meters(tmp_path: Path) -> None:
    """Les coordonnées des LINE sont en mètres (pas en mm ni en cm)."""
    # Mur de 3 mètres de long
    pair = _make_porteur_pair()
    doc = _export_and_read(tmp_path, [pair.face_a, pair.face_b], [pair])
    msp = doc.modelspace()
    lines = [e for e in msp if e.dxftype() == "LINE"]
    assert lines, "Aucune LINE dans le DXF"

    # Toutes les coordonnées doivent être dans des plages métriques raisonnables
    # (pas des milliers de millimètres ou des fractions de millimètre)
    for line in lines:
        xs = [line.dxf.start[0], line.dxf.end[0]]
        ys = [line.dxf.start[1], line.dxf.end[1]]
        for x in xs:
            assert abs(x) < 1000.0, f"Coordonnée X suspecte (pas en mètres) : {x}"
        for y in ys:
            assert abs(y) < 1000.0, f"Coordonnée Y suspecte (pas en mètres) : {y}"

    # Vérifier que le mur de 3 m est bien à 3 m (pas 3000 mm)
    porteur_lines = [
        e for e in lines if e.dxf.layer == "MURS_PORTEURS"
    ]
    assert porteur_lines, "Aucune LINE sur MURS_PORTEURS"
    # La ligne la plus longue doit faire ~3 m
    max_length = max(
        abs(ln.dxf.end[0] - ln.dxf.start[0]) +
        abs(ln.dxf.end[1] - ln.dxf.start[1])
        for ln in porteur_lines
    )
    # Entre 2 m et 5 m (le mur fait exactement 3 m)
    assert 2.0 <= max_length <= 5.0, f"Longueur suspecte : {max_length}"
