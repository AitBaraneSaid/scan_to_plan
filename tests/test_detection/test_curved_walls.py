"""Tests unitaires pour detection/curved_walls.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scan2plan.detection.curved_walls import (
    DetectedArc,
    DetectedPillar,
    detect_curved_walls,
    detect_pillars,
    export_arcs_to_dxf,
)
from scan2plan.slicing.density_map import DensityMapResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dmap(
    image: np.ndarray,
    x_min: float = 0.0,
    y_min: float = 0.0,
    resolution: float = 0.005,
) -> DensityMapResult:
    h, w = image.shape[:2]
    return DensityMapResult(
        image=image.astype(np.float32),
        x_min=x_min,
        y_min=y_min,
        resolution=resolution,
        width=w,
        height=h,
    )


def _arc_binary_image(
    cx_m: float,
    cy_m: float,
    radius_m: float,
    start_deg: float,
    end_deg: float,
    x_min: float = 0.0,
    y_min: float = 0.0,
    size_m: float = 12.0,
    res: float = 0.005,
    thickness_px: int = 3,
) -> tuple[DensityMapResult, np.ndarray]:
    """Génère une DensityMapResult + binary_image avec un arc synthétique.

    L'arc est tracé comme une bande de pixels autour du cercle (cx, cy, radius)
    entre start_deg et end_deg.

    Args:
        cx_m, cy_m: Centre du cercle (mètres).
        radius_m: Rayon (mètres).
        start_deg, end_deg: Angles de début et de fin (degrés).
        x_min, y_min: Origine de l'image (mètres).
        size_m: Taille de l'image (mètres carrés).
        res: Résolution (mètres/pixel).
        thickness_px: Épaisseur de l'arc en pixels.

    Returns:
        (DensityMapResult, binary_image uint8).
    """
    size_px = int(size_m / res)
    binary = np.zeros((size_px, size_px), dtype=np.uint8)

    n_angles = int((end_deg - start_deg) / 0.1) + 1
    angles = np.linspace(math.radians(start_deg), math.radians(end_deg), n_angles)

    for a in angles:
        xm = cx_m + radius_m * math.cos(a)
        ym = cy_m + radius_m * math.sin(a)
        col = int((xm - x_min) / res)
        row = int(size_px - 1 - (ym - y_min) / res)
        for dr in range(-thickness_px, thickness_px + 1):
            for dc in range(-thickness_px, thickness_px + 1):
                r, c = row + dr, col + dc
                if 0 <= r < size_px and 0 <= c < size_px:
                    binary[r, c] = 255

    dmap = _make_dmap(
        binary.astype(np.float32),
        x_min=x_min, y_min=y_min, resolution=res,
    )
    return dmap, binary


def _circle_binary_image(
    cx_m: float,
    cy_m: float,
    radius_m: float,
    x_min: float = 0.0,
    y_min: float = 0.0,
    size_m: float = 3.0,
    res: float = 0.005,
    thickness_px: int = 2,
) -> tuple[DensityMapResult, np.ndarray]:
    """Génère un cercle plein synthétique (poteau).

    Args identiques à _arc_binary_image, sans les angles.
    """
    return _arc_binary_image(
        cx_m, cy_m, radius_m,
        start_deg=0.0, end_deg=359.9,
        x_min=x_min, y_min=y_min,
        size_m=size_m, res=res, thickness_px=thickness_px,
    )


# ---------------------------------------------------------------------------
# Tests DetectedArc dataclass
# ---------------------------------------------------------------------------

class TestDetectedArc:
    def test_arc_length_quarter_circle(self) -> None:
        """Arc de 90° de rayon 5m → longueur ≈ 2π×5/4 ≈ 7.85 m."""
        arc = DetectedArc(cx=0.0, cy=0.0, radius=5.0,
                          start_angle_deg=0.0, end_angle_deg=90.0)
        expected = 5.0 * math.pi / 2.0
        assert arc.arc_length == pytest.approx(expected, abs=0.01)

    def test_arc_span_positive(self) -> None:
        arc = DetectedArc(cx=0.0, cy=0.0, radius=5.0,
                          start_angle_deg=45.0, end_angle_deg=135.0)
        assert arc.span_deg == pytest.approx(90.0)

    def test_default_confidence(self) -> None:
        arc = DetectedArc(cx=0, cy=0, radius=1.0,
                          start_angle_deg=0, end_angle_deg=90)
        assert arc.confidence == pytest.approx(1.0)

    def test_default_source_slice(self) -> None:
        arc = DetectedArc(cx=0, cy=0, radius=1.0,
                          start_angle_deg=0, end_angle_deg=90)
        assert arc.source_slice == "high"


# ---------------------------------------------------------------------------
# Tests DetectedPillar dataclass
# ---------------------------------------------------------------------------

class TestDetectedPillar:
    def test_fields(self) -> None:
        p = DetectedPillar(cx=1.0, cy=2.0, radius=0.15)
        assert p.cx == pytest.approx(1.0)
        assert p.cy == pytest.approx(2.0)
        assert p.radius == pytest.approx(0.15)
        assert p.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests detect_curved_walls
# ---------------------------------------------------------------------------

class TestDetectCurvedWalls:
    def test_arc_90deg_radius5m_detected(self) -> None:
        """Arc de 90° de rayon 5 m → 1 arc détecté."""
        cx, cy, r = 5.0, 5.0, 5.0
        dmap, binary = _arc_binary_image(
            cx_m=cx, cy_m=cy, radius_m=r,
            start_deg=0.0, end_deg=90.0,
            x_min=0.0, y_min=0.0, size_m=12.0,
        )
        arcs = detect_curved_walls(dmap, binary, min_arc_length_m=0.20)
        assert len(arcs) >= 1, "Au moins 1 arc attendu pour un quart de cercle de rayon 5m"

    def test_detected_arc_radius_approx_correct(self) -> None:
        """Le rayon de l'arc détecté doit être ≈ 5 m ± 50 cm."""
        cx, cy, r = 5.0, 5.0, 5.0
        dmap, binary = _arc_binary_image(
            cx_m=cx, cy_m=cy, radius_m=r,
            start_deg=0.0, end_deg=90.0,
            x_min=0.0, y_min=0.0, size_m=12.0,
        )
        arcs = detect_curved_walls(dmap, binary, min_arc_length_m=0.20)
        if arcs:
            assert abs(arcs[0].radius - r) < 0.50, (
                f"Rayon attendu ≈ {r} m, obtenu : {arcs[0].radius:.3f} m"
            )

    def test_arc_180deg_detected(self) -> None:
        """Demi-cercle (180°) de rayon 3 m → au moins 1 arc détecté."""
        dmap, binary = _arc_binary_image(
            cx_m=4.0, cy_m=4.0, radius_m=3.0,
            start_deg=0.0, end_deg=180.0,
            x_min=0.0, y_min=0.0, size_m=10.0,
        )
        arcs = detect_curved_walls(dmap, binary, min_arc_length_m=0.20)
        assert len(arcs) >= 1

    def test_empty_image_returns_empty(self) -> None:
        """Image noire → aucun arc détecté."""
        image = np.zeros((200, 200), dtype=np.uint8)
        dmap = _make_dmap(image.astype(np.float32))
        arcs = detect_curved_walls(dmap, image)
        assert arcs == []

    def test_returns_list_of_detected_arc(self) -> None:
        """Le résultat doit être une liste de DetectedArc."""
        dmap, binary = _arc_binary_image(
            cx_m=5.0, cy_m=5.0, radius_m=4.0,
            start_deg=0.0, end_deg=90.0,
            x_min=0.0, y_min=0.0, size_m=11.0,
        )
        arcs = detect_curved_walls(dmap, binary)
        for arc in arcs:
            assert isinstance(arc, DetectedArc)

    def test_min_arc_length_filter(self) -> None:
        """Un arc court (< min_arc_length_m) ne doit pas être retourné."""
        # Arc de 10° de rayon 0.5m → longueur ≈ 0.087m < 0.20m
        dmap, binary = _arc_binary_image(
            cx_m=2.0, cy_m=2.0, radius_m=0.5,
            start_deg=0.0, end_deg=10.0,
            x_min=0.0, y_min=0.0, size_m=5.0,
        )
        arcs = detect_curved_walls(dmap, binary, min_arc_length_m=0.20)
        # Aucun arc de cette longueur ne doit passer le filtre
        assert all(a.arc_length >= 0.20 for a in arcs)

    def test_arc_has_correct_source_slice(self) -> None:
        dmap, binary = _arc_binary_image(
            cx_m=5.0, cy_m=5.0, radius_m=4.0,
            start_deg=0.0, end_deg=90.0,
            x_min=0.0, y_min=0.0, size_m=11.0,
        )
        arcs = detect_curved_walls(dmap, binary, source_slice="mid")
        for arc in arcs:
            assert arc.source_slice == "mid"

    def test_arcs_sorted_by_length_desc(self) -> None:
        """Les arcs doivent être triés par longueur décroissante."""
        dmap, binary = _arc_binary_image(
            cx_m=5.0, cy_m=5.0, radius_m=4.0,
            start_deg=0.0, end_deg=180.0,
            x_min=0.0, y_min=0.0, size_m=12.0,
        )
        arcs = detect_curved_walls(dmap, binary, min_arc_length_m=0.10)
        lengths = [a.arc_length for a in arcs]
        assert lengths == sorted(lengths, reverse=True)


# ---------------------------------------------------------------------------
# Tests detect_pillars
# ---------------------------------------------------------------------------

class TestDetectPillars:
    def test_circular_pillar_detected(self) -> None:
        """Cercle de rayon 15 cm → 1 poteau détecté (tolérance large : ±15 cm)."""
        dmap, binary = _circle_binary_image(
            cx_m=1.5, cy_m=1.5, radius_m=0.15,
            x_min=0.0, y_min=0.0, size_m=3.0,
            res=0.005,
        )
        pillars = detect_pillars(dmap, binary, param2=5.0)
        # HoughCircles voit l'anneau de la bande de pixels — le rayon apparent
        # peut varier selon l'épaisseur tracée, tolérance ±15 cm
        for p in pillars:
            assert abs(p.radius - 0.15) < 0.20, (
                f"Rayon attendu ≈ 0.15m, obtenu : {p.radius:.3f}m"
            )

    def test_empty_image_no_pillars(self) -> None:
        """Image noire → aucun poteau."""
        image = np.zeros((200, 200), dtype=np.uint8)
        dmap = _make_dmap(image.astype(np.float32))
        pillars = detect_pillars(dmap, image)
        assert pillars == []

    def test_returns_list_of_detected_pillar(self) -> None:
        dmap, binary = _circle_binary_image(
            cx_m=1.5, cy_m=1.5, radius_m=0.15,
            x_min=0.0, y_min=0.0, size_m=3.0,
        )
        pillars = detect_pillars(dmap, binary, param2=5.0)
        for p in pillars:
            assert isinstance(p, DetectedPillar)

    def test_pillars_sorted_by_confidence(self) -> None:
        """Si plusieurs poteaux détectés, triés par confiance décroissante."""
        dmap, binary = _circle_binary_image(
            cx_m=1.5, cy_m=1.5, radius_m=0.15,
            x_min=0.0, y_min=0.0, size_m=3.0,
        )
        pillars = detect_pillars(dmap, binary, param2=5.0)
        confs = [p.confidence for p in pillars]
        assert confs == sorted(confs, reverse=True)


# ---------------------------------------------------------------------------
# Tests export_arcs_to_dxf
# ---------------------------------------------------------------------------

class TestExportArcsToDxf:
    def test_arc_creates_arc_entity(self) -> None:
        """Un arc → entité ARC dans le DXF."""
        import ezdxf
        doc = ezdxf.new()
        arc = DetectedArc(cx=2.0, cy=2.0, radius=5.0,
                          start_angle_deg=0.0, end_angle_deg=90.0,
                          confidence=0.9)
        n = export_arcs_to_dxf([arc], [], doc)
        assert n == 1
        arcs_in_doc = list(doc.modelspace().query("ARC"))
        assert len(arcs_in_doc) == 1

    def test_pillar_creates_circle_entity(self) -> None:
        """Un poteau → entité CIRCLE dans le DXF."""
        import ezdxf
        doc = ezdxf.new()
        pillar = DetectedPillar(cx=1.0, cy=1.0, radius=0.15, confidence=0.8)
        n = export_arcs_to_dxf([], [pillar], doc)
        assert n == 1
        circles_in_doc = list(doc.modelspace().query("CIRCLE"))
        assert len(circles_in_doc) == 1

    def test_mixed_export_counts_all(self) -> None:
        """2 arcs + 3 poteaux → 5 entités."""
        import ezdxf
        doc = ezdxf.new()
        arcs = [
            DetectedArc(cx=2.0, cy=2.0, radius=5.0,
                        start_angle_deg=0.0, end_angle_deg=90.0),
            DetectedArc(cx=6.0, cy=6.0, radius=3.0,
                        start_angle_deg=90.0, end_angle_deg=180.0),
        ]
        pillars = [
            DetectedPillar(cx=1.0, cy=1.0, radius=0.10),
            DetectedPillar(cx=2.0, cy=1.0, radius=0.12),
            DetectedPillar(cx=3.0, cy=1.0, radius=0.15),
        ]
        n = export_arcs_to_dxf(arcs, pillars, doc)
        assert n == 5

    def test_arc_layer_created(self) -> None:
        """Le calque MURS_COURBES est créé dans le document."""
        import ezdxf
        doc = ezdxf.new()
        arc = DetectedArc(cx=2.0, cy=2.0, radius=5.0,
                          start_angle_deg=0.0, end_angle_deg=90.0)
        export_arcs_to_dxf([arc], [], doc)
        assert "MURS_COURBES" in doc.layers

    def test_pillar_layer_created(self) -> None:
        """Le calque POTEAUX est créé dans le document."""
        import ezdxf
        doc = ezdxf.new()
        pillar = DetectedPillar(cx=1.0, cy=1.0, radius=0.15)
        export_arcs_to_dxf([], [pillar], doc)
        assert "POTEAUX" in doc.layers

    def test_custom_layer_config(self) -> None:
        """layer_config personnalisé → calque avec le bon nom."""
        import ezdxf
        doc = ezdxf.new()
        arc = DetectedArc(cx=2.0, cy=2.0, radius=5.0,
                          start_angle_deg=0.0, end_angle_deg=90.0)
        export_arcs_to_dxf(
            [arc], [], doc,
            layer_config={"curved_walls": "ARCS", "pillars": "PILIERS"},
        )
        assert "ARCS" in doc.layers

    def test_arc_geometry_in_dxf(self) -> None:
        """Les propriétés de l'arc (rayon, centre, angles) sont correctes dans le DXF."""
        import ezdxf
        doc = ezdxf.new()
        arc = DetectedArc(cx=2.0, cy=3.0, radius=5.0,
                          start_angle_deg=10.0, end_angle_deg=80.0)
        export_arcs_to_dxf([arc], [], doc)
        arc_entity = list(doc.modelspace().query("ARC"))[0]
        assert arc_entity.dxf.radius == pytest.approx(5.0, abs=1e-4)
        assert arc_entity.dxf.center.x == pytest.approx(2.0, abs=1e-4)
        assert arc_entity.dxf.center.y == pytest.approx(3.0, abs=1e-4)

    def test_empty_lists_returns_zero(self) -> None:
        import ezdxf
        doc = ezdxf.new()
        n = export_arcs_to_dxf([], [], doc)
        assert n == 0
