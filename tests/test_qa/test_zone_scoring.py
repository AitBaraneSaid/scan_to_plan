"""Tests unitaires pour qa/zone_scoring.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from scan2plan.detection.line_detection import DetectedSegment
from scan2plan.detection.openings import Opening
from scan2plan.qa.zone_scoring import (
    ZoneMap,
    ZoneScore,
    _clipped_length,
    _point_in_zone,
    _segment_intersects_zone,
    compute_zone_scores,
    export_low_confidence_zones_to_dxf,
    generate_confidence_heatmap,
    generate_pdf_report,
)
from scan2plan.slicing.density_map import DensityMapResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_density_map(
    width_m: float = 5.0,
    height_m: float = 4.0,
    resolution: float = 0.1,
    fill_fraction: float = 0.5,
) -> DensityMapResult:
    """Crée une DensityMapResult synthétique."""
    w = int(width_m / resolution)
    h = int(height_m / resolution)
    image = np.zeros((h, w), dtype=np.uint16)
    # Remplir une fraction des pixels
    n_occupied = int(image.size * fill_fraction)
    if n_occupied > 0:
        flat = image.ravel()
        flat[:n_occupied] = 10
        image = flat.reshape(h, w)
    return DensityMapResult(
        image=image,
        x_min=0.0,
        y_min=0.0,
        resolution=resolution,
        width=w,
        height=h,
    )


def _make_segment(
    x1: float, y1: float, x2: float, y2: float,
    source: str = "high",
    confidence: float = 0.9,
) -> DetectedSegment:
    return DetectedSegment(
        x1=x1, y1=y1, x2=x2, y2=y2,
        source_slice=source,
        confidence=confidence,
    )


def _make_opening(
    wall: DetectedSegment,
    opening_type: str = "door",
) -> Opening:
    return Opening(
        type=opening_type,
        wall_segment=wall,
        position_start=0.1,
        position_end=0.9,
        width=0.8,
        confidence=0.8,
    )


def _make_zone(
    x_min: float = 0.0, y_min: float = 0.0,
    x_max: float = 1.0, y_max: float = 1.0,
    col: int = 0, row: int = 0,
) -> ZoneScore:
    return ZoneScore(
        col=col, row=row,
        x_min=x_min, y_min=y_min,
        x_max=x_max, y_max=y_max,
    )


# ---------------------------------------------------------------------------
# Tests ZoneScore
# ---------------------------------------------------------------------------

class TestZoneScore:
    def test_cx_cy(self) -> None:
        zone = _make_zone(x_min=0.0, y_min=0.0, x_max=2.0, y_max=4.0)
        assert zone.cx == pytest.approx(1.0)
        assert zone.cy == pytest.approx(2.0)

    def test_defaults(self) -> None:
        zone = ZoneScore(col=0, row=0, x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)
        assert zone.density_score == pytest.approx(0.0)
        assert zone.segment_score == pytest.approx(0.0)
        assert zone.topology_score == pytest.approx(0.0)
        assert zone.opening_score == pytest.approx(1.0)
        assert zone.total_score == pytest.approx(0.0)
        assert zone.is_low_confidence is False

    def test_cx_cy_non_unit(self) -> None:
        zone = _make_zone(x_min=1.5, y_min=2.5, x_max=4.5, y_max=6.5)
        assert zone.cx == pytest.approx(3.0)
        assert zone.cy == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# Tests ZoneMap
# ---------------------------------------------------------------------------

class TestZoneMap:
    def _make_zone_map(self, n_rows: int = 2, n_cols: int = 3) -> ZoneMap:
        grid = []
        for r in range(n_rows):
            row = []
            for c in range(n_cols):
                z = ZoneScore(
                    col=c, row=r,
                    x_min=float(c), y_min=float(r),
                    x_max=float(c + 1), y_max=float(r + 1),
                    total_score=float(r * n_cols + c) / (n_rows * n_cols),
                )
                row.append(z)
            grid.append(row)
        return ZoneMap(
            zones=grid,
            n_cols=n_cols,
            n_rows=n_rows,
            cell_size_m=1.0,
            x_min=0.0,
            y_min=0.0,
            global_score=0.5,
        )

    def test_score_matrix_shape(self) -> None:
        zm = self._make_zone_map(n_rows=2, n_cols=3)
        mat = zm.score_matrix()
        assert mat.shape == (2, 3)

    def test_score_matrix_dtype(self) -> None:
        zm = self._make_zone_map()
        assert zm.score_matrix().dtype == np.float32

    def test_score_matrix_values(self) -> None:
        zm = self._make_zone_map(n_rows=1, n_cols=2)
        mat = zm.score_matrix()
        assert mat[0, 0] == pytest.approx(0.0)
        assert mat[0, 1] == pytest.approx(1.0 / 2.0)


# ---------------------------------------------------------------------------
# Tests _point_in_zone
# ---------------------------------------------------------------------------

class TestPointInZone:
    def test_inside(self) -> None:
        zone = _make_zone(0.0, 0.0, 2.0, 2.0)
        assert _point_in_zone(1.0, 1.0, zone) is True

    def test_on_border(self) -> None:
        zone = _make_zone(0.0, 0.0, 2.0, 2.0)
        assert _point_in_zone(0.0, 0.0, zone) is True
        assert _point_in_zone(2.0, 2.0, zone) is True

    def test_outside_x(self) -> None:
        zone = _make_zone(0.0, 0.0, 2.0, 2.0)
        assert _point_in_zone(3.0, 1.0, zone) is False

    def test_outside_y(self) -> None:
        zone = _make_zone(0.0, 0.0, 2.0, 2.0)
        assert _point_in_zone(1.0, 5.0, zone) is False


# ---------------------------------------------------------------------------
# Tests _segment_intersects_zone
# ---------------------------------------------------------------------------

class TestSegmentIntersectsZone:
    def test_fully_inside(self) -> None:
        seg = _make_segment(0.2, 0.2, 0.8, 0.8)
        zone = _make_zone(0.0, 0.0, 1.0, 1.0)
        assert _segment_intersects_zone(seg, zone) is True

    def test_crosses_zone(self) -> None:
        seg = _make_segment(-1.0, 0.5, 2.0, 0.5)
        zone = _make_zone(0.0, 0.0, 1.0, 1.0)
        assert _segment_intersects_zone(seg, zone) is True

    def test_fully_outside(self) -> None:
        seg = _make_segment(5.0, 0.0, 6.0, 0.0)
        zone = _make_zone(0.0, 0.0, 1.0, 1.0)
        assert _segment_intersects_zone(seg, zone) is False

    def test_touches_corner(self) -> None:
        seg = _make_segment(-1.0, 0.0, 0.0, -1.0)
        zone = _make_zone(0.0, 0.0, 1.0, 1.0)
        # AABB test — bounding boxes touch
        assert _segment_intersects_zone(seg, zone) is True

    def test_vertical_segment_inside(self) -> None:
        seg = _make_segment(0.5, 0.0, 0.5, 1.0)
        zone = _make_zone(0.0, 0.0, 1.0, 1.0)
        assert _segment_intersects_zone(seg, zone) is True

    def test_horizontal_segment_outside_y(self) -> None:
        seg = _make_segment(0.0, 5.0, 1.0, 5.0)
        zone = _make_zone(0.0, 0.0, 1.0, 1.0)
        assert _segment_intersects_zone(seg, zone) is False


# ---------------------------------------------------------------------------
# Tests _clipped_length
# ---------------------------------------------------------------------------

class TestClippedLength:
    def test_segment_fully_inside(self) -> None:
        seg = _make_segment(0.2, 0.5, 0.8, 0.5)
        zone = _make_zone(0.0, 0.0, 1.0, 1.0)
        assert _clipped_length(seg, zone) == pytest.approx(0.6, abs=1e-6)

    def test_segment_fully_outside(self) -> None:
        seg = _make_segment(5.0, 5.0, 6.0, 5.0)
        zone = _make_zone(0.0, 0.0, 1.0, 1.0)
        assert _clipped_length(seg, zone) == pytest.approx(0.0, abs=1e-6)

    def test_segment_half_inside(self) -> None:
        # Segment horizontal de 0 à 2, zone de 0 à 1 → clipped = 1.0
        seg = _make_segment(0.0, 0.5, 2.0, 0.5)
        zone = _make_zone(0.0, 0.0, 1.0, 1.0)
        assert _clipped_length(seg, zone) == pytest.approx(1.0, abs=1e-6)

    def test_zero_length_segment(self) -> None:
        seg = _make_segment(0.5, 0.5, 0.5, 0.5)
        zone = _make_zone(0.0, 0.0, 1.0, 1.0)
        assert _clipped_length(seg, zone) == pytest.approx(0.0, abs=1e-6)

    def test_diagonal_segment_crossing(self) -> None:
        # Segment de (-1, 0) à (1, 0) — horizontal traversant zone [0,0]-[1,1]
        seg = _make_segment(-1.0, 0.5, 1.0, 0.5)
        zone = _make_zone(0.0, 0.0, 1.0, 1.0)
        assert _clipped_length(seg, zone) == pytest.approx(1.0, abs=1e-6)

    def test_vertical_segment_fully_inside(self) -> None:
        seg = _make_segment(0.5, 0.1, 0.5, 0.9)
        zone = _make_zone(0.0, 0.0, 1.0, 1.0)
        assert _clipped_length(seg, zone) == pytest.approx(0.8, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests compute_zone_scores
# ---------------------------------------------------------------------------

class TestComputeZoneScores:
    def test_returns_zone_map(self) -> None:
        dm = _make_density_map()
        result = compute_zone_scores(dm, [], [])
        assert isinstance(result, ZoneMap)

    def test_grid_dimensions(self) -> None:
        dm = _make_density_map(width_m=4.0, height_m=3.0, resolution=0.1)
        result = compute_zone_scores(dm, [], [], cell_size_m=1.0)
        assert result.n_cols == 4
        assert result.n_rows == 3
        assert len(result.zones) == 3
        assert len(result.zones[0]) == 4

    def test_empty_image_no_segments(self) -> None:
        dm = _make_density_map(fill_fraction=0.0)
        result = compute_zone_scores(dm, [], [])
        # All cells are empty → low scores
        for row in result.zones:
            for zone in row:
                assert zone.density_score == pytest.approx(0.0)

    def test_full_image_high_density_score(self) -> None:
        dm = _make_density_map(fill_fraction=1.0)
        result = compute_zone_scores(dm, [], [])
        for row in result.zones:
            for zone in row:
                assert zone.density_score == pytest.approx(1.0, abs=0.05)

    def test_global_score_in_range(self) -> None:
        dm = _make_density_map(fill_fraction=0.5)
        segs = [_make_segment(0.5, 0.5, 2.0, 0.5)]
        result = compute_zone_scores(dm, segs, [])
        assert 0.0 <= result.global_score <= 1.0

    def test_low_confidence_zones_populated(self) -> None:
        # Image vide → tout à faible confiance
        dm = _make_density_map(fill_fraction=0.0)
        result = compute_zone_scores(dm, [], [], low_confidence_threshold=0.5)
        assert len(result.low_confidence_zones) > 0

    def test_no_low_confidence_when_full(self) -> None:
        dm = _make_density_map(fill_fraction=1.0)
        segs = [_make_segment(0.0, 2.0, 5.0, 2.0)]
        result = compute_zone_scores(dm, segs, [], low_confidence_threshold=0.01)
        # Nearly all should be above 0.01
        assert result.global_score > 0.01

    def test_segment_score_nonzero_with_segment(self) -> None:
        dm = _make_density_map()
        segs = [_make_segment(0.0, 2.0, 5.0, 2.0)]
        result = compute_zone_scores(dm, segs, [], cell_size_m=1.0)
        # At least one cell should have segment_score > 0
        has_seg_score = any(
            zone.segment_score > 0
            for row in result.zones
            for zone in row
        )
        assert has_seg_score

    def test_topology_score_all_segments_connected(self) -> None:
        dm = _make_density_map(width_m=2.0, height_m=2.0)
        # Segment starts and ends inside the single cell grid
        segs = [_make_segment(0.1, 0.5, 0.9, 0.5)]
        result = compute_zone_scores(dm, segs, [], cell_size_m=3.0)
        # 1 cell, both endpoints inside → topology_score = 1.0
        assert result.zones[0][0].topology_score == pytest.approx(1.0)

    def test_opening_score_no_openings(self) -> None:
        dm = _make_density_map()
        segs = [_make_segment(0.0, 0.5, 1.0, 0.5)]
        result = compute_zone_scores(dm, segs, [], cell_size_m=2.0)
        # No openings → opening_score == 1.0
        assert result.zones[0][0].opening_score == pytest.approx(1.0)

    def test_opening_score_penalized_many_openings(self) -> None:
        """Beaucoup d'ouvertures sur un petit mur → ouverture_score < 1."""
        dm = _make_density_map(width_m=2.0, height_m=2.0)
        wall = _make_segment(0.1, 0.5, 0.3, 0.5)  # mur court ~0.2m
        # 5 ouvertures sur 0.2 m → ratio = 25/m → pénalité
        openings = [_make_opening(wall) for _ in range(5)]
        result = compute_zone_scores(dm, [wall], openings, cell_size_m=3.0)
        assert result.zones[0][0].opening_score < 1.0

    def test_total_score_is_weighted_sum(self) -> None:
        dm = _make_density_map(fill_fraction=1.0)
        segs = [_make_segment(0.0, 0.5, 1.0, 0.5)]
        result = compute_zone_scores(dm, segs, [], cell_size_m=5.0)
        zone = result.zones[0][0]
        expected = (
            0.35 * zone.density_score
            + 0.30 * zone.segment_score
            + 0.20 * zone.topology_score
            + 0.15 * zone.opening_score
        )
        assert zone.total_score == pytest.approx(expected, abs=1e-6)

    def test_single_cell_grid(self) -> None:
        """Grille 1×1 : toute l'image dans une cellule."""
        dm = _make_density_map(width_m=2.0, height_m=2.0, fill_fraction=0.8)
        result = compute_zone_scores(dm, [], [], cell_size_m=10.0)
        assert result.n_rows == 1
        assert result.n_cols == 1

    def test_zones_cover_full_plan(self) -> None:
        dm = _make_density_map(width_m=3.0, height_m=2.0, resolution=0.1)
        result = compute_zone_scores(dm, [], [], cell_size_m=1.0)
        # Les cellules doivent couvrir l'emprise complète
        x_covered = sum(
            result.zones[0][c].x_max - result.zones[0][c].x_min
            for c in range(result.n_cols)
        )
        assert x_covered == pytest.approx(3.0, abs=0.1)


# ---------------------------------------------------------------------------
# Tests generate_confidence_heatmap
# ---------------------------------------------------------------------------

class TestGenerateConfidenceHeatmap:
    def _zone_map_simple(self) -> ZoneMap:
        grid = [[ZoneScore(
            col=0, row=0, x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0,
            total_score=0.75,
        )]]
        return ZoneMap(
            zones=grid, n_cols=1, n_rows=1,
            cell_size_m=1.0, x_min=0.0, y_min=0.0,
            global_score=0.75,
        )

    def test_returns_ndarray(self) -> None:
        zm = self._zone_map_simple()
        result = generate_confidence_heatmap(zm)
        assert isinstance(result, np.ndarray)

    def test_matrix_shape(self) -> None:
        dm = _make_density_map(width_m=3.0, height_m=2.0)
        segs: list = []
        zone_map = compute_zone_scores(dm, segs, [], cell_size_m=1.0)
        mat = generate_confidence_heatmap(zone_map)
        assert mat.shape == (zone_map.n_rows, zone_map.n_cols)

    def test_matrix_dtype_float32(self) -> None:
        zm = self._zone_map_simple()
        mat = generate_confidence_heatmap(zm)
        assert mat.dtype == np.float32

    def test_values_match_score_matrix(self) -> None:
        zm = self._zone_map_simple()
        mat = generate_confidence_heatmap(zm)
        expected = zm.score_matrix()
        np.testing.assert_array_almost_equal(mat, expected)

    def test_saves_png_when_path_given(self) -> None:
        zm = self._zone_map_simple()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "heatmap.png"
            generate_confidence_heatmap(zm, output_path=out)
            assert out.exists()
            assert out.stat().st_size > 0

    def test_no_output_without_path(self) -> None:
        zm = self._zone_map_simple()
        # Ne doit pas lever d'exception
        mat = generate_confidence_heatmap(zm, output_path=None)
        assert mat is not None


# ---------------------------------------------------------------------------
# Tests export_low_confidence_zones_to_dxf
# ---------------------------------------------------------------------------

class TestExportLowConfidenceZonesToDxf:
    def _make_doc(self):
        import ezdxf
        return ezdxf.new("R2013")

    def test_no_zones_returns_zero(self) -> None:
        doc = self._make_doc()
        zm = ZoneMap(
            zones=[], n_cols=0, n_rows=0,
            cell_size_m=1.0, x_min=0.0, y_min=0.0,
            low_confidence_zones=[],
            global_score=1.0,
        )
        n = export_low_confidence_zones_to_dxf(zm, doc)
        assert n == 0

    def test_one_zone_returns_four_lines(self) -> None:
        doc = self._make_doc()
        zone = ZoneScore(
            col=0, row=0, x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0,
            total_score=0.1, is_low_confidence=True,
        )
        zm = ZoneMap(
            zones=[[zone]], n_cols=1, n_rows=1,
            cell_size_m=1.0, x_min=0.0, y_min=0.0,
            low_confidence_zones=[zone],
            global_score=0.1,
        )
        n = export_low_confidence_zones_to_dxf(zm, doc)
        assert n == 4

    def test_two_zones_returns_eight_lines(self) -> None:
        doc = self._make_doc()
        z1 = ZoneScore(col=0, row=0, x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0,
                       total_score=0.1, is_low_confidence=True)
        z2 = ZoneScore(col=1, row=0, x_min=1.0, y_min=0.0, x_max=2.0, y_max=1.0,
                       total_score=0.1, is_low_confidence=True)
        zm = ZoneMap(
            zones=[[z1, z2]], n_cols=2, n_rows=1,
            cell_size_m=1.0, x_min=0.0, y_min=0.0,
            low_confidence_zones=[z1, z2],
            global_score=0.1,
        )
        n = export_low_confidence_zones_to_dxf(zm, doc)
        assert n == 8

    def test_layer_created(self) -> None:
        doc = self._make_doc()
        zone = ZoneScore(col=0, row=0, x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0,
                         total_score=0.1, is_low_confidence=True)
        zm = ZoneMap(
            zones=[[zone]], n_cols=1, n_rows=1,
            cell_size_m=1.0, x_min=0.0, y_min=0.0,
            low_confidence_zones=[zone],
            global_score=0.1,
        )
        export_low_confidence_zones_to_dxf(zm, doc, layer_name="INCERTAIN")
        assert "INCERTAIN" in doc.layers

    def test_custom_layer_name(self) -> None:
        doc = self._make_doc()
        zone = ZoneScore(col=0, row=0, x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0,
                         total_score=0.1, is_low_confidence=True)
        zm = ZoneMap(
            zones=[[zone]], n_cols=1, n_rows=1,
            cell_size_m=1.0, x_min=0.0, y_min=0.0,
            low_confidence_zones=[zone],
            global_score=0.1,
        )
        export_low_confidence_zones_to_dxf(zm, doc, layer_name="ZONE_FAIBLE")
        assert "ZONE_FAIBLE" in doc.layers

    def test_lines_on_correct_layer(self) -> None:
        doc = self._make_doc()
        zone = ZoneScore(col=0, row=0, x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0,
                         total_score=0.1, is_low_confidence=True)
        zm = ZoneMap(
            zones=[[zone]], n_cols=1, n_rows=1,
            cell_size_m=1.0, x_min=0.0, y_min=0.0,
            low_confidence_zones=[zone],
            global_score=0.1,
        )
        export_low_confidence_zones_to_dxf(zm, doc, layer_name="INCERTAIN")
        msp = doc.modelspace()
        lines = [e for e in msp if e.dxftype() == "LINE" and e.dxf.layer == "INCERTAIN"]
        assert len(lines) == 4

    def test_layer_reused_if_exists(self) -> None:
        import ezdxf
        doc = ezdxf.new("R2013")
        doc.layers.new("INCERTAIN", dxfattribs={"color": 7})
        zone = ZoneScore(col=0, row=0, x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0,
                         total_score=0.1, is_low_confidence=True)
        zm = ZoneMap(
            zones=[[zone]], n_cols=1, n_rows=1,
            cell_size_m=1.0, x_min=0.0, y_min=0.0,
            low_confidence_zones=[zone],
            global_score=0.1,
        )
        # Should not raise even if layer already exists
        n = export_low_confidence_zones_to_dxf(zm, doc, layer_name="INCERTAIN")
        assert n == 4


# ---------------------------------------------------------------------------
# Tests generate_pdf_report
# ---------------------------------------------------------------------------

class TestGeneratePdfReport:
    def _make_zone_map(self) -> ZoneMap:
        dm = _make_density_map()
        return compute_zone_scores(dm, [], [])

    def test_creates_pdf_file(self) -> None:
        zm = self._make_zone_map()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.pdf"
            result = generate_pdf_report(zm, [], [], out, title="Test Rapport")
            assert result.exists()
            assert result.suffix == ".pdf"

    def test_extension_forced_to_pdf(self) -> None:
        zm = self._make_zone_map()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.txt"
            result = generate_pdf_report(zm, [], [], out, title="Test")
            assert result.suffix == ".pdf"

    def test_pdf_not_empty(self) -> None:
        zm = self._make_zone_map()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.pdf"
            result = generate_pdf_report(zm, [], [], out)
            assert result.stat().st_size > 1000  # PDF non trivial

    def test_with_segments_and_openings(self) -> None:
        dm = _make_density_map()
        wall = _make_segment(0.0, 1.0, 3.0, 1.0)
        op = _make_opening(wall)
        zm = compute_zone_scores(dm, [wall], [op])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.pdf"
            result = generate_pdf_report(zm, [wall], [op], out, title="Test complet")
            assert result.exists()

    def test_parent_dir_created(self) -> None:
        """generate_pdf_report crée le répertoire parent si nécessaire."""
        zm = self._make_zone_map()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "subdir" / "nested" / "report.pdf"
            result = generate_pdf_report(zm, [], [], out)
            assert result.exists()

    def test_returns_path_object(self) -> None:
        zm = self._make_zone_map()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.pdf"
            result = generate_pdf_report(zm, [], [], out)
            assert isinstance(result, Path)
