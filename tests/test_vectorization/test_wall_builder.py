"""Tests unitaires pour vectorization/wall_builder.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scan2plan.detection.line_detection import DetectedSegment
from scan2plan.slicing.density_map import DensityMapResult
from scan2plan.vectorization.wall_builder import (
    build_double_line_walls,
    estimate_wall_thickness,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(
    x1: float, y1: float, x2: float, y2: float,
    source: str = "high", confidence: float = 1.0,
) -> DetectedSegment:
    return DetectedSegment(x1=x1, y1=y1, x2=x2, y2=y2,
                           source_slice=source, confidence=confidence)


def _make_wall_dmap(
    wall_y_m: float,
    half_thickness_m: float,
    wall_x1_m: float = 0.0,
    wall_x2_m: float = 4.0,
    res: float = 0.005,
) -> tuple[DensityMapResult, np.ndarray]:
    """Crée une DensityMapResult + binary_image avec un mur horizontal synthétique.

    Le mur est centré sur wall_y_m, d'épaisseur 2×half_thickness_m,
    s'étendant de wall_x1_m à wall_x2_m.

    Args:
        wall_y_m: Coordonnée Y du centre du mur (mètres).
        half_thickness_m: Demi-épaisseur du mur (mètres).
        wall_x1_m: Début du mur en X (mètres).
        wall_x2_m: Fin du mur en X (mètres).
        res: Résolution en mètres/pixel.

    Returns:
        (DensityMapResult, binary_image).
    """
    margin_m = 0.5
    x_min = wall_x1_m - margin_m
    y_min = wall_y_m - half_thickness_m - margin_m
    x_max = wall_x2_m + margin_m
    y_max = wall_y_m + half_thickness_m + margin_m

    width_px = int((x_max - x_min) / res) + 1
    height_px = int((y_max - y_min) / res) + 1

    # Image de densité (float) : on met de la densité dans la zone mur
    image = np.zeros((height_px, width_px), dtype=np.float32)

    # Convertir les bornes métriques du mur en pixels
    col1 = int((wall_x1_m - x_min) / res)
    col2 = int((wall_x2_m - x_min) / res)
    # Rangées : row 0 = Y max → Y_mur_max est en haut dans l'image
    row_top = int(height_px - 1 - (wall_y_m + half_thickness_m - y_min) / res)
    row_bot = int(height_px - 1 - (wall_y_m - half_thickness_m - y_min) / res)

    row_top = max(0, row_top)
    row_bot = min(height_px - 1, row_bot)
    col1 = max(0, col1)
    col2 = min(width_px - 1, col2)

    image[row_top:row_bot + 1, col1:col2 + 1] = 10.0

    binary = (image > 0).astype(np.uint8) * 255

    dmap = DensityMapResult(
        image=image,
        x_min=x_min,
        y_min=y_min,
        resolution=res,
        width=width_px,
        height=height_px,
    )
    return dmap, binary


# ---------------------------------------------------------------------------
# Tests estimate_wall_thickness
# ---------------------------------------------------------------------------

class TestEstimateWallThickness:
    def test_20cm_wall_estimated_correctly(self) -> None:
        """Mur horizontal de 20 cm → épaisseur estimée ≈ 20 cm ± 2 cm."""
        seg = _seg(0.5, 1.0, 3.5, 1.0)  # axe central à y=1.0, horizontal
        dmap, binary = _make_wall_dmap(
            wall_y_m=1.0,
            half_thickness_m=0.10,  # 20 cm total
            wall_x1_m=0.5,
            wall_x2_m=3.5,
        )
        thickness = estimate_wall_thickness(seg, dmap, binary)
        assert 0.16 <= thickness <= 0.24, (
            f"Épaisseur attendue ≈ 0.20 m, obtenue : {thickness:.3f} m"
        )

    def test_10cm_wall_estimated(self) -> None:
        """Mur de 10 cm → épaisseur estimée ≈ 10 cm ± 2 cm."""
        seg = _seg(0.5, 1.0, 3.5, 1.0)
        dmap, binary = _make_wall_dmap(
            wall_y_m=1.0,
            half_thickness_m=0.05,  # 10 cm total
        )
        thickness = estimate_wall_thickness(seg, dmap, binary)
        assert 0.06 <= thickness <= 0.14, (
            f"Épaisseur attendue ≈ 0.10 m, obtenue : {thickness:.3f} m"
        )

    def test_30cm_wall_estimated(self) -> None:
        """Mur de 30 cm → épaisseur estimée ≈ 30 cm ± 3 cm."""
        seg = _seg(0.5, 1.0, 3.5, 1.0)
        dmap, binary = _make_wall_dmap(
            wall_y_m=1.0,
            half_thickness_m=0.15,  # 30 cm total
        )
        thickness = estimate_wall_thickness(seg, dmap, binary)
        assert 0.24 <= thickness <= 0.36, (
            f"Épaisseur attendue ≈ 0.30 m, obtenue : {thickness:.3f} m"
        )

    def test_degenerate_segment_returns_zero(self) -> None:
        """Segment de longueur nulle → retourne 0.0 (pas de mesure)."""
        seg = _seg(1.0, 1.0, 1.0, 1.0)
        dmap, binary = _make_wall_dmap(1.0, 0.10)
        result = estimate_wall_thickness(seg, dmap, binary)
        assert result == 0.0

    def test_empty_image_returns_zero(self) -> None:
        """Image noire (pas de mur visible) → retourne 0.0."""
        seg = _seg(0.5, 1.0, 3.5, 1.0)
        dmap, binary = _make_wall_dmap(1.0, 0.10)
        binary_empty = np.zeros_like(binary)
        result = estimate_wall_thickness(seg, dmap, binary_empty)
        assert result == 0.0

    def test_returns_float(self) -> None:
        seg = _seg(0.5, 1.0, 3.5, 1.0)
        dmap, binary = _make_wall_dmap(1.0, 0.10)
        result = estimate_wall_thickness(seg, dmap, binary)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Tests build_double_line_walls — segments simples
# ---------------------------------------------------------------------------

class TestBuildDoubleLineWalls:
    def test_single_horizontal_segment_two_parallel_lines(self) -> None:
        """Segment horizontal → 2 lignes parallèles écartées de l'épaisseur."""
        seg = _seg(0.0, 0.0, 4.0, 0.0)
        pairs = build_double_line_walls([seg], [0.20], default_thickness=0.15)
        assert len(pairs) == 1
        line_pos, line_neg = pairs[0]
        # Les deux lignes doivent être horizontales (même y pour x1 et x2)
        assert abs(line_pos.y1 - line_pos.y2) < 1e-6
        assert abs(line_neg.y1 - line_neg.y2) < 1e-6

    def test_spacing_equals_thickness(self) -> None:
        """L'écartement entre les deux lignes doit être égal à l'épaisseur."""
        seg = _seg(0.0, 0.0, 4.0, 0.0)
        thickness = 0.20
        pairs = build_double_line_walls([seg], [thickness])
        line_pos, line_neg = pairs[0]
        # Distance perpendiculaire entre les deux lignes
        dy = abs(line_pos.y1 - line_neg.y1)
        assert dy == pytest.approx(thickness, abs=1e-6)

    def test_default_thickness_used_when_zero(self) -> None:
        """Épaisseur ≤ 0 → utilise default_thickness."""
        seg = _seg(0.0, 0.0, 4.0, 0.0)
        default = 0.15
        pairs = build_double_line_walls([seg], [0.0], default_thickness=default)
        line_pos, line_neg = pairs[0]
        dy = abs(line_pos.y1 - line_neg.y1)
        assert dy == pytest.approx(default, abs=1e-6)

    def test_length_preserved(self) -> None:
        """Les deux lignes parallèles ont la même longueur que le segment original."""
        seg = _seg(1.0, 2.0, 5.0, 2.0)
        pairs = build_double_line_walls([seg], [0.20])
        line_pos, line_neg = pairs[0]
        assert line_pos.length == pytest.approx(seg.length, abs=1e-6)
        assert line_neg.length == pytest.approx(seg.length, abs=1e-6)

    def test_vertical_segment_offsets_in_x(self) -> None:
        """Segment vertical → les lignes parallèles sont décalées en X."""
        seg = _seg(2.0, 0.0, 2.0, 3.0)
        pairs = build_double_line_walls([seg], [0.20])
        line_pos, line_neg = pairs[0]
        dx = abs(line_pos.x1 - line_neg.x1)
        assert dx == pytest.approx(0.20, abs=1e-6)

    def test_source_and_confidence_preserved(self) -> None:
        """Les métadonnées source_slice et confidence sont préservées."""
        seg = DetectedSegment(x1=0, y1=0, x2=4, y2=0,
                              source_slice="mid", confidence=0.7)
        pairs = build_double_line_walls([seg], [0.20])
        line_pos, line_neg = pairs[0]
        assert line_pos.source_slice == "mid"
        assert line_pos.confidence == pytest.approx(0.7)
        assert line_neg.source_slice == "mid"

    def test_multiple_segments(self) -> None:
        """3 segments → 3 paires."""
        segs = [
            _seg(0.0, 0.0, 4.0, 0.0),
            _seg(4.0, 0.0, 4.0, 3.0),
            _seg(4.0, 3.0, 0.0, 3.0),
        ]
        pairs = build_double_line_walls(segs, [0.20, 0.15, 0.20])
        assert len(pairs) == 3

    def test_mismatched_lengths_raises(self) -> None:
        """Listes de longueurs différentes → ValueError."""
        segs = [_seg(0, 0, 4, 0), _seg(0, 1, 4, 1)]
        with pytest.raises(ValueError, match="même longueur"):
            build_double_line_walls(segs, [0.20])

    def test_symmetry_around_axis(self) -> None:
        """Les deux lignes sont symétriques par rapport à l'axe original."""
        seg = _seg(0.0, 1.0, 4.0, 1.0)
        pairs = build_double_line_walls([seg], [0.20])
        line_pos, line_neg = pairs[0]
        mid_y = (line_pos.y1 + line_neg.y1) / 2.0
        assert mid_y == pytest.approx(seg.y1, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests build_double_line_walls — intersections en L
# ---------------------------------------------------------------------------

class TestLIntersection:
    def test_l_corner_lines_do_not_overshoot(self) -> None:
        """Deux murs en L à angle droit : les 4 lignes ne dépassent pas le coin."""
        # Mur horizontal : (0,0) → (4,0)
        # Mur vertical   : (4,-1) → (4,3)
        # L'angle est exactement 90°
        seg_h = _seg(0.0, 0.0, 4.0, 0.0)
        seg_v = _seg(4.0, -1.0, 4.0, 3.0)
        thickness = 0.20  # 20 cm chacun

        pairs = build_double_line_walls([seg_h, seg_v], [thickness, thickness])
        (h_pos, h_neg), (v_pos, v_neg) = pairs

        # Les lignes horizontales doivent se terminer autour de x=4 (± épaisseur)
        h_pos_xmax = max(h_pos.x1, h_pos.x2)
        h_neg_xmax = max(h_neg.x1, h_neg.x2)
        assert h_pos_xmax <= 4.0 + thickness + 0.05
        assert h_neg_xmax <= 4.0 + thickness + 0.05

    def test_l_corner_returns_four_segments(self) -> None:
        """Coin en L → toujours 4 segments (2 paires)."""
        seg_h = _seg(0.0, 0.0, 4.0, 0.0)
        seg_v = _seg(4.0, 0.0, 4.0, 3.0)
        pairs = build_double_line_walls([seg_h, seg_v], [0.20, 0.20])
        assert len(pairs) == 2
        assert all(len(p) == 2 for p in pairs)

    def test_l_corner_90deg_lines_are_perpendicular(self) -> None:
        """Les lignes d'un mur horizontal restent horizontales après résolution du coin."""
        seg_h = _seg(0.0, 0.0, 4.0, 0.0)
        seg_v = _seg(4.0, 0.0, 4.0, 3.0)
        pairs = build_double_line_walls([seg_h, seg_v], [0.20, 0.20])
        (h_pos, h_neg), _ = pairs
        # Lignes horizontales : delta Y ≈ 0
        assert abs(h_pos.y2 - h_pos.y1) < 0.01
        assert abs(h_neg.y2 - h_neg.y1) < 0.01

    def test_parallel_segments_not_modified(self) -> None:
        """Deux murs parallèles (même direction) : aucune coupure de coin."""
        seg1 = _seg(0.0, 0.0, 4.0, 0.0)
        seg2 = _seg(0.0, 3.0, 4.0, 3.0)
        pairs_before = build_double_line_walls([seg1], [0.20])
        pairs_after = build_double_line_walls([seg1, seg2], [0.20, 0.20])
        # Les lignes du premier mur ne doivent pas être modifiées (pas de coin)
        (b_pos, b_neg) = pairs_before[0]
        (a_pos, a_neg) = pairs_after[0]
        assert a_pos.x1 == pytest.approx(b_pos.x1, abs=0.01)
        assert a_pos.x2 == pytest.approx(b_pos.x2, abs=0.01)

    def test_four_wall_rectangle_double_line(self) -> None:
        """Rectangle de 4 murs en double ligne : 4 paires produites."""
        segs = [
            _seg(0.0, 0.0, 4.0, 0.0),
            _seg(4.0, 0.0, 4.0, 3.0),
            _seg(4.0, 3.0, 0.0, 3.0),
            _seg(0.0, 3.0, 0.0, 0.0),
        ]
        thicknesses = [0.20] * 4
        pairs = build_double_line_walls(segs, thicknesses)
        assert len(pairs) == 4
