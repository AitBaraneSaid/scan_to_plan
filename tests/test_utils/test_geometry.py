"""Tests unitaires pour utils/geometry.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scan2plan.utils.geometry import (
    angle_between_segments,
    line_intersection,
    perpendicular_distance_point_to_line,
    perpendicular_distance_segment_to_segment,
    segment_angle,
    segment_length,
    segments_overlap_or_gap,
)


class TestSegmentLength:
    def test_horizontal(self) -> None:
        assert segment_length((0.0, 0.0, 3.0, 0.0)) == pytest.approx(3.0)

    def test_vertical(self) -> None:
        assert segment_length((0.0, 0.0, 0.0, 4.0)) == pytest.approx(4.0)

    def test_diagonal(self) -> None:
        assert segment_length((0.0, 0.0, 3.0, 4.0)) == pytest.approx(5.0)

    def test_zero_length(self) -> None:
        assert segment_length((1.0, 2.0, 1.0, 2.0)) == pytest.approx(0.0)

    def test_negative_coords(self) -> None:
        assert segment_length((-1.0, -1.0, 2.0, 3.0)) == pytest.approx(5.0)


class TestSegmentAngle:
    def test_horizontal_right(self) -> None:
        assert segment_angle((0.0, 0.0, 1.0, 0.0)) == pytest.approx(0.0)

    def test_horizontal_left(self) -> None:
        """Direction opposée → même angle (droite non orientée)."""
        assert segment_angle((1.0, 0.0, 0.0, 0.0)) == pytest.approx(0.0)

    def test_vertical(self) -> None:
        assert segment_angle((0.0, 0.0, 0.0, 1.0)) == pytest.approx(math.pi / 2)

    def test_45_degrees(self) -> None:
        assert segment_angle((0.0, 0.0, 1.0, 1.0)) == pytest.approx(math.pi / 4)

    def test_degenerate_zero_length(self) -> None:
        """Segment dégénéré → 0.0 par convention."""
        assert segment_angle((1.0, 1.0, 1.0, 1.0)) == pytest.approx(0.0)

    def test_result_in_range(self) -> None:
        """L'angle doit toujours être dans [0, π)."""
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (1, 1), (-1, 1)]:
            a = segment_angle((0.0, 0.0, float(dx), float(dy)))
            assert 0.0 <= a < math.pi


class TestAngleBetweenSegments:
    def test_parallel_zero(self) -> None:
        """Deux segments parallèles → angle = 0."""
        seg1 = (0.0, 0.0, 1.0, 0.0)
        seg2 = (0.0, 1.0, 1.0, 1.0)
        assert angle_between_segments(seg1, seg2) == pytest.approx(0.0, abs=1e-9)

    def test_perpendicular(self) -> None:
        """Deux segments perpendiculaires → angle = π/2."""
        seg1 = (0.0, 0.0, 1.0, 0.0)
        seg2 = (0.0, 0.0, 0.0, 1.0)
        assert angle_between_segments(seg1, seg2) == pytest.approx(math.pi / 2)

    def test_45_degrees(self) -> None:
        seg1 = (0.0, 0.0, 1.0, 0.0)
        seg2 = (0.0, 0.0, 1.0, 1.0)
        assert angle_between_segments(seg1, seg2) == pytest.approx(math.pi / 4)

    def test_opposite_directions_same_angle(self) -> None:
        """Directions opposées → angle 0 (droites identiques)."""
        seg1 = (0.0, 0.0, 1.0, 0.0)
        seg2 = (1.0, 0.0, 0.0, 0.0)
        assert angle_between_segments(seg1, seg2) == pytest.approx(0.0, abs=1e-9)

    def test_result_in_range(self) -> None:
        """L'angle doit toujours être dans [0, π/2]."""
        seg1 = (0.0, 0.0, 1.0, 0.0)
        seg2 = (0.0, 0.0, -1.0, 1.0)
        a = angle_between_segments(seg1, seg2)
        assert 0.0 <= a <= math.pi / 2


class TestPerpendicularDistancePointToLine:
    def test_point_on_line(self) -> None:
        """Point sur la droite → distance 0."""
        assert perpendicular_distance_point_to_line((0.5, 0.0), (0.0, 0.0, 1.0, 0.0)) == pytest.approx(0.0)

    def test_point_above_horizontal(self) -> None:
        assert perpendicular_distance_point_to_line((0.5, 0.1), (0.0, 0.0, 1.0, 0.0)) == pytest.approx(0.1)

    def test_point_beside_vertical(self) -> None:
        assert perpendicular_distance_point_to_line((0.05, 0.5), (0.0, 0.0, 0.0, 1.0)) == pytest.approx(0.05)

    def test_degenerate_segment(self) -> None:
        """Segment dégénéré → distance euclidienne au point de départ."""
        d = perpendicular_distance_point_to_line((3.0, 4.0), (0.0, 0.0, 0.0, 0.0))
        assert d == pytest.approx(5.0)


class TestPerpendicularDistanceSegmentToSegment:
    def test_parallel_offset(self) -> None:
        """Deux segments horizontaux décalés de 0.05 m en Y."""
        seg1 = (0.0, 0.0, 1.0, 0.0)
        seg2 = (0.0, 0.05, 1.0, 0.05)
        assert perpendicular_distance_segment_to_segment(seg1, seg2) == pytest.approx(0.05)

    def test_collinear_zero(self) -> None:
        """Segments colinéaires → distance ≈ 0."""
        seg1 = (0.0, 0.0, 1.0, 0.0)
        seg2 = (1.5, 0.0, 2.5, 0.0)
        assert perpendicular_distance_segment_to_segment(seg1, seg2) == pytest.approx(0.0, abs=1e-9)


class TestSegmentsOverlapOrGap:
    def test_gap_positive(self) -> None:
        """Segments non chevauchants → gap positif."""
        seg1 = (0.0, 0.0, 1.0, 0.0)
        seg2 = (1.1, 0.0, 2.0, 0.0)
        assert segments_overlap_or_gap(seg1, seg2) == pytest.approx(0.1, abs=0.01)

    def test_overlap_negative(self) -> None:
        """Segments chevauchants → valeur négative."""
        seg1 = (0.0, 0.0, 1.5, 0.0)
        seg2 = (1.0, 0.0, 2.5, 0.0)
        result = segments_overlap_or_gap(seg1, seg2)
        assert result < 0

    def test_touching_zero(self) -> None:
        """Segments qui se touchent exactement → 0."""
        seg1 = (0.0, 0.0, 1.0, 0.0)
        seg2 = (1.0, 0.0, 2.0, 0.0)
        assert segments_overlap_or_gap(seg1, seg2) == pytest.approx(0.0, abs=1e-9)

    def test_degenerate_seg1(self) -> None:
        result = segments_overlap_or_gap((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 2.0, 0.0))
        assert result == pytest.approx(0.0)


class TestLineIntersection:
    def test_perpendicular_cross(self) -> None:
        """Deux segments perpendiculaires qui se croisent en (0.5, 0)."""
        seg1 = (0.0, 0.0, 1.0, 0.0)
        seg2 = (0.5, -1.0, 0.5, 1.0)
        pt = line_intersection(seg1, seg2)
        assert pt is not None
        assert pt[0] == pytest.approx(0.5)
        assert pt[1] == pytest.approx(0.0)

    def test_parallel_no_intersection(self) -> None:
        """Deux segments parallèles → None."""
        seg1 = (0.0, 0.0, 1.0, 0.0)
        seg2 = (0.0, 1.0, 1.0, 1.0)
        assert line_intersection(seg1, seg2) is None

    def test_collinear_no_intersection(self) -> None:
        """Segments colinéaires → parallèles → None."""
        seg1 = (0.0, 0.0, 1.0, 0.0)
        seg2 = (2.0, 0.0, 3.0, 0.0)
        assert line_intersection(seg1, seg2) is None

    def test_known_intersection(self) -> None:
        """X de (0,0)→(2,2) et (0,2)→(2,0) se croisent en (1,1)."""
        seg1 = (0.0, 0.0, 2.0, 2.0)
        seg2 = (0.0, 2.0, 2.0, 0.0)
        pt = line_intersection(seg1, seg2)
        assert pt is not None
        assert pt[0] == pytest.approx(1.0)
        assert pt[1] == pytest.approx(1.0)

    def test_returns_tuple_of_floats(self) -> None:
        pt = line_intersection((0.0, 0.0, 1.0, 0.0), (0.5, -1.0, 0.5, 1.0))
        assert pt is not None
        assert isinstance(pt[0], float)
        assert isinstance(pt[1], float)
