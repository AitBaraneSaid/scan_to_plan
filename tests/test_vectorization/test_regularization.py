"""Tests unitaires pour detection/orientation.py et vectorization/regularization.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scan2plan.detection.line_detection import DetectedSegment
from scan2plan.detection.orientation import detect_dominant_orientations
from scan2plan.vectorization.regularization import (
    align_parallel_segments,
    regularize_segments,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    source: str = "high",
    confidence: float = 1.0,
) -> DetectedSegment:
    return DetectedSegment(x1=x1, y1=y1, x2=x2, y2=y2,
                           source_slice=source, confidence=confidence)


def _seg_from_angle_len(
    angle_deg: float,
    length: float = 1.0,
    cx: float = 0.0,
    cy: float = 0.0,
) -> DetectedSegment:
    """Crée un segment centré en (cx, cy), orienté à angle_deg, de longueur length."""
    a = math.radians(angle_deg)
    half = length / 2.0
    return _seg(
        cx - math.cos(a) * half,
        cy - math.sin(a) * half,
        cx + math.cos(a) * half,
        cy + math.sin(a) * half,
    )


# ---------------------------------------------------------------------------
# Tests detect_dominant_orientations
# ---------------------------------------------------------------------------

class TestDetectDominantOrientations:
    def test_two_orthogonal_directions(self) -> None:
        """4 segments ≈0° et ≈90° (bruit ±2°) → 2 pics à ~0° et ~90°."""
        segs = [
            _seg_from_angle_len(2.0, 3.0),    # ≈ 0°
            _seg_from_angle_len(-1.5, 2.5),   # ≈ 0°
            _seg_from_angle_len(91.0, 3.0),   # ≈ 90°
            _seg_from_angle_len(88.5, 2.0),   # ≈ 90°
        ]
        angles = detect_dominant_orientations(segs, bin_size_deg=1.0)

        assert len(angles) == 2
        degrees = sorted(math.degrees(a) % 180.0 for a in angles)
        # Un pic autour de 0° (peut aussi apparaître à ~178-180° ≡ 0°)
        # Un pic autour de 90°
        assert any(d < 10.0 or d > 170.0 for d in degrees), f"Pas de pic à ~0° : {degrees}"
        assert any(80.0 < d < 100.0 for d in degrees), f"Pas de pic à ~90° : {degrees}"

    def test_single_direction(self) -> None:
        """Tous les segments horizontaux → au moins 1 pic à ~0°."""
        segs = [_seg_from_angle_len(0.0, 2.0) for _ in range(5)]
        angles = detect_dominant_orientations(segs)
        assert len(angles) >= 1
        best = min(math.degrees(a) % 180.0 for a in angles)
        # 0° ou ~180° (même direction)
        assert best < 15.0 or best > 165.0

    def test_empty_segments_returns_empty(self) -> None:
        assert detect_dominant_orientations([]) == []

    def test_length_weighting(self) -> None:
        """Un seul long segment horizontal doit dominer sur plusieurs courts verticaux."""
        segs = [
            _seg_from_angle_len(0.0, 10.0),   # 1 segment horizontal long
            _seg_from_angle_len(90.0, 0.3),   # petit segment vertical
            _seg_from_angle_len(90.0, 0.2),   # petit segment vertical
        ]
        angles = detect_dominant_orientations(segs)
        assert len(angles) >= 1
        # Le premier pic (le plus fort) doit être à ~0°
        first_deg = math.degrees(angles[0]) % 180.0
        assert first_deg < 15.0 or first_deg > 165.0

    def test_returns_radians(self) -> None:
        """Les angles retournés doivent être en radians, dans [0, π)."""
        segs = [_seg_from_angle_len(0.0, 3.0), _seg_from_angle_len(90.0, 3.0)]
        angles = detect_dominant_orientations(segs)
        for a in angles:
            assert 0.0 <= a < math.pi + 0.01  # +0.01 pour la tolérance sur 179.5°


# ---------------------------------------------------------------------------
# Tests regularize_segments
# ---------------------------------------------------------------------------

class TestRegularizeSegments:
    def test_snap_to_dominant_88_to_90(self) -> None:
        """Segment à 88° avec dominant à 90°, tolérance 5° → snappé à 90°."""
        seg = _seg_from_angle_len(88.0, 2.0)
        result = regularize_segments([seg], [math.radians(90.0)], snap_tolerance_deg=5.0)

        assert len(result) == 1
        angle_out = math.degrees(math.atan2(
            result[0].y2 - result[0].y1,
            result[0].x2 - result[0].x1,
        )) % 180.0
        assert abs(angle_out - 90.0) < 0.5, f"Angle obtenu : {angle_out:.2f}°"

    def test_no_snap_oblique_45(self) -> None:
        """Segment à 45° avec dominants 0° et 90°, tolérance 5° → inchangé."""
        seg = _seg_from_angle_len(45.0, 2.0)
        result = regularize_segments(
            [seg],
            [math.radians(0.0), math.radians(90.0)],
            snap_tolerance_deg=5.0,
        )
        angle_out = math.degrees(math.atan2(
            result[0].y2 - result[0].y1,
            result[0].x2 - result[0].x1,
        )) % 180.0
        assert abs(angle_out - 45.0) < 1.0, f"Angle obtenu : {angle_out:.2f}°"

    def test_length_preserved_after_snap(self) -> None:
        """Le segment pivote mais sa longueur reste identique."""
        seg = _seg_from_angle_len(91.0, 2.0)
        result = regularize_segments([seg], [math.radians(90.0)], snap_tolerance_deg=5.0)

        original_len = seg.length
        result_len = result[0].length
        assert original_len == pytest.approx(result_len, abs=1e-6)

    def test_center_preserved_after_snap(self) -> None:
        """Le centre du segment doit rester au même endroit après pivotement."""
        seg = _seg_from_angle_len(89.0, 3.0, cx=2.0, cy=1.0)
        result = regularize_segments([seg], [math.radians(90.0)], snap_tolerance_deg=5.0)

        cx_in = (seg.x1 + seg.x2) / 2.0
        cy_in = (seg.y1 + seg.y2) / 2.0
        cx_out = (result[0].x1 + result[0].x2) / 2.0
        cy_out = (result[0].y1 + result[0].y2) / 2.0
        assert cx_out == pytest.approx(cx_in, abs=1e-6)
        assert cy_out == pytest.approx(cy_in, abs=1e-6)

    def test_empty_segments_returns_empty(self) -> None:
        result = regularize_segments([], [0.0])
        assert result == []

    def test_empty_dominant_angles_returns_unchanged(self) -> None:
        seg = _seg_from_angle_len(45.0, 1.0)
        result = regularize_segments([seg], [])
        assert len(result) == 1
        assert result[0].x1 == pytest.approx(seg.x1)

    def test_source_slice_and_confidence_preserved(self) -> None:
        """Les métadonnées du segment doivent être préservées après snapping."""
        seg = DetectedSegment(x1=0, y1=0, x2=0, y2=1, source_slice="mid", confidence=0.7)
        result = regularize_segments([seg], [math.radians(90.0)], snap_tolerance_deg=5.0)
        assert result[0].source_slice == "mid"
        assert result[0].confidence == pytest.approx(0.7)

    def test_multiple_segments_some_snapped(self) -> None:
        """Parmi 3 segments, seul celui dans la tolérance est snappé."""
        segs = [
            _seg_from_angle_len(2.0, 1.0),   # dans tolérance → snappé à 0°
            _seg_from_angle_len(45.0, 1.0),  # hors tolérance → inchangé
            _seg_from_angle_len(88.0, 1.0),  # dans tolérance → snappé à 90°
        ]
        dominants = [math.radians(0.0), math.radians(90.0)]
        result = regularize_segments(segs, dominants, snap_tolerance_deg=5.0)

        angle0 = math.degrees(math.atan2(result[0].y2 - result[0].y1,
                                         result[0].x2 - result[0].x1)) % 180.0
        angle1 = math.degrees(math.atan2(result[1].y2 - result[1].y1,
                                         result[1].x2 - result[1].x1)) % 180.0
        angle2 = math.degrees(math.atan2(result[2].y2 - result[2].y1,
                                         result[2].x2 - result[2].x1)) % 180.0

        assert abs(angle0) < 0.5 or abs(angle0 - 180.0) < 0.5  # snappé à 0°
        assert abs(angle1 - 45.0) < 1.0                          # inchangé
        assert abs(angle2 - 90.0) < 0.5                          # snappé à 90°


# ---------------------------------------------------------------------------
# Tests align_parallel_segments
# ---------------------------------------------------------------------------

class TestAlignParallelSegments:
    def test_two_close_parallel_segments_aligned(self) -> None:
        """Deux segments horizontaux séparés de 1cm → fusionnés sur une droite."""
        s1 = _seg(0.0, 1.000, 2.0, 1.000)  # y = 1.000
        s2 = _seg(0.5, 1.005, 2.5, 1.005)  # y = 1.005 (décalage 5mm)
        result = align_parallel_segments(
            [s1, s2],
            [math.radians(0.0)],
            alignment_tolerance=0.02,
        )
        assert len(result) == 1
        # Le segment fusionné doit s'étendre de ~0 à ~2.5
        xs = sorted([result[0].x1, result[0].x2])
        assert xs[0] == pytest.approx(0.0, abs=0.05)
        assert xs[1] == pytest.approx(2.5, abs=0.05)

    def test_two_far_parallel_not_merged(self) -> None:
        """Deux segments horizontaux séparés de 5cm > tolerance 2cm → non fusionnés."""
        s1 = _seg(0.0, 1.00, 2.0, 1.00)
        s2 = _seg(0.0, 1.05, 2.0, 1.05)   # 5cm de séparation
        result = align_parallel_segments(
            [s1, s2],
            [math.radians(0.0)],
            alignment_tolerance=0.02,
        )
        assert len(result) == 2

    def test_empty_returns_empty(self) -> None:
        result = align_parallel_segments([], [0.0])
        assert result == []

    def test_single_segment_unchanged(self) -> None:
        s = _seg(0.0, 0.0, 3.0, 0.0)
        result = align_parallel_segments([s], [0.0])
        assert len(result) == 1

    def test_perpendicular_segments_not_merged(self) -> None:
        """Un horizontal et un vertical ne doivent pas être fusionnés ensemble."""
        s_h = _seg(0.0, 1.0, 3.0, 1.0)   # horizontal
        s_v = _seg(1.5, 0.0, 1.5, 3.0)   # vertical
        result = align_parallel_segments(
            [s_h, s_v],
            [math.radians(0.0), math.radians(90.0)],
            alignment_tolerance=0.02,
        )
        # Les deux restent séparés (directions différentes)
        assert len(result) == 2
