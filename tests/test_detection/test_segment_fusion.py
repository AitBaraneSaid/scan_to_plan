"""Tests unitaires pour detection/segment_fusion.py."""

from __future__ import annotations

import math

import pytest

from scan2plan.detection.line_detection import DetectedSegment
from scan2plan.detection.segment_fusion import fuse_collinear_segments


def _seg(x1: float, y1: float, x2: float, y2: float, conf: float = 0.8) -> DetectedSegment:
    """Fabrique un DetectedSegment avec des valeurs par défaut."""
    return DetectedSegment(x1=x1, y1=y1, x2=x2, y2=y2, source_slice="mid", confidence=conf)


class TestFuseCollinearSegments:
    def test_fuse_two_collinear(self) -> None:
        """2 segments alignés avec un gap de 5 cm → fusionnés en 1."""
        segments = [
            _seg(0.0, 0.0, 1.0, 0.0),
            _seg(1.05, 0.0, 2.0, 0.0),
        ]
        result = fuse_collinear_segments(segments, angle_tolerance_deg=3.0,
                                         perpendicular_dist=0.03, max_gap=0.20)
        assert len(result) == 1
        assert result[0].length == pytest.approx(2.0, abs=0.15)

    def test_fuse_three_fragments(self) -> None:
        """3 segments colinéaires fragmentés → fusionnés en 1."""
        segments = [
            _seg(0.0, 0.0, 1.0, 0.0),
            _seg(1.05, 0.0, 2.0, 0.0),
            _seg(2.05, 0.0, 3.0, 0.0),
        ]
        result = fuse_collinear_segments(segments, angle_tolerance_deg=3.0,
                                         perpendicular_dist=0.03, max_gap=0.20)
        assert len(result) == 1
        assert result[0].length == pytest.approx(3.0, abs=0.15)

    def test_no_fuse_perpendicular(self) -> None:
        """2 segments perpendiculaires → pas fusionnés."""
        segments = [
            _seg(0.0, 0.0, 1.0, 0.0),
            _seg(0.5, -0.5, 0.5, 0.5),
        ]
        result = fuse_collinear_segments(segments, angle_tolerance_deg=3.0,
                                         perpendicular_dist=0.03, max_gap=0.20)
        assert len(result) == 2

    def test_no_fuse_distant_parallel(self) -> None:
        """2 segments parallèles mais distants de 50 cm en Y → pas fusionnés."""
        segments = [
            _seg(0.0, 0.0, 1.0, 0.0),
            _seg(0.0, 0.50, 1.0, 0.50),
        ]
        result = fuse_collinear_segments(segments, angle_tolerance_deg=3.0,
                                         perpendicular_dist=0.03, max_gap=0.20)
        assert len(result) == 2

    def test_no_fuse_large_gap(self) -> None:
        """Gap de 1.0 m > max_gap=0.20 → pas de fusion."""
        segments = [
            _seg(0.0, 0.0, 1.0, 0.0),
            _seg(2.0, 0.0, 3.0, 0.0),
        ]
        result = fuse_collinear_segments(segments, angle_tolerance_deg=3.0,
                                         perpendicular_dist=0.03, max_gap=0.20)
        assert len(result) == 2

    def test_empty_input(self) -> None:
        """Entrée vide → sortie vide."""
        result = fuse_collinear_segments([], angle_tolerance_deg=3.0,
                                         perpendicular_dist=0.03, max_gap=0.20)
        assert result == []

    def test_single_segment_unchanged(self) -> None:
        """Un seul segment ne doit pas être modifié."""
        seg = _seg(0.0, 0.0, 2.0, 0.0, conf=0.9)
        result = fuse_collinear_segments([seg], angle_tolerance_deg=3.0,
                                         perpendicular_dist=0.03, max_gap=0.20)
        assert len(result) == 1
        assert result[0].x1 == pytest.approx(seg.x1, abs=1e-6)
        assert result[0].x2 == pytest.approx(seg.x2, abs=1e-6)

    def test_convergence_within_max_iterations(self, caplog: pytest.LogCaptureFixture) -> None:
        """L'algorithme doit converger en moins de 10 itérations."""
        import logging
        import re
        segments = [_seg(float(i) * 1.05, 0.0, float(i) * 1.05 + 1.0, 0.0) for i in range(5)]
        with caplog.at_level(logging.INFO, logger="scan2plan.detection.segment_fusion"):
            result = fuse_collinear_segments(segments, angle_tolerance_deg=3.0,
                                             perpendicular_dist=0.03, max_gap=0.20)
        assert len(result) <= len(segments)
        # Extraire le nombre de passes depuis le log "N passes (max 10)"
        for record in caplog.records:
            m = re.search(r"en (\d+) passes", record.message)
            if m:
                n_passes = int(m.group(1))
                assert n_passes < 10, f"Convergence trop lente : {n_passes} passes"
                break

    def test_result_is_list_of_detected_segments(self) -> None:
        segments = [_seg(0.0, 0.0, 1.0, 0.0), _seg(1.05, 0.0, 2.0, 0.0)]
        result = fuse_collinear_segments(segments)
        assert isinstance(result, list)
        for seg in result:
            assert isinstance(seg, DetectedSegment)

    def test_confidence_max_preserved(self) -> None:
        """Lors d'une fusion, la confiance max des deux segments est conservée."""
        segments = [
            _seg(0.0, 0.0, 1.0, 0.0, conf=0.6),
            _seg(1.05, 0.0, 2.0, 0.0, conf=0.9),
        ]
        result = fuse_collinear_segments(segments)
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.9)

    def test_fuse_nearly_collinear_with_slight_angle(self) -> None:
        """Deux segments avec un angle de 1° (< tolérance 3°) → fusionnés."""
        angle_rad = math.radians(1.0)
        segments = [
            _seg(0.0, 0.0, 1.0, 0.0),
            _seg(1.05, 0.0, 2.05, math.sin(angle_rad)),
        ]
        result = fuse_collinear_segments(segments, angle_tolerance_deg=3.0,
                                         perpendicular_dist=0.05, max_gap=0.20)
        assert len(result) == 1
