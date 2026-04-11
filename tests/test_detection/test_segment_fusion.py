"""Tests unitaires pour detection/segment_fusion.py."""

from __future__ import annotations

import numpy as np
import pytest

from scan2plan.detection.segment_fusion import fuse_collinear_segments


class TestFuseCollinearSegments:
    def test_three_fragments_fused_to_one(self) -> None:
        """Trois segments colinéaires fragmentés doivent être fusionnés en 1."""
        # Segment horizontal Y=0, fragmenté en 3 morceaux
        segments = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [1.05, 0.0, 2.0, 0.0],
            [2.05, 0.0, 3.0, 0.0],
        ])
        result = fuse_collinear_segments(
            segments,
            angle_tolerance_deg=3.0,
            perpendicular_dist=0.03,
            max_gap=0.20,
        )
        assert len(result) == 1
        # Le segment fusionné doit couvrir toute la longueur
        length = np.hypot(result[0, 2] - result[0, 0], result[0, 3] - result[0, 1])
        assert length == pytest.approx(3.0, abs=0.15)

    def test_perpendicular_segments_not_fused(self) -> None:
        """Deux segments perpendiculaires ne doivent pas être fusionnés."""
        segments = np.array([
            [0.0, 0.0, 1.0, 0.0],  # horizontal
            [0.5, -0.5, 0.5, 0.5],  # vertical
        ])
        result = fuse_collinear_segments(
            segments,
            angle_tolerance_deg=3.0,
            perpendicular_dist=0.03,
            max_gap=0.20,
        )
        assert len(result) == 2

    def test_empty_input_returns_empty(self) -> None:
        """Un tableau vide doit retourner un tableau vide."""
        segments = np.zeros((0, 4), dtype=np.float64)
        result = fuse_collinear_segments(segments, 3.0, 0.03, 0.20)
        assert len(result) == 0

    def test_single_segment_unchanged(self) -> None:
        """Un seul segment ne doit pas être modifié."""
        segments = np.array([[0.0, 0.0, 2.0, 0.0]])
        result = fuse_collinear_segments(segments, 3.0, 0.03, 0.20)
        assert len(result) == 1

    def test_large_gap_prevents_fusion(self) -> None:
        """Un gap trop grand doit empêcher la fusion."""
        segments = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 3.0, 0.0],  # gap = 1.0 m > max_gap=0.20
        ])
        result = fuse_collinear_segments(
            segments,
            angle_tolerance_deg=3.0,
            perpendicular_dist=0.03,
            max_gap=0.20,
        )
        assert len(result) == 2
