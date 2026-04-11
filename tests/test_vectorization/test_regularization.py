"""Tests unitaires pour vectorization/regularization.py."""

from __future__ import annotations

import numpy as np
import pytest

from scan2plan.vectorization.regularization import regularize_segments


class TestRegularizeSegments:
    def test_segment_at_89_snapped_to_90(self) -> None:
        """Un segment à 89° doit être snappé à 90° (π/2)."""
        angle_89 = np.deg2rad(89)
        seg = np.array([[0.0, 0.0, np.cos(angle_89), np.sin(angle_89)]])
        dominant = [np.deg2rad(0.0), np.deg2rad(90.0)]
        result = regularize_segments(seg, dominant, snap_tolerance_deg=5.0)

        # L'angle résultant doit être très proche de 90°
        dx = result[0, 2] - result[0, 0]
        dy = result[0, 3] - result[0, 1]
        angle_result = np.degrees(np.arctan2(dy, dx)) % 180
        assert abs(angle_result - 90.0) < 0.5

    def test_segment_beyond_tolerance_not_snapped(self) -> None:
        """Un segment à 45° ne doit pas être snappé à 0° ou 90° (tolérance=5°)."""
        angle_45 = np.deg2rad(45)
        seg = np.array([[0.0, 0.0, np.cos(angle_45), np.sin(angle_45)]])
        dominant = [np.deg2rad(0.0), np.deg2rad(90.0)]
        result = regularize_segments(seg, dominant, snap_tolerance_deg=5.0)

        dx = result[0, 2] - result[0, 0]
        dy = result[0, 3] - result[0, 1]
        angle_result = np.degrees(np.arctan2(dy, dx)) % 180
        # Doit rester proche de 45°
        assert abs(angle_result - 45.0) < 1.0

    def test_empty_orientations_returns_unchanged(self) -> None:
        """Sans orientations dominantes, les segments sont retournés inchangés."""
        seg = np.array([[0.0, 0.0, 1.0, 0.5]])
        result = regularize_segments(seg, [], snap_tolerance_deg=5.0)
        np.testing.assert_array_equal(result, seg)

    def test_empty_segments_returns_empty(self) -> None:
        """Un tableau vide doit retourner un tableau vide."""
        seg = np.zeros((0, 4))
        result = regularize_segments(seg, [0.0], snap_tolerance_deg=5.0)
        assert len(result) == 0

    def test_length_preserved_after_snap(self) -> None:
        """La longueur du segment doit être conservée après le snapping."""
        angle_91 = np.deg2rad(91)
        seg = np.array([[0.0, 0.0, np.cos(angle_91) * 2, np.sin(angle_91) * 2]])
        dominant = [np.deg2rad(90.0)]
        result = regularize_segments(seg, dominant, snap_tolerance_deg=5.0)

        original_len = np.hypot(seg[0, 2] - seg[0, 0], seg[0, 3] - seg[0, 1])
        result_len = np.hypot(result[0, 2] - result[0, 0], result[0, 3] - result[0, 1])
        assert original_len == pytest.approx(result_len, abs=0.01)
