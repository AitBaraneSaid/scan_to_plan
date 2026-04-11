"""Tests unitaires pour slicing/slicer.py."""

from __future__ import annotations

import numpy as np
import pytest

from scan2plan.slicing.slicer import InsufficientPointsError, extract_slice


class TestExtractSlice:
    def test_extracts_correct_z_range(self, simple_room_points: np.ndarray) -> None:
        """Les points extraits doivent tous être dans la fenêtre Z demandée."""
        z_floor = 0.0
        height = 1.10
        thickness = 0.10
        result = extract_slice(simple_room_points, z_floor, height, thickness)

        z_low = z_floor + height - thickness / 2
        z_high = z_floor + height + thickness / 2
        assert (result[:, 2] >= z_low - 1e-6).all()
        assert (result[:, 2] <= z_high + 1e-6).all()

    def test_returns_less_than_total(self, simple_room_points: np.ndarray) -> None:
        """La slice doit contenir moins de points que le nuage complet."""
        result = extract_slice(simple_room_points, 0.0, 1.10, 0.10)
        assert len(result) < len(simple_room_points)

    def test_empty_slice_raises(self) -> None:
        """Une hauteur hors de la plage Z doit lever InsufficientPointsError."""
        pts = np.random.default_rng(0).random((500, 3))
        pts[:, 2] *= 2.5  # Z ∈ [0, 2.5]
        with pytest.raises(InsufficientPointsError):
            extract_slice(pts, z_floor=0.0, height=10.0, thickness=0.10)

    def test_shape_is_n_by_3(self, simple_room_points: np.ndarray) -> None:
        result = extract_slice(simple_room_points, 0.0, 0.20, 0.10)
        assert result.ndim == 2
        assert result.shape[1] == 3
