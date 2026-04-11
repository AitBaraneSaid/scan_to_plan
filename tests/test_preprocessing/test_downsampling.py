"""Tests unitaires pour preprocessing/downsampling.py."""

from __future__ import annotations

import numpy as np
import pytest

from scan2plan.preprocessing.downsampling import voxel_downsample


class TestVoxelDownsample:
    def test_reduces_point_count(self, simple_room_points: np.ndarray) -> None:
        """Le downsampling doit réduire le nombre de points."""
        result = voxel_downsample(simple_room_points, voxel_size=0.1)
        assert len(result) < len(simple_room_points)

    def test_shape_is_n_by_3(self, simple_room_points: np.ndarray) -> None:
        result = voxel_downsample(simple_room_points, voxel_size=0.05)
        assert result.ndim == 2
        assert result.shape[1] == 3

    def test_dtype_is_float64(self, simple_room_points: np.ndarray) -> None:
        result = voxel_downsample(simple_room_points, voxel_size=0.05)
        assert result.dtype == np.float64

    def test_invalid_voxel_size_raises(self, simple_room_points: np.ndarray) -> None:
        with pytest.raises(ValueError, match="voxel_size"):
            voxel_downsample(simple_room_points, voxel_size=0.0)

    def test_negative_voxel_size_raises(self, simple_room_points: np.ndarray) -> None:
        with pytest.raises(ValueError):
            voxel_downsample(simple_room_points, voxel_size=-0.01)

    def test_empty_points_raises(self) -> None:
        with pytest.raises(ValueError, match="vide"):
            voxel_downsample(np.zeros((0, 3)), voxel_size=0.01)

    def test_bounding_box_preserved(self, simple_room_points: np.ndarray) -> None:
        """La bounding box du résultat doit être proche de celle de l'entrée."""
        result = voxel_downsample(simple_room_points, voxel_size=0.05)
        for dim in range(3):
            assert result[:, dim].min() >= simple_room_points[:, dim].min() - 0.1
            assert result[:, dim].max() <= simple_room_points[:, dim].max() + 0.1
