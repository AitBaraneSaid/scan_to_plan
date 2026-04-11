"""Tests unitaires pour slicing/density_map.py."""

from __future__ import annotations

import numpy as np
import pytest

from scan2plan.slicing.density_map import DensityMap, compute_density_map


class TestComputeDensityMap:
    def test_returns_density_map_instance(self, simple_room_points: np.ndarray) -> None:
        result = compute_density_map(simple_room_points, resolution=0.05)
        assert isinstance(result, DensityMap)

    def test_image_is_2d(self, simple_room_points: np.ndarray) -> None:
        result = compute_density_map(simple_room_points, resolution=0.05)
        assert result.image.ndim == 2

    def test_image_dtype_uint16(self, simple_room_points: np.ndarray) -> None:
        result = compute_density_map(simple_room_points, resolution=0.05)
        assert result.image.dtype == np.uint16

    def test_nonzero_pixels_cover_room_footprint(self, simple_room_points: np.ndarray) -> None:
        """Les pixels non-nuls doivent correspondre aux zones avec des points."""
        result = compute_density_map(simple_room_points, resolution=0.05)
        assert result.image.max() > 0

    def test_x_min_is_below_data_min(self, simple_room_points: np.ndarray) -> None:
        """x_min doit être inférieur ou égal au minimum des X du nuage (marge incluse)."""
        result = compute_density_map(simple_room_points, resolution=0.05)
        assert result.x_min <= simple_room_points[:, 0].min()

    def test_resolution_stored(self, simple_room_points: np.ndarray) -> None:
        res = 0.02
        result = compute_density_map(simple_room_points, resolution=res)
        assert result.resolution == pytest.approx(res)

    def test_invalid_resolution_raises(self, simple_room_points: np.ndarray) -> None:
        with pytest.raises(ValueError, match="resolution"):
            compute_density_map(simple_room_points, resolution=0.0)

    def test_empty_slice_raises(self) -> None:
        with pytest.raises(ValueError, match="vide"):
            compute_density_map(np.zeros((0, 3)), resolution=0.01)
