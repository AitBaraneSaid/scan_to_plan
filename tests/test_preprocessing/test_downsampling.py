"""Tests unitaires pour preprocessing/downsampling.py."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import KDTree

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

    def test_uniform_density(self, simple_room_points: np.ndarray) -> None:
        """La variance des distances au plus proche voisin doit diminuer après downsampling.

        Le downsampling voxel uniformise la densité : les écarts entre voisins
        deviennent plus réguliers, donc la variance diminue.
        """
        result = voxel_downsample(simple_room_points, voxel_size=0.05)

        tree_orig = KDTree(simple_room_points)
        d_orig, _ = tree_orig.query(simple_room_points, k=2)
        var_orig = float(np.var(d_orig[:, 1]))

        tree_down = KDTree(result)
        d_down, _ = tree_down.query(result, k=2)
        var_down = float(np.var(d_down[:, 1]))

        assert var_down < var_orig, (
            f"Variance avant : {var_orig:.6f}, après : {var_down:.6f} — "
            "la densité devrait être plus uniforme après downsampling."
        )

    def test_empty_input_raises(self) -> None:
        """Un nuage vide doit lever ValueError."""
        with pytest.raises(ValueError, match="vide"):
            voxel_downsample(np.zeros((0, 3)), voxel_size=0.01)

    def test_invalid_voxel_size_raises(self, simple_room_points: np.ndarray) -> None:
        with pytest.raises(ValueError, match="voxel_size"):
            voxel_downsample(simple_room_points, voxel_size=0.0)

    def test_negative_voxel_size_raises(self, simple_room_points: np.ndarray) -> None:
        with pytest.raises(ValueError):
            voxel_downsample(simple_room_points, voxel_size=-0.01)

    def test_bounding_box_preserved(self, simple_room_points: np.ndarray) -> None:
        """La bounding box du résultat doit rester dans celle de l'entrée."""
        result = voxel_downsample(simple_room_points, voxel_size=0.05)
        for dim in range(3):
            assert result[:, dim].min() >= simple_room_points[:, dim].min() - 0.1
            assert result[:, dim].max() <= simple_room_points[:, dim].max() + 0.1

    def test_default_voxel_size(self, simple_room_points: np.ndarray) -> None:
        """Le paramètre par défaut voxel_size=0.005 doit fonctionner sans argument."""
        result = voxel_downsample(simple_room_points)
        assert result.shape[1] == 3
        assert len(result) > 0
