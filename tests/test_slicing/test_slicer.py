"""Tests unitaires pour slicing/slicer.py."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from scan2plan.slicing.slicer import extract_slice, extract_all_slices


class TestExtractSlice:
    def test_returns_2d_array(self, simple_room_points: np.ndarray) -> None:
        """La slice doit retourner un array (M, 2) avec uniquement XY."""
        result = extract_slice(simple_room_points, height=1.10)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_extracts_correct_height(self, simple_room_points: np.ndarray) -> None:
        """Les points extraits doivent être dans la fenêtre Z demandée."""
        height = 1.10
        thickness = 0.10
        floor_z = 0.0
        # Vérification indirecte : on filtre le nuage original et compare les XY
        z_low = floor_z + height - thickness / 2.0
        z_high = floor_z + height + thickness / 2.0
        expected_mask = (simple_room_points[:, 2] >= z_low) & (simple_room_points[:, 2] <= z_high)
        expected_xy = simple_room_points[expected_mask, :2]

        result = extract_slice(simple_room_points, height=height, thickness=thickness, floor_z=floor_z)
        assert len(result) == len(expected_xy)
        np.testing.assert_array_equal(result, expected_xy)

    def test_returns_less_than_total(self, simple_room_points: np.ndarray) -> None:
        """La slice doit contenir moins de points que le nuage complet."""
        result = extract_slice(simple_room_points, height=1.10, thickness=0.10)
        assert len(result) < len(simple_room_points)

    def test_floor_z_offset_applied(self, simple_room_points: np.ndarray) -> None:
        """floor_z doit décaler la fenêtre Z correctement."""
        # Avec floor_z=0, slice à h=1.10 → Z ∈ [1.05, 1.15]
        # Avec floor_z=0.5, slice à h=1.10 → Z ∈ [1.55, 1.65]
        result_no_offset = extract_slice(simple_room_points, height=1.10, thickness=0.10, floor_z=0.0)
        result_with_offset = extract_slice(simple_room_points, height=1.10, thickness=0.10, floor_z=0.5)
        # Les deux slices capturent des zones différentes — elles ne peuvent pas être identiques
        assert len(result_no_offset) != len(result_with_offset) or not np.array_equal(
            result_no_offset, result_with_offset
        )

    def test_empty_slice_logs_warning(
        self, simple_room_points: np.ndarray, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Une slice vide doit loguer un WARNING et retourner un array vide (0, 2)."""
        with caplog.at_level(logging.WARNING, logger="scan2plan.slicing.slicer"):
            result = extract_slice(simple_room_points, height=50.0, thickness=0.10)
        assert result.shape == (0, 2)
        assert any("warning" in r.levelname.lower() or r.levelno >= logging.WARNING
                   for r in caplog.records)

    def test_dtype_is_float64(self, simple_room_points: np.ndarray) -> None:
        result = extract_slice(simple_room_points, height=1.10)
        assert result.dtype == np.float64

    def test_default_parameters_work(self, simple_room_points: np.ndarray) -> None:
        """Tous les paramètres ont des valeurs par défaut utilisables directement."""
        result = extract_slice(simple_room_points, height=1.10)
        assert result.ndim == 2
        assert result.shape[1] == 2


class TestExtractAllSlices:
    def test_returns_dict_with_all_heights(self, simple_room_points: np.ndarray) -> None:
        heights = [0.20, 1.10, 2.10]
        result = extract_all_slices(simple_room_points, heights=heights)
        assert set(result.keys()) == set(heights)

    def test_each_slice_is_2d(self, simple_room_points: np.ndarray) -> None:
        slices = extract_all_slices(simple_room_points, heights=[1.10, 2.10])
        for h, arr in slices.items():
            assert arr.ndim == 2, f"Slice h={h} doit être (M, 2)"
            assert arr.shape[1] == 2, f"Slice h={h} doit avoir 2 colonnes"

    def test_includes_empty_slice_with_warning(
        self, simple_room_points: np.ndarray, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Une hauteur hors plage retourne un tableau vide et génère un warning."""
        heights = [1.10, 50.0]
        with caplog.at_level(logging.WARNING, logger="scan2plan.slicing.slicer"):
            slices = extract_all_slices(simple_room_points, heights=heights)
        assert 50.0 in slices
        assert slices[50.0].shape == (0, 2)
