"""Tests unitaires pour detection/morphology.py."""

from __future__ import annotations

import numpy as np
import pytest

from scan2plan.detection.morphology import binarize_density_map, morphological_cleanup


def _make_density_map(shape: tuple[int, int] = (100, 100)) -> np.ndarray:
    """Crée une density map synthétique avec une ligne horizontale de points."""
    dm = np.zeros(shape, dtype=np.uint16)
    mid = shape[0] // 2
    dm[mid - 2 : mid + 2, 10:90] = 5
    return dm


class TestBinarizeDensityMap:
    def test_returns_uint8(self) -> None:
        dm = _make_density_map()
        result = binarize_density_map(dm)
        assert result.dtype == np.uint8

    def test_shape_preserved(self) -> None:
        dm = _make_density_map((80, 120))
        result = binarize_density_map(dm)
        assert result.shape == (80, 120)

    def test_values_are_0_or_255(self) -> None:
        dm = _make_density_map()
        result = binarize_density_map(dm)
        unique = set(np.unique(result))
        assert unique <= {0, 255}

    def test_occupied_pixels_detected(self) -> None:
        """Les pixels avec des points doivent être binarisés à 255."""
        dm = _make_density_map()
        result = binarize_density_map(dm)
        assert result.max() == 255

    def test_empty_map_gives_all_zeros(self) -> None:
        dm = np.zeros((50, 50), dtype=np.uint16)
        result = binarize_density_map(dm)
        assert result.max() == 0

    def test_unknown_method_raises(self) -> None:
        dm = _make_density_map()
        with pytest.raises(ValueError, match="Méthode"):
            binarize_density_map(dm, method="adaptive")

    def test_default_method_is_otsu(self) -> None:
        """Appel sans argument doit fonctionner (méthode par défaut = otsu)."""
        dm = _make_density_map()
        result = binarize_density_map(dm)
        assert result.shape == dm.shape


class TestMorphologicalCleanup:
    def test_shape_preserved(self) -> None:
        binary = binarize_density_map(_make_density_map((80, 120)))
        result = morphological_cleanup(binary)
        assert result.shape == (80, 120)

    def test_dtype_is_uint8(self) -> None:
        binary = binarize_density_map(_make_density_map())
        result = morphological_cleanup(binary)
        assert result.dtype == np.uint8

    def test_closing_fills_small_gap(self) -> None:
        """La fermeture doit combler un petit trou dans une ligne."""
        binary = np.zeros((50, 50), dtype=np.uint8)
        binary[25, 5:20] = 255
        binary[25, 22:40] = 255

        cleaned = morphological_cleanup(binary, kernel_size=5, close_iterations=2, open_iterations=0)
        assert cleaned[25, 21] == 255, "Le gap doit être comblé par la fermeture."

    def test_opening_removes_isolated_pixel(self) -> None:
        """L'ouverture doit supprimer un pixel isolé."""
        binary = np.zeros((50, 50), dtype=np.uint8)
        binary[5, 5] = 255

        cleaned = morphological_cleanup(binary, kernel_size=5, close_iterations=0, open_iterations=1)
        assert cleaned[5, 5] == 0, "Le pixel isolé doit être supprimé par l'ouverture."

    def test_values_are_0_or_255(self) -> None:
        binary = binarize_density_map(_make_density_map())
        result = morphological_cleanup(binary)
        unique = set(np.unique(result))
        assert unique <= {0, 255}

    def test_default_parameters_work(self) -> None:
        binary = binarize_density_map(_make_density_map())
        result = morphological_cleanup(binary)
        assert result.shape == binary.shape
