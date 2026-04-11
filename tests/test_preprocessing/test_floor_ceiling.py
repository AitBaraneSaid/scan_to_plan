"""Tests unitaires pour preprocessing/floor_ceiling.py."""

from __future__ import annotations

import numpy as np
import pytest

from scan2plan.preprocessing.floor_ceiling import (
    detect_floor_and_ceiling,
    filter_vertical_range,
)


class TestDetectFloorAndCeiling:
    def test_detects_correct_floor_altitude(self, simple_room_points: np.ndarray) -> None:
        """Le sol doit être détecté proche de Z=0 (tolérance 5 cm)."""
        z_floor, _ = detect_floor_and_ceiling(
            simple_room_points,
            ransac_distance=0.02,
            ransac_iterations=500,
            normal_tolerance_deg=10.0,
        )
        assert abs(z_floor) < 0.05, f"Sol attendu ≈0.0 m, obtenu {z_floor:.4f} m"

    def test_detects_correct_ceiling_altitude(self, simple_room_points: np.ndarray) -> None:
        """Le plafond doit être détecté proche de Z=2.5 m (tolérance 5 cm)."""
        _, z_ceiling = detect_floor_and_ceiling(
            simple_room_points,
            ransac_distance=0.02,
            ransac_iterations=500,
            normal_tolerance_deg=10.0,
        )
        assert abs(z_ceiling - 2.5) < 0.05, f"Plafond attendu ≈2.5 m, obtenu {z_ceiling:.4f} m"


class TestFilterVerticalRange:
    def test_removes_points_outside_range(self) -> None:
        """Les points hors de [z_floor, z_ceiling] doivent être supprimés."""
        pts = np.array([
            [0.0, 0.0, -0.5],   # sous le sol
            [0.0, 0.0,  1.0],   # dans la plage
            [0.0, 0.0,  3.0],   # au-dessus du plafond
        ], dtype=np.float64)
        result = filter_vertical_range(pts, z_floor=0.0, z_ceiling=2.5)
        assert len(result) == 1
        assert result[0, 2] == pytest.approx(1.0)

    def test_all_points_valid(self, simple_room_points: np.ndarray) -> None:
        """Après filtrage avec les bonnes bornes, tous les points doivent rester."""
        z_min = simple_room_points[:, 2].min() - 0.01
        z_max = simple_room_points[:, 2].max() + 0.01
        result = filter_vertical_range(simple_room_points, z_floor=z_min, z_ceiling=z_max)
        assert len(result) == len(simple_room_points)
