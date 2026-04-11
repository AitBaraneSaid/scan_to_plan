"""Tests unitaires pour preprocessing/floor_ceiling.py."""

from __future__ import annotations

import numpy as np
import pytest

from scan2plan.preprocessing.floor_ceiling import (
    NoFloorDetectedError,
    NoCeilingDetectedError,
    detect_floor,
    detect_ceiling,
    filter_vertical_range,
)


class TestDetectFloor:
    def test_detect_floor_on_synthetic(self, simple_room_points: np.ndarray) -> None:
        """Le sol du nuage synthétique est à Z≈0 — tolérance 2 cm."""
        z_floor, _ = detect_floor(simple_room_points)
        assert abs(z_floor) < 0.02, f"Sol attendu ≈0.0 m, obtenu {z_floor:.4f} m"

    def test_returns_boolean_mask(self, simple_room_points: np.ndarray) -> None:
        """Le masque doit être un array booléen de longueur N."""
        _, mask = detect_floor(simple_room_points)
        assert mask.dtype == bool
        assert mask.shape == (len(simple_room_points),)

    def test_mask_selects_floor_points(self, simple_room_points: np.ndarray) -> None:
        """Les points sélectionnés par le masque doivent être proches du sol."""
        z_floor, mask = detect_floor(simple_room_points)  # les deux utilisés
        floor_pts_z = simple_room_points[mask, 2]
        # Tous les inliers doivent être dans ±distance_threshold du plan sol
        assert np.all(np.abs(floor_pts_z - z_floor) < 0.05)

    def test_no_floor_raises(self) -> None:
        """Un nuage sur un plan vertical ne doit pas lever de sol horizontal."""
        rng = np.random.default_rng(42)
        # Points sur le plan X=0 (normal = [1,0,0] — vertical, pas horizontal)
        # Peu d'inliers attendus dans un plan horizontal avec seuil très strict
        vertical_wall = rng.uniform(-1, 1, (300, 3))
        vertical_wall[:, 0] = rng.uniform(-0.001, 0.001, 300)  # X ≈ 0, pas de plan horizontal
        with pytest.raises(NoFloorDetectedError):
            detect_floor(
                vertical_wall,
                distance_threshold=0.001,
                num_iterations=200,
                normal_tolerance_deg=1.0,  # tolérance ultra-stricte : seul un plan quasi-parfait qualifie
            )

    def test_default_parameters_work(self, simple_room_points: np.ndarray) -> None:
        """Tous les paramètres ont des valeurs par défaut utilisables directement."""
        z_floor, _ = detect_floor(simple_room_points)
        assert isinstance(z_floor, float)


class TestDetectCeiling:
    # floor_z fixé à 0.0 (sol synthétique connu) pour isoler detect_ceiling
    # et éviter la propagation d'erreur si detect_floor détecte le plafond en premier.
    _FLOOR_Z = 0.0
    _MIN_HEIGHT = 2.0

    def test_detect_ceiling_on_synthetic(self, simple_room_points: np.ndarray) -> None:
        """Le plafond du nuage synthétique est à Z≈2.5 m — tolérance 2 cm."""
        z_ceiling, _ = detect_ceiling(
            simple_room_points, floor_z=self._FLOOR_Z, min_height=self._MIN_HEIGHT
        )
        assert abs(z_ceiling - 2.5) < 0.02, (
            f"Plafond attendu ≈2.5 m, obtenu {z_ceiling:.4f} m"
        )

    def test_returns_boolean_mask(self, simple_room_points: np.ndarray) -> None:
        """Le masque retourné doit correspondre au nuage original complet."""
        _, mask = detect_ceiling(
            simple_room_points, floor_z=self._FLOOR_Z, min_height=self._MIN_HEIGHT
        )
        assert mask.dtype == bool
        assert mask.shape == (len(simple_room_points),)

    def test_ceiling_above_floor(self, simple_room_points: np.ndarray) -> None:
        """Le plafond doit être plus haut que le sol."""
        z_ceiling, _ = detect_ceiling(
            simple_room_points, floor_z=self._FLOOR_Z, min_height=self._MIN_HEIGHT
        )
        assert z_ceiling > self._FLOOR_Z

    def test_no_ceiling_raises_when_no_high_points(self) -> None:
        """Si aucun point n'est au-dessus de floor_z + min_height, lever NoCeilingDetectedError."""
        pts = np.zeros((50, 3), dtype=np.float64)  # tous à Z=0
        with pytest.raises(NoCeilingDetectedError):
            detect_ceiling(pts, floor_z=0.0, min_height=2.0)


class TestFilterVerticalRange:
    def test_removes_points_outside_range(self) -> None:
        """Les points hors de [z_min - margin, z_max + margin] doivent être supprimés."""
        pts = np.array([
            [0.0, 0.0, -0.5],   # trop bas (hors margin=0.05)
            [0.0, 0.0,  1.0],   # dans la plage
            [0.0, 0.0,  3.0],   # trop haut (hors margin=0.05)
        ], dtype=np.float64)
        result = filter_vertical_range(pts, z_min=0.0, z_max=2.5, margin=0.05)
        assert len(result) == 1
        assert result[0, 2] == pytest.approx(1.0)

    def test_margin_includes_boundary_points(self) -> None:
        """Les points dans la marge doivent être conservés."""
        pts = np.array([
            [0.0, 0.0, -0.03],   # dans la marge basse (margin=0.05)
            [0.0, 0.0,  2.53],   # dans la marge haute
        ], dtype=np.float64)
        result = filter_vertical_range(pts, z_min=0.0, z_max=2.5, margin=0.05)
        assert len(result) == 2

    def test_all_points_in_full_range(self, simple_room_points: np.ndarray) -> None:
        """Avec des bornes englobant tout le nuage, aucun point ne doit être perdu."""
        z_min = simple_room_points[:, 2].min()
        z_max = simple_room_points[:, 2].max()
        result = filter_vertical_range(simple_room_points, z_min=z_min, z_max=z_max, margin=0.1)
        assert len(result) == len(simple_room_points)

    def test_shape_preserved(self, simple_room_points: np.ndarray) -> None:
        """Le résultat doit toujours avoir 3 colonnes."""
        result = filter_vertical_range(simple_room_points, z_min=0.0, z_max=2.5)
        assert result.ndim == 2
        assert result.shape[1] == 3
