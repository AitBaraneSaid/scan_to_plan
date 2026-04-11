"""Tests unitaires pour src/scan2plan/io/readers.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scan2plan.io.readers import (
    UnsupportedFormatError,
    _log_point_cloud_info,
    read_point_cloud,
)


# ---------------------------------------------------------------------------
# Tests de dispatch par extension
# ---------------------------------------------------------------------------

class TestReadPointCloudDispatch:
    """Vérifie que read_point_cloud dispatche correctement selon l'extension."""

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """Un fichier .pts doit lever UnsupportedFormatError."""
        fake = tmp_path / "scan.pts"
        fake.write_text("dummy content")
        with pytest.raises(UnsupportedFormatError, match=r"\.pts"):
            read_point_cloud(fake)

    def test_rcs_extension_raises(self, tmp_path: Path) -> None:
        """Un fichier .rcs (Autodesk) doit lever UnsupportedFormatError."""
        fake = tmp_path / "scan.rcs"
        fake.write_text("dummy content")
        with pytest.raises(UnsupportedFormatError, match="non supporté"):
            read_point_cloud(fake)

    def test_no_extension_raises(self, tmp_path: Path) -> None:
        """Un fichier sans extension doit lever UnsupportedFormatError."""
        fake = tmp_path / "scan"
        fake.write_text("dummy content")
        with pytest.raises(UnsupportedFormatError):
            read_point_cloud(fake)

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Un fichier inexistant doit lever FileNotFoundError."""
        missing = tmp_path / "nonexistent.e57"
        with pytest.raises(FileNotFoundError):
            read_point_cloud(missing)

    def test_error_message_lists_supported_formats(self, tmp_path: Path) -> None:
        """Le message d'erreur doit lister les formats acceptés."""
        fake = tmp_path / "scan.dwg"
        fake.write_text("dummy")
        with pytest.raises(UnsupportedFormatError) as exc_info:
            read_point_cloud(fake)
        msg = str(exc_info.value)
        assert ".e57" in msg
        assert ".las" in msg or ".laz" in msg


# ---------------------------------------------------------------------------
# Tests avec le nuage synthétique (.npy via conftest)
# ---------------------------------------------------------------------------

class TestSimpleRoomFixture:
    """Vérifie les propriétés géométriques du nuage synthétique simple_room."""

    def test_shape_is_n_by_3(self, simple_room_points: np.ndarray) -> None:
        """Le nuage doit avoir exactement 3 colonnes (X, Y, Z)."""
        assert simple_room_points.ndim == 2
        assert simple_room_points.shape[1] == 3

    def test_dtype_is_float64(self, simple_room_points: np.ndarray) -> None:
        """Le nuage doit être en float64."""
        assert simple_room_points.dtype == np.float64

    def test_approximately_ten_thousand_points(self, simple_room_points: np.ndarray) -> None:
        """Le nuage doit contenir ~10 000 points (±5 %)."""
        n = len(simple_room_points)
        assert 9_000 <= n <= 11_000, f"Attendu ~10 000 points, reçu {n}"

    def test_z_range_matches_room_height(self, simple_room_points: np.ndarray) -> None:
        """Le sol doit être ≈ Z=0 et le plafond ≈ Z=2.5 m (bruit σ=2 mm)."""
        z_min = simple_room_points[:, 2].min()
        z_max = simple_room_points[:, 2].max()
        assert z_min > -0.02, f"Sol trop bas : Z_min={z_min:.4f} m"
        assert z_max < 2.52, f"Plafond trop haut : Z_max={z_max:.4f} m"

    def test_x_range_matches_room_width(self, simple_room_points: np.ndarray) -> None:
        """La plage X doit correspondre à ±2.0 m (largeur 4 m, centrée en 0)."""
        x_min = simple_room_points[:, 0].min()
        x_max = simple_room_points[:, 0].max()
        assert x_min > -2.02
        assert x_max < 2.02

    def test_y_range_matches_room_depth(self, simple_room_points: np.ndarray) -> None:
        """La plage Y doit correspondre à ±1.5 m (profondeur 3 m, centrée en 0)."""
        y_min = simple_room_points[:, 1].min()
        y_max = simple_room_points[:, 1].max()
        assert y_min > -1.52
        assert y_max < 1.52

    def test_no_nan_or_inf(self, simple_room_points: np.ndarray) -> None:
        """Le nuage ne doit contenir aucune valeur non-finie."""
        assert np.isfinite(simple_room_points).all()


# ---------------------------------------------------------------------------
# Tests de la fonction de logging interne
# ---------------------------------------------------------------------------

class TestLogPointCloudInfo:
    """Vérifie que _log_point_cloud_info n'échoue pas sur des données valides."""

    def test_does_not_raise_on_valid_points(self, simple_room_points: np.ndarray) -> None:
        """Doit s'exécuter sans exception."""
        _log_point_cloud_info(simple_room_points, Path("test.e57"), "E57")

    def test_does_not_raise_on_single_point(self) -> None:
        """Doit fonctionner même avec un seul point."""
        pts = np.array([[1.0, 2.0, 3.0]])
        _log_point_cloud_info(pts, Path("single.las"), "LAS/LAZ")
