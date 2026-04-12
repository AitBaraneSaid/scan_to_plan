"""Tests unitaires pour config_profiles.py."""

from __future__ import annotations

import numpy as np
import pytest

from scan2plan.config_profiles import (
    AVAILABLE_PROFILES,
    CalibrationResult,
    apply_profile,
    auto_calibrate,
    calibrate_slice_heights,
    load_profile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_thin_wall_image(
    size_px: int = 400,
    wall_thickness_px: int = 14,  # ≈ 7 cm à 5 mm/px
    n_walls: int = 4,
) -> np.ndarray:
    """Image synthétique avec des murs fins (profil récent)."""
    image = np.zeros((size_px, size_px), dtype=np.float32)
    spacing = size_px // (n_walls + 1)
    for i in range(1, n_walls + 1):
        pos = i * spacing
        image[pos: pos + wall_thickness_px, 20: size_px - 20] = 10.0
        image[20: size_px - 20, pos: pos + wall_thickness_px] = 10.0
    return image


def _make_thick_wall_image(
    size_px: int = 400,
    wall_thickness_px: int = 50,  # ≈ 25 cm à 5 mm/px
    n_walls: int = 3,
) -> np.ndarray:
    """Image synthétique avec des murs épais (profil ancien)."""
    image = np.zeros((size_px, size_px), dtype=np.float32)
    spacing = size_px // (n_walls + 1)
    for i in range(1, n_walls + 1):
        pos = i * spacing
        image[pos: pos + wall_thickness_px, 20: size_px - 20] = 10.0
    return image


def _make_open_space_image(
    size_px: int = 400,
    wall_thickness_px: int = 10,
    n_walls: int = 2,
) -> np.ndarray:
    """Image synthétique avec peu de murs (profil bureau)."""
    image = np.zeros((size_px, size_px), dtype=np.float32)
    spacing = size_px // (n_walls + 1)
    for i in range(1, n_walls + 1):
        pos = i * spacing
        image[pos: pos + wall_thickness_px, 20: size_px - 20] = 5.0
    return image


# ---------------------------------------------------------------------------
# Tests load_profile
# ---------------------------------------------------------------------------

class TestLoadProfile:
    def test_load_recent(self) -> None:
        data = load_profile("recent")
        assert isinstance(data, dict)
        assert "regularization" in data

    def test_load_ancien(self) -> None:
        data = load_profile("ancien")
        assert isinstance(data, dict)
        assert "segment_fusion" in data

    def test_load_bureau(self) -> None:
        data = load_profile("bureau")
        assert isinstance(data, dict)
        assert "hough" in data

    def test_unknown_profile_raises(self) -> None:
        with pytest.raises(ValueError, match="inconnu"):
            load_profile("inexistant")

    def test_recent_snap_tolerance_stricter_than_default(self) -> None:
        """Le profil 'recent' a une tolérance angulaire plus stricte que le défaut (5°)."""
        data = load_profile("recent")
        assert data["regularization"]["snap_tolerance_deg"] < 5.0

    def test_ancien_snap_tolerance_more_lenient_than_default(self) -> None:
        """Le profil 'ancien' a une tolérance angulaire plus souple (> 5°)."""
        data = load_profile("ancien")
        assert data["regularization"]["snap_tolerance_deg"] > 5.0

    def test_ancien_kernel_larger_than_recent(self) -> None:
        """Le profil 'ancien' a un noyau morphologique plus grand que 'recent'."""
        recent = load_profile("recent")
        ancien = load_profile("ancien")
        assert ancien["morphology"]["kernel_size"] > recent["morphology"]["kernel_size"]

    def test_all_profiles_loadable(self) -> None:
        for name in AVAILABLE_PROFILES:
            data = load_profile(name)
            assert isinstance(data, dict)
            assert len(data) > 0

    def test_profile_does_not_contain_all_sections(self) -> None:
        """Les profils sont des surcharges partielles, pas une config complète."""
        data = load_profile("recent")
        # Les profils ne doivent pas redéfinir 'dxf' ou 'density_map' inutilement
        assert "dxf" not in data


# ---------------------------------------------------------------------------
# Tests apply_profile
# ---------------------------------------------------------------------------

class TestApplyProfile:
    def test_apply_recent_changes_snap_tolerance(self) -> None:
        from scan2plan.config import ScanConfig
        cfg = ScanConfig()
        default_snap = cfg.regularization.snap_tolerance_deg
        apply_profile(cfg, "recent")
        assert cfg.regularization.snap_tolerance_deg != default_snap
        assert cfg.regularization.snap_tolerance_deg < default_snap

    def test_apply_ancien_changes_angle_tolerance(self) -> None:
        from scan2plan.config import ScanConfig
        cfg = ScanConfig()
        apply_profile(cfg, "ancien")
        assert cfg.segment_fusion.angle_tolerance_deg > 3.0  # > défaut

    def test_apply_profile_returns_config(self) -> None:
        """apply_profile retourne la config pour permettre le chaining."""
        from scan2plan.config import ScanConfig
        cfg = ScanConfig()
        returned = apply_profile(cfg, "recent")
        assert returned is cfg

    def test_apply_bureau_reduces_hough_threshold(self) -> None:
        from scan2plan.config import ScanConfig
        cfg = ScanConfig()
        apply_profile(cfg, "bureau")
        # Le profil bureau utilise threshold: 40, défaut = 50
        assert cfg.hough.threshold < 50

    def test_non_overridden_values_preserved(self) -> None:
        """Les valeurs non surchargées par le profil restent inchangées."""
        from scan2plan.config import ScanConfig
        cfg = ScanConfig()
        original_voxel = cfg.preprocessing.voxel_size
        # Le profil bureau surcharge voxel_size, recent ne le surcharge pas
        apply_profile(cfg, "recent")
        # Selon le profil recent.yaml, voxel_size = 0.005 (même que défaut)
        assert cfg.preprocessing.voxel_size == pytest.approx(original_voxel, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests auto_calibrate
# ---------------------------------------------------------------------------

class TestAutoCalibrateRecent:
    def test_thin_walls_low_ceiling_suggests_recent(self) -> None:
        """Murs fins + hauteur standard → profil 'recent'."""
        image = _make_thin_wall_image()
        result = auto_calibrate(image, ceiling_height_m=2.50, resolution_m=0.005)
        assert result.suggested_profile == "recent", (
            f"Profil attendu : 'recent', obtenu : '{result.suggested_profile}'. "
            f"Épaisseur médiane : {result.median_wall_thickness_m:.3f}m, "
            f"ratio ouvert : {result.open_space_ratio:.2%}"
        )

    def test_recent_result_has_high_confidence(self) -> None:
        image = _make_thin_wall_image()
        result = auto_calibrate(image, ceiling_height_m=2.50, resolution_m=0.005)
        assert result.confidence > 0.3


class TestAutoCalibrateAncien:
    def test_thick_walls_high_ceiling_suggests_ancien(self) -> None:
        """Murs épais + hauts plafonds → profil 'ancien'."""
        image = _make_thick_wall_image()
        result = auto_calibrate(image, ceiling_height_m=3.20, resolution_m=0.005)
        assert result.suggested_profile == "ancien", (
            f"Profil attendu : 'ancien', obtenu : '{result.suggested_profile}'. "
            f"Épaisseur médiane : {result.median_wall_thickness_m:.3f}m"
        )

    def test_ancien_result_type(self) -> None:
        image = _make_thick_wall_image()
        result = auto_calibrate(image, ceiling_height_m=3.20, resolution_m=0.005)
        assert isinstance(result, CalibrationResult)


class TestAutoCalibrateBureau:
    def test_sparse_walls_suggests_bureau(self) -> None:
        """Peu de murs (grands espaces) → profil 'bureau'."""
        image = _make_open_space_image()
        result = auto_calibrate(image, ceiling_height_m=2.60, resolution_m=0.005)
        assert result.suggested_profile == "bureau", (
            f"Profil attendu : 'bureau', obtenu : '{result.suggested_profile}'. "
            f"Ratio occupé : {result.open_space_ratio:.2%}"
        )


class TestAutoCalibrateGeneral:
    def test_returns_calibration_result(self) -> None:
        image = _make_thin_wall_image()
        result = auto_calibrate(image, ceiling_height_m=2.50, resolution_m=0.005)
        assert isinstance(result, CalibrationResult)

    def test_result_has_all_fields(self) -> None:
        image = _make_thin_wall_image()
        result = auto_calibrate(image, ceiling_height_m=2.50, resolution_m=0.005)
        assert result.suggested_profile in AVAILABLE_PROFILES
        assert 0.0 <= result.confidence <= 1.0
        assert result.ceiling_height_m == pytest.approx(2.50)
        assert result.median_wall_thickness_m >= 0.0
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    def test_open_space_ratio_correct(self) -> None:
        """Image à moitié occupée → ratio ≈ 0.5."""
        image = np.zeros((100, 100), dtype=np.float32)
        image[:50, :] = 1.0  # moitié haute occupée
        result = auto_calibrate(image, ceiling_height_m=2.50, resolution_m=0.005)
        assert result.open_space_ratio == pytest.approx(0.5, abs=0.01)

    def test_empty_image_does_not_crash(self) -> None:
        """Image vide → retourne un résultat valide sans exception."""
        image = np.zeros((200, 200), dtype=np.float32)
        result = auto_calibrate(image, ceiling_height_m=2.50, resolution_m=0.005)
        assert result.suggested_profile in AVAILABLE_PROFILES


# ---------------------------------------------------------------------------
# Tests calibrate_slice_heights
# ---------------------------------------------------------------------------

class TestCalibrateSliceHeights:
    def test_standard_ceiling_250(self) -> None:
        """Plafond 2.50 m → 3 hauteurs plausibles."""
        heights = calibrate_slice_heights(2.50)
        assert len(heights) == 3
        # Haute > médiane > basse
        assert heights[0] > heights[1] > heights[2]

    def test_heights_in_plausible_range(self) -> None:
        for h_ceiling in [2.20, 2.50, 2.70, 3.00, 3.50]:
            heights = calibrate_slice_heights(h_ceiling)
            assert 1.80 <= heights[0] <= 3.50
            assert 0.90 <= heights[1] <= 1.50
            assert 0.10 <= heights[2] <= 0.40

    def test_low_ceiling_returns_lower_heights(self) -> None:
        """Plafond bas → slice haute plus basse."""
        h_low = calibrate_slice_heights(2.20)
        h_high = calibrate_slice_heights(3.50)
        assert h_low[0] < h_high[0]

    def test_returns_list_of_three(self) -> None:
        heights = calibrate_slice_heights(2.80)
        assert isinstance(heights, list)
        assert len(heights) == 3
        assert all(isinstance(h, float) for h in heights)

    def test_rounded_to_5cm(self) -> None:
        """Les hauteurs sont arrondies au multiple de 5 cm le plus proche."""
        heights = calibrate_slice_heights(2.50)
        for h in heights:
            assert abs(round(h / 0.05) * 0.05 - h) < 1e-6

    def test_high_ceiling_320(self) -> None:
        """Plafond 3.20 m (ancien) → slice haute ≥ 2.50 m."""
        heights = calibrate_slice_heights(3.20)
        assert heights[0] >= 2.50
