"""Tests unitaires pour slicing/density_map.py et utils/coordinate.py."""

from __future__ import annotations

import numpy as np
import pytest

from scan2plan.slicing.density_map import DensityMapResult, create_density_map
from scan2plan.utils.coordinate import metric_to_pixel, pixel_to_metric, segments_pixel_to_metric


# ---------------------------------------------------------------------------
# Fixture locale : une grille 2D (XY seulement)
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_room_2d(simple_room_points: np.ndarray) -> np.ndarray:
    """Projection XY du nuage synthétique."""
    return simple_room_points[:, :2]


# ---------------------------------------------------------------------------
# Tests DensityMapResult
# ---------------------------------------------------------------------------

class TestCreateDensityMap:
    def test_returns_density_map_result_instance(self, simple_room_2d: np.ndarray) -> None:
        result = create_density_map(simple_room_2d, resolution=0.05)
        assert isinstance(result, DensityMapResult)

    def test_image_is_2d(self, simple_room_2d: np.ndarray) -> None:
        result = create_density_map(simple_room_2d, resolution=0.05)
        assert result.image.ndim == 2

    def test_image_dtype_uint16(self, simple_room_2d: np.ndarray) -> None:
        result = create_density_map(simple_room_2d, resolution=0.05)
        assert result.image.dtype == np.uint16

    def test_image_shape_matches_width_height(self, simple_room_2d: np.ndarray) -> None:
        """result.image.shape doit être (height, width)."""
        result = create_density_map(simple_room_2d, resolution=0.05)
        assert result.image.shape == (result.height, result.width)

    def test_nonzero_pixels_exist(self, simple_room_2d: np.ndarray) -> None:
        """Il doit y avoir des pixels non nuls dans la density map."""
        result = create_density_map(simple_room_2d, resolution=0.05)
        assert result.image.max() > 0

    def test_x_min_below_data_min(self, simple_room_2d: np.ndarray) -> None:
        """x_min doit être inférieur au minimum des X (marge incluse)."""
        result = create_density_map(simple_room_2d, resolution=0.05)
        assert result.x_min <= simple_room_2d[:, 0].min()

    def test_resolution_stored(self, simple_room_2d: np.ndarray) -> None:
        res = 0.02
        result = create_density_map(simple_room_2d, resolution=res)
        assert result.resolution == pytest.approx(res)

    def test_invalid_resolution_raises(self, simple_room_2d: np.ndarray) -> None:
        with pytest.raises(ValueError, match="resolution"):
            create_density_map(simple_room_2d, resolution=0.0)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            create_density_map(np.zeros((0, 2)), resolution=0.01)

    def test_point_appears_in_correct_column(self) -> None:
        """Un point au centre doit tomber dans la colonne attendue."""
        pts = np.array([[2.0, 2.0]], dtype=np.float64)
        result = create_density_map(pts, resolution=1.0, margin=0.0)
        # Avec x_min=2.0, résolution=1.0 → col 0
        assert result.image[:, 0].max() > 0

    def test_y_axis_inversion(self) -> None:
        """Un point avec Y élevé doit apparaître dans les premières lignes (row bas = Y haut)."""
        # Deux points avec des Y très différents
        pt_high_y = np.array([[0.0, 10.0]], dtype=np.float64)
        pt_low_y = np.array([[0.0, 0.0]], dtype=np.float64)

        result_high = create_density_map(pt_high_y, resolution=1.0, margin=0.0)
        result_low = create_density_map(pt_low_y, resolution=1.0, margin=0.0)

        # pt_high_y → row 0 (Y haut = row haut en image)
        # pt_low_y → row max (Y bas = row bas en image)
        assert result_high.image[0, 0] > 0, "Y élevé doit être en row 0"
        assert result_low.image[-1, 0] > 0, "Y bas doit être en dernière row"


# ---------------------------------------------------------------------------
# Tests pixel_to_metric / metric_to_pixel
# ---------------------------------------------------------------------------

class TestCoordinateConversion:
    def test_roundtrip_metric_to_pixel_to_metric(self) -> None:
        """metric_to_pixel → pixel_to_metric doit redonner les coordonnées d'origine (± 0.5 px)."""
        x_min, y_min, resolution, image_height = 0.0, 0.0, 0.005, 1000

        for x, y in [(1.0, 1.5), (2.3, 0.7), (0.025, 2.495)]:
            col, row = metric_to_pixel(x, y, x_min, y_min, resolution, image_height)
            x_back, y_back = pixel_to_metric(col, row, x_min, y_min, resolution, image_height)
            assert abs(x_back - x) <= resolution, f"Erreur X pour ({x}, {y})"
            assert abs(y_back - y) <= resolution, f"Erreur Y pour ({x}, {y})"

    def test_x_axis_direct(self) -> None:
        """L'axe X n'est pas inversé : px croît avec x."""
        x_min, y_min, resolution, image_height = 0.0, 0.0, 1.0, 10
        col, _ = metric_to_pixel(3.0, 0.0, x_min, y_min, resolution, image_height)
        assert col == 3

    def test_y_axis_inverted(self) -> None:
        """L'axe Y image est inversé : Y métrique bas → row élevé."""
        x_min, y_min, resolution, image_height = 0.0, 0.0, 1.0, 10
        _, row_low = metric_to_pixel(0.0, 0.0, x_min, y_min, resolution, image_height)
        _, row_high = metric_to_pixel(0.0, 9.0, x_min, y_min, resolution, image_height)
        # Y=0 (bas) → row élevé ; Y=9 (haut) → row 0
        assert row_low > row_high

    def test_pixel_to_metric_origin(self) -> None:
        """pixel_to_metric(0, H-1) doit donner (x_min, y_min)."""
        x_min, y_min, resolution, image_height = 1.0, 2.0, 0.005, 100
        x, y = pixel_to_metric(0, image_height - 1, x_min, y_min, resolution, image_height)
        assert x == pytest.approx(x_min)
        assert y == pytest.approx(y_min)


# ---------------------------------------------------------------------------
# Tests segments_pixel_to_metric
# ---------------------------------------------------------------------------

class TestSegmentsPixelToMetric:
    def test_shape_preserved(self) -> None:
        segs = np.array([[10, 20, 30, 40]], dtype=np.float64)
        result = segments_pixel_to_metric(segs, x_min=0.0, y_min=0.0,
                                          resolution=0.005, image_height=100)
        assert result.shape == (1, 4)

    def test_x_coordinates_correct(self) -> None:
        """Les colonnes X doivent être converties correctement (pas d'inversion)."""
        segs = np.array([[10, 50, 30, 50]], dtype=np.float64)
        result = segments_pixel_to_metric(segs, x_min=0.0, y_min=0.0,
                                          resolution=1.0, image_height=100)
        assert result[0, 0] == pytest.approx(10.0)
        assert result[0, 2] == pytest.approx(30.0)

    def test_y_coordinates_inverted(self) -> None:
        """Les coordonnées Y doivent être inversées par rapport à l'axe image."""
        segs = np.array([[0, 0, 0, 99]], dtype=np.float64)  # row 0 et row 99
        result = segments_pixel_to_metric(segs, x_min=0.0, y_min=0.0,
                                          resolution=1.0, image_height=100)
        # row 0 → py_geo = 99 → y = 99.0
        # row 99 → py_geo = 0 → y = 0.0
        assert result[0, 1] == pytest.approx(99.0)
        assert result[0, 3] == pytest.approx(0.0)

    def test_roundtrip_with_metric_to_pixel(self) -> None:
        """Un segment créé depuis des coordonnées métriques doit se retrouver intact."""
        x_min, y_min, resolution = 0.0, 0.0, 0.005
        image_height = 1000

        x1, y1, x2, y2 = 1.0, 2.0, 3.0, 4.0
        col1, row1 = metric_to_pixel(x1, y1, x_min, y_min, resolution, image_height)
        col2, row2 = metric_to_pixel(x2, y2, x_min, y_min, resolution, image_height)
        segs_px = np.array([[col1, row1, col2, row2]], dtype=np.float64)

        result = segments_pixel_to_metric(segs_px, x_min, y_min, resolution, image_height)
        assert abs(result[0, 0] - x1) <= resolution
        assert abs(result[0, 1] - y1) <= resolution
        assert abs(result[0, 2] - x2) <= resolution
        assert abs(result[0, 3] - y2) <= resolution
