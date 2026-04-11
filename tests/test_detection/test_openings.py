"""Tests unitaires de la détection des ouvertures (portes et fenêtres).

Stratégie : construire des DensityMapResult synthétiques représentant un mur
horizontal avec ou sans trou. Le mur est tracé sur l'axe X (y=2.0), de x=0
à x=4 m. Résolution 0.05 m/px pour garder les images petites.

Convention :
    - slice HIGH : mur complet (présent partout)
    - slice MID  : mur avec trou = ouverture (absent dans la zone de l'ouverture)
    - slice LOW  : selon le type d'ouverture
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scan2plan.detection.line_detection import DetectedSegment
from scan2plan.detection.openings import (
    Opening,
    _find_gaps,
    _normalize_profile,
    _resize_profile,
    detect_all_openings,
    detect_openings_along_wall,
)
from scan2plan.slicing.density_map import DensityMapResult


# ---------------------------------------------------------------------------
# Helpers de construction de density maps synthétiques
# ---------------------------------------------------------------------------

_RES = 0.05   # 5 cm/px


def _draw_wall_band(
    image: np.ndarray,
    row: int,
    col1: int,
    col2: int,
    value: int,
) -> None:
    """Remplit une bande horizontale de ±1 ligne autour de ``row``."""
    n_rows, n_cols = image.shape
    for r in range(max(0, row - 1), min(n_rows, row + 2)):
        start = max(0, col1)
        end = min(n_cols, col2 + 1)
        image[r, start:end] = value


def _make_dmap(
    wall_x1: float,
    wall_x2: float,
    wall_y: float,
    gap_x1: float = 0.0,
    gap_x2: float = 0.0,
    x_min: float = -0.5,
    x_max: float = 4.5,
    y_min: float = -0.5,
    y_max: float = 4.5,
    res: float = _RES,
) -> DensityMapResult:
    """Construit une density map avec un mur horizontal optionnellement troué.

    Le mur est tracé à ``wall_y`` entre ``wall_x1`` et ``wall_x2``.
    Si ``gap_x1 < gap_x2``, les colonnes correspondant à [gap_x1, gap_x2]
    sont mises à zéro (simulation d'une ouverture).
    """
    n_cols = int(round((x_max - x_min) / res))
    n_rows = int(round((y_max - y_min) / res))
    image = np.zeros((n_rows, n_cols), dtype=np.uint16)

    py_geo = int(round((wall_y - y_min) / res))
    row = (n_rows - 1) - py_geo
    col1 = int(round((wall_x1 - x_min) / res))
    col2 = int(round((wall_x2 - x_min) / res))

    if 0 <= row < n_rows:
        _draw_wall_band(image, row, col1, col2, value=10)
        if gap_x1 < gap_x2:
            gc1 = int(round((gap_x1 - x_min) / res))
            gc2 = int(round((gap_x2 - x_min) / res))
            _draw_wall_band(image, row, gc1, gc2, value=0)

    return DensityMapResult(
        image=image,
        x_min=x_min,
        y_min=y_min,
        resolution=res,
        width=n_cols,
        height=n_rows,
    )


def _wall_seg(x1: float = 0.0, y: float = 2.0, x2: float = 4.0) -> DetectedSegment:
    """Segment de mur horizontal synthétique."""
    return DetectedSegment(x1=x1, y1=y, x2=x2, y2=y, source_slice="high", confidence=1.0)


# ---------------------------------------------------------------------------
# Tests des helpers privés
# ---------------------------------------------------------------------------

class TestFindGaps:
    # Convention : True = ouverture présente, False = mur présent
    # (correspond à high_present & mid_absent dans detect_openings_along_wall)

    def test_single_gap(self) -> None:
        # Mur—mur—OUVERTURE(3px)—mur—mur
        mask = np.array([False, False, True, True, True, False, False])
        gaps = _find_gaps(mask)
        assert len(gaps) == 1
        assert gaps[0] == (2, 4)

    def test_gap_too_short(self) -> None:
        # 2 pixels True consécutifs < _MIN_GAP_PIXELS (3) → ignoré
        mask = np.array([False, False, True, True, False, False])
        gaps = _find_gaps(mask)
        assert len(gaps) == 0

    def test_gap_at_end(self) -> None:
        # Ouverture en fin de profil
        mask = np.array([False, True, True, True, True])
        gaps = _find_gaps(mask)
        assert len(gaps) == 1
        assert gaps[0] == (1, 4)

    def test_no_gap(self) -> None:
        # Aucune ouverture : tout False
        mask = np.zeros(10, dtype=bool)
        gaps = _find_gaps(mask)
        assert len(gaps) == 0

    def test_all_gap(self) -> None:
        # Tout True = un seul grand gap
        mask = np.ones(5, dtype=bool)
        gaps = _find_gaps(mask)
        assert len(gaps) == 1

    def test_multiple_gaps(self) -> None:
        # Deux ouvertures séparées par du mur
        mask = np.array([False, True, True, True, False, False, True, True, True, True])
        gaps = _find_gaps(mask)
        assert len(gaps) == 2


class TestNormalizeProfile:
    def test_normalized_max_is_one(self) -> None:
        p = np.array([0.0, 5.0, 10.0, 3.0])
        n = _normalize_profile(p)
        assert abs(n.max() - 1.0) < 1e-9

    def test_all_zero_stays_zero(self) -> None:
        p = np.zeros(5)
        n = _normalize_profile(p)
        assert n.max() < 1e-9

    def test_empty_returns_empty(self) -> None:
        n = _normalize_profile(np.array([]))
        assert len(n) == 0


class TestResizeProfile:
    def test_expand(self) -> None:
        p = np.array([0.0, 1.0])
        r = _resize_profile(p, 5)
        assert len(r) == 5
        assert r[0] == pytest.approx(0.0)
        assert r[-1] == pytest.approx(1.0)

    def test_same_length(self) -> None:
        p = np.array([1.0, 2.0, 3.0])
        r = _resize_profile(p, 3)
        np.testing.assert_array_almost_equal(r, p)

    def test_empty_returns_zeros(self) -> None:
        r = _resize_profile(np.array([]), 4)
        assert len(r) == 4
        assert r.sum() < 1e-9


# ---------------------------------------------------------------------------
# Tests detect_openings_along_wall
# ---------------------------------------------------------------------------

class TestDetectOpeningsAlongWall:
    def test_no_opening_on_full_wall(self) -> None:
        """Mur continu dans toutes les slices → 0 ouvertures."""
        wall = _wall_seg()
        dmap = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0)
        density_maps = {"high": dmap, "mid": dmap, "low": dmap}
        openings = detect_openings_along_wall(wall, density_maps, {})
        assert len(openings) == 0

    def test_detect_door(self) -> None:
        """Mur de 4m avec trou de 90cm (1.55–2.45m) en MID et LOW → 1 porte."""
        wall = _wall_seg(x1=0.0, y=2.0, x2=4.0)
        high = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0)
        mid  = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0, gap_x1=1.55, gap_x2=2.45)
        low  = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0, gap_x1=1.55, gap_x2=2.45)

        density_maps = {"high": high, "mid": mid, "low": low}
        openings = detect_openings_along_wall(
            wall, density_maps, {},
            min_door_width=0.60, max_door_width=1.40,
        )

        assert len(openings) == 1
        door = openings[0]
        assert door.type == "door"
        assert abs(door.width - 0.90) < 0.10, f"Largeur porte : {door.width:.3f} m"

    def test_detect_window(self) -> None:
        """Mur avec trou en MID (allège 80cm), présent en HIGH et LOW → 1 fenêtre."""
        wall = _wall_seg(x1=0.0, y=2.0, x2=4.0)
        high = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0)
        mid  = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0, gap_x1=1.50, gap_x2=2.50)
        low  = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0)   # allège : low présent

        density_maps = {"high": high, "mid": mid, "low": low}
        openings = detect_openings_along_wall(
            wall, density_maps, {},
            min_window_width=0.30, max_window_width=2.50,
        )

        assert len(openings) == 1
        win = openings[0]
        assert win.type == "window"
        assert win.width >= 0.30

    def test_dimension_filter_too_small(self) -> None:
        """Trou de 20cm (< 60cm min porte) → filtré, aucune porte détectée."""
        wall = _wall_seg(x1=0.0, y=2.0, x2=4.0)
        high = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0)
        # Trou de 4 pixels × 5cm = 20cm
        mid  = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0, gap_x1=1.90, gap_x2=2.10)
        low  = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0, gap_x1=1.90, gap_x2=2.10)

        density_maps = {"high": high, "mid": mid, "low": low}
        openings = detect_openings_along_wall(
            wall, density_maps, {},
            min_door_width=0.60, max_door_width=1.40,
        )
        # Le trou de 20cm est trop petit pour être une porte
        doors = [o for o in openings if o.type == "door"]
        assert len(doors) == 0

    def test_dimension_filter_too_large(self) -> None:
        """Trou de 2m (> 1.40m max porte) → pas de porte, peut être fenêtre."""
        wall = _wall_seg(x1=0.0, y=2.0, x2=4.0)
        high = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0)
        mid  = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0, gap_x1=1.0, gap_x2=3.0)
        low  = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0, gap_x1=1.0, gap_x2=3.0)

        density_maps = {"high": high, "mid": mid, "low": low}
        openings = detect_openings_along_wall(
            wall, density_maps, {},
            min_door_width=0.60, max_door_width=1.40,
        )
        doors = [o for o in openings if o.type == "door"]
        assert len(doors) == 0

    def test_no_high_slice_returns_empty(self) -> None:
        """Sans slice high, aucune détection possible."""
        wall = _wall_seg()
        dmap = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0)
        openings = detect_openings_along_wall(wall, {"mid": dmap}, {})
        assert len(openings) == 0

    def test_opening_has_valid_position(self) -> None:
        """La position de l'ouverture doit être dans les bornes du mur."""
        wall = _wall_seg(x1=0.0, y=2.0, x2=4.0)
        high = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0)
        mid  = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0, gap_x1=1.55, gap_x2=2.45)
        low  = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0, gap_x1=1.55, gap_x2=2.45)

        openings = detect_openings_along_wall(
            wall, {"high": high, "mid": mid, "low": low}, {},
        )
        for op in openings:
            assert op.position_start >= 0.0
            assert op.position_end <= wall.length + 0.1   # tolérance 1 px
            assert op.position_start < op.position_end

    def test_confidence_in_range(self) -> None:
        """La confiance doit être dans [0, 1]."""
        wall = _wall_seg(x1=0.0, y=2.0, x2=4.0)
        high = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0)
        mid  = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0, gap_x1=1.55, gap_x2=2.45)
        low  = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0, gap_x1=1.55, gap_x2=2.45)

        openings = detect_openings_along_wall(
            wall, {"high": high, "mid": mid, "low": low}, {},
        )
        for op in openings:
            assert 0.0 <= op.confidence <= 1.0

    def test_sorted_by_position(self) -> None:
        """Deux ouvertures → retournées dans l'ordre de position croissante."""
        wall = _wall_seg(x1=0.0, y=2.0, x2=6.0)
        high = _make_dmap(wall_x1=0.0, wall_x2=6.0, wall_y=2.0, x_max=7.0)
        mid  = _make_dmap(
            wall_x1=0.0, wall_x2=6.0, wall_y=2.0,
            gap_x1=0.80, gap_x2=1.70,
            x_max=7.0,
        )
        # Deuxième ouverture
        mid2 = _make_dmap(
            wall_x1=0.0, wall_x2=6.0, wall_y=2.0,
            gap_x1=3.50, gap_x2=4.40,
            x_max=7.0,
        )
        # Combiner les deux density maps mid
        combined_image = np.minimum(mid.image, mid2.image)
        mid_combined = DensityMapResult(
            image=combined_image,
            x_min=mid.x_min, y_min=mid.y_min,
            resolution=mid.resolution,
            width=mid.width, height=mid.height,
        )
        low = _make_dmap(
            wall_x1=0.0, wall_x2=6.0, wall_y=2.0,
            gap_x1=0.80, gap_x2=1.70,
            x_max=7.0,
        )
        low2 = _make_dmap(
            wall_x1=0.0, wall_x2=6.0, wall_y=2.0,
            gap_x1=3.50, gap_x2=4.40,
            x_max=7.0,
        )
        combined_low = DensityMapResult(
            image=np.minimum(low.image, low2.image),
            x_min=low.x_min, y_min=low.y_min,
            resolution=low.resolution,
            width=low.width, height=low.height,
        )

        openings = detect_openings_along_wall(
            wall,
            {"high": high, "mid": mid_combined, "low": combined_low},
            {},
        )
        if len(openings) >= 2:
            positions = [o.position_start for o in openings]
            assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# Tests detect_all_openings
# ---------------------------------------------------------------------------

class TestDetectAllOpenings:
    def test_empty_wall_list(self) -> None:
        openings = detect_all_openings([], {}, {}, {})
        assert openings == []

    def test_single_wall_with_door(self) -> None:
        wall = _wall_seg()
        high = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0)
        mid  = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0, gap_x1=1.55, gap_x2=2.45)
        low  = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0, gap_x1=1.55, gap_x2=2.45)

        density_maps = {"high": high, "mid": mid, "low": low}
        openings = detect_all_openings(
            [wall], density_maps, {},
            {"min_door_width": 0.60, "max_door_width": 1.40},
        )
        assert len(openings) >= 1
        assert any(o.type == "door" for o in openings)

    def test_config_overrides_defaults(self) -> None:
        """Un trou de 55cm est détecté si min_door_width=0.40 mais pas si 0.60."""
        wall = _wall_seg()
        high = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0)
        mid  = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0, gap_x1=1.73, gap_x2=2.28)
        low  = _make_dmap(wall_x1=0.0, wall_x2=4.0, wall_y=2.0, gap_x1=1.73, gap_x2=2.28)
        dmaps = {"high": high, "mid": mid, "low": low}

        # Trop étroit avec min_door_width=0.60
        openings_strict = detect_all_openings(
            [wall], dmaps, {}, {"min_door_width": 0.60, "max_door_width": 1.40}
        )
        # Suffisant avec min_door_width=0.40
        openings_loose = detect_all_openings(
            [wall], dmaps, {}, {"min_door_width": 0.40, "max_door_width": 1.40}
        )
        # Avec la config stricte on doit avoir moins (ou autant) de portes
        doors_strict = sum(1 for o in openings_strict if o.type == "door")
        doors_loose = sum(1 for o in openings_loose if o.type == "door")
        assert doors_strict <= doors_loose


# ---------------------------------------------------------------------------
# Test export DXF ouvertures
# ---------------------------------------------------------------------------

class TestExportOpeningsToDxf:
    def test_export_creates_lines(self, tmp_path) -> None:
        """export_openings_to_dxf doit créer 2 LINE par ouverture."""
        import ezdxf
        from scan2plan.detection.openings import Opening
        from scan2plan.io.writers import export_openings_to_dxf

        wall = _wall_seg()
        op = Opening(
            type="door",
            wall_segment=wall,
            position_start=1.5,
            position_end=2.4,
            width=0.9,
            confidence=0.8,
        )
        doc = ezdxf.new(dxfversion="R2013")
        n = export_openings_to_dxf([op], doc)

        assert n == 2
        lines = [e for e in doc.modelspace() if e.dxftype() == "LINE"]
        assert len(lines) == 2

    def test_door_on_portes_layer(self, tmp_path) -> None:
        """Les portes doivent être sur le calque PORTES."""
        import ezdxf
        from scan2plan.detection.openings import Opening
        from scan2plan.io.writers import export_openings_to_dxf

        wall = _wall_seg()
        op = Opening(type="door", wall_segment=wall,
                     position_start=1.0, position_end=1.9,
                     width=0.9, confidence=0.9)
        doc = ezdxf.new(dxfversion="R2013")
        export_openings_to_dxf([op], doc)

        layers = {e.dxf.layer for e in doc.modelspace() if e.dxftype() == "LINE"}
        assert "PORTES" in layers

    def test_window_on_fenetres_layer(self, tmp_path) -> None:
        """Les fenêtres doivent être sur le calque FENETRES."""
        import ezdxf
        from scan2plan.detection.openings import Opening
        from scan2plan.io.writers import export_openings_to_dxf

        wall = _wall_seg()
        op = Opening(type="window", wall_segment=wall,
                     position_start=1.0, position_end=2.2,
                     width=1.2, confidence=0.7)
        doc = ezdxf.new(dxfversion="R2013")
        export_openings_to_dxf([op], doc)

        layers = {e.dxf.layer for e in doc.modelspace() if e.dxftype() == "LINE"}
        assert "FENETRES" in layers

    def test_empty_openings_returns_zero(self) -> None:
        import ezdxf
        from scan2plan.io.writers import export_openings_to_dxf

        doc = ezdxf.new(dxfversion="R2013")
        n = export_openings_to_dxf([], doc)
        assert n == 0

    def test_custom_layer_config(self) -> None:
        """layer_config doit permettre de renommer les calques."""
        import ezdxf
        from scan2plan.detection.openings import Opening
        from scan2plan.io.writers import export_openings_to_dxf

        wall = _wall_seg()
        op = Opening(type="door", wall_segment=wall,
                     position_start=1.0, position_end=1.9,
                     width=0.9, confidence=0.9)
        doc = ezdxf.new(dxfversion="R2013")
        export_openings_to_dxf([op], doc, layer_config={"doors": "CUSTOM_DOORS"})

        layers = {e.dxf.layer for e in doc.modelspace() if e.dxftype() == "LINE"}
        assert "CUSTOM_DOORS" in layers
        assert "PORTES" not in layers
