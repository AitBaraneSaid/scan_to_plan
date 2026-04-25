"""Tests unitaires pour light_topology.

Vérifie que snap_endpoints et close_corners font uniquement des corrections
locales de bruit, sans jamais franchir une embrasure architecturale.
"""

from __future__ import annotations

import numpy as np
import pytest

from scan2plan.detection.line_detection import DetectedSegment
from scan2plan.vectorization.light_topology import (
    apply_light_topology,
    close_corners,
    snap_endpoints,
)


def seg(
    x1: float, y1: float, x2: float, y2: float, conf: float = 0.9
) -> DetectedSegment:
    return DetectedSegment(x1=x1, y1=y1, x2=x2, y2=y2, source_slice="high", confidence=conf)


# ---------------------------------------------------------------------------
# Tests snap_endpoints
# ---------------------------------------------------------------------------


class TestSnapEndpoints:
    def test_snap_close_endpoints(self) -> None:
        """2 segments perpendiculaires avec extrémités à 2 cm → snappées sur le même point."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)     # horizontal, extrémité droite à (1.0, 0.0)
        s2 = seg(1.02, 0.0, 1.02, 1.0)   # vertical,  extrémité basse à (1.02, 0.0)
        result = snap_endpoints([s1, s2], tolerance=0.03)

        assert len(result) == 2
        # Les extrémités proches doivent converger vers le même point
        r1_x2, r1_y2 = result[0].x2, result[0].y2
        r2_x1, r2_y1 = result[1].x1, result[1].y1
        assert abs(r1_x2 - r2_x1) < 0.001
        assert abs(r1_y2 - r2_y1) < 0.001

    def test_no_snap_far_endpoints(self) -> None:
        """2 segments avec extrémités à 15 cm → pas de snap."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)
        s2 = seg(1.15, 0.0, 1.15, 1.0)
        result = snap_endpoints([s1, s2], tolerance=0.03)

        # Les extrémités restent inchangées
        assert result[0].x2 == pytest.approx(1.0)
        assert result[1].x1 == pytest.approx(1.15)

    def test_centroid_is_mean(self) -> None:
        """3 extrémités proches → centroïde = moyenne des 3."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)     # p2 à (1.0, 0.0)
        s2 = seg(1.02, 0.0, 1.02, 1.0)   # p1 à (1.02, 0.0)
        s3 = seg(0.5, 0.01, 1.01, 0.01)  # p2 à (1.01, 0.01)
        result = snap_endpoints([s1, s2, s3], tolerance=0.03)

        # Les 3 extrémités proches convergent vers leur centroïde
        x_expected = (1.0 + 1.02 + 1.01) / 3.0
        y_expected = (0.0 + 0.0 + 0.01) / 3.0
        assert abs(result[0].x2 - x_expected) < 0.01
        assert abs(result[1].x1 - x_expected) < 0.01
        assert abs(result[0].y2 - y_expected) < 0.01

    def test_segment_not_destroyed(self) -> None:
        """Après snap, aucun segment ne fait moins de 1 cm."""
        # Segment déjà court (5 cm) avec extrémité proche d'une autre
        s1 = seg(0.0, 0.0, 0.05, 0.0)
        s2 = seg(0.04, 0.0, 1.0, 0.0)
        result = snap_endpoints([s1, s2], tolerance=0.03)
        for r in result:
            assert r.length >= 0.01

    def test_empty_input(self) -> None:
        """Liste vide → liste vide."""
        assert snap_endpoints([]) == []

    def test_single_segment_unchanged(self) -> None:
        """Un seul segment → retourné inchangé."""
        s = seg(0.0, 0.0, 3.0, 0.0)
        result = snap_endpoints([s], tolerance=0.03)
        assert len(result) == 1
        assert result[0].x1 == pytest.approx(0.0)
        assert result[0].x2 == pytest.approx(3.0)

    def test_source_slice_preserved(self) -> None:
        """source_slice et confidence sont préservés après snap."""
        s = DetectedSegment(x1=0.0, y1=0.0, x2=1.0, y2=0.0,
                            source_slice="mid", confidence=0.75)
        result = snap_endpoints([s])
        assert result[0].source_slice == "mid"
        assert result[0].confidence == pytest.approx(0.75)

    def test_already_connected_unchanged(self) -> None:
        """Extrémités déjà au même point → pas de perturbation."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)
        s2 = seg(1.0, 0.0, 1.0, 1.0)
        result = snap_endpoints([s1, s2], tolerance=0.03)
        assert result[0].x2 == pytest.approx(1.0)
        assert result[1].x1 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests close_corners
# ---------------------------------------------------------------------------


class TestCloseCorners:
    def test_close_corner(self) -> None:
        """2 segments perpendiculaires avec gap de 5 cm → coin fermé."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)       # horizontal, finit à (1.0, 0.0)
        s2 = seg(1.05, 0.0, 1.05, 1.0)     # vertical, commence à (1.05, 0.0) — gap 5 cm
        result = close_corners([s1, s2], max_extension=0.08, min_angle_deg=60.0)

        assert len(result) == 2
        # Le bout du s1 et le bas du s2 doivent se rejoindre au point d'intersection
        r1_x2 = result[0].x2
        r2_x1 = result[1].x1
        assert abs(r1_x2 - r2_x1) < 0.01

    def test_no_close_parallel(self) -> None:
        """2 segments parallèles à 5 cm → PAS de fermeture (angle < 60°)."""
        s1 = seg(0.0, 0.00, 2.0, 0.00)
        s2 = seg(0.0, 0.05, 2.0, 0.05)
        orig_s1_x2 = s1.x2
        orig_s2_x2 = s2.x2
        result = close_corners([s1, s2], max_extension=0.08, min_angle_deg=60.0)

        # Aucun changement — pas un coin
        assert result[0].x2 == pytest.approx(orig_s1_x2)
        assert result[1].x2 == pytest.approx(orig_s2_x2)

    def test_no_extend_too_far(self) -> None:
        """2 segments perpendiculaires avec gap de 15 cm → PAS de fermeture (> 8 cm)."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)
        s2 = seg(1.15, 0.0, 1.15, 1.0)   # gap 15 cm
        orig_x2 = s1.x2
        result = close_corners([s1, s2], max_extension=0.08)
        assert result[0].x2 == pytest.approx(orig_x2)

    def test_preserve_door_opening(self) -> None:
        """Gap de 80 cm entre deux segments colinéaires → pas de fermeture."""
        # Deux tronçons de mur avec porte de 80 cm entre eux
        left = seg(0.0, 0.0, 1.0, 0.0)    # finit à x=1.0
        right = seg(1.80, 0.0, 3.0, 0.0)  # commence à x=1.80 → gap 80 cm
        # Mur perpendiculaire fermant le coin gauche
        perp = seg(0.0, 0.0, 0.0, 2.0)
        result = close_corners([left, right, perp], max_extension=0.08)

        # Le gap de 80 cm ne doit pas être comblé
        right_result = next(r for r in result if r.x1 > 1.5)
        assert right_result.x1 > 1.70  # la porte reste ouverte

    def test_corner_exact_angle_90(self) -> None:
        """Coin exactement à 90° → fermé."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)
        s2 = seg(1.04, 0.0, 1.04, 1.0)   # gap 4 cm
        result = close_corners([s1, s2], max_extension=0.08)
        # Les extrémités proches doivent se rejoindre
        assert abs(result[0].x2 - result[1].x1) < 0.01

    def test_empty_input(self) -> None:
        """Liste vide → liste vide."""
        assert close_corners([]) == []

    def test_single_segment_unchanged(self) -> None:
        """Un seul segment → retourné inchangé."""
        s = seg(0.0, 0.0, 3.0, 0.0)
        result = close_corners([s])
        assert len(result) == 1
        assert result[0].x2 == pytest.approx(3.0)

    def test_count_stable(self) -> None:
        """Le nombre de segments ne change presque pas (±2 max sur 10 segments)."""
        segments = [
            seg(0.0, 0.0, 4.0, 0.0),
            seg(4.05, 0.0, 4.05, 3.0),
            seg(4.0, 3.05, 0.0, 3.05),
            seg(0.0, 3.0, 0.0, 0.05),
            seg(1.0, 0.0, 1.0, 1.5),
            seg(2.0, 0.0, 2.0, 1.5),
            seg(1.0, 1.55, 2.0, 1.55),
            seg(0.0, 1.5, 0.95, 1.5),
            seg(2.05, 1.5, 4.0, 1.5),
            seg(3.0, 0.0, 3.0, 1.5),
        ]
        result = close_corners(segments, max_extension=0.08)
        assert abs(len(result) - len(segments)) <= 2


# ---------------------------------------------------------------------------
# Tests apply_light_topology
# ---------------------------------------------------------------------------


class TestApplyLightTopology:
    def test_combined_snap_and_close(self) -> None:
        """Snap + fermeture de coin : extrémités à 2 cm et gap résiduel de 3 cm."""
        # Après snap les extrémités seront à ~0.5 cm, puis close_corners ferme le reste
        s1 = seg(0.0, 0.0, 1.0, 0.0)
        s2 = seg(1.02, 0.0, 1.02, 1.0)
        result = apply_light_topology([s1, s2], snap_tolerance=0.03, corner_max_extension=0.08)
        assert len(result) == 2
        # Après snap, les extrémités sont fusionnées : coin fermé
        assert abs(result[0].x2 - result[1].x1) < 0.01

    def test_no_destruction_on_clean_plan(self) -> None:
        """Plan propre (coins déjà fermés) → aucun segment détruit."""
        segments = [
            seg(0.0, 0.0, 4.0, 0.0),
            seg(4.0, 0.0, 4.0, 3.0),
            seg(4.0, 3.0, 0.0, 3.0),
            seg(0.0, 3.0, 0.0, 0.0),
        ]
        result = apply_light_topology(segments)
        assert len(result) == 4

    def test_empty_input(self) -> None:
        """Liste vide → liste vide."""
        assert apply_light_topology([]) == []
