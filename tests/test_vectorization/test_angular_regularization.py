"""Tests unitaires pour angular_regularization.

Vérifie que snap_angles fait UNIQUEMENT de la rotation :
- Pas de fusion, pas de suppression, pas de déplacement transversal.
- Le nombre de segments en sortie est identique à l'entrée.
"""

from __future__ import annotations

import numpy as np
import pytest

from scan2plan.detection.line_detection import DetectedSegment
from scan2plan.vectorization.angular_regularization import (
    detect_dominant_orientations,
    snap_angles,
)


def seg(
    x1: float, y1: float, x2: float, y2: float, conf: float = 0.9
) -> DetectedSegment:
    return DetectedSegment(x1=x1, y1=y1, x2=x2, y2=y2, source_slice="high", confidence=conf)


# ---------------------------------------------------------------------------
# Tests snap_angles
# ---------------------------------------------------------------------------


class TestSnapAngles:
    def test_snap_to_90(self) -> None:
        """Segment à 88° avec dominant à 90° → snappé à 90°."""
        angle_88 = np.deg2rad(88.0)
        s = seg(0.0, 0.0, np.cos(angle_88), np.sin(angle_88))
        dominant = [np.deg2rad(90.0)]
        result = snap_angles([s], dominant, tolerance_deg=5.0)

        result_angle = np.degrees(
            np.arctan2(result[0].y2 - result[0].y1, result[0].x2 - result[0].x1) % np.pi
        )
        assert abs(result_angle - 90.0) < 0.01

    def test_snap_to_0(self) -> None:
        """Segment à 2° avec dominant à 0° → snappé à 0°."""
        angle_2 = np.deg2rad(2.0)
        s = seg(0.0, 0.0, np.cos(angle_2), np.sin(angle_2))
        result = snap_angles([s], [0.0], tolerance_deg=5.0)

        result_angle = np.degrees(
            np.arctan2(result[0].y2 - result[0].y1, result[0].x2 - result[0].x1) % np.pi
        )
        assert abs(result_angle) < 0.01 or abs(result_angle - 180.0) < 0.01

    def test_no_snap_oblique(self) -> None:
        """Segment à 45° avec dominants 0° et 90° → inchangé (oblique)."""
        angle_45 = np.deg2rad(45.0)
        s = seg(0.0, 0.0, np.cos(angle_45), np.sin(angle_45))
        dominants = [0.0, np.deg2rad(90.0)]
        result = snap_angles([s], dominants, tolerance_deg=5.0)

        # Le segment est retourné inchangé
        assert result[0].x1 == pytest.approx(s.x1)
        assert result[0].y1 == pytest.approx(s.y1)
        assert result[0].x2 == pytest.approx(s.x2)
        assert result[0].y2 == pytest.approx(s.y2)

    def test_length_preserved(self) -> None:
        """Le segment garde exactement la même longueur après rotation."""
        s = seg(0.0, 0.0, 2.0, 0.07)  # légèrement incliné, longueur ≈ 2.001 m
        original_length = s.length
        result = snap_angles([s], [0.0], tolerance_deg=5.0)
        assert abs(result[0].length - original_length) < 1e-6

    def test_center_preserved(self) -> None:
        """Le milieu du segment ne bouge pas après rotation."""
        s = seg(1.0, 1.0, 3.0, 1.06)  # centre à (2.0, ~1.03)
        cx_before = (s.x1 + s.x2) / 2.0
        cy_before = (s.y1 + s.y2) / 2.0

        result = snap_angles([s], [0.0], tolerance_deg=5.0)

        cx_after = (result[0].x1 + result[0].x2) / 2.0
        cy_after = (result[0].y1 + result[0].y2) / 2.0

        assert abs(cx_after - cx_before) < 1e-6
        assert abs(cy_after - cy_before) < 1e-6

    def test_no_count_change(self) -> None:
        """Même nombre de segments en entrée et en sortie."""
        segments = [
            seg(0.0, 0.0, 2.0, 0.03),   # ~1°, sera snappé
            seg(0.0, 1.0, 0.0, 3.0),    # vertical, sera snappé
            seg(0.0, 5.0, 2.0, 7.0),    # 45°, pas snappé
            seg(3.0, 0.0, 5.0, 0.0),    # horizontal exact
        ]
        dominants = [0.0, np.deg2rad(90.0)]
        result = snap_angles(segments, dominants, tolerance_deg=5.0)
        assert len(result) == len(segments)

    def test_empty_input(self) -> None:
        """Liste vide → liste vide."""
        assert snap_angles([], [0.0]) == []

    def test_no_dominant_angles(self) -> None:
        """Pas d'angles dominants → retour sans modification."""
        s = seg(0.0, 0.0, 1.0, 0.05)
        result = snap_angles([s], [])
        assert result[0].x1 == pytest.approx(s.x1)
        assert result[0].x2 == pytest.approx(s.x2)

    def test_source_slice_preserved(self) -> None:
        """source_slice et confidence sont préservés après snap."""
        s = DetectedSegment(x1=0.0, y1=0.0, x2=1.0, y2=0.04,
                            source_slice="mid", confidence=0.75)
        result = snap_angles([s], [0.0], tolerance_deg=5.0)
        assert result[0].source_slice == "mid"
        assert result[0].confidence == pytest.approx(0.75)

    def test_horizontal_segment_unchanged(self) -> None:
        """Segment parfaitement horizontal avec dominant 0° → inchangé géométriquement."""
        s = seg(0.0, 0.0, 3.0, 0.0)
        result = snap_angles([s], [0.0], tolerance_deg=5.0)
        assert result[0].x1 == pytest.approx(0.0)
        assert result[0].y1 == pytest.approx(0.0, abs=1e-9)
        assert result[0].x2 == pytest.approx(3.0)
        assert result[0].y2 == pytest.approx(0.0, abs=1e-9)

    def test_just_within_tolerance(self) -> None:
        """Segment à exactement (tolérance - ε)° → snappé."""
        angle = np.deg2rad(4.9)   # tolérance=5°, diff=4.9° → snappé
        s = seg(0.0, 0.0, np.cos(angle), np.sin(angle))
        result = snap_angles([s], [0.0], tolerance_deg=5.0)
        result_angle = float(np.degrees(
            np.arctan2(result[0].y2 - result[0].y1, result[0].x2 - result[0].x1) % np.pi
        ))
        assert abs(result_angle) < 0.01 or abs(result_angle - 180.0) < 0.01

    def test_just_outside_tolerance(self) -> None:
        """Segment à (tolérance + 1)° → PAS snappé."""
        angle = np.deg2rad(6.0)   # tolérance=5°, diff=6° → oblique
        s = seg(0.0, 0.0, np.cos(angle), np.sin(angle))
        original_y2 = s.y2
        result = snap_angles([s], [0.0], tolerance_deg=5.0)
        assert result[0].y2 == pytest.approx(original_y2)


# ---------------------------------------------------------------------------
# Tests detect_dominant_orientations
# ---------------------------------------------------------------------------


class TestDetectDominantOrientations:
    def test_detect_two_orientations(self) -> None:
        """10 segments à ~0° et 10 à ~90° → 2 pics détectés."""
        segs_h = [seg(0.0, float(i), 3.0, float(i)) for i in range(10)]
        segs_v = [seg(float(i), 0.0, float(i), 3.0) for i in range(10)]
        angles = detect_dominant_orientations(segs_h + segs_v)
        assert len(angles) == 2
        # Les deux angles doivent être proches de 0° et 90°
        degs = sorted(np.degrees(a) for a in angles)
        assert degs[0] < 10.0       # pic horizontal
        assert degs[1] > 80.0       # pic vertical

    def test_single_orientation(self) -> None:
        """20 segments tous horizontaux → 1 pic dominant à 0°."""
        segs = [seg(0.0, float(i), 3.0, float(i)) for i in range(20)]
        angles = detect_dominant_orientations(segs)
        assert len(angles) >= 1
        degs = [np.degrees(a) % 180 for a in angles]
        assert any(d < 5.0 or d > 175.0 for d in degs)

    def test_empty_returns_default(self) -> None:
        """Liste vide → [0.0] (fallback)."""
        result = detect_dominant_orientations([])
        assert result == [0.0]

    def test_long_segments_weighted_more(self) -> None:
        """Un segment long de 10 m à 0° doit dominer 50 segments courts à 90°."""
        long_h = seg(0.0, 0.0, 10.0, 0.0)
        short_v = [seg(float(i), 0.0, float(i), 0.1) for i in range(50)]
        angles = detect_dominant_orientations([long_h] + short_v)
        degs = [np.degrees(a) % 180 for a in angles]
        # L'angle horizontal doit être dans les pics
        assert any(d < 5.0 or d > 175.0 for d in degs)

    def test_perpendicular_peaks(self) -> None:
        """Les deux pics d'un bâtiment orthogonal sont à ~90° l'un de l'autre."""
        segs_h = [seg(0.0, float(i), 5.0, float(i)) for i in range(15)]
        segs_v = [seg(float(i), 0.0, float(i), 4.0) for i in range(15)]
        angles = detect_dominant_orientations(segs_h + segs_v)
        if len(angles) >= 2:
            angles_deg = sorted(np.degrees(a) % 180 for a in angles)
            diff = angles_deg[1] - angles_deg[0]
            # Les deux pics doivent être séparés d'environ 90°
            assert 75.0 < diff < 105.0
