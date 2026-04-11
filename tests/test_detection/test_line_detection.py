"""Tests unitaires pour detection/line_detection.py."""

from __future__ import annotations

import numpy as np
import pytest

from scan2plan.detection.line_detection import DetectedSegment, detect_lines_hough
from scan2plan.slicing.density_map import DensityMapResult


def _make_dmap(width: int = 400, height: int = 400, resolution: float = 0.005) -> DensityMapResult:
    """Crée un DensityMapResult synthétique avec x_min=y_min=0."""
    image = np.zeros((height, width), dtype=np.uint16)
    return DensityMapResult(
        image=image,
        x_min=0.0,
        y_min=0.0,
        resolution=resolution,
        width=width,
        height=height,
    )


def _horizontal_line_image(width: int = 400, height: int = 400) -> np.ndarray:
    """Image binaire avec une ligne horizontale de 200 pixels au centre."""
    img = np.zeros((height, width), dtype=np.uint8)
    mid = height // 2
    img[mid, 100:300] = 255
    return img


def _rectangle_image(width: int = 400, height: int = 400) -> np.ndarray:
    """Image binaire avec un rectangle plein."""
    img = np.zeros((height, width), dtype=np.uint8)
    img[50:350, 50] = 255
    img[50:350, 349] = 255
    img[50, 50:350] = 255
    img[349, 50:350] = 255
    return img


class TestDetectedSegment:
    def test_length(self) -> None:
        seg = DetectedSegment(x1=0.0, y1=0.0, x2=3.0, y2=4.0, source_slice="mid", confidence=0.5)
        assert seg.length == pytest.approx(5.0)

    def test_as_tuple(self) -> None:
        seg = DetectedSegment(x1=1.0, y1=2.0, x2=3.0, y2=4.0, source_slice="high", confidence=0.8)
        assert seg.as_tuple() == (1.0, 2.0, 3.0, 4.0)

    def test_confidence_in_range(self) -> None:
        seg = DetectedSegment(x1=0.0, y1=0.0, x2=2.0, y2=0.0, source_slice="low", confidence=1.0)
        assert 0.0 <= seg.confidence <= 1.0


class TestDetectLinesHough:
    def test_detect_horizontal_line(self) -> None:
        """Image avec une ligne horizontale de 200 px → au moins 1 segment détecté."""
        img = _horizontal_line_image()
        dmap = _make_dmap()
        segments = detect_lines_hough(
            img, dmap, rho=1, theta_deg=1.0, threshold=50,
            min_line_length=100, max_line_gap=10,
        )
        assert len(segments) >= 1

    def test_horizontal_line_orientation(self) -> None:
        """Le segment détecté sur une ligne horizontale doit être quasi-horizontal."""
        img = _horizontal_line_image()
        dmap = _make_dmap()
        segments = detect_lines_hough(
            img, dmap, rho=1, theta_deg=1.0, threshold=30,
            min_line_length=100, max_line_gap=10,
        )
        assert len(segments) >= 1
        seg = segments[0]
        angle_deg = abs(np.degrees(np.arctan2(seg.y2 - seg.y1, seg.x2 - seg.x1)))
        assert angle_deg < 5.0 or angle_deg > 175.0

    def test_no_detection_on_empty(self) -> None:
        """Image noire → 0 segments."""
        img = np.zeros((200, 200), dtype=np.uint8)
        dmap = _make_dmap(200, 200)
        segments = detect_lines_hough(
            img, dmap, rho=1, theta_deg=1.0, threshold=10,
            min_line_length=20, max_line_gap=5,
        )
        assert len(segments) == 0

    def test_returns_list_of_detected_segments(self) -> None:
        img = _horizontal_line_image()
        dmap = _make_dmap()
        segments = detect_lines_hough(img, dmap, threshold=30, min_line_length=80)
        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg, DetectedSegment)

    def test_segments_in_metric_coords(self) -> None:
        """Les coordonnées des segments doivent être en mètres, pas en pixels.

        Avec résolution=0.005 m/px et image 400×400 px, les coords métriques
        couvrent une zone de 0 à 2.0 m. Les valeurs en pixels (0-400) ne
        peuvent pas être confondues avec des mètres.
        """
        img = _horizontal_line_image(400, 400)
        dmap = _make_dmap(width=400, height=400, resolution=0.005)
        segments = detect_lines_hough(
            img, dmap, rho=1, theta_deg=1.0, threshold=30,
            min_line_length=80, max_line_gap=10,
        )
        assert len(segments) >= 1
        for seg in segments:
            # Avec résolution=0.005 et image 400 px, les valeurs métriques sont dans [0, 2.0]
            # Valeurs en pixels seraient dans [0, 400] — facilement discriminables
            assert abs(seg.x1) < 10.0, f"x1={seg.x1} semble être en pixels, pas en mètres"
            assert abs(seg.x2) < 10.0, f"x2={seg.x2} semble être en pixels, pas en mètres"

    def test_confidence_between_0_and_1(self) -> None:
        """Le score de confiance de chaque segment doit être dans [0, 1]."""
        img = _horizontal_line_image()
        dmap = _make_dmap()
        segments = detect_lines_hough(img, dmap, threshold=30, min_line_length=80)
        for seg in segments:
            assert 0.0 <= seg.confidence <= 1.0

    def test_source_slice_stored(self) -> None:
        img = _horizontal_line_image()
        dmap = _make_dmap()
        segments = detect_lines_hough(img, dmap, threshold=30, min_line_length=80, source_slice="high")
        for seg in segments:
            assert seg.source_slice == "high"

    def test_detect_rectangle(self) -> None:
        """Image avec un rectangle → au moins 4 segments (un par côté)."""
        img = _rectangle_image()
        dmap = _make_dmap()
        segments = detect_lines_hough(
            img, dmap, rho=1, theta_deg=1.0, threshold=30,
            min_line_length=50, max_line_gap=20,
        )
        assert len(segments) >= 4
