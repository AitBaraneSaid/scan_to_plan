"""Tests unitaires pour micro_fuse_segments.

Vérifie que la fusion ultra-conservative recolle les fragments Hough (gap < 5 cm)
sans jamais franchir une ouverture architecturale (porte ≥ 70 cm, fenêtre ≥ 30 cm).
"""

from __future__ import annotations

import numpy as np
import pytest

from scan2plan.detection.line_detection import DetectedSegment
from scan2plan.detection.micro_fusion import (
    _compute_gap,
    _merge_two_segments,
    micro_fuse_segments,
)


# ---------------------------------------------------------------------------
# Helpers de construction
# ---------------------------------------------------------------------------

def seg(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> DetectedSegment:
    """Construit un DetectedSegment horizontal ou quelconque."""
    return DetectedSegment(x1=x1, y1=y1, x2=x2, y2=y2, source_slice="high", confidence=conf)


# ---------------------------------------------------------------------------
# Tests _compute_gap
# ---------------------------------------------------------------------------

class TestComputeGap:
    def test_positive_gap(self) -> None:
        """Gap positif entre deux segments bout-à-bout."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)
        s2 = seg(1.10, 0.0, 2.0, 0.0)
        gap = _compute_gap(s1, s2)
        assert abs(gap - 0.10) < 0.001

    def test_negative_gap_overlap(self) -> None:
        """Chevauchement → gap négatif."""
        s1 = seg(0.0, 0.0, 1.10, 0.0)
        s2 = seg(1.00, 0.0, 2.00, 0.0)
        gap = _compute_gap(s1, s2)
        assert gap < 0.0

    def test_zero_gap_touching(self) -> None:
        """Segments qui se touchent exactement → gap ≈ 0."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)
        s2 = seg(1.0, 0.0, 2.0, 0.0)
        gap = _compute_gap(s1, s2)
        assert abs(gap) < 0.001

    def test_symmetry(self) -> None:
        """Le gap est symétrique : gap(a,b) == gap(b,a)."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)
        s2 = seg(1.08, 0.0, 2.0, 0.0)
        assert abs(_compute_gap(s1, s2) - _compute_gap(s2, s1)) < 0.001

    def test_diagonal_segments(self) -> None:
        """Gap calculé sur des segments diagonaux."""
        # Segment à 45° de longueur √2
        s1 = seg(0.0, 0.0, 1.0, 1.0)
        # Gap de 0.10*√2 ≈ 0.141 m
        s2 = seg(1.10, 1.10, 2.0, 2.0)
        gap = _compute_gap(s1, s2)
        assert abs(gap - 0.10 * np.sqrt(2)) < 0.01


# ---------------------------------------------------------------------------
# Tests _merge_two_segments
# ---------------------------------------------------------------------------

class TestMergeTwoSegments:
    def test_merge_length(self) -> None:
        """La longueur du segment fusionné couvre les deux segments."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)
        s2 = seg(1.03, 0.0, 2.0, 0.0)
        merged = _merge_two_segments(s1, s2)
        assert merged.length > 2.0 - 0.01  # couvre les deux

    def test_merge_preserves_direction(self) -> None:
        """Le segment fusionné reste horizontal."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)
        s2 = seg(1.03, 0.01, 2.0, 0.01)  # léger décalage vertical
        merged = _merge_two_segments(s1, s2)
        # La direction doit rester quasi-horizontale
        angle = abs(np.arctan2(merged.y2 - merged.y1, merged.x2 - merged.x1))
        assert angle < np.deg2rad(5.0)

    def test_merge_confidence_max(self) -> None:
        """La confidence du segment fusionné est le max des deux."""
        s1 = seg(0.0, 0.0, 1.0, 0.0, conf=0.7)
        s2 = seg(1.03, 0.0, 2.0, 0.0, conf=0.9)
        merged = _merge_two_segments(s1, s2)
        assert merged.confidence == pytest.approx(0.9)

    def test_merge_source_slice_preserved(self) -> None:
        """Le source_slice du premier segment est préservé."""
        s1 = DetectedSegment(0.0, 0.0, 1.0, 0.0, source_slice="high", confidence=0.9)
        s2 = DetectedSegment(1.03, 0.0, 2.0, 0.0, source_slice="mid", confidence=0.9)
        merged = _merge_two_segments(s1, s2)
        assert merged.source_slice == "high"


# ---------------------------------------------------------------------------
# Tests micro_fuse_segments — comportements principaux
# ---------------------------------------------------------------------------

class TestMicroFuseSegments:
    def test_fuse_micro_gap(self) -> None:
        """2 segments colinéaires séparés de 3 cm → fusionnés en 1."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)
        s2 = seg(1.03, 0.0, 2.0, 0.0)
        result = micro_fuse_segments([s1, s2], max_gap=0.05)
        assert len(result) == 1
        assert result[0].length > 1.9

    def test_preserve_door_gap(self) -> None:
        """2 segments colinéaires séparés de 80 cm → PAS fusionnés (porte)."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)
        s2 = seg(1.80, 0.0, 3.0, 0.0)
        result = micro_fuse_segments([s1, s2], max_gap=0.05)
        assert len(result) == 2

    def test_preserve_window_gap(self) -> None:
        """2 segments colinéaires séparés de 60 cm → PAS fusionnés (fenêtre)."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)
        s2 = seg(1.60, 0.0, 3.0, 0.0)
        result = micro_fuse_segments([s1, s2], max_gap=0.05)
        assert len(result) == 2

    def test_fuse_overlapping(self) -> None:
        """2 segments qui se chevauchent de 5 cm → fusionnés en 1."""
        s1 = seg(0.0, 0.0, 1.05, 0.0)
        s2 = seg(1.00, 0.0, 2.00, 0.0)
        result = micro_fuse_segments([s1, s2], max_gap=0.05)
        assert len(result) == 1
        assert result[0].length > 1.9

    def test_no_fuse_perpendicular(self) -> None:
        """2 segments perpendiculaires → PAS fusionnés."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)          # horizontal
        s2 = seg(1.03, 0.0, 1.03, 1.0)         # vertical
        result = micro_fuse_segments([s1, s2], max_gap=0.05)
        assert len(result) == 2

    def test_no_fuse_parallel_offset(self) -> None:
        """2 segments parallèles décalés de 15 cm perpendiculairement → PAS fusionnés.

        Ce sont les deux faces d'un mur — ne jamais les fusionner.
        """
        s1 = seg(0.0, 0.00, 2.0, 0.00)
        s2 = seg(0.0, 0.15, 2.0, 0.15)   # 15 cm de décalage perpendiculaire
        result = micro_fuse_segments([s1, s2], max_gap=0.05)
        assert len(result) == 2

    def test_three_fragments(self) -> None:
        """3 fragments colinéaires avec gaps de 2 cm → fusionnés en 1."""
        s1 = seg(0.00, 0.0, 1.00, 0.0)
        s2 = seg(1.02, 0.0, 2.00, 0.0)
        s3 = seg(2.02, 0.0, 3.00, 0.0)
        result = micro_fuse_segments([s1, s2, s3], max_gap=0.05)
        assert len(result) == 1
        assert result[0].length > 2.9

    def test_convergence(self) -> None:
        """La fusion converge en < 10 itérations sur 10 fragments."""
        fragments = [seg(float(i) * 1.02, 0.0, float(i) * 1.02 + 1.0, 0.0)
                     for i in range(10)]
        result = micro_fuse_segments(fragments, max_gap=0.05, max_iterations=10)
        # Doit converger (pas lever d'exception ni boucler indéfiniment)
        assert isinstance(result, list)

    def test_preserves_short_wall(self) -> None:
        """Un segment court isolé (25 cm) est conservé tel quel."""
        s_short = seg(5.0, 0.0, 5.25, 0.0)  # 25 cm, loin de tout
        s_far = seg(0.0, 0.0, 1.0, 0.0)     # segment distinct, 5 m de distance
        result = micro_fuse_segments([s_short, s_far], max_gap=0.05)
        assert len(result) == 2
        lengths = sorted([r.length for r in result])
        assert lengths[0] == pytest.approx(0.25, abs=0.01)

    def test_empty_input(self) -> None:
        """Liste vide → liste vide."""
        assert micro_fuse_segments([]) == []

    def test_single_segment(self) -> None:
        """Un seul segment → retourné inchangé."""
        s = seg(0.0, 0.0, 1.0, 0.0)
        result = micro_fuse_segments([s])
        assert len(result) == 1
        assert result[0].x1 == pytest.approx(0.0)
        assert result[0].x2 == pytest.approx(1.0)

    def test_count_reduction_realistic(self) -> None:
        """Simulation réaliste : 20 murs de 3m fragmentés en 3 morceaux chacun.

        Chaque mur a des micro-gaps de 2 cm entre fragments,
        et les murs sont séparés de 1 m les uns des autres (perpendiculairement).
        Résultat attendu : 20 segments (un par mur), pas 60.
        """
        segments = []
        for wall_idx in range(20):
            y = float(wall_idx) * 1.0   # murs espacés de 1 m
            # 3 fragments avec gaps de 2 cm
            segments.append(seg(0.00, y, 0.98, y))
            segments.append(seg(1.00, y, 1.98, y))
            segments.append(seg(2.00, y, 3.00, y))

        result = micro_fuse_segments(segments, max_gap=0.05, perpendicular_tolerance=0.02)
        # Chaque groupe de 3 fragments doit fusionner → 20 segments
        assert len(result) == 20

    def test_gap_exactly_at_threshold(self) -> None:
        """Gap exactement à max_gap=5 cm → fusionné (limite incluse)."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)
        s2 = seg(1.05, 0.0, 2.0, 0.0)
        result = micro_fuse_segments([s1, s2], max_gap=0.05)
        assert len(result) == 1

    def test_gap_just_above_threshold(self) -> None:
        """Gap légèrement > max_gap → PAS fusionné."""
        s1 = seg(0.0, 0.0, 1.0, 0.0)
        s2 = seg(1.06, 0.0, 2.0, 0.0)
        result = micro_fuse_segments([s1, s2], max_gap=0.05)
        assert len(result) == 2

    def test_angle_tolerance_respected(self) -> None:
        """Segments avec écart angulaire > tolérance → PAS fusionnés."""
        # s2 est légèrement incliné (10°, bien au-delà de 3°)
        angle = np.deg2rad(10.0)
        s1 = seg(0.0, 0.0, 1.0, 0.0)
        s2 = DetectedSegment(
            x1=1.02,
            y1=0.0,
            x2=1.02 + np.cos(angle),
            y2=np.sin(angle),
            source_slice="high",
            confidence=0.9,
        )
        result = micro_fuse_segments([s1, s2], max_gap=0.05, angle_tolerance_deg=3.0)
        assert len(result) == 2

    def test_diagonal_micro_gap(self) -> None:
        """2 segments diagonaux (45°) séparés d'un micro-gap → fusionnés."""
        # Segment à 45° de longueur √2 ≈ 1.414 m
        s1 = DetectedSegment(0.0, 0.0, 1.0, 1.0, source_slice="high", confidence=0.9)
        # Gap de 0.03*√2 ≈ 4.2 cm dans la direction diagonale
        s2 = DetectedSegment(1.03, 1.03, 2.0, 2.0, source_slice="high", confidence=0.9)
        result = micro_fuse_segments([s1, s2], max_gap=0.05)
        assert len(result) == 1
