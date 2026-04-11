"""Tests unitaires du filtrage multi-slice (murs vs mobilier vs ouvertures)."""

from __future__ import annotations

import numpy as np
import pytest

from scan2plan.detection.line_detection import DetectedSegment
from scan2plan.detection.multi_slice_filter import (
    SegmentMatch,
    classify_segments,
    get_door_candidates,
    get_window_candidates,
    match_segments_across_slices,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(x1: float, y1: float, x2: float, y2: float, source: str = "high") -> DetectedSegment:
    """Crée un DetectedSegment de test avec confidence=1.0."""
    return DetectedSegment(x1=x1, y1=y1, x2=x2, y2=y2, source_slice=source, confidence=1.0)


def _wall_seg(source: str = "high") -> DetectedSegment:
    """Segment de mur horizontal à y=1.0, de x=0 à x=4."""
    return _seg(0.0, 1.0, 4.0, 1.0, source)


def _furniture_seg(source: str = "mid") -> DetectedSegment:
    """Segment de meuble décalé (x=5 à x=9) — ne correspond à aucun mur high."""
    return _seg(5.0, 1.0, 9.0, 1.0, source)


# ---------------------------------------------------------------------------
# Tests SegmentMatch
# ---------------------------------------------------------------------------

class TestSegmentMatch:
    def test_default_classification_is_wall(self) -> None:
        m = SegmentMatch(segment_high=_wall_seg(), segment_mid=None, segment_low=None)
        assert m.classification == "wall"

    def test_furniture_classification_stored(self) -> None:
        m = SegmentMatch(
            segment_high=_furniture_seg(),
            segment_mid=None,
            segment_low=None,
            classification="furniture",
        )
        assert m.classification == "furniture"


# ---------------------------------------------------------------------------
# Tests match_segments_across_slices
# ---------------------------------------------------------------------------

class TestMatchSegmentsAcrossSlices:
    def test_wall_all_three_slices(self) -> None:
        """Un segment présent en high/mid/low doit produire un SegmentMatch complet."""
        segs = {
            "high": [_wall_seg("high")],
            "mid": [_wall_seg("mid")],
            "low": [_wall_seg("low")],
        }
        matches = match_segments_across_slices(segs)
        assert len(matches) == 1
        m = matches[0]
        assert m.segment_high is not None
        assert m.segment_mid is not None
        assert m.segment_low is not None

    def test_furniture_only_mid(self) -> None:
        """Un segment présent uniquement en mid doit être classifié 'furniture'."""
        segs = {
            "high": [],
            "mid": [_furniture_seg("mid")],
            "low": [],
        }
        matches = match_segments_across_slices(segs)
        assert len(matches) == 1
        assert matches[0].classification == "furniture"

    def test_furniture_only_low(self) -> None:
        """Un segment présent uniquement en low → 'furniture'."""
        segs = {
            "high": [],
            "mid": [],
            "low": [_furniture_seg("low")],
        }
        matches = match_segments_across_slices(segs)
        assert len(matches) == 1
        assert matches[0].classification == "furniture"

    def test_high_only_present(self) -> None:
        """Un segment présent uniquement en high → SegmentMatch avec mid=None, low=None."""
        segs = {
            "high": [_wall_seg("high")],
        }
        matches = match_segments_across_slices(segs)
        assert len(matches) == 1
        assert matches[0].segment_mid is None
        assert matches[0].segment_low is None

    def test_no_match_large_offset(self) -> None:
        """Un segment mid très éloigné du high ne doit pas être apparié."""
        segs = {
            "high": [_wall_seg("high")],
            "mid": [_seg(20.0, 20.0, 24.0, 20.0, "mid")],
        }
        matches = match_segments_across_slices(segs)
        # 1 match pour le high (mid=None) + 1 furniture pour le mid orphelin
        assert len(matches) == 2
        high_match = next(m for m in matches if m.segment_high.source_slice == "high")
        assert high_match.segment_mid is None

    def test_angle_mismatch_no_pairing(self) -> None:
        """Un segment mid perpendiculaire au high ne doit pas être apparié."""
        segs = {
            "high": [_seg(0.0, 0.0, 4.0, 0.0, "high")],   # horizontal
            "mid": [_seg(2.0, -2.0, 2.0, 2.0, "mid")],    # vertical, même zone
        }
        matches = match_segments_across_slices(segs, distance_tolerance=0.50)
        high_match = next(m for m in matches if m.classification != "furniture")
        assert high_match.segment_mid is None

    def test_combined_wall_and_furniture(self) -> None:
        """Un mur haut + un meuble mid orphelin → 2 matches dont 1 furniture."""
        segs = {
            "high": [_wall_seg("high")],
            "mid": [_wall_seg("mid"), _furniture_seg("mid")],
            "low": [_wall_seg("low")],
        }
        matches = match_segments_across_slices(segs)
        furniture_count = sum(1 for m in matches if m.classification == "furniture")
        assert furniture_count == 1

    def test_empty_slices(self) -> None:
        matches = match_segments_across_slices({})
        assert matches == []

    def test_missing_mid_and_low_keys(self) -> None:
        """Clés 'mid' et 'low' absentes → 1 seul match, mid/low = None."""
        segs = {"high": [_wall_seg("high")]}
        matches = match_segments_across_slices(segs)
        assert len(matches) == 1
        assert matches[0].segment_mid is None
        assert matches[0].segment_low is None


# ---------------------------------------------------------------------------
# Tests classify_segments
# ---------------------------------------------------------------------------

class TestClassifySegments:
    def test_wall_detected_all_slices(self) -> None:
        """Segment HIGH + MID + LOW → classifié 'wall', retourné dans la liste."""
        match = SegmentMatch(
            segment_high=_wall_seg("high"),
            segment_mid=_wall_seg("mid"),
            segment_low=_wall_seg("low"),
        )
        walls = classify_segments([match])
        assert len(walls) == 1
        assert match.classification == "wall"

    def test_wall_high_mid_no_low(self) -> None:
        """Segment HIGH + MID, pas LOW → 'door_candidate' (LOW absent = ouverture basse)."""
        match = SegmentMatch(
            segment_high=_wall_seg("high"),
            segment_mid=_wall_seg("mid"),
            segment_low=None,
        )
        walls = classify_segments([match])
        assert len(walls) == 1
        assert match.classification == "door_candidate"

    def test_furniture_filtered(self) -> None:
        """Segment pré-classifié 'furniture' → exclu de la liste de retour."""
        match = SegmentMatch(
            segment_high=_furniture_seg("mid"),
            segment_mid=None,
            segment_low=None,
            classification="furniture",
        )
        walls = classify_segments([match])
        assert len(walls) == 0

    def test_door_candidate(self) -> None:
        """HIGH présent, LOW absent → 'door_candidate', inclus dans les murs."""
        # Forcer la condition door : high présent, low absent
        match2 = SegmentMatch(
            segment_high=_wall_seg("high"),
            segment_mid=None,
            segment_low=None,
        )
        walls = classify_segments([match2])
        assert len(walls) == 1
        assert match2.classification == "door_candidate"

    def test_window_candidate(self) -> None:
        """HIGH + LOW, MID absent → 'window_candidate', inclus dans les murs."""
        match = SegmentMatch(
            segment_high=_wall_seg("high"),
            segment_mid=None,
            segment_low=_wall_seg("low"),
        )
        walls = classify_segments([match])
        assert len(walls) == 1
        assert match.classification == "window_candidate"

    def test_mixed_batch(self) -> None:
        """Batch mélangé : 2 murs, 1 furniture, 1 porte, 1 fenêtre."""
        matches = [
            SegmentMatch(_wall_seg(), _wall_seg("mid"), _wall_seg("low")),           # wall
            SegmentMatch(_wall_seg(), _wall_seg("mid"), _wall_seg("low")),           # wall
            SegmentMatch(_furniture_seg("mid"), None, None, classification="furniture"),
            SegmentMatch(_wall_seg(), None, None),                                    # door_candidate
            SegmentMatch(_wall_seg(), None, _wall_seg("low")),                       # window_candidate
        ]
        walls = classify_segments(matches)
        assert len(walls) == 4  # 2 murs + 1 porte + 1 fenêtre
        assert matches[0].classification == "wall"
        assert matches[1].classification == "wall"
        assert matches[3].classification == "door_candidate"
        assert matches[4].classification == "window_candidate"


# ---------------------------------------------------------------------------
# Tests get_door_candidates / get_window_candidates
# ---------------------------------------------------------------------------

class TestGetCandidates:
    def _make_matches_classified(self) -> list[SegmentMatch]:
        matches = [
            SegmentMatch(_wall_seg(), _wall_seg("mid"), _wall_seg("low")),
            SegmentMatch(_wall_seg(), None, None),           # door
            SegmentMatch(_wall_seg(), None, _wall_seg("low")),  # window
            SegmentMatch(_furniture_seg("mid"), None, None, classification="furniture"),
        ]
        classify_segments(matches)
        return matches

    def test_get_door_candidates_count(self) -> None:
        matches = self._make_matches_classified()
        doors = get_door_candidates(matches)
        assert len(doors) == 1

    def test_get_window_candidates_count(self) -> None:
        matches = self._make_matches_classified()
        windows = get_window_candidates(matches)
        assert len(windows) == 1

    def test_no_candidates_when_all_walls(self) -> None:
        matches = [SegmentMatch(_wall_seg(), _wall_seg("mid"), _wall_seg("low"))]
        classify_segments(matches)
        assert get_door_candidates(matches) == []
        assert get_window_candidates(matches) == []


# ---------------------------------------------------------------------------
# Tests d'intégration : scénarios métier synthétiques
# ---------------------------------------------------------------------------

class TestIntegrationScenarios:
    def test_wall_detected_floor_to_ceiling(self) -> None:
        """Mur du sol au plafond : présent en HIGH, MID, LOW → classifié 'wall'."""
        wall = _wall_seg
        segs = {
            "high": [wall("high")],
            "mid": [wall("mid")],
            "low": [wall("low")],
        }
        matches = match_segments_across_slices(segs)
        walls = classify_segments(matches)

        assert len(walls) == 1
        assert matches[0].classification == "wall"

    def test_furniture_filtered_no_high(self) -> None:
        """Meuble bas (0-1.5m) → absent en HIGH, présent en MID et/ou LOW → 'furniture'."""
        wall_high = _wall_seg("high")
        wall_mid = _wall_seg("mid")
        wall_low = _wall_seg("low")
        # Meuble décalé spatialement
        furn_mid = _furniture_seg("mid")
        furn_low = _furniture_seg("low")

        segs = {
            "high": [wall_high],
            "mid": [wall_mid, furn_mid],
            "low": [wall_low, furn_low],
        }
        matches = match_segments_across_slices(segs)
        walls = classify_segments(matches)

        # 1 mur confirmé + le meuble filtré
        assert any(m.classification == "wall" for m in matches)
        assert any(m.classification == "furniture" for m in matches)
        # Les meubles ne sont pas dans walls
        wall_sources = {w.source_slice for w in walls}
        assert "high" in wall_sources

    def test_door_candidate_high_no_low(self) -> None:
        """Mur en HIGH sans correspondant en LOW → ouverture basse (porte candidate)."""
        segs = {
            "high": [_wall_seg("high")],
            "mid": [],
            "low": [],
        }
        matches = match_segments_across_slices(segs)
        classify_segments(matches)
        doors = get_door_candidates(matches)

        assert len(doors) == 1

    def test_window_candidate_high_and_low_no_mid(self) -> None:
        """Présent en HIGH et LOW, absent en MID → fenêtre candidate."""
        segs = {
            "high": [_wall_seg("high")],
            "mid": [],
            "low": [_wall_seg("low")],
        }
        matches = match_segments_across_slices(segs)
        classify_segments(matches)
        windows = get_window_candidates(matches)

        assert len(windows) == 1
