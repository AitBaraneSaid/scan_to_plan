"""Tests unitaires du validateur QA (qa/validator.py et qa/metrics.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scan2plan.detection.line_detection import DetectedSegment
from scan2plan.detection.openings import Opening
from scan2plan.qa.metrics import QAReport
from scan2plan.qa.validator import generate_qa_report, validate_plan
from scan2plan.vectorization.topology import WallGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(
    x1: float, y1: float, x2: float, y2: float,
    source: str = "high", confidence: float = 1.0,
) -> DetectedSegment:
    return DetectedSegment(x1=x1, y1=y1, x2=x2, y2=y2,
                           source_slice=source, confidence=confidence)


def _opening(type_: str = "door") -> Opening:
    return Opening(
        type=type_,
        wall_segment=_seg(0, 0, 1, 0),
        position_start=0.2,
        position_end=1.0,
        width=0.8,
        confidence=0.9,
    )


def _rect_graph(w: float = 4.0, h: float = 3.0) -> WallGraph:
    """Rectangle fermé de w×h — 4 nœuds, 4 arêtes, 4 segments."""
    nodes = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    segments = [
        _seg(0.0, 0.0, w, 0.0),
        _seg(w, 0.0, w, h),
        _seg(w, h, 0.0, h),
        _seg(0.0, h, 0.0, 0.0),
    ]
    return WallGraph(nodes=nodes, edges=edges, segments=segments)


# ---------------------------------------------------------------------------
# Tests QAReport dataclass
# ---------------------------------------------------------------------------

class TestQAReport:
    def test_default_values(self) -> None:
        r = QAReport()
        assert r.total_wall_length == 0.0
        assert r.num_segments == 0
        assert r.num_rooms_detected == 0
        assert r.score == 100.0
        assert r.warnings == []

    def test_is_valid_threshold(self) -> None:
        r = QAReport(score=50.0)
        assert r.is_valid is True

    def test_is_invalid_below_50(self) -> None:
        r = QAReport(score=49.9)
        assert r.is_valid is False

    def test_summary_contains_score(self) -> None:
        r = QAReport(score=75.0, num_segments=4, num_rooms_detected=1)
        text = r.summary()
        assert "75" in text
        assert "4" in text


# ---------------------------------------------------------------------------
# Tests validate_plan — graphe vide
# ---------------------------------------------------------------------------

class TestValidatePlanEmpty:
    def test_empty_graph_score_zero(self) -> None:
        g = WallGraph()
        report = validate_plan(g, [])
        assert report.score == 0.0

    def test_empty_graph_has_warning(self) -> None:
        g = WallGraph()
        report = validate_plan(g, [])
        assert any("segment" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# Tests validate_plan — rectangle fermé (cas idéal)
# ---------------------------------------------------------------------------

class TestValidatePlanRectangle:
    def test_rectangle_score_high(self) -> None:
        """Rectangle parfait : aucun gap, 1 pièce → score élevé."""
        g = _rect_graph()
        report = validate_plan(g, [])
        # 4 nœuds de degré 2 → 0 gap → score 100 + bonus si ≥2 pièces
        # Mais 1 seule pièce → pas de bonus. Pas de pénalités → score 100.
        assert report.score >= 95.0

    def test_rectangle_one_room(self) -> None:
        g = _rect_graph()
        report = validate_plan(g, [])
        assert report.num_rooms_detected == 1

    def test_rectangle_no_gaps(self) -> None:
        g = _rect_graph()
        report = validate_plan(g, [])
        assert report.num_gaps == 0

    def test_rectangle_no_orphans(self) -> None:
        g = _rect_graph()
        report = validate_plan(g, [])
        assert report.num_orphan_segments == 0

    def test_rectangle_total_length(self) -> None:
        g = _rect_graph(w=4.0, h=3.0)
        report = validate_plan(g, [])
        # 4+3+4+3 = 14 m
        assert report.total_wall_length == pytest.approx(14.0, abs=0.01)

    def test_rectangle_avg_confidence_one(self) -> None:
        g = _rect_graph()
        report = validate_plan(g, [])
        assert report.avg_confidence == pytest.approx(1.0)

    def test_openings_counted(self) -> None:
        g = _rect_graph()
        openings = [_opening("door"), _opening("window")]
        report = validate_plan(g, openings)
        assert report.num_openings == 2


# ---------------------------------------------------------------------------
# Tests validate_plan — graphe avec gaps
# ---------------------------------------------------------------------------

class TestValidatePlanGaps:
    def test_single_pendant_node_gap(self) -> None:
        """Segment isolé (deux nœuds de degré 1) → 2 gaps, 1 orphelin."""
        g = WallGraph(
            nodes=[(0.0, 0.0), (4.0, 0.0)],
            edges=[(0, 1)],
            segments=[_seg(0.0, 0.0, 4.0, 0.0)],
        )
        report = validate_plan(g, [])
        assert report.num_gaps == 2
        assert report.num_orphan_segments == 1

    def test_gap_penalizes_score(self) -> None:
        """Chaque gap enlève 10 points."""
        g = WallGraph(
            nodes=[(0.0, 0.0), (4.0, 0.0)],
            edges=[(0, 1)],
            segments=[_seg(0.0, 0.0, 4.0, 0.0)],
        )
        report = validate_plan(g, [])
        # 2 gaps → -20, 1 orphelin → -5, pas de pièce → -5
        # 100 - 20 - 5 - 5 = 70
        assert report.score == pytest.approx(70.0, abs=1.0)

    def test_gap_triggers_warning(self) -> None:
        g = WallGraph(
            nodes=[(0.0, 0.0), (4.0, 0.0)],
            edges=[(0, 1)],
            segments=[_seg(0.0, 0.0, 4.0, 0.0)],
        )
        report = validate_plan(g, [])
        assert any("gap" in w.lower() or "connectée" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# Tests validate_plan — micro-segments
# ---------------------------------------------------------------------------

class TestValidatePlanMicroSegments:
    def test_micro_segment_counted(self) -> None:
        """Segment de 5 cm = micro-segment."""
        g = WallGraph(
            nodes=[(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0), (0.05, 0.0)],
            edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 4)],
            segments=[
                _seg(0.0, 0.0, 4.0, 0.0),
                _seg(4.0, 0.0, 4.0, 3.0),
                _seg(4.0, 3.0, 0.0, 3.0),
                _seg(0.0, 3.0, 0.0, 0.0),
                _seg(0.0, 0.0, 0.05, 0.0),  # 5 cm
            ],
        )
        report = validate_plan(g, [])
        assert report.num_micro_segments == 1

    def test_micro_segment_triggers_warning(self) -> None:
        g = WallGraph(
            nodes=[(0.0, 0.0), (0.05, 0.0)],
            edges=[(0, 1)],
            segments=[_seg(0.0, 0.0, 0.05, 0.0)],
        )
        report = validate_plan(g, [])
        assert any("micro" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# Tests validate_plan — score
# ---------------------------------------------------------------------------

class TestQAScore:
    def test_score_capped_at_100(self) -> None:
        g = _rect_graph()
        report = validate_plan(g, [])
        assert report.score <= 100.0

    def test_score_minimum_zero(self) -> None:
        """Nombreux problèmes → score minimal = 0, jamais négatif."""
        g = WallGraph(
            nodes=[(0.0, 0.0), (1.0, 0.0)],
            edges=[(0, 1)],
            segments=[_seg(0.0, 0.0, 0.05, 0.0)] * 20,
        )
        report = validate_plan(g, [])
        assert report.score >= 0.0

    def test_two_rooms_adds_bonus(self) -> None:
        """Graphe à 2 pièces fermées → bonus de +10 au score."""
        # Deux rectangles partageant un côté
        nodes = [(0, 0), (4, 0), (4, 3), (0, 3), (8, 0), (8, 3)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 5), (5, 2)]
        segments = [
            _seg(0, 0, 4, 0), _seg(4, 0, 4, 3), _seg(4, 3, 0, 3), _seg(0, 3, 0, 0),
            _seg(4, 0, 8, 0), _seg(8, 0, 8, 3), _seg(8, 3, 4, 3),
        ]
        g = WallGraph(nodes=nodes, edges=edges, segments=segments)
        report = validate_plan(g, [])
        assert report.num_rooms_detected == 2
        # Tous nœuds de degré ≥ 2 → 0 gap. Score = 100 + 10 = 110, capped à 100
        assert report.score == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Tests validate_plan — confiance faible
# ---------------------------------------------------------------------------

class TestLowConfidence:
    def test_low_confidence_segment_in_zones(self) -> None:
        g = WallGraph(
            nodes=[(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)],
            edges=[(0, 1), (1, 2), (2, 3), (3, 0)],
            segments=[
                _seg(0.0, 0.0, 4.0, 0.0, confidence=0.3),  # faible
                _seg(4.0, 0.0, 4.0, 3.0, confidence=1.0),
                _seg(4.0, 3.0, 0.0, 3.0, confidence=1.0),
                _seg(0.0, 3.0, 0.0, 0.0, confidence=1.0),
            ],
        )
        report = validate_plan(g, [])
        assert len(report.low_confidence_zones) == 1
        # Centroïde du segment horizontal bas : (2.0, 0.0)
        cx, cy = report.low_confidence_zones[0]
        assert cx == pytest.approx(2.0)
        assert cy == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests generate_qa_report
# ---------------------------------------------------------------------------

class TestGenerateQaReport:
    def test_writes_json_file(self, tmp_path: Path) -> None:
        g = _rect_graph()
        report = validate_plan(g, [])
        out = tmp_path / "qa.json"
        generate_qa_report(report, out)
        assert out.exists()

    def test_json_contains_score(self, tmp_path: Path) -> None:
        g = _rect_graph()
        report = validate_plan(g, [])
        out = tmp_path / "qa.json"
        generate_qa_report(report, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "score" in data
        assert isinstance(data["score"], float)

    def test_json_adds_extension(self, tmp_path: Path) -> None:
        """Passer un chemin sans .json → l'extension est ajoutée."""
        g = _rect_graph()
        report = validate_plan(g, [])
        out = tmp_path / "qa_report"  # pas d'extension
        generate_qa_report(report, out)
        assert (tmp_path / "qa_report.json").exists()

    def test_json_structure_complete(self, tmp_path: Path) -> None:
        g = _rect_graph()
        report = validate_plan(g, [_opening()])
        out = tmp_path / "qa.json"
        generate_qa_report(report, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        expected_keys = {
            "score", "total_wall_length_m", "num_segments", "num_rooms_detected",
            "num_openings", "num_gaps", "num_orphan_segments", "num_micro_segments",
            "avg_confidence", "low_confidence_zones", "warnings",
        }
        assert expected_keys.issubset(data.keys())
        assert data["num_openings"] == 1
