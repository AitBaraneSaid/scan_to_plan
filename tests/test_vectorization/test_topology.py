"""Tests unitaires de la reconstruction topologique (topology.py)."""

from __future__ import annotations

import math

import pytest

from scan2plan.detection.line_detection import DetectedSegment
from scan2plan.vectorization.topology import (
    WallGraph,
    build_wall_graph,
    clean_topology,
    detect_rooms,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(
    x1: float, y1: float, x2: float, y2: float,
    source: str = "high", confidence: float = 1.0,
) -> DetectedSegment:
    return DetectedSegment(x1=x1, y1=y1, x2=x2, y2=y2,
                           source_slice=source, confidence=confidence)


def _rect_with_gap(w: float = 4.0, h: float = 3.0, gap: float = 0.03) -> list[DetectedSegment]:
    """Rectangle de w×h mètres avec des gaps de ``gap`` mètres aux 4 coins."""
    g = gap
    return [
        # Bas : (0,0) → (w,0), avec gap à gauche et à droite
        _seg(g, 0.0, w - g, 0.0),
        # Droite : (w,0) → (w,h)
        _seg(w, g, w, h - g),
        # Haut : (w,h) → (0,h)
        _seg(w - g, h, g, h),
        # Gauche : (0,h) → (0,0)
        _seg(0.0, h - g, 0.0, g),
    ]


# ---------------------------------------------------------------------------
# Tests WallGraph dataclass
# ---------------------------------------------------------------------------

class TestWallGraph:
    def test_default_is_empty(self) -> None:
        g = WallGraph()
        assert g.nodes == []
        assert g.edges == []
        assert g.segments == []
        assert g.openings == []

    def test_stores_openings(self) -> None:
        from scan2plan.detection.openings import Opening
        op = Opening(type="door", wall_segment=_seg(0, 0, 1, 0),
                     position_start=0.2, position_end=1.1, width=0.9, confidence=0.8)
        g = WallGraph(openings=[op])
        assert len(g.openings) == 1


# ---------------------------------------------------------------------------
# Tests build_wall_graph — cas de base
# ---------------------------------------------------------------------------

class TestBuildWallGraph:
    def test_empty_segments_returns_empty_graph(self) -> None:
        g = build_wall_graph([], [])
        assert len(g.nodes) == 0
        assert len(g.edges) == 0

    def test_single_segment_produces_two_nodes(self) -> None:
        segs = [_seg(0.0, 0.0, 3.0, 0.0)]
        g = build_wall_graph(segs, [], min_segment_length=0.05)
        assert len(g.nodes) == 2
        assert len(g.edges) == 1

    def test_too_short_segment_removed(self) -> None:
        """Un segment de 5 cm est supprimé si min_segment_length=0.10."""
        segs = [_seg(0.0, 0.0, 0.05, 0.0)]
        g = build_wall_graph(segs, [], min_segment_length=0.10)
        assert len(g.edges) == 0

    def test_openings_stored_in_graph(self) -> None:
        from scan2plan.detection.openings import Opening
        op = Opening(type="door", wall_segment=_seg(0, 0, 4, 0),
                     position_start=1.5, position_end=2.4, width=0.9, confidence=0.8)
        segs = [_seg(0.0, 0.0, 4.0, 0.0)]
        g = build_wall_graph(segs, [op])
        assert len(g.openings) == 1

    def test_rectangular_room_four_nodes_four_edges(self) -> None:
        """Rectangle 4×3 m avec gaps de 3 cm aux coins → 4 nœuds, 4 arêtes."""
        segs = _rect_with_gap(w=4.0, h=3.0, gap=0.03)
        g = build_wall_graph(segs, [], intersection_distance=0.08)
        assert len(g.nodes) == 4, f"Nœuds : {len(g.nodes)}"
        assert len(g.edges) == 4, f"Arêtes : {len(g.edges)}"

    def test_two_perpendicular_segments_share_corner(self) -> None:
        """Deux segments perpendiculaires dont les extrémités sont à 3 cm → 3 nœuds."""
        # Segment horizontal : (0,0)→(4-0.03, 0) = (0,0)→(3.97, 0)
        # Segment vertical   : (4, 0.03)→(4, 3)
        segs = [
            _seg(0.0, 0.0, 3.97, 0.0),
            _seg(4.0, 0.03, 4.0, 3.0),
        ]
        g = build_wall_graph(segs, [], intersection_distance=0.08)
        # Après résolution, les deux extrémités proches de (4,0) convergent → 3 nœuds
        assert len(g.nodes) == 3


# ---------------------------------------------------------------------------
# Tests build_wall_graph — scénarios métier
# ---------------------------------------------------------------------------

class TestBuildWallGraphScenarios:
    def test_rectangular_room_one_cycle(self) -> None:
        """Rectangle fermé → 1 cycle détecté (= 1 pièce)."""
        segs = _rect_with_gap(w=4.0, h=3.0, gap=0.03)
        g = build_wall_graph(segs, [], intersection_distance=0.08)
        cycles = detect_rooms(g)
        assert len(cycles) == 1, f"Cycles : {cycles}"

    def test_two_rooms_sharing_wall(self) -> None:
        """Deux rectangles partageant un mur commun → 2 cycles.

        Géométrie : pièce 1 = (0,0)→(4,3), pièce 2 = (4,0)→(8,3),
        mur partagé à x=4.
        """
        gap = 0.03
        segs = [
            # Pièce 1
            _seg(gap, 0.0, 4.0 - gap, 0.0),      # bas P1
            _seg(4.0, gap, 4.0, 3.0 - gap),       # droite P1 = gauche P2 (mur partagé)
            _seg(4.0 - gap, 3.0, gap, 3.0),       # haut P1
            _seg(0.0, 3.0 - gap, 0.0, gap),       # gauche P1
            # Pièce 2
            _seg(4.0 + gap, 0.0, 8.0 - gap, 0.0),  # bas P2
            _seg(8.0, gap, 8.0, 3.0 - gap),         # droite P2
            _seg(8.0 - gap, 3.0, 4.0 + gap, 3.0),  # haut P2
        ]
        g = build_wall_graph(segs, [], intersection_distance=0.08)
        cycles = detect_rooms(g)
        assert len(cycles) == 2, f"Cycles attendus : 2, obtenus : {len(cycles)}"


# ---------------------------------------------------------------------------
# Tests clean_topology
# ---------------------------------------------------------------------------

class TestCleanTopology:
    def test_orphan_short_segment_removed(self) -> None:
        """Segment pendant court (5 cm) avec un nœud de degré 1 → supprimé."""
        # Triangle : nœud A-B-C fermé + nœud D pendant à C avec 5cm
        g = WallGraph(
            nodes=[(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (4.05, 3.0)],
            edges=[(0, 1), (1, 2), (2, 0), (2, 3)],
            segments=[
                _seg(0.0, 0.0, 4.0, 0.0),
                _seg(4.0, 0.0, 4.0, 3.0),
                _seg(4.0, 3.0, 0.0, 0.0),
                _seg(4.0, 3.0, 4.05, 3.0),  # 5 cm pendant
            ],
        )
        cleaned = clean_topology(g, min_segment_length=0.10)
        seg_lengths = [s.length for s in cleaned.segments]
        assert all(l >= 0.10 or l == pytest.approx(0.0, abs=1e-6)
                   for l in seg_lengths), f"Segments restants : {seg_lengths}"
        # Le segment de 5 cm doit être absent
        assert not any(l < 0.06 for l in seg_lengths)

    def test_duplicate_edges_removed(self) -> None:
        """Arête en doublon (même paire de nœuds) → une seule conservée."""
        g = WallGraph(
            nodes=[(0.0, 0.0), (4.0, 0.0)],
            edges=[(0, 1), (0, 1)],
            segments=[
                _seg(0.0, 0.0, 4.0, 0.0),
                _seg(0.0, 0.0, 4.0, 0.0),
            ],
        )
        cleaned = clean_topology(g)
        assert len(cleaned.edges) == 1

    def test_isolated_nodes_removed(self) -> None:
        """Nœuds sans aucune arête → supprimés après compactage."""
        g = WallGraph(
            nodes=[(0.0, 0.0), (4.0, 0.0), (10.0, 10.0)],  # (10,10) isolé
            edges=[(0, 1)],
            segments=[_seg(0.0, 0.0, 4.0, 0.0)],
        )
        cleaned = clean_topology(g)
        assert len(cleaned.nodes) == 2

    def test_nodes_merged_when_close(self) -> None:
        """Deux nœuds à 1 cm de distance → fusionnés en un seul."""
        # Deux segments quasi-colinéaires dont les extrémités internes
        # sont à 1 cm : on les présentera comme un graphe déjà construit
        # avec 4 nœuds dont deux très proches.
        g = WallGraph(
            nodes=[(0.0, 0.0), (2.00, 0.0), (2.01, 0.0), (4.0, 0.0)],
            edges=[(0, 1), (2, 3)],
            segments=[
                _seg(0.0, 0.0, 2.00, 0.0),
                _seg(2.01, 0.0, 4.0, 0.0),
            ],
        )
        cleaned = clean_topology(g)
        # Les nœuds 1 et 2 (dist=1cm < seuil 2cm) doivent être fusionnés
        assert len(cleaned.nodes) <= 3

    def test_empty_graph_stays_empty(self) -> None:
        g = WallGraph()
        cleaned = clean_topology(g)
        assert cleaned.nodes == []
        assert cleaned.edges == []


# ---------------------------------------------------------------------------
# Tests detect_rooms
# ---------------------------------------------------------------------------

class TestDetectRooms:
    def test_no_cycle_returns_empty(self) -> None:
        """Graphe linéaire (chemin) → aucun cycle."""
        g = WallGraph(
            nodes=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            edges=[(0, 1), (1, 2)],
            segments=[_seg(0, 0, 1, 0), _seg(1, 0, 2, 0)],
        )
        cycles = detect_rooms(g)
        assert len(cycles) == 0

    def test_triangle_one_cycle(self) -> None:
        """Triangle → 1 cycle."""
        g = WallGraph(
            nodes=[(0.0, 0.0), (3.0, 0.0), (1.5, 2.0)],
            edges=[(0, 1), (1, 2), (2, 0)],
            segments=[
                _seg(0, 0, 3, 0),
                _seg(3, 0, 1.5, 2),
                _seg(1.5, 2, 0, 0),
            ],
        )
        cycles = detect_rooms(g)
        assert len(cycles) == 1
        assert len(cycles[0]) == 3  # 3 nœuds dans le cycle

    def test_two_adjacent_rectangles_two_cycles(self) -> None:
        """Deux rectangles partageant un côté → 2 cycles."""
        # Nœuds : 0=(0,0), 1=(4,0), 2=(4,3), 3=(0,3),
        #          4=(8,0), 5=(8,3)
        g = WallGraph(
            nodes=[(0,0),(4,0),(4,3),(0,3),(8,0),(8,3)],
            edges=[(0,1),(1,2),(2,3),(3,0),(1,4),(4,5),(5,2)],
            segments=[
                _seg(0,0,4,0), _seg(4,0,4,3), _seg(4,3,0,3), _seg(0,3,0,0),
                _seg(4,0,8,0), _seg(8,0,8,3), _seg(8,3,4,3),
            ],
        )
        cycles = detect_rooms(g)
        assert len(cycles) == 2

    def test_empty_graph_returns_empty(self) -> None:
        cycles = detect_rooms(WallGraph())
        assert cycles == []
