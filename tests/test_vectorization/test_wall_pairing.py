"""Tests du module wall_pairing.

Couvre chaque helper géométrique, chaque branche de l'algorithme,
et chaque cas métier critique. Aucun test ne dépend de fichiers scan
réels — tous les segments sont construits programmatiquement.
"""

from __future__ import annotations

import numpy as np
import pytest

from scan2plan.vectorization.wall_pairing import (
    FacePair,
    FacePairingResult,
    PairingConfig,
    Segment,
    _angle_diff,
    _build_corridor_polygon,
    _build_median_segment,
    _compute_overlap,
    _corridor_is_free,
    _perpendicular_distance,
    apply_median_pairing,
    find_wall_pairs,
    pair_wall_faces,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def seg(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str = "",
    confidence: float = 1.0,
) -> Segment:
    """Raccourci de construction d'un Segment."""
    return Segment(x1=x1, y1=y1, x2=x2, y2=y2, label=label, confidence=confidence)


# ---------------------------------------------------------------------------
# TestAngleDiff
# ---------------------------------------------------------------------------


class TestAngleDiff:
    def test_parallel(self) -> None:
        """Deux angles identiques → différence nulle."""
        assert _angle_diff(0.0, 0.0) == pytest.approx(0.0)

    def test_perpendicular(self) -> None:
        """Angle de 90° → différence = π/2."""
        assert _angle_diff(0.0, np.pi / 2) == pytest.approx(np.pi / 2)

    def test_antiparallel(self) -> None:
        """Un segment à 0 et un à π sont bidirectionnellement parallèles → diff ≈ 0."""
        assert _angle_diff(0.0, np.pi) == pytest.approx(0.0, abs=1e-9)

    def test_near_pi(self) -> None:
        """0.01 rad et π-0.01 rad → diff ≈ 0.02 rad."""
        assert _angle_diff(0.01, np.pi - 0.01) == pytest.approx(0.02, abs=1e-9)

    def test_small_diff(self) -> None:
        """Petite différence d'angle."""
        assert _angle_diff(0.5, 0.52) == pytest.approx(0.02, abs=1e-9)


# ---------------------------------------------------------------------------
# TestPerpendicularDistance
# ---------------------------------------------------------------------------


class TestPerpendicularDistance:
    def test_horizontal_20cm(self) -> None:
        """Deux segments horizontaux distants de 20 cm."""
        sa = seg(0, 0, 4, 0)
        sb = seg(0, 0.20, 4, 0.20)
        assert _perpendicular_distance(sa, sb) == pytest.approx(0.20, abs=1e-6)

    def test_vertical_15cm(self) -> None:
        """Deux segments verticaux distants de 15 cm."""
        sa = seg(0, 0, 0, 3)
        sb = seg(0.15, 0, 0.15, 3)
        assert _perpendicular_distance(sa, sb) == pytest.approx(0.15, abs=1e-6)

    def test_same_line(self) -> None:
        """Deux segments sur la même droite → distance nulle."""
        sa = seg(0, 0, 4, 0)
        sb = seg(1, 0, 3, 0)
        assert _perpendicular_distance(sa, sb) == pytest.approx(0.0, abs=1e-6)

    def test_diagonal(self) -> None:
        """Segments diagonaux parallèles — distance perpendiculaire exacte."""
        # Deux segments à 45° décalés perpendiculairement de 0.20 m
        # Décalage perp. = (0, d) projeté sur la normale (−sin45, cos45)
        d = 0.20
        sa = seg(0, 0, 3, 3)
        # Décaler sb perpendiculairement à (1/√2, 1/√2) → normal = (−1/√2, 1/√2)
        nx, ny = -1 / np.sqrt(2), 1 / np.sqrt(2)
        sb = seg(d * nx, d * ny, 3 + d * nx, 3 + d * ny)
        assert _perpendicular_distance(sa, sb) == pytest.approx(d, abs=1e-6)


# ---------------------------------------------------------------------------
# TestComputeOverlap
# ---------------------------------------------------------------------------


class TestComputeOverlap:
    def test_full_overlap(self) -> None:
        """Segments identiques → overlap = longueur du segment."""
        sa = seg(0, 0, 4, 0)
        sb = seg(0, 0.20, 4, 0.20)
        overlap, _, _ = _compute_overlap(sa, sb)
        assert overlap == pytest.approx(4.0, abs=1e-6)

    def test_partial_overlap(self) -> None:
        """Sa=[0,4], Sb=[2,6] → overlap = 2.0."""
        sa = seg(0, 0, 4, 0)
        sb = seg(2, 0.20, 6, 0.20)
        overlap, t_start, t_end = _compute_overlap(sa, sb)
        assert overlap == pytest.approx(2.0, abs=1e-6)
        assert t_start == pytest.approx(2.0, abs=1e-6)
        assert t_end == pytest.approx(4.0, abs=1e-6)

    def test_no_overlap(self) -> None:
        """Sa=[0,2], Sb=[3,5] → pas de chevauchement."""
        sa = seg(0, 0, 2, 0)
        sb = seg(3, 0.20, 5, 0.20)
        overlap, _, _ = _compute_overlap(sa, sb)
        assert overlap == pytest.approx(0.0, abs=1e-6)

    def test_contained(self) -> None:
        """Sb=[1,3] entièrement contenu dans Sa=[0,4] → overlap = 2.0."""
        sa = seg(0, 0, 4, 0)
        sb = seg(1, 0.20, 3, 0.20)
        overlap, _, _ = _compute_overlap(sa, sb)
        assert overlap == pytest.approx(2.0, abs=1e-6)

    def test_touching(self) -> None:
        """Sa=[0,2], Sb=[2,4] → ils se touchent, overlap ≈ 0."""
        sa = seg(0, 0, 2, 0)
        sb = seg(2, 0.20, 4, 0.20)
        overlap, _, _ = _compute_overlap(sa, sb)
        assert overlap == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TestCorridorPolygon
# ---------------------------------------------------------------------------


class TestCorridorPolygon:
    def _overlap(self, sa: Segment, sb: Segment) -> tuple[float, float, float]:
        return _compute_overlap(sa, sb)

    def test_builds_valid_polygon(self) -> None:
        """Deux segments parallèles standard → Polygon valide avec area > 0."""
        sa = seg(0, 0, 4, 0)
        sb = seg(0, 0.20, 4, 0.20)
        overlap, t_start, t_end = self._overlap(sa, sb)
        poly = _build_corridor_polygon(sa, sb, t_start, t_end, margin=0.02)
        assert poly is not None
        assert poly.is_valid
        assert poly.area > 0

    def test_too_short_returns_none(self) -> None:
        """Chevauchement de 3 cm avec margin 2 cm → corridor longitudinal = -1 cm → None."""
        sa = seg(0, 0, 0.03, 0)
        sb = seg(0, 0.20, 0.03, 0.20)
        overlap, t_start, t_end = self._overlap(sa, sb)
        poly = _build_corridor_polygon(sa, sb, t_start, t_end, margin=0.02)
        assert poly is None

    def test_too_narrow_returns_none(self) -> None:
        """Distance perpendiculaire 3 cm et margin 2 cm → largeur effective 3-4=-1 cm → None."""
        sa = seg(0, 0, 4, 0)
        sb = seg(0, 0.03, 4, 0.03)
        overlap, t_start, t_end = self._overlap(sa, sb)
        poly = _build_corridor_polygon(sa, sb, t_start, t_end, margin=0.02)
        assert poly is None

    def test_polygon_dimensions(self) -> None:
        """Dimensions du corridor : largeur ≈ dist - 2*margin, longueur ≈ overlap - 2*margin."""
        margin = 0.02
        sa = seg(0, 0, 4, 0)
        sb = seg(0, 0.20, 4, 0.20)
        overlap, t_start, t_end = self._overlap(sa, sb)
        poly = _build_corridor_polygon(sa, sb, t_start, t_end, margin=margin)
        assert poly is not None
        expected_len = (t_end - t_start) - 2 * margin   # 4.0 - 0.04 = 3.96
        expected_wid = 0.20 - 2 * margin                 # 0.20 - 0.04 = 0.16
        assert poly.area == pytest.approx(expected_len * expected_wid, rel=0.05)


# ---------------------------------------------------------------------------
# TestCorridorIsFree
# ---------------------------------------------------------------------------


class TestCorridorIsFree:
    """Tests de la fonction _corridor_is_free."""

    def _setup_corridor(
        self,
        extra_segs: list[Segment],
    ) -> tuple[object, list[Segment], float, float]:
        """Construit un corridor standard 4m × 20cm et retourne le contexte."""
        sa = seg(0, 0, 4, 0)
        sb = seg(0, 0.20, 4, 0.20)
        all_segs = [sa, sb] + extra_segs
        overlap, t_start, t_end = _compute_overlap(sa, sb)
        margin = 0.02
        corridor = _build_corridor_polygon(sa, sb, t_start, t_end, margin=margin)
        pair_angle = sa.angle
        corridor_width = 0.20 - 2 * margin
        return corridor, all_segs, pair_angle, corridor_width

    def test_empty_corridor(self) -> None:
        """Aucun segment tiers → corridor libre."""
        corridor, all_segs, pair_angle, corridor_width = self._setup_corridor([])
        assert _corridor_is_free(corridor, all_segs, (0, 1), pair_angle, corridor_width) is True

    def test_parallel_segment_blocks(self) -> None:
        """Segment parallèle à mi-hauteur → bloque."""
        intrus = seg(1, 0.10, 3, 0.10)
        corridor, all_segs, pair_angle, corridor_width = self._setup_corridor([intrus])
        assert _corridor_is_free(corridor, all_segs, (0, 1), pair_angle, corridor_width) is False

    def test_perpendicular_corner_wall_ok(self) -> None:
        """Mur perpendiculaire au coin qui traverse l'épaisseur → autorisé."""
        corner = seg(0, -1, 0, 1)   # traverse au coin, intersection ≈ 16 cm ≤ corridor_width + 5cm
        corridor, all_segs, pair_angle, corridor_width = self._setup_corridor([corner])
        assert _corridor_is_free(corridor, all_segs, (0, 1), pair_angle, corridor_width) is True

    def test_perpendicular_at_middle_ok(self) -> None:
        """Mur perpendiculaire au milieu du corridor → autorisé (traverse sur ~16 cm)."""
        cross = seg(2, -0.5, 2, 0.70)
        corridor, all_segs, pair_angle, corridor_width = self._setup_corridor([cross])
        assert _corridor_is_free(corridor, all_segs, (0, 1), pair_angle, corridor_width) is True

    def test_tiny_segment_ignored(self) -> None:
        """Micro-segment de 3 cm à l'intérieur → trop court, ignoré."""
        tiny = seg(2, 0.10, 2.03, 0.10)
        corridor, all_segs, pair_angle, corridor_width = self._setup_corridor([tiny])
        assert _corridor_is_free(corridor, all_segs, (0, 1), pair_angle, corridor_width) is True

    def test_long_parallel_inside_blocks(self) -> None:
        """Segment parallèle de 3 m à l'intérieur du corridor → bloque."""
        long_intrus = seg(0.5, 0.10, 3.5, 0.10)
        corridor, all_segs, pair_angle, corridor_width = self._setup_corridor([long_intrus])
        assert _corridor_is_free(corridor, all_segs, (0, 1), pair_angle, corridor_width) is False


# ---------------------------------------------------------------------------
# TestBuildMedianSegment
# ---------------------------------------------------------------------------


class TestBuildMedianSegment:
    def test_horizontal_median(self) -> None:
        """Mur horizontal 20 cm → médian à y = 0.10, longueur = 4.0 m."""
        sa = seg(0, 0, 4, 0)
        sb = seg(0, 0.20, 4, 0.20)
        med = _build_median_segment(sa, sb)
        assert med.length == pytest.approx(4.0, abs=0.01)
        assert (med.y1 + med.y2) / 2 == pytest.approx(0.10, abs=1e-6)

    def test_vertical_median(self) -> None:
        """Mur vertical 20 cm → médian à x = 0.10, longueur = 3.0 m."""
        sa = seg(0, 0, 0, 3)
        sb = seg(0.20, 0, 0.20, 3)
        med = _build_median_segment(sa, sb)
        assert med.length == pytest.approx(3.0, abs=0.01)
        assert (med.x1 + med.x2) / 2 == pytest.approx(0.10, abs=1e-6)

    def test_covers_full_extent(self) -> None:
        """Sa=[0,4], Sb=[1,5] décalés → médian couvre l'emprise totale [0,5]."""
        sa = seg(0, 0, 4, 0)
        sb = seg(1, 0.20, 5, 0.20)
        med = _build_median_segment(sa, sb)
        assert med.length == pytest.approx(5.0, abs=0.01)

    def test_label_is_wall_paired(self) -> None:
        """Le segment médian porte le label 'wall_paired'."""
        sa = seg(0, 0, 4, 0)
        sb = seg(0, 0.20, 4, 0.20)
        med = _build_median_segment(sa, sb)
        assert med.label == "wall_paired"

    def test_confidence_is_max(self) -> None:
        """La confiance du médian est le max des deux faces."""
        sa = seg(0, 0, 4, 0, confidence=0.8)
        sb = seg(0, 0.20, 4, 0.20, confidence=0.9)
        med = _build_median_segment(sa, sb)
        assert med.confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# TestFindWallPairs
# ---------------------------------------------------------------------------


class TestFindWallPairs:
    def test_simple_pair(self) -> None:
        """Deux faces d'un mur de 20 cm → 1 paire confirmée."""
        r = find_wall_pairs([seg(0, 0, 4, 0), seg(0, 0.20, 4, 0.20)])
        assert r.num_pairs_confirmed == 1
        assert r.pairs[0].thickness == pytest.approx(0.20, abs=0.01)
        assert len(r.unpaired_segments) == 0

    def test_perpendicular_not_paired(self) -> None:
        """Deux segments perpendiculaires → 0 paires."""
        r = find_wall_pairs([seg(0, 0, 4, 0), seg(2, -1, 2, 1)])
        assert r.num_pairs_confirmed == 0
        assert len(r.unpaired_segments) == 2

    def test_too_far_apart(self) -> None:
        """Parallèles à 50 cm → au-delà de max_distance (30 cm), pas une paire."""
        r = find_wall_pairs([seg(0, 0, 4, 0), seg(0, 0.50, 4, 0.50)])
        assert r.num_pairs_confirmed == 0

    def test_too_close(self) -> None:
        """Parallèles à 2 cm → en dessous de min_distance (4 cm), c'est de la fusion."""
        r = find_wall_pairs([seg(0, 0, 4, 0), seg(0, 0.02, 4, 0.02)])
        assert r.num_pairs_confirmed == 0

    def test_no_longitudinal_overlap(self) -> None:
        """Parallèles mais sans chevauchement longitudinal → pas de paire."""
        r = find_wall_pairs([seg(0, 0, 2, 0), seg(3, 0.20, 5, 0.20)])
        assert r.num_pairs_confirmed == 0

    def test_short_jambage_excluded(self) -> None:
        """Jambage de porte de 10 cm → trop court (< min_segment_length 20 cm)."""
        config = PairingConfig(min_segment_length=0.20)
        r = find_wall_pairs([seg(0, 0, 4, 0), seg(2, 0.15, 2.10, 0.15)], config)
        assert r.num_pairs_confirmed == 0

    def test_corridor_blocked_by_parallel_intruder(self) -> None:
        """Segment parallèle intercalé → la paire 0↔0.20 est bloquée."""
        segs = [
            seg(0, 0, 4, 0),       # face A
            seg(0, 0.20, 4, 0.20), # face B
            seg(1, 0.10, 3, 0.10), # parallèle intercalé
        ]
        r = find_wall_pairs(segs)
        # La paire 0.0↔0.20 (20 cm) doit être bloquée par le parallèle intérieur
        for p in r.pairs:
            assert p.thickness < 0.18, (
                f"La paire de 20 cm ne devrait pas être confirmée, épaisseur={p.thickness}"
            )

    def test_four_walls_rectangle(self) -> None:
        """Pièce rectangulaire 4×3m, murs de 20cm → 4 paires.

        Test critique : les murs perpendiculaires aux coins traversent les
        corridors mais ne doivent PAS bloquer les paires.
        """
        segs = [
            seg(0, 0, 4, 0),       seg(0, 0.20, 4, 0.20),    # bas
            seg(0, 3, 4, 3),       seg(0, 3.20, 4, 3.20),    # haut
            seg(0, 0, 0, 3),       seg(0.20, 0, 0.20, 3),    # gauche
            seg(4, 0, 4, 3),       seg(4.20, 0, 4.20, 3),    # droite
        ]
        r = find_wall_pairs(segs)
        assert r.num_pairs_confirmed == 4
        assert len(r.unpaired_segments) == 0

    def test_conflict_resolution(self) -> None:
        """Trois parallèles dont deux candidats conflictuels → 1 paire, 1 non apparié."""
        # seg[0] est candidat avec seg[1] ET avec seg[2]
        # La meilleure paire (score max) gagne ; l'autre reste non appariée
        segs = [
            seg(0, 0, 4, 0),       # ref
            seg(0, 0.15, 4, 0.15), # à 15 cm
            seg(0, 0.25, 4, 0.25), # à 25 cm
        ]
        r = find_wall_pairs(segs)
        assert r.num_pairs_confirmed == 1
        assert len(r.unpaired_segments) == 1

    def test_two_partitions_with_cross_wall(self) -> None:
        """Deux cloisons de 7 cm dos à dos avec mur de refend → 2 paires.

        Attendu : paire (0cm↔7cm) et paire (13cm↔20cm). La paire (0cm↔20cm)
        et (7cm↔13cm) sont bloquées par le refend perpendiculaire.
        """
        segs = [
            seg(0, 0, 4, 0),           # Face ext cloison 1
            seg(0, 0.07, 4, 0.07),     # Face int cloison 1
            seg(0, 0.13, 4, 0.13),     # Face int cloison 2
            seg(0, 0.20, 4, 0.20),     # Face ext cloison 2
            seg(2, -0.1, 2, 0.30),     # Refend perpendiculaire
        ]
        r = find_wall_pairs(segs)
        assert r.num_pairs_confirmed == 2

    def test_two_rooms_shared_wall(self) -> None:
        """Deux pièces adjacentes avec mur partagé → 7 paires."""
        segs = [
            seg(0, 0, 4, 0),          seg(0, 0.20, 4, 0.20),       # P1 bas
            seg(0, 3, 4, 3),          seg(0, 3.20, 4, 3.20),       # P1 haut
            seg(0, 0, 0, 3),          seg(0.20, 0, 0.20, 3),       # P1 gauche
            seg(4, 0, 4, 3),          seg(4.20, 0, 4.20, 3),       # mur partagé
            seg(4.20, 0, 8.20, 0),    seg(4.20, 0.20, 8.20, 0.20), # P2 bas
            seg(4.20, 3, 8.20, 3),    seg(4.20, 3.20, 8.20, 3.20), # P2 haut
            seg(8.20, 0, 8.20, 3),    seg(8.40, 0, 8.40, 3),       # P2 droite
        ]
        r = find_wall_pairs(segs)
        assert r.num_pairs_confirmed == 7


# ---------------------------------------------------------------------------
# TestRobustness
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_noisy_positions(self) -> None:
        """Segments avec bruit gaussien ±2 cm → paire quand même détectée."""
        rng = np.random.default_rng(42)
        n = 0.02
        sa = seg(
            float(rng.uniform(-n, n)), float(rng.uniform(-n, n)),
            4.0 + float(rng.uniform(-n, n)), float(rng.uniform(-n, n)),
        )
        sb = seg(
            float(rng.uniform(-n, n)), 0.20 + float(rng.uniform(-n, n)),
            4.0 + float(rng.uniform(-n, n)), 0.20 + float(rng.uniform(-n, n)),
        )
        r = find_wall_pairs([sa, sb])
        assert r.num_pairs_confirmed == 1

    def test_2_degrees_ok(self) -> None:
        """Segments à 2° d'écart (< tolérance par défaut 3°) → paire confirmée."""
        a = np.deg2rad(2)
        r = find_wall_pairs([
            seg(0, 0, 4, 0),
            seg(0, 0.20, 4 * np.cos(a), 0.20 + 4 * np.sin(a)),
        ])
        assert r.num_pairs_confirmed == 1

    def test_5_degrees_rejected(self) -> None:
        """Segments à 5° d'écart (> tolérance par défaut 3°) → pas de paire."""
        a = np.deg2rad(5)
        r = find_wall_pairs([
            seg(0, 0, 4, 0),
            seg(0, 0.20, 4 * np.cos(a), 0.20 + 4 * np.sin(a)),
        ])
        assert r.num_pairs_confirmed == 0

    def test_thin_partition_5cm(self) -> None:
        """Cloison de 5 cm (> min_distance 4 cm) → pairée."""
        r = find_wall_pairs([seg(0, 0, 4, 0), seg(0, 0.05, 4, 0.05)])
        assert r.num_pairs_confirmed == 1
        assert r.pairs[0].thickness == pytest.approx(0.05, abs=0.01)

    def test_thick_wall_25cm(self) -> None:
        """Mur porteur de 25 cm (< max_distance 30 cm) → pairé."""
        r = find_wall_pairs([seg(0, 0, 4, 0), seg(0, 0.25, 4, 0.25)])
        assert r.num_pairs_confirmed == 1

    def test_empty_input(self) -> None:
        """Liste vide → 0 paires, 0 non appariés."""
        r = find_wall_pairs([])
        assert r.num_pairs_confirmed == 0
        assert len(r.unpaired_segments) == 0

    def test_single_segment(self) -> None:
        """Un seul segment → 0 paires, 1 non apparié."""
        r = find_wall_pairs([seg(0, 0, 4, 0)])
        assert r.num_pairs_confirmed == 0
        assert len(r.unpaired_segments) == 1


# ---------------------------------------------------------------------------
# TestApplyPairing
# ---------------------------------------------------------------------------


class TestApplyPairing:
    def test_reduces_count(self) -> None:
        """1 paire + 1 perpendiculaire → 2 segments en sortie."""
        out = apply_median_pairing([
            seg(0, 0, 4, 0),
            seg(0, 0.20, 4, 0.20),
            seg(2, -1, 2, 1),
        ])
        assert len(out) == 2

    def test_median_has_correct_label(self) -> None:
        """Le segment médian en sortie porte le label 'wall_paired'."""
        out = apply_median_pairing([seg(0, 0, 4, 0), seg(0, 0.20, 4, 0.20)])
        assert len(out) == 1
        assert out[0].label == "wall_paired"

    def test_passthrough_when_no_pairs(self) -> None:
        """Aucune paire possible → tous les segments passent tels quels."""
        out = apply_median_pairing([seg(0, 0, 4, 0), seg(2, -1, 2, 1)])
        assert len(out) == 2


# ---------------------------------------------------------------------------
# TestPairWallFaces — mode sans fusion (segments originaux conservés)
# ---------------------------------------------------------------------------


class TestPairWallFaces:
    def test_segments_unchanged(self) -> None:
        """Les segments en sortie sont IDENTIQUES aux segments en entrée."""
        segments = [
            seg(0, 0, 4, 0),       # face_a mur horizontal
            seg(0, 0.20, 4, 0.20), # face_b mur horizontal
            seg(2, -1, 2, 1),      # mur perpendiculaire non appariable
        ]
        result = pair_wall_faces(segments)
        # all_segments = même liste (même contenu)
        assert len(result.all_segments) == len(segments)
        for orig, out in zip(segments, result.all_segments):
            assert out.x1 == orig.x1
            assert out.y1 == orig.y1
            assert out.x2 == orig.x2
            assert out.y2 == orig.y2

    def test_all_segments_count(self) -> None:
        """all_segments contient TOUS les segments, pairés et non pairés."""
        segments = [
            seg(0, 0, 4, 0),
            seg(0, 0.20, 4, 0.20),
            seg(2, -1, 2, 1),
            seg(10, 10, 12, 10),   # isolé, loin de tout
        ]
        result = pair_wall_faces(segments)
        assert len(result.all_segments) == 4

    def test_pairs_detected_rectangle(self) -> None:
        """Pièce rectangulaire 4×3m avec murs de 20cm → 4 paires détectées."""
        cfg = PairingConfig(
            min_segment_length=0.10,
            min_overlap_abs=0.10,
            min_overlap_ratio=0.10,
        )
        # 4 murs × 2 faces chacun
        segments = [
            # Mur bas (y=0 / y=0.20)
            seg(0, 0, 4, 0),
            seg(0, 0.20, 4, 0.20),
            # Mur haut (y=3 / y=3.20)
            seg(0, 3.0, 4, 3.0),
            seg(0, 3.20, 4, 3.20),
            # Mur gauche (x=0 / x=0.20)
            seg(0, 0, 0, 3),
            seg(0.20, 0, 0.20, 3),
            # Mur droit (x=4 / x=4.20)
            seg(4.0, 0, 4.0, 3),
            seg(4.20, 0, 4.20, 3),
        ]
        result = pair_wall_faces(segments, cfg)
        assert result.num_pairs == 4

    def test_thickness_correct(self) -> None:
        """L'épaisseur de la paire correspond à la distance entre les faces."""
        segments = [
            seg(0, 0, 4, 0),       # y=0
            seg(0, 0.18, 4, 0.18), # y=0.18 → épaisseur = 0.18 m
        ]
        cfg = PairingConfig(min_overlap_abs=0.10, min_overlap_ratio=0.10)
        result = pair_wall_faces(segments, cfg)
        assert len(result.paired_faces) == 1
        assert abs(result.paired_faces[0].thickness - 0.18) < 0.005

    def test_unpaired_preserved(self) -> None:
        """Les segments non appariés sont dans unpaired_segments."""
        segments = [
            seg(0, 0, 4, 0),
            seg(0, 0.20, 4, 0.20),
            seg(2, -1, 2, 1),  # perpendiculaire → non appariable
        ]
        result = pair_wall_faces(segments)
        assert len(result.unpaired_segments) == 1
        assert result.unpaired_segments[0].x1 == pytest.approx(2.0)

    def test_returns_facepairingresult(self) -> None:
        """pair_wall_faces retourne un FacePairingResult."""
        result = pair_wall_faces([seg(0, 0, 4, 0), seg(0, 0.20, 4, 0.20)])
        assert isinstance(result, FacePairingResult)
        assert isinstance(result.paired_faces[0], FacePair)

    def test_no_pairs_all_unpaired(self) -> None:
        """Segments non appariables → paired_faces vide, tout dans unpaired."""
        segments = [seg(0, 0, 4, 0), seg(2, -1, 2, 1)]
        result = pair_wall_faces(segments)
        assert result.num_pairs == 0
        assert len(result.unpaired_segments) == 2
        assert len(result.all_segments) == 2

    def test_empty_input(self) -> None:
        """Liste vide → résultat vide cohérent."""
        result = pair_wall_faces([])
        assert result.num_pairs == 0
        assert result.paired_faces == []
        assert result.unpaired_segments == []
        assert result.all_segments == []

    def test_face_a_and_b_are_original_objects(self) -> None:
        """face_a et face_b dans FacePair sont les mêmes objets que l'entrée."""
        fa = seg(0, 0, 4, 0)
        fb = seg(0, 0.20, 4, 0.20)
        cfg = PairingConfig(min_overlap_abs=0.10, min_overlap_ratio=0.10)
        result = pair_wall_faces([fa, fb], cfg)
        assert len(result.paired_faces) == 1
        pair = result.paired_faces[0]
        # Les coordonnées doivent correspondre exactement aux originaux
        assert pair.face_a.x1 == fa.x1 and pair.face_a.y1 == fa.y1
        assert pair.face_b.x1 == fb.x1 and pair.face_b.y1 == fb.y1
