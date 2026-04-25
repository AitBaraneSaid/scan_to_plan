"""Tests unitaires pour clean_parasites.

Vérifie que les segments parasites courts et isolés sont supprimés
sans toucher aux cloisons courtes légitimes.
"""

from __future__ import annotations

import pytest

from scan2plan.detection.line_detection import DetectedSegment
from scan2plan.detection.segment_cleanup import clean_parasites


def seg(
    x1: float, y1: float, x2: float, y2: float, conf: float = 0.9
) -> DetectedSegment:
    return DetectedSegment(x1=x1, y1=y1, x2=x2, y2=y2, source_slice="high", confidence=conf)


class TestCleanParasites:
    def test_remove_isolated_short(self) -> None:
        """Segment de 8 cm sans aucun voisin → supprimé."""
        parasite = seg(0.0, 0.0, 0.08, 0.0)
        result = clean_parasites([parasite], min_length=0.15)
        assert len(result) == 0

    def test_keep_short_with_parallel(self) -> None:
        """Segment de 10 cm avec un parallèle à 15 cm et chevauchement → gardé (face de cloison)."""
        short = seg(0.0, 0.0, 0.10, 0.0)
        parallel = seg(0.0, 0.15, 2.0, 0.15)   # parallèle, 15 cm de décalage
        result = clean_parasites([short, parallel], min_length=0.15)
        # short doit être conservé car il a un voisin parallèle avec chevauchement
        assert any(abs(r.length - short.length) < 0.01 for r in result)

    def test_keep_short_connected(self) -> None:
        """Segment de 12 cm avec un perpendiculaire à 5 cm d'une extrémité → gardé."""
        short = seg(0.0, 0.0, 0.12, 0.0)
        # Perpendiculaire partant de l'extrémité droite du short
        perp = seg(0.12, 0.0, 0.12, 2.0)
        result = clean_parasites([short, perp], min_length=0.15)
        assert len(result) == 2

    def test_keep_all_long(self) -> None:
        """Tous les segments ≥ 15 cm → tous gardés, aucune modification."""
        segments = [
            seg(0.0, 0.0, 0.20, 0.0),
            seg(0.0, 1.0, 3.0, 1.0),
            seg(2.0, 0.0, 2.0, 4.0),
        ]
        result = clean_parasites(segments, min_length=0.15)
        assert len(result) == 3

    def test_real_case(self) -> None:
        """10 vrais segments + 5 parasites → seuls les 5 parasites sont supprimés."""
        # 4 murs longs formant un rectangle
        walls = [
            seg(0.0, 0.0, 5.0, 0.0),    # mur bas
            seg(0.0, 4.0, 5.0, 4.0),    # mur haut
            seg(0.0, 0.0, 0.0, 4.0),    # mur gauche
            seg(5.0, 0.0, 5.0, 4.0),    # mur droit
        ]
        # 2 cloisons intérieures (doubles faces)
        partitions = [
            seg(2.0, 0.0, 2.0, 1.8),    # cloison face A
            seg(2.15, 0.0, 2.15, 1.8),  # cloison face B (face parallèle)
            seg(2.0, 2.2, 2.0, 4.0),    # cloison face A suite
            seg(2.15, 2.2, 2.15, 4.0),  # cloison face B suite
        ]
        # 2 embrasures courtes connectées perpendiculairement
        jambs = [
            seg(2.0, 1.8, 2.15, 1.8),   # tableau de porte (12 cm, connecté)
        ]
        # 5 parasites : courts et isolés
        parasites = [
            seg(8.0, 8.0, 8.07, 8.0),   # 7 cm, loin de tout
            seg(9.0, 9.0, 9.08, 9.0),   # 8 cm, loin de tout
            seg(10.0, 0.0, 10.09, 0.0), # 9 cm, loin de tout
            seg(0.0, 10.0, 0.10, 10.0), # 10 cm, loin de tout
            seg(7.0, 7.0, 7.06, 7.0),   # 6 cm, loin de tout
        ]

        all_segments = walls + partitions + jambs + parasites
        result = clean_parasites(all_segments, min_length=0.15)

        # Les 5 parasites doivent être supprimés
        assert len(result) == len(all_segments) - 5

        # Vérifier que les parasites ne sont plus dans le résultat
        parasite_coords = {(p.x1, p.y1, p.x2, p.y2) for p in parasites}
        result_coords = {(r.x1, r.y1, r.x2, r.y2) for r in result}
        assert parasite_coords.isdisjoint(result_coords)

    def test_empty_input(self) -> None:
        """Liste vide → liste vide."""
        assert clean_parasites([]) == []

    def test_single_long_segment(self) -> None:
        """Un seul segment long → retourné inchangé."""
        s = seg(0.0, 0.0, 3.0, 0.0)
        result = clean_parasites([s])
        assert len(result) == 1

    def test_single_short_isolated(self) -> None:
        """Un seul segment court sans aucun voisin → supprimé."""
        s = seg(0.0, 0.0, 0.05, 0.0)
        result = clean_parasites([s], min_length=0.15)
        assert len(result) == 0

    def test_parallel_without_overlap_not_saved(self) -> None:
        """Segment court avec parallèle mais SANS chevauchement longitudinal → supprimé.

        Le parallèle est entièrement à côté (bout-à-bout), pas en face.
        """
        short = seg(0.0, 0.0, 0.10, 0.0)
        # Parallèle qui commence là où le court finit → pas de chevauchement
        parallel_no_overlap = seg(0.50, 0.15, 2.0, 0.15)
        result = clean_parasites([short, parallel_no_overlap], min_length=0.15)
        # short n'a pas de chevauchement avec parallel_no_overlap → supprimé
        result_lengths = [r.length for r in result]
        assert not any(abs(l - 0.10) < 0.01 for l in result_lengths)

    def test_perpendicular_far_not_saved(self) -> None:
        """Segment court avec perpendiculaire trop loin → supprimé."""
        short = seg(0.0, 0.0, 0.10, 0.0)
        # Perpendiculaire à 50 cm de distance (> 10 cm de seuil)
        perp_far = seg(0.60, 0.0, 0.60, 2.0)
        result = clean_parasites(
            [short, perp_far],
            min_length=0.15,
            perpendicular_search_distance=0.10,
        )
        result_lengths = [r.length for r in result]
        assert not any(abs(l - 0.10) < 0.01 for l in result_lengths)

    def test_no_removal_above_min_length(self) -> None:
        """Aucun segment au-dessus de min_length n'est supprimé, même s'il est isolé."""
        isolated_but_long = seg(50.0, 50.0, 50.20, 50.0)  # 20 cm, loin de tout
        result = clean_parasites([isolated_but_long], min_length=0.15)
        assert len(result) == 1
