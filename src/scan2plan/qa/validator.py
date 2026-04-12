"""Validations de cohérence géométrique du plan produit (V1)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from scan2plan.qa.metrics import QAReport

if TYPE_CHECKING:
    from scan2plan.detection.openings import Opening
    from scan2plan.vectorization.topology import WallGraph

logger = logging.getLogger(__name__)

# Seuils de validation
_MICRO_SEGMENT_M = 0.10  # Segment < 10 cm = micro-segment parasite
_LOW_CONFIDENCE = 0.50  # Confiance < 0.5 = zone à vérifier
_MIN_ROOM_AREA_M2 = 0.25  # Pièce < 0.25 m² = probablement un artefact
_MAX_ROOM_AREA_M2 = 500.0  # Pièce > 500 m² = probablement erronée

# Pénalités de score
_PENALTY_GAP = 10
_PENALTY_ORPHAN = 5
_PENALTY_MICRO = 2
_PENALTY_NO_ROOMS = 5
_BONUS_MULTI_ROOMS = 10


def validate_plan(
    wall_graph: "WallGraph",
    openings: "list[Opening]",
) -> QAReport:
    """Analyse le graphe de murs et produit un rapport de qualité.

    Métriques calculées :
    - Longueur totale des murs.
    - Nombre de segments et de pièces.
    - Gaps (nœuds de degré 1 = extremité non connectée).
    - Segments orphelins (< seuil de longueur ET degré 1).
    - Micro-segments parasites (< 10 cm).
    - Confiance moyenne et zones à faible confiance.
    - Score global de 0 à 100.

    Args:
        wall_graph: Graphe topologique des murs (WallGraph).
        openings: Liste des ouvertures détectées.

    Returns:
        QAReport complet.
    """
    report = QAReport()
    report.num_openings = len(openings)

    segments = wall_graph.segments
    nodes = wall_graph.nodes
    edges = wall_graph.edges

    if not segments:
        report.warnings.append("Aucun segment de mur dans le graphe.")
        report.score = 0.0
        return report

    # ---- Métriques de base ------------------------------------------------
    report.num_segments = len(segments)
    report.total_wall_length = sum(s.length for s in segments)

    confidences = [s.confidence for s in segments]
    report.avg_confidence = sum(confidences) / len(confidences)

    # Zones à faible confiance : centroïde du segment
    for seg in segments:
        if seg.confidence < _LOW_CONFIDENCE:
            cx = (seg.x1 + seg.x2) / 2.0
            cy = (seg.y1 + seg.y2) / 2.0
            report.low_confidence_zones.append((cx, cy))

    # ---- Micro-segments ---------------------------------------------------
    report.num_micro_segments = sum(1 for s in segments if s.length < _MICRO_SEGMENT_M)

    # ---- Degrés des nœuds -------------------------------------------------
    degree = _compute_node_degrees(nodes, edges)
    degree_one_nodes = {n for n, d in degree.items() if d == 1}
    report.num_gaps = len(degree_one_nodes)

    # Orphelins = segments dont les DEUX nœuds sont de degré 1
    report.num_orphan_segments = _count_orphan_segments(edges, degree_one_nodes)

    # ---- Pièces détectées -------------------------------------------------
    from scan2plan.vectorization.topology import detect_rooms

    rooms = detect_rooms(wall_graph)
    report.num_rooms_detected = len(rooms)

    # ---- Avertissements métier --------------------------------------------
    _add_warnings(report)

    # ---- Score ------------------------------------------------------------
    report.score = _compute_score(report)

    logger.info(
        "QA : score=%.0f, segments=%d, pièces=%d, gaps=%d, orphelins=%d, micro=%d.",
        report.score,
        report.num_segments,
        report.num_rooms_detected,
        report.num_gaps,
        report.num_orphan_segments,
        report.num_micro_segments,
    )
    return report


def generate_qa_report(report: QAReport, output_path: Path) -> None:
    """Sérialise le rapport QA en JSON.

    Args:
        report: Rapport QA à écrire.
        output_path: Fichier de sortie (.json).
    """
    data = {
        "score": round(report.score, 1),
        "total_wall_length_m": round(report.total_wall_length, 3),
        "num_segments": report.num_segments,
        "num_rooms_detected": report.num_rooms_detected,
        "num_openings": report.num_openings,
        "num_gaps": report.num_gaps,
        "num_orphan_segments": report.num_orphan_segments,
        "num_micro_segments": report.num_micro_segments,
        "avg_confidence": round(report.avg_confidence, 3),
        "low_confidence_zones": [
            {"x": round(x, 3), "y": round(y, 3)} for x, y in report.low_confidence_zones
        ],
        "warnings": report.warnings,
    }
    output_path = output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Rapport QA écrit : %s", output_path)


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------


def _compute_node_degrees(
    nodes: list[tuple[float, float]],
    edges: list[tuple[int, int]],
) -> dict[int, int]:
    """Calcule le degré de chaque nœud.

    Args:
        nodes: Liste de positions (x, y).
        edges: Liste de paires (i, j).

    Returns:
        Dictionnaire {index_nœud: degré}.
    """
    degree: dict[int, int] = dict.fromkeys(range(len(nodes)), 0)
    for i, j in edges:
        degree[i] = degree.get(i, 0) + 1
        degree[j] = degree.get(j, 0) + 1
    return degree


def _count_orphan_segments(
    edges: list[tuple[int, int]],
    degree_one_nodes: set[int],
) -> int:
    """Compte les segments dont les deux nœuds sont de degré 1 (= orphelins).

    Args:
        edges: Arêtes du graphe.
        degree_one_nodes: Ensemble des nœuds de degré 1.

    Returns:
        Nombre de segments orphelins.
    """
    return sum(1 for i, j in edges if i in degree_one_nodes and j in degree_one_nodes)


def _add_warnings(report: QAReport) -> None:
    """Ajoute les avertissements métier au rapport.

    Args:
        report: Rapport en cours de construction (modifié en place).
    """
    if report.num_gaps > 0:
        report.warnings.append(
            f"{report.num_gaps} extrémité(s) de mur non connectée(s) (gaps). "
            "Vérifiez les coins et jonctions."
        )
    if report.num_orphan_segments > 0:
        report.warnings.append(
            f"{report.num_orphan_segments} segment(s) orphelin(s) isolé(s). "
            "Probablement du mobilier ou un artefact Hough."
        )
    if report.num_micro_segments > 0:
        report.warnings.append(
            f"{report.num_micro_segments} micro-segment(s) < 10 cm détecté(s). "
            "Appliquer clean_topology() ou augmenter min_segment_length."
        )
    if report.num_rooms_detected == 0:
        report.warnings.append(
            "Aucune pièce fermée détectée. "
            "Le plan manque probablement de murs ou contient des gaps importants."
        )
    if report.avg_confidence < _LOW_CONFIDENCE:
        report.warnings.append(
            f"Confiance moyenne faible ({report.avg_confidence:.2f}). "
            "Résultat à valider attentivement."
        )
    if report.low_confidence_zones:
        report.warnings.append(
            f"{len(report.low_confidence_zones)} zone(s) à confiance < {_LOW_CONFIDENCE:.0%}. "
            "Calque INCERTAIN à vérifier dans AutoCAD."
        )


def _compute_score(report: QAReport) -> float:
    """Calcule le score QA de 0 à 100.

    Règles :
    - Départ à 100.
    - -10 par gap de mur.
    - -5 par segment orphelin.
    - -2 par micro-segment.
    - -5 si aucune pièce fermée.
    - +10 si ≥ 2 pièces détectées.
    - Score minimal = 0.

    Args:
        report: Rapport avec toutes les métriques calculées.

    Returns:
        Score entre 0 et 100.
    """
    score = 100.0
    score -= report.num_gaps * _PENALTY_GAP
    score -= report.num_orphan_segments * _PENALTY_ORPHAN
    score -= report.num_micro_segments * _PENALTY_MICRO
    if report.num_rooms_detected == 0:
        score -= _PENALTY_NO_ROOMS
    if report.num_rooms_detected >= 2:
        score += _BONUS_MULTI_ROOMS
    return max(0.0, min(100.0, score))
