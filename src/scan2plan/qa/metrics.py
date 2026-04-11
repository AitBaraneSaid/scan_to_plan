"""Métriques de qualité sur le plan produit."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QAReport:
    """Rapport de contrôle qualité du plan produit (V1).

    Attributes:
        total_wall_length: Longueur totale des murs (mètres).
        num_segments: Nombre de segments dans le graphe.
        num_rooms_detected: Nombre de cycles (pièces fermées) détectés.
        num_openings: Nombre d'ouvertures (portes + fenêtres).
        num_gaps: Nombre de gaps entre segments de murs (nœuds de degré 1).
        num_orphan_segments: Nombre de segments orphelins (pendants).
        num_micro_segments: Nombre de segments < 10 cm.
        avg_confidence: Confiance moyenne des segments.
        low_confidence_zones: Positions (x, y) des segments à faible confiance.
        warnings: Messages d'avertissement pour l'opérateur.
        score: Score QA de 0 à 100 (100 = plan parfait).
    """

    total_wall_length: float = 0.0
    num_segments: int = 0
    num_rooms_detected: int = 0
    num_openings: int = 0
    num_gaps: int = 0
    num_orphan_segments: int = 0
    num_micro_segments: int = 0
    avg_confidence: float = 0.0
    low_confidence_zones: list[tuple[float, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    score: float = 100.0

    @property
    def is_valid(self) -> bool:
        """True si le score est suffisant pour un usage production."""
        return self.score >= 50.0

    def summary(self) -> str:
        """Résumé lisible du rapport QA."""
        lines = [
            f"Score QA : {self.score:.0f}/100",
            f"  Segments : {self.num_segments} ({self.total_wall_length:.1f} m total)",
            f"  Pièces   : {self.num_rooms_detected}",
            f"  Ouvertures : {self.num_openings}",
            f"  Gaps : {self.num_gaps}  |  Orphelins : {self.num_orphan_segments}"
            f"  |  Micro-segments : {self.num_micro_segments}",
            f"  Confiance moy. : {self.avg_confidence:.2f}",
        ]
        if self.warnings:
            lines.append("  Avertissements :")
            lines.extend(f"    - {w}" for w in self.warnings)
        return "\n".join(lines)
