"""QA avancé : scoring par zone géographique du plan.

Découpe le plan en cellules de grille, calcule un score de confiance par cellule
basé sur quatre critères, produit une carte de chaleur et exporte les zones faibles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from scan2plan.detection.line_detection import DetectedSegment
    from scan2plan.detection.openings import Opening
    from scan2plan.slicing.density_map import DensityMapResult

logger = logging.getLogger(__name__)

# Score de confiance en dessous duquel une zone est signalée
_LOW_CONFIDENCE_THRESHOLD = 0.4

# Taille de cellule de grille par défaut (mètres)
_DEFAULT_CELL_SIZE_M = 1.0

# Poids des critères dans le score composite [0..1]
_W_DENSITY = 0.35      # densité de points
_W_SEGMENTS = 0.30     # présence de segments
_W_TOPOLOGY = 0.20     # cohérence topologique locale
_W_OPENINGS = 0.15     # ouvertures plausibles


@dataclass
class ZoneScore:
    """Score de confiance pour une cellule de grille.

    Attributes:
        col: Indice de colonne dans la grille.
        row: Indice de rangée dans la grille.
        x_min, y_min, x_max, y_max: Emprise de la cellule (mètres).
        density_score: Score basé sur la densité de points [0, 1].
        segment_score: Score basé sur la présence de segments [0, 1].
        topology_score: Score de cohérence topologique locale [0, 1].
        opening_score: Score basé sur les ouvertures plausibles [0, 1].
        total_score: Score composite pondéré [0, 1].
        is_low_confidence: True si total_score < seuil.
    """

    col: int
    row: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    density_score: float = 0.0
    segment_score: float = 0.0
    topology_score: float = 0.0
    opening_score: float = 1.0
    total_score: float = 0.0
    is_low_confidence: bool = False

    @property
    def cx(self) -> float:
        """Centre X de la cellule."""
        return (self.x_min + self.x_max) / 2.0

    @property
    def cy(self) -> float:
        """Centre Y de la cellule."""
        return (self.y_min + self.y_max) / 2.0


@dataclass
class ZoneMap:
    """Carte de scores par zone pour un plan complet.

    Attributes:
        zones: Grille de ZoneScore, shape (n_rows, n_cols).
        n_cols: Nombre de colonnes.
        n_rows: Nombre de rangées.
        cell_size_m: Taille d'une cellule (mètres).
        x_min, y_min: Origine de la grille.
        low_confidence_zones: Liste des cellules à faible confiance.
        global_score: Score moyen pondéré par la surface occupée.
    """

    zones: list[list[ZoneScore]]
    n_cols: int
    n_rows: int
    cell_size_m: float
    x_min: float
    y_min: float
    low_confidence_zones: list[ZoneScore] = field(default_factory=list)
    global_score: float = 0.0

    def score_matrix(self) -> np.ndarray:
        """Retourne les scores comme matrice NumPy (n_rows, n_cols)."""
        return np.array(
            [[self.zones[r][c].total_score for c in range(self.n_cols)]
             for r in range(self.n_rows)],
            dtype=np.float32,
        )


def compute_zone_scores(
    density_map: "DensityMapResult",
    segments: "list[DetectedSegment]",
    openings: "list[Opening]",
    cell_size_m: float = _DEFAULT_CELL_SIZE_M,
    low_confidence_threshold: float = _LOW_CONFIDENCE_THRESHOLD,
) -> ZoneMap:
    """Découpe le plan en grille et calcule un score de confiance par cellule.

    Critères par cellule :
    - **density_score** : fraction de pixels occupés dans la cellule (density map).
    - **segment_score** : longueur de murs dans la cellule / longueur max globale.
    - **topology_score** : ratio segments dont les extrémités tombent dans la cellule
      (connexité locale) — 1.0 si aucun segment isolé.
    - **opening_score** : 1.0 si ≤ 1 ouverture par mètre de mur, pénalisé sinon.

    Args:
        density_map: DensityMapResult de la slice médiane (référence de densité).
        segments: Segments de murs après topologie.
        openings: Ouvertures détectées.
        cell_size_m: Taille d'une cellule de grille (mètres).
        low_confidence_threshold: Score en dessous duquel une zone est signalée.

    Returns:
        ``ZoneMap`` avec les scores par cellule et la liste des zones faibles.
    """
    # Emprise du plan
    x_min, y_min = density_map.x_min, density_map.y_min
    res = density_map.resolution
    h_img, w_img = density_map.image.shape[:2]
    x_max = x_min + w_img * res
    y_max = y_min + h_img * res

    n_cols = max(1, int(np.ceil((x_max - x_min) / cell_size_m)))
    n_rows = max(1, int(np.ceil((y_max - y_min) / cell_size_m)))

    # Pré-calculer la longueur max de murs pour la normalisation
    all_lengths = [s.length for s in segments] if segments else [0.0]
    max_wall_length = max(all_lengths) if all_lengths else 1.0

    # Construire la grille
    grid: list[list[ZoneScore]] = []
    for r in range(n_rows):
        row_zones: list[ZoneScore] = []
        for c in range(n_cols):
            zx_min = x_min + c * cell_size_m
            zy_min = y_min + r * cell_size_m
            zx_max = min(x_max, zx_min + cell_size_m)
            zy_max = min(y_max, zy_min + cell_size_m)
            zone = ZoneScore(col=c, row=r,
                             x_min=zx_min, y_min=zy_min,
                             x_max=zx_max, y_max=zy_max)
            _score_zone(zone, density_map, segments, openings, max_wall_length, res)
            row_zones.append(zone)
        grid.append(row_zones)

    # Identifier les zones faibles et calculer le score global
    low: list[ZoneScore] = []
    scores = []
    for row in grid:
        for zone in row:
            scores.append(zone.total_score)
            if zone.is_low_confidence:
                low.append(zone)

    global_score = float(np.mean(scores)) if scores else 0.0

    zone_map = ZoneMap(
        zones=grid,
        n_cols=n_cols,
        n_rows=n_rows,
        cell_size_m=cell_size_m,
        x_min=x_min,
        y_min=y_min,
        low_confidence_zones=low,
        global_score=round(global_score, 3),
    )

    logger.info(
        "Zone scoring : %d×%d cellules, %d zones faibles, score global %.2f.",
        n_cols, n_rows, len(low), global_score,
    )
    return zone_map


def generate_confidence_heatmap(
    zone_map: ZoneMap,
    output_path: Path | None = None,
) -> np.ndarray:
    """Génère une carte de chaleur des scores de confiance.

    Produit une image float32 (n_rows, n_cols) avec les scores [0, 1],
    colorée en rouge-vert (rouge = faible, vert = élevé).

    Si ``output_path`` est fourni, sauvegarde la figure en PNG.

    Args:
        zone_map: Carte de scores par zone.
        output_path: Chemin PNG optionnel.

    Returns:
        Matrice NumPy (n_rows, n_cols) float32 des scores.
    """
    matrix = zone_map.score_matrix()

    if output_path is not None:
        _save_heatmap_png(matrix, zone_map, output_path)

    return matrix


def export_low_confidence_zones_to_dxf(
    zone_map: ZoneMap,
    doc: Any,
    layer_name: str = "INCERTAIN",
) -> int:
    """Exporte les zones à faible confiance comme rectangles hachurés dans le DXF.

    Chaque zone faible est représentée par 4 entités LINE (bordure du rectangle)
    sur le calque INCERTAIN.

    Args:
        zone_map: Carte de zones avec les scores.
        doc: Document ezdxf ouvert (modifié en place).
        layer_name: Nom du calque de destination.

    Returns:
        Nombre d'entités LINE ajoutées.
    """
    if layer_name not in doc.layers:
        doc.layers.new(layer_name, dxfattribs={"color": 2})  # jaune

    msp = doc.modelspace()
    n = 0

    for zone in zone_map.low_confidence_zones:
        corners = [
            (zone.x_min, zone.y_min),
            (zone.x_max, zone.y_min),
            (zone.x_max, zone.y_max),
            (zone.x_min, zone.y_max),
        ]
        for k in range(4):
            p1 = (*corners[k], 0.0)
            p2 = (*corners[(k + 1) % 4], 0.0)
            msp.add_line(start=p1, end=p2, dxfattribs={"layer": layer_name})
            n += 1

    logger.info(
        "export_low_confidence_zones_to_dxf : %d zones → %d entités LINE.",
        len(zone_map.low_confidence_zones), n,
    )
    return n


def generate_pdf_report(
    zone_map: ZoneMap,
    segments: "list[DetectedSegment]",
    openings: "list[Opening]",
    output_path: Path,
    title: str = "Rapport QA Scan2Plan",
) -> Path:
    """Produit un rapport PDF avec le plan annoté et les métriques QA.

    Le PDF contient :
    - Page 1 : carte de chaleur de confiance + zones faibles surlignées.
    - Page 2 : métriques tabulaires (score global, nb zones faibles, etc.).

    Nécessite matplotlib.

    Args:
        zone_map: Carte de scores par zone.
        segments: Segments de murs.
        openings: Ouvertures détectées.
        output_path: Chemin du fichier PDF de sortie.
        title: Titre affiché en en-tête du rapport.

    Returns:
        Chemin du fichier PDF créé.

    Raises:
        ImportError: Si matplotlib n'est pas disponible.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        raise ImportError(
            "matplotlib est requis pour générer le rapport PDF. "
            "Installer avec : pip install matplotlib"
        ) from exc

    output_path = output_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(output_path)) as pdf:
        _write_heatmap_page(pdf, zone_map, segments, openings, title)
        _write_metrics_page(pdf, zone_map, segments, openings, title)

    logger.info("Rapport PDF QA généré : %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Helpers privés — scoring
# ---------------------------------------------------------------------------

def _score_zone(
    zone: ZoneScore,
    density_map: "DensityMapResult",
    segments: "list[DetectedSegment]",
    openings: "list[Opening]",
    max_wall_length: float,
    res: float,
) -> None:
    """Calcule et assigne les scores à une cellule (modifié en place).

    Args:
        zone: Cellule à scorer.
        density_map: DensityMapResult.
        segments: Segments du plan.
        openings: Ouvertures du plan.
        max_wall_length: Longueur max de mur (pour normalisation).
        res: Résolution de la density map (mètres/pixel).
    """
    zone.density_score = _density_score(zone, density_map, res)
    zone.segment_score = _segment_score(zone, segments, max_wall_length)
    zone.topology_score = _topology_score(zone, segments)
    zone.opening_score = _opening_score(zone, segments, openings)

    zone.total_score = (
        _W_DENSITY * zone.density_score
        + _W_SEGMENTS * zone.segment_score
        + _W_TOPOLOGY * zone.topology_score
        + _W_OPENINGS * zone.opening_score
    )
    zone.is_low_confidence = zone.total_score < _LOW_CONFIDENCE_THRESHOLD


def _density_score(
    zone: ZoneScore,
    density_map: "DensityMapResult",
    res: float,
) -> float:
    """Fraction de pixels occupés dans la cellule.

    Args:
        zone: Cellule.
        density_map: DensityMapResult.
        res: Résolution (mètres/pixel).

    Returns:
        Score [0, 1] — 0 = cellule vide, 1 = pleine.
    """
    dm = density_map
    h, w = dm.image.shape[:2]

    col1 = int((zone.x_min - dm.x_min) / res)
    col2 = int((zone.x_max - dm.x_min) / res)
    # row 0 = Y max → y_max correspond à la rangée du haut
    row1 = int(h - 1 - (zone.y_max - dm.y_min) / res)
    row2 = int(h - 1 - (zone.y_min - dm.y_min) / res)

    col1 = max(0, min(col1, w - 1))
    col2 = max(0, min(col2, w))
    row1 = max(0, min(row1, h - 1))
    row2 = max(0, min(row2, h))

    if col1 >= col2 or row1 >= row2:
        return 0.0

    patch = dm.image[row1:row2, col1:col2]
    if patch.size == 0:
        return 0.0
    return float((patch > 0).sum()) / float(patch.size)


def _segment_score(
    zone: ZoneScore,
    segments: "list[DetectedSegment]",
    max_wall_length: float,
) -> float:
    """Longueur totale de murs dans la cellule, normalisée par le max global.

    Args:
        zone: Cellule.
        segments: Segments du plan.
        max_wall_length: Longueur maximale d'un segment (normalisation).

    Returns:
        Score [0, 1].
    """
    total_len = 0.0
    for seg in segments:
        if _segment_intersects_zone(seg, zone):
            total_len += _clipped_length(seg, zone)

    if max_wall_length < 1e-9:
        return 0.0
    return float(min(1.0, total_len / max_wall_length))


def _topology_score(
    zone: ZoneScore,
    segments: "list[DetectedSegment]",
) -> float:
    """Ratio de segments dont au moins une extrémité est dans la cellule.

    Une extrémité "dans la cellule" signifie que le segment est connecté à
    quelque chose de visible dans cette zone. Si tous les segments traversant
    la zone ont au moins une extrémité dedans, la topologie est cohérente.

    Args:
        zone: Cellule.
        segments: Segments du plan.

    Returns:
        Score [0, 1] — 1.0 si aucun segment ou tous bien connectés.
    """
    crossing = [s for s in segments if _segment_intersects_zone(s, zone)]
    if not crossing:
        return 1.0

    connected = sum(
        1 for s in crossing
        if _point_in_zone(s.x1, s.y1, zone) or _point_in_zone(s.x2, s.y2, zone)
    )
    return float(connected) / float(len(crossing))


def _opening_score(
    zone: ZoneScore,
    segments: "list[DetectedSegment]",
    openings: "list[Opening]",
) -> float:
    """Score basé sur la plausibilité du nombre d'ouvertures par rapport aux murs.

    Si la longueur de murs dans la zone est nulle, le score est 1.0.
    Sinon, le ratio ouvertures / (longueur en mètres) est calculé.
    Un ratio > 2 ouvertures/m est suspect → pénalité.

    Args:
        zone: Cellule.
        segments: Segments du plan.
        openings: Ouvertures détectées.

    Returns:
        Score [0, 1].
    """
    wall_len = sum(
        _clipped_length(s, zone) for s in segments
        if _segment_intersects_zone(s, zone)
    )
    if wall_len < 0.10:
        return 1.0  # pas de mur dans la zone → neutre

    n_openings = sum(
        1 for op in openings
        if _point_in_zone(
            (op.wall_segment.x1 + op.wall_segment.x2) / 2.0,
            (op.wall_segment.y1 + op.wall_segment.y2) / 2.0,
            zone,
        )
    )

    ratio = n_openings / wall_len  # ouvertures par mètre de mur
    return float(max(0.0, 1.0 - min(1.0, ratio / 2.0)))


# ---------------------------------------------------------------------------
# Helpers privés — géométrie de zone
# ---------------------------------------------------------------------------

def _point_in_zone(x: float, y: float, zone: ZoneScore) -> bool:
    """True si le point (x, y) est dans la cellule (bornes incluses)."""
    return zone.x_min <= x <= zone.x_max and zone.y_min <= y <= zone.y_max


def _segment_intersects_zone(seg: "DetectedSegment", zone: ZoneScore) -> bool:
    """True si le segment passe au moins partiellement dans la cellule.

    Utilise un test de bounding-box AABB rapide.

    Args:
        seg: Segment de mur.
        zone: Cellule de grille.

    Returns:
        True si les AABB se chevauchent.
    """
    seg_x_min = min(seg.x1, seg.x2)
    seg_x_max = max(seg.x1, seg.x2)
    seg_y_min = min(seg.y1, seg.y2)
    seg_y_max = max(seg.y1, seg.y2)
    return (
        seg_x_max >= zone.x_min and seg_x_min <= zone.x_max
        and seg_y_max >= zone.y_min and seg_y_min <= zone.y_max
    )


def _clipped_length(seg: "DetectedSegment", zone: ZoneScore) -> float:
    """Longueur de la portion du segment à l'intérieur de la cellule.

    Utilise le clipping de Cohen-Sutherland simplifié par projection paramétrique.

    Args:
        seg: Segment de mur.
        zone: Cellule de grille.

    Returns:
        Longueur clippée en mètres (≥ 0).
    """
    total_len = seg.length
    if total_len < 1e-9:
        return 0.0

    t0, t1 = 0.0, 1.0
    dx = seg.x2 - seg.x1
    dy = seg.y2 - seg.y1

    for edge_p, edge_d, is_x in [
        (seg.x1, dx, True),
        (seg.y1, dy, False),
    ]:
        lo = zone.x_min if is_x else zone.y_min
        hi = zone.x_max if is_x else zone.y_max

        if abs(edge_d) < 1e-12:
            val = edge_p
            if val < lo or val > hi:
                return 0.0
        else:
            t_lo = (lo - edge_p) / edge_d
            t_hi = (hi - edge_p) / edge_d
            if edge_d < 0:
                t_lo, t_hi = t_hi, t_lo
            t0 = max(t0, t_lo)
            t1 = min(t1, t_hi)
            if t0 > t1:
                return 0.0

    return float((t1 - t0) * total_len)


# ---------------------------------------------------------------------------
# Helpers privés — visualisation
# ---------------------------------------------------------------------------

def _save_heatmap_png(
    matrix: np.ndarray,
    zone_map: ZoneMap,
    output_path: Path,
) -> None:
    """Sauvegarde la carte de chaleur en PNG sans afficher la figure.

    Args:
        matrix: Matrice des scores (n_rows, n_cols).
        zone_map: Carte de zones (pour les métadonnées).
        output_path: Chemin de sortie (extension .png assurée).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(
        matrix,
        origin="lower",
        vmin=0.0, vmax=1.0,
        cmap="RdYlGn",
        interpolation="nearest",
        aspect="auto",
    )
    plt.colorbar(img, ax=ax, label="Score de confiance [0-1]")
    ax.set_title("Carte de confiance par zone")
    ax.set_xlabel("Colonne (cellules)")
    ax.set_ylabel("Rangée (cellules)")

    # Marquer les zones faibles
    for zone in zone_map.low_confidence_zones:
        ax.add_patch(plt.Rectangle(
            (zone.col - 0.5, zone.row - 0.5), 1, 1,
            fill=False, edgecolor="black", linewidth=1.5, linestyle="--",
        ))

    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap sauvegardée : %s", output_path)


def _write_heatmap_page(
    pdf: Any,
    zone_map: ZoneMap,
    segments: "list[DetectedSegment]",
    openings: "list[Opening]",
    title: str,
) -> None:
    """Écrit la page de carte de chaleur dans le PDF.

    Args:
        pdf: Objet PdfPages matplotlib.
        zone_map: Carte de zones.
        segments: Segments de murs.
        openings: Ouvertures.
        title: Titre du rapport.
    """
    import matplotlib.pyplot as plt

    matrix = zone_map.score_matrix()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Carte de chaleur
    ax = axes[0]
    img = ax.imshow(
        matrix, origin="lower", vmin=0.0, vmax=1.0,
        cmap="RdYlGn", interpolation="nearest", aspect="auto",
    )
    plt.colorbar(img, ax=ax, label="Confiance")
    ax.set_title("Confiance par zone")
    ax.set_xlabel("Col")
    ax.set_ylabel("Row")
    for zone in zone_map.low_confidence_zones:
        ax.add_patch(plt.Rectangle(
            (zone.col - 0.5, zone.row - 0.5), 1, 1,
            fill=False, edgecolor="black", linewidth=1.2, linestyle="--",
        ))

    # Plan avec segments et zones faibles
    ax2 = axes[1]
    ax2.set_aspect("equal")
    ax2.set_title("Plan avec zones à faible confiance")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")

    for seg in segments:
        ax2.plot([seg.x1, seg.x2], [seg.y1, seg.y2],
                 color="steelblue", linewidth=0.8)

    for zone in zone_map.low_confidence_zones:
        ax2.add_patch(plt.Rectangle(
            (zone.x_min, zone.y_min),
            zone.x_max - zone.x_min,
            zone.y_max - zone.y_min,
            fill=True, facecolor="red", alpha=0.2,
            edgecolor="red", linewidth=0.8,
        ))

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _write_metrics_page(
    pdf: Any,
    zone_map: ZoneMap,
    segments: "list[DetectedSegment]",
    openings: "list[Opening]",
    title: str,
) -> None:
    """Écrit la page de métriques tabulaires dans le PDF.

    Args:
        pdf: Objet PdfPages matplotlib.
        zone_map: Carte de zones.
        segments: Segments de murs.
        openings: Ouvertures.
        title: Titre du rapport.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    fig.suptitle(f"{title} — Métriques", fontsize=13, fontweight="bold")

    n_total = zone_map.n_rows * zone_map.n_cols
    rows_data = [
        ["Métrique", "Valeur"],
        ["Score global", f"{zone_map.global_score:.2f} / 1.00"],
        ["Cellules analysées", str(n_total)],
        ["Cellules à faible confiance", str(len(zone_map.low_confidence_zones))],
        ["Taux zones faibles", f"{len(zone_map.low_confidence_zones)/max(1,n_total):.1%}"],
        ["Taille cellule", f"{zone_map.cell_size_m:.2f} m"],
        ["Segments de murs", str(len(segments))],
        ["Ouvertures", str(len(openings))],
        ["Longueur totale murs", f"{sum(s.length for s in segments):.1f} m"],
    ]

    tbl = ax.table(
        cellText=rows_data[1:],
        colLabels=rows_data[0],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.4, 1.8)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
