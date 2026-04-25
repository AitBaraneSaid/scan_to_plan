"""Appariement des faces de murs : identifie les doubles lignes (faces opposées).

Un scanner laser intérieur capte les deux faces d'un mur physique. Après
Hough + fusion colinéaire, chaque mur de 15–25 cm produit DEUX segments
parallèles. Ce module identifie ces paires et produit les métadonnées
d'épaisseur, sans modifier ni supprimer les segments originaux.

Deux modes d'utilisation
------------------------
- ``pair_wall_faces`` (mode principal) : retourne les segments ORIGINAUX inchangés
  + les métadonnées de pairing. Aucun segment n'est perdu, fusionné ou remplacé.
- ``apply_median_pairing`` (V2 — cotations) : remplace les paires par leur axe
  médian. À utiliser uniquement pour la génération des cotations d'épaisseur.

Nomenclature géométrique
------------------------
- **face** : un segment issu du pipeline Hough représentant une face scannée.
- **paire** : deux faces parallèles d'un même mur physique.
- **axe médian** : segment central calculé comme la moyenne pondérée des deux faces.
- **corridor** : espace rectangulaire entre les deux faces d'une paire candidate.
  Si un troisième segment traverse ce corridor, la paire est rejetée (les
  deux faces appartiennent à des murs distincts).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """Segment de mur en coordonnées métriques.

    Args:
        x1: Coordonnée X de la première extrémité (m).
        y1: Coordonnée Y de la première extrémité (m).
        x2: Coordonnée X de la deuxième extrémité (m).
        y2: Coordonnée Y de la deuxième extrémité (m).
        label: Étiquette sémantique du segment.
        confidence: Score de confiance (0–1).
        source_slice: Slice d'origine (``"high"``, ``"mid"``, ``"low"``).
    """

    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""
    confidence: float = 1.0
    source_slice: str = "mid"

    # ------------------------------------------------------------------
    # Propriétés calculées (aucune valeur stockée)
    # ------------------------------------------------------------------

    @property
    def length(self) -> float:
        """Longueur euclidienne du segment (m)."""
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    @property
    def angle(self) -> float:
        """Angle du segment en radians dans ``[0, π)``.

        Les segments sont bidirectionnels : un angle de π rad est identique
        à 0 rad pour nos calculs.
        """
        return float(np.arctan2(self.y2 - self.y1, self.x2 - self.x1) % np.pi)

    @property
    def midpoint(self) -> tuple[float, float]:
        """Centre du segment."""
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def direction(self) -> np.ndarray:
        """Vecteur unitaire dans la direction du segment (shape ``(2,)``)."""
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        length = np.hypot(dx, dy)
        if length < 1e-9:
            return np.array([1.0, 0.0])
        return np.array([dx / length, dy / length])

    @property
    def normal(self) -> np.ndarray:
        """Vecteur unitaire perpendiculaire au segment (rotation +90°)."""
        d = self.direction
        return np.array([-d[1], d[0]])

    @property
    def p1(self) -> np.ndarray:
        """Première extrémité sous forme de tableau NumPy."""
        return np.array([self.x1, self.y1])

    @property
    def p2(self) -> np.ndarray:
        """Deuxième extrémité sous forme de tableau NumPy."""
        return np.array([self.x2, self.y2])

    @property
    def as_tuple(self) -> tuple[float, float, float, float]:
        """Représentation (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    def as_shapely(self):  # type: ignore[return]
        """Retourne un ``shapely.geometry.LineString``."""
        from shapely.geometry import LineString  # import local : évite la dépendance au niveau module

        return LineString([(self.x1, self.y1), (self.x2, self.y2)])


@dataclass
class FacePair:
    """Paire de faces opposées d'un mur physique (segments originaux, inchangés).

    Les segments sont retournés TELS QUELS — pas de fusion, pas de médiane.
    Cette structure transporte uniquement les métadonnées de l'appariement.

    Args:
        face_a: Première face (segment original inchangé).
        face_b: Deuxième face (segment original inchangé).
        thickness: Distance perpendiculaire entre les deux faces (m).
        overlap_length: Longueur de chevauchement longitudinal (m).
        score: Score de confiance de l'appariement (0–1).
    """

    face_a: Segment
    face_b: Segment
    thickness: float
    overlap_length: float
    score: float = 0.0


@dataclass
class FacePairingResult:
    """Résultat de ``pair_wall_faces`` — segments originaux + métadonnées.

    Args:
        paired_faces: Paires de faces identifiées.
        unpaired_segments: Segments non appariés (cloisons, jambages).
        all_segments: TOUS les segments originaux inchangés (pairés + non pairés).
        num_pairs: Nombre de paires confirmées.
    """

    paired_faces: list[FacePair]
    unpaired_segments: list[Segment]
    all_segments: list[Segment]
    num_pairs: int = 0


@dataclass
class WallPair:
    """Paire de faces opposées d'un mur physique avec axe médian.

    Utilisé par ``apply_median_pairing`` (V2 — cotations uniquement).
    Pour le pipeline principal, utiliser ``FacePair`` / ``pair_wall_faces``.

    Args:
        face_a: Premier segment (face).
        face_b: Deuxième segment (face opposée).
        thickness: Distance perpendiculaire entre les deux faces (m).
        overlap_length: Longueur de chevauchement longitudinal (m).
        median_segment: Axe médian calculé (segment central).
        score: Score de confiance de l'appariement (0–1). Plus c'est proche
            de 1, plus la paire est fiable.
    """

    face_a: Segment
    face_b: Segment
    thickness: float
    overlap_length: float
    median_segment: Segment
    score: float = 0.0


@dataclass
class PairingResult:
    """Résultat complet de l'appariement (interne à find_wall_pairs).

    Args:
        pairs: Paires de faces confirmées (avec médiane).
        unpaired_segments: Segments non appariés (cloisons, jambages, bruit).
        num_candidates_tested: Nombre de paires candidates évaluées.
        num_pairs_confirmed: Nombre de paires acceptées.
        num_pairs_rejected_corridor: Paires rejetées car un tiers segment
            traverse le corridor.
        num_pairs_rejected_conflict: Paires rejetées car un segment appartient
            à plusieurs paires candidates (conflit résolu par score).
    """

    pairs: list[WallPair]
    unpaired_segments: list[Segment]
    num_candidates_tested: int = 0
    num_pairs_confirmed: int = 0
    num_pairs_rejected_corridor: int = 0
    num_pairs_rejected_conflict: int = 0


@dataclass
class PairingConfig:
    """Paramètres de l'algorithme d'appariement.

    Args:
        angle_tolerance_deg: Tolérance angulaire maximale entre deux segments
            parallèles (degrés). Au-delà, ils ne peuvent pas être une paire.
        min_distance: Distance perpendiculaire minimale pour une paire (m).
            En dessous, c'est de la fusion colinéaire, pas du pairing.
        max_distance: Distance perpendiculaire maximale (m). Au-delà, les
            deux segments appartiennent à des murs distincts.
        min_overlap_abs: Chevauchement longitudinal minimum absolu (m).
        min_overlap_ratio: Chevauchement minimum relatif à la longueur du
            segment le plus court (0–1).
        corridor_margin: Rétrécissement du corridor de chaque côté (m).
            Évite les faux positifs sur les intersections rasantes.
        typical_wall_thickness: Épaisseur typique d'un mur (m). Utilisée
            pour pondérer le score (une paire à 15 cm score mieux que 28 cm).
        min_segment_length: Longueur minimale pour qu'un segment soit candidat
            au pairing (m). Filtre les jambages courts.
        corridor_intersection_threshold: Longueur minimale d'intersection avec
            le corridor pour qu'un segment bloque la paire (m).
    """

    angle_tolerance_deg: float = 3.0
    min_distance: float = 0.04
    max_distance: float = 0.30
    min_overlap_abs: float = 0.30
    min_overlap_ratio: float = 0.40
    corridor_margin: float = 0.02
    typical_wall_thickness: float = 0.15
    min_segment_length: float = 0.20
    corridor_intersection_threshold: float = 0.05


# ---------------------------------------------------------------------------
# Helpers géométriques (NumPy pur — appelés O(n²), pas de Shapely ici)
# ---------------------------------------------------------------------------


def _angle_diff(a1: float, a2: float) -> float:
    """Différence angulaire entre deux angles en radians, résultat dans ``[0, π/2]``.

    Les segments sont bidirectionnels : un segment à θ et un segment à θ+π
    ont la même orientation. La symétrie est gérée via ``% π``.

    Args:
        a1: Premier angle en radians.
        a2: Deuxième angle en radians.

    Returns:
        Différence angulaire dans ``[0, π/2]``.

    Examples:
        >>> _angle_diff(0.0, np.pi)
        0.0
        >>> _angle_diff(0.0, np.pi / 2)
        1.5707963...
    """
    diff = abs(a1 - a2) % np.pi
    return float(min(diff, np.pi - diff))


def _perpendicular_distance(seg_a: Segment, seg_b: Segment) -> float:
    """Distance perpendiculaire entre deux segments quasi-parallèles.

    Projette le milieu de ``seg_b`` sur la droite porteuse de ``seg_a``
    via le produit scalaire avec la normale de ``seg_a``.

    Hypothèse : les segments sont suffisamment parallèles (``_angle_diff < π/36``).
    Pour des segments non parallèles, le résultat n'est pas significatif.

    Args:
        seg_a: Segment de référence.
        seg_b: Segment dont on mesure la distance.

    Returns:
        Distance perpendiculaire signée en mètres (valeur absolue à prendre
        par l'appelant si la direction n'importe pas).
    """
    mid_b = np.array(seg_b.midpoint)
    return float(abs(np.dot(mid_b - seg_a.p1, seg_a.normal)))


def _project_on_line(
    point: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
) -> float:
    """Projette un point sur une droite paramétrée par ``(origin, direction)``.

    Retourne le paramètre scalaire ``t`` tel que la projection vaut
    ``origin + t * direction``.

    Args:
        point: Point à projeter (shape ``(2,)``).
        origin: Point d'origine de la droite (shape ``(2,)``).
        direction: Vecteur unitaire de direction (shape ``(2,)``).

    Returns:
        Paramètre ``t`` de la projection (en mètres si ``direction`` est unitaire).
    """
    return float(np.dot(point - origin, direction))


def _compute_overlap(
    seg_a: Segment,
    seg_b: Segment,
) -> tuple[float, float, float]:
    """Chevauchement longitudinal entre deux segments parallèles.

    Algorithme :

    1. Utiliser la direction de ``seg_a`` comme axe de référence.
    2. Projeter les 4 extrémités sur cet axe.
    3. Calculer l'intersection des intervalles ``[a_min, a_max] ∩ [b_min, b_max]``.
    4. ``overlap = max(0, min(a_max, b_max) - max(a_min, b_min))``.

    Args:
        seg_a: Premier segment (fournit l'axe de projection).
        seg_b: Deuxième segment.

    Returns:
        ``(overlap_length, t_start, t_end)`` où ``t_start`` et ``t_end``
        sont les bornes de la zone de chevauchement sur l'axe de ``seg_a``
        (en mètres depuis ``seg_a.p1``).
        ``overlap_length`` vaut 0 si les segments ne se chevauchent pas.
    """
    origin = seg_a.p1
    direction = seg_a.direction

    # Projections sur l'axe de seg_a
    ta1 = _project_on_line(seg_a.p1, origin, direction)
    ta2 = _project_on_line(seg_a.p2, origin, direction)
    tb1 = _project_on_line(seg_b.p1, origin, direction)
    tb2 = _project_on_line(seg_b.p2, origin, direction)

    a_min, a_max = min(ta1, ta2), max(ta1, ta2)
    b_min, b_max = min(tb1, tb2), max(tb1, tb2)

    t_start = max(a_min, b_min)
    t_end = min(a_max, b_max)
    overlap = max(0.0, t_end - t_start)

    return overlap, t_start, t_end


def _build_corridor_polygon(
    seg_a: Segment,
    seg_b: Segment,
    t_start: float,
    t_end: float,
    margin: float = 0.02,
) -> object | None:
    """Construit le polygone corridor entre deux faces sur leur zone de chevauchement.

    Le corridor est un rectangle aligné sur l'axe de ``seg_a``, rétréci
    longitudinalement et perpendiculairement de ``margin`` pour éviter les
    faux positifs aux intersections rasantes.

    Nécessite Shapely — appelé une seule fois par paire candidate.

    Args:
        seg_a: Première face (fournit l'axe de référence).
        seg_b: Deuxième face.
        t_start: Début de la zone de chevauchement (paramètre sur l'axe de seg_a).
        t_end: Fin de la zone de chevauchement.
        margin: Rétrécissement de chaque côté, longitudinal et perpendiculaire (m).

    Returns:
        ``shapely.geometry.Polygon`` représentant le corridor, ou ``None``
        si le corridor est trop petit pour être significatif.
    """
    from shapely.geometry import Polygon

    origin = seg_a.p1
    direction = seg_a.direction
    normal = seg_a.normal

    # Rétrécissement longitudinal
    t_start_m = t_start + margin
    t_end_m = t_end - margin
    if t_end_m - t_start_m < 0.05:
        return None

    # Distances perpendiculaires des deux faces à l'axe de seg_a
    dist_a = float(np.dot(seg_a.midpoint - origin, normal))  # type: ignore[arg-type]
    mid_b = np.array(seg_b.midpoint)
    dist_b = float(np.dot(mid_b - origin, normal))

    d_min = min(dist_a, dist_b) + margin
    d_max = max(dist_a, dist_b) - margin
    if d_max - d_min < 0.01:
        return None

    # 4 coins du rectangle corridor
    c0 = origin + direction * t_start_m + normal * d_min
    c1 = origin + direction * t_end_m + normal * d_min
    c2 = origin + direction * t_end_m + normal * d_max
    c3 = origin + direction * t_start_m + normal * d_max

    poly = Polygon([(c0[0], c0[1]), (c1[0], c1[1]), (c2[0], c2[1]), (c3[0], c3[1])])
    if not poly.is_valid or poly.area < 1e-6:
        return None
    return poly


def _corridor_is_free(
    corridor: object,
    all_segments: list[Segment],
    exclude_indices: tuple[int, int],
    pair_angle: float,
    corridor_width: float,
    threshold: float = 0.05,
) -> bool:
    """Teste si le corridor entre deux faces ne contient aucun segment bloquant.

    Distingue trois catégories de segments tiers :

    - **Parallèle** (angle_diff ≤ 30°) : mur intercalé → bloque inconditionnellement.
    - **Perpendiculaire** (angle_diff > 30°), intersection courte : mur de refend
      ou de coin traversant l'épaisseur du mur → autorisé.
    - **Perpendiculaire**, intersection longue (> corridor_width + 5 cm) : segment
      courant à l'intérieur du corridor → bloque.

    Args:
        corridor: ``shapely.geometry.Polygon`` du corridor à tester.
        all_segments: Tous les segments du pipeline.
        exclude_indices: Indices ``(i, j)`` des deux faces de la paire (ignorés).
        pair_angle: Angle de la paire (angle de ``seg_a``) en radians.
        corridor_width: Largeur effective du corridor après marge (m).
        threshold: Longueur minimale d'intersection pour qu'elle soit significative (m).

    Returns:
        ``True`` si le corridor est libre, ``False`` si un segment le bloque.
    """
    i_exc, j_exc = exclude_indices

    for k, seg in enumerate(all_segments):
        if k == i_exc or k == j_exc:
            continue
        if seg.length < threshold:
            continue

        sk_line = seg.as_shapely()
        if not corridor.intersects(sk_line):  # type: ignore[union-attr]
            continue

        intersection = corridor.intersection(sk_line)  # type: ignore[union-attr]
        if intersection.is_empty or intersection.length < threshold:
            continue

        angle_diff = _angle_diff(seg.angle, pair_angle)

        if angle_diff > np.deg2rad(30.0):
            # Perpendiculaire : autorisé sauf si l'intersection est trop longue
            if intersection.length > corridor_width + 0.05:
                return False
        else:
            # Parallèle ou quasi-parallèle : mur intercalé → bloque
            return False

    return True


def _build_median_segment(seg_a: Segment, seg_b: Segment) -> Segment:
    """Construit l'axe médian entre deux segments appariés.

    L'axe médian couvre l'emprise totale des deux segments (union des
    projections longitudinales), pas seulement leur chevauchement. Les
    extrémités non chevauchantes correspondent à des zones d'occultation
    d'un côté ou de l'autre — le mur physique existe quand même.

    Args:
        seg_a: Première face.
        seg_b: Deuxième face.

    Returns:
        Nouveau ``Segment`` centré entre les deux faces, avec
        ``label="wall_paired"``.
    """
    origin = seg_a.p1
    direction = seg_a.direction
    normal = seg_a.normal

    # Emprise totale : projections longitudinales des 4 extrémités
    projs = [
        _project_on_line(seg_a.p1, origin, direction),
        _project_on_line(seg_a.p2, origin, direction),
        _project_on_line(seg_b.p1, origin, direction),
        _project_on_line(seg_b.p2, origin, direction),
    ]
    t_min, t_max = min(projs), max(projs)

    # Position perpendiculaire médiane
    mid_a = np.array(seg_a.midpoint)
    mid_b = np.array(seg_b.midpoint)
    perp_a = float(np.dot(mid_a - origin, normal))
    perp_b = float(np.dot(mid_b - origin, normal))
    perp_median = (perp_a + perp_b) / 2.0

    p_start = origin + direction * t_min + normal * perp_median
    p_end = origin + direction * t_max + normal * perp_median

    return Segment(
        x1=float(p_start[0]),
        y1=float(p_start[1]),
        x2=float(p_end[0]),
        y2=float(p_end[1]),
        label="wall_paired",
        confidence=max(seg_a.confidence, seg_b.confidence),
        source_slice=seg_a.source_slice,
    )


# ---------------------------------------------------------------------------
# Algorithme principal
# ---------------------------------------------------------------------------


def find_wall_pairs(
    segments: list[Segment],
    config: PairingConfig | None = None,
) -> PairingResult:
    """Identifie et apparie les paires de faces opposées de murs.

    Trois phases :

    1. **Identification des candidats** (NumPy pur, O(n²)) : filtre par
       angle, distance perpendiculaire et chevauchement longitudinal.
    2. **Test du corridor** (Shapely) : rejette les paires dont le corridor
       contient un segment tiers bloquant.
    3. **Résolution gloutonne des conflits** : trie par score décroissant,
       chaque segment ne peut appartenir qu'à une seule paire.

    Args:
        segments: Liste de segments issus du pipeline Hough + fusion.
        config: Paramètres d'appariement. Valeurs par défaut si ``None``.

    Returns:
        ``PairingResult`` avec les paires confirmées, les segments non appariés
        et les statistiques de rejet.
    """
    if config is None:
        config = PairingConfig()

    if len(segments) < 2:
        logger.info("find_wall_pairs : moins de 2 segments, appariement ignoré.")
        return PairingResult(pairs=[], unpaired_segments=list(segments))

    angle_tol_rad = float(np.deg2rad(config.angle_tolerance_deg))
    n = len(segments)

    # ------------------------------------------------------------------
    # Phase 1 — Identification des candidats (NumPy pur)
    # ------------------------------------------------------------------
    candidates: list[tuple[int, int, float, float, float, float]] = []

    for i in range(n):
        si = segments[i]
        if si.length < config.min_segment_length:
            continue
        for j in range(i + 1, n):
            sj = segments[j]
            if sj.length < config.min_segment_length:
                continue
            if _angle_diff(si.angle, sj.angle) > angle_tol_rad:
                continue
            d_perp = _perpendicular_distance(si, sj)
            if d_perp < config.min_distance or d_perp > config.max_distance:
                continue
            overlap, t_start, t_end = _compute_overlap(si, sj)
            shorter = min(si.length, sj.length)
            if overlap < config.min_overlap_abs or overlap < config.min_overlap_ratio * shorter:
                continue
            candidates.append((i, j, d_perp, overlap, t_start, t_end))

    logger.info("find_wall_pairs Phase 1 : %d paires candidates", len(candidates))

    # ------------------------------------------------------------------
    # Phase 2 — Test du corridor (Shapely)
    # ------------------------------------------------------------------
    validated: list[tuple[int, int, float, float, float]] = []
    n_rejected_corridor = 0

    for i, j, d_perp, overlap, t_start, t_end in candidates:
        si, sj = segments[i], segments[j]
        corridor = _build_corridor_polygon(si, sj, t_start, t_end, config.corridor_margin)
        if corridor is None:
            n_rejected_corridor += 1
            continue

        corridor_width = d_perp - 2.0 * config.corridor_margin
        if not _corridor_is_free(
            corridor, segments, (i, j),
            si.angle, corridor_width,
            config.corridor_intersection_threshold,
        ):
            n_rejected_corridor += 1
            continue

        longer = max(si.length, sj.length)
        overlap_score = overlap / longer if longer > 1e-9 else 0.0
        thickness_err = abs(d_perp - config.typical_wall_thickness)
        thickness_score = 1.0 - min(1.0, thickness_err / config.typical_wall_thickness)
        score = 0.6 * overlap_score + 0.4 * thickness_score

        validated.append((i, j, d_perp, overlap, score))

    logger.info(
        "find_wall_pairs Phase 2 : %d validées, %d rejetées (corridor)",
        len(validated), n_rejected_corridor,
    )

    # ------------------------------------------------------------------
    # Phase 3 — Résolution gloutonne des conflits
    # ------------------------------------------------------------------
    validated.sort(key=lambda x: x[4], reverse=True)

    used: set[int] = set()
    pairs: list[WallPair] = []
    n_conflicts = 0

    for i, j, d_perp, overlap, score in validated:
        if i in used or j in used:
            n_conflicts += 1
            continue
        used.add(i)
        used.add(j)
        median = _build_median_segment(segments[i], segments[j])
        pairs.append(WallPair(
            face_a=segments[i],
            face_b=segments[j],
            thickness=d_perp,
            overlap_length=overlap,
            median_segment=median,
            score=score,
        ))

    unpaired = [seg for k, seg in enumerate(segments) if k not in used]

    logger.info(
        "find_wall_pairs Phase 3 : %d paires confirmées, %d conflits, %d non appariés",
        len(pairs), n_conflicts, len(unpaired),
    )

    return PairingResult(
        pairs=pairs,
        unpaired_segments=unpaired,
        num_candidates_tested=len(candidates),
        num_pairs_confirmed=len(pairs),
        num_pairs_rejected_corridor=n_rejected_corridor,
        num_pairs_rejected_conflict=n_conflicts,
    )


def pair_wall_faces(
    segments: list[Segment],
    config: PairingConfig | None = None,
) -> FacePairingResult:
    """Identifie les paires de faces opposées sans modifier les segments.

    CONTRAINTE CRITIQUE : cette fonction ne modifie PAS les segments.
    Elle retourne les segments ORIGINAUX avec des métadonnées de pairing.
    Aucun segment n'est supprimé, fusionné, ni remplacé par une médiane.

    Utilise le même algorithme en 3 phases que ``find_wall_pairs`` :
    1. Identification des candidats (NumPy pur, O(n²)).
    2. Test du corridor libre (Shapely).
    3. Résolution gloutonne des conflits (score décroissant).

    Args:
        segments: Segments issus du pipeline (regularization → ici).
        config: Paramètres d'appariement. Valeurs par défaut si ``None``.

    Returns:
        ``FacePairingResult`` avec :
        - ``paired_faces`` : les ``FacePair`` identifiées (segments originaux).
        - ``unpaired_segments`` : segments sans paire.
        - ``all_segments`` : TOUS les segments originaux, inchangés.
        - ``num_pairs`` : nombre de paires confirmées.

    Example:
        >>> result = pair_wall_faces(segments)
        >>> result.all_segments == segments  # même objets, même ordre
        True
        >>> len(result.all_segments) == len(segments)
        True
    """
    if config is None:
        config = PairingConfig()

    pairing = find_wall_pairs(segments, config)

    paired_faces = [
        FacePair(
            face_a=wp.face_a,
            face_b=wp.face_b,
            thickness=wp.thickness,
            overlap_length=wp.overlap_length,
            score=wp.score,
        )
        for wp in pairing.pairs
    ]

    logger.info(
        "pair_wall_faces : %d paires identifiées sur %d segments (%d non appariés).",
        len(paired_faces),
        len(segments),
        len(pairing.unpaired_segments),
    )

    return FacePairingResult(
        paired_faces=paired_faces,
        unpaired_segments=pairing.unpaired_segments,
        all_segments=list(segments),
        num_pairs=len(paired_faces),
    )


def apply_median_pairing(
    segments: list[Segment],
    config: PairingConfig | None = None,
) -> list[Segment]:
    """Remplace les paires par leur axe médian — V2, pour cotations uniquement.

    NE PAS utiliser dans le pipeline principal. Cette fonction détruit
    l'information de double face et produit des segments qui ne correspondent
    plus aux faces réelles scannées.

    Pour le pipeline principal (export DXF double ligne, topologie, etc.),
    utiliser ``pair_wall_faces`` qui conserve les segments originaux.

    Args:
        segments: Segments issus du pipeline Hough + fusion.
        config: Paramètres d'appariement. Valeurs par défaut si ``None``.

    Returns:
        Liste de segments avec les paires remplacées par leurs axes médians
        + les segments non appariés inchangés.
    """
    result = find_wall_pairs(segments, config)
    output: list[Segment] = [wp.median_segment for wp in result.pairs]
    output.extend(result.unpaired_segments)
    logger.info(
        "apply_median_pairing : %d -> %d segments (%d paires → médianes).",
        len(segments), len(output), len(result.pairs),
    )
    return output
