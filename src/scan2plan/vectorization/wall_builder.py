"""Construction des entités mur — V1 simple ligne, V2 double ligne avec épaisseur."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scan2plan.detection.line_detection import DetectedSegment
    from scan2plan.slicing.density_map import DensityMapResult

logger = logging.getLogger(__name__)

# Seuils métier
_MIN_WALL_THICKNESS_M = 0.03   # 3 cm — en dessous = artefact
_MAX_WALL_THICKNESS_M = 0.80   # 80 cm — au-dessus = mobilier
_PROFILE_SAMPLES = 5            # nombre de profils perpendiculaires par mur
_PROFILE_HALF_LENGTH_M = 0.50  # demi-longueur du profil perpendiculaire (m)
_DENSITY_THRESHOLD = 0.1       # fraction du max pour considérer « haute densité »
_MIN_ANGLE_DEG = 20.0          # angle min pour traiter deux murs comme distincts (L/T)


def estimate_wall_thickness(
    segment: "DetectedSegment",
    density_map: "DensityMapResult",
    binary_image: np.ndarray,
) -> float:
    """Estime l'épaisseur d'un mur à partir de la density map.

    Algorithme :
    1. Extraire _PROFILE_SAMPLES profils perpendiculaires répartis le long du mur.
    2. Pour chaque profil, projeter les pixels de la binary_image sur cet axe et
       mesurer la largeur de la zone de haute densité (valeur = 255).
    3. L'épaisseur est la médiane de ces mesures, convertie en mètres.

    Si la mesure est hors des bornes plausibles (_MIN/_MAX_WALL_THICKNESS_M),
    retourne 0.0 pour signaler l'échec (l'appelant utilisera default_thickness).

    Args:
        segment: Segment de mur (axe central).
        density_map: DensityMapResult de la slice haute (meilleure qualité).
        binary_image: Image binaire correspondante (np.uint8, valeurs 0 ou 255).

    Returns:
        Épaisseur estimée en mètres, ou 0.0 si non mesurable.
    """
    length = segment.length
    if length < 1e-6:
        return 0.0

    # Direction du mur et perpendiculaire
    dx = (segment.x2 - segment.x1) / length
    dy = (segment.y2 - segment.y1) / length
    px, py = -dy, dx  # perpendiculaire unitaire

    res = density_map.resolution
    h, w = binary_image.shape[:2]

    thicknesses: list[float] = []

    for k in range(_PROFILE_SAMPLES):
        # Point le long du mur : éviter les 10% extrêmes pour les coins
        t = length * (0.1 + 0.8 * (k + 0.5) / _PROFILE_SAMPLES)
        cx = segment.x1 + t * dx
        cy = segment.y1 + t * dy

        thickness = _measure_profile_thickness(
            cx, cy, px, py, binary_image, density_map, res, h, w
        )
        if thickness > 0.0:
            thicknesses.append(thickness)

    if not thicknesses:
        return 0.0

    median_thickness = float(np.median(thicknesses))
    if median_thickness < _MIN_WALL_THICKNESS_M or median_thickness > _MAX_WALL_THICKNESS_M:
        return 0.0

    logger.debug(
        "Épaisseur mur estimée : %.3f m (médiane de %d mesures).",
        median_thickness,
        len(thicknesses),
    )
    return median_thickness


def build_double_line_walls(
    segments: "list[DetectedSegment]",
    thicknesses: list[float],
    default_thickness: float = 0.15,
) -> "list[tuple[DetectedSegment, DetectedSegment]]":
    """Construit les doubles lignes pour chaque mur.

    Chaque mur simple (1 segment) devient 2 segments parallèles décalés
    de ±épaisseur/2 perpendiculairement à l'axe du mur.

    Les intersections en L (deux murs à ≥ _MIN_ANGLE_DEG l'un de l'autre dont
    les extrémités se touchent) sont résolues : les 4 lignes (2 par mur) sont
    coupées au point d'intersection de leurs droites supports, pour que le coin
    soit propre sans débordement ni gap.

    Args:
        segments: Axes des murs (après régularisation et topologie).
        thicknesses: Épaisseur de chaque segment (même ordre, même longueur).
            Une valeur ≤ 0 → utilise ``default_thickness``.
        default_thickness: Épaisseur de repli si non mesurable (mètres).

    Returns:
        Liste de paires (ligne_gauche, ligne_droite) dans le repère métrique.
        La « gauche » est le côté avec la normale +perpendiculaire.
    """
    if len(segments) != len(thicknesses):
        raise ValueError(
            f"segments ({len(segments)}) et thicknesses ({len(thicknesses)}) "
            "doivent avoir la même longueur."
        )

    from scan2plan.detection.line_detection import DetectedSegment as DS

    # 1. Construire les paires brutes (non coupées aux coins)
    pairs: list[tuple[DS, DS]] = []
    half_widths: list[float] = []
    for seg, thick in zip(segments, thicknesses):
        hw = (thick if thick > 0.0 else default_thickness) / 2.0
        half_widths.append(hw)
        pairs.append(_offset_segment(seg, hw))

    # 2. Résolution des intersections L/T pour chaque paire de murs proches
    _resolve_double_line_corners(pairs, segments, half_widths)

    logger.info(
        "build_double_line_walls : %d murs → %d paires de lignes.",
        len(segments),
        len(pairs),
    )
    return pairs


# ---------------------------------------------------------------------------
# Helpers privés — estimation épaisseur
# ---------------------------------------------------------------------------

def _measure_profile_thickness(
    cx: float,
    cy: float,
    px: float,
    py: float,
    binary_image: np.ndarray,
    density_map: "DensityMapResult",
    res: float,
    h: int,
    w: int,
) -> float:
    """Mesure la largeur de la zone occupée sur un profil perpendiculaire.

    Échantillonne le profil perpendiculaire au mur passant par (cx, cy),
    sur une longueur ±_PROFILE_HALF_LENGTH_M, et mesure la largeur de la
    zone de pixels à 255 (mur).

    Args:
        cx, cy: Centre du profil en coordonnées métriques.
        px, py: Direction perpendiculaire unitaire.
        binary_image: Image binaire (0 ou 255).
        density_map: DensityMapResult pour la conversion métrique → pixel.
        res: Résolution (mètres/pixel).
        h, w: Dimensions de l'image.

    Returns:
        Épaisseur mesurée en mètres, ou 0.0 si pas de zone détectée.
    """
    n_samples = max(10, int(2 * _PROFILE_HALF_LENGTH_M / res))
    offsets = np.linspace(-_PROFILE_HALF_LENGTH_M, _PROFILE_HALF_LENGTH_M, n_samples)

    # Coordonnées métriques des points du profil
    xs = cx + offsets * px
    ys = cy + offsets * py

    # Conversion en pixels (cf. DensityMapResult : row 0 = Y max)
    cols = ((xs - density_map.x_min) / res).astype(int)
    rows = (density_map.height - 1 - (ys - density_map.y_min) / res).astype(int)

    # Masque de validité (dans l'image)
    valid = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    if not valid.any():
        return 0.0

    # Valeurs du profil (1 = mur, 0 = vide)
    profile = np.zeros(n_samples, dtype=np.uint8)
    profile[valid] = (binary_image[rows[valid], cols[valid]] > 0).astype(np.uint8)

    return _profile_width_m(profile, offsets)


def _profile_width_m(profile: np.ndarray, offsets: np.ndarray) -> float:
    """Mesure la largeur de la zone continue centrale de 1 dans le profil.

    Cherche le segment de 1 consécutifs le plus large contenant le centre.
    Si le centre (offset = 0) est à 0, cherche le segment le plus proche.

    Args:
        profile: Tableau binaire (0 ou 1), longueur N.
        offsets: Positions métriques correspondantes, longueur N.

    Returns:
        Largeur en mètres, ou 0.0 si aucun pixel occupé.
    """
    indices = np.where(profile > 0)[0]
    if len(indices) == 0:
        return 0.0

    # Groupes consécutifs
    gaps = np.where(np.diff(indices) > 1)[0]
    starts = np.concatenate([[indices[0]], indices[gaps + 1]])
    ends = np.concatenate([indices[gaps], [indices[-1]]])

    # Centre du profil
    center_idx = len(profile) // 2

    # Chercher le groupe contenant ou le plus proche du centre
    best_start, best_end = starts[0], ends[0]
    best_dist = float("inf")
    for s, e in zip(starts, ends):
        mid = (s + e) // 2
        dist = abs(mid - center_idx)
        if s <= center_idx <= e:
            # Groupe contenant le centre → priorité absolue
            best_start, best_end = s, e
            break
        if dist < best_dist:
            best_dist = dist
            best_start, best_end = s, e

    return float(abs(offsets[best_end] - offsets[best_start]))


# ---------------------------------------------------------------------------
# Helpers privés — double ligne
# ---------------------------------------------------------------------------

def _offset_segment(
    seg: "DetectedSegment",
    half_width: float,
) -> "tuple[DetectedSegment, DetectedSegment]":
    """Génère deux segments parallèles décalés de ±half_width.

    Args:
        seg: Axe central du mur.
        half_width: Demi-épaisseur en mètres.

    Returns:
        (ligne_positive, ligne_négative) — décalées dans les deux sens de la normale.
    """
    from scan2plan.detection.line_detection import DetectedSegment as DS

    length = seg.length
    if length < 1e-9:
        return (seg, seg)

    dx = (seg.x2 - seg.x1) / length
    dy = (seg.y2 - seg.y1) / length
    nx, ny = -dy, dx  # normale unitaire

    ox, oy = nx * half_width, ny * half_width

    line_pos = DS(
        x1=seg.x1 + ox, y1=seg.y1 + oy,
        x2=seg.x2 + ox, y2=seg.y2 + oy,
        source_slice=seg.source_slice,
        confidence=seg.confidence,
    )
    line_neg = DS(
        x1=seg.x1 - ox, y1=seg.y1 - oy,
        x2=seg.x2 - ox, y2=seg.y2 - oy,
        source_slice=seg.source_slice,
        confidence=seg.confidence,
    )
    return (line_pos, line_neg)


def _resolve_double_line_corners(
    pairs: "list[tuple[DetectedSegment, DetectedSegment]]",
    axes: "list[DetectedSegment]",
    half_widths: list[float],
) -> None:
    """Résout les coins L/T pour toutes les paires de murs proches.

    Modifie ``pairs`` en place. Pour chaque paire de murs dont les axes
    se croisent (ou dont les extrémités sont proches), coupe les 4 lignes
    au bon point d'intersection.

    Seules les intersections à angle > _MIN_ANGLE_DEG sont traitées (évite
    de couper des murs quasi-parallèles).

    Args:
        pairs: Doubles lignes à modifier en place.
        axes: Axes centraux correspondants.
        half_widths: Demi-épaisseurs correspondantes.
    """
    from scan2plan.utils.geometry import angle_between_segments

    n = len(axes)
    for i in range(n):
        for j in range(i + 1, n):
            angle_rad = angle_between_segments(axes[i].as_tuple(), axes[j].as_tuple())
            angle_deg = float(np.degrees(angle_rad))
            if angle_deg < _MIN_ANGLE_DEG:
                continue

            # Chercher si les axes sont proches en extrémité (coin L/T)
            corner = _find_corner_type(axes[i], axes[j])
            if corner is None:
                continue

            pairs[i], pairs[j] = _cut_corner(
                pairs[i], pairs[j], axes[i], axes[j], corner
            )


def _find_corner_type(
    a: "DetectedSegment",
    b: "DetectedSegment",
) -> str | None:
    """Détermine le type de coin entre deux axes de murs.

    Vérifie si une extrémité de ``a`` est proche d'une extrémité de ``b``
    (coin L) ou si une extrémité de ``a`` est proche du milieu de ``b``
    (coin T). Seuil : 2× l'épaisseur maximale raisonnablement possible.

    Args:
        a: Premier axe.
        b: Deuxième axe.

    Returns:
        ``"L"`` pour un coin en L, ``"T"`` pour un T, ``None`` si éloignés.
    """
    threshold = _MAX_WALL_THICKNESS_M * 2  # 160 cm — large pour couvrir coins décalés

    endpoints_a = [(a.x1, a.y1), (a.x2, a.y2)]
    endpoints_b = [(b.x1, b.y1), (b.x2, b.y2)]

    # Vérification coin L : une extrémité de a proche d'une extrémité de b
    for pa in endpoints_a:
        for pb in endpoints_b:
            dist = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
            if dist < threshold:
                return "L"

    # Vérification coin T : une extrémité de a proche d'un point intérieur de b
    for pa in endpoints_a:
        if _point_near_segment_interior(pa, b, threshold):
            return "T"
    for pb in endpoints_b:
        if _point_near_segment_interior(pb, a, threshold):
            return "T"

    return None


def _point_near_segment_interior(
    pt: tuple[float, float],
    seg: "DetectedSegment",
    threshold: float,
) -> bool:
    """True si le point est proche du segment mais pas des extrémités.

    Args:
        pt: Point (x, y).
        seg: Segment à tester.
        threshold: Distance maximale pour être « proche ».

    Returns:
        True si le point est à < threshold du segment et pas à une extrémité.
    """
    from scan2plan.utils.geometry import perpendicular_distance_point_to_line

    dist = perpendicular_distance_point_to_line(pt, seg.as_tuple())
    if dist >= threshold:
        return False

    # Vérifier que la projection est dans le segment (pas en dehors)
    length = seg.length
    if length < 1e-9:
        return False
    dx = (seg.x2 - seg.x1) / length
    dy = (seg.y2 - seg.y1) / length
    t = (pt[0] - seg.x1) * dx + (pt[1] - seg.y1) * dy
    return float(threshold) < t < length - float(threshold)


def _cut_corner(
    pair_a: "tuple[DetectedSegment, DetectedSegment]",
    pair_b: "tuple[DetectedSegment, DetectedSegment]",
    axis_a: "DetectedSegment",
    axis_b: "DetectedSegment",
    corner_type: str,
) -> "tuple[tuple[DetectedSegment, DetectedSegment], tuple[DetectedSegment, DetectedSegment]]":
    """Coupe les 4 lignes d'un coin L ou T au bon point d'intersection.

    Pour un coin L :
    - La ligne extérieure de a rencontre la ligne extérieure de b → couper les deux.
    - La ligne intérieure de a rencontre la ligne intérieure de b → couper les deux.

    Pour un coin T :
    - La ligne de a qui est côté b est coupée à la ligne correspondante de b.

    Args:
        pair_a: (ligne_pos, ligne_neg) du mur a.
        pair_b: (ligne_pos, ligne_neg) du mur b.
        axis_a: Axe central du mur a.
        axis_b: Axe central du mur b.
        corner_type: ``"L"`` ou ``"T"``.

    Returns:
        (pair_a_coupée, pair_b_coupée).
    """

    # Identifier les extrémités de contact entre les axes
    # On cherche quelle extrémité de axis_a est proche de axis_b
    ep_a_idx = _nearest_endpoint_index(axis_a, axis_b)

    # Couper les 4 combinaisons (pos×pos, pos×neg, neg×pos, neg×neg)
    # entre les lignes de pair_a et pair_b
    a_pos, a_neg = pair_a
    b_pos, b_neg = pair_b

    # Pour chaque ligne de a, trouver la ligne de b la plus proche et couper
    a_pos = _trim_line_at_intersection(a_pos, b_pos, ep_a_idx) or a_pos
    a_pos = _trim_line_at_intersection(a_pos, b_neg, ep_a_idx) or a_pos
    a_neg = _trim_line_at_intersection(a_neg, b_pos, ep_a_idx) or a_neg
    a_neg = _trim_line_at_intersection(a_neg, b_neg, ep_a_idx) or a_neg

    ep_b_idx = _nearest_endpoint_index(axis_b, axis_a)
    b_pos = _trim_line_at_intersection(b_pos, a_pos, ep_b_idx) or b_pos
    b_pos = _trim_line_at_intersection(b_pos, a_neg, ep_b_idx) or b_pos
    b_neg = _trim_line_at_intersection(b_neg, a_pos, ep_b_idx) or b_neg
    b_neg = _trim_line_at_intersection(b_neg, a_neg, ep_b_idx) or b_neg

    return (a_pos, a_neg), (b_pos, b_neg)


def _nearest_endpoint_index(
    seg: "DetectedSegment",
    other: "DetectedSegment",
) -> int:
    """Retourne 1 si l'extrémité (x1,y1) est la plus proche de other, 2 sinon.

    Args:
        seg: Segment dont on cherche l'extrémité la plus proche de other.
        other: Segment de référence.

    Returns:
        1 si (x1,y1) est plus proche, 2 si (x2,y2) l'est.
    """
    d1 = min(
        np.hypot(seg.x1 - other.x1, seg.y1 - other.y1),
        np.hypot(seg.x1 - other.x2, seg.y1 - other.y2),
    )
    d2 = min(
        np.hypot(seg.x2 - other.x1, seg.y2 - other.y1),
        np.hypot(seg.x2 - other.x2, seg.y2 - other.y2),
    )
    return 1 if d1 <= d2 else 2


def _trim_line_at_intersection(
    line: "DetectedSegment",
    other: "DetectedSegment",
    endpoint_idx: int,
) -> "DetectedSegment | None":
    """Raccourcit l'extrémité ``endpoint_idx`` de ``line`` jusqu'à l'intersection
    avec ``other``.

    Ne modifie ``line`` que si l'intersection existe et est du bon côté
    (entre l'intérieur du segment et l'extrémité).

    Args:
        line: Ligne à raccourcir.
        other: Ligne avec laquelle on cherche l'intersection.
        endpoint_idx: 1 = modifier l'extrémité (x1,y1), 2 = modifier (x2,y2).

    Returns:
        Nouveau ``DetectedSegment`` raccourci, ou ``None`` si pas d'intersection
        utilisable.
    """
    from scan2plan.detection.line_detection import DetectedSegment as DS
    from scan2plan.utils.geometry import line_intersection

    pt = line_intersection(line.as_tuple(), other.as_tuple())
    if pt is None:
        return None

    ix, iy = pt

    # Vérifier que l'intersection est du bon côté (proche de l'extrémité visée)
    if endpoint_idx == 1:
        # L'intersection doit être entre (x1,y1) et (x2,y2) — pas au-delà de x2,y2
        t = _project_t(line, ix, iy)
        if t > 1.0 + 0.1 or t < -0.5:
            return None
        return DS(
            x1=ix, y1=iy,
            x2=line.x2, y2=line.y2,
            source_slice=line.source_slice,
            confidence=line.confidence,
        )
    else:
        t = _project_t(line, ix, iy)
        if t < -0.1 or t > 1.5:
            return None
        return DS(
            x1=line.x1, y1=line.y1,
            x2=ix, y2=iy,
            source_slice=line.source_slice,
            confidence=line.confidence,
        )


def _project_t(seg: "DetectedSegment", x: float, y: float) -> float:
    """Paramètre t de la projection de (x, y) sur le segment [0=start, 1=end].

    Args:
        seg: Segment de référence.
        x, y: Point à projeter.

    Returns:
        Paramètre t (peut être hors [0,1] si la projection est à l'extérieur).
    """
    length = seg.length
    if length < 1e-9:
        return 0.0
    dx = (seg.x2 - seg.x1) / length
    dy = (seg.y2 - seg.y1) / length
    return float(((x - seg.x1) * dx + (y - seg.y1) * dy) / length)


def build_wall_entities(segments: np.ndarray) -> np.ndarray:
    """Retourne les segments de murs prêts pour l'export DXF (V1 — ligne simple).

    En V2, utiliser ``build_double_line_walls`` à la place.

    Args:
        segments: Array (M, 4) float64 — segments [x1, y1, x2, y2] en mètres.

    Returns:
        Array (M, 4) float64 — entités mur prêtes pour l'export.
    """
    logger.debug("Construction de %d entités mur (ligne simple).", len(segments))
    return segments.astype(np.float64)
