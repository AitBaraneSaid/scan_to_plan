"""Détection de murs courbes (arcs) et poteaux cylindriques.

Algorithme :
- Poteaux : Hough circulaire (cv2.HoughCircles) sur la density map normalisée.
- Murs courbes : zones où les résidus d'un ajustement linéaire local sont élevés.
  Sur ces zones, ajustement d'arc de cercle par moindres carrés.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scan2plan.slicing.density_map import DensityMapResult

logger = logging.getLogger(__name__)

# Limites métriques pour les poteaux (rayon)
_MIN_PILLAR_RADIUS_M = 0.05   # 5 cm — poteau minimal
_MAX_PILLAR_RADIUS_M = 0.40   # 40 cm — poteau maximal (au-delà = mur)

# Limites pour les murs courbes
_MIN_ARC_RADIUS_M = 0.50      # 50 cm — rayon minimal d'un arc (en dessous = coin ?)
_MAX_ARC_RADIUS_M = 50.0      # 50 m — rayon maximal (au-delà ≈ droite)
_MIN_ARC_LENGTH_M = 0.30      # 30 cm — longueur minimale d'arc significatif
_MIN_ARC_ANGLE_DEG = 10.0     # 10° — angle d'arc minimal

# Paramètres de détection des zones courbes
_RESIDUE_WINDOW_PX = 30       # fenêtre locale pour calculer le résidu linéaire
_HIGH_RESIDUE_RATIO = 0.15    # fraction du diamètre max de la fenêtre
_MIN_CURVED_ZONE_PX = 20      # taille minimale d'une zone courbe en pixels


@dataclass
class DetectedArc:
    """Arc de cercle détecté (mur courbe).

    Attributes:
        cx: Coordonnée X du centre (mètres).
        cy: Coordonnée Y du centre (mètres).
        radius: Rayon de l'arc (mètres).
        start_angle_deg: Angle de début (degrés, convention mathématique).
        end_angle_deg: Angle de fin (degrés, convention mathématique).
        confidence: Score de confiance dans [0, 1].
        source_slice: Slice d'origine.
    """

    cx: float
    cy: float
    radius: float
    start_angle_deg: float
    end_angle_deg: float
    confidence: float = 1.0
    source_slice: str = "high"

    @property
    def arc_length(self) -> float:
        """Longueur de l'arc en mètres."""
        span = abs(self.end_angle_deg - self.start_angle_deg)
        span = min(span, 360.0 - span)
        return self.radius * math.radians(span)

    @property
    def span_deg(self) -> float:
        """Étendue angulaire de l'arc en degrés (toujours positive)."""
        span = abs(self.end_angle_deg - self.start_angle_deg)
        return min(span, 360.0 - span)


@dataclass
class DetectedPillar:
    """Poteau cylindrique détecté.

    Attributes:
        cx: Coordonnée X du centre (mètres).
        cy: Coordonnée Y du centre (mètres).
        radius: Rayon du poteau (mètres).
        confidence: Score de confiance dans [0, 1].
        source_slice: Slice d'origine.
    """

    cx: float
    cy: float
    radius: float
    confidence: float = 1.0
    source_slice: str = "high"


def detect_pillars(
    density_map: "DensityMapResult",
    binary_image: np.ndarray,
    source_slice: str = "high",
    min_radius_m: float = _MIN_PILLAR_RADIUS_M,
    max_radius_m: float = _MAX_PILLAR_RADIUS_M,
    param1: float = 50.0,
    param2: float = 15.0,
) -> list[DetectedPillar]:
    """Détecte les poteaux cylindriques par Hough circulaire.

    Utilise cv2.HoughCircles sur l'image binaire normalisée.
    Filtre les cercles dont le rayon est hors des bornes métriques plausibles.

    Args:
        density_map: DensityMapResult pour la conversion pixel → métrique.
        binary_image: Image binaire (uint8, 0 ou 255).
        source_slice: Identifiant de la slice d'origine.
        min_radius_m: Rayon minimal d'un poteau (mètres).
        max_radius_m: Rayon maximal d'un poteau (mètres).
        param1: Seuil du détecteur de Canny interne à HoughCircles.
        param2: Seuil d'accumulation (plus petit = plus de faux positifs).

    Returns:
        Liste de ``DetectedPillar`` triés par confiance décroissante.
        Liste vide si cv2 n'est pas disponible ou si aucun poteau trouvé.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("cv2 non disponible — détection de poteaux ignorée.")
        return []

    res = density_map.resolution
    min_radius_px = max(1, int(min_radius_m / res))
    max_radius_px = max(min_radius_px + 1, int(max_radius_m / res))

    # HoughCircles requiert une image uint8 8 bits, non nulle
    img_norm = _normalize_to_uint8(binary_image)
    if img_norm.max() == 0:
        return []

    circles = cv2.HoughCircles(
        img_norm,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(2, min_radius_px * 2),
        param1=param1,
        param2=param2,
        minRadius=min_radius_px,
        maxRadius=max_radius_px,
    )

    if circles is None:
        logger.debug("detect_pillars : aucun poteau détecté.")
        return []

    pillars: list[DetectedPillar] = []
    for cx_px, cy_px, r_px in circles[0]:
        cx_m, cy_m = _pixel_to_metric(density_map, float(cx_px), float(cy_px))
        r_m = float(r_px) * res

        if not (min_radius_m <= r_m <= max_radius_m):
            continue

        # Confiance : plus le rayon est proche de la valeur centrale, mieux c'est
        mid_r = (min_radius_m + max_radius_m) / 2.0
        conf = 1.0 - abs(r_m - mid_r) / (max_radius_m - min_radius_m)
        conf = max(0.1, min(1.0, conf))

        pillars.append(DetectedPillar(
            cx=cx_m, cy=cy_m, radius=r_m,
            confidence=round(conf, 3),
            source_slice=source_slice,
        ))

    # Trier par confiance décroissante
    pillars.sort(key=lambda p: p.confidence, reverse=True)
    logger.info(
        "detect_pillars [%s] : %d poteau(x) détecté(s).",
        source_slice, len(pillars),
    )
    return pillars


def detect_curved_walls(
    density_map: "DensityMapResult",
    binary_image: np.ndarray,
    source_slice: str = "high",
    min_radius_m: float = _MIN_ARC_RADIUS_M,
    max_radius_m: float = _MAX_ARC_RADIUS_M,
    min_arc_length_m: float = _MIN_ARC_LENGTH_M,
) -> list[DetectedArc]:
    """Détecte les murs courbes par analyse des résidus + ajustement d'arc.

    Algorithme :
    1. Extraire les points de contour de binary_image (pixels de mur en bordure).
    2. Pour chaque fenêtre glissante de _RESIDUE_WINDOW_PX pixels sur ces contours :
       a. Ajuster une droite par moindres carrés.
       b. Calculer le résidu max (distance point → droite).
       c. Si résidu > _HIGH_RESIDUE_RATIO × diamètre de la fenêtre : zone courbe.
    3. Sur chaque zone courbe continue, ajuster un cercle par moindres carrés.
    4. Calculer les angles de début/fin, filtrer par longueur et rayon plausibles.

    Args:
        density_map: DensityMapResult pour la conversion pixel → métrique.
        binary_image: Image binaire (uint8, 0 ou 255).
        source_slice: Identifiant de la slice d'origine.
        min_radius_m: Rayon minimal d'arc plausible (mètres).
        max_radius_m: Rayon maximal d'arc plausible (mètres).
        min_arc_length_m: Longueur minimale d'arc retenu (mètres).

    Returns:
        Liste de ``DetectedArc`` triés par longueur d'arc décroissante.
    """
    # Extraire les points de contour en coordonnées métriques
    contour_pts_m = _extract_contour_points_metric(density_map, binary_image)
    if len(contour_pts_m) < _MIN_CURVED_ZONE_PX:
        return []

    # Identifier les zones courbes (indices dans contour_pts_m)
    curved_zones = _find_curved_zones(contour_pts_m, density_map.resolution)

    arcs: list[DetectedArc] = []
    for zone_indices in curved_zones:
        if len(zone_indices) < 6:  # minimum pour fitter un cercle
            continue
        pts = contour_pts_m[zone_indices]
        arc = _fit_arc(pts, source_slice, min_radius_m, max_radius_m, min_arc_length_m)
        if arc is not None:
            arcs.append(arc)

    # Dédupliquer les arcs très proches
    arcs = _deduplicate_arcs(arcs)
    arcs.sort(key=lambda a: a.arc_length, reverse=True)

    logger.info(
        "detect_curved_walls [%s] : %d arc(s) détecté(s).",
        source_slice, len(arcs),
    )
    return arcs


# ---------------------------------------------------------------------------
# Export DXF
# ---------------------------------------------------------------------------

def export_arcs_to_dxf(
    arcs: list[DetectedArc],
    pillars: list[DetectedPillar],
    doc: "Any",
    layer_config: dict | None = None,
) -> int:
    """Ajoute les arcs et poteaux dans un document DXF existant.

    - Arcs → entités ARC sur le calque MURS_COURBES (magenta ACI 6).
    - Poteaux → entités CIRCLE sur le calque POTEAUX (vert foncé ACI 3).

    Args:
        arcs: Arcs de murs courbes à exporter.
        pillars: Poteaux cylindriques à exporter.
        doc: Document ezdxf ouvert (modifié en place).
        layer_config: Configuration des calques (optionnel).

    Returns:
        Nombre total d'entités ajoutées.
    """
    arc_layer = (layer_config or {}).get("curved_walls", "MURS_COURBES")
    pillar_layer = (layer_config or {}).get("pillars", "POTEAUX")

    _ensure_layer(doc, arc_layer, color=6)    # magenta
    _ensure_layer(doc, pillar_layer, color=3)  # vert

    msp = doc.modelspace()
    n = 0

    for arc in arcs:
        # ARC DXF : centre, rayon, angle_début, angle_fin (degrés CCW depuis +X)
        msp.add_arc(
            center=(arc.cx, arc.cy, 0.0),
            radius=arc.radius,
            start_angle=arc.start_angle_deg,
            end_angle=arc.end_angle_deg,
            dxfattribs={"layer": arc_layer},
        )
        n += 1

    for pillar in pillars:
        msp.add_circle(
            center=(pillar.cx, pillar.cy, 0.0),
            radius=pillar.radius,
            dxfattribs={"layer": pillar_layer},
        )
        n += 1

    logger.info(
        "export_arcs_to_dxf : %d arcs + %d poteaux → %d entités.",
        len(arcs), len(pillars), n,
    )
    return n


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------

def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalise une image en uint8 [0, 255].

    Args:
        image: Image d'entrée (n'importe quel dtype).

    Returns:
        Image uint8 normalisée.
    """
    if image.dtype == np.uint8:
        return image
    max_val = float(image.max())
    if max_val < 1e-9:
        return np.zeros_like(image, dtype=np.uint8)
    return ((image / max_val) * 255).astype(np.uint8)


def _pixel_to_metric(
    density_map: "DensityMapResult",
    col: float,
    row: float,
) -> tuple[float, float]:
    """Convertit des coordonnées pixel (col, row) en métriques (x, y).

    Respecte la convention Y-inversé de DensityMapResult (row 0 = Y max).

    Args:
        density_map: DensityMapResult de référence.
        col: Colonne (axe X de l'image).
        row: Rangée (axe Y de l'image, 0 = haut).

    Returns:
        (x_m, y_m) en mètres.
    """
    res = density_map.resolution
    x_m = density_map.x_min + col * res
    y_m = density_map.y_min + (density_map.height - 1 - row) * res
    return float(x_m), float(y_m)


def _extract_contour_points_metric(
    density_map: "DensityMapResult",
    binary_image: np.ndarray,
) -> np.ndarray:
    """Extrait les points de contour de l'image binaire en coordonnées métriques.

    Utilise la détection de contours par erosion (points de mur adjacents au vide).
    Retourne un tableau (N, 2) float64 avec les positions métriques.

    Args:
        density_map: DensityMapResult pour la conversion.
        binary_image: Image binaire (uint8, 0 ou 255).

    Returns:
        Tableau (N, 2) des positions métriques des points de contour.
    """
    try:
        import cv2
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary_image, kernel, iterations=1)
        contour_mask = (binary_image > 0) & (eroded == 0)
    except ImportError:
        # Fallback sans cv2 : utiliser tous les pixels non nuls
        contour_mask = binary_image > 0

    rows, cols = np.nonzero(contour_mask)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.float64)

    res = density_map.resolution
    xs = density_map.x_min + cols * res
    ys = density_map.y_min + (density_map.height - 1 - rows) * res
    return np.column_stack([xs, ys]).astype(np.float64)


def _find_curved_zones(
    pts: np.ndarray,
    resolution_m: float,
) -> list[np.ndarray]:
    """Identifie les zones courbes dans un ensemble de points.

    Pour chaque point, calcule le résidu de l'ajustement linéaire local
    sur une fenêtre de _RESIDUE_WINDOW_PX voisins ordonnés par chaîne
    de plus proches voisins. Les zones avec résidu élevé sont groupées.

    Si l'ensemble de points est globalement courbe (résidu global élevé par
    rapport à un ajustement linéaire), retourne tous les points comme une
    seule zone courbe. Ceci couvre le cas d'un arc partiel isolé.

    Args:
        pts: Tableau (N, 2) de points en coordonnées métriques, non ordonnés.
        resolution_m: Résolution en mètres/pixel (pour le seuil de résidu).

    Returns:
        Liste de tableaux d'indices, un par zone courbe continue.
    """
    n = len(pts)
    if n < _MIN_CURVED_ZONE_PX:
        return []

    # Cas rapide : si le résidu global est élevé, tout est courbe
    global_residue = _linear_fit_residue(pts)
    global_threshold = resolution_m * _RESIDUE_WINDOW_PX * _HIGH_RESIDUE_RATIO
    if global_residue > global_threshold and n >= _MIN_CURVED_ZONE_PX:
        return [np.arange(n)]

    # Cas général : ordonnancement par chaîne de plus proches voisins
    order = _chain_order(pts)
    pts_ordered = pts[order]

    half = _RESIDUE_WINDOW_PX // 2
    is_curved = np.zeros(n, dtype=bool)

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half)
        window = pts_ordered[start:end]
        if len(window) < 5:
            continue
        residue = _linear_fit_residue(window)
        if residue > global_threshold:
            is_curved[i] = True

    zones = _group_curved_runs(is_curved, order)
    logger.debug(
        "_find_curved_zones : %d zones courbes sur %d points.", len(zones), n
    )
    return zones


def _group_curved_runs(
    is_curved: np.ndarray,
    order: np.ndarray,
) -> list[np.ndarray]:
    """Regroupe les runs consécutifs True dans is_curved en zones d'indices.

    Seules les zones d'au moins _MIN_CURVED_ZONE_PX points sont conservées.

    Args:
        is_curved: Masque booléen de longueur N.
        order: Tableau d'indices (N,) — order[i] = indice original du i-ème point.

    Returns:
        Liste de tableaux d'indices originaux, un par zone courbe.
    """
    zones: list[np.ndarray] = []
    n = len(is_curved)
    in_zone = False
    zone_start = 0

    for i in range(n):
        if is_curved[i] and not in_zone:
            zone_start = i
            in_zone = True
        elif not is_curved[i] and in_zone:
            zone_indices = order[zone_start:i]
            if len(zone_indices) >= _MIN_CURVED_ZONE_PX:
                zones.append(zone_indices)
            in_zone = False

    if in_zone:
        zone_indices = order[zone_start:]
        if len(zone_indices) >= _MIN_CURVED_ZONE_PX:
            zones.append(zone_indices)

    return zones


def _chain_order(pts: np.ndarray) -> np.ndarray:
    """Ordonne les points en chaîne de plus proches voisins (greedy).

    Commence au point le plus à gauche et à chaque étape choisit
    le voisin non visité le plus proche.

    Args:
        pts: Tableau (N, 2) de points.

    Returns:
        Tableau d'indices représentant l'ordre de la chaîne.
    """
    n = len(pts)
    if n <= 1:
        return np.arange(n)

    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=int)

    # Partir du point le plus à gauche (x minimal)
    current = int(np.argmin(pts[:, 0]))
    order[0] = current
    visited[current] = True

    for step in range(1, n):
        # Distance de current à tous les non-visités
        diff = pts - pts[current]
        dists = diff[:, 0] ** 2 + diff[:, 1] ** 2
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        order[step] = nxt
        visited[nxt] = True
        current = nxt

    return order


def _linear_fit_residue(pts: np.ndarray) -> float:
    """Résidu maximal de l'ajustement linéaire d'un ensemble de points.

    Ajuste une droite par SVD sur les points centrés et retourne la distance
    maximale d'un point à la droite ajustée.

    Args:
        pts: Tableau (N, 2) de points.

    Returns:
        Distance maximale en mètres (≥ 0).
    """
    centered = pts - pts.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        # vt[1] = vecteur perpendiculaire à la droite principale
        normal = vt[1]
        distances = np.abs(centered @ normal)
        return float(distances.max())
    except np.linalg.LinAlgError:
        return 0.0


def _fit_arc(
    pts: np.ndarray,
    source_slice: str,
    min_radius_m: float,
    max_radius_m: float,
    min_arc_length_m: float,
) -> DetectedArc | None:
    """Ajuste un arc de cercle sur un ensemble de points par moindres carrés.

    Utilise l'algorithme de Coope (forme linéarisée) pour l'initialisation,
    puis scipy.optimize.least_squares pour le raffinement.

    Args:
        pts: Tableau (N, 2) de points en coordonnées métriques.
        source_slice: Identifiant de la slice.
        min_radius_m: Rayon minimal acceptable.
        max_radius_m: Rayon maximal acceptable.
        min_arc_length_m: Longueur minimale de l'arc pour être conservé.

    Returns:
        ``DetectedArc`` si l'ajustement est bon, ``None`` sinon.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        # Fallback : utiliser seulement l'initialisation linéaire
        cx, cy, r = _fit_circle_algebraic(pts)
        if cx is None:
            return None
        return _make_arc(pts, cx, cy, r, source_slice,
                         min_radius_m, max_radius_m, min_arc_length_m)

    # Initialisation par méthode algébrique (Coope)
    cx0, cy0, r0 = _fit_circle_algebraic(pts)
    if cx0 is None:
        return None

    # Raffinement par moindres carrés
    def residuals(params: np.ndarray) -> np.ndarray:
        cx, cy, r = params
        dist = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        return dist - r

    try:
        result = least_squares(
            residuals,
            x0=[cx0, cy0, r0],
            bounds=(
                [-np.inf, -np.inf, min_radius_m],
                [np.inf, np.inf, max_radius_m],
            ),
            max_nfev=200,
        )
        if not result.success and result.cost > 1.0:
            return None
        cx, cy, r = float(result.x[0]), float(result.x[1]), float(result.x[2])
    except (ValueError, RuntimeError):
        cx, cy, r = cx0, cy0, r0

    return _make_arc(pts, cx, cy, r, source_slice,
                     min_radius_m, max_radius_m, min_arc_length_m)


def _fit_circle_algebraic(
    pts: np.ndarray,
) -> tuple[float, float, float] | tuple[None, None, None]:
    """Ajuste un cercle par méthode algébrique (Coope / Kasa).

    Résout le système linéaire : 2x·cx + 2y·cy + d = x²+y² où d = cx²+cy²-r².

    Args:
        pts: Tableau (N, 2) de points.

    Returns:
        (cx, cy, r) ou (None, None, None) si le système est singulier.
    """
    if len(pts) < 3:
        return None, None, None

    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(pts))])
    b = x ** 2 + y ** 2

    try:
        result, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, None

    if rank < 3:
        return None, None, None

    cx, cy = float(result[0]), float(result[1])
    r = float(np.sqrt(max(0.0, result[2] + cx ** 2 + cy ** 2)))
    return cx, cy, r


def _make_arc(
    pts: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    source_slice: str,
    min_radius_m: float,
    max_radius_m: float,
    min_arc_length_m: float,
) -> DetectedArc | None:
    """Construit un DetectedArc à partir du cercle ajusté et des points.

    Calcule les angles de début/fin depuis les positions angulaires des points
    par rapport au centre du cercle. Filtre par rayon et longueur d'arc.

    Args:
        pts: Points utilisés pour l'ajustement.
        cx, cy: Centre du cercle ajusté.
        r: Rayon du cercle ajusté.
        source_slice: Identifiant de la slice.
        min_radius_m: Rayon minimal acceptable.
        max_radius_m: Rayon maximal acceptable.
        min_arc_length_m: Longueur minimale de l'arc (mètres).

    Returns:
        ``DetectedArc`` si valide, ``None`` sinon.
    """
    if not (min_radius_m <= r <= max_radius_m):
        return None

    # Angles de chaque point par rapport au centre
    ang = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    start_deg = float(ang.min())
    end_deg = float(ang.max())

    # Longueur de l'arc
    span_deg = end_deg - start_deg
    arc_len = r * math.radians(span_deg)
    if arc_len < min_arc_length_m:
        return None

    if span_deg < _MIN_ARC_ANGLE_DEG:
        return None

    # Résidu moyen = mesure de qualité → confiance
    distances = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    mean_residue = float(np.abs(distances - r).mean())
    conf = max(0.1, 1.0 - min(1.0, mean_residue / (r + 1e-9) * 10))

    return DetectedArc(
        cx=round(cx, 4),
        cy=round(cy, 4),
        radius=round(r, 4),
        start_angle_deg=round(start_deg, 2),
        end_angle_deg=round(end_deg, 2),
        confidence=round(conf, 3),
        source_slice=source_slice,
    )


def _deduplicate_arcs(arcs: list[DetectedArc], tol_m: float = 0.30) -> list[DetectedArc]:
    """Supprime les arcs dont le centre et le rayon sont très proches.

    Garde l'arc avec la confiance la plus haute de chaque groupe.

    Args:
        arcs: Arcs à dédupliquer.
        tol_m: Distance maximale entre centres pour considérer deux arcs identiques.

    Returns:
        Liste d'arcs dédupliqués.
    """
    if len(arcs) <= 1:
        return arcs

    kept: list[DetectedArc] = []
    used = [False] * len(arcs)
    arcs_sorted = sorted(arcs, key=lambda a: a.confidence, reverse=True)

    for i, arc in enumerate(arcs_sorted):
        if used[i]:
            continue
        kept.append(arc)
        for j in range(i + 1, len(arcs_sorted)):
            if used[j]:
                continue
            other = arcs_sorted[j]
            dist = math.hypot(arc.cx - other.cx, arc.cy - other.cy)
            if dist < tol_m and abs(arc.radius - other.radius) < tol_m:
                used[j] = True

    return kept


def _ensure_layer(doc: "Any", name: str, color: int) -> None:
    """Ajoute un calque au document s'il n'existe pas.

    Args:
        doc: Document ezdxf.
        name: Nom du calque.
        color: Couleur ACI.
    """
    if name not in doc.layers:
        doc.layers.new(name, dxfattribs={"color": color})
