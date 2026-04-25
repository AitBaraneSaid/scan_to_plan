"""Topologie légère : snap d'extrémités + fermeture de coins.

Remplace _resolve_intersections(distance=0.50) qui détruisait des segments
en étendant trop agressivement. Philosophie : toucher le moins possible.

Deux opérations seulement :
1. Snap des extrémités proches (< 3 cm) → corrige le bruit de quantification Hough.
2. Fermeture de coins (< 8 cm, angle > 60°) → referme les coins de pièces ouverts.

Aucune de ces deux opérations ne peut franchir une embrasure de porte (> 50 cm).
"""

from __future__ import annotations

import logging

import numpy as np

from scan2plan.detection.line_detection import DetectedSegment

logger = logging.getLogger(__name__)

Segment = DetectedSegment

# Longueur minimale d'un segment après snap — en dessous, c'est un artefact
_MIN_SEGMENT_LENGTH = 0.01  # 1 cm


def snap_endpoints(
    segments: list[Segment],
    tolerance: float = 0.03,
) -> list[Segment]:
    """Moyenne les extrémités de segments qui sont à moins de ``tolerance``.

    Algorithme :
    1. Collecter toutes les extrémités (2 par segment → 2N points).
    2. Grouper par proximité : union-find sur les paires à distance < tolerance.
    3. Pour chaque groupe, calculer le centroïde.
    4. Remplacer chaque extrémité par le centroïde de son groupe.
    5. Reconstruire les segments ; supprimer ceux raccourcis à < 1 cm.

    Le seuil de 3 cm est l'imprécision typique de Hough à 5 mm/pixel.
    Aucune embrasure de porte ne fait 3 cm — ce seuil est architecturalement
    inoffensif.

    Args:
        segments: Segments issus du pipeline (après snap angulaire et pairing).
        tolerance: Distance maximale entre deux extrémités pour les fusionner (m).

    Returns:
        Segments avec les extrémités proches fusionnées sur leur centroïde.
        Les artefacts de moins de 1 cm sont supprimés.

    Example:
        >>> s1 = DetectedSegment(0, 0, 1, 0, "high", 0.9)
        >>> s2 = DetectedSegment(1.02, 0, 1.02, 1, "high", 0.9)
        >>> result = snap_endpoints([s1, s2], tolerance=0.03)
        >>> abs(result[0].x2 - result[1].x1) < 0.001
        True
    """
    if not segments:
        return []

    n = len(segments)
    # Extraire les 2N extrémités : indice i→ (i//2 = segment, i%2 = endpoint)
    endpoints = np.zeros((2 * n, 2), dtype=float)
    for i, seg in enumerate(segments):
        endpoints[2 * i] = [seg.x1, seg.y1]
        endpoints[2 * i + 1] = [seg.x2, seg.y2]

    # Union-Find pour grouper les extrémités proches
    parent = list(range(2 * n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # O(N²) — acceptable pour N < 500 segments
    for i in range(2 * n):
        for j in range(i + 1, 2 * n):
            if np.hypot(endpoints[i, 0] - endpoints[j, 0],
                        endpoints[i, 1] - endpoints[j, 1]) < tolerance:
                union(i, j)

    # Calculer le centroïde de chaque groupe
    groups: dict[int, list[int]] = {}
    for i in range(2 * n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    new_endpoints = endpoints.copy()
    for members in groups.values():
        centroid = endpoints[members].mean(axis=0)
        for m in members:
            new_endpoints[m] = centroid

    # Reconstruire les segments
    result: list[Segment] = []
    n_removed = 0
    for i, seg in enumerate(segments):
        x1, y1 = new_endpoints[2 * i]
        x2, y2 = new_endpoints[2 * i + 1]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < _MIN_SEGMENT_LENGTH:
            n_removed += 1
            continue
        result.append(DetectedSegment(
            x1=float(x1), y1=float(y1),
            x2=float(x2), y2=float(y2),
            source_slice=seg.source_slice,
            confidence=seg.confidence,
        ))

    logger.info(
        "snap_endpoints (tol=%.3fm) : %d → %d segments (%d supprimés).",
        tolerance, n, len(result), n_removed,
    )
    return result


def close_corners(
    segments: list[Segment],
    max_extension: float = 0.08,
    min_angle_deg: float = 60.0,
) -> list[Segment]:
    """Étend les segments quasi-perpendiculaires proches jusqu'à leur intersection.

    Pour chaque paire (Si, Sj) :
    1. Angle entre Si et Sj > min_angle_deg (quasi-perpendiculaires).
    2. Il existe une extrémité de Si et une extrémité de Sj distantes de < max_extension.
    3. Calculer le point d'intersection P des droites porteuses.
    4. Vérifier que P prolonge les deux segments (pas de l'autre côté).
    5. Vérifier que la distance extrémité→P < max_extension pour les deux.
    6. Étendre Si et Sj jusqu'à P.

    Le seuil de 8 cm garantit qu'aucune embrasure de porte (> 50 cm) ne sera
    franchie. Les coins de pièces ont typiquement des gaps de 3–8 cm après Hough.

    Args:
        segments: Segments après snap_endpoints.
        max_extension: Extension maximale autorisée par extrémité (m).
        min_angle_deg: Angle minimal entre deux segments pour les considérer
            quasi-perpendiculaires et candidats à la fermeture (degrés).

    Returns:
        Segments avec les coins fermés.

    Example:
        >>> s1 = DetectedSegment(0, 0, 1, 0, "high", 0.9)    # horizontal
        >>> s2 = DetectedSegment(1.05, 0.05, 1.05, 1, "high", 0.9)  # vertical, gap 5cm
        >>> result = close_corners([s1, s2], max_extension=0.08)
        >>> abs(result[0].x2 - result[1].x1) < 0.001
        True
    """
    if not segments:
        return []

    min_angle_rad = float(np.deg2rad(min_angle_deg))
    # Travailler sur une liste mutable de tuples (x1, y1, x2, y2, source, conf)
    pts: list[list[float]] = [
        [seg.x1, seg.y1, seg.x2, seg.y2] for seg in segments
    ]
    n = len(segments)

    for i in range(n):
        for j in range(i + 1, n):
            angle = _angle_between(pts[i], pts[j])
            if angle < min_angle_rad:
                continue  # trop parallèles → pas un coin

            # Chercher la paire d'extrémités la plus proche
            ei, ej, dist = _closest_endpoint_pair(pts[i], pts[j])
            if dist > max_extension:
                continue  # trop loin → pas un coin à fermer

            # Calculer le point d'intersection des droites porteuses
            p = _line_intersection(pts[i], pts[j])
            if p is None:
                continue

            # Vérifier que P prolonge (et ne dépasse pas de max_extension) les deux
            ext_i = _extension_distance(pts[i], ei, p)
            ext_j = _extension_distance(pts[j], ej, p)
            if ext_i > max_extension or ext_j > max_extension:
                continue
            if not _is_prolongation(pts[i], ei, p):
                continue
            if not _is_prolongation(pts[j], ej, p):
                continue

            # Étendre les extrémités vers P
            if ei == 0:
                pts[i][0], pts[i][1] = p[0], p[1]
            else:
                pts[i][2], pts[i][3] = p[0], p[1]

            if ej == 0:
                pts[j][0], pts[j][1] = p[0], p[1]
            else:
                pts[j][2], pts[j][3] = p[0], p[1]

    # Reconstruire les segments
    result: list[Segment] = []
    n_extended = 0
    for i, seg in enumerate(segments):
        x1, y1, x2, y2 = pts[i]
        changed = (
            abs(x1 - seg.x1) > 1e-9 or abs(y1 - seg.y1) > 1e-9 or
            abs(x2 - seg.x2) > 1e-9 or abs(y2 - seg.y2) > 1e-9
        )
        if changed:
            n_extended += 1
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < _MIN_SEGMENT_LENGTH:
            continue
        result.append(DetectedSegment(
            x1=float(x1), y1=float(y1),
            x2=float(x2), y2=float(y2),
            source_slice=seg.source_slice,
            confidence=seg.confidence,
        ))

    logger.info(
        "close_corners (max=%.3fm, min_angle=%.0f°) : %d segments, %d coins fermés.",
        max_extension, min_angle_deg, len(result), n_extended,
    )
    return result


def apply_light_topology(
    segments: list[Segment],
    snap_tolerance: float = 0.03,
    corner_max_extension: float = 0.08,
    corner_min_angle_deg: float = 60.0,
) -> list[Segment]:
    """Applique le snap puis la fermeture de coins.

    Remplace ``_resolve_intersections(distance=0.50)`` qui détruisait des
    segments en étendant trop agressivement (89 → 37 segments).

    Philosophie : toucher le moins possible. Corriger le bruit de Hough (snap
    3 cm), fermer les coins évidents (8 cm, > 60°). Ne jamais franchir une
    embrasure architecturale.

    Args:
        segments: Segments après snap angulaire + pairing.
        snap_tolerance: Seuil de fusion des extrémités proches (m).
        corner_max_extension: Extension maximale par coin (m).
        corner_min_angle_deg: Angle minimal pour fermer un coin (degrés).

    Returns:
        Segments avec la topologie légère appliquée.
    """
    snapped = snap_endpoints(segments, snap_tolerance)
    closed = close_corners(snapped, corner_max_extension, corner_min_angle_deg)
    logger.info(
        "apply_light_topology : %d → %d segments (snap=%.3fm, coin=%.3fm).",
        len(segments), len(closed), snap_tolerance, corner_max_extension,
    )
    return closed


# ---------------------------------------------------------------------------
# Helpers géométriques privés
# ---------------------------------------------------------------------------


def _angle_between(p: list[float], q: list[float]) -> float:
    """Différence angulaire entre deux segments, dans [0, π/2].

    Args:
        p: ``[x1, y1, x2, y2]`` du premier segment.
        q: ``[x1, y1, x2, y2]`` du second segment.

    Returns:
        Différence angulaire en radians dans ``[0, π/2]``.
    """
    ap = float(np.arctan2(p[3] - p[1], p[2] - p[0]) % np.pi)
    aq = float(np.arctan2(q[3] - q[1], q[2] - q[0]) % np.pi)
    diff = abs(ap - aq) % np.pi
    return float(min(diff, np.pi - diff))


def _closest_endpoint_pair(
    p: list[float],
    q: list[float],
) -> tuple[int, int, float]:
    """Trouve la paire d'extrémités (ep, eq) la plus proche entre deux segments.

    Args:
        p: ``[x1, y1, x2, y2]`` du premier segment.
        q: ``[x1, y1, x2, y2]`` du second segment.

    Returns:
        ``(idx_p, idx_q, distance)`` où ``idx`` vaut 0 (p1) ou 1 (p2),
        et ``distance`` est la distance euclidienne minimale.
    """
    endpoints_p = [(p[0], p[1]), (p[2], p[3])]
    endpoints_q = [(q[0], q[1]), (q[2], q[3])]

    best_dist = float("inf")
    best_ei, best_ej = 0, 0
    for ei, ep in enumerate(endpoints_p):
        for ej, eq in enumerate(endpoints_q):
            d = float(np.hypot(ep[0] - eq[0], ep[1] - eq[1]))
            if d < best_dist:
                best_dist = d
                best_ei, best_ej = ei, ej

    return best_ei, best_ej, best_dist


def _line_intersection(
    p: list[float],
    q: list[float],
) -> tuple[float, float] | None:
    """Calcule le point d'intersection des droites porteuses de deux segments.

    Args:
        p: ``[x1, y1, x2, y2]`` du premier segment.
        q: ``[x1, y1, x2, y2]`` du second segment.

    Returns:
        ``(x, y)`` du point d'intersection, ou ``None`` si parallèles.
    """
    dx1, dy1 = p[2] - p[0], p[3] - p[1]
    dx2, dy2 = q[2] - q[0], q[3] - q[1]
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-9:
        return None  # parallèles
    t = ((q[0] - p[0]) * dy2 - (q[1] - p[1]) * dx2) / denom
    x = p[0] + t * dx1
    y = p[1] + t * dy1
    return float(x), float(y)


def _extension_distance(
    seg: list[float],
    endpoint_idx: int,
    target: tuple[float, float],
) -> float:
    """Distance entre l'extrémité ``endpoint_idx`` du segment et ``target``.

    Args:
        seg: ``[x1, y1, x2, y2]``.
        endpoint_idx: 0 → (x1, y1), 1 → (x2, y2).
        target: Point cible ``(x, y)``.

    Returns:
        Distance euclidienne en mètres.
    """
    if endpoint_idx == 0:
        return float(np.hypot(target[0] - seg[0], target[1] - seg[1]))
    return float(np.hypot(target[0] - seg[2], target[1] - seg[3]))


def _is_prolongation(
    seg: list[float],
    endpoint_idx: int,
    target: tuple[float, float],
) -> bool:
    """Vérifie que ``target`` prolonge le segment depuis ``endpoint_idx``.

    "Prolonge" signifie que le vecteur (extrémité → target) pointe dans le
    même sens que (extrémité → autre extrémité), i.e. produit scalaire ≥ 0.
    Cela garantit qu'on n'étend pas un segment de l'autre côté.

    Args:
        seg: ``[x1, y1, x2, y2]``.
        endpoint_idx: 0 (p1) ou 1 (p2).
        target: Point cible ``(x, y)``.

    Returns:
        ``True`` si ``target`` est dans le sens de prolongation.
    """
    if endpoint_idx == 0:
        # On étend depuis p1 → la direction naturelle est p1→p2
        # Le vecteur p1→target doit pointer dans la direction opposée : p1←p2
        # (on va vers l'extérieur du segment, pas vers l'intérieur)
        along_x = seg[0] - seg[2]   # vecteur p2→p1 (direction "extérieure" depuis p1)
        along_y = seg[1] - seg[3]
        to_x = target[0] - seg[0]
        to_y = target[1] - seg[1]
    else:
        # On étend depuis p2 → la direction "extérieure" est p1→p2
        along_x = seg[2] - seg[0]
        along_y = seg[3] - seg[1]
        to_x = target[0] - seg[2]
        to_y = target[1] - seg[3]

    dot = along_x * to_x + along_y * to_y
    return dot >= 0.0
