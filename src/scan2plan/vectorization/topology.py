"""Reconstruction topologique : connexion des murs aux intersections.

Construit un graphe dont les nœuds sont les intersections de murs et les
arêtes sont les segments de murs. Permet de détecter les pièces comme des
cycles fermés dans le graphe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from scan2plan.detection.line_detection import DetectedSegment
from scan2plan.detection.openings import Opening
from scan2plan.utils.geometry import (
    angle_between_segments,
    line_intersection,
)

logger = logging.getLogger(__name__)

# Distance max entre deux extrémités pour qu'elles soient considérées proches
_ENDPOINT_TOLERANCE = 0.10  # 10 cm

# Angle minimal pour qu'une intersection soit considérée non-parallèle (radians)
_MIN_INTERSECTION_ANGLE_RAD = float(np.deg2rad(30.0))

# Distance en-dessous de laquelle deux nœuds sont fusionnés
_NODE_MERGE_DIST = 0.02  # 2 cm


@dataclass
class WallGraph:
    """Graphe topologique des murs.

    Attributes:
        nodes: Liste de points d'intersection (x, y) en mètres.
        edges: Liste de paires d'indices de nœuds ``(i, j)`` définissant
            les arêtes (une arête = un segment de mur).
        segments: Segments géométriques des arêtes (même ordre que ``edges``).
        openings: Ouvertures associées (non liées aux arêtes pour l'instant).
    """

    nodes: list[tuple[float, float]] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)
    segments: list[DetectedSegment] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)


def build_wall_graph(
    segments: list[DetectedSegment],
    openings: list[Opening],
    intersection_distance: float = 0.05,
    min_segment_length: float = 0.10,
) -> WallGraph:
    """Construit le graphe topologique des murs.

    Algorithme :
    1. Pour chaque paire de segments non-parallèles dont une extrémité de
       l'un est proche d'une extrémité de l'autre (< ``intersection_distance``),
       calculer le point d'intersection des droites porteuses et prolonger /
       raccourcir les segments jusqu'à ce point.
    2. Les extrémités de chaque segment (après ajustement) deviennent les
       nœuds du graphe.
    3. Fusionner les nœuds très proches (< ``_NODE_MERGE_DIST``).
    4. Construire les arêtes.
    5. Supprimer les segments orphelins (une seule extrémité connectée) dont
       la longueur est < ``min_segment_length``.

    Args:
        segments: Segments de murs régularisés.
        openings: Ouvertures détectées (stockées dans le graphe sans traitement).
        intersection_distance: Distance maximale entre deux extrémités pour
            tenter une résolution d'intersection (mètres).
        min_segment_length: Longueur minimale d'un segment gardé dans le graphe
            (mètres). Les segments plus courts sont supprimés.

    Returns:
        ``WallGraph`` avec la topologie complète.
    """
    if not segments:
        return WallGraph(openings=list(openings))

    # Étape 1 : résoudre les intersections → ajuster les extrémités
    adjusted = _resolve_intersections(segments, intersection_distance)

    # Étape 2 : filtrer les segments trop courts
    adjusted = [s for s in adjusted if s.length >= min_segment_length]

    if not adjusted:
        return WallGraph(openings=list(openings))

    # Étape 3 : collecter toutes les extrémités comme nœuds candidats
    raw_nodes: list[tuple[float, float]] = []
    for seg in adjusted:
        raw_nodes.append((seg.x1, seg.y1))
        raw_nodes.append((seg.x2, seg.y2))

    # Étape 4 : fusionner les nœuds proches
    merged_nodes, node_map = _merge_nodes(raw_nodes, _NODE_MERGE_DIST)

    # Étape 5 : construire les arêtes (un segment → une arête)
    edges: list[tuple[int, int]] = []
    edge_segments: list[DetectedSegment] = []
    for idx, seg in enumerate(adjusted):
        ni = int(node_map[2 * idx])
        nj = int(node_map[2 * idx + 1])
        if ni != nj:
            edges.append((ni, nj))
            edge_segments.append(seg)

    graph = WallGraph(
        nodes=merged_nodes,
        edges=edges,
        segments=edge_segments,
        openings=list(openings),
    )

    # Étape 6 : nettoyer la topologie
    graph = clean_topology(graph, min_segment_length=min_segment_length)

    logger.info(
        "build_wall_graph : %d nœuds, %d arêtes, %d ouvertures.",
        len(graph.nodes),
        len(graph.edges),
        len(graph.openings),
    )
    return graph


def detect_rooms(graph: WallGraph) -> list[list[int]]:
    """Détecte les pièces comme des cycles minimaux dans le graphe des murs.

    Utilise NetworkX pour trouver les cycles de base (minimum cycle basis).
    Chaque cycle représente une pièce fermée.

    Args:
        graph: Graphe topologique des murs.

    Returns:
        Liste de cycles. Chaque cycle est une liste d'indices de nœuds
        formant une pièce fermée. Liste vide si aucun cycle.
    """
    import networkx as nx

    G = _build_nx_graph(graph)

    if G.number_of_edges() == 0:
        return []

    try:
        cycles = nx.minimum_cycle_basis(G)
    except Exception as exc:
        logger.warning("detect_rooms : erreur lors de la détection de cycles : %s", exc)
        return []

    logger.info("detect_rooms : %d pièce(s) détectée(s).", len(cycles))
    return cycles


def clean_topology(
    graph: WallGraph,
    min_segment_length: float = 0.10,
) -> WallGraph:
    """Nettoie la topologie du graphe.

    Opérations :
    - Supprimer les arêtes en doublon (même paire de nœuds).
    - Fusionner les nœuds très proches (< ``_NODE_MERGE_DIST``).
    - Supprimer les segments pendants courts dont le nœud extrême est de
      degré 1 et la longueur < ``min_segment_length``.
    - Supprimer les nœuds isolés (degré 0).

    Args:
        graph: Graphe à nettoyer (modifié en place puis retourné).
        min_segment_length: Longueur minimale des segments conservés (mètres).

    Returns:
        Graphe nettoyé (nouvelle instance).
    """
    # 1. Dédupliquer les arêtes
    edges, segs = _deduplicate_edges(graph.edges, graph.segments)

    # 2. Re-fusionner les nœuds proches (au cas où de nouveaux ont été créés)
    nodes, edges, segs = _remerge_nodes(graph.nodes, edges, segs)

    # 3. Supprimer les segments pendants courts
    edges, segs = _remove_short_pending(nodes, edges, segs, min_segment_length)

    # 4. Compacter : supprimer les nœuds sans aucune arête
    nodes, edges, segs = _compact_graph(nodes, edges, segs)

    return WallGraph(
        nodes=nodes,
        edges=edges,
        segments=segs,
        openings=graph.openings,
    )


# ---------------------------------------------------------------------------
# Helpers privés — résolution des intersections
# ---------------------------------------------------------------------------

def _resolve_intersections(
    segments: list[DetectedSegment],
    distance: float,
) -> list[DetectedSegment]:
    """Ajuste les extrémités des segments aux points d'intersection.

    Pour chaque paire (i, j) de segments non-parallèles dont une extrémité
    de l'un est proche (< ``distance``) du point d'intersection de leurs
    droites porteuses, déplace ces extrémités à ce point d'intersection.

    Args:
        segments: Segments originaux.
        distance: Seuil de proximité des extrémités (mètres).

    Returns:
        Nouveaux segments avec extrémités ajustées.
    """
    pts: list[list[float]] = [[s.x1, s.y1, s.x2, s.y2] for s in segments]

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            _snap_pair_to_intersection(pts, i, j, distance)

    return [
        DetectedSegment(
            x1=pt[0], y1=pt[1], x2=pt[2], y2=pt[3],
            source_slice=seg.source_slice,
            confidence=seg.confidence,
        )
        for seg, pt in zip(segments, pts)
    ]


def _snap_pair_to_intersection(
    pts: list[list[float]],
    i: int,
    j: int,
    distance: float,
) -> None:
    """Tente de snapper les extrémités proches des segments i et j à leur intersection.

    Modifie ``pts`` en place.

    Args:
        pts: Coordonnées mutables ``[[x1,y1,x2,y2], ...]``.
        i: Indice du premier segment.
        j: Indice du deuxième segment.
        distance: Distance max entre extrémité et point d'intersection.
    """
    angle = angle_between_segments(
        (pts[i][0], pts[i][1], pts[i][2], pts[i][3]),
        (pts[j][0], pts[j][1], pts[j][2], pts[j][3]),
    )
    if angle < _MIN_INTERSECTION_ANGLE_RAD:
        return

    inter = line_intersection(
        (pts[i][0], pts[i][1], pts[i][2], pts[i][3]),
        (pts[j][0], pts[j][1], pts[j][2], pts[j][3]),
    )
    if inter is None:
        return

    ix, iy = inter
    _snap_endpoints_to_point(pts[i], ix, iy, distance)
    _snap_endpoints_to_point(pts[j], ix, iy, distance)


def _snap_endpoints_to_point(
    pt: list[float],
    ix: float,
    iy: float,
    distance: float,
) -> None:
    """Déplace une extrémité d'un segment vers (ix, iy) si elle est assez proche.

    Vérifie les deux extrémités (indices 0-1 et 2-3) et déplace la première
    qui est à moins de ``distance`` du point cible.

    Args:
        pt: Coordonnées ``[x1, y1, x2, y2]`` du segment, modifiées en place.
        ix: Coordonnée X du point cible.
        iy: Coordonnée Y du point cible.
        distance: Seuil de proximité (mètres).
    """
    for k in (0, 2):
        if float(np.hypot(pt[k] - ix, pt[k + 1] - iy)) <= distance:
            pt[k] = ix
            pt[k + 1] = iy


# ---------------------------------------------------------------------------
# Helpers privés — gestion des nœuds
# ---------------------------------------------------------------------------

def _merge_nodes(
    raw: list[tuple[float, float]],
    threshold: float,
) -> tuple[list[tuple[float, float]], list[int]]:
    """Fusionne les nœuds proches en un représentant commun.

    Args:
        raw: Liste brute de points (doublons possibles).
        threshold: Distance en dessous de laquelle deux points sont fusionnés.

    Returns:
        ``(merged_nodes, node_map)`` où ``node_map[i]`` est l'indice dans
        ``merged_nodes`` du point ``raw[i]``.
    """
    merged: list[tuple[float, float]] = []
    node_map: list[int] = []

    for pt in raw:
        found = None
        for mi, mp in enumerate(merged):
            if float(np.hypot(pt[0] - mp[0], pt[1] - mp[1])) < threshold:
                found = mi
                break
        if found is None:
            node_map.append(len(merged))
            merged.append(pt)
        else:
            node_map.append(found)

    return merged, node_map


def _remerge_nodes(
    nodes: list[tuple[float, float]],
    edges: list[tuple[int, int]],
    segs: list[DetectedSegment],
) -> tuple[list[tuple[float, float]], list[tuple[int, int]], list[DetectedSegment]]:
    """Refusionne les nœuds proches dans un graphe existant.

    Args:
        nodes: Nœuds courants.
        edges: Arêtes courantes.
        segs: Segments des arêtes.

    Returns:
        ``(new_nodes, new_edges, new_segs)`` après refusion.
    """
    merged_nodes, node_map = _merge_nodes(nodes, _NODE_MERGE_DIST)
    # node_map[i] = cluster index for old node i
    # Build a mapping: cluster index → new compact index
    n_clusters = len(merged_nodes)
    cluster_to_new: list[int] = [-1] * n_clusters
    new_nodes: list[tuple[float, float]] = []
    for old_idx, cluster in enumerate(node_map):
        if cluster_to_new[cluster] == -1:
            cluster_to_new[cluster] = len(new_nodes)
            new_nodes.append(nodes[old_idx])

    # old node index → new compact index
    remapped = [cluster_to_new[node_map[i]] for i in range(len(nodes))]
    new_edges = [(int(remapped[a]), int(remapped[b])) for a, b in edges]
    return new_nodes, new_edges, segs


# ---------------------------------------------------------------------------
# Helpers privés — nettoyage
# ---------------------------------------------------------------------------

def _deduplicate_edges(
    edges: list[tuple[int, int]],
    segs: list[DetectedSegment],
) -> tuple[list[tuple[int, int]], list[DetectedSegment]]:
    """Supprime les arêtes en doublon (même paire de nœuds).

    Args:
        edges: Arêtes (i, j).
        segs: Segments associés.

    Returns:
        ``(edges_dedup, segs_dedup)``.
    """
    seen: set[frozenset[int]] = set()
    new_edges: list[tuple[int, int]] = []
    new_segs: list[DetectedSegment] = []
    for (a, b), seg in zip(edges, segs):
        key = frozenset({a, b})
        if key not in seen:
            seen.add(key)
            new_edges.append((a, b))
            new_segs.append(seg)
    return new_edges, new_segs


def _node_degrees(
    n_nodes: int,
    edges: list[tuple[int, int]],
) -> list[int]:
    """Calcule le degré de chaque nœud.

    Args:
        n_nodes: Nombre de nœuds.
        edges: Arêtes.

    Returns:
        Liste de degrés, indexée par numéro de nœud.
    """
    deg = [0] * n_nodes
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return deg


def _remove_short_pending(
    nodes: list[tuple[float, float]],
    edges: list[tuple[int, int]],
    segs: list[DetectedSegment],
    min_length: float,
) -> tuple[list[tuple[int, int]], list[DetectedSegment]]:
    """Supprime les segments pendants courts de façon itérative.

    Un segment pendant est une arête dont un nœud extrême a degré 1.
    Si ce segment est plus court que ``min_length``, il est supprimé.
    On itère jusqu'à stabilisation.

    Args:
        nodes: Nœuds du graphe.
        edges: Arêtes.
        segs: Segments associés.
        min_length: Longueur minimale conservée (mètres).

    Returns:
        ``(edges_clean, segs_clean)``.
    """
    changed = True
    cur_edges = list(edges)
    cur_segs = list(segs)

    while changed:
        changed = False
        deg = _node_degrees(len(nodes), cur_edges)
        keep_edges: list[tuple[int, int]] = []
        keep_segs: list[DetectedSegment] = []
        for (a, b), seg in zip(cur_edges, cur_segs):
            is_pending = deg[a] == 1 or deg[b] == 1
            if is_pending and seg.length < min_length:
                changed = True
            else:
                keep_edges.append((a, b))
                keep_segs.append(seg)
        cur_edges, cur_segs = keep_edges, keep_segs

    return cur_edges, cur_segs


def _compact_graph(
    nodes: list[tuple[float, float]],
    edges: list[tuple[int, int]],
    segs: list[DetectedSegment],
) -> tuple[list[tuple[float, float]], list[tuple[int, int]], list[DetectedSegment]]:
    """Supprime les nœuds isolés et reindex les indices d'arêtes.

    Args:
        nodes: Nœuds (certains peuvent être devenus orphelins).
        edges: Arêtes.
        segs: Segments associés.

    Returns:
        ``(new_nodes, new_edges, new_segs)`` compactés.
    """
    used: set[int] = set()
    for a, b in edges:
        used.add(a)
        used.add(b)

    old_to_new = {old: new for new, old in enumerate(sorted(used))}
    new_nodes = [nodes[i] for i in sorted(used)]
    new_edges = [(old_to_new[a], old_to_new[b]) for a, b in edges]
    return new_nodes, new_edges, segs


# ---------------------------------------------------------------------------
# Helper privé — conversion vers NetworkX
# ---------------------------------------------------------------------------

def _build_nx_graph(graph: WallGraph) -> "networkx.Graph":
    """Convertit un WallGraph en graphe NetworkX non orienté.

    Args:
        graph: Graphe topologique.

    Returns:
        Graphe NetworkX avec les nœuds et arêtes du WallGraph.
    """
    import networkx as nx

    G: networkx.Graph = nx.Graph()
    G.add_nodes_from(range(len(graph.nodes)))
    for a, b in graph.edges:
        G.add_edge(a, b)
    return G
