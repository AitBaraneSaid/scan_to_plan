"""Script de diagnostic visuel du pipeline Scan2Plan.

Exécute le pipeline étape par étape et génère une image PNG après chaque étape.
Permet de valider visuellement que chaque étape fait ce qu'elle doit faire.

Usage:
    python scripts/debug_pipeline.py <fichier.e57|.las> [output_dir]

Sorties (dans output_dir/) :
    00_nuage_stats.png         Histogramme Z + sol/plafond
    01_density_maps.png        3 density maps côte à côte
    01_binary_maps.png         3 images binaires côte à côte
    02_hough_brut.png          Segments Hough bruts (vert)
    03_multi_slice.png         Murs confirmés (jaune), mobilier (rouge)
    04_micro_fusion.png        Après micro-fusion (cyan)
    05_cleanup.png             Gardés (blanc) / supprimés (rouge pointillé)
    06_regularization.png      Segments régularisés + directions dominantes
    06_angle_histogram.png     Histogramme angulaire avec pics
    07_pairing.png             Paires par couleur, non pairés en gris
    07_pairing_thickness.png   Paires colorées par épaisseur
    08_snap.png                Après snap, cercles jaunes aux points de snap
    09_corners.png             Après fermeture coins, croix rouges
    10_final.png               Résultat final par calque DXF
    10_final_vs_hough.png      Comparaison Hough brut vs final
    11_recap.png               Grille 2x5 toutes étapes
    recap.txt                  Rapport texte complet
    plan.dxf                   Fichier DXF final
"""

from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Pas d'affichage — sauvegarde uniquement

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Import des modules pipeline
# ---------------------------------------------------------------------------

# Ajouter le dossier src au path pour l'import autonome
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from scan2plan.config import ScanConfig
from scan2plan.detection.line_detection import DetectedSegment, detect_lines_hough
from scan2plan.detection.morphology import binarize_density_map, morphological_cleanup
from scan2plan.detection.multi_slice_filter import (
    classify_segments,
    get_door_candidates,
    match_segments_across_slices,
)
from scan2plan.detection.segment_fusion import fuse_collinear_segments
from scan2plan.io.dxf_face_export import export_dxf_faces
from scan2plan.io.readers import read_point_cloud
from scan2plan.preprocessing.downsampling import voxel_downsample
from scan2plan.preprocessing.floor_ceiling import (
    detect_ceiling,
    detect_floor,
    detect_floor_rdc,
    filter_vertical_range,
)
from scan2plan.preprocessing.outlier_removal import remove_statistical_outliers
from scan2plan.slicing.density_map import DensityMapResult, create_density_map
from scan2plan.slicing.slicer import extract_multi_slices
from scan2plan.utils.coordinate import metric_to_pixel
from scan2plan.vectorization.angular_regularization import (
    detect_dominant_orientations,
    snap_angles,
)
from scan2plan.vectorization.light_topology import (
    apply_light_topology,
    close_corners,
    snap_endpoints,
)
from scan2plan.vectorization.wall_pairing import (
    FacePair,
    PairingConfig,
    Segment,
    pair_wall_faces,
)

# ---------------------------------------------------------------------------
# Constantes visuelles
# ---------------------------------------------------------------------------

_DPI = 150
_LINEWIDTH = 1.5
_FACECOLOR = "black"
_PAIR_COLORS = [
    "#FF4444", "#44FF44", "#4488FF", "#FFFF44",
    "#FF44FF", "#44FFFF", "#FF8800", "#AA44FF",
]
_GRAY_UNPAIRED = "#666666"


# ---------------------------------------------------------------------------
# Conversion métrique → pixel pour un segment
# ---------------------------------------------------------------------------

def _seg_to_px(
    seg: DetectedSegment | Segment,
    dmap: DensityMapResult,
) -> tuple[int, int, int, int]:
    """Retourne (col1, row1, col2, row2) en pixels."""
    c1, r1 = metric_to_pixel(
        seg.x1, seg.y1,
        dmap.x_min, dmap.y_min, dmap.resolution, dmap.height,
    )
    c2, r2 = metric_to_pixel(
        seg.x2, seg.y2,
        dmap.x_min, dmap.y_min, dmap.resolution, dmap.height,
    )
    return c1, r1, c2, r2


# ---------------------------------------------------------------------------
# Dessin de base
# ---------------------------------------------------------------------------

def _ax_background(ax: plt.Axes, dmap: DensityMapResult) -> None:
    """Affiche la density map en niveaux de gris sur un axe."""
    img = np.log1p(dmap.image.astype(float))
    img = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img
    ax.imshow(img, cmap="gray", origin="upper",
              extent=(0, dmap.width, dmap.height, 0))
    ax.set_facecolor(_FACECOLOR)
    ax.set_xlim(0, dmap.width)
    ax.set_ylim(dmap.height, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_segs(
    ax: plt.Axes,
    segs: list,
    dmap: DensityMapResult,
    color: str | list[str] = "white",
    linewidth: float = _LINEWIDTH,
    linestyle: str = "-",
    alpha: float = 1.0,
) -> None:
    """Trace des segments sur un axe matplotlib (coordonnées pixel)."""
    for i, seg in enumerate(segs):
        c1, r1, c2, r2 = _seg_to_px(seg, dmap)
        col = color[i] if isinstance(color, list) else color
        ax.plot([c1, c2], [r1, r2], color=col,
                linewidth=linewidth, linestyle=linestyle, alpha=alpha)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Sauvegarde la figure et la ferme."""
    fig.savefig(str(path), dpi=_DPI, bbox_inches="tight", facecolor=_FACECOLOR)
    plt.close(fig)
    print(f"    -> {path.name}")


# ---------------------------------------------------------------------------
# plot_segments_on_density  (fonction centrale)
# ---------------------------------------------------------------------------

def plot_segments_on_density(
    dmap: DensityMapResult,
    segments: list,
    title: str,
    output_path: Path,
    colors: list[str] | str = "white",
    highlight_segments: list | None = None,
    highlight_color: str = "red",
    markers: list[tuple[float, float]] | None = None,
    marker_style: str = "o",
    marker_color: str = "yellow",
    figsize: tuple[int, int] = (12, 12),
) -> None:
    """Dessine des segments sur la density map et sauvegarde en PNG.

    Args:
        dmap: Density map de référence (fond + conversion métrique/pixel).
        segments: Segments principaux à afficher.
        title: Titre de l'image (inclure le compteur de segments).
        output_path: Chemin de sortie PNG.
        colors: Couleur(s) des segments.
        highlight_segments: Segments secondaires (dessinés en pointillé).
        highlight_color: Couleur des segments secondaires.
        markers: Points à marquer (coordonnées métriques).
        marker_style: Style de marqueur ("o", "x", "+").
        marker_color: Couleur des marqueurs.
        figsize: Taille de la figure en pouces.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=_FACECOLOR)
    _ax_background(ax, dmap)
    _draw_segs(ax, segments, dmap, color=colors)

    if highlight_segments:
        _draw_segs(ax, highlight_segments, dmap,
                   color=highlight_color, linestyle="--", alpha=0.7)

    if markers:
        for mx, my in markers:
            c, r = metric_to_pixel(mx, my, dmap.x_min, dmap.y_min,
                                   dmap.resolution, dmap.height)
            ax.plot(c, r, marker_style, color=marker_color,
                    markersize=6, markeredgewidth=1.5)

    ax.set_title(title, color="white", fontsize=13, pad=6)
    _save_fig(fig, output_path)


def plot_comparison(
    dmap: DensityMapResult,
    segs_before: list,
    segs_after: list,
    title_before: str,
    title_after: str,
    output_path: Path,
    color_before: str = "#00FF00",
    color_after: str = "white",
) -> None:
    """Figure 1×2 : avant (gauche) / après (droite) sur la même density map.

    Args:
        dmap: Density map de fond.
        segs_before: Segments de la colonne gauche.
        segs_after: Segments de la colonne droite.
        title_before: Titre colonne gauche.
        title_after: Titre colonne droite.
        output_path: Chemin de sortie.
        color_before: Couleur colonne gauche.
        color_after: Couleur colonne droite.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), facecolor=_FACECOLOR)
    for ax, segs, title, color in (
        (ax1, segs_before, title_before, color_before),
        (ax2, segs_after, title_after, color_after),
    ):
        _ax_background(ax, dmap)
        _draw_segs(ax, segs, dmap, color=color)
        ax.set_title(title, color="white", fontsize=13, pad=6)
    plt.tight_layout()
    _save_fig(fig, output_path)


def plot_recap_grid(
    image_paths: list[Path],
    titles: list[str],
    output_path: Path,
) -> None:
    """Grille 2×5 avec les miniatures de toutes les étapes.

    Args:
        image_paths: Chemins des PNG (jusqu'à 10 images).
        titles: Titres des sous-figures.
        output_path: Chemin de sortie.
    """
    n = min(len(image_paths), 10)
    fig, axes = plt.subplots(2, 5, figsize=(30, 12), facecolor=_FACECOLOR)
    for i, ax in enumerate(axes.flat):
        ax.set_facecolor(_FACECOLOR)
        if i < n and image_paths[i].exists():
            try:
                img = plt.imread(str(image_paths[i]))
                ax.imshow(img)
            except Exception:
                pass
        title = titles[i] if i < len(titles) else ""
        ax.set_title(title, color="white", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Couleurs pour le pairing
# ---------------------------------------------------------------------------

def generate_pair_colors(
    segments: list,
    face_pairs: list[FacePair],
) -> list[str]:
    """Une couleur par paire (alternée), gris pour les non pairés.

    Args:
        segments: Tous les segments (même ordre que la liste à colorier).
        face_pairs: Paires de faces.

    Returns:
        Liste de couleurs hex, une par segment.
    """
    pair_map: dict[int, int] = {}  # id(seg) → index paire
    for pi, fp in enumerate(face_pairs):
        pair_map[id(fp.face_a)] = pi
        pair_map[id(fp.face_b)] = pi

    colors = []
    for seg in segments:
        pi = pair_map.get(id(seg))
        if pi is not None:
            colors.append(_PAIR_COLORS[pi % len(_PAIR_COLORS)])
        else:
            colors.append(_GRAY_UNPAIRED)
    return colors


def generate_thickness_colors(
    segments: list,
    face_pairs: list[FacePair],
) -> list[str]:
    """Couleur par épaisseur (bleu=5cm → vert=15cm → rouge=25cm), gris=non pairé.

    Args:
        segments: Tous les segments.
        face_pairs: Paires de faces avec épaisseur.

    Returns:
        Liste de couleurs hex.
    """
    thick_map: dict[int, float] = {}
    for fp in face_pairs:
        thick_map[id(fp.face_a)] = fp.thickness
        thick_map[id(fp.face_b)] = fp.thickness

    cmap = cm.get_cmap("RdYlGn_r")
    colors = []
    for seg in segments:
        t = thick_map.get(id(seg))
        if t is not None:
            # Normaliser 5cm→25cm sur [0,1]
            norm_t = float(np.clip((t - 0.05) / 0.20, 0.0, 1.0))
            rgba = cmap(norm_t)
            colors.append(mcolors.to_hex(rgba))
        else:
            colors.append(_GRAY_UNPAIRED)
    return colors


# ---------------------------------------------------------------------------
# Étape 0 — Prétraitement
# ---------------------------------------------------------------------------

def step0_preprocessing(
    input_path: Path,
    cfg: ScanConfig,
    output_dir: Path,
) -> tuple[np.ndarray, float, float]:
    """Lit, downsample, SOR, détecte sol/plafond, filtre verticalement.

    Returns:
        ``(points_filtered, z_floor, z_ceiling)``
    """
    print("\n[Étape 0] Chargement et prétraitement...")
    t0 = time.monotonic()

    # Lecture
    points_raw = read_point_cloud(input_path)
    print(f"  Points bruts          : {len(points_raw):>12,}")

    # Downsampling
    cfg_pre = cfg.preprocessing
    points = voxel_downsample(points_raw, cfg_pre.voxel_size)
    print(f"  Après downsampling    : {len(points):>12,}  (voxel={cfg_pre.voxel_size*100:.0f}mm)")

    # SOR
    points = remove_statistical_outliers(points, cfg_pre.sor_k_neighbors, cfg_pre.sor_std_ratio)
    print(f"  Après SOR             : {len(points):>12,}")

    # Sol / plafond
    z_floor_hint, z_ceiling_hint = detect_floor_rdc(points)
    cfg_fc = cfg.floor_ceiling
    margin = 0.30

    mask_f = (points[:, 2] >= z_floor_hint - margin) & (points[:, 2] <= z_floor_hint + margin)
    pts_f = points[mask_f]
    z_floor = detect_floor(
        pts_f,
        distance_threshold=cfg_fc.ransac_distance,
        num_iterations=cfg_fc.ransac_iterations,
        normal_tolerance_deg=cfg_fc.normal_tolerance_deg,
    )[0] if len(pts_f) >= 3 else z_floor_hint

    mask_c = (points[:, 2] >= z_ceiling_hint - margin) & (points[:, 2] <= z_ceiling_hint + margin)
    pts_c = points[mask_c]
    z_ceiling = detect_ceiling(
        pts_c,
        z_floor,
        distance_threshold=cfg_fc.ransac_distance,
        num_iterations=cfg_fc.ransac_iterations,
        normal_tolerance_deg=cfg_fc.normal_tolerance_deg,
    )[0] if len(pts_c) >= 3 else z_ceiling_hint

    print(f"  Altitude sol          :  {z_floor:+.3f} m")
    print(f"  Altitude plafond      :  {z_ceiling:+.3f} m")
    print(f"  Hauteur sous plafond  :  {z_ceiling - z_floor:.2f} m")

    # Filtrage vertical
    points = filter_vertical_range(points, z_floor, z_ceiling)
    print(f"  Après filtrage vert.  : {len(points):>12,}")

    # Image histogramme Z
    _plot_z_histogram(points_raw, z_floor, z_ceiling, output_dir / "00_nuage_stats.png")
    print(f"  Temps : {time.monotonic()-t0:.1f} s")
    return points, z_floor, z_ceiling


def _plot_z_histogram(
    points: np.ndarray,
    z_floor: float,
    z_ceiling: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_FACECOLOR)
    ax.set_facecolor(_FACECOLOR)
    z = points[:, 2]
    ax.hist(z, bins=200, color="#4488FF", alpha=0.8)
    ax.axvline(z_floor, color="#FF4444", linewidth=2, label=f"Sol : {z_floor:.3f} m")
    ax.axvline(z_ceiling, color="#44FF44", linewidth=2, label=f"Plafond : {z_ceiling:.3f} m")
    ax.set_xlabel("Z (m)", color="white")
    ax.set_ylabel("Nb points", color="white")
    ax.set_title(f"Distribution Z  ({len(points):,} points)", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#222222", labelcolor="white", fontsize=10)
    ax.spines[:].set_color("#555555")
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Étape 1 — Slices et density maps
# ---------------------------------------------------------------------------

def step1_slicing(
    points: np.ndarray,
    z_floor: float,
    cfg: ScanConfig,
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, DensityMapResult], dict[str, np.ndarray]]:
    """Extrait slices, density maps et images binaires.

    Returns:
        ``(slices_xy, density_maps, binary_maps)``
    """
    print("\n[Étape 1] Slices et density maps...")
    t0 = time.monotonic()

    cfg_sl = cfg.slicing
    slices_xy = extract_multi_slices(points, heights=cfg_sl.heights,
                                     thickness=cfg_sl.thickness, floor_z=z_floor)
    density_maps: dict[str, DensityMapResult] = {}
    binary_maps: dict[str, np.ndarray] = {}
    cfg_morph = cfg.morphology

    for label, pts in slices_xy.items():
        print(f"  Slice {label:4s} : {len(pts):>8,} points")
        if len(pts) == 0:
            continue
        dmap = create_density_map(pts, cfg.density_map.resolution)
        density_maps[label] = dmap
        binary_raw = binarize_density_map(dmap.image)
        binary = morphological_cleanup(binary_raw, cfg_morph.kernel_size,
                                       cfg_morph.close_iterations, cfg_morph.open_iterations)
        binary_maps[label] = binary

    _plot_density_grid(density_maps, output_dir / "01_density_maps.png", "Density maps")
    _plot_binary_grid(binary_maps, output_dir / "01_binary_maps.png", "Images binaires")
    print(f"  Temps : {time.monotonic()-t0:.1f} s")
    return slices_xy, density_maps, binary_maps


def _plot_density_grid(
    dmaps: dict[str, DensityMapResult],
    output_path: Path,
    title: str,
) -> None:
    labels = list(dmaps.keys())
    n = len(labels)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8), facecolor=_FACECOLOR)
    if n == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        dmap = dmaps[label]
        img = np.log1p(dmap.image.astype(float))
        img = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img
        ax.imshow(img, cmap="hot", origin="upper")
        ax.set_title(f"Slice {label}  {dmap.height}×{dmap.width}px",
                     color="white", fontsize=11)
        ax.axis("off")
        ax.set_facecolor(_FACECOLOR)
    fig.suptitle(title, color="white", fontsize=14)
    plt.tight_layout()
    _save_fig(fig, output_path)


def _plot_binary_grid(
    binary_maps: dict[str, np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    labels = list(binary_maps.keys())
    n = len(labels)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8), facecolor=_FACECOLOR)
    if n == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        ax.imshow(binary_maps[label], cmap="gray", origin="upper")
        ax.set_title(f"Binaire {label}", color="white", fontsize=11)
        ax.axis("off")
        ax.set_facecolor(_FACECOLOR)
    fig.suptitle(title, color="white", fontsize=14)
    plt.tight_layout()
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Étape 2 — Hough brut
# ---------------------------------------------------------------------------

def step2_hough(
    density_maps: dict[str, DensityMapResult],
    binary_maps: dict[str, np.ndarray],
    cfg: ScanConfig,
    output_dir: Path,
) -> tuple[dict[str, list[DetectedSegment]], list[DetectedSegment]]:
    """Détection Hough sur chaque slice.

    Returns:
        ``(segments_by_slice, all_segments)``
    """
    print("\n[Étape 2] Hough brut...")
    t0 = time.monotonic()
    cfg_h = cfg.hough
    segs_by_slice: dict[str, list[DetectedSegment]] = {}
    all_segs: list[DetectedSegment] = []

    for label in density_maps:
        binary = binary_maps.get(label)
        dmap = density_maps[label]
        if binary is None:
            segs_by_slice[label] = []
            continue
        segs = detect_lines_hough(binary, dmap,
                                  rho=cfg_h.rho, theta_deg=cfg_h.theta_deg,
                                  threshold=cfg_h.threshold,
                                  min_line_length=cfg_h.min_line_length,
                                  max_line_gap=cfg_h.max_line_gap,
                                  source_slice=label)
        segs_by_slice[label] = segs
        all_segs.extend(segs)
        print(f"  Slice {label:4s} : {len(segs):>5} segments")

    print(f"  TOTAL Hough brut : {len(all_segs)}")

    # Image sur slice high (ou mid si high absent)
    ref_label = "high" if "high" in density_maps else next(iter(density_maps))
    dmap_ref = density_maps[ref_label]
    segs_ref = segs_by_slice.get(ref_label, [])
    plot_segments_on_density(
        dmap_ref, segs_ref,
        f"02 — Hough brut  [{ref_label}]  |  {len(segs_ref)} segments",
        output_dir / "02_hough_brut.png",
        colors="#00FF00",
    )
    print(f"  Temps : {time.monotonic()-t0:.1f} s")
    return segs_by_slice, all_segs


# ---------------------------------------------------------------------------
# Étape 3 — Multi-slice filter
# ---------------------------------------------------------------------------

def step3_multifilter(
    segs_by_slice: dict[str, list[DetectedSegment]],
    density_maps: dict[str, DensityMapResult],
    cfg: ScanConfig,
    output_dir: Path,
) -> list[DetectedSegment]:
    """Filtre multi-slice : garde les murs, élimine le mobilier.

    Returns:
        Segments de murs confirmés.
    """
    print("\n[Étape 3] Multi-slice filter...")
    t0 = time.monotonic()
    cfg_msf = cfg._data.get("multi_slice_filter", {})
    angle_tol = float(cfg_msf.get("angle_tolerance_deg", 5.0))
    dist_tol = float(cfg_msf.get("distance_tolerance", 0.10))

    all_before = sum(len(v) for v in segs_by_slice.values())
    matches = match_segments_across_slices(segs_by_slice,
                                          angle_tolerance_deg=angle_tol,
                                          distance_tolerance=dist_tol)
    wall_segs = classify_segments(matches)

    # Mobilier = segments high non confirmés
    high_segs = segs_by_slice.get("high", [])
    wall_set = {id(s) for s in wall_segs}
    furniture_segs = [s for s in high_segs if id(s) not in wall_set]

    print(f"  Segments avant       : {all_before}")
    print(f"  Murs confirmés       : {len(wall_segs)}")
    print(f"  Mobilier supprimé    : {len(furniture_segs)}")

    dmap_ref = density_maps.get("mid") or density_maps.get("high") or next(iter(density_maps.values()))
    fig, ax = plt.subplots(figsize=(12, 12), facecolor=_FACECOLOR)
    _ax_background(ax, dmap_ref)
    _draw_segs(ax, wall_segs, dmap_ref, color="#FFFF00")
    _draw_segs(ax, furniture_segs, dmap_ref, color="#FF0000", linestyle="--", alpha=0.6)

    yellow_patch = mpatches.Patch(color="#FFFF00", label=f"Murs ({len(wall_segs)})")
    red_patch = mpatches.Patch(color="#FF0000", label=f"Mobilier ({len(furniture_segs)})")
    ax.legend(handles=[yellow_patch, red_patch], loc="upper right",
              facecolor="#222222", labelcolor="white", fontsize=10)
    ax.set_title(f"03 — Multi-slice filter  |  {len(wall_segs)} murs  {len(furniture_segs)} mobilier",
                 color="white", fontsize=13, pad=6)
    _save_fig(fig, output_dir / "03_multi_slice.png")
    print(f"  Temps : {time.monotonic()-t0:.1f} s")
    return wall_segs


# ---------------------------------------------------------------------------
# Étape 4 — Micro-fusion
# ---------------------------------------------------------------------------

def step4_microfusion(
    wall_segs: list[DetectedSegment],
    dmap_ref: DensityMapResult,
    cfg: ScanConfig,
    output_dir: Path,
) -> list[DetectedSegment]:
    """Fusion avec gap ≤ 5 cm.

    Returns:
        Segments après micro-fusion.
    """
    print("\n[Étape 4] Micro-fusion (gap ≤ 5 cm)...")
    t0 = time.monotonic()
    cfg_sf = cfg.segment_fusion
    fused = fuse_collinear_segments(wall_segs, cfg_sf.angle_tolerance_deg,
                                    cfg_sf.perpendicular_dist, max_gap=0.05)
    n_fused = len(wall_segs) - len(fused)
    print(f"  Avant  : {len(wall_segs)}")
    print(f"  Après  : {len(fused)}  (-{n_fused} fusions)")

    plot_segments_on_density(
        dmap_ref, fused,
        f"04 — Micro-fusion  |  {len(wall_segs)} → {len(fused)} segments",
        output_dir / "04_micro_fusion.png",
        colors="#00FFFF",
    )
    print(f"  Temps : {time.monotonic()-t0:.1f} s")
    return fused


# ---------------------------------------------------------------------------
# Étape 5 — Nettoyage parasites
# ---------------------------------------------------------------------------

def step5_cleanup(
    segs: list[DetectedSegment],
    dmap_ref: DensityMapResult,
    cfg: ScanConfig,
    output_dir: Path,
) -> list[DetectedSegment]:
    """Supprime les segments courts (< min_segment_length).

    Returns:
        Segments nettoyés.
    """
    print("\n[Étape 5] Nettoyage parasites...")
    t0 = time.monotonic()
    min_len = float(cfg._data.get("topology", {}).get("min_segment_length", 0.10))
    kept = [s for s in segs if s.length >= min_len]
    removed = [s for s in segs if s.length < min_len]

    avg_len_removed = (
        float(np.mean([s.length for s in removed])) if removed else 0.0
    )
    print(f"  Avant  : {len(segs)}")
    print(f"  Après  : {len(kept)}  (-{len(removed)} parasites, "
          f"longueur moy. supprimés : {avg_len_removed*100:.1f} cm)")

    fig, ax = plt.subplots(figsize=(12, 12), facecolor=_FACECOLOR)
    _ax_background(ax, dmap_ref)
    _draw_segs(ax, kept, dmap_ref, color="white")
    _draw_segs(ax, removed, dmap_ref, color="red", linestyle="--", alpha=0.7)

    w_patch = mpatches.Patch(color="white", label=f"Gardés ({len(kept)})")
    r_patch = mpatches.Patch(color="red", label=f"Supprimés ({len(removed)})")
    ax.legend(handles=[w_patch, r_patch], loc="upper right",
              facecolor="#222222", labelcolor="white", fontsize=10)
    ax.set_title(f"05 — Cleanup  |  {len(segs)} → {len(kept)} segments",
                 color="white", fontsize=13, pad=6)
    _save_fig(fig, output_dir / "05_cleanup.png")
    print(f"  Temps : {time.monotonic()-t0:.1f} s")
    return kept


# ---------------------------------------------------------------------------
# Étape 6 — Régularisation angulaire
# ---------------------------------------------------------------------------

def step6_regularization(
    segs: list[DetectedSegment],
    dmap_ref: DensityMapResult,
    cfg: ScanConfig,
    output_dir: Path,
) -> tuple[list[DetectedSegment], list[float]]:
    """Snap angulaire sur les directions dominantes.

    Returns:
        ``(regularized_segs, dominant_angles_rad)``
    """
    print("\n[Étape 6] Régularisation angulaire...")
    t0 = time.monotonic()
    snap_tol = float(cfg._data.get("regularization", {}).get("snap_tolerance_deg", 5.0))
    dominant_angles = detect_dominant_orientations(segs)
    regularized = snap_angles(segs, dominant_angles, tolerance_deg=snap_tol)

    angles_deg = [float(np.degrees(a)) % 180 for a in dominant_angles]
    print(f"  Orientations dominantes : {[f'{a:.1f}°' for a in angles_deg]}")
    snapped_count = sum(
        1 for s, r in zip(segs, regularized)
        if abs(s.x1 - r.x1) > 1e-4 or abs(s.y1 - r.y1) > 1e-4
    )
    print(f"  Segments snappés : {snapped_count} / {len(segs)}")

    # Image segments régularisés + directions dominantes
    fig, ax = plt.subplots(figsize=(12, 12), facecolor=_FACECOLOR)
    _ax_background(ax, dmap_ref)
    _draw_segs(ax, regularized, dmap_ref, color="#00FF00")

    # Tracer les directions dominantes comme lignes pointillées
    cx = dmap_ref.width / 2.0
    cy = dmap_ref.height / 2.0
    diag = float(np.hypot(dmap_ref.width, dmap_ref.height))
    for angle_rad in dominant_angles:
        ux = float(np.cos(angle_rad)) * diag / 2
        uy = -float(np.sin(angle_rad)) * diag / 2  # inversion Y image
        ax.plot([cx - ux, cx + ux], [cy - uy, cy + uy],
                color="#FF8800", linewidth=1, linestyle="--", alpha=0.6)

    ax.set_title(
        f"06 — Régularisation  |  {len(regularized)} seg  |  "
        f"directions : {[f'{a:.1f}°' for a in angles_deg]}",
        color="white", fontsize=11, pad=6,
    )
    _save_fig(fig, output_dir / "06_regularization.png")

    # Histogramme angulaire
    _plot_angle_histogram(segs, dominant_angles, output_dir / "06_angle_histogram.png")
    print(f"  Temps : {time.monotonic()-t0:.1f} s")
    return regularized, dominant_angles


def _plot_angle_histogram(
    segs: list[DetectedSegment],
    dominant_angles: list[float],
    output_path: Path,
) -> None:
    angles = [float(np.arctan2(s.y2 - s.y1, s.x2 - s.x1) % np.pi) for s in segs]
    weights = [s.length for s in segs]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_FACECOLOR)
    ax.set_facecolor(_FACECOLOR)
    bins = np.linspace(0, np.pi, 181)
    ax.hist(np.degrees(angles), bins=np.degrees(bins),
            weights=weights, color="#4488FF", alpha=0.8)
    for a in dominant_angles:
        a_deg = float(np.degrees(a)) % 180
        ax.axvline(a_deg, color="#FF4444", linewidth=2, label=f"{a_deg:.1f}°")
    ax.set_xlabel("Angle (°)", color="white")
    ax.set_ylabel("Longueur pondérée (m)", color="white")
    ax.set_title("Histogramme angulaire (pondéré par longueur)", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#222222", labelcolor="white", fontsize=10)
    ax.spines[:].set_color("#555555")
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Étape 7 — Face pairing
# ---------------------------------------------------------------------------

def step7_pairing(
    segs: list[DetectedSegment],
    dmap_ref: DensityMapResult,
    cfg: ScanConfig,
    output_dir: Path,
) -> tuple[list[Segment], list[FacePair], list[Segment]]:
    """Apparie les faces de murs sans calculer de médiane.

    Returns:
        ``(pairing_segs, face_pairs, unpaired_segs)``
    """
    print("\n[Étape 7] Face pairing...")
    t0 = time.monotonic()

    pairing_segs = [
        Segment(x1=s.x1, y1=s.y1, x2=s.x2, y2=s.y2,
                label="wall", confidence=s.confidence, source_slice=s.source_slice)
        for s in segs
    ]
    cfg_wp = cfg.wall_pairing
    pairing_config = PairingConfig(
        angle_tolerance_deg=cfg_wp.angle_tolerance_deg,
        min_distance=cfg_wp.min_distance,
        max_distance=cfg_wp.max_distance,
        min_overlap_abs=cfg_wp.min_overlap_abs,
        min_overlap_ratio=cfg_wp.min_overlap_ratio,
        corridor_margin=cfg_wp.corridor_margin,
        typical_wall_thickness=cfg_wp.typical_wall_thickness,
        min_segment_length=cfg_wp.min_segment_length,
        corridor_intersection_threshold=cfg_wp.corridor_intersection_threshold,
    )
    face_result = pair_wall_faces(pairing_segs, pairing_config)
    face_pairs = face_result.paired_faces
    unpaired = face_result.unpaired_segments

    thicknesses = [fp.thickness for fp in face_pairs]
    if thicknesses:
        print(f"  Paires détectées   : {len(face_pairs)}")
        print(f"  Épaisseur min/max  : {min(thicknesses)*100:.0f} / {max(thicknesses)*100:.0f} cm")
        print(f"  Épaisseur moyenne  : {np.mean(thicknesses)*100:.0f} cm")
    print(f"  Non pairés         : {len(unpaired)}")

    all_wp_segs = [fp.face_a for fp in face_pairs] + \
                  [fp.face_b for fp in face_pairs] + unpaired

    colors_pair = generate_pair_colors(all_wp_segs, face_pairs)
    plot_segments_on_density(
        dmap_ref, all_wp_segs,
        f"07 — Pairing  |  {len(face_pairs)} paires  {len(unpaired)} simples",
        output_dir / "07_pairing.png",
        colors=colors_pair,
    )

    colors_thick = generate_thickness_colors(all_wp_segs, face_pairs)
    plot_segments_on_density(
        dmap_ref, all_wp_segs,
        f"07 — Pairing par épaisseur  |  {len(face_pairs)} paires",
        output_dir / "07_pairing_thickness.png",
        colors=colors_thick,
    )
    print(f"  Temps : {time.monotonic()-t0:.1f} s")
    return pairing_segs, face_pairs, unpaired


# ---------------------------------------------------------------------------
# Étapes 8 & 9 — Snap + Close corners (avec marqueurs)
# ---------------------------------------------------------------------------

def step8_snap(
    segs: list[DetectedSegment],
    dmap_ref: DensityMapResult,
    output_dir: Path,
    snap_tolerance: float = 0.03,
) -> tuple[list[DetectedSegment], list[tuple[float, float]]]:
    """Snap des extrémités proches.

    Returns:
        ``(snapped_segs, snap_points_metric)``
    """
    print("\n[Étape 8] Snap extrémités (tol=3 cm)...")
    t0 = time.monotonic()

    # Capturer les centroïdes créés par le snap
    from scan2plan.vectorization.light_topology import snap_endpoints

    before_coords = {(round(s.x1, 4), round(s.y1, 4)): None for s in segs}
    before_coords.update({(round(s.x2, 4), round(s.y2, 4)): None for s in segs})

    snapped = snap_endpoints(segs, tolerance=snap_tolerance)

    # Points qui ont bougé → marqueurs
    snap_pts: list[tuple[float, float]] = []
    for s_after in snapped:
        for pt in ((s_after.x1, s_after.y1), (s_after.x2, s_after.y2)):
            key = (round(pt[0], 4), round(pt[1], 4))
            if key not in before_coords:
                snap_pts.append(pt)

    # Dédupliquer les marqueurs proches
    snap_pts_dedup: list[tuple[float, float]] = []
    for pt in snap_pts:
        if not any(abs(pt[0] - p[0]) < 0.005 and abs(pt[1] - p[1]) < 0.005
                   for p in snap_pts_dedup):
            snap_pts_dedup.append(pt)

    print(f"  Segments avant/après : {len(segs)} / {len(snapped)}")
    print(f"  Points de snap       : {len(snap_pts_dedup)}")

    plot_segments_on_density(
        dmap_ref, snapped,
        f"08 — Snap ({snap_tolerance*100:.0f} cm)  |  {len(snap_pts_dedup)} points snappés",
        output_dir / "08_snap.png",
        colors="white",
        markers=snap_pts_dedup,
        marker_style="o",
        marker_color="yellow",
    )
    print(f"  Temps : {time.monotonic()-t0:.1f} s")
    return snapped, snap_pts_dedup


def step9_corners(
    segs: list[DetectedSegment],
    dmap_ref: DensityMapResult,
    output_dir: Path,
    max_extension: float = 0.08,
) -> tuple[list[DetectedSegment], list[tuple[float, float]]]:
    """Fermeture de coins.

    Returns:
        ``(closed_segs, corner_points_metric)``
    """
    print("\n[Étape 9] Fermeture de coins (max=8 cm)...")
    t0 = time.monotonic()

    before_set = {
        (round(s.x1, 4), round(s.y1, 4), round(s.x2, 4), round(s.y2, 4))
        for s in segs
    }
    closed = close_corners(segs, max_extension=max_extension)

    # Nouveaux points d'extrémité → coins fermés
    corner_pts: list[tuple[float, float]] = []
    for s in closed:
        key = (round(s.x1, 4), round(s.y1, 4), round(s.x2, 4), round(s.y2, 4))
        if key not in before_set:
            for pt in ((s.x1, s.y1), (s.x2, s.y2)):
                corner_pts.append(pt)

    corner_pts_dedup: list[tuple[float, float]] = []
    for pt in corner_pts:
        if not any(abs(pt[0] - p[0]) < 0.01 and abs(pt[1] - p[1]) < 0.01
                   for p in corner_pts_dedup):
            corner_pts_dedup.append(pt)

    print(f"  Segments avant/après : {len(segs)} / {len(closed)}")
    print(f"  Coins fermés         : {len(corner_pts_dedup)}")

    plot_segments_on_density(
        dmap_ref, closed,
        f"09 — Coins fermés  |  {len(corner_pts_dedup)} intersections créées",
        output_dir / "09_corners.png",
        colors="white",
        markers=corner_pts_dedup,
        marker_style="x",
        marker_color="red",
    )
    print(f"  Temps : {time.monotonic()-t0:.1f} s")
    return closed, corner_pts_dedup


# ---------------------------------------------------------------------------
# Étape 10 — Résultat final
# ---------------------------------------------------------------------------

def step10_final(
    final_segs: list[DetectedSegment],
    face_pairs: list[FacePair],
    hough_segs: list[DetectedSegment],
    dmap_ref: DensityMapResult,
    output_dir: Path,
    output_dxf: Path,
) -> None:
    """Export DXF et images finales.

    Args:
        final_segs: Segments finaux après topologie.
        face_pairs: Paires de faces.
        hough_segs: Segments Hough bruts (pour la comparaison).
        dmap_ref: Density map de référence.
        output_dir: Répertoire de sortie.
        output_dxf: Chemin du DXF de sortie.
    """
    print("\n[Étape 10] Export final...")
    t0 = time.monotonic()

    # Export DXF
    export_dxf_faces(final_segs, face_pairs, output_dxf)

    # Couleurs par calque DXF
    _PARTITION_THRESHOLD = 0.12
    paired_ids: set[int] = set()
    paired_thick: dict[int, float] = {}
    for fp in face_pairs:
        for face in (fp.face_a, fp.face_b):
            paired_ids.add(id(face))
            paired_thick[id(face)] = fp.thickness

    colors_final = []
    for seg in final_segs:
        t = paired_thick.get(id(seg))
        if t is None:
            colors_final.append("#00FFFF")     # MURS_SIMPLE
        elif t > _PARTITION_THRESHOLD:
            colors_final.append("white")       # MURS_PORTEURS
        else:
            colors_final.append("#00FF88")     # CLOISONS

    plot_segments_on_density(
        dmap_ref, final_segs,
        f"10 — Final  |  {len(final_segs)} segments  {len(face_pairs)} paires",
        output_dir / "10_final.png",
        colors=colors_final,
    )

    plot_comparison(
        dmap_ref,
        hough_segs[:len([s for s in hough_segs if s.source_slice == "high"])
                   or len(hough_segs)],
        final_segs,
        f"Hough brut  ({len(hough_segs)} seg)",
        f"Final  ({len(final_segs)} seg)",
        output_dir / "10_final_vs_hough.png",
        color_before="#00FF00",
        color_after="white",
    )
    print(f"  DXF exporté : {output_dxf.name}")
    print(f"  Temps : {time.monotonic()-t0:.1f} s")


# ---------------------------------------------------------------------------
# Étape 11 — Récapitulatif
# ---------------------------------------------------------------------------

def step11_recap(
    output_dir: Path,
    stats: dict,
    all_images: list[Path],
    titles: list[str],
) -> None:
    """Génère 11_recap.png et recap.txt.

    Args:
        output_dir: Répertoire de sortie.
        stats: Dictionnaire des statistiques collectées.
        all_images: Chemins des images des étapes 0-10.
        titles: Titres courts pour la grille.
    """
    print("\n[Étape 11] Récapitulatif...")
    plot_recap_grid(all_images, titles, output_dir / "11_recap.png")
    _write_recap_txt(output_dir / "recap.txt", stats)


def _write_recap_txt(output_path: Path, stats: dict) -> None:
    """Écrit le fichier recap.txt."""
    sep = "=" * 62
    thin = "─" * 62
    lines = [
        sep,
        "SCAN2PLAN — Rapport de traitement",
        f"Fichier  : {stats.get('input_name', '?')}",
        f"Date     : {datetime.date.today().isoformat()}",
        sep,
        "",
        "PRÉTRAITEMENT",
        f"  Points bruts            : {stats.get('n_raw', 0):>12,}",
        f"  Après downsampling      : {stats.get('n_downsampled', 0):>12,}",
        f"  Après SOR               : {stats.get('n_sor', 0):>12,}",
        f"  Altitude sol            : {stats.get('z_floor', 0):>+10.3f} m",
        f"  Altitude plafond        : {stats.get('z_ceiling', 0):>+10.3f} m",
        f"  Hauteur sous plafond    : {stats.get('z_ceiling', 0) - stats.get('z_floor', 0):>10.2f} m",
        "",
        "SLICING",
    ]
    for label, n in stats.get("slice_counts", {}).items():
        h = stats.get("slice_heights", {}).get(label, 0.0)
        lines.append(f"  Slice {label:4s} ({h:.2f}m)         : {n:>12,} points")
    lines += [
        "",
        "PIPELINE SEGMENTS",
        f"  {'Étape':<28} {'Segments':>8}    {'Variation':>10}",
        f"  {thin}",
    ]

    seg_steps = stats.get("seg_steps", [])
    prev = None
    for name, n in seg_steps:
        if prev is None:
            variation = "—"
        else:
            diff = n - prev
            variation = f"{diff:+d}" if diff != 0 else "±0"
        lines.append(f"  {name:<28} {n:>8}    {variation:>10}")
        prev = n

    lines += [
        f"  {thin}",
        f"  {'FINAL':<28} {seg_steps[-1][1] if seg_steps else '?':>8}",
        "",
        "PAIRING",
        f"  Paires de faces         : {stats.get('n_pairs', 0):>6}",
    ]
    thicknesses = stats.get("thicknesses", [])
    if thicknesses:
        lines += [
            f"  Épaisseur min           : {min(thicknesses):>5.2f} m ({min(thicknesses)*100:.0f} cm)",
            f"  Épaisseur max           : {max(thicknesses):>5.2f} m ({max(thicknesses)*100:.0f} cm)",
            f"  Épaisseur moyenne       : {np.mean(thicknesses):>5.2f} m ({np.mean(thicknesses)*100:.0f} cm)",
        ]
    lines += [
        f"  Segments non pairés     : {stats.get('n_unpaired', 0):>6}",
        "",
        "CALQUES DXF",
        f"  MURS_PORTEURS           : {stats.get('n_porteurs', 0):>6} segments",
        f"  CLOISONS                : {stats.get('n_cloisons', 0):>6} segments",
        f"  MURS_SIMPLE             : {stats.get('n_unpaired', 0):>6} segments",
        "",
        sep,
        f"Temps total : {stats.get('elapsed', 0):.1f} secondes",
        f"DXF exporté : {stats.get('dxf_name', '?')}",
        sep,
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"    -> {output_path.name}")


# ---------------------------------------------------------------------------
# Orchestrateur principal
# ---------------------------------------------------------------------------

def run_debug_pipeline(input_path: Path, output_dir: Path) -> None:
    """Exécute le pipeline complet avec visualisation de chaque étape.

    Args:
        input_path: Chemin du fichier nuage de points (E57, LAS, LAZ, NPY).
        output_dir: Répertoire de sortie pour les PNG, le DXF et le recap.
    """
    t_total = time.monotonic()
    print("=" * 62)
    print("SCAN2PLAN — Pipeline de debug visuel")
    print(f"Fichier  : {input_path.name}")
    print(f"Sortie   : {output_dir}")
    print("=" * 62)

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = ScanConfig()
    stats: dict = {
        "input_name": input_path.name,
        "dxf_name": "plan.dxf",
        "seg_steps": [],
        "slice_counts": {},
        "slice_heights": {},
    }

    # ------------------------------------------------------------------
    # Étape 0 — Prétraitement
    # ------------------------------------------------------------------
    try:
        points, z_floor, z_ceiling = step0_preprocessing(input_path, cfg, output_dir)
        stats.update({
            "z_floor": z_floor,
            "z_ceiling": z_ceiling,
            "n_sor": len(points),
        })
        for label, h in zip(("high", "mid", "low"), cfg.slicing.heights):
            stats["slice_heights"][label] = float(z_floor + h)
    except Exception as exc:
        print(f"  [ERREUR étape 0] {exc}")
        return

    # ------------------------------------------------------------------
    # Étape 1 — Slicing
    # ------------------------------------------------------------------
    try:
        slices_xy, density_maps, binary_maps = step1_slicing(
            points, z_floor, cfg, output_dir
        )
        for label, pts in slices_xy.items():
            stats["slice_counts"][label] = len(pts)
        dmap_ref = (density_maps.get("mid") or density_maps.get("high")
                    or next(iter(density_maps.values())))
    except Exception as exc:
        print(f"  [ERREUR étape 1] {exc}")
        return

    # ------------------------------------------------------------------
    # Étape 2 — Hough brut
    # ------------------------------------------------------------------
    try:
        segs_by_slice, hough_all = step2_hough(density_maps, binary_maps, cfg, output_dir)
        stats["seg_steps"].append(("Hough brut", len(hough_all)))
        high_segs = segs_by_slice.get("high", hough_all)
    except Exception as exc:
        print(f"  [ERREUR étape 2] {exc}")
        return

    # ------------------------------------------------------------------
    # Étape 3 — Multi-slice filter
    # ------------------------------------------------------------------
    try:
        wall_segs = step3_multifilter(segs_by_slice, density_maps, cfg, output_dir)
        stats["seg_steps"].append(("Multi-slice filter", len(wall_segs)))
    except Exception as exc:
        print(f"  [ERREUR étape 3] {exc}")
        wall_segs = hough_all

    # ------------------------------------------------------------------
    # Étape 4 — Micro-fusion
    # ------------------------------------------------------------------
    try:
        micro_fused = step4_microfusion(wall_segs, dmap_ref, cfg, output_dir)
        stats["seg_steps"].append(("Micro-fusion (5cm)", len(micro_fused)))
    except Exception as exc:
        print(f"  [ERREUR étape 4] {exc}")
        micro_fused = wall_segs

    # ------------------------------------------------------------------
    # Étape 5 — Cleanup
    # ------------------------------------------------------------------
    try:
        cleaned = step5_cleanup(micro_fused, dmap_ref, cfg, output_dir)
        stats["seg_steps"].append(("Nettoyage parasites", len(cleaned)))
    except Exception as exc:
        print(f"  [ERREUR étape 5] {exc}")
        cleaned = micro_fused

    # ------------------------------------------------------------------
    # Étape 6 — Régularisation
    # ------------------------------------------------------------------
    try:
        regularized, dominant_angles = step6_regularization(cleaned, dmap_ref, cfg, output_dir)
        stats["seg_steps"].append(("Régularisation", len(regularized)))
    except Exception as exc:
        print(f"  [ERREUR étape 6] {exc}")
        regularized = cleaned
        dominant_angles = []

    # ------------------------------------------------------------------
    # Étape 7 — Face pairing
    # ------------------------------------------------------------------
    try:
        pairing_segs, face_pairs, unpaired_segs = step7_pairing(
            regularized, dmap_ref, cfg, output_dir
        )
        stats["seg_steps"].append(("Face pairing", len(regularized)))
        thicknesses = [fp.thickness for fp in face_pairs]
        stats.update({
            "n_pairs": len(face_pairs),
            "thicknesses": thicknesses,
            "n_unpaired": len(unpaired_segs),
            "n_porteurs": sum(1 for t in thicknesses if t > 0.12) * 2,
            "n_cloisons": sum(1 for t in thicknesses if t <= 0.12) * 2,
        })
    except Exception as exc:
        print(f"  [ERREUR étape 7] {exc}")
        face_pairs = []
        unpaired_segs = [
            Segment(x1=s.x1, y1=s.y1, x2=s.x2, y2=s.y2,
                    label="wall", confidence=s.confidence, source_slice=s.source_slice)
            for s in regularized
        ]

    # ------------------------------------------------------------------
    # Étapes 8 & 9 — Snap + Corners (sur les DetectedSegment)
    # ------------------------------------------------------------------
    try:
        snapped, snap_pts = step8_snap(regularized, dmap_ref, output_dir)
        stats["seg_steps"].append((f"Snap ({len(snap_pts)} pts)", len(snapped)))
        closed, corner_pts = step9_corners(snapped, dmap_ref, output_dir)
        stats["seg_steps"].append((f"Corners ({len(corner_pts)} coins)", len(closed)))
        final_segs = closed
    except Exception as exc:
        print(f"  [ERREUR étapes 8-9] {exc}")
        final_segs = regularized

    # ------------------------------------------------------------------
    # Étape 10 — Export final
    # ------------------------------------------------------------------
    output_dxf = output_dir / "plan.dxf"
    try:
        step10_final(final_segs, face_pairs, high_segs, dmap_ref, output_dir, output_dxf)
    except Exception as exc:
        print(f"  [ERREUR étape 10] {exc}")

    # ------------------------------------------------------------------
    # Étape 11 — Récapitulatif
    # ------------------------------------------------------------------
    stats["elapsed"] = time.monotonic() - t_total

    _recap_images = [
        output_dir / "00_nuage_stats.png",
        output_dir / "02_hough_brut.png",
        output_dir / "03_multi_slice.png",
        output_dir / "04_micro_fusion.png",
        output_dir / "05_cleanup.png",
        output_dir / "06_regularization.png",
        output_dir / "07_pairing.png",
        output_dir / "08_snap.png",
        output_dir / "09_corners.png",
        output_dir / "10_final.png",
    ]
    _recap_titles = [
        "00 Nuage Z",
        "02 Hough brut",
        "03 Multi-slice",
        "04 Micro-fusion",
        "05 Cleanup",
        "06 Régularisation",
        "07 Pairing",
        "08 Snap",
        "09 Coins",
        "10 Final",
    ]
    step11_recap(output_dir, stats, _recap_images, _recap_titles)

    # ------------------------------------------------------------------
    # Bilan console
    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("TERMINÉ")
    print(f"  Résultats dans   : {output_dir}")
    print(f"  DXF              : {output_dxf.name}")
    print(f"  Segments final   : {len(final_segs)}")
    print(f"  Paires de faces  : {len(face_pairs)}")
    if stats.get("seg_steps"):
        n_hough = stats["seg_steps"][0][1]
        retention = len(final_segs) / n_hough * 100 if n_hough else 0
        print(f"  Rétention        : {len(final_segs)}/{n_hough}  ({retention:.0f}%)")
    print(f"  Temps total      : {stats['elapsed']:.1f} s")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_pipeline.py <fichier.e57|.las|.npy> [output_dir]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"Erreur : fichier introuvable : {input_file}")
        sys.exit(1)

    output_dir = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else input_file.parent / f"debug_{input_file.stem}"
    )
    run_debug_pipeline(input_file, output_dir)
