"""Benchmarks de performance — temps d'exécution par étape.

Mesure le temps de chaque étape du pipeline sur des données synthétiques
de taille croissante (simulant un appartement de 50 à 200 m²).

Utilisation :
    python tests/benchmarks/bench_pipeline.py
    python tests/benchmarks/bench_pipeline.py --n-points 2000000 --output bench_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Générateurs de données synthétiques
# ---------------------------------------------------------------------------

def _make_synthetic_room(
    n_points: int = 500_000,
    width_m: float = 10.0,
    depth_m: float = 8.0,
    height_m: float = 2.80,
    seed: int = 42,
) -> np.ndarray:
    """Génère un nuage de points synthétique d'une pièce rectangulaire.

    Distribue des points sur les 6 faces (4 murs + sol + plafond) avec du bruit.

    Args:
        n_points: Nombre total de points.
        width_m: Largeur de la pièce (axe X).
        depth_m: Profondeur de la pièce (axe Y).
        height_m: Hauteur de la pièce (axe Z).
        seed: Graine aléatoire pour la reproductibilité.

    Returns:
        Tableau NumPy (N, 3) float32 avec les coordonnées XYZ.
    """
    rng = np.random.default_rng(seed)
    pts_per_face = n_points // 6
    noise = 0.005  # 5 mm de bruit

    faces = []

    # Sol (Z = 0)
    xs = rng.uniform(0, width_m, pts_per_face)
    ys = rng.uniform(0, depth_m, pts_per_face)
    zs = rng.normal(0, noise, pts_per_face)
    faces.append(np.column_stack([xs, ys, zs]))

    # Plafond (Z = height_m)
    xs = rng.uniform(0, width_m, pts_per_face)
    ys = rng.uniform(0, depth_m, pts_per_face)
    zs = rng.normal(height_m, noise, pts_per_face)
    faces.append(np.column_stack([xs, ys, zs]))

    # Mur X=0
    ys = rng.uniform(0, depth_m, pts_per_face)
    zs = rng.uniform(0, height_m, pts_per_face)
    xs = rng.normal(0, noise, pts_per_face)
    faces.append(np.column_stack([xs, ys, zs]))

    # Mur X=width_m
    ys = rng.uniform(0, depth_m, pts_per_face)
    zs = rng.uniform(0, height_m, pts_per_face)
    xs = rng.normal(width_m, noise, pts_per_face)
    faces.append(np.column_stack([xs, ys, zs]))

    # Mur Y=0
    xs = rng.uniform(0, width_m, pts_per_face)
    zs = rng.uniform(0, height_m, pts_per_face)
    ys = rng.normal(0, noise, pts_per_face)
    faces.append(np.column_stack([xs, ys, zs]))

    # Mur Y=depth_m
    xs = rng.uniform(0, width_m, pts_per_face)
    zs = rng.uniform(0, height_m, pts_per_face)
    ys = rng.normal(depth_m, noise, pts_per_face)
    faces.append(np.column_stack([xs, ys, zs]))

    return np.vstack(faces).astype(np.float32)


# ---------------------------------------------------------------------------
# Mesure de temps
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Résultat de benchmark pour une étape."""

    step: str
    n_points_in: int
    n_points_out: int
    duration_ms: float
    notes: str = ""


class Timer:
    """Context manager de mesure de temps."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


# ---------------------------------------------------------------------------
# Benchmarks par étape
# ---------------------------------------------------------------------------

def bench_density_map(points_2d: np.ndarray, resolution: float = 0.005) -> StepResult:
    """Benchmark : génération de density map."""
    from scan2plan.slicing.density_map import create_density_map

    with Timer() as t:
        result = create_density_map(points_2d, resolution=resolution)

    return StepResult(
        step="density_map",
        n_points_in=len(points_2d),
        n_points_out=int((result.image > 0).sum()),
        duration_ms=t.elapsed_ms,
        notes=f"image {result.width}×{result.height}px, résolution {resolution*1000:.0f}mm/px",
    )


def bench_morphology(image: np.ndarray) -> StepResult:
    """Benchmark : binarisation et nettoyage morphologique."""
    from scan2plan.detection.morphology import binarize_density_map, morphological_cleanup

    with Timer() as t:
        binary = binarize_density_map(image)
        _ = morphological_cleanup(binary, kernel_size=5, close_iterations=2, open_iterations=1)

    return StepResult(
        step="morphology",
        n_points_in=image.size,
        n_points_out=int(binary.sum()),
        duration_ms=t.elapsed_ms,
        notes=f"image {image.shape[1]}x{image.shape[0]}px",
    )


def bench_hough(binary_image: np.ndarray) -> StepResult:
    """Benchmark : détection de segments par Hough probabiliste."""
    from scan2plan.detection.line_detection import detect_lines_hough
    from scan2plan.slicing.density_map import DensityMapResult

    dm = DensityMapResult(
        image=binary_image,
        x_min=0.0, y_min=0.0,
        resolution=0.005,
        width=binary_image.shape[1],
        height=binary_image.shape[0],
    )

    with Timer() as t:
        segments = detect_lines_hough(
            binary_image, dm,
            rho=1, theta_deg=0.5, threshold=50,
            min_line_length=50, max_line_gap=20,
            source_slice="mid",
        )

    return StepResult(
        step="hough",
        n_points_in=int(binary_image.sum()),
        n_points_out=len(segments),
        duration_ms=t.elapsed_ms,
        notes=f"{len(segments)} segments détectés",
    )


def bench_segment_fusion(segments: list) -> StepResult:
    """Benchmark : fusion de segments colinéaires."""
    from scan2plan.detection.segment_fusion import fuse_collinear_segments

    with Timer() as t:
        fused = fuse_collinear_segments(
            segments,
            angle_tolerance_deg=3.0,
            perpendicular_dist=0.03,
            max_gap=0.20,
        )

    return StepResult(
        step="segment_fusion",
        n_points_in=len(segments),
        n_points_out=len(fused),
        duration_ms=t.elapsed_ms,
        notes=f"{len(segments)} → {len(fused)} segments",
    )


def bench_zone_scoring(density_map, segments: list) -> StepResult:
    """Benchmark : scoring par zone."""
    from scan2plan.qa.zone_scoring import compute_zone_scores

    with Timer() as t:
        zone_map = compute_zone_scores(density_map, segments, [], cell_size_m=1.0)

    return StepResult(
        step="zone_scoring",
        n_points_in=len(segments),
        n_points_out=zone_map.n_cols * zone_map.n_rows,
        duration_ms=t.elapsed_ms,
        notes=f"grille {zone_map.n_cols}×{zone_map.n_rows}, score global {zone_map.global_score:.2f}",
    )


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------

def run_benchmarks(n_points: int = 500_000) -> list[StepResult]:
    """Exécute la suite de benchmarks et retourne les résultats."""
    print(f"\n{'='*60}")
    print(f"Benchmarks Scan2Plan — {n_points:,} points")
    print(f"{'='*60}\n")

    results: list[StepResult] = []

    # Générer le nuage de points synthétique
    print("Génération du nuage de points synthétique...")
    points = _make_synthetic_room(n_points=n_points)
    print(f"  -> {len(points):,} points generes (10m x 8m x 2.8m)\n")

    # Extraire une slice 2D à hauteur de coupe standard (~1.10 m)
    mask = (points[:, 2] >= 1.05) & (points[:, 2] <= 1.15)
    points_2d = points[mask, :2]
    print(f"Slice médiane (1.05-1.15m) : {len(points_2d):,} points\n")

    # --- Étape 1 : Density map ---
    print("1/5  Density map...")
    r = bench_density_map(points_2d)
    results.append(r)
    print(f"     {r.duration_ms:.1f} ms — {r.notes}")

    # Récupérer la density map pour les étapes suivantes
    from scan2plan.slicing.density_map import create_density_map
    dm = create_density_map(points_2d, resolution=0.005)

    # --- Étape 2 : Morphologie ---
    print("2/5  Morphologie (binarisation + nettoyage)...")
    r = bench_morphology(dm.image.astype(np.uint8))
    results.append(r)
    print(f"     {r.duration_ms:.1f} ms — {r.notes}")

    from scan2plan.detection.morphology import binarize_density_map, morphological_cleanup
    binary = morphological_cleanup(binarize_density_map(dm.image), kernel_size=5,
                                   close_iterations=2, open_iterations=1)

    # --- Étape 3 : Hough ---
    print("3/5  Détection Hough...")
    r = bench_hough(binary)
    results.append(r)
    print(f"     {r.duration_ms:.1f} ms — {r.notes}")

    from scan2plan.detection.line_detection import detect_lines_hough
    segments = detect_lines_hough(
        binary, dm,
        rho=1, theta_deg=0.5, threshold=50,
        min_line_length=50, max_line_gap=20,
        source_slice="mid",
    )

    # --- Étape 4 : Fusion ---
    print("4/5  Fusion de segments colinéaires...")
    r = bench_segment_fusion(segments)
    results.append(r)
    print(f"     {r.duration_ms:.1f} ms — {r.notes}")

    from scan2plan.detection.segment_fusion import fuse_collinear_segments
    fused = fuse_collinear_segments(segments, angle_tolerance_deg=3.0,
                                    perpendicular_dist=0.03, max_gap=0.20)

    # --- Étape 5 : Zone scoring ---
    print("5/5  Zone scoring QA...")
    r = bench_zone_scoring(dm, fused)
    results.append(r)
    print(f"     {r.duration_ms:.1f} ms — {r.notes}")

    # Résumé
    total_ms = sum(r.duration_ms for r in results)
    print(f"\n{'='*60}")
    print(f"Total pipeline 2D (5 étapes) : {total_ms:.1f} ms ({total_ms/1000:.2f} s)")
    print(f"{'='*60}\n")

    # Tableau récapitulatif
    print(f"{'Étape':<20} {'Entrée':>10} {'Sortie':>10} {'Durée ms':>10}")
    print("-" * 54)
    for r in results:
        print(f"{r.step:<20} {r.n_points_in:>10,} {r.n_points_out:>10,} {r.duration_ms:>10.1f}")
    print()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmarks de performance Scan2Plan"
    )
    parser.add_argument(
        "--n-points", type=int, default=500_000,
        help="Nombre de points dans le nuage synthétique (défaut : 500 000)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Chemin JSON de sortie pour les résultats (optionnel)"
    )
    args = parser.parse_args()

    results = run_benchmarks(n_points=args.n_points)

    if args.output:
        out_path = Path(args.output)
        data = {
            "n_points": args.n_points,
            "steps": [asdict(r) for r in results],
            "total_ms": sum(r.duration_ms for r in results),
        }
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"Résultats sauvegardés : {out_path}")


if __name__ == "__main__":
    main()
