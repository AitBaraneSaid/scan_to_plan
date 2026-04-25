"""Analyse la distribution Z d'un nuage E57 pour identifier les niveaux d'étage."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pye57


def analyze_z(filepath: str, sample_ratio: float = 0.05) -> None:
    path = Path(filepath)
    e57 = pye57.E57(str(path))
    header = e57.get_header(0)
    total = header.point_count

    print(f"\nFichier : {path.name}  ({total:,} points)")
    print(f"Échantillon : {sample_ratio*100:.0f}% = ~{int(total*sample_ratio):,} points\n")

    # Lecture complète (pye57 ne supporte pas la lecture partielle par champ)
    data = e57.read_scan(0, ignore_missing_fields=True)
    z = np.asarray(data["cartesianZ"], dtype=np.float32)

    # Sous-échantillonnage aléatoire
    if sample_ratio < 1.0:
        idx = np.random.default_rng(42).choice(len(z), size=int(len(z) * sample_ratio), replace=False)
        z = z[idx]

    print(f"Z min : {z.min():.3f} m")
    print(f"Z max : {z.max():.3f} m")
    print(f"Z moy : {z.mean():.3f} m")
    print(f"Z med : {np.median(z):.3f} m")

    # Histogramme par tranches de 0.5 m
    z_min = float(np.floor(z.min()))
    z_max = float(np.ceil(z.max()))
    bins = np.arange(z_min, z_max + 0.5, 0.5)
    counts, edges = np.histogram(z, bins=bins)
    total_sampled = len(z)

    print(f"\nDistribution Z (tranches 0.5 m) — seuil affiché > 0.5% :")
    print(f"{'Tranche Z':>20}  {'Points':>10}  {'%':>6}  Barre")
    print("-" * 60)

    max_count = counts.max()
    for i, (lo, hi, cnt) in enumerate(zip(edges[:-1], edges[1:], counts)):
        pct = cnt / total_sampled * 100
        if pct < 0.5:
            continue
        bar = "#" * int(cnt / max_count * 40)
        print(f"  [{lo:6.2f} – {hi:6.2f}] m  {cnt:>10,}  {pct:>5.1f}%  {bar}")

    # Détecter les pics (planchers/plafonds potentiels)
    print("\nPics détectés (densité locale maximale) :")
    from scipy.signal import find_peaks
    peaks, props = find_peaks(counts, prominence=counts.max() * 0.05, distance=4)
    for p in peaks:
        lo, hi = edges[p], edges[p + 1]
        pct = counts[p] / total_sampled * 100
        print(f"  Z ≈ {(lo+hi)/2:.2f} m  ({pct:.1f}% des points) — plancher/plafond probable")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python scripts/analyze_z_distribution.py <fichier.e57> [ratio]")
        sys.exit(1)
    ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    analyze_z(sys.argv[1], ratio)
