"""Génération des nuages de points synthétiques pour les tests.

Exécuter ce script une fois pour (re)générer les fixtures :

    python tests/fixtures/generate_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


_FIXTURE_DIR = Path(__file__).parent

# Paramètres de la pièce synthétique
_ROOM_WIDTH = 4.0    # mètres (X)
_ROOM_DEPTH = 3.0    # mètres (Y)
_ROOM_HEIGHT = 2.5   # mètres (Z)
_NOISE_STD = 0.002   # σ = 2 mm
_N_TOTAL = 10_000
_RNG_SEED = 42


def generate_simple_room(
    width: float = _ROOM_WIDTH,
    depth: float = _ROOM_DEPTH,
    height: float = _ROOM_HEIGHT,
    n_points: int = _N_TOTAL,
    noise_std: float = _NOISE_STD,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Génère un nuage de points synthétique d'une pièce rectangulaire.

    La pièce est centrée en (0, 0) avec :
    - Sol à Z = 0
    - Plafond à Z = height
    - 4 murs à X = ±width/2 et Y = ±depth/2

    Les points sont répartis uniformément sur les 6 surfaces avec un léger
    bruit gaussien isotrope de σ = ``noise_std``.

    Args:
        width: Largeur de la pièce en mètres (axe X).
        depth: Profondeur de la pièce en mètres (axe Y).
        height: Hauteur sous plafond en mètres (axe Z).
        n_points: Nombre total de points à générer.
        noise_std: Écart-type du bruit gaussien (mètres).
        rng: Générateur aléatoire NumPy (reproductibilité des tests).

    Returns:
        Array (N, 3) float64 — nuage synthétique en mètres.

    Example:
        >>> pts = generate_simple_room()
        >>> pts.shape
        (10000, 3)
        >>> pts[:, 2].min() >= -0.01  # sol ≈ Z=0
        True
    """
    if rng is None:
        rng = np.random.default_rng(_RNG_SEED)

    # Surfaces : 6 faces, répartir les points proportionnellement aux aires
    areas = _compute_face_areas(width, depth, height)
    total_area = sum(areas.values())
    n_per_face = {
        face: max(1, int(round(area / total_area * n_points)))
        for face, area in areas.items()
    }

    cloud_parts: list[np.ndarray] = []

    # Sol (Z = 0)
    n = n_per_face["floor"]
    x = rng.uniform(-width / 2, width / 2, n)
    y = rng.uniform(-depth / 2, depth / 2, n)
    z = np.zeros(n)
    cloud_parts.append(np.column_stack([x, y, z]))

    # Plafond (Z = height)
    n = n_per_face["ceiling"]
    x = rng.uniform(-width / 2, width / 2, n)
    y = rng.uniform(-depth / 2, depth / 2, n)
    z = np.full(n, height)
    cloud_parts.append(np.column_stack([x, y, z]))

    # Mur X = -width/2 (gauche)
    n = n_per_face["wall_x_neg"]
    x = np.full(n, -width / 2)
    y = rng.uniform(-depth / 2, depth / 2, n)
    z = rng.uniform(0, height, n)
    cloud_parts.append(np.column_stack([x, y, z]))

    # Mur X = +width/2 (droit)
    n = n_per_face["wall_x_pos"]
    x = np.full(n, width / 2)
    y = rng.uniform(-depth / 2, depth / 2, n)
    z = rng.uniform(0, height, n)
    cloud_parts.append(np.column_stack([x, y, z]))

    # Mur Y = -depth/2 (avant)
    n = n_per_face["wall_y_neg"]
    x = rng.uniform(-width / 2, width / 2, n)
    y = np.full(n, -depth / 2)
    z = rng.uniform(0, height, n)
    cloud_parts.append(np.column_stack([x, y, z]))

    # Mur Y = +depth/2 (arrière)
    n = n_per_face["wall_y_pos"]
    x = rng.uniform(-width / 2, width / 2, n)
    y = np.full(n, depth / 2)
    z = rng.uniform(0, height, n)
    cloud_parts.append(np.column_stack([x, y, z]))

    points = np.vstack(cloud_parts).astype(np.float64)

    # Bruit gaussien
    points += rng.normal(0, noise_std, points.shape)

    return points


def _compute_face_areas(
    width: float,
    depth: float,
    height: float,
) -> dict[str, float]:
    """Calcule l'aire de chaque face de la pièce.

    Args:
        width: Largeur (axe X).
        depth: Profondeur (axe Y).
        height: Hauteur (axe Z).

    Returns:
        Dictionnaire {nom_face: aire_m²}.
    """
    return {
        "floor": width * depth,
        "ceiling": width * depth,
        "wall_x_neg": depth * height,
        "wall_x_pos": depth * height,
        "wall_y_neg": width * height,
        "wall_y_pos": width * height,
    }


def main() -> None:
    """Génère et sauvegarde les fixtures de test."""
    output_path = _FIXTURE_DIR / "simple_room.npy"
    points = generate_simple_room()
    np.save(output_path, points)
    print(f"Fixture générée : {output_path}")
    print(f"  Shape      : {points.shape}")
    print(f"  X range    : [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}] m")
    print(f"  Y range    : [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}] m")
    print(f"  Z range    : [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}] m")


if __name__ == "__main__":
    main()
