"""Fixtures pytest partagées pour l'ensemble de la suite de tests Scan2Plan."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.generate_fixtures import generate_simple_room

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_SIMPLE_ROOM_NPY = _FIXTURE_DIR / "simple_room.npy"


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures() -> None:
    """Génère les fixtures .npy si elles n'existent pas encore.

    Scope session : exécuté une seule fois par session pytest.
    Autouse : actif sans avoir à le déclarer explicitement dans chaque test.
    """
    if not _SIMPLE_ROOM_NPY.exists():
        points = generate_simple_room()
        np.save(_SIMPLE_ROOM_NPY, points)


@pytest.fixture(scope="session")
def simple_room_points() -> np.ndarray:
    """Charge le nuage synthétique de la pièce rectangulaire 4×3×2.5 m.

    Returns:
        Array (N, 3) float64 — ~10 000 points avec bruit gaussien σ=2 mm.
    """
    return np.load(_SIMPLE_ROOM_NPY)


@pytest.fixture(scope="session")
def default_config():
    """Charge la configuration par défaut du pipeline.

    Returns:
        Instance de ScanConfig avec les paramètres de config/default_params.yaml.
    """
    from scan2plan.config import ScanConfig

    return ScanConfig()
