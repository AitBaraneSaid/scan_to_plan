"""Tests unitaires pour preprocessing/outlier_removal.py."""

from __future__ import annotations

import numpy as np
import pytest

from scan2plan.preprocessing.outlier_removal import remove_statistical_outliers

# Points aberrants isolés — dispersés loin du nuage principal.
# IMPORTANT : les outliers sont placés isolément (pas en cluster dense)
# car le SOR détecte les points dont les k voisins sont loin, pas les clusters distants.
_OUTLIER_POSITION = np.array([100.0, 100.0, 100.0])
_N_OUTLIERS = 5  # peu de points, très dispersés → chacun est isolé


def _add_outliers(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ajoute des points aberrants isolés dispersés loin du nuage principal.

    Chaque outlier est espacé de 5 m de ses voisins pour garantir qu'il
    est bien isolé (distance k-NN élevée) et donc détecté par le SOR.

    Args:
        points: Nuage de points d'origine.
        rng: Générateur aléatoire pour le placement des outliers.

    Returns:
        Nuage augmenté avec les outliers en fin de tableau.
    """
    outliers = np.array([
        [100.0,   0.0,   0.0],
        [  0.0, 100.0,   0.0],
        [  0.0,   0.0, 100.0],
        [100.0, 100.0,   0.0],
        [  0.0, 100.0, 100.0],
    ], dtype=np.float64)
    return np.vstack([points, outliers])


class TestRemoveStatisticalOutliers:
    def test_removes_outliers(self, simple_room_points: np.ndarray) -> None:
        """Les points aberrants isolés (à > 50 m du nuage principal) doivent être supprimés.

        Le SOR détecte les points dont la distance moyenne aux k voisins est anormalement
        élevée. Des points isolés loin du nuage sont bien détectés ; des clusters denses
        à distance ne le sont pas (leurs voisins sont proches entre eux).
        """
        rng = np.random.default_rng(0)
        contaminated = _add_outliers(simple_room_points, rng)
        assert len(contaminated) == len(simple_room_points) + _N_OUTLIERS

        result = remove_statistical_outliers(contaminated, nb_neighbors=20, std_ratio=2.0)

        # Aucun point du résultat ne doit être à plus de 10 m du nuage original
        near_outlier = result[np.abs(result).max(axis=1) > 10.0]
        assert len(near_outlier) == 0, (
            f"{len(near_outlier)} points aberrants isolés ont survécu au SOR."
        )

    def test_preserves_inliers(self, simple_room_points: np.ndarray) -> None:
        """Après SOR, le nombre de points restants doit être proche de l'original.

        Tolérance : pas plus de 5 % de perte sur le nuage propre.
        """
        rng = np.random.default_rng(1)
        contaminated = _add_outliers(simple_room_points, rng)

        result = remove_statistical_outliers(contaminated, nb_neighbors=20, std_ratio=2.0)

        # Au moins 95 % des points originaux doivent être conservés
        min_expected = int(len(simple_room_points) * 0.95)
        assert len(result) >= min_expected, (
            f"Trop de points supprimés : {len(result)} conservés, "
            f"minimum attendu {min_expected} (95 % de {len(simple_room_points)})."
        )

    def test_result_shape(self, simple_room_points: np.ndarray) -> None:
        """Le résultat doit avoir exactement 3 colonnes."""
        result = remove_statistical_outliers(simple_room_points, nb_neighbors=20, std_ratio=2.0)
        assert result.ndim == 2
        assert result.shape[1] == 3

    def test_result_dtype_float64(self, simple_room_points: np.ndarray) -> None:
        result = remove_statistical_outliers(simple_room_points)
        assert result.dtype == np.float64

    def test_default_parameters(self, simple_room_points: np.ndarray) -> None:
        """Les paramètres par défaut (nb_neighbors=20, std_ratio=2.0) doivent fonctionner."""
        result = remove_statistical_outliers(simple_room_points)
        assert len(result) > 0

    def test_invalid_nb_neighbors_raises(self, simple_room_points: np.ndarray) -> None:
        with pytest.raises(ValueError, match="nb_neighbors"):
            remove_statistical_outliers(simple_room_points, nb_neighbors=0)

    def test_invalid_std_ratio_raises(self, simple_room_points: np.ndarray) -> None:
        with pytest.raises(ValueError, match="std_ratio"):
            remove_statistical_outliers(simple_room_points, std_ratio=0.0)

    def test_no_outliers_in_clean_cloud(self, simple_room_points: np.ndarray) -> None:
        """Sur un nuage propre, le SOR ne doit pas supprimer trop de points (< 10 %)."""
        result = remove_statistical_outliers(simple_room_points, nb_neighbors=20, std_ratio=2.0)
        assert len(result) >= int(len(simple_room_points) * 0.90)
