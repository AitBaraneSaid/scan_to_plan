"""Chargement et validation de la configuration du pipeline Scan2Plan."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Chemin vers le fichier de paramètres par défaut distribué avec le package
_DEFAULT_PARAMS_PATH = Path(__file__).parent.parent.parent / "config" / "default_params.yaml"


class ConfigValidationError(ValueError):
    """Erreur levée quand la configuration contient des valeurs invalides."""


class ScanConfig:
    """Charge, valide et expose la configuration du pipeline Scan2Plan.

    Args:
        user_config_path: Chemin vers un fichier YAML utilisateur optionnel.
            Les valeurs de ce fichier surchargent les valeurs par défaut.

    Raises:
        ConfigValidationError: Si une valeur de configuration est invalide.
        FileNotFoundError: Si le fichier de configuration par défaut est introuvable.

    Example:
        >>> cfg = ScanConfig()
        >>> cfg.preprocessing.voxel_size
        0.005
        >>> cfg = ScanConfig(Path("my_config.yaml"))
    """

    def __init__(self, user_config_path: Path | None = None) -> None:
        self._data = self._load_defaults()
        if user_config_path is not None:
            self._merge_user_config(user_config_path)
        self._validate()
        logger.info(
            "Configuration chargée. voxel_size=%.4f m, %d slices définies.",
            self.preprocessing.voxel_size,
            len(self.slicing.heights),
        )

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------

    def _load_defaults(self) -> dict[str, Any]:
        """Charge le fichier de paramètres par défaut.

        Returns:
            Dictionnaire brut des paramètres.

        Raises:
            FileNotFoundError: Si le fichier par défaut est absent.
        """
        if not _DEFAULT_PARAMS_PATH.exists():
            raise FileNotFoundError(
                f"Fichier de configuration par défaut introuvable : {_DEFAULT_PARAMS_PATH}"
            )
        with _DEFAULT_PARAMS_PATH.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        logger.debug("Fichier de configuration par défaut chargé : %s", _DEFAULT_PARAMS_PATH)
        return data

    def _merge_user_config(self, path: Path) -> None:
        """Fusionne un fichier de configuration utilisateur (deep merge).

        Args:
            path: Chemin vers le fichier YAML utilisateur.

        Raises:
            FileNotFoundError: Si le fichier utilisateur est introuvable.
        """
        if not path.exists():
            raise FileNotFoundError(f"Fichier de configuration utilisateur introuvable : {path}")
        with path.open(encoding="utf-8") as fh:
            user_data = yaml.safe_load(fh) or {}
        self._data = _deep_merge(self._data, user_data)
        logger.info("Configuration utilisateur fusionnée depuis : %s", path)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Valide la cohérence des valeurs de configuration.

        Raises:
            ConfigValidationError: Si une valeur est invalide.
        """
        pre = self._data.get("preprocessing", {})
        if pre.get("voxel_size", 0) <= 0:
            raise ConfigValidationError("preprocessing.voxel_size doit être > 0")
        if pre.get("sor_k_neighbors", 0) < 1:
            raise ConfigValidationError("preprocessing.sor_k_neighbors doit être >= 1")
        if pre.get("sor_std_ratio", 0) <= 0:
            raise ConfigValidationError("preprocessing.sor_std_ratio doit être > 0")

        slicing = self._data.get("slicing", {})
        if not slicing.get("heights"):
            raise ConfigValidationError("slicing.heights ne peut pas être vide")
        if slicing.get("thickness", 0) <= 0:
            raise ConfigValidationError("slicing.thickness doit être > 0")

        dm = self._data.get("density_map", {})
        if dm.get("resolution", 0) <= 0:
            raise ConfigValidationError("density_map.resolution doit être > 0")

        fc = self._data.get("floor_ceiling", {})
        if fc.get("ransac_distance", 0) <= 0:
            raise ConfigValidationError("floor_ceiling.ransac_distance doit être > 0")
        if fc.get("ransac_iterations", 0) < 1:
            raise ConfigValidationError("floor_ceiling.ransac_iterations doit être >= 1")

        logger.debug("Configuration validée avec succès.")

    # ------------------------------------------------------------------
    # Accesseurs typés (namespaces)
    # ------------------------------------------------------------------

    @property
    def preprocessing(self) -> "_PreprocessingConfig":
        return _PreprocessingConfig(self._data["preprocessing"])

    @property
    def floor_ceiling(self) -> "_FloorCeilingConfig":
        return _FloorCeilingConfig(self._data["floor_ceiling"])

    @property
    def slicing(self) -> "_SlicingConfig":
        return _SlicingConfig(self._data["slicing"])

    @property
    def density_map(self) -> "_DensityMapConfig":
        return _DensityMapConfig(self._data["density_map"])

    @property
    def morphology(self) -> "_MorphologyConfig":
        return _MorphologyConfig(self._data["morphology"])

    @property
    def hough(self) -> "_HoughConfig":
        return _HoughConfig(self._data["hough"])

    @property
    def segment_fusion(self) -> "_SegmentFusionConfig":
        return _SegmentFusionConfig(self._data["segment_fusion"])

    @property
    def regularization(self) -> "_RegularizationConfig":
        return _RegularizationConfig(self._data["regularization"])

    @property
    def topology(self) -> "_TopologyConfig":
        return _TopologyConfig(self._data["topology"])

    @property
    def dxf(self) -> "_DxfConfig":
        return _DxfConfig(self._data["dxf"])

    def raw(self) -> dict[str, Any]:
        """Retourne le dictionnaire brut de configuration.

        Returns:
            Copie du dictionnaire de configuration.
        """
        import copy

        return copy.deepcopy(self._data)


# ------------------------------------------------------------------
# Namespaces de configuration (accès typé sans dataclass externe)
# ------------------------------------------------------------------


class _PreprocessingConfig:
    def __init__(self, data: dict[str, Any]) -> None:
        self._d = data

    @property
    def voxel_size(self) -> float:
        return float(self._d["voxel_size"])

    @property
    def sor_k_neighbors(self) -> int:
        return int(self._d["sor_k_neighbors"])

    @property
    def sor_std_ratio(self) -> float:
        return float(self._d["sor_std_ratio"])


class _FloorCeilingConfig:
    def __init__(self, data: dict[str, Any]) -> None:
        self._d = data

    @property
    def ransac_distance(self) -> float:
        return float(self._d["ransac_distance"])

    @property
    def ransac_iterations(self) -> int:
        return int(self._d["ransac_iterations"])

    @property
    def normal_tolerance_deg(self) -> float:
        return float(self._d["normal_tolerance_deg"])


class _SlicingConfig:
    def __init__(self, data: dict[str, Any]) -> None:
        self._d = data

    @property
    def heights(self) -> list[float]:
        return [float(h) for h in self._d["heights"]]

    @property
    def thickness(self) -> float:
        return float(self._d["thickness"])


class _DensityMapConfig:
    def __init__(self, data: dict[str, Any]) -> None:
        self._d = data

    @property
    def resolution(self) -> float:
        return float(self._d["resolution"])


class _MorphologyConfig:
    def __init__(self, data: dict[str, Any]) -> None:
        self._d = data

    @property
    def kernel_size(self) -> int:
        return int(self._d["kernel_size"])

    @property
    def close_iterations(self) -> int:
        return int(self._d["close_iterations"])

    @property
    def open_iterations(self) -> int:
        return int(self._d["open_iterations"])


class _HoughConfig:
    def __init__(self, data: dict[str, Any]) -> None:
        self._d = data

    @property
    def rho(self) -> int:
        return int(self._d["rho"])

    @property
    def theta_deg(self) -> float:
        return float(self._d["theta_deg"])

    @property
    def threshold(self) -> int:
        return int(self._d["threshold"])

    @property
    def min_line_length(self) -> int:
        return int(self._d["min_line_length"])

    @property
    def max_line_gap(self) -> int:
        return int(self._d["max_line_gap"])


class _SegmentFusionConfig:
    def __init__(self, data: dict[str, Any]) -> None:
        self._d = data

    @property
    def angle_tolerance_deg(self) -> float:
        return float(self._d["angle_tolerance_deg"])

    @property
    def perpendicular_dist(self) -> float:
        return float(self._d["perpendicular_dist"])

    @property
    def max_gap(self) -> float:
        return float(self._d["max_gap"])


class _RegularizationConfig:
    def __init__(self, data: dict[str, Any]) -> None:
        self._d = data

    @property
    def snap_tolerance_deg(self) -> float:
        return float(self._d["snap_tolerance_deg"])


class _TopologyConfig:
    def __init__(self, data: dict[str, Any]) -> None:
        self._d = data

    @property
    def intersection_distance(self) -> float:
        return float(self._d["intersection_distance"])

    @property
    def min_segment_length(self) -> float:
        return float(self._d["min_segment_length"])


class _DxfConfig:
    def __init__(self, data: dict[str, Any]) -> None:
        self._d = data

    @property
    def version(self) -> str:
        return str(self._d["version"])

    @property
    def layers(self) -> dict[str, str]:
        return dict(self._d["layers"])


# ------------------------------------------------------------------
# Utilitaire
# ------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Fusionne récursivement deux dictionnaires, override prend priorité.

    Args:
        base: Dictionnaire de base (paramètres par défaut).
        override: Dictionnaire de surcharge (paramètres utilisateur).

    Returns:
        Nouveau dictionnaire fusionné.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
