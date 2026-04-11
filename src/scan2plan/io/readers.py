"""Lecture de nuages de points : E57, LAS/LAZ, dispatch par extension."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Extensions reconnues
_E57_EXTENSIONS = {".e57"}
_LAS_EXTENSIONS = {".las", ".laz"}
_NPY_EXTENSIONS = {".npy"}   # Tableaux NumPy — usage interne et tests


class UnsupportedFormatError(ValueError):
    """Levée quand l'extension du fichier n'est pas supportée."""


def read_e57(path: Path) -> np.ndarray:
    """Lit un fichier E57 et retourne le nuage de points en mètres.

    Gère les fichiers multi-scans : les matrices de transformation de chaque
    scan sont appliquées pour ramener tous les points dans le repère global
    du projet.

    Args:
        path: Chemin vers le fichier .e57.

    Returns:
        Array NumPy de forme (N, 3) float64, coordonnées XYZ en mètres
        dans le repère global.

    Raises:
        FileNotFoundError: Si le fichier est introuvable.
        RuntimeError: Si le fichier est corrompu ou vide.

    Example:
        >>> points = read_e57(Path("scan.e57"))
        >>> points.shape
        (1234567, 3)
    """
    import pye57

    if not path.exists():
        raise FileNotFoundError(f"Fichier E57 introuvable : {path}")

    logger.info("Lecture E57 : %s", path)

    e57 = pye57.E57(str(path))
    all_points: list[np.ndarray] = []

    scan_count = e57.scan_count
    logger.debug("Nombre de scans dans le fichier : %d", scan_count)

    for scan_index in range(scan_count):
        raw = e57.read_scan_raw(scan_index)
        xyz = _extract_xyz_from_raw(raw)
        if xyz is None or len(xyz) == 0:
            logger.warning("Scan %d : aucun point valide, ignoré.", scan_index)
            continue

        transform = _get_scan_transform(e57, scan_index)
        if transform is not None:
            xyz = _apply_transform(xyz, transform)
            logger.debug("Scan %d : matrice de transformation appliquée.", scan_index)

        logger.debug("Scan %d : %d points chargés.", scan_index, len(xyz))
        all_points.append(xyz)

    if not all_points:
        raise RuntimeError(f"Aucun point valide trouvé dans le fichier E57 : {path}")

    points = np.vstack(all_points).astype(np.float64)
    _log_point_cloud_info(points, path, "E57")
    return points


def _extract_xyz_from_raw(raw: dict) -> np.ndarray | None:
    """Extrait les coordonnées XYZ depuis le dictionnaire brut d'un scan pye57.

    Args:
        raw: Dictionnaire retourné par ``e57.read_scan_raw()``.

    Returns:
        Array (N, 3) float64 ou None si les données sont absentes.
    """
    try:
        x = np.asarray(raw["cartesianX"], dtype=np.float64)
        y = np.asarray(raw["cartesianY"], dtype=np.float64)
        z = np.asarray(raw["cartesianZ"], dtype=np.float64)
    except KeyError:
        logger.warning("Le scan ne contient pas de coordonnées cartésiennes (cartesianX/Y/Z).")
        return None

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return np.column_stack([x[mask], y[mask], z[mask]])


def _get_scan_transform(e57: "pye57.E57", scan_index: int) -> np.ndarray | None:
    """Extrait la matrice de transformation 4×4 d'un scan E57.

    Args:
        e57: Objet pye57.E57 ouvert.
        scan_index: Indice du scan.

    Returns:
        Matrice 4×4 float64, ou None si aucune transformation n'est définie.
    """
    try:
        header = e57.get_header(scan_index)
        rot = header.rotation_matrix
        trans = header.translation
        if rot is None or trans is None:
            return None
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.asarray(rot, dtype=np.float64)
        matrix[:3, 3] = np.asarray(trans, dtype=np.float64)
        return matrix
    except Exception as exc:  # noqa: BLE001
        logger.debug("Impossible de lire la transformation du scan %d : %s", scan_index, exc)
        return None


def _apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Applique une transformation homogène 4×4 à un nuage de points.

    Args:
        points: Array (N, 3) float64.
        matrix: Matrice de transformation 4×4.

    Returns:
        Array (N, 3) float64 transformé.
    """
    ones = np.ones((len(points), 1), dtype=np.float64)
    homogeneous = np.hstack([points, ones])
    transformed = (matrix @ homogeneous.T).T
    return transformed[:, :3]


def read_las(path: Path) -> np.ndarray:
    """Lit un fichier LAS ou LAZ et retourne le nuage de points en mètres.

    Les coordonnées sont automatiquement converties via les facteurs d'échelle
    et les offsets définis dans l'en-tête LAS (format de stockage entier → mètres).

    Args:
        path: Chemin vers le fichier .las ou .laz.

    Returns:
        Array NumPy de forme (N, 3) float64, coordonnées XYZ en mètres.

    Raises:
        FileNotFoundError: Si le fichier est introuvable.
        RuntimeError: Si le fichier est vide ou corrompu.

    Example:
        >>> points = read_las(Path("scan.laz"))
        >>> points.shape
        (987654, 3)
    """
    import laspy

    if not path.exists():
        raise FileNotFoundError(f"Fichier LAS/LAZ introuvable : {path}")

    logger.info("Lecture LAS/LAZ : %s", path)

    las = laspy.read(str(path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)

    if len(x) == 0:
        raise RuntimeError(f"Fichier LAS/LAZ vide : {path}")

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    points = np.column_stack([x[mask], y[mask], z[mask]])

    if not mask.all():
        logger.warning("%d points avec coordonnées non finies supprimés.", (~mask).sum())

    _log_point_cloud_info(points, path, "LAS/LAZ")
    return points


def read_point_cloud(path: Path) -> np.ndarray:
    """Lit un nuage de points en dispatchant selon l'extension du fichier.

    Formats supportés : .e57, .las, .laz

    Args:
        path: Chemin vers le fichier de nuage de points.

    Returns:
        Array NumPy de forme (N, 3) float64, coordonnées XYZ en mètres.

    Raises:
        FileNotFoundError: Si le fichier est introuvable.
        UnsupportedFormatError: Si l'extension n'est pas reconnue.

    Example:
        >>> points = read_point_cloud(Path("relevé.e57"))
        >>> points = read_point_cloud(Path("relevé.laz"))
    """
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    suffix = path.suffix.lower()
    logger.debug("Format détecté : %s pour le fichier %s", suffix, path.name)

    if suffix in _E57_EXTENSIONS:
        return read_e57(path)
    if suffix in _LAS_EXTENSIONS:
        return read_las(path)
    if suffix in _NPY_EXTENSIONS:
        return _read_npy(path)

    supported = sorted(_E57_EXTENSIONS | _LAS_EXTENSIONS | _NPY_EXTENSIONS)
    raise UnsupportedFormatError(
        f"Format non supporté : '{suffix}'. "
        f"Formats acceptés : {supported}. "
        f"Convertissez le fichier en E57 ou LAS avant de relancer le pipeline."
    )


def _read_npy(path: Path) -> np.ndarray:
    """Charge un tableau NumPy (N, 3) depuis un fichier .npy.

    Usage interne : tests, fichiers intermédiaires sauvegardés par le pipeline.

    Args:
        path: Chemin vers le fichier .npy.

    Returns:
        Array (N, 3) float64.

    Raises:
        ValueError: Si le tableau n'est pas de forme (N, 3).
    """
    points = np.load(str(path)).astype(np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Le fichier .npy doit contenir un tableau (N, 3), "
            f"reçu : {points.shape}"
        )
    _log_point_cloud_info(points, path, "NPY")
    return points


def _log_point_cloud_info(points: np.ndarray, path: Path, fmt: str) -> None:
    """Logge les informations de base sur le nuage chargé.

    Args:
        points: Array (N, 3) float64.
        path: Chemin source (pour le log).
        fmt: Nom du format (pour le log).
    """
    n = len(points)
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)
    logger.info(
        "[%s] %s — %d points chargés. "
        "BBox X=[%.3f, %.3f] Y=[%.3f, %.3f] Z=[%.3f, %.3f] (m)",
        fmt,
        path.name,
        n,
        x_min, x_max,
        y_min, y_max,
        z_min, z_max,
    )
