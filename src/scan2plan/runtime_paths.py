"""Helpers de resolution de chemins en source et dans un bundle PyInstaller."""

from __future__ import annotations

import sys
from pathlib import Path


def get_config_dir(module_file: str | Path | None = None) -> Path:
    """Retourne le dossier ``config/`` du projet ou du bundle.

    La recherche couvre les cas suivants :
    - execution depuis le depot (`src/scan2plan/...`);
    - execution depuis un bundle PyInstaller (`sys._MEIPASS`);
    - execution via un binaire place a cote d'un dossier `config/`.

    Args:
        module_file: Fichier d'ancrage optionnel. Par defaut, ce module.

    Returns:
        Chemin absolu vers le dossier ``config``.

    Raises:
        FileNotFoundError: Si aucun dossier ``config`` n'est trouve.
    """
    anchor = Path(module_file) if module_file is not None else Path(__file__)
    anchor = anchor.resolve()

    roots: list[Path] = []

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        roots.append(Path(sys._MEIPASS))

    roots.extend(
        [
            anchor.parents[2],  # depot: repo/src/scan2plan/module.py
            anchor.parents[1],  # fallback: package parent
            Path(sys.executable).resolve().parent,
            Path.cwd(),
        ]
    )

    seen: set[Path] = set()
    ordered_roots: list[Path] = []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        ordered_roots.append(root)
        config_dir = root / "config"
        if config_dir.is_dir():
            return config_dir

    searched = ", ".join(str(root / "config") for root in ordered_roots)
    raise FileNotFoundError(f"Dossier de configuration introuvable. Chemins testes : {searched}")
