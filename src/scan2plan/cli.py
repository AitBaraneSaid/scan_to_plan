"""Point d'entrée CLI du pipeline Scan2Plan."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="scan2plan",
    help="Transforme un nuage de points 3D indoor en plan 2D DXF.",
    add_completion=False,
)


@app.command()
def process(
    input_file: Path = typer.Argument(
        ...,
        help="Chemin vers le fichier nuage de points (.e57, .las, .laz).",
        exists=True,
        readable=True,
    ),
    output_file: Path = typer.Argument(
        ...,
        help="Chemin vers le fichier DXF de sortie.",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Fichier de configuration YAML utilisateur (surcharge les paramètres par défaut).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Active les logs DEBUG.",
    ),
) -> None:
    """Lance le pipeline Scan2Plan sur un fichier de nuage de points.

    Exemple :

        scan2plan scan.e57 output/plan.dxf

        scan2plan scan.laz output/plan.dxf --config my_params.yaml --verbose
    """
    _configure_logging(verbose)

    from scan2plan.config import ScanConfig
    from scan2plan.pipeline import run_pipeline

    config = ScanConfig(user_config_path=config_file)
    run_pipeline(input_file, output_file, config)


def _configure_logging(verbose: bool) -> None:
    """Configure le logging selon le niveau de verbosité.

    Args:
        verbose: Si True, active le niveau DEBUG. Sinon, niveau INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
