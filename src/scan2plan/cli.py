"""Point d'entrée CLI du pipeline Scan2Plan."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import typer

app = typer.Typer(
    name="scan2plan",
    help="Scan2Plan — Conversion automatique nuage de points 3D → plan 2D DXF.",
    add_completion=False,
)


@app.command()
def process(
    input_file: Path = typer.Argument(
        ...,
        help="Fichier nuage de points (.e57, .las, .laz).",
        exists=True,
        readable=True,
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Fichier DXF de sortie. Défaut : même nom que l'entrée avec extension .dxf.",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Fichier de configuration YAML (surcharge les paramètres par défaut).",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Générer les visualisations de debug (density maps, segments).",
    ),
    save_intermediates: bool = typer.Option(
        False,
        "--save-intermediates",
        help="Sauvegarder les résultats intermédiaires (.npy, .png).",
    ),
    voxel_size: Optional[float] = typer.Option(
        None,
        "--voxel-size",
        help="Surcharger la taille du voxel de downsampling (mètres).",
        min=0.001,
    ),
    slice_height: Optional[float] = typer.Option(
        None,
        "--slice-height",
        help="Surcharger la hauteur de coupe principale (mètres relatifs au sol).",
        min=0.1,
    ),
    qa_report: Optional[Path] = typer.Option(
        None,
        "--qa-report",
        help="Chemin du fichier JSON de rapport QA. Si omis, pas de fichier JSON produit.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Activer les logs DEBUG.",
    ),
) -> None:
    """Transforme un nuage de points 3D en plan 2D DXF.

    Exemples :

        scan2plan process mon_scan.e57

        scan2plan process mon_scan.e57 --output plan.dxf --debug

        scan2plan process mon_scan.las --config my_params.yaml --voxel-size 0.01

        scan2plan process mon_scan.e57 --qa-report rapport_qa.json
    """
    _configure_logging(verbose)

    from scan2plan.config import ScanConfig
    from scan2plan.pipeline import Scan2PlanPipeline

    # Chemin de sortie par défaut
    if output_file is None:
        output_file = input_file.with_suffix(".dxf")

    # Chargement de la configuration
    config = ScanConfig(user_config_path=config_file)

    # Surcharges CLI
    if voxel_size is not None:
        config._data["preprocessing"]["voxel_size"] = voxel_size
    if slice_height is not None:
        heights = config._data["slicing"]["heights"]
        if len(heights) > 1:
            heights[1] = slice_height
        else:
            heights[0] = slice_height

    # Exécution du pipeline
    pipeline = Scan2PlanPipeline(config)
    result = pipeline.run(
        input_path=input_file,
        output_path=output_file,
        save_intermediates=save_intermediates,
        debug_visualizations=debug,
    )

    # Affichage du résumé et du rapport QA
    typer.echo(result.summary())
    _display_qa(result)
    _display_pipeline_warnings(result)

    # Export du rapport QA en JSON si demandé
    if qa_report is not None and result.qa_report is not None:
        from scan2plan.qa.validator import generate_qa_report
        generate_qa_report(result.qa_report, qa_report)
        typer.echo(f"\nRapport QA écrit : {qa_report.with_suffix('.json')}")

    # Code de retour
    if not result.success:
        raise typer.Exit(code=1)


@app.command()
def info(
    input_file: Path = typer.Argument(
        ...,
        help="Fichier nuage de points (.e57, .las, .laz).",
        exists=True,
        readable=True,
    ),
) -> None:
    """Affiche les informations d'un fichier nuage de points.

    Exemple :

        scan2plan info mon_scan.e57
    """
    _configure_logging(verbose=False)

    from scan2plan.io.readers import read_point_cloud

    try:
        points = read_point_cloud(input_file)
    except Exception as exc:
        typer.echo(f"Erreur lors de la lecture : {exc}", err=True)
        raise typer.Exit(code=1) from None

    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)

    typer.echo(f"Fichier  : {input_file}")
    typer.echo(f"Format   : {input_file.suffix.upper().lstrip('.')}")
    typer.echo(f"Points   : {len(points):,}")
    typer.echo("Bounding box :")
    typer.echo(f"  X : [{x_min:.3f}, {x_max:.3f}] m  (largeur {x_max - x_min:.3f} m)")
    typer.echo(f"  Y : [{y_min:.3f}, {y_max:.3f}] m  (profondeur {y_max - y_min:.3f} m)")
    typer.echo(f"  Z : [{z_min:.3f}, {z_max:.3f}] m  (hauteur {z_max - z_min:.3f} m)")


def _display_qa(result: "Any") -> None:
    """Affiche le score QA et ses avertissements.

    Args:
        result: PipelineResult avec qa_report optionnel.
    """
    if result.qa_report is None:
        return
    typer.echo(f"\nScore QA : {result.qa_score:.0f}/100")
    for w in result.qa_report.warnings:
        typer.echo(f"  [!] {w}")


def _display_pipeline_warnings(result: "Any") -> None:
    """Affiche les avertissements pipeline non déjà couverts par le QA.

    Args:
        result: PipelineResult avec warnings et qa_report optionnel.
    """
    qa_warnings: set[str] = set()
    if result.qa_report is not None:
        qa_warnings = set(result.qa_report.warnings)
    pipeline_warnings = [w for w in result.warnings if w not in qa_warnings]
    if not pipeline_warnings:
        return
    typer.echo("\nAvertissements pipeline :")
    for w in pipeline_warnings:
        typer.echo(f"  [!] {w}")


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
        stream=sys.stderr,
    )
