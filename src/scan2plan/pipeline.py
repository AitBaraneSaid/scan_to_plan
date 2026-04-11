"""Orchestrateur du pipeline Scan2Plan complet."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scan2plan.config import ScanConfig
from scan2plan.detection.line_detection import DetectedSegment, detect_lines_hough
from scan2plan.detection.morphology import binarize_density_map, morphological_cleanup
from scan2plan.detection.segment_fusion import fuse_collinear_segments
from scan2plan.io.readers import read_point_cloud
from scan2plan.io.writers import export_dxf
from scan2plan.preprocessing.downsampling import voxel_downsample
from scan2plan.preprocessing.floor_ceiling import (
    detect_ceiling,
    detect_floor,
    filter_vertical_range,
)
from scan2plan.preprocessing.outlier_removal import remove_statistical_outliers
from scan2plan.slicing.density_map import DensityMapResult, create_density_map
from scan2plan.slicing.slicer import extract_slice

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Métriques et statut d'exécution du pipeline.

    Attributes:
        input_path: Fichier nuage de points traité.
        output_path: Fichier DXF produit.
        num_points_original: Points lus depuis le fichier source.
        num_points_after_preprocessing: Points après downsampling + SOR + filtrage vertical.
        num_points_in_slice: Points dans la slice médiane utilisée pour la détection.
        num_segments_detected: Segments bruts issus de Hough.
        num_segments_after_fusion: Segments après fusion colinéaire.
        floor_z: Altitude du sol détecté (mètres).
        ceiling_z: Altitude du plafond détecté (mètres).
        execution_time_seconds: Durée totale d'exécution.
        success: True si le pipeline s'est terminé avec succès.
        warnings: Liste de messages d'avertissement émis pendant l'exécution.
    """

    input_path: Path
    output_path: Path
    num_points_original: int = 0
    num_points_after_preprocessing: int = 0
    num_points_in_slice: int = 0
    num_segments_detected: int = 0
    num_segments_after_fusion: int = 0
    floor_z: float = 0.0
    ceiling_z: float = 0.0
    execution_time_seconds: float = 0.0
    success: bool = False
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Résumé lisible du résultat du pipeline."""
        status = "OK" if self.success else "ECHEC"
        return (
            f"[{status}] {self.input_path.name} → {self.output_path.name}\n"
            f"  Points : {self.num_points_original:,} → {self.num_points_after_preprocessing:,} "
            f"(après prétraitement)\n"
            f"  Slice  : {self.num_points_in_slice:,} points\n"
            f"  Segments : {self.num_segments_detected} détectés → "
            f"{self.num_segments_after_fusion} après fusion\n"
            f"  Sol/Plafond : {self.floor_z:.3f} m / {self.ceiling_z:.3f} m\n"
            f"  Durée  : {self.execution_time_seconds:.1f} s"
        )


class Scan2PlanPipeline:
    """Orchestrateur du pipeline Scan2Plan complet.

    Coordonne l'exécution séquentielle de toutes les étapes du pipeline,
    gère le passage des résultats intermédiaires, le logging, et les
    sauvegardes optionnelles.

    Args:
        config: Configuration du pipeline (paramètres YAML + surcharges).

    Example:
        >>> pipeline = Scan2PlanPipeline(ScanConfig())
        >>> result = pipeline.run(Path("scan.e57"), Path("plan.dxf"))
        >>> result.success
        True
    """

    def __init__(self, config: ScanConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        input_path: Path,
        output_path: Path,
        save_intermediates: bool = False,
        debug_visualizations: bool = False,
    ) -> PipelineResult:
        """Exécute le pipeline complet de Scan2Plan.

        Étapes :
        1. Lecture du nuage de points.
        2. Voxel downsampling.
        3. Statistical Outlier Removal.
        4. Détection du sol par RANSAC.
        5. Détection du plafond par RANSAC.
        6. Filtrage vertical.
        7. Extraction de slice médiane (~1.10 m).
        8. Création de density map.
        9. Binarisation + nettoyage morphologique.
        10. Détection de segments par Hough.
        11. Fusion de segments colinéaires.
        12. Export DXF.

        Args:
            input_path: Chemin du fichier nuage de points (E57, LAS, LAZ).
            output_path: Chemin du fichier DXF de sortie.
            save_intermediates: Si True, sauvegarde les density maps en PNG
                et les nuages intermédiaires en .npy dans le répertoire de sortie.
            debug_visualizations: Si True, génère et sauvegarde les plots
                matplotlib de chaque étape (nécessite matplotlib).

        Returns:
            ``PipelineResult`` avec toutes les métriques d'exécution.
        """
        t_start = time.monotonic()
        result = PipelineResult(input_path=input_path, output_path=output_path)

        self.logger.info("=== Scan2Plan — démarrage du pipeline MVP ===")
        self.logger.info("Entrée  : %s", input_path)
        self.logger.info("Sortie  : %s", output_path)

        try:
            # ---- Étape 1 : Lecture ----------------------------------------
            points = self._step("Lecture", lambda: read_point_cloud(input_path))
            result.num_points_original = len(points)

            # ---- Étape 2 : Downsampling ------------------------------------
            cfg_pre = self.config.preprocessing
            points = self._step(
                "Downsampling",
                lambda: voxel_downsample(points, cfg_pre.voxel_size),
            )

            # ---- Étape 3 : SOR ---------------------------------------------
            points = self._step(
                "Outlier removal",
                lambda: remove_statistical_outliers(
                    points, cfg_pre.sor_k_neighbors, cfg_pre.sor_std_ratio
                ),
            )

            # ---- Étape 4 : Sol ---------------------------------------------
            cfg_fc = self.config.floor_ceiling
            z_floor, _ = self._step(
                "Détection sol",
                lambda: detect_floor(
                    points,
                    distance_threshold=cfg_fc.ransac_distance,
                    num_iterations=cfg_fc.ransac_iterations,
                    normal_tolerance_deg=cfg_fc.normal_tolerance_deg,
                ),
            )
            result.floor_z = z_floor

            # ---- Étape 5 : Plafond -----------------------------------------
            z_ceiling, _ = self._step(
                "Détection plafond",
                lambda: detect_ceiling(
                    points,
                    floor_z=z_floor,
                    distance_threshold=cfg_fc.ransac_distance,
                    num_iterations=cfg_fc.ransac_iterations,
                    normal_tolerance_deg=cfg_fc.normal_tolerance_deg,
                ),
            )
            result.ceiling_z = z_ceiling

            # ---- Étape 6 : Filtrage vertical --------------------------------
            points = self._step(
                "Filtrage vertical",
                lambda: filter_vertical_range(points, z_floor, z_ceiling),
            )
            result.num_points_after_preprocessing = len(points)

            if save_intermediates:
                self._save_npy(points, output_path, "points_preprocessed")

            # ---- Étape 7 : Slice -------------------------------------------
            cfg_sl = self.config.slicing
            median_height = cfg_sl.heights[1] if len(cfg_sl.heights) > 1 else 1.10
            slice_xy = self._step(
                f"Slice h={median_height:.2f} m",
                lambda: extract_slice(
                    points,
                    height=median_height,
                    thickness=cfg_sl.thickness,
                    floor_z=z_floor,
                ),
            )
            result.num_points_in_slice = len(slice_xy)

            # ---- Étape 8 : Density map -------------------------------------
            dmap: DensityMapResult = self._step(
                "Density map",
                lambda: create_density_map(slice_xy, self.config.density_map.resolution),
            )

            if save_intermediates:
                self._save_density_map_png(dmap, output_path, "density_map")

            # ---- Étape 9 : Binarisation + morphologie ----------------------
            cfg_morph = self.config.morphology
            binary_raw = self._step("Binarisation Otsu", lambda: binarize_density_map(dmap.image))
            binary = self._step(
                "Nettoyage morphologique",
                lambda: morphological_cleanup(
                    binary_raw,
                    cfg_morph.kernel_size,
                    cfg_morph.close_iterations,
                    cfg_morph.open_iterations,
                ),
            )

            if save_intermediates:
                self._save_image_png(binary, output_path, "binary_cleaned")

            # ---- Étape 10 : Hough ------------------------------------------
            cfg_h = self.config.hough
            detected: list[DetectedSegment] = self._step(
                "Hough",
                lambda: detect_lines_hough(
                    binary,
                    dmap,
                    rho=cfg_h.rho,
                    theta_deg=cfg_h.theta_deg,
                    threshold=cfg_h.threshold,
                    min_line_length=cfg_h.min_line_length,
                    max_line_gap=cfg_h.max_line_gap,
                    source_slice="mid",
                ),
            )
            result.num_segments_detected = len(detected)

            if not detected:
                msg = "Aucun segment détecté par Hough — le DXF ne sera pas produit."
                self.logger.error(msg)
                result.warnings.append(msg)
                result.execution_time_seconds = time.monotonic() - t_start
                return result

            # ---- Étape 11 : Fusion -----------------------------------------
            cfg_sf = self.config.segment_fusion
            fused: list[DetectedSegment] = self._step(
                "Fusion segments",
                lambda: fuse_collinear_segments(
                    detected,
                    cfg_sf.angle_tolerance_deg,
                    cfg_sf.perpendicular_dist,
                    cfg_sf.max_gap,
                ),
            )
            result.num_segments_after_fusion = len(fused)

            if debug_visualizations:
                self._save_segment_visualization(dmap, fused, output_path)

            # ---- Étape 12 : Export DXF -------------------------------------
            final_path = self._step(
                "Export DXF",
                lambda: export_dxf(
                    fused,
                    output_path,
                    version=self.config.dxf.version,
                    layer_config=self.config.dxf.layers,
                ),
            )
            result.output_path = final_path
            result.success = True

        except Exception as exc:
            self.logger.error("Pipeline échoué : %s", exc, exc_info=True)
            result.warnings.append(f"Erreur fatale : {exc}")

        result.execution_time_seconds = time.monotonic() - t_start
        self.logger.info(result.summary())
        return result

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _step(self, name: str, fn: "Any") -> "Any":
        """Exécute une étape en loggant le début, la fin, et le temps.

        Args:
            name: Nom lisible de l'étape (pour le logging).
            fn: Callable sans argument retournant le résultat de l'étape.

        Returns:
            Résultat de ``fn()``.

        Raises:
            Re-lève toute exception levée par ``fn``.
        """
        self.logger.info("[%s] démarrage…", name)
        t0 = time.monotonic()
        result = fn()
        elapsed = time.monotonic() - t0
        self.logger.info("[%s] terminé en %.2f s.", name, elapsed)
        return result

    def _save_npy(self, array: np.ndarray, output_path: Path, stem: str) -> None:
        """Sauvegarde un tableau NumPy dans le répertoire de sortie."""
        out = output_path.parent / f"{output_path.stem}_{stem}.npy"
        np.save(str(out), array)
        self.logger.debug("Intermédiaire sauvegardé : %s", out)

    def _save_density_map_png(
        self, dmap: DensityMapResult, output_path: Path, stem: str
    ) -> None:
        """Sauvegarde la density map en PNG (nécessite matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            from scan2plan.utils.visualization import save_figure
            fig, ax = plt.subplots()
            ax.imshow(dmap.image, cmap="hot", origin="upper")
            ax.set_title(stem)
            out = output_path.parent / f"{output_path.stem}_{stem}.png"
            save_figure(fig, out)
            plt.close(fig)
        except Exception as exc:
            self.logger.warning("Impossible de sauvegarder la density map : %s", exc)

    def _save_image_png(self, image: np.ndarray, output_path: Path, stem: str) -> None:
        """Sauvegarde une image binaire en PNG (nécessite matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            from scan2plan.utils.visualization import save_figure
            fig, ax = plt.subplots()
            ax.imshow(image, cmap="gray", origin="upper")
            ax.set_title(stem)
            out = output_path.parent / f"{output_path.stem}_{stem}.png"
            save_figure(fig, out)
            plt.close(fig)
        except Exception as exc:
            self.logger.warning("Impossible de sauvegarder l'image binaire : %s", exc)

    def _save_segment_visualization(
        self,
        dmap: DensityMapResult,
        segments: "list[DetectedSegment]",
        output_path: Path,
    ) -> None:
        """Sauvegarde la visualisation des segments détectés en PNG."""
        try:
            import matplotlib.pyplot as plt
            from scan2plan.utils.visualization import plot_detected_segments, save_figure
            plot_detected_segments(dmap, segments, "Segments détectés")
            fig = plt.gcf()
            out = output_path.parent / f"{output_path.stem}_segments.png"
            save_figure(fig, out)
            plt.close(fig)
        except Exception as exc:
            self.logger.warning("Impossible de sauvegarder la visualisation segments : %s", exc)
