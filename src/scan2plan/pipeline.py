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
from scan2plan.detection.multi_slice_filter import (
    classify_segments,
    get_door_candidates,
    get_window_candidates,
    match_segments_across_slices,
)
from scan2plan.detection.openings import detect_all_openings
from scan2plan.detection.orientation import detect_dominant_orientations
from scan2plan.detection.segment_fusion import fuse_collinear_segments
from scan2plan.io.readers import read_point_cloud
from scan2plan.io.writers import export_dxf_v1
from scan2plan.preprocessing.downsampling import voxel_downsample
from scan2plan.preprocessing.floor_ceiling import (
    detect_ceiling,
    detect_floor,
    filter_vertical_range,
)
from scan2plan.preprocessing.outlier_removal import remove_statistical_outliers
from scan2plan.qa.metrics import QAReport
from scan2plan.qa.validator import validate_plan
from scan2plan.slicing.density_map import DensityMapResult, create_density_map
from scan2plan.slicing.slicer import extract_multi_slices
from scan2plan.vectorization.regularization import (
    align_parallel_segments,
    regularize_segments,
)
from scan2plan.vectorization.topology import WallGraph, build_wall_graph, clean_topology

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
        num_segments_final: Segments après régularisation et topologie.
        num_rooms_detected: Pièces fermées détectées dans le graphe.
        num_openings: Ouvertures détectées (portes + fenêtres).
        floor_z: Altitude du sol détecté (mètres).
        ceiling_z: Altitude du plafond détecté (mètres).
        qa_score: Score QA de 0 à 100.
        qa_report: Rapport QA complet (None si non disponible).
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
    num_segments_final: int = 0
    num_rooms_detected: int = 0
    num_openings: int = 0
    floor_z: float = 0.0
    ceiling_z: float = 0.0
    qa_score: float = 0.0
    qa_report: QAReport | None = None
    execution_time_seconds: float = 0.0
    success: bool = False
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Résumé lisible du résultat du pipeline."""
        status = "OK" if self.success else "ECHEC"
        lines = [
            f"[{status}] {self.input_path.name} → {self.output_path.name}",
            f"  Points : {self.num_points_original:,} → {self.num_points_after_preprocessing:,} "
            f"(après prétraitement)",
            f"  Slice  : {self.num_points_in_slice:,} points",
            f"  Segments : {self.num_segments_detected} détectés → "
            f"{self.num_segments_after_fusion} après fusion → "
            f"{self.num_segments_final} après topologie",
            f"  Pièces : {self.num_rooms_detected}  |  Ouvertures : {self.num_openings}",
            f"  Sol/Plafond : {self.floor_z:.3f} m / {self.ceiling_z:.3f} m",
            f"  Score QA : {self.qa_score:.0f}/100",
            f"  Durée  : {self.execution_time_seconds:.1f} s",
        ]
        return "\n".join(lines)


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
        """Exécute le pipeline complet de Scan2Plan V1.

        Étapes :
        1. Lecture du nuage de points.
        2. Voxel downsampling.
        3. Statistical Outlier Removal.
        4. Détection du sol par RANSAC.
        5. Détection du plafond par RANSAC.
        6. Filtrage vertical.
        7. Extraction de 3 slices (high/mid/low).
        8. Density map + binarisation + Hough pour chaque slice.
        9. Filtrage multi-slice : murs vs mobilier vs ouvertures.
        10. Fusion de segments colinéaires.
        11. Détection des orientations dominantes + régularisation angulaire.
        12. Détection des ouvertures (portes/fenêtres).
        13. Reconstruction topologique (graphe de murs + intersections).
        14. Export DXF V1 multi-calques.
        15. Contrôle qualité automatique.

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

        self.logger.info("=== Scan2Plan — démarrage du pipeline V1 ===")
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

            # ---- Étape 7 : Multi-slices ------------------------------------
            cfg_sl = self.config.slicing
            slices_xy: dict[str, np.ndarray] = self._step(
                "Extraction multi-slices",
                lambda: extract_multi_slices(
                    points,
                    heights=cfg_sl.heights,
                    thickness=cfg_sl.thickness,
                    floor_z=z_floor,
                ),
            )
            mid_slice = slices_xy.get("mid", slices_xy.get("high", np.empty((0, 2))))
            result.num_points_in_slice = len(mid_slice)

            # ---- Étape 8 : Density map + Hough pour chaque slice -----------
            cfg_h = self.config.hough
            cfg_morph = self.config.morphology
            segments_by_slice: dict[str, list[DetectedSegment]] = {}
            density_maps: dict[str, DensityMapResult] = {}
            binary_images: dict[str, np.ndarray] = {}
            all_detected: list[DetectedSegment] = []

            for label, slice_xy in slices_xy.items():
                if len(slice_xy) == 0:
                    segments_by_slice[label] = []
                    continue

                dmap: DensityMapResult = self._step(
                    f"Density map [{label}]",
                    lambda s=slice_xy: create_density_map(s, self.config.density_map.resolution),
                )
                density_maps[label] = dmap

                if save_intermediates and label == "mid":
                    self._save_density_map_png(dmap, output_path, f"density_map_{label}")

                binary_raw = self._step(
                    f"Binarisation [{label}]",
                    lambda d=dmap: binarize_density_map(d.image),
                )
                binary = self._step(
                    f"Morphologie [{label}]",
                    lambda b=binary_raw: morphological_cleanup(
                        b,
                        cfg_morph.kernel_size,
                        cfg_morph.close_iterations,
                        cfg_morph.open_iterations,
                    ),
                )
                binary_images[label] = binary

                segs: list[DetectedSegment] = self._step(
                    f"Hough [{label}]",
                    lambda b=binary, d=dmap, lbl=label: detect_lines_hough(
                        b,
                        d,
                        rho=cfg_h.rho,
                        theta_deg=cfg_h.theta_deg,
                        threshold=cfg_h.threshold,
                        min_line_length=cfg_h.min_line_length,
                        max_line_gap=cfg_h.max_line_gap,
                        source_slice=lbl,
                    ),
                )
                segments_by_slice[label] = segs
                all_detected.extend(segs)

            result.num_segments_detected = len(all_detected)

            if not any(segments_by_slice.get("high", [])):
                msg = "Aucun segment détecté en slice haute — résultat potentiellement incomplet."
                self.logger.warning(msg)
                result.warnings.append(msg)

            # ---- Étape 9 : Filtrage multi-slice ----------------------------
            cfg_msf = self.config._data.get("multi_slice_filter", {})
            angle_tol = float(cfg_msf.get("angle_tolerance_deg", 5.0))
            dist_tol = float(cfg_msf.get("distance_tolerance", 0.10))

            matches = self._step(
                "Recoupement multi-slice",
                lambda: match_segments_across_slices(
                    segments_by_slice,
                    angle_tolerance_deg=angle_tol,
                    distance_tolerance=dist_tol,
                ),
            )
            wall_segments: list[DetectedSegment] = self._step(
                "Classification segments",
                lambda: classify_segments(matches),
            )

            self._door_candidates = get_door_candidates(matches)
            self._window_candidates = get_window_candidates(matches)
            self.logger.info(
                "Ouvertures candidates : %d portes, %d fenêtres.",
                len(self._door_candidates),
                len(self._window_candidates),
            )

            if not wall_segments:
                msg = "Aucun segment de mur confirmé après filtrage multi-slice."
                self.logger.error(msg)
                result.warnings.append(msg)
                result.execution_time_seconds = time.monotonic() - t_start
                return result

            # ---- Étape 10 : Fusion -----------------------------------------
            cfg_sf = self.config.segment_fusion
            fused: list[DetectedSegment] = self._step(
                "Fusion segments",
                lambda: fuse_collinear_segments(
                    wall_segments,
                    cfg_sf.angle_tolerance_deg,
                    cfg_sf.perpendicular_dist,
                    cfg_sf.max_gap,
                ),
            )
            result.num_segments_after_fusion = len(fused)

            # ---- Étape 11 : Orientations dominantes + régularisation --------
            dominant_angles: list[float] = self._step(
                "Orientations dominantes",
                lambda: detect_dominant_orientations(fused),
            )

            cfg_reg = self.config._data.get("regularization", {})
            snap_tol = float(cfg_reg.get("snap_tolerance_deg", 5.0))

            regularized: list[DetectedSegment] = self._step(
                "Régularisation angulaire",
                lambda: regularize_segments(fused, dominant_angles, snap_tol),
            )
            regularized = self._step(
                "Alignement parallèle",
                lambda: align_parallel_segments(regularized, dominant_angles),
            )

            # ---- Étape 12 : Détection des ouvertures -----------------------
            cfg_op = self.config._data.get("openings", {})
            openings = self._step(
                "Détection ouvertures",
                lambda: detect_all_openings(
                    self._door_candidates + self._window_candidates,
                    density_maps,
                    binary_images,
                    cfg_op,
                ),
            )
            result.num_openings = len(openings)

            # ---- Étape 13 : Reconstruction topologique ---------------------
            cfg_top = self.config._data.get("topology", {})
            inter_dist = float(cfg_top.get("intersection_distance", 0.05))
            min_seg = float(cfg_top.get("min_segment_length", 0.10))

            wall_graph: WallGraph = self._step(
                "Graphe topologique",
                lambda: build_wall_graph(
                    regularized,
                    openings,
                    intersection_distance=inter_dist,
                    min_segment_length=min_seg,
                ),
            )
            wall_graph = self._step(
                "Nettoyage topologie",
                lambda: clean_topology(wall_graph, min_segment_length=min_seg),
            )

            result.num_segments_final = len(wall_graph.segments)

            if debug_visualizations:
                mid_slice_xy = slices_xy.get("mid", np.empty((0, 2)))
                if len(mid_slice_xy) > 0:
                    dmap_mid = create_density_map(mid_slice_xy, self.config.density_map.resolution)
                    self._save_segment_visualization(dmap_mid, wall_graph.segments, output_path)

            # ---- Étape 14 : Export DXF V1 ----------------------------------
            final_path = self._step(
                "Export DXF V1",
                lambda: export_dxf_v1(
                    wall_graph,
                    openings,
                    output_path,
                    config=self.config,
                ),
            )
            result.output_path = final_path

            # ---- Étape 15 : Contrôle qualité --------------------------------
            qa: QAReport = self._step(
                "Contrôle qualité",
                lambda: validate_plan(wall_graph, openings),
            )
            result.qa_score = qa.score
            result.qa_report = qa
            result.num_rooms_detected = qa.num_rooms_detected
            result.warnings.extend(qa.warnings)

            self.logger.info(qa.summary())
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
