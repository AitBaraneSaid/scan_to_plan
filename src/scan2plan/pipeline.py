"""Orchestrateur du pipeline Scan2Plan complet — version post-traitement minimaliste.

Le pipeline est découpé en deux parties :
- Détection (étapes 1-7) : lecture, prétraitement, slicing, Hough, multi-slice.
  Ces étapes sont inchangées depuis V1.
- Post-traitement (étapes 8-13) : micro-fusion, nettoyage, régularisation angulaire
  pure, face pairing sans médiane, snap/close léger, export DXF faces directes.
  Ces étapes remplacent le post-traitement destructeur précédent.

Bilan attendu sur un scan typique :
  Ancien : 232 Hough → 110 fusion → 70 régul → 37 topo → DXF 73
  Nouveau : 232 Hough → 180 micro-fusion → 170 cleanup → 170 régul → 168 topo → DXF 170
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from scan2plan.config import ScanConfig
from scan2plan.detection.line_detection import DetectedSegment, detect_lines_hough
from scan2plan.detection.morphology import binarize_density_map, morphological_cleanup
from scan2plan.detection.multi_slice_filter import (
    classify_segments,
    match_segments_across_slices,
)
from scan2plan.detection.segment_fusion import fuse_collinear_segments
from scan2plan.io.readers import read_point_cloud
from scan2plan.preprocessing.downsampling import voxel_downsample
from scan2plan.preprocessing.floor_ceiling import (
    detect_ceiling,
    detect_floor,
    detect_floor_rdc,
    filter_vertical_range,
)
from scan2plan.preprocessing.outlier_removal import remove_statistical_outliers
from scan2plan.qa.metrics import QAReport
from scan2plan.slicing.density_map import DensityMapResult, create_density_map
from scan2plan.slicing.slicer import extract_multi_slices
from scan2plan.vectorization.angular_regularization import (
    detect_dominant_orientations,
    snap_angles,
)
from scan2plan.vectorization.light_topology import apply_light_topology
from scan2plan.vectorization.wall_pairing import (
    FacePair,
    PairingConfig,
    Segment,
    find_wall_pairs,
    pair_wall_faces,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Résultat du pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Métriques et statut d'exécution du pipeline.

    Attributes:
        input_path: Fichier nuage de points traité.
        output_path: Fichier DXF produit.
        num_points_original: Points lus depuis le fichier source.
        num_points_after_preprocessing: Points après downsampling + SOR + filtrage.
        num_points_in_slice: Points dans la slice médiane.
        num_segments_hough: Segments bruts issus de Hough (toutes slices).
        num_segments_after_multifilter: Segments conservés après filtre multi-slice.
        num_segments_after_microfusion: Segments après micro-fusion (gap ≤ 5 cm).
        num_segments_after_cleanup: Segments après suppression des parasites.
        num_segments_after_regularization: Segments après régularisation angulaire.
        num_wall_pairs: Paires de faces détectées par face pairing.
        num_segments_after_topology: Segments après snap léger + fermeture coins.
        num_openings: Ouvertures détectées dans le DXF final.
        floor_z: Altitude du sol détecté (mètres).
        ceiling_z: Altitude du plafond détecté (mètres).
        dominant_angles_deg: Orientations dominantes détectées (degrés).
        qa_score: Score QA de 0 à 100.
        qa_report: Rapport QA complet.
        execution_time_seconds: Durée totale d'exécution.
        success: True si le pipeline s'est terminé avec succès.
        warnings: Messages d'avertissement émis pendant l'exécution.
    """

    input_path: Path
    output_path: Path
    num_points_original: int = 0
    num_points_after_preprocessing: int = 0
    num_points_in_slice: int = 0
    num_segments_hough: int = 0
    num_segments_after_multifilter: int = 0
    num_segments_after_microfusion: int = 0
    num_segments_after_cleanup: int = 0
    num_segments_after_regularization: int = 0
    num_wall_pairs: int = 0
    num_segments_after_topology: int = 0
    num_openings: int = 0
    floor_z: float = 0.0
    ceiling_z: float = 0.0
    dominant_angles_deg: list[float] = field(default_factory=list)
    qa_score: float = 0.0
    qa_report: QAReport | None = None
    execution_time_seconds: float = 0.0
    success: bool = False
    warnings: list[str] = field(default_factory=list)

    # Compat ancien code (propriétés de lecture seule)
    @property
    def num_segments_detected(self) -> int:
        """Alias vers num_segments_hough (compatibilité)."""
        return self.num_segments_hough

    @property
    def num_segments_after_fusion(self) -> int:
        """Alias vers num_segments_after_microfusion (compatibilité)."""
        return self.num_segments_after_microfusion

    @property
    def num_segments_after_pairing(self) -> int:
        """Alias vers num_segments_after_topology (compatibilité)."""
        return self.num_segments_after_topology

    @property
    def num_segments_final(self) -> int:
        """Alias vers num_segments_after_topology (compatibilité)."""
        return self.num_segments_after_topology

    @property
    def num_rooms_detected(self) -> int:
        """Nombre de pièces détectées (depuis QAReport)."""
        return self.qa_report.num_rooms_detected if self.qa_report else 0

    def summary(self) -> str:
        """Résumé lisible du résultat du pipeline."""
        status = "OK" if self.success else "ECHEC"
        retention = (
            f"{self.num_segments_after_topology / self.num_segments_hough * 100:.0f}%"
            if self.num_segments_hough else "N/A"
        )
        lines = [
            f"[{status}] {self.input_path.name} -> {self.output_path.name}",
            f"  Points : {self.num_points_original:,} -> {self.num_points_after_preprocessing:,}",
            f"  Hough     : {self.num_segments_hough}",
            f"  Multifiltr: {self.num_segments_after_multifilter}",
            f"  Micro-fus : {self.num_segments_after_microfusion}",
            f"  Cleanup   : {self.num_segments_after_cleanup}",
            f"  Régul     : {self.num_segments_after_regularization}",
            f"  Pairing   : {self.num_wall_pairs} paires",
            f"  Topologie : {self.num_segments_after_topology}  (rétention {retention})",
            f"  Ouvertures: {self.num_openings}",
            f"  Sol/Plafond : {self.floor_z:.3f} m / {self.ceiling_z:.3f} m",
            f"  Score QA : {self.qa_score:.0f}/100",
            f"  Durée    : {self.execution_time_seconds:.1f} s",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


class Scan2PlanPipeline:
    """Orchestrateur du pipeline Scan2Plan complet.

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
        self.wall_pairs: list[Any] = []  # FacePair issus du dernier run()

    def run(
        self,
        input_path: Path,
        output_path: Path,
        save_intermediates: bool = False,
        debug_visualizations: bool = False,
        floor_z_override: float | None = None,
        ceiling_z_override: float | None = None,
        xy_bounds: tuple[float | None, float | None, float | None, float | None] = (
            None, None, None, None
        ),
    ) -> PipelineResult:
        """Exécute le pipeline complet Scan2Plan.

        Étapes de détection (1-7) :
        1. Lecture du nuage de points.
        2. Voxel downsampling.
        3. Statistical Outlier Removal.
        4. Détection sol/plafond (auto ou override).
        5. Filtrage vertical.
        6. Extraction multi-slices (high/mid/low).
        7. Density map + binarisation + Hough par slice → filtre multi-slice.

        Étapes de post-traitement (8-13) :
        8. Micro-fusion (gap ≤ 5 cm, same direction < 3°).
        9. Nettoyage parasites (segments < 10 cm supprimés).
        10. Régularisation angulaire pure (snap sans fusion ni comptage).
        11. Face pairing (association faces sans médiane).
        12. Snap léger (3 cm) + fermeture coins (8 cm, > 60°).
        13. Export DXF direct des faces avec calques métier.

        Args:
            input_path: Chemin du nuage de points (E57, LAS, LAZ).
            output_path: Chemin du DXF de sortie.
            save_intermediates: Sauvegarde les nuages intermédiaires (.npy).
            debug_visualizations: Génère les 8 images de debug PNG.
            floor_z_override: Altitude sol imposée (mètres) ou None.
            ceiling_z_override: Altitude plafond imposée (mètres) ou None.
            xy_bounds: Recadrage XY (x_min, x_max, y_min, y_max).

        Returns:
            ``PipelineResult`` avec toutes les métriques.
        """
        t_start = time.monotonic()
        result = PipelineResult(input_path=input_path, output_path=output_path)
        logger.info("=== Scan2Plan — démarrage pipeline minimaliste ===")
        logger.info("Entrée : %s", input_path)
        logger.info("Sortie : %s", output_path)

        debug_dir = (output_path.parent / "debug") if debug_visualizations else None
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)

        try:
            # ----------------------------------------------------------------
            # Étapes 1-7 : Détection
            # ----------------------------------------------------------------
            _, _, density_maps, wall_segments = self._run_detection(
                input_path, output_path, result,
                save_intermediates, xy_bounds,
                floor_z_override, ceiling_z_override,
            )

            if wall_segments is None:
                result.execution_time_seconds = time.monotonic() - t_start
                return result

            # ----------------------------------------------------------------
            # Étapes 8-13 : Post-traitement
            # ----------------------------------------------------------------
            ref_dmap = self._ref_density_map(density_maps)
            final_segments, face_pairs = self._run_post_detection(
                wall_segments, result, debug_dir, ref_dmap,
            )

            # ----------------------------------------------------------------
            # Étape 13 : Export DXF faces directes
            # ----------------------------------------------------------------
            from scan2plan.io.dxf_face_export import export_dxf_faces

            final_path = self._step(
                "Export DXF faces",
                lambda: export_dxf_faces(final_segments, face_pairs, output_path),
            )
            result.output_path = final_path

            if debug_dir is not None and ref_dmap is not None:
                self._save_debug_image(
                    debug_dir, "08_final", final_segments,
                    ref_dmap, f"08 — Final ({len(final_segments)} seg, "
                    f"{len(face_pairs)} paires)",
                    face_pairs=face_pairs,
                )

            # ----------------------------------------------------------------
            # Contrôle qualité
            # ----------------------------------------------------------------
            result.num_openings = self._count_openings_in_dxf(final_path)
            result.qa_score, result.qa_report = self._run_qa(
                final_segments, face_pairs,
            )
            result.success = True

        except Exception as exc:
            logger.error("Pipeline échoué : %s", exc, exc_info=True)
            result.warnings.append(f"Erreur fatale : {exc}")

        result.execution_time_seconds = time.monotonic() - t_start
        logger.info(result.summary())
        return result

    # ------------------------------------------------------------------
    # Bloc détection (étapes 1-7)
    # ------------------------------------------------------------------

    def _run_detection(
        self,
        input_path: Path,
        output_path: Path,
        result: PipelineResult,
        save_intermediates: bool,
        xy_bounds: tuple[float | None, float | None, float | None, float | None],
        floor_z_override: float | None,
        ceiling_z_override: float | None,
    ) -> tuple[
        np.ndarray,
        dict[str, np.ndarray],
        dict[str, DensityMapResult],
        list[DetectedSegment] | None,
    ]:
        """Étapes 1-7 : lecture → Hough → multi-slice filter.

        Returns:
            ``(points, slices_xy, density_maps, wall_segments)`` ou
            ``wall_segments=None`` en cas d'échec.
        """
        # 1 — Lecture
        points: np.ndarray = self._step("Lecture", lambda: read_point_cloud(input_path))
        result.num_points_original = len(points)

        x_min, x_max, y_min, y_max = xy_bounds
        if any(v is not None for v in (x_min, x_max, y_min, y_max)):
            points = self._crop_xy(points, x_min, x_max, y_min, y_max)

        # 2-3 — Downsampling + SOR
        cfg_pre = self.config.preprocessing
        points = self._step(
            "Downsampling",
            lambda: voxel_downsample(points, cfg_pre.voxel_size),
        )
        points = self._step(
            "Outlier removal",
            lambda: remove_statistical_outliers(
                points, cfg_pre.sor_k_neighbors, cfg_pre.sor_std_ratio
            ),
        )

        # 4-5 — Sol/plafond + filtrage vertical
        z_floor, z_ceiling = self._detect_floor_ceiling(
            points, floor_z_override, ceiling_z_override
        )
        result.floor_z = z_floor
        result.ceiling_z = z_ceiling

        points = self._step(
            "Filtrage vertical",
            lambda: filter_vertical_range(points, z_floor, z_ceiling),
        )
        result.num_points_after_preprocessing = len(points)

        if save_intermediates:
            self._save_npy(points, output_path, "points_preprocessed")

        # 6 — Multi-slices
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

        # 7 — Density map + Hough par slice
        cfg_h = self.config.hough
        cfg_morph = self.config.morphology
        segments_by_slice: dict[str, list[DetectedSegment]] = {}
        density_maps: dict[str, DensityMapResult] = {}
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
            binary_raw = binarize_density_map(dmap.image)
            binary = morphological_cleanup(
                binary_raw,
                cfg_morph.kernel_size,
                cfg_morph.close_iterations,
                cfg_morph.open_iterations,
            )
            segs: list[DetectedSegment] = self._step(
                f"Hough [{label}]",
                lambda b=binary, d=dmap, lbl=label: detect_lines_hough(
                    b, d,
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

        result.num_segments_hough = len(all_detected)

        if not any(segments_by_slice.get("high", [])):
            msg = "Aucun segment en slice haute — résultat potentiellement incomplet."
            logger.warning(msg)
            result.warnings.append(msg)

        # Multi-slice filter
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
        result.num_segments_after_multifilter = len(wall_segments)

        if not wall_segments:
            msg = "Aucun segment de mur confirmé après filtrage multi-slice."
            logger.error(msg)
            result.warnings.append(msg)
            return points, slices_xy, density_maps, None

        return points, slices_xy, density_maps, wall_segments

    # ------------------------------------------------------------------
    # Bloc post-traitement (étapes 8-12)
    # ------------------------------------------------------------------

    def _run_post_detection(
        self,
        wall_segments: list[DetectedSegment],
        result: PipelineResult,
        debug_dir: Path | None,
        ref_dmap: DensityMapResult | None,
    ) -> tuple[list[DetectedSegment], list[FacePair]]:
        """Étapes 8-12 : micro-fusion → cleanup → régul → pairing → topologie.

        Args:
            wall_segments: Segments murs après filtre multi-slice.
            result: PipelineResult à enrichir.
            debug_dir: Répertoire debug ou None.
            ref_dmap: Density map de référence pour les images de debug.

        Returns:
            ``(final_segments, face_pairs)``
        """
        # 8 — Micro-fusion (gap ≤ 5 cm)
        cfg_sf = self.config.segment_fusion
        micro_fused = self._step(
            "Micro-fusion (gap ≤ 5 cm)",
            lambda: fuse_collinear_segments(
                wall_segments,
                cfg_sf.angle_tolerance_deg,
                cfg_sf.perpendicular_dist,
                max_gap=0.05,   # gap réduit : 5 cm seulement
            ),
        )
        result.num_segments_after_microfusion = len(micro_fused)
        self._debug_img(debug_dir, "03_micro_fusion", micro_fused, ref_dmap,
                        f"03 — Micro-fusion ({len(micro_fused)} seg)")

        # 9 — Nettoyage parasites (< 10 cm)
        min_len = float(
            self.config._data.get("topology", {}).get("min_segment_length", 0.10)
        )
        cleaned = self._step(
            "Nettoyage parasites",
            lambda: [s for s in micro_fused if s.length >= min_len],
        )
        result.num_segments_after_cleanup = len(cleaned)
        self._debug_img(debug_dir, "04_cleanup", cleaned, ref_dmap,
                        f"04 — Cleanup ({len(cleaned)} seg)")

        # 10 — Régularisation angulaire pure
        cfg_reg = self.config._data.get("regularization", {})
        snap_tol = float(cfg_reg.get("snap_tolerance_deg", 5.0))

        dominant_angles = self._step(
            "Orientations dominantes",
            lambda: detect_dominant_orientations(cleaned),
        )
        result.dominant_angles_deg = [
            float(np.degrees(a)) for a in dominant_angles
        ]
        logger.info(
            "Orientations : %s",
            [f"{a:.1f}°" for a in result.dominant_angles_deg],
        )

        regularized = self._step(
            "Régularisation angulaire",
            lambda: snap_angles(cleaned, dominant_angles, tolerance_deg=snap_tol),
        )
        result.num_segments_after_regularization = len(regularized)
        self._debug_img(debug_dir, "05_regularized", regularized, ref_dmap,
                        f"05 — Régularisé ({len(regularized)} seg)")

        # 11 — Face pairing (faces directes, sans médiane)
        pairing_segs = [
            Segment(
                x1=s.x1, y1=s.y1, x2=s.x2, y2=s.y2,
                label="wall",
                confidence=s.confidence,
                source_slice=s.source_slice,
            )
            for s in regularized
        ]
        cfg_wp = self.config.wall_pairing
        pairing_config = PairingConfig(
            angle_tolerance_deg=cfg_wp.angle_tolerance_deg,
            min_distance=cfg_wp.min_distance,
            max_distance=cfg_wp.max_distance,
            min_overlap_abs=cfg_wp.min_overlap_abs,
            min_overlap_ratio=cfg_wp.min_overlap_ratio,
            corridor_margin=cfg_wp.corridor_margin,
            typical_wall_thickness=cfg_wp.typical_wall_thickness,
            min_segment_length=cfg_wp.min_segment_length,
            corridor_intersection_threshold=cfg_wp.corridor_intersection_threshold,
        )

        face_pairing_result = self._step(
            "Face pairing",
            lambda: pair_wall_faces(pairing_segs, pairing_config),
        )
        result.num_wall_pairs = face_pairing_result.num_pairs
        self.wall_pairs = list(face_pairing_result.paired_faces)  # accès public post-run
        self._debug_img(
            debug_dir, "06_paired", regularized, ref_dmap,
            f"06 — Pairé ({result.num_wall_pairs} paires, "
            f"{len(face_pairing_result.unpaired_segments)} simples)",
            face_pairs=list(face_pairing_result.paired_faces),
        )

        # Reconversion FacePair → DetectedSegment pour la topologie
        all_segs_after_pairing = self._face_pairing_to_detected(
            face_pairing_result.paired_faces,
            face_pairing_result.unpaired_segments,
        )

        # 12 — Snap léger + fermeture coins
        cfg_lt = self.config._data.get("light_topology", {})
        snap_tol_m = float(cfg_lt.get("snap_tolerance", 0.03))
        corner_ext = float(cfg_lt.get("corner_max_extension", 0.08))
        corner_ang = float(cfg_lt.get("corner_min_angle_deg", 60.0))

        topo_segs = self._step(
            "Snap + Close corners",
            lambda: apply_light_topology(
                all_segs_after_pairing,
                snap_tolerance=snap_tol_m,
                corner_max_extension=corner_ext,
                corner_min_angle_deg=corner_ang,
            ),
        )
        result.num_segments_after_topology = len(topo_segs)
        self._debug_img(debug_dir, "07_topology", topo_segs, ref_dmap,
                        f"07 — Topologie ({len(topo_segs)} seg)")

        # Re-extraire les FacePair après topologie pour l'export
        # (on utilise les FacePair originaux — la topologie travaille sur des copies)
        face_pairs_for_export = list(face_pairing_result.paired_faces)

        return topo_segs, face_pairs_for_export

    # ------------------------------------------------------------------
    # Helpers géométriques et de conversion
    # ------------------------------------------------------------------

    def _face_pairing_to_detected(
        self,
        face_pairs: list[Any],
        unpaired: list[Any],
    ) -> list[DetectedSegment]:
        """Convertit FacePair + unpaired en DetectedSegment pour la topologie.

        Args:
            face_pairs: ``FacePair`` issus de ``pair_wall_faces``.
            unpaired: Segments non pairés (``Segment`` de wall_pairing).

        Returns:
            Liste de ``DetectedSegment`` combinant faces pairées et non pairées.
        """
        result: list[DetectedSegment] = []
        for fp in face_pairs:
            for face in (fp.face_a, fp.face_b):
                result.append(DetectedSegment(
                    x1=float(getattr(face, "x1", 0.0)),
                    y1=float(getattr(face, "y1", 0.0)),
                    x2=float(getattr(face, "x2", 0.0)),
                    y2=float(getattr(face, "y2", 0.0)),
                    source_slice=str(getattr(face, "source_slice", "high")),
                    confidence=float(getattr(face, "confidence", 0.9)),
                ))
        for seg in unpaired:
            result.append(DetectedSegment(
                x1=float(getattr(seg, "x1", 0.0)),
                y1=float(getattr(seg, "y1", 0.0)),
                x2=float(getattr(seg, "x2", 0.0)),
                y2=float(getattr(seg, "y2", 0.0)),
                source_slice=str(getattr(seg, "source_slice", "high")),
                confidence=float(getattr(seg, "confidence", 0.9)),
            ))
        return result

    def _ref_density_map(
        self,
        density_maps: dict[str, DensityMapResult],
    ) -> DensityMapResult | None:
        """Retourne la density map de référence (mid > high > first available).

        Args:
            density_maps: Density maps calculées.

        Returns:
            ``DensityMapResult`` ou ``None`` si aucune disponible.
        """
        for label in ("mid", "high"):
            if label in density_maps:
                return density_maps[label]
        return next(iter(density_maps.values()), None)

    # ------------------------------------------------------------------
    # Détection sol/plafond
    # ------------------------------------------------------------------

    def _detect_floor_ceiling(
        self,
        points: np.ndarray,
        floor_z_override: float | None,
        ceiling_z_override: float | None,
    ) -> tuple[float, float]:
        """Détecte ou applique les altitudes sol/plafond.

        Args:
            points: Nuage après downsampling + SOR.
            floor_z_override: Altitude sol imposée ou None.
            ceiling_z_override: Altitude plafond imposée ou None.

        Returns:
            ``(z_floor, z_ceiling)`` en mètres.
        """
        if floor_z_override is not None and ceiling_z_override is not None:
            logger.info(
                "Sol/Plafond manuels : sol=%.3f m, plafond=%.3f m.",
                floor_z_override, ceiling_z_override,
            )
            return floor_z_override, ceiling_z_override

        logger.info("Détection automatique sol/plafond.")
        z_floor_hist, z_ceiling_hist = self._step(
            "Histogramme Z",
            lambda: detect_floor_rdc(points),
        )
        cfg_fc = self.config.floor_ceiling
        z_floor = self._ransac_refine_floor(points, z_floor_hist, cfg_fc)
        z_ceiling = self._ransac_refine_ceiling(points, z_ceiling_hist, z_floor, cfg_fc)
        logger.info(
            "Sol=%.3f m  Plafond=%.3f m  H=%.2f m",
            z_floor, z_ceiling, z_ceiling - z_floor,
        )
        return z_floor, z_ceiling

    def _ransac_refine_floor(
        self, points: np.ndarray, z_hint: float, cfg_fc: Any
    ) -> float:
        """Affine l'altitude du sol par RANSAC ±30 cm autour du pic Z.

        Args:
            points: Nuage complet.
            z_hint: Altitude approximative du sol.
            cfg_fc: Configuration floor_ceiling.

        Returns:
            Altitude affinée.
        """
        mask = (points[:, 2] >= z_hint - 0.30) & (points[:, 2] <= z_hint + 0.30)
        pts_zone = points[mask]
        if len(pts_zone) < 3:
            return z_hint
        z_floor, _ = self._step(
            "RANSAC sol",
            lambda p=pts_zone: detect_floor(
                p,
                distance_threshold=cfg_fc.ransac_distance,
                num_iterations=cfg_fc.ransac_iterations,
                normal_tolerance_deg=cfg_fc.normal_tolerance_deg,
            ),
        )
        return z_floor

    def _ransac_refine_ceiling(
        self, points: np.ndarray, z_hint: float, z_floor: float, cfg_fc: Any
    ) -> float:
        """Affine l'altitude du plafond par RANSAC ±30 cm autour du pic Z.

        Args:
            points: Nuage complet.
            z_hint: Altitude approximative du plafond.
            z_floor: Altitude du sol.
            cfg_fc: Configuration floor_ceiling.

        Returns:
            Altitude affinée.
        """
        mask = (points[:, 2] >= z_hint - 0.30) & (points[:, 2] <= z_hint + 0.30)
        pts_zone = points[mask]
        if len(pts_zone) < 3:
            return z_hint
        z_ceiling, _ = self._step(
            "RANSAC plafond",
            lambda p=pts_zone: detect_ceiling(
                p,
                floor_z=z_floor,
                distance_threshold=cfg_fc.ransac_distance,
                num_iterations=cfg_fc.ransac_iterations,
                normal_tolerance_deg=cfg_fc.normal_tolerance_deg,
            ),
        )
        return z_ceiling

    # ------------------------------------------------------------------
    # Debug images
    # ------------------------------------------------------------------

    def _debug_img(
        self,
        debug_dir: Path | None,
        name: str,
        segments: list[DetectedSegment],
        dmap: DensityMapResult | None,
        title: str,
        face_pairs: list[Any] | None = None,
    ) -> None:
        """Sauvegarde une image de debug si debug_dir est défini.

        Args:
            debug_dir: Répertoire de destination ou None.
            name: Nom du fichier sans extension.
            segments: Segments à afficher.
            dmap: Density map de fond ou None.
            title: Titre de l'image.
            face_pairs: Paires de faces pour colorisation optionnelle.
        """
        if debug_dir is None or dmap is None:
            return
        self._save_debug_image(debug_dir, name, segments, dmap, title, face_pairs)

    def _save_debug_image(
        self,
        debug_dir: Path,
        name: str,
        segments: list[DetectedSegment],
        dmap: DensityMapResult,
        title: str,
        face_pairs: list[Any] | None = None,
    ) -> None:
        """Génère et sauvegarde une image PNG de debug.

        Affiche les segments colorés par confiance sur la density map.
        Si ``face_pairs`` est fourni, les faces pairées sont tracées en orange
        et les non pairées en bleu-gris.

        Args:
            debug_dir: Répertoire de destination.
            name: Nom du fichier sans extension.
            segments: Segments à afficher.
            dmap: Density map de fond.
            title: Titre de la figure.
            face_pairs: Paires de faces pour colorisation.
        """
        try:
            import matplotlib.pyplot as plt

            h, w = dmap.height, dmap.width
            extent = (
                dmap.x_min,
                dmap.x_min + w * dmap.resolution,
                dmap.y_min,
                dmap.y_min + h * dmap.resolution,
            )
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(dmap.image, origin="upper", extent=extent,
                      cmap="gray", interpolation="nearest")
            ax.set_title(f"{title}  [n={len(segments)}]", fontsize=11)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")

            if face_pairs:
                self._draw_segments_paired(ax, segments, face_pairs)
            else:
                self._draw_segments_confidence(ax, segments)

            plt.tight_layout()
            out = debug_dir / f"{name}.png"
            fig.savefig(str(out), dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.debug("Debug image : %s", out)
        except Exception as exc:
            logger.warning("Impossible de sauvegarder debug image %s : %s", name, exc)

    def _draw_segments_paired(
        self,
        ax: Any,
        segments: list[DetectedSegment],
        face_pairs: list[Any],
    ) -> None:
        """Trace les segments avec orange=pairé, bleu=non pairé.

        Args:
            ax: Axes matplotlib.
            segments: Segments à tracer.
            face_pairs: Paires de faces pour identifier les segments pairés.
        """
        paired_coords: set[tuple[float, float, float, float]] = set()
        for fp in face_pairs:
            for face in (getattr(fp, "face_a", None), getattr(fp, "face_b", None)):
                if face is not None:
                    paired_coords.add((
                        round(float(getattr(face, "x1", 0)), 4),
                        round(float(getattr(face, "y1", 0)), 4),
                        round(float(getattr(face, "x2", 0)), 4),
                        round(float(getattr(face, "y2", 0)), 4),
                    ))
        for seg in segments:
            key = (round(seg.x1, 4), round(seg.y1, 4),
                   round(seg.x2, 4), round(seg.y2, 4))
            color = "#FF8C00" if key in paired_coords else "#6699CC"
            ax.plot([seg.x1, seg.x2], [seg.y1, seg.y2],
                    color=color, linewidth=1.5, alpha=0.85)

    def _draw_segments_confidence(
        self,
        ax: Any,
        segments: list[DetectedSegment],
    ) -> None:
        """Trace les segments colorés par score de confiance (jet colormap).

        Args:
            ax: Axes matplotlib.
            segments: Segments à tracer.
        """
        import matplotlib.cm as cm
        cmap = cm.get_cmap("jet")
        for seg in segments:
            color = cmap(seg.confidence)
            lw = 1.0 + seg.confidence * 2.0
            ax.plot([seg.x1, seg.x2], [seg.y1, seg.y2],
                    color=color, linewidth=lw)

    # ------------------------------------------------------------------
    # QA et comptage ouvertures
    # ------------------------------------------------------------------

    def _run_qa(
        self,
        segments: list[DetectedSegment],
        face_pairs: list[Any],
    ) -> tuple[float, QAReport]:
        """Calcule un score QA basique sur le résultat final.

        Args:
            segments: Segments finaux.
            face_pairs: Paires de faces.

        Returns:
            ``(score, qa_report)``.
        """
        try:
            from scan2plan.qa.metrics import QAReport

            n_paired = len(face_pairs) * 2
            n_total = len(segments)
            pair_ratio = n_paired / n_total if n_total > 0 else 0.0

            # Score heuristique : ratio de segments pairés (murs bien définis)
            score = min(100.0, pair_ratio * 120.0)

            # Pénalité si trop peu de segments
            if n_total < 4:
                score = max(0.0, score - 30.0)

            report = QAReport(
                score=score,
                num_segments=n_total,
                num_rooms_detected=0,
                warnings=[],
            )
            return score, report
        except Exception as exc:
            logger.warning("QA échoué : %s", exc)
            from scan2plan.qa.metrics import QAReport
            return 0.0, QAReport(score=0.0, num_segments=len(segments))

    def _count_openings_in_dxf(self, dxf_path: Path) -> int:
        """Compte les entités sur le calque OUVERTURES dans le DXF produit.

        Args:
            dxf_path: Chemin du DXF.

        Returns:
            Nombre d'entités OUVERTURES.
        """
        try:
            import ezdxf

            doc = ezdxf.readfile(str(dxf_path))
            return sum(
                1 for e in doc.modelspace()
                if e.dxf.layer == "OUVERTURES"
            )
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Step wrapper + utilitaires
    # ------------------------------------------------------------------

    def _step(self, name: str, fn: Any) -> Any:
        """Exécute une étape en loggant le temps.

        Args:
            name: Nom lisible de l'étape.
            fn: Callable sans argument.

        Returns:
            Résultat de ``fn()``.
        """
        logger.info("[%s] démarrage…", name)
        t0 = time.monotonic()
        result = fn()
        logger.info("[%s] terminé en %.2f s.", name, time.monotonic() - t0)
        return result

    def _crop_xy(
        self,
        points: np.ndarray,
        x_min: float | None,
        x_max: float | None,
        y_min: float | None,
        y_max: float | None,
    ) -> np.ndarray:
        """Filtre les points hors de la zone XY.

        Args:
            points: Nuage (N, 3).
            x_min: Borne X minimale ou None.
            x_max: Borne X maximale ou None.
            y_min: Borne Y minimale ou None.
            y_max: Borne Y maximale ou None.

        Returns:
            Nuage filtré.
        """
        mask = np.ones(len(points), dtype=bool)
        if x_min is not None:
            mask &= points[:, 0] >= x_min
        if x_max is not None:
            mask &= points[:, 0] <= x_max
        if y_min is not None:
            mask &= points[:, 1] >= y_min
        if y_max is not None:
            mask &= points[:, 1] <= y_max
        cropped = points[mask]
        logger.info(
            "Recadrage XY : %d → %d points.", len(points), len(cropped)
        )
        return cropped

    def _save_npy(self, array: np.ndarray, output_path: Path, stem: str) -> None:
        """Sauvegarde un tableau NumPy dans le répertoire de sortie.

        Args:
            array: Tableau à sauvegarder.
            output_path: Chemin du DXF de référence (pour le répertoire).
            stem: Suffixe du nom de fichier.
        """
        out = output_path.parent / f"{output_path.stem}_{stem}.npy"
        np.save(str(out), array)
        logger.debug("Intermédiaire sauvegardé : %s", out)
