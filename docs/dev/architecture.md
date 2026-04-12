# Architecture

## Arborescence

```
scan2plan/
├── config/
│   ├── default_params.yaml         # Paramètres par défaut
│   └── profiles/
│       ├── recent.yaml             # Profil logement neuf
│       ├── ancien.yaml             # Profil bâtiment ancien
│       └── bureau.yaml             # Profil locaux professionnels
│
├── src/scan2plan/
│   ├── cli.py                      # Point d'entrée CLI (typer)
│   ├── pipeline.py                 # Orchestrateur du pipeline complet
│   ├── config.py                   # Chargement et validation de la configuration
│   ├── config_profiles.py          # Profils et auto-calibrage
│   │
│   ├── io/
│   │   ├── readers.py              # Lecture E57, LAS, PTS
│   │   └── writers.py              # Export DXF V0 + V1
│   │
│   ├── preprocessing/
│   │   ├── downsampling.py         # Voxel downsampling (open3d)
│   │   ├── outlier_removal.py      # Statistical Outlier Removal (open3d)
│   │   └── floor_ceiling.py        # Détection sol/plafond RANSAC (open3d)
│   │
│   ├── slicing/
│   │   ├── slicer.py               # Extraction de tranches horizontales
│   │   └── density_map.py          # Projection 2D, histogram2d
│   │
│   ├── detection/
│   │   ├── morphology.py           # Binarisation, nettoyage morphologique
│   │   ├── line_detection.py       # Hough probabiliste, DetectedSegment
│   │   ├── segment_fusion.py       # Fusion de segments colinéaires
│   │   ├── orientation.py          # Détection des orientations dominantes
│   │   ├── openings.py             # Détection portes/fenêtres, Opening
│   │   └── curved_walls.py         # Arcs de murs, poteaux (DetectedArc, DetectedPillar)
│   │
│   ├── vectorization/
│   │   ├── regularization.py       # Snapping angulaire
│   │   ├── topology.py             # Graphe de murs, WallGraph
│   │   └── wall_builder.py         # Double ligne, épaisseur réelle
│   │
│   ├── qa/
│   │   ├── metrics.py              # QAReport
│   │   ├── validator.py            # validate_plan, generate_qa_report
│   │   └── zone_scoring.py         # ZoneScore, ZoneMap, heatmap, PDF
│   │
│   └── utils/
│       ├── geometry.py             # Fonctions géométriques 2D
│       ├── coordinate.py           # Conversion pixel ↔ métrique
│       └── visualization.py        # Visualisation matplotlib (debug)
│
└── tests/
    └── ...                         # Miroir de src/scan2plan/
```

## Principes

- **Un module = une responsabilité.** `slicing` ne fait que du slicing.
- **I/O isolées.** Le reste du pipeline ne connaît que des NumPy arrays.
- **Configuration centralisée.** Un seul YAML, surcharges partielles possibles.
- **Pas de magic numbers.** Tous les seuils viennent de la config.
- **Testabilité.** Chaque fonction est pure et testable avec des données synthétiques.

## Types de données centraux

| Type | Module | Description |
|------|--------|-------------|
| `DetectedSegment` | `detection/line_detection.py` | Segment de mur `(x1,y1,x2,y2,source_slice,confidence)` |
| `DensityMapResult` | `slicing/density_map.py` | Image 2D + métadonnées géographiques |
| `Opening` | `detection/openings.py` | Ouverture (porte/fenêtre) avec position et largeur |
| `WallGraph` | `vectorization/topology.py` | Graphe topologique des murs |
| `DetectedArc` | `detection/curved_walls.py` | Arc de mur courbe |
| `DetectedPillar` | `detection/curved_walls.py` | Poteau circulaire |
| `QAReport` | `qa/metrics.py` | Métriques globales de qualité |
| `ZoneMap` | `qa/zone_scoring.py` | Carte de confiance par zone |
