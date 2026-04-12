# Pipeline — Vue d'ensemble

Le pipeline Scan2Plan transforme un nuage de points 3D en plan 2D DXF en 15 étapes organisées en 3 couches :

```
Nuage de points 3D (E57 / LAS)
         │
   ┌─────▼──────┐
   │    3D      │  Étapes 1-6 : contexte global
   │ (Open3D)   │  sol, plafond, orientations dominantes
   └─────┬──────┘
         │
   ┌─────▼──────┐
   │    2D      │  Étapes 7-12 : extraction des murs
   │ (OpenCV)   │  slicing, density maps, Hough, fusion
   └─────┬──────┘
         │
   ┌─────▼──────┐
   │ Vectoriel  │  Étapes 13-16 : qualité finale
   │  (ezdxf)   │  régularisation, topologie, export DXF
   └─────┬──────┘
         │
   Plan DXF + Rapport QA
```

## Étapes détaillées

| # | Étape | Module | Sortie |
|---|-------|--------|--------|
| 1 | Lecture du nuage | `io/readers.py` | `np.ndarray (N×3)` |
| 2 | Voxel downsampling | `preprocessing/downsampling.py` | Nuage réduit |
| 3 | Statistical Outlier Removal | `preprocessing/outlier_removal.py` | Nuage propre |
| 4 | Détection sol/plafond RANSAC | `preprocessing/floor_ceiling.py` | `z_floor`, `z_ceiling` |
| 5 | Filtrage vertical | `pipeline.py` | Nuage intérieur |
| 6 | Orientations dominantes | `detection/orientation.py` | Angles dominants |
| 7 | Extraction slices | `slicing/slicer.py` | 3 nuages 2D |
| 8 | Density maps | `slicing/density_map.py` | 3 images float32 |
| 9 | Binarisation + morphologie | `detection/morphology.py` | 3 images binaires |
| 10 | Hough probabiliste | `detection/line_detection.py` | Liste de segments |
| 11 | Fusion colinéaire | `detection/segment_fusion.py` | Segments fusionnés |
| 12 | Recoupement multi-slice | `pipeline.py` | Segments filtrés |
| 13 | Détection ouvertures | `detection/openings.py` | Liste d'ouvertures |
| 14 | Régularisation | `vectorization/regularization.py` | Segments régularisés |
| 15 | Topologie | `vectorization/topology.py` | `WallGraph` |
| 16 | Export DXF | `io/writers.py` | Fichier DXF |
| — | QA | `qa/validator.py`, `qa/zone_scoring.py` | Rapport JSON/PDF |
