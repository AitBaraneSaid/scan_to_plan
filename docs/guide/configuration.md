# Configuration

Tous les paramètres du pipeline sont centralisés dans `config/default_params.yaml`.
Les surcharges partielles (par fichier utilisateur ou profil) sont fusionnées de façon récursive.

## Paramètres complets

```yaml
preprocessing:
  voxel_size: 0.005          # Taille du voxel downsampling (m). Plus grand → plus rapide, moins précis.
  sor_k_neighbors: 20        # K voisins pour le Statistical Outlier Removal.
  sor_std_ratio: 2.0         # Seuil σ pour le SOR. Plus bas → plus agressif.

floor_ceiling:
  ransac_distance: 0.02      # Tolérance RANSAC pour la détection du plan sol/plafond (m).
  ransac_iterations: 1000    # Nombre d'itérations RANSAC.
  normal_tolerance_deg: 10   # Tolérance d'horizontalité (degrés).

slicing:
  heights:                   # Hauteurs des slices relatives au sol (m).
    - 2.10                   # Haute : au-dessus des ouvertures et du mobilier.
    - 1.10                   # Médiane : hauteur de coupe standard.
    - 0.20                   # Basse : détection du bas des portes.
  thickness: 0.10            # Épaisseur de chaque tranche (m).

density_map:
  resolution: 0.005          # Résolution du raster (m/pixel). Défaut : 5 mm/px.

morphology:
  kernel_size: 5             # Taille de l'élément structurant (pixels).
  close_iterations: 2        # Fermeture morphologique (comble les trous dans les murs).
  open_iterations: 1         # Ouverture morphologique (supprime les artefacts).

hough:
  rho: 1                     # Résolution spatiale Hough (pixels).
  theta_deg: 0.5             # Résolution angulaire Hough (degrés).
  threshold: 50              # Seuil d'accumulation. Plus bas → plus de segments détectés.
  min_line_length: 50        # Longueur minimale de segment (pixels ≈ 25 cm à 5 mm/px).
  max_line_gap: 20           # Gap maximal dans un segment (pixels ≈ 10 cm).

segment_fusion:
  angle_tolerance_deg: 3.0   # Tolérance angulaire pour la colinéarité (degrés).
  perpendicular_dist: 0.03   # Distance perpendiculaire maximale pour fusionner (m).
  max_gap: 0.20              # Gap maximal entre segments à fusionner (m).

regularization:
  snap_tolerance_deg: 5.0    # Tolérance pour le snapping angulaire (degrés).

topology:
  intersection_distance: 0.05  # Distance max pour chercher une intersection (m).
  min_segment_length: 0.10     # Longueur minimale d'un segment conservé (m).

dxf:
  version: "R2013"           # Version DXF (R2010 ou R2013).
  layers:
    walls: "MURS"
    partitions: "CLOISONS"
    doors: "PORTES"
    windows: "FENETRES"
    uncertain: "INCERTAIN"
```

## Règles de surcharge

Les fichiers utilisateur sont des **surcharges partielles** : seules les clés présentes sont modifiées.

```yaml
# Exemple : ne surcharger que le seuil Hough
hough:
  threshold: 40
```

Cette surcharge ne touche pas les autres paramètres Hough (`rho`, `theta_deg`, etc.).

## Validation

La configuration est validée au chargement. Les erreurs courantes :

| Erreur | Cause |
|--------|-------|
| `voxel_size must be positive` | Valeur ≤ 0 |
| `snap_tolerance_deg out of range` | Valeur hors [0°, 45°] |
| `heights must be sorted descending` | Hauteurs non ordonnées |
