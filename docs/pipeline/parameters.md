# Paramètres du pipeline

Voir [Configuration](../guide/configuration.md) pour la référence complète des paramètres YAML.

## Impact des paramètres critiques

### `voxel_size`

Taille du voxel pour le downsampling (mètres).

| Valeur | Usage | Impact |
|--------|-------|--------|
| 0.005 (5 mm) | Défaut — appartements standards | Précis, lent sur gros fichiers |
| 0.008 (8 mm) | Bureaux, grandes surfaces | Compromis vitesse/précision |
| 0.010 (10 mm) | Gros fichiers > 500 Mo | Rapide, peut manquer les cloisons fines |

!!! warning
    Un voxel trop grand (> 15 mm) peut ne pas détecter les cloisons de 5 cm (profil récent).

### `hough.threshold`

Nombre minimal de votes dans l'espace de Hough pour qu'un segment soit retenu.

- **Trop bas** (< 30) → beaucoup de faux positifs (mobilier, bruit)
- **Trop haut** (> 80) → segments manqués (cloisons courtes, surfaces peu denses)
- **Défaut** : 50 (bon équilibre sur un appartement standard)

### `segment_fusion.max_gap`

Gap maximal entre deux segments pour les fusionner (mètres).

- **0.10 m** : conservateur — ne fusionne que les segments très proches
- **0.20 m** : défaut — comble les petites discontinuités
- **0.40 m** : permissif (profil bureau) — cloisons avec ouvertures intégrées

### `regularization.snap_tolerance_deg`

Tolérance angulaire pour le snapping vers les directions dominantes.

- **3°** (profil récent) : logements neufs, murs orthogonaux
- **5°** (défaut) : tolérance standard
- **8°** (profil ancien) : bâtiments anciens avec angles variables

!!! tip "Règle pratique"
    Si le plan résultant a des murs légèrement obliques qui devraient être droits,
    augmenter la tolérance. Si des murs réellement obliques sont forcés à 90°,
    la diminuer.
