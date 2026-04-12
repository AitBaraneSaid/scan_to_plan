# Profils de paramètres

Trois profils prédéfinis couvrent les cas d'usage courants.

## Profils disponibles

### `recent` — Logement neuf

Cloisons légères (7-10 cm), angles droits, plafond ~2.50 m.

```yaml
regularization:
  snap_tolerance_deg: 3.0   # Plus strict : les murs sont orthogonaux
morphology:
  kernel_size: 3             # Petit noyau : ne pas noyer les cloisons fines
segment_fusion:
  angle_tolerance_deg: 2.0
```

### `ancien` — Bâtiment ancien

Murs épais (20-50 cm), angles variables, plafonds hauts (3.00-3.50 m).

```yaml
regularization:
  snap_tolerance_deg: 8.0   # Plus souple : les murs ne sont pas parfaitement orthogonaux
morphology:
  kernel_size: 7             # Grand noyau : combler les lacunes dans les murs épais
segment_fusion:
  angle_tolerance_deg: 5.0
  max_gap: 0.30
```

### `bureau` — Locaux professionnels

Grands espaces, cloisons vitrées fines, faux plafond ~2.60 m.

```yaml
hough:
  threshold: 40              # Seuil bas : détecter les cloisons vitrées peu denses
  max_line_gap: 25           # Gap large : montants de cloisons vitrées
segment_fusion:
  max_gap: 0.40              # Cloisons avec ouvertures intégrées
```

## Utilisation

```python
from scan2plan.config import ScanConfig
from scan2plan.config_profiles import apply_profile

cfg = ScanConfig()
apply_profile(cfg, "ancien")  # Retourne cfg pour le chaining
```

## Auto-calibrage

L'auto-calibrage analyse la density map et la hauteur sous plafond pour suggérer automatiquement le profil le plus adapté.

```python
from scan2plan.config_profiles import auto_calibrate, calibrate_slice_heights

# Analyser la density map
result = auto_calibrate(
    density_map_image,
    ceiling_height_m=2.85,
    resolution_m=0.005,
)

print(f"Profil suggéré : {result.suggested_profile}")
print(f"Confiance : {result.confidence:.0%}")
print(f"Épaisseur médiane des murs : {result.median_wall_thickness_m*100:.0f} cm")
print(f"Raisonnement : {result.reasoning}")

# Adapter les hauteurs de slices à la hauteur détectée
heights = calibrate_slice_heights(result.ceiling_height_m)
print(f"Slices suggérées : {heights}")
```

## Règles de décision

| Critère | Valeur | Profil suggéré |
|---------|--------|----------------|
| Densité de murs < 30 % | Peu de pixels occupés | `bureau` |
| Épaisseur médiane > 20 cm | Murs épais | `ancien` |
| Hauteur sous plafond > 2.85 m | Hauts plafonds | `ancien` |
| Sinon | — | `recent` |
