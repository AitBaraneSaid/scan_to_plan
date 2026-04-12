# Prise en main

## Traitement d'un fichier E57

```bash
# Activation du venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# Traitement basique
scan2plan scan.e57 output/plan.dxf

# Avec configuration personnalisée
scan2plan scan.e57 output/plan.dxf --config mon_appart.yaml

# Mode verbeux
scan2plan scan.e57 output/plan.dxf --verbose

# Avec rapport QA
scan2plan scan.e57 output/plan.dxf --qa-report output/rapport_qa.json
```

## Fichier de configuration minimal

```yaml
# mon_appart.yaml — surcharge partielle des paramètres
preprocessing:
  voxel_size: 0.01   # voxel plus grand pour un gros scan

slicing:
  heights: [2.10, 1.10, 0.20]
  thickness: 0.10
```

## Utiliser un profil prédéfini

Les profils sont dans `config/profiles/`. Trois profils disponibles :

| Profil | Usage |
|--------|-------|
| `recent` | Logements neufs, cloisons fines (7-10 cm), angles droits |
| `ancien` | Bâtiments anciens, murs épais (20-50 cm), angles variables |
| `bureau` | Locaux professionnels, grands espaces, faux plafond |

```python
from scan2plan.config import ScanConfig
from scan2plan.config_profiles import apply_profile, auto_calibrate

# Application d'un profil manuellement
cfg = ScanConfig()
apply_profile(cfg, "ancien")

# Auto-calibrage à partir d'une density map
result = auto_calibrate(density_map_image, ceiling_height_m=3.20, resolution_m=0.005)
print(f"Profil suggéré : {result.suggested_profile} (confiance {result.confidence:.0%})")
```

## Résultat attendu

Le DXF produit contient les calques suivants :

```
MURS         ← murs confirmés (blanc)
CLOISONS     ← cloisons fines (cyan)
PORTES       ← ouvertures portes (rouge)
FENETRES     ← ouvertures fenêtres (bleu)
MURS_COURBES ← arcs de murs courbes (magenta)
POTEAUX      ← poteaux circulaires (vert)
INCERTAIN    ← zones à vérifier (jaune, pointillés)
```

Ouvrir le DXF dans AutoCAD, vérifier les zones `INCERTAIN`, corriger les erreurs et valider.
