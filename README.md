# Scan2Plan

Moteur Python de transformation d'un nuage de points 3D indoor en plan 2D vectoriel DXF, directement exploitable dans AutoCAD.

## Contexte

Les topographes utilisent des scanners laser terrestres (Leica RTC360) pour relever des appartements et bâtiments. Le passage du nuage de points 3D au plan 2D est aujourd'hui un travail essentiellement manuel, représentant plusieurs heures par relevé.

Scan2Plan automatise ce traitement : **réduction cible de 50 à 80 % du temps de post-traitement**.

## Pipeline de traitement

```
Fichier E57 / LAS
       │
       ▼
  Voxel downsampling (Open3D)
       │
       ▼
  Statistical Outlier Removal (Open3D)
       │
       ▼
  Détection sol + plafond RANSAC (Open3D)
       │
       ▼
  Extraction slices horizontales (NumPy)
       │
       ▼
  Density maps (histogram2d)
       │
       ▼
  Binarisation Otsu + morphologie (OpenCV)
       │
       ▼
  Détection segments Hough probabiliste (OpenCV)
       │
       ▼
  Fusion segments colinéaires
       │
       ▼
  Régularisation angulaire
       │
       ▼
  Export DXF structuré en calques (ezdxf)
```

## Prérequis système

- **Python 3.12** (open3d ne supporte pas Python 3.13+)
- Windows 10/11 64-bit, Linux ou macOS
- 8 Go de RAM minimum (16 Go recommandés pour les gros scans)

## Installation

### 1. Cloner le dépôt

```bash
git clone <url-du-depot>
cd 3d_scan_to_2d_plan
```

### 2. Lancer le script d'installation

**Windows :**
```bash
setup.bat
```

**Linux / macOS :**
```bash
bash setup.sh
```

Le script installe automatiquement Python 3.12 si nécessaire, crée le venv `.venv/` et installe toutes les dépendances.

### 3. Installation manuelle (alternative)

```bash
# Créer le venv avec Python 3.12
py -3.12 -m venv .venv          # Windows
python3.12 -m venv .venv        # Linux/macOS

# Activer
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS

# Installer le package et ses dépendances
pip install -e ".[dev]"
```

## Utilisation

### CLI

```bash
# Activer le venv (si pas déjà actif)
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS

# Traiter un fichier E57
scan2plan scan.e57 output/plan.dxf

# Avec un fichier de configuration personnalisé
scan2plan scan.laz output/plan.dxf --config ma_config.yaml

# Mode verbeux (logs DEBUG)
scan2plan scan.e57 output/plan.dxf --verbose
```

### Configuration

Les paramètres par défaut sont dans [config/default_params.yaml](config/default_params.yaml).

Pour surcharger, créer un fichier YAML partiel :

```yaml
# mon_appartement.yaml
preprocessing:
  voxel_size: 0.01   # voxel plus grand pour un gros scan

slicing:
  heights: [2.10, 1.10, 0.20]
  thickness: 0.10
```

```bash
scan2plan scan.e57 plan.dxf --config mon_appartement.yaml
```

### Calques DXF produits

| Calque | Contenu |
|--------|---------|
| `MURS` | Murs porteurs et cloisons confirmés |
| `CLOISONS` | Cloisons fines (V1) |
| `PORTES` | Ouvertures de type porte (V1) |
| `FENETRES` | Ouvertures de type fenêtre (V1) |
| `INCERTAIN` | Segments à faible confiance — à valider manuellement |

## Formats supportés

| Format | Lecture | Remarque |
|--------|---------|----------|
| E57 | ✓ | Format natif RTC360, multi-scans avec matrices de transformation |
| LAS | ✓ | Standard lidar |
| LAZ | ✓ | LAS compressé (recommandé pour les gros fichiers) |
| RCP/RCS | ✗ | Propriétaire Autodesk — convertir en E57 ou LAS avant |

## Lancer les tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -v          # Windows
.venv/bin/python -m pytest tests/ -v                  # Linux/macOS

# Avec couverture de code
.venv/Scripts/python.exe -m pytest tests/ --cov=scan2plan --cov-report=term-missing
```

## Architecture du projet

```
scan2plan/
├── config/                 # Paramètres par défaut (YAML)
├── src/scan2plan/
│   ├── cli.py              # Point d'entrée CLI (typer)
│   ├── pipeline.py         # Orchestrateur du pipeline
│   ├── config.py           # Chargement et validation de la configuration
│   ├── io/                 # Lecture E57/LAS, export DXF
│   ├── preprocessing/      # Downsampling, SOR, détection sol/plafond
│   ├── slicing/            # Extraction de tranches, density maps
│   ├── detection/          # Hough, fusion de segments, orientations
│   ├── vectorization/      # Régularisation, topologie, construction murs
│   ├── qa/                 # Contrôle qualité automatique
│   └── utils/              # Géométrie 2D, coordonnées, visualisation
├── tests/                  # Suite de tests pytest
│   └── fixtures/           # Nuages synthétiques pour les tests
├── setup.bat               # Script d'installation Windows
├── setup.sh                # Script d'installation Linux/macOS
└── pyproject.toml          # Métadonnées et dépendances
```

## Roadmap

- **V0.1 (actuel)** — Squelette + I/O + pipeline MVP (slice unique, Hough, DXF)
- **V1** — Multi-slice, détection ouvertures, régularisation, topologie, calques métier
- **V2+** — Double ligne (épaisseur réelle), murs courbes, poteaux, interface de validation

## Limites connues

- Les portes **fermées** sont indiscernables d'un mur — correction manuelle requise.
- Les meubles **plaqués sol-plafond** peuvent être détectés comme murs.
- Les zones **occulées** par du mobilier ne peuvent pas être reconstruites automatiquement.
- open3d ne supporte pas Python 3.13+ — Python 3.12 requis.
