# Installation

## Prérequis système

| Composant | Version minimale | Recommandé |
|-----------|-----------------|------------|
| Python | 3.12 | 3.12.x |
| RAM | 8 Go | 16 Go |
| Espace disque | 2 Go | 5 Go |
| OS | Windows 10+, Ubuntu 20.04+, macOS 12+ | — |

!!! note "Python 3.12 obligatoire pour open3d"
    open3d (traitement nuage de points) ne supporte pas Python 3.13+.
    Si vous n'avez besoin que du pipeline 2D (density maps, Hough, DXF), Python 3.13+ fonctionne sans open3d.

---

## Installation via script (recommandé)

=== "Windows"

    ```bat
    git clone <url-du-depot>
    cd 3d_scan_to_2d_plan
    setup.bat
    ```

    Le script :
    1. Vérifie Python 3.12 (installe via winget si absent)
    2. Crée le venv `.venv\`
    3. Installe scan2plan et toutes ses dépendances
    4. Lance les tests de validation

=== "Linux / macOS"

    ```bash
    git clone <url-du-depot>
    cd 3d_scan_to_2d_plan
    bash setup.sh
    ```

---

## Installation manuelle

```bash
# 1. Cloner
git clone <url-du-depot>
cd 3d_scan_to_2d_plan

# 2. Créer le venv Python 3.12
py -3.12 -m venv .venv          # Windows
python3.12 -m venv .venv        # Linux/macOS

# 3. Activer
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS

# 4. Installer
pip install -e ".[dev]"

# 5. Optionnel : avec open3d (nécessite Python 3.12)
pip install -e ".[dev,pointcloud]"
```

---

## Installation via Docker

```bash
# Construire l'image
docker build -t scan2plan:latest .

# Vérifier
docker run --rm scan2plan:latest --help
```

---

## Validation de l'installation

```bash
# Vérifier la CLI
scan2plan --help

# Lancer les tests
python -m pytest tests/ -v

# Vérifier la couverture
python -m pytest tests/ --cov=scan2plan --cov-report=term-missing
```

Si tous les tests passent (hors les tests open3d si non installé), l'installation est correcte.
