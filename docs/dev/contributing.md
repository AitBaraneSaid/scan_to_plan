# Contribuer

## Environnement de développement

```bash
git clone <url-du-depot>
cd 3d_scan_to_2d_plan

# Créer le venv Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# Installer en mode éditable + outils dev
pip install -e ".[dev,pointcloud]"
```

## Avant chaque commit

```bash
# Lint
ruff check src/

# Format
ruff format src/

# Type check
mypy src/scan2plan/

# Tests
python -m pytest tests/ -v
```

## Standards de code

- **Python 3.12+** minimum.
- **Type hints** sur toutes les fonctions publiques.
- **Docstrings Google style** sur toutes les fonctions publiques.
- **Fonctions ≤ 40 lignes** de corps — découper si plus long.
- **Fichiers ≤ 300 lignes** — découper en sous-modules si plus long.
- **Pas de magic numbers** — tous les seuils viennent de la configuration.
- **Pas de `except: pass`**.

## Ajouter un test

1. Créer un fichier `tests/test_<module>/test_<feature>.py`.
2. Utiliser des données synthétiques générées programmatiquement (pas de fichiers volumineux).
3. Couvrir au minimum : cas nominal, cas limite, cas d'erreur.

## Ajouter un profil

1. Créer `config/profiles/<nom>.yaml` avec les surcharges partielles.
2. Ajouter `"<nom>"` à `AVAILABLE_PROFILES` dans `src/scan2plan/config_profiles.py`.
3. Ajouter des tests dans `tests/test_config/test_profiles.py`.
4. Documenter dans `docs/guide/profiles.md`.
