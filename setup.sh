#!/usr/bin/env bash
# setup.sh — Installation de l'environnement Scan2Plan sur Linux / macOS
set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${BOLD}[INFO]${NC} $*"; }
ok()      { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERREUR]${NC} $*"; exit 1; }

echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD} Scan2Plan — Installation de l'environnement Python${NC}"
echo -e "${BOLD}============================================================${NC}"
echo

# ----------------------------------------------------------------
# 1. Localiser Python 3.12
# ----------------------------------------------------------------
PYTHON=""
for candidate in python3.12 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        if [[ "$version" == "3.12" ]]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    warn "Python 3.12 non trouvé dans le PATH."
    echo
    echo "  Sur Ubuntu/Debian :"
    echo "    sudo add-apt-repository ppa:deadsnakes/ppa"
    echo "    sudo apt install python3.12 python3.12-venv"
    echo
    echo "  Sur macOS avec Homebrew :"
    echo "    brew install python@3.12"
    echo
    echo "  Puis relancez ce script."
    exit 1
fi

ok "$("$PYTHON" --version) détecté : $(command -v "$PYTHON")"

# ----------------------------------------------------------------
# 2. Créer le venv si absent
# ----------------------------------------------------------------
if [[ -f ".venv/bin/python" ]]; then
    ok "Venv .venv/ existant détecté, on le réutilise."
else
    info "Création du venv Python 3.12 dans .venv/ ..."
    "$PYTHON" -m venv .venv
    ok "Venv créé."
fi

VENV_PYTHON=".venv/bin/python"

# ----------------------------------------------------------------
# 3. Mettre à jour pip (via python -m pip pour éviter les pip corrompus)
# ----------------------------------------------------------------
info "Mise à jour de pip..."
"$VENV_PYTHON" -m pip install --upgrade pip

# ----------------------------------------------------------------
# 4. Installer le package et toutes ses dépendances
# ----------------------------------------------------------------
info "Installation de scan2plan et de ses dépendances..."
echo "     (open3d peut prendre quelques minutes au premier téléchargement)"
echo
"$VENV_PYTHON" -m pip install -e ".[dev,docs]" || error "L'installation a échoué. Vérifiez votre connexion internet."

# ----------------------------------------------------------------
# 5. Vérifier la CLI
# ----------------------------------------------------------------
echo
info "Vérification de la CLI scan2plan..."
if .venv/bin/scan2plan --help &>/dev/null; then
    ok "CLI scan2plan opérationnelle."
else
    warn "La CLI scan2plan n'est pas accessible. Activez le venv : source .venv/bin/activate"
fi

# ----------------------------------------------------------------
# 6. Lancer les tests
# ----------------------------------------------------------------
info "Lancement de la suite de tests..."
echo
"$VENV_PYTHON" -m pytest tests/ -v --tb=short && ok "Tous les tests passent." || warn "Certains tests ont échoué. Vérifiez les logs ci-dessus."

# ----------------------------------------------------------------
# 7. Résumé
# ----------------------------------------------------------------
echo
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD} Installation terminée.${NC}"
echo
echo "  Pour utiliser scan2plan :"
echo "    source .venv/bin/activate"
echo "    scan2plan --help"
echo
echo "  Pour lancer les tests :"
echo "    .venv/bin/python -m pytest tests/ -v"
echo -e "${BOLD}============================================================${NC}"
