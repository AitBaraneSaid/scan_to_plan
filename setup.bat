@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  Scan2Plan — Installation de l'environnement Python
echo ============================================================
echo.

:: ----------------------------------------------------------------
:: 1. Verifier que Python 3.12 est disponible
:: ----------------------------------------------------------------
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Python 3.12 non detecte. Installation via winget...
    winget install Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo [ERREUR] Echec de l'installation de Python 3.12.
        echo          Installez-le manuellement depuis https://www.python.org/downloads/release/python-3121/
        echo          puis relancez ce script.
        pause
        exit /b 1
    )
    echo [OK] Python 3.12 installe.
) else (
    for /f "tokens=*" %%v in ('py -3.12 --version 2^>^&1') do echo [OK] %%v detecte.
)

:: ----------------------------------------------------------------
:: 2. Creer le venv si absent
:: ----------------------------------------------------------------
if exist ".venv\Scripts\python.exe" (
    echo [OK] Venv .venv\ existant detecte, on le reutilise.
) else (
    echo [INFO] Creation du venv Python 3.12 dans .venv\ ...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo [ERREUR] Impossible de creer le venv.
        pause
        exit /b 1
    )
    echo [OK] Venv cree.
)

:: ----------------------------------------------------------------
:: 3. Mettre a jour pip dans le venv
::    Utiliser python.exe -m pip (et non pip.exe) pour eviter les
::    erreurs si le pip embarque dans le venv est corrompu ou obsolete
:: ----------------------------------------------------------------
echo [INFO] Mise a jour de pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERREUR] Impossible de mettre a jour pip.
    pause
    exit /b 1
)

:: ----------------------------------------------------------------
:: 4. Installer le package et toutes ses dependances
::    Utiliser python.exe -m pip pour la meme raison qu'a l'etape 3
:: ----------------------------------------------------------------
echo [INFO] Installation de scan2plan et de ses dependances...
echo        (open3d peut prendre quelques minutes au premier telechargement)
echo.
.venv\Scripts\python.exe -m pip install -e ".[dev,docs]"
if errorlevel 1 (
    echo.
    echo [ERREUR] L'installation a echoue. Verifiez votre connexion internet
    echo          et relancez le script.
    pause
    exit /b 1
)

:: ----------------------------------------------------------------
:: 5. Verifier que la CLI est operationnelle
:: ----------------------------------------------------------------
echo.
echo [INFO] Verification de la CLI scan2plan...
.venv\Scripts\scan2plan.exe --help >nul 2>&1
if errorlevel 1 (
    echo [AVERTISSEMENT] La CLI scan2plan n'est pas accessible.
    echo                 Activez le venv et relancez : .venv\Scripts\activate
) else (
    echo [OK] CLI scan2plan operationnelle.
)

:: ----------------------------------------------------------------
:: 6. Formater et verifier le code avec ruff
::    Appel via python.exe -m ruff (contourne les problemes de PATH PowerShell)
:: ----------------------------------------------------------------
echo.
echo [INFO] Formatage du code source (ruff format)...
.venv\Scripts\python.exe -m ruff format src\
if errorlevel 1 (
    echo [AVERTISSEMENT] ruff format a echoue.
) else (
    echo [OK] Code formate.
)

echo [INFO] Verification du code (ruff check)...
.venv\Scripts\python.exe -m ruff check src\
if errorlevel 1 (
    echo [AVERTISSEMENT] ruff check a signale des problemes. Voir les logs ci-dessus.
) else (
    echo [OK] Aucun probleme de linting.
)

:: ----------------------------------------------------------------
:: 8. Lancer les tests pour valider l'installation
:: ----------------------------------------------------------------
echo [INFO] Lancement de la suite de tests...
echo.
.venv\Scripts\python.exe -m pytest tests\ -v --tb=short
if errorlevel 1 (
    echo.
    echo [AVERTISSEMENT] Certains tests ont echoue. Verifiez les logs ci-dessus.
) else (
    echo.
    echo [OK] Tous les tests passent.
)

:: ----------------------------------------------------------------
:: 9. Resume
:: ----------------------------------------------------------------
echo.
echo ============================================================
echo  Installation terminee.
echo.
echo  Pour utiliser scan2plan :
echo    .venv\Scripts\activate
echo    scan2plan --help
echo.
echo  Pour lancer les tests :
echo    .venv\Scripts\python.exe -m pytest tests\ -v
echo ============================================================
echo.
pause
