# Scan2Plan

**Moteur Python de transformation d'un nuage de points 3D indoor en plan 2D vectoriel DXF.**

Scan2Plan automatise le passage d'un relevé laser (E57/LAS) à un plan 2D exploitable dans AutoCAD, réduisant le temps de post-traitement de **50 à 80 %**.

---

## Démarrage rapide

=== "Windows"

    ```bat
    setup.bat
    scan2plan scan.e57 output/plan.dxf
    ```

=== "Linux / macOS"

    ```bash
    bash setup.sh
    scan2plan scan.e57 output/plan.dxf
    ```

=== "Docker"

    ```bash
    docker pull scan2plan:latest
    docker run --rm \
      -v /chemin/scan.e57:/data/input.e57 \
      -v /chemin/output:/output \
      scan2plan:latest /data/input.e57 /output/plan.dxf
    ```

---

## Fonctionnalités V1

| Fonctionnalité | Statut |
|----------------|--------|
| Lecture E57 / LAS / LAZ | ✓ |
| Voxel downsampling + débruitage | ✓ |
| Détection sol/plafond RANSAC | ✓ |
| Multi-slice (3 hauteurs) | ✓ |
| Density maps + Hough probabiliste | ✓ |
| Fusion de segments colinéaires | ✓ |
| Détection des ouvertures (portes/fenêtres) | ✓ |
| Régularisation angulaire | ✓ |
| Reconstruction topologique | ✓ |
| Export DXF structuré en calques | ✓ |
| Double ligne avec épaisseur réelle | ✓ |
| Détection murs courbes et poteaux | ✓ |
| Profils de paramètres (récent/ancien/bureau) | ✓ |
| Auto-calibrage | ✓ |
| QA avancé avec scoring par zone | ✓ |
| Rapport PDF de qualité | ✓ |

---

## Calques DXF produits

| Calque | Contenu |
|--------|---------|
| `MURS` | Murs porteurs et cloisons confirmés |
| `CLOISONS` | Cloisons fines |
| `PORTES` | Ouvertures de type porte |
| `FENETRES` | Ouvertures de type fenêtre |
| `MURS_COURBES` | Murs et arcs courbes |
| `POTEAUX` | Poteaux circulaires |
| `INCERTAIN` | Zones à faible confiance — à valider manuellement |

---

## Limites connues

!!! warning "Points d'attention"
    - Les **portes fermées** sont indiscernables d'un mur — correction manuelle requise.
    - Les **meubles plaqués sol-plafond** peuvent être détectés comme murs.
    - Les **zones occultées** par du mobilier ne peuvent pas être reconstruites automatiquement.
    - open3d requiert **Python 3.12** — non compatible 3.13+.
