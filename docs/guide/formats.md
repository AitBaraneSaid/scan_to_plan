# Formats supportés

## Entrée

| Format | Extension | Lecture | Bibliothèque | Remarque |
|--------|-----------|---------|-------------|----------|
| **E57** | `.e57` | ✓ | pye57 | Format natif RTC360. Supporte multi-scans + matrices de transformation. |
| **LAS** | `.las` | ✓ | laspy | Standard lidar. |
| **LAZ** | `.laz` | ✓ | laspy+lazrs | LAS compressé — 10× plus petit. Recommandé pour les gros fichiers. |
| **PTS** | `.pts` | Partiel | — | ASCII texte. Très volumineux. Convertir en LAS. |
| **RCP/RCS** | `.rcp/.rcs` | ✗ | — | Propriétaire Autodesk. Exporter en E57 ou LAS depuis ReCap. |

## Sortie

| Format | Extension | Usage |
|--------|-----------|-------|
| **DXF** | `.dxf` | Plan 2D vectoriel pour AutoCAD (R2013, entités LINE/LWPOLYLINE/ARC/CIRCLE). |
| **JSON** | `.json` | Rapport QA automatique. |
| **PDF** | `.pdf` | Rapport visuel avec heatmap de confiance. |
| **PNG** | `.png` | Heatmap de confiance par zone. |

## Conversion de formats

Si votre fichier est en RCP/RCS, convertir d'abord dans Autodesk ReCap :

```
ReCap → Exporter → Point Cloud → Format E57 ou LAS
```

Si votre fichier est en PTS :

```bash
# Avec CloudCompare (gratuit)
CloudCompare -AUTO_SAVE OFF -O scan.pts -C_EXPORT_FMT LAS -SAVE_CLOUDS
```
