# Tests

## Lancer les tests

```bash
# Suite complète
python -m pytest tests/ -v

# Avec couverture
python -m pytest tests/ --cov=scan2plan --cov-report=term-missing

# Un seul module
python -m pytest tests/test_qa/ -v

# Un seul test
python -m pytest tests/test_qa/test_zone_scoring.py::TestComputeZoneScores::test_grid_dimensions -v
```

## Organisation

```
tests/
├── conftest.py                    # Fixtures partagées
├── test_io/
│   └── test_readers.py
├── test_preprocessing/
│   ├── test_downsampling.py       # Skip si open3d absent
│   ├── test_outlier_removal.py    # Skip si open3d absent
│   └── test_floor_ceiling.py     # Skip si open3d absent
├── test_slicing/
│   ├── test_slicer.py
│   └── test_density_map.py
├── test_detection/
│   ├── test_morphology.py
│   ├── test_line_detection.py
│   ├── test_segment_fusion.py
│   ├── test_openings.py
│   └── test_curved_walls.py
├── test_vectorization/
│   ├── test_regularization.py
│   ├── test_topology.py
│   └── test_wall_builder.py
├── test_config/
│   └── test_profiles.py
├── test_qa/
│   ├── test_validator.py
│   └── test_zone_scoring.py
└── fixtures/
    └── generate_fixtures.py
```

## Tests open3d

Les tests qui dépendent d'open3d (`test_preprocessing/`) sont automatiquement sautés si open3d n'est pas installé. C'est le comportement attendu sur Python 3.13+.

## Couverture cible

| Module | Couverture cible |
|--------|-----------------|
| `detection/` | > 85 % |
| `vectorization/` | > 85 % |
| `qa/` | > 90 % |
| `slicing/` | > 85 % |
| `io/` | > 75 % |
| `config*` | > 90 % |
