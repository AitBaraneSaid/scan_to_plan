"""Tests unitaires pour io/writers.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from scan2plan.detection.line_detection import DetectedSegment
from scan2plan.io.writers import export_dxf, setup_dxf_layers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(x1: float, y1: float, x2: float, y2: float) -> DetectedSegment:
    return DetectedSegment(x1=x1, y1=y1, x2=x2, y2=y2, source_slice="mid", confidence=0.9)


def _four_segments() -> list[DetectedSegment]:
    """Rectangle 4 m × 3 m centré en (0, 0)."""
    return [
        _seg(0.0, 0.0, 4.0, 0.0),   # bas
        _seg(4.0, 0.0, 4.0, 3.0),   # droit
        _seg(4.0, 3.0, 0.0, 3.0),   # haut
        _seg(0.0, 3.0, 0.0, 0.0),   # gauche
    ]


# ---------------------------------------------------------------------------
# Tests export_dxf
# ---------------------------------------------------------------------------

class TestExportDxf:
    def test_export_creates_file(self, tmp_path: Path) -> None:
        """export_dxf doit créer le fichier DXF sur le disque."""
        out = tmp_path / "plan.dxf"
        result = export_dxf(_four_segments(), out)
        assert result.exists()
        assert result.suffix == ".dxf"

    def test_returns_correct_path(self, tmp_path: Path) -> None:
        """La valeur de retour doit être le chemin effectif du fichier."""
        out = tmp_path / "plan.dxf"
        result = export_dxf(_four_segments(), out)
        assert result == out

    def test_extension_forced_to_dxf(self, tmp_path: Path) -> None:
        """Si le chemin n'a pas d'extension .dxf, elle doit être ajoutée."""
        out = tmp_path / "plan.txt"
        result = export_dxf(_four_segments(), out)
        assert result.suffix == ".dxf"
        assert result.exists()

    def test_dxf_is_readable_by_ezdxf(self, tmp_path: Path) -> None:
        """Le DXF produit doit être lisible par ezdxf sans erreur (roundtrip)."""
        import ezdxf
        out = tmp_path / "plan.dxf"
        export_dxf(_four_segments(), out)
        doc = ezdxf.readfile(str(out))
        assert doc is not None

    def test_dxf_has_correct_number_of_entities(self, tmp_path: Path) -> None:
        """Le DXF doit contenir exactement autant de LINE que de segments."""
        import ezdxf
        segs = _four_segments()
        out = tmp_path / "plan.dxf"
        export_dxf(segs, out)
        doc = ezdxf.readfile(str(out))
        msp = doc.modelspace()
        lines = [e for e in msp if e.dxftype() == "LINE"]
        assert len(lines) == len(segs)

    def test_dxf_coordinates(self, tmp_path: Path) -> None:
        """Les coordonnées des entités LINE doivent correspondre aux segments en mètres."""
        import ezdxf
        segs = [_seg(1.0, 2.0, 3.0, 4.0)]
        out = tmp_path / "plan.dxf"
        export_dxf(segs, out)
        doc = ezdxf.readfile(str(out))
        msp = doc.modelspace()
        lines = [e for e in msp if e.dxftype() == "LINE"]
        assert len(lines) == 1
        line = lines[0]
        assert line.dxf.start.x == pytest.approx(1.0)
        assert line.dxf.start.y == pytest.approx(2.0)
        assert line.dxf.start.z == pytest.approx(0.0)
        assert line.dxf.end.x == pytest.approx(3.0)
        assert line.dxf.end.y == pytest.approx(4.0)
        assert line.dxf.end.z == pytest.approx(0.0)

    def test_dxf_layer(self, tmp_path: Path) -> None:
        """Toutes les entités LINE doivent être sur le calque 'MURS_DETECTES'."""
        import ezdxf
        out = tmp_path / "plan.dxf"
        export_dxf(_four_segments(), out)
        doc = ezdxf.readfile(str(out))
        msp = doc.modelspace()
        lines = [e for e in msp if e.dxftype() == "LINE"]
        for line in lines:
            assert line.dxf.layer == "MURS_DETECTES"

    def test_murs_detectes_layer_exists(self, tmp_path: Path) -> None:
        """Le calque 'MURS_DETECTES' doit exister dans le document."""
        import ezdxf
        out = tmp_path / "plan.dxf"
        export_dxf(_four_segments(), out)
        doc = ezdxf.readfile(str(out))
        assert "MURS_DETECTES" in doc.layers

    def test_empty_segments(self, tmp_path: Path) -> None:
        """Liste vide → DXF valide avec 0 entités LINE, sans erreur."""
        import ezdxf
        out = tmp_path / "empty.dxf"
        export_dxf([], out)
        assert out.exists()
        doc = ezdxf.readfile(str(out))
        msp = doc.modelspace()
        lines = [e for e in msp if e.dxftype() == "LINE"]
        assert len(lines) == 0

    def test_parent_dir_created(self, tmp_path: Path) -> None:
        """Le répertoire parent doit être créé si nécessaire."""
        out = tmp_path / "subdir" / "deep" / "plan.dxf"
        export_dxf(_four_segments(), out)
        assert out.exists()

    def test_with_layer_config(self, tmp_path: Path) -> None:
        """Avec layer_config, les calques métier doivent être présents dans le DXF."""
        import ezdxf
        layer_config = {
            "walls": "MURS",
            "partitions": "CLOISONS",
            "doors": "PORTES",
            "windows": "FENETRES",
            "uncertain": "INCERTAIN",
        }
        out = tmp_path / "plan_layers.dxf"
        export_dxf(_four_segments(), out, layer_config=layer_config)
        doc = ezdxf.readfile(str(out))
        for name in layer_config.values():
            assert name in doc.layers, f"Calque '{name}' absent du DXF"

    def test_r2010_version(self, tmp_path: Path) -> None:
        """La version R2010 doit aussi fonctionner sans erreur."""
        import ezdxf
        out = tmp_path / "plan_r2010.dxf"
        export_dxf(_four_segments(), out, version="R2010")
        doc = ezdxf.readfile(str(out))
        assert doc is not None


# ---------------------------------------------------------------------------
# Tests setup_dxf_layers
# ---------------------------------------------------------------------------

class TestSetupDxfLayers:
    def test_layers_created(self) -> None:
        """setup_dxf_layers doit créer tous les calques spécifiés."""
        import ezdxf
        doc = ezdxf.new("R2013")
        layer_config = {"walls": "MURS", "doors": "PORTES", "windows": "FENETRES"}
        setup_dxf_layers(doc, layer_config)
        for name in layer_config.values():
            assert name in doc.layers

    def test_existing_layer_not_duplicated(self) -> None:
        """Un calque déjà existant ne doit pas provoquer d'erreur."""
        import ezdxf
        doc = ezdxf.new("R2013")
        doc.layers.add("MURS")
        layer_config = {"walls": "MURS"}
        setup_dxf_layers(doc, layer_config)
        assert "MURS" in doc.layers

    def test_distinct_colors(self) -> None:
        """Les calques MURS, PORTES, FENETRES doivent avoir des couleurs distinctes."""
        import ezdxf
        doc = ezdxf.new("R2013")
        layer_config = {"walls": "MURS", "doors": "PORTES", "windows": "FENETRES"}
        setup_dxf_layers(doc, layer_config)
        colors = [doc.layers.get(name).dxf.color for name in layer_config.values()]
        assert len(set(colors)) > 1, "Les calques doivent avoir des couleurs distinctes"
