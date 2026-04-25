"""Tests pour la resolution des chemins runtime."""

from __future__ import annotations

import sys
from pathlib import Path

from scan2plan.runtime_paths import get_config_dir


def test_get_config_dir_finds_repo_config() -> None:
    config_dir = get_config_dir(__file__)
    assert config_dir.name == "config"
    assert (config_dir / "default_params.yaml").is_file()


def test_get_config_dir_prefers_meipass(monkeypatch: object, tmp_path: Path) -> None:
    bundled_config = tmp_path / "config"
    bundled_config.mkdir()
    (bundled_config / "default_params.yaml").write_text("preprocessing: {}\n", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert get_config_dir(__file__) == bundled_config
