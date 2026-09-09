"""Tests for AI4AO.paths -- data-directory resolution and save-path helpers."""
from pathlib import Path

from AI4AO import paths


def test_default_data_dir_is_repo_data(monkeypatch):
    monkeypatch.delenv("AI4AO_DATA_DIR", raising=False)
    assert paths.get_data_dir() == paths.REPO_ROOT / "Data"


def test_env_var_overrides_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AI4AO_DATA_DIR", str(tmp_path / "elsewhere"))
    assert paths.get_data_dir() == tmp_path / "elsewhere"


def test_env_var_expands_user(monkeypatch):
    monkeypatch.setenv("AI4AO_DATA_DIR", "~/ai4ao_data")
    assert paths.get_data_dir() == Path.home() / "ai4ao_data"


def test_ensure_parent_creates_missing_dirs(tmp_path):
    target = tmp_path / "a" / "b" / "c.pth"
    paths.ensure_parent(target)
    assert target.parent.is_dir()
    paths.ensure_parent(target)  # idempotent


def test_instrument_dir_creates_and_returns(monkeypatch, tmp_path):
    monkeypatch.setenv("AI4AO_DATA_DIR", str(tmp_path))
    d = paths.instrument_dir("Rama")
    assert d == tmp_path / "Rama"
    assert d.is_dir()

    assert paths.instrument_dir("Ekarus", create=False) == tmp_path / "Ekarus"
    assert not (tmp_path / "Ekarus").exists()
