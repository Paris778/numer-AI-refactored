"""Persistence and atomic publication for the experiment layout (nmr/experiment_store.py).

Covers: scaffold-creation on first run, run.json persistence, staging +
atomic publish of export slots, discard, and immutability (re-publish rejected).
"""

from __future__ import annotations

import json

import pytest

from nmr import experiment_store, paths


def test_record_run_creates_scaffold_and_run(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    run_id = "a" * 64
    p = experiment_store.record_run("ender-xgb-v1", run_id, {"scorecard": {}})
    assert p.name == "run.json"
    assert paths.experiment_dir("ender-xgb-v1").joinpath("meta.json").is_file()
    assert paths.experiment_dir("ender-xgb-v1").joinpath("base_config.yaml").is_file()
    assert paths.experiment_dir("ender-xgb-v1").joinpath("README.md").is_file()
    assert json.loads(p.read_text())["scorecard"] == {}


def test_stage_publish_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    run_id = "a" * 64
    staging = experiment_store.stage_export("f", "partial", run_id)
    assert staging.name == f".tmp-{run_id}"
    (staging / "predict.pkl").write_bytes(b"x")
    (staging / "export.json").write_text("{}")
    final = experiment_store.publish_staged_export("f", "partial", run_id)
    assert final == paths.export_dir("f", "partial", run_id)
    assert not staging.exists()
    assert (final / "predict.pkl").read_bytes() == b"x"


def test_discard_staged(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    staging = experiment_store.stage_export("f", "full", "a" * 64)
    (staging / "x").write_text("x")
    experiment_store.discard_staged_export("f", "full", "a" * 64)
    assert not staging.exists()


def test_republish_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    run_id = "a" * 64
    experiment_store.stage_export("f", "full", run_id)
    experiment_store.publish_staged_export("f", "full", run_id)
    staging = experiment_store.stage_export("f", "full", run_id)
    (staging / "x").write_text("x")
    with pytest.raises(ValueError):
        experiment_store.publish_staged_export("f", "full", run_id)  # slot exists
