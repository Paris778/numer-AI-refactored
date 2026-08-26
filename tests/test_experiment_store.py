"""Persistence and atomic publication for the experiment layout (nmr/experiment_store.py).

Covers: scaffold-creation on first run, run.json persistence, the
RunResult recorder (parquets + run.json), staging + atomic publish of export
slots, discard, and immutability (re-publish rejected).
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from nmr import experiment_store, paths
from nmr.evaluation import MetricSummary
from nmr.runner import RunResult


def test_record_run_creates_scaffold_and_run(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    run_id = "a" * 64
    p = experiment_store.record_run("ender-xgb-v1", run_id, {"scorecard": {}})
    assert p.name == "run.json"
    assert paths.experiment_dir("ender-xgb-v1").joinpath("meta.json").is_file()
    assert paths.experiment_dir("ender-xgb-v1").joinpath("base_config.yaml").is_file()
    assert paths.experiment_dir("ender-xgb-v1").joinpath("README.md").is_file()
    assert json.loads(p.read_text())["scorecard"] == {}


def _result(run_id: str, *, with_validation_preds: bool = False) -> RunResult:
    oof = pl.DataFrame({"id": ["a", "b"], "era": ["1", "1"], "prediction": [0.1, 0.9]})
    return RunResult(
        run_id=run_id,
        oof=oof,
        metrics=MetricSummary(mean=0.1, std=0.2, sharpe=0.5, max_drawdown=0.05),
        artifact=None,
        manifest={"config": {"run": {"name": "fam-a"}}, "oof_device": "cpu"},
        validation_predictions=(
            pl.DataFrame({"era": ["0575", "0575"], "id": ["x", "y"], "prediction": [0.2, 0.8]})
            if with_validation_preds
            else None
        ),
    )


def test_record_run_result_writes_parquets_and_run_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    result = _result("a" * 64, with_validation_preds=True)
    run_dir = experiment_store.record_run_result("fam-a", result)
    assert run_dir == paths.run_dir("fam-a", "a" * 64)
    assert pl.read_parquet(run_dir / "oof.parquet").height == 2
    persisted = pl.read_parquet(run_dir / "validation_preds.parquet")
    assert persisted["prediction"].to_list() == [0.2, 0.8]
    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == "a" * 64
    assert payload["manifest"]["config"]["run"]["name"] == "fam-a"
    assert payload["oof_path"] == "oof.parquet"
    # no validation predictions -> no validation_preds.parquet
    other = _result("b" * 64)
    other_dir = experiment_store.record_run_result("fam-a", other)
    assert not (other_dir / "validation_preds.parquet").exists()


def test_record_run_result_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    result = _result("a" * 64)
    run_dir = experiment_store.record_run_result("fam-a", result)
    original = (run_dir / "run.json").read_text(encoding="utf-8")
    assert experiment_store.record_run_result("fam-a", result) == run_dir
    assert (run_dir / "run.json").read_text(encoding="utf-8") == original


def test_record_run_result_rejects_bad_slug_and_run_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    with pytest.raises(ValueError, match="slug"):
        experiment_store.record_run_result("Bad Name", _result("a" * 64))
    with pytest.raises(ValueError, match="run_id"):
        experiment_store.record_run_result("fam-a", _result("not-a-run-id"))


def test_record_run_result_atomic_write_failure_keeps_previous_run_json(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    result = _result("a" * 64)
    run_dir = experiment_store.record_run_result("fam-a", result)
    stable_json = (run_dir / "run.json").read_text(encoding="utf-8")

    import nmr._atomicio as atomicio_module

    def fail_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(atomicio_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        experiment_store.record_run_result("fam-a", result)

    assert (run_dir / "run.json").read_text(encoding="utf-8") == stable_json


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
