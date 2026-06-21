"""Tests for run registry persistence and promotion semantics."""

from __future__ import annotations

import json

import polars as pl
import pytest

from nmr.evaluation import MetricSummary
from nmr.registry import RunRegistry
from nmr.runner import RunResult


def _result(run_id: str, sharpe: float) -> RunResult:
    return RunResult(
        run_id=run_id,
        oof=pl.DataFrame(
            {
                "id": ["a", "b"],
                "era": ["1", "1"],
                "prediction": [0.1, 0.9],
            }
        ),
        metrics=MetricSummary(mean=0.1, std=0.2, sharpe=sharpe, max_drawdown=0.05),
        artifact=None,
        manifest={"run_id": run_id},
    )


def test_record_is_idempotent_and_writes_json_atomically(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    result = _result("run-a", sharpe=0.7)

    run_dir = registry.record(result)
    original = (run_dir / "run.json").read_text(encoding="utf-8")
    run_dir_again = registry.record(result)
    repeated = (run_dir_again / "run.json").read_text(encoding="utf-8")

    assert run_dir == run_dir_again
    assert original == repeated


def test_atomic_write_failure_keeps_previous_run_json(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = RunRegistry(tmp_path)
    result = _result("run-a", sharpe=0.7)
    run_dir = registry.record(result)
    stable_json = (run_dir / "run.json").read_text(encoding="utf-8")

    import nmr.registry as registry_module

    def fail_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(registry_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        registry.record(result)

    assert (run_dir / "run.json").read_text(encoding="utf-8") == stable_json


def test_list_best_and_promote_are_deterministic_and_idempotent(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    registry.record(_result("run-a", sharpe=0.3))
    registry.record(_result("run-b", sharpe=0.8))

    listed = registry.list()
    assert {entry["run_id"] for entry in listed} == {"run-a", "run-b"}

    best = registry.best("sharpe")
    assert best is not None
    assert best["run_id"] == "run-b"

    run_b_json_before = (tmp_path / "run-b" / "run.json").read_text(encoding="utf-8")
    champion_path = registry.promote("run-b")
    champion_again = registry.promote("run-b")
    assert champion_path == champion_again
    assert json.loads(champion_path.read_text(encoding="utf-8")) == {"run_id": "run-b"}

    run_b_json_after = (tmp_path / "run-b" / "run.json").read_text(encoding="utf-8")
    assert run_b_json_before == run_b_json_after


def test_best_returns_none_for_empty_registry(tmp_path) -> None:
    assert RunRegistry(tmp_path).best() is None
