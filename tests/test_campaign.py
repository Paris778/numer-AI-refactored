# tests/test_campaign.py
from __future__ import annotations

import json

import polars as pl
import pytest

from nmr.campaign import (
    CampaignConfig,
    CampaignLog,
    CampaignRun,
    build_campaign_log,
    campaign_id,
    write_campaign_log,
)


def _write_config(tmp_path, name: str, content: str) -> None:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_campaign_id_is_deterministic_and_path_independent(tmp_path) -> None:
    a = _write_config(tmp_path, "a.yaml", "run:\n  name: x\n")
    b = _write_config(tmp_path, "b.yaml", "run:\n  name: y\n")
    c = _write_config(tmp_path, "c.yaml", "run:\n  name: x\n")  # same content as a

    assert campaign_id("camp", [a, b]) == campaign_id("camp", [a, b])
    assert campaign_id("camp", [a, b]) == campaign_id("camp", [b, a])  # order-independent: identity is the set of configs
    assert campaign_id("camp", [a, b]) != campaign_id("other", [a, b])
    # identical content, different file name -> identical id (path-independent)
    assert campaign_id("camp", [a, b]) == campaign_id("camp", [c, b])
    assert len(campaign_id("camp", [a, b])) == 64
    assert campaign_id("camp", [a, b]).isalnum()


def test_build_campaign_log_validates_inputs(tmp_path) -> None:
    cfg = _write_config(tmp_path, "a.yaml", "run:\n  name: x\n")
    with pytest.raises(ValueError, match="name"):
        build_campaign_log("", [cfg], runs=())
    with pytest.raises(ValueError, match="config_paths"):
        build_campaign_log("camp", [], runs=())
    with pytest.raises(FileNotFoundError):
        build_campaign_log("camp", [tmp_path / "missing.yaml"], runs=())
    with pytest.raises(ValueError, match="status"):
        build_campaign_log(
            "camp", [cfg], runs=[CampaignRun(str(cfg), run_id=None, status="bogus")]
        )


def test_write_campaign_log_atomic_and_schema(tmp_path) -> None:
    cfg = _write_config(tmp_path, "a.yaml", "run:\n  name: x\n")
    log = build_campaign_log(
        "camp",
        [cfg],
        runs=[
            CampaignRun(str(cfg), run_id="a" * 64, status="recorded"),
            CampaignRun(str(cfg), run_id=None, status="error", error="boom"),
        ],
    )
    out_dir = tmp_path / "campaigns"
    written = write_campaign_log(log, out_dir)
    assert written == out_dir / f"{log.campaign_id}.json"
    assert written.exists()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["campaign_id"] == log.campaign_id
    assert payload["name"] == "camp"
    assert payload["configs"][0]["path"] == str(cfg)
    assert len(payload["configs"][0]["sha256"]) == 64
    assert payload["runs"][0]["status"] == "recorded"
    assert payload["runs"][1]["error"] == "boom"
    assert set(payload) == {"campaign_id", "name", "configs", "runs"}


def test_write_campaign_log_is_idempotent(tmp_path) -> None:
    cfg = _write_config(tmp_path, "a.yaml", "run:\n  name: x\n")
    log = build_campaign_log("camp", [cfg], runs=())
    p1 = write_campaign_log(log, tmp_path / "out")
    p2 = write_campaign_log(log, tmp_path / "out")
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")


import subprocess
import sys

import pytest

import run_campaign
from nmr.runner import ExperimentRunner, RunResult
from nmr.evaluation import MetricSummary


def _stub_run(tmp_path, monkeypatch) -> None:
    import polars as pl

    def fake_run(self, *, deploy: bool = False) -> RunResult:
        return RunResult(
            run_id="a" * 64,
            oof=pl.DataFrame({"id": ["x"], "era": ["1"], "prediction": [0.5]}),
            metrics=MetricSummary(mean=0.1, std=0.2, sharpe=0.5, max_drawdown=0.05),
            artifact=None,
            manifest={"run_id": "a" * 64, "oof_device": "cpu"},
        )

    monkeypatch.setattr(ExperimentRunner, "run", fake_run)
    monkeypatch.setattr(
        ExperimentRunner,
        "compute_run_id",
        staticmethod(lambda config: "a" * 64),
    )


def test_run_campaign_main_records_and_writes_log(tmp_path, monkeypatch) -> None:
    cfg = _write_config(tmp_path, "a.yaml", "run:\n  name: x\n")
    _stub_run(tmp_path, monkeypatch)
    registry_dir = tmp_path / "registry"
    campaigns_dir = tmp_path / "campaigns"
    rc = run_campaign.main([
        "--config", str(cfg), "--name", "camp",
        "--registry", str(registry_dir), "--campaigns-dir", str(campaigns_dir),
    ])
    assert rc == 0
    assert (registry_dir / ("a" * 64) / "run.json").exists()
    logs = list(campaigns_dir.glob("*.json"))
    assert len(logs) == 1
    payload = json.loads(logs[0].read_text(encoding="utf-8"))
    assert payload["runs"][0]["status"] == "recorded"
    assert payload["runs"][0]["run_id"] == "a" * 64


def test_run_campaign_dry_run_writes_nothing(tmp_path, monkeypatch) -> None:
    cfg = _write_config(tmp_path, "a.yaml", "run:\n  name: x\n")
    _stub_run(tmp_path, monkeypatch)
    rc = run_campaign.main([
        "--config", str(cfg), "--name", "camp",
        "--registry", str(tmp_path / "registry"),
        "--campaigns-dir", str(tmp_path / "campaigns"),
        "--dry-run",
    ])
    assert rc == 0
    assert not (tmp_path / "registry").exists()
    assert not (tmp_path / "campaigns").exists()


def test_run_campaign_error_records_and_returns_1(tmp_path, monkeypatch) -> None:
    cfg = _write_config(tmp_path, "a.yaml", "run:\n  name: x\n")
    _stub_run(tmp_path, monkeypatch)

    def boom(self, *, deploy: bool = False):
        raise RuntimeError("training failed")

    monkeypatch.setattr(ExperimentRunner, "run", boom)
    rc = run_campaign.main([
        "--config", str(cfg), "--name", "camp",
        "--registry", str(tmp_path / "registry"),
        "--campaigns-dir", str(tmp_path / "campaigns"),
    ])
    assert rc == 1
    logs = list((tmp_path / "campaigns").glob("*.json"))
    assert len(logs) == 1
    payload = json.loads(logs[0].read_text(encoding="utf-8"))
    assert payload["runs"][0]["status"] == "error"
    assert "training failed" in payload["runs"][0]["error"]


def test_run_campaign_rejects_no_configs(tmp_path, monkeypatch, capsys) -> None:
    _stub_run(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        run_campaign.main(["--name", "camp"])
