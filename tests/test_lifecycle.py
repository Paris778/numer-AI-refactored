"""Tests for lifecycle: export validity, total stage derivation, ordering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nmr import lifecycle, paths
from nmr.deployment import serialize_predict
from nmr.lifecycle import ExportVersion, StakedRecord


@pytest.fixture(autouse=True)
def _isolated_experiments_root(tmp_path, monkeypatch) -> None:
    """Route all experiment paths under tmp_path; never touch the real repo root."""
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")


def _write_export(slug, scope, run_id, *, training_scope=None, sha=None, meta_json=None):
    slot = paths.export_dir(slug, scope, run_id)
    slot.mkdir(parents=True, exist_ok=True)
    # Write a real deploy artifact: valid_export()'s validity predicate calls
    # load_predict(), which hash-verifies and cloudpickle-loads predict.pkl.
    def dummy_predict(live_features, live_benchmark_models=None):
        return live_features

    serialize_predict(dummy_predict, path=slot / "predict.pkl", feature_names=["f1"])
    if sha is not None:
        (slot / "predict.pkl.manifest.json").write_text(json.dumps({"sha256": sha}))
    ts = training_scope or scope
    (slot / "export.json").write_text(
        json.dumps({"family": slug, "training_scope": ts,
                    "promoted_from_run_id": run_id, "promoted_at": "2026-08-26T10:00:00+00:00",
                    "config": {}})
    )
    if scope == "partial":
        (slot / "scorecard.json").write_text(json.dumps({"schema_version": 3}))
    if meta_json is not None:
        paths.experiment_dir(slug).joinpath("meta.json").write_text(json.dumps(meta_json))
    return slot


def test_stage_derivation_total():
    # uninitialized: dir exists, no run.json
    paths.experiment_dir("fam1").mkdir(parents=True, exist_ok=True)
    assert lifecycle.derive_stage("fam1", None) == ("uninitialized", "none")
    # research: run.json present, no exports
    paths.run_json_path("fam1", "a" * 64).parent.mkdir(parents=True, exist_ok=True)
    paths.run_json_path("fam1", "a" * 64).write_text("{}")
    assert lifecycle.derive_stage("fam1", None) == ("research", "none")
    # partial: valid partial export, no full
    _write_export("fam1", "partial", "b" * 64)
    assert lifecycle.derive_stage("fam1", None) == ("partial", "none")
    # degraded: valid full export, dangling pointer
    _write_export("fam1", "full", "c" * 64)
    assert lifecycle.derive_stage("fam1", None) == ("degraded", "degraded")
    # full: pointer at valid slot
    paths.current_pointer_path("fam1").write_text(json.dumps({"run_id": "c" * 64}))
    assert lifecycle.derive_stage("fam1", None) == ("full", "full")
    # staked: active stake on valid full
    staked = StakedRecord(run_id="c" * 64, scope="full", numerai_model_id="m1",
                          staked_at="2026-08-26T11:00:00+00:00", status="active")
    assert lifecycle.derive_stage("fam1", staked) == ("staked", "full")


def test_staked_stale_when_export_invalid():
    staked = StakedRecord(run_id="dead" * 16, scope="full", numerai_model_id="m1",
                          staked_at="2026-08-26T11:00:00+00:00", status="active")
    assert lifecycle.derive_stage("fam1", staked)[0] != "staked"


def test_export_identity_binding():
    # manifest promoted_from_run_id != slot dir run_id -> invalid
    slot = _write_export("fam2", "full", "a" * 64)
    (slot / "export.json").write_text(
        json.dumps({"family": "fam2", "training_scope": "full",
                    "promoted_from_run_id": "b" * 64, "promoted_at": "x", "config": {}})
    )
    assert lifecycle.valid_export("fam2", "full", "a" * 64) is None


def test_sort_exports_deterministic():
    e1 = ExportVersion(family="f", scope="partial", run_id="b" * 64, slot_dir=Path("."),
                       training_scope="partial", promoted_at="2026-08-26T10:00:00+00:00",
                       training_rows=None, training_era_range=None, config={},
                       tier4_gate_passed=None, rehearsal=False)
    e2 = ExportVersion(family="f", scope="partial", run_id="a" * 64, slot_dir=Path("."),
                       training_scope="partial", promoted_at="2026-08-26T10:00:00+00:00",
                       training_rows=None, training_era_range=None, config={},
                       tier4_gate_passed=None, rehearsal=False)
    e3 = ExportVersion(family="f", scope="partial", run_id="c" * 64, slot_dir=Path("."),
                       training_scope="partial", promoted_at="2026-08-25T10:00:00+00:00",
                       training_rows=None, training_era_range=None, config={},
                       tier4_gate_passed=None, rehearsal=False)
    got = lifecycle.sort_exports([e1, e2, e3])
    assert [e.run_id for e in got] == ["a" * 64, "b" * 64, "c" * 64]
