"""Tests for the promotion writer (nmr.promote) — the money-path terminus.

Covers: gate enforcement + override recording, config normalization recording,
supplemental feature-set identity, slot/pointer overwrite guards, the
spawned include_validation path, and manifest/artifact validity (load_predict
round-trip with (0,1) output).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from nmr import experiment_store, lifecycle, paths
from nmr.benchmark import Tier4GateConfig
from nmr.config import (
    DataConfig,
    EnsembleConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    RiskConfig,
    RunConfig,
    SplitConfig,
)
from nmr.data import IngestionAgent
from nmr.deployment import load_predict
from nmr.promote import (
    PromotionResult,
    promote_full_version,
    rehearse_promotion,
)
from nmr.promote import (
    _full_history_frame as _orig_full_history_frame,
)
from nmr.runner import ExperimentRunner
from nmr.scorecard import CROSSCHECK_N_TRIALS

_RID = "a" * 64


@pytest.fixture(autouse=True)
def _isolated_experiments_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the experiment layout to tmp — promotions never touch the repo tree."""
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")


def _make_data(root: Path, *, validation_eras: int = 8) -> Path:
    version_dir = root / "vtest"
    version_dir.mkdir(parents=True, exist_ok=True)
    train_eras = 8
    total_eras = train_eras + validation_eras
    rows = []
    for era in range(1, total_eras + 1):
        for idx in range(8):
            f1 = idx * 0.1
            f2 = (idx % 4) * 0.1
            # Era-dependent, NON-proportional slopes so the per-era target
            # ordering (and therefore per-era corr) varies: a near-constant
            # corr series makes scipy's bias=False skew/kurt collapse to NaN
            # and deflated_sharpe raises — the cross-check needs finite
            # moments. (Measured: corr std ~0.05 across eras.)
            rows.append(
                {
                    "era": f"{era:04d}",
                    "id": f"{era}_{idx}",
                    "f1": f1,
                    "f2": f2,
                    "target": (0.5 + 0.03 * era) * f1 - (0.05 * era) * f2 + 0.01 * era,
                }
            )
    pl.DataFrame([r for r in rows if int(r["era"]) <= train_eras]).write_parquet(
        version_dir / "train.parquet"
    )
    val_rows = [r for r in rows if int(r["era"]) > train_eras]
    pl.DataFrame(val_rows).write_parquet(version_dir / "validation.parquet")
    # meta_model.parquet over the validation eras — required by the partial
    # cross-check (MMC) and the runner's validation stage.
    pl.DataFrame(val_rows).select(["era", "id"]).with_columns(
        pl.lit(0.35).alias("numerai_meta_model")
    ).write_parquet(version_dir / "meta_model.parquet")
    # Rehearsal needs features.json + live files in the same layout as v5.3.
    version_dir.joinpath("features.json").write_text(
        json.dumps({"feature_sets": {"small": ["f1", "f2"]}}), encoding="utf-8"
    )
    pl.DataFrame(
        {
            "era": ["0017"] * 5,
            "id": [f"live_{i}" for i in range(5)],
            "f1": [0.0, 0.1, 0.2, 0.3, 0.05],
            "f2": [0.0, 0.1, 0.0, 0.1, 0.2],
            "target": [0.5] * 5,
        }
    ).write_parquet(version_dir / "live.parquet")
    pl.DataFrame(
        {
            "era": ["0017"] * 5,
            "id": [f"live_{i}" for i in range(5)],
            "benchmark": [0.5] * 5,
        }
    ).write_parquet(version_dir / "live_benchmark_models.parquet")
    return root


def _config(tmp_data: Path) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            version="vtest",
            feature_set="small",
            targets=("target",),
            data_dir=tmp_data,
        ),
        split=SplitConfig(purge_eras=8),
        model=ModelConfig(backend="lightgbm", preset="fast"),
        evaluation=EvalConfig(),
        risk=RiskConfig(neutralization_proportion=0.0),
        ensemble=EnsembleConfig(),
        run=RunConfig(
            seed=42, artifacts_dir=tmp_data / "artifacts", name="brb1-lgbm-v6"
        ),
    )


def _stored_config_dict(tmp_data: Path) -> dict:
    stored = json.loads(json.dumps(dataclasses.asdict(_config(tmp_data)), default=str))
    stored["data"].pop("horizon", None)  # legacy stored configs predate the field
    stored["split"]["embargo_eras"] = 4  # legacy stored value (all 29 registry rows)
    return stored


def _passing_scorecard() -> dict:
    return {
        "corr": 0.05, "corr_sharpe_ac": 1.0, "fnc": 0.03,
        "gain_to_pain_ratio": 2.0, "cagr_1y": 0.1,
        "deflated_sharpe": 0.9, "turnover_mean": None,
    }


def _write_registry(
    *,
    run_id: str = _RID,
    stored_config: dict,
    scorecard: dict | None = None,
    feature_cols: list[str] | None = None,
    weights: list[float] | None = None,
    supplemental_sha: str | None = None,
) -> Path:
    """Write the run record through experiment_store (experiments layout)."""
    manifest = {
        "config": stored_config,
        "feature_cols": feature_cols if feature_cols is not None else ["f1", "f2"],
        "weights": weights if weights is not None else [1.0],
    }
    if supplemental_sha is not None:
        manifest["supplemental_feature_sets_sha256"] = supplemental_sha
    payload = {
        "run_id": run_id,
        "metrics": {"mean": 0.1, "sharpe": 0.5},
        "manifest": manifest,
        "scorecard": scorecard if scorecard is not None else _passing_scorecard(),
    }
    return experiment_store.record_run("brb1-lgbm-v6", run_id, payload)


def _promote(tmp_path: Path, **kwargs) -> object:
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data))
    return promote_full_version(
        _RID,
        "brb1-lgbm-v6",
        **kwargs,
    )


def test_promote_happy_path_manifest_and_artifact(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data))
    result = promote_full_version(
        _RID, "brb1-lgbm-v6"
    )

    assert result.scope == "full"
    assert result.cross_check_path is None
    assert result.artifact_path == (
        paths.export_dir("brb1-lgbm-v6", "full", _RID) / "predict.pkl"
    )
    assert result.artifact_path.is_file()
    assert result.tier4_gate_passed is True
    assert result.override_used is False

    # lifecycle read-side validates the published export (pointer + slot).
    version = lifecycle.valid_export("brb1-lgbm-v6", "full", _RID)
    assert version is not None
    assert version.family == "brb1-lgbm-v6"
    assert version.run_id == _RID
    assert version.scope == "full"
    assert version.training_scope == "full"
    assert [v.run_id for v in lifecycle.scan_valid_exports("brb1-lgbm-v6", "full")] == [_RID]
    pointer = paths.current_pointer_path("brb1-lgbm-v6")
    assert json.loads(pointer.read_text(encoding="utf-8"))["run_id"] == _RID

    # The git-tracked promotion record is export.json (not manifest.json).
    record = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.manifest_path.name == "export.json"
    assert record["tier4_gate_passed"] is True
    assert record["override_used"] is False
    assert record["training_scope"] == "full"
    normalizations = {n["field"]: n for n in record["config_normalizations"]}
    assert normalizations["split.embargo_eras"] == {"field": "split.embargo_eras", "from": 4, "to": 0}
    assert normalizations["data.horizon"] == {"field": "data.horizon", "from": None, "to": "20D"}
    # First-class training provenance (review directive 2026-08-18): a genuine
    # full version is explicitly NOT a rehearsal and states what it trained on.
    assert record["rehearsal"] is False
    assert record["training_rows"] > 0
    assert len(record["training_era_range"]) == 2

    # The artifact loads and the raw contract output is strictly in (0,1).
    predict_fn = load_predict(result.artifact_path)
    live = pd.DataFrame(
        {"f1": [0.0, 0.1, 0.2, 0.3, 0.05, 0.15], "f2": [0.0, 0.1, 0.0, 0.1, 0.2, 0.0]},
        index=[f"id_{i}" for i in range(6)],
    )
    prediction = predict_fn(live)
    values = prediction["prediction"].to_numpy()
    assert ((values > 0) & (values < 1)).all()

    # D5 acceptance gate: the real contract with BOTH arguments, raw output
    # validated unaided by the official validator (no build_submission in the
    # assertion path).
    from nmr.submission import accept_promoted_artifact

    bench = pd.DataFrame({"dummy": [0.1] * 6}, index=live.index)
    raw = accept_promoted_artifact(
        result.artifact_path, live_features=live, live_benchmark_models=bench
    )
    assert list(raw.columns) == ["prediction"]
    assert ((raw["prediction"] > 0) & (raw["prediction"] < 1)).all()


def test_promote_gate_refusal_and_override(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    failing = _passing_scorecard()
    failing["corr"] = 0.01  # below corr_min 0.0286
    _write_registry(stored_config=_stored_config_dict(data), scorecard=failing)

    with pytest.raises(ValueError, match="tier-4 promotion gate"):
        promote_full_version(
            _RID, "brb1-lgbm-v6"
        )
    # The rehearsal path: override records the failure in the artifact's own
    # manifest — a failed-gate artifact carries its verdict with it.
    result = promote_full_version(
        _RID,
        "brb1-lgbm-v6",
        override_gate=True,
    )
    assert result.tier4_gate_passed is False
    assert result.override_used is True
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["tier4_gate_passed"] is False
    assert manifest["override_used"] is True
    assert manifest["tier4_receipts"]["corr"]["passed"] is False


def test_promote_missing_scorecard_refused(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data), scorecard={}
    )
    with pytest.raises(ValueError, match="no validation scorecard"):
        promote_full_version(
            _RID, "brb1-lgbm-v6"
        )


def test_promote_supplemental_identity_mismatch_refused(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    # Stored run carries a supplemental SHA but the config has no path.
    _write_registry(
        stored_config=_stored_config_dict(data),
        supplemental_sha="f" * 64,
    )
    with pytest.raises(ValueError, match="supplemental feature-set identity"):
        promote_full_version(
            _RID, "brb1-lgbm-v6"
        )


def test_promote_supplemental_identity_match(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    supp = data / "vtest" / "derived_feature_sets.json"
    supp.write_text(json.dumps({"sets": {"medium": ["f1", "f2"]}}), encoding="utf-8")
    stored = _stored_config_dict(data)
    stored["data"]["supplemental_feature_sets"] = str(supp)
    sha = ExperimentRunner._supplemental_fingerprint(supp)
    _write_registry(stored_config=stored, supplemental_sha=sha
    )
    result = promote_full_version(
        _RID, "brb1-lgbm-v6"
    )
    assert result.artifact_path.is_file()


def test_promote_overwrite_and_repoint_guards(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data), run_id=_RID)
    promote_full_version(
        _RID, "brb1-lgbm-v6"
    )
    # Same slot again WITHOUT force: immutable — refused (a slot is never
    # overwritten; force only gates repointing current.json).
    with pytest.raises(ValueError, match="already exists"):
        promote_full_version(
            _RID, "brb1-lgbm-v6"
        )
    # A second run repointing current.json away: refused without force.
    other = "b" * 64
    _write_registry(stored_config=_stored_config_dict(data), run_id=other)
    with pytest.raises(ValueError, match="repointing requires force"):
        promote_full_version(
            other, "brb1-lgbm-v6"
        )
    promote_full_version(
        other, "brb1-lgbm-v6", force=True,
    )
    assert sorted(v.run_id for v in lifecycle.scan_valid_exports("brb1-lgbm-v6", "full")) == [_RID, other]
    pointer = json.loads(
        paths.current_pointer_path("brb1-lgbm-v6").read_text(encoding="utf-8")
    )
    assert pointer["run_id"] == other


def test_promote_force_recovery_repoints_existing_valid_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BLOCKING 1 (b): force=True against an existing VALID slot for a
    DIFFERENT run_id repoints the pointer to that slot WITHOUT refitting (the
    fit path is never entered — asserted via the spy) — the slot is
    immutable, the pointer is repaired."""
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data), run_id=_RID)
    promote_full_version(_RID, "brb1-lgbm-v6")  # slot A + pointer A
    other = "b" * 64
    _write_registry(stored_config=_stored_config_dict(data), run_id=other)
    promote_full_version(other, "brb1-lgbm-v6", force=True)  # slot B + pointer B

    def _must_not_fit(*args, **kwargs):
        raise AssertionError("recovery must not refit")

    monkeypatch.setattr("nmr.promote._build_deploy_pipeline", _must_not_fit)
    result = promote_full_version(_RID, "brb1-lgbm-v6", force=True)
    pointer = json.loads(
        paths.current_pointer_path("brb1-lgbm-v6").read_text(encoding="utf-8")
    )
    assert pointer["run_id"] == _RID
    assert result.artifact_path == paths.export_dir("brb1-lgbm-v6", "full", _RID) / "predict.pkl"
    # The slot itself is untouched (immutability preserved).
    assert lifecycle.valid_export("brb1-lgbm-v6", "full", _RID) is not None
    assert lifecycle.current_full_status("brb1-lgbm-v6") == "full"


def test_promote_pointer_write_failure_recoverable_via_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BLOCKING 1 (a): a promotion whose current.json pointer write fails
    leaves the slot published + pointer missing (the family reads 'degraded');
    re-running with force=True repoints the pointer at the existing valid slot
    and the family returns to 'full' — the documented recovery is executable."""
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data), run_id=_RID)

    import nmr.promote as promote_module

    original_atomic_write_text = promote_module.atomic_write_text

    def _fail_pointer_write(path, text, *, encoding="utf-8"):
        if Path(path).name == "current.json":
            raise OSError("simulated pointer-write failure")
        return original_atomic_write_text(path, text, encoding=encoding)

    # Patch the promotion writer's own atomic_write_text binding so the
    # pointer write (and only that) fails; the staging export.json writes
    # pass through.
    with monkeypatch.context() as ctx:
        ctx.setattr(promote_module, "atomic_write_text", _fail_pointer_write)
        with pytest.raises(OSError, match="simulated pointer-write failure"):
            promote_full_version(_RID, "brb1-lgbm-v6")

    # Slot published, pointer missing -> degraded.
    slot = paths.export_dir("brb1-lgbm-v6", "full", _RID)
    assert slot.is_dir()
    assert not paths.current_pointer_path("brb1-lgbm-v6").exists()
    assert lifecycle.current_full_status("brb1-lgbm-v6") == "degraded"

    # force=True now repairs the pointer WITHOUT refitting.
    def _must_not_fit(*args, **kwargs):
        raise AssertionError("recovery must not refit")

    with monkeypatch.context() as ctx:
        ctx.setattr("nmr.promote._build_deploy_pipeline", _must_not_fit)
        result = promote_full_version(_RID, "brb1-lgbm-v6", force=True)
    assert result.artifact_path == slot / "predict.pkl"
    assert lifecycle.current_full_status("brb1-lgbm-v6") == "full"
    assert json.loads(
        paths.current_pointer_path("brb1-lgbm-v6").read_text(encoding="utf-8")
    )["run_id"] == _RID


def test_promote_force_recovery_refuses_invalid_slot(tmp_path: Path) -> None:
    """BLOCKING 1: force=True against an existing but INVALID slot refuses to
    repoint — recovery only ever points at a valid export."""
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data), run_id=_RID)
    promote_full_version(_RID, "brb1-lgbm-v6")
    slot = paths.export_dir("brb1-lgbm-v6", "full", _RID)
    (slot / "predict.pkl").unlink()  # hollow the slot -> invalid
    with pytest.raises(ValueError, match="not a VALID full export"):
        promote_full_version(_RID, "brb1-lgbm-v6", force=True)


def test_promote_rejects_invalid_run_id_and_family(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data))
    with pytest.raises(ValueError, match="64-char"):
        promote_full_version(
            "not-a-run-id", "brb1-lgbm-v6",
        )
    with pytest.raises(ValueError, match="invalid family name"):
        promote_full_version(
            _RID, "../evil"
        )
    with pytest.raises(FileNotFoundError, match="has no record"):
        promote_full_version(
            "c" * 64, "brb1-lgbm-v6",
        )


def test_promote_spawn_path_trains_on_train_plus_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the fresh-process full-history path at small scale (the D7
    rehearsal forces the same path via NMR_FULL_HISTORY_SPAWN_MIN_BYTES) and
    prove include_validation wiring: the child re-reads train+validation."""
    monkeypatch.setenv("NMR_FULL_HISTORY_SPAWN_MIN_BYTES", "1")
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data))
    result = promote_full_version(
        _RID, "brb1-lgbm-v6",
    )
    assert result.artifact_path.is_file()
    predict_fn = load_predict(result.artifact_path)
    live = pd.DataFrame(
        {"f1": [0.05, 0.15, 0.25], "f2": [0.0, 0.1, 0.2]},
        index=[f"id_{i}" for i in range(3)],
    )
    values = predict_fn(live)["prediction"].to_numpy()
    assert ((values > 0) & (values < 1)).all()


def test_rehearse_promotion_end_to_end(tmp_path: Path) -> None:
    """D7 Stage-1 rehearsal: truncated data dir, forced spawn path, measured
    peak RAM estimate, and the acceptance criterion on the live frame."""
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data))
    result = rehearse_promotion(
        _RID,
        "brb1-lgbm-v6",
        rehearsal_data_root=tmp_path / "rehearsal",
        train_eras=6,
        validation_eras=6,
    )
    assert result.acceptance_passed is True
    assert result.artifact_path.is_file()
    assert result.train_validation_rows > 0
    estimate = json.loads(result.ram_estimate_path.read_text(encoding="utf-8"))
    assert estimate["train_validation_rows"] == result.train_validation_rows
    assert estimate["peak_bytes"] is None or estimate["peak_bytes"] > 0
    assert estimate["measured_at"]
    # The rehearsal artifact states its own scope and is never the current
    # pointer (review directive 2026-08-18).
    slot_manifest = json.loads(
        (result.artifact_path.parent / "export.json").read_text(encoding="utf-8")
    )
    assert slot_manifest["rehearsal"] is True
    assert slot_manifest["training_rows"] == result.train_validation_rows
    assert not paths.current_pointer_path("brb1-lgbm-v6").exists()


def _gate() -> Tier4GateConfig:
    return Tier4GateConfig(
        corr_min=0.0286,
        corr_sharpe_ac_min=0.5,
        fnc_min=0.01,
        deflated_sharpe_min=0.3,
        gain_to_pain_min=1.0,
        cagr_min=0.05,
        turnover_max=0.05,
    )


def test_evaluate_gate_missing_evidence_fails() -> None:
    """A hard field with no measured value is a failure — never promoted on faith."""
    from nmr.promote import _evaluate_gate

    scorecard = {k: v for k, v in _passing_scorecard().items() if k != "corr"}
    passed, receipts = _evaluate_gate(scorecard, _gate())
    assert passed is False
    assert receipts["corr"]["measured"] is None
    assert receipts["corr"]["passed"] is False
    assert receipts["cagr_1y"]["passed"] is True  # other fields still evaluated


def test_evaluate_gate_strict_cagr_fails_at_threshold() -> None:
    """cagr_1y uses strict `>`: equality at the threshold fails (promote.py:165)."""
    from nmr.promote import _evaluate_gate

    scorecard = _passing_scorecard()
    scorecard["cagr_1y"] = 0.05  # exactly cagr_min — strict needs strictly greater
    passed, receipts = _evaluate_gate(scorecard, _gate())
    assert passed is False
    assert receipts["cagr_1y"]["passed"] is False


def test_load_run_record_corrupt_json(tmp_path: Path) -> None:
    from nmr.promote import _load_run_record

    run_dir = paths.run_dir("brb1-lgbm-v6", _RID)
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt run.json"):
        _load_run_record("brb1-lgbm-v6", _RID)


def test_load_run_record_non_mapping(tmp_path: Path) -> None:
    from nmr.promote import _load_run_record

    run_dir = paths.run_dir("brb1-lgbm-v6", _RID)
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a mapping"):
        _load_run_record("brb1-lgbm-v6", _RID)


def test_load_run_record_missing_fails_loud(tmp_path: Path) -> None:
    from nmr.promote import _load_run_record

    with pytest.raises(FileNotFoundError, match="has no record"):
        _load_run_record("brb1-lgbm-v6", _RID)


def test_ram_guard_curve_path_passes_when_under_guard(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Zero-intercept/zero-slope curve → extrapolated commit ≈ 0 → guard passes,
    exercising the fitted-curve branch end to end (promote.py:259-297)."""
    from nmr.promote import _ram_guard

    data = _make_data(tmp_path / "data")
    config = _config(data)
    reports = paths.shared_reports_dir(config.run.artifacts_dir)
    reports.mkdir(parents=True)
    (reports / "ram_curve.json").write_text(
        json.dumps(
            {
                "fit": {"intercept_gib": 0.0, "slope_gib_per_row": 0.0},
                "fit_ws": {"intercept_gib": 0.0, "slope_gib_per_row": 0.0},
                "points": [{"parent_commit_gib": 0.0, "parent_ws_gib": 0.0}],
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.INFO, logger="nmr.promote"):
        _ram_guard(config, scope="full")  # must not raise
    assert "extrapolated full-version combined commit" in caplog.text


def _huge_curve(*, commit_slope: float, ws_slope: float, ws_intercept: float = 0.0) -> str:
    return json.dumps(
        {
            "fit": {"intercept_gib": 0.0, "slope_gib_per_row": commit_slope},
            "fit_ws": {"intercept_gib": ws_intercept, "slope_gib_per_row": ws_slope},
            "points": [{"parent_commit_gib": 0.0, "parent_ws_gib": 0.0}],
        }
    )


def test_ram_guard_over_ceiling_raises(tmp_path: Path) -> None:
    """combined commit > 45 GiB ceiling → RuntimeError naming the guard."""
    from nmr.promote import _ram_guard

    data = _make_data(tmp_path / "data")
    reports = paths.shared_reports_dir(_config(data).run.artifacts_dir)
    reports.mkdir(parents=True)
    (reports / "ram_curve.json").write_text(_huge_curve(commit_slope=1e9, ws_slope=0.0), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exceeds the 45 GiB guard"):
        _ram_guard(_config(data), scope="full")


def test_ram_guard_over_commit_limit_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """combined commit under the ceiling but over the machine commit limit."""
    from nmr.models import _machine_memory_limits
    from nmr.promote import _ram_guard

    _, commit_limit = _machine_memory_limits()
    if commit_limit is None:
        pytest.skip("platform reports no commit limit (Unix)")
    monkeypatch.setattr("nmr.promote._RAM_GUARD_BYTES", 2**70)
    data = _make_data(tmp_path / "data")
    reports = paths.shared_reports_dir(_config(data).run.artifacts_dir)
    reports.mkdir(parents=True)
    (reports / "ram_curve.json").write_text(_huge_curve(commit_slope=1e9, ws_slope=0.0), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exceeds the machine commit limit"):
        _ram_guard(_config(data), scope="full")


def test_ram_guard_over_working_set_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """commit small but combined working set > 85% of physical RAM → thrash refusal."""
    from nmr.models import _machine_memory_limits
    from nmr.promote import _ram_guard

    physical, _ = _machine_memory_limits()
    if physical is None:
        pytest.skip("platform reports no physical RAM")
    monkeypatch.setattr("nmr.promote._RAM_GUARD_BYTES", 2**70)
    data = _make_data(tmp_path / "data")
    reports = paths.shared_reports_dir(_config(data).run.artifacts_dir)
    reports.mkdir(parents=True)
    (reports / "ram_curve.json").write_text(_huge_curve(commit_slope=0.0, ws_slope=1e9), encoding="utf-8")
    with pytest.raises(RuntimeError, match="would thrash"):
        _ram_guard(_config(data), scope="full")


def _write_estimate(reports: Path, payload: dict) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "full_version_ram_estimate.json").write_text(json.dumps(payload), encoding="utf-8")


def test_ram_guard_estimate_path_passes_when_under_guard(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No curve on disk → single-point estimate, through-origin extrapolation."""
    from nmr.promote import _ram_guard

    data = _make_data(tmp_path / "data")
    config = _config(data)
    _write_estimate(
        paths.shared_reports_dir(config.run.artifacts_dir),
        {
            "peak_commit_bytes": 1,
            "peak_bytes": 1,
            "parent_peak_commit_bytes": 0,
            "parent_peak_bytes": 0,
            "train_validation_rows": 1,
        },
    )
    with caplog.at_level(logging.WARNING, logger="nmr.promote"):
        _ram_guard(config, scope="full")  # must not raise
    assert "single-point estimate extrapolation" in caplog.text


def test_ram_guard_estimate_missing_dual_metric_skips(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from nmr.promote import _ram_guard

    data = _make_data(tmp_path / "data")
    config = _config(data)
    _write_estimate(
        paths.shared_reports_dir(config.run.artifacts_dir),
        {"peak_bytes": 1, "parent_peak_bytes": 0, "train_validation_rows": 1},
    )
    with caplog.at_level(logging.WARNING, logger="nmr.promote"):
        _ram_guard(config, scope="full")  # must not raise
    assert "lacks dual-metric data" in caplog.text


def test_ram_guard_corrupt_curve_falls_back_to_estimate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from nmr.promote import _ram_guard

    data = _make_data(tmp_path / "data")
    config = _config(data)
    reports = paths.shared_reports_dir(config.run.artifacts_dir)
    reports.mkdir(parents=True)
    (reports / "ram_curve.json").write_text("{corrupt", encoding="utf-8")
    _write_estimate(
        reports,
        {
            "peak_commit_bytes": 1,
            "peak_bytes": 1,
            "parent_peak_commit_bytes": 0,
            "parent_peak_bytes": 0,
            "train_validation_rows": 1,
        },
    )
    with caplog.at_level(logging.WARNING, logger="nmr.promote"):
        _ram_guard(config, scope="full")  # must not raise
    assert "unreadable RAM curve" in caplog.text


def test_ram_guard_corrupt_estimate_skips(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from nmr.promote import _ram_guard

    data = _make_data(tmp_path / "data")
    config = _config(data)
    reports = paths.shared_reports_dir(config.run.artifacts_dir)
    reports.mkdir(parents=True)
    (reports / "full_version_ram_estimate.json").write_text("{corrupt", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="nmr.promote"):
        _ram_guard(config, scope="full")  # must not raise
    assert "unreadable RAM estimate" in caplog.text


def test_promote_manifest_without_config_refused(tmp_path: Path) -> None:
    experiment_store.record_run(
        "brb1-lgbm-v6",
        _RID,
        {"run_id": _RID, "manifest": {}, "scorecard": _passing_scorecard()},
    )
    with pytest.raises(ValueError, match="no config dict"):
        promote_full_version(
            _RID, "brb1-lgbm-v6"
        )


def test_promote_gate_missing_from_yaml_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data))
    monkeypatch.setattr(
        "nmr.promote.load_benchmark_file", lambda path: types.SimpleNamespace(gate=None)
    )
    with pytest.raises(ValueError, match="no gate"):
        promote_full_version(
            _RID, "brb1-lgbm-v6"
        )


def test_promote_corrupt_current_pointer_requires_force(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data))
    pointer = paths.current_pointer_path("brb1-lgbm-v6")
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text("{corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="repointing requires force"):
        promote_full_version(
            _RID, "brb1-lgbm-v6"
        )


def test_promote_missing_feature_cols_refused(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data), feature_cols=[])
    with pytest.raises(ValueError, match="no feature_cols"):
        promote_full_version(
            _RID, "brb1-lgbm-v6"
        )


def test_promote_weight_count_mismatch_refused(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data), weights=[1.0, 1.0]
    )
    with pytest.raises(ValueError, match="do not match targets"):
        promote_full_version(
            _RID, "brb1-lgbm-v6"
        )


def test_build_truncated_data_missing_asset(tmp_path: Path) -> None:
    from nmr.promote import _build_truncated_data

    stored = _stored_config_dict(tmp_path / "data")
    stored["data"]["data_dir"] = str(tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="data assets missing"):
        _build_truncated_data(
            stored, tmp_path / "rehearsal", train_eras=1, validation_eras=1
        )


def test_build_truncated_data_insufficient_eras(tmp_path: Path) -> None:
    from nmr.promote import _build_truncated_data

    data = _make_data(tmp_path / "data")  # 8 train eras
    stored = _stored_config_dict(data)
    with pytest.raises(ValueError, match="rehearsal needs 9/1"):
        _build_truncated_data(
            stored, tmp_path / "rehearsal", train_eras=9, validation_eras=1
        )


def test_measure_full_history_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs the real promotion training path (spawn forced, train+validation)
    and returns measured peaks — the curve measurement, exercised at toy scale."""
    from nmr.promote import measure_full_history_peak

    monkeypatch.setenv("NMR_FULL_HISTORY_SPAWN_MIN_BYTES", "1")
    data = _make_data(tmp_path / "data")
    stored = _stored_config_dict(data)
    child_ws, child_commit, parent_ws, parent_commit, rows = measure_full_history_peak(
        stored,
        feature_cols=["f1", "f2"],
        target_cols=["target"],
        weights=[1.0],
        data_dir=data,
        seed=42,
    )
    assert rows > 0
    assert child_ws is None or child_ws > 0
    assert child_commit is None or child_commit > 0
    assert parent_ws is None or parent_ws > 0
    assert parent_commit is None or parent_commit > 0


def test_rehearse_restores_env_and_removes_stale_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env override is restored to its prior value and a stale current.json
    that references a REHEARSAL slot is removed (a rehearsal is never the full
    version — final review I4: only rehearsal-slot pointers are unlinked)."""
    monkeypatch.setenv("NMR_FULL_HISTORY_SPAWN_MIN_BYTES", "7777")
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data), run_id=_RID)
    # A prior rehearsal published a rehearsal slot; the stale pointer still
    # references it.
    slot = paths.export_dir("brb1-lgbm-v6", "full", _RID)
    slot.mkdir(parents=True, exist_ok=True)
    (slot / "export.json").write_text(
        json.dumps({"rehearsal": True}), encoding="utf-8"
    )
    pointer = paths.current_pointer_path("brb1-lgbm-v6")
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps({"run_id": _RID}), encoding="utf-8")
    other = "b" * 64
    _write_registry(stored_config=_stored_config_dict(data), run_id=other)
    result = rehearse_promotion(
        other,
        "brb1-lgbm-v6",
        rehearsal_data_root=tmp_path / "rehearsal",
        train_eras=6,
        validation_eras=6,
    )
    assert result.acceptance_passed is True
    assert os.environ["NMR_FULL_HISTORY_SPAWN_MIN_BYTES"] == "7777"
    assert not pointer.exists()


def test_rehearse_preserves_genuine_full_pointer(tmp_path: Path) -> None:
    """Final review I4: rehearsing in a family with a REAL full export must
    leave the genuine current.json untouched — the family stays 'full' instead
    of silently dropping to 'degraded' (the regression: the rehearsal path
    unlinked the pointer unconditionally)."""
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data), run_id=_RID)
    promote_full_version(_RID, "brb1-lgbm-v6")  # genuine full export + pointer
    assert lifecycle.current_full_status("brb1-lgbm-v6") == "full"
    other = "b" * 64
    _write_registry(stored_config=_stored_config_dict(data), run_id=other)
    rehearse_promotion(
        other,
        "brb1-lgbm-v6",
        rehearsal_data_root=tmp_path / "rehearsal",
        train_eras=6,
        validation_eras=6,
    )
    pointer = paths.current_pointer_path("brb1-lgbm-v6")
    assert pointer.is_file()
    assert json.loads(pointer.read_text(encoding="utf-8"))["run_id"] == _RID
    assert lifecycle.current_full_status("brb1-lgbm-v6") == "full"


def test_rehearse_manifest_without_config_refused(tmp_path: Path) -> None:
    experiment_store.record_run(
        "brb1-lgbm-v6",
        _RID,
        {"run_id": _RID, "manifest": {}, "scorecard": _passing_scorecard()},
    )
    with pytest.raises(ValueError, match="no config dict"):
        rehearse_promotion(
            _RID, "brb1-lgbm-v6"
        )


def _fake_promotion_result(tmp_path: Path) -> PromotionResult:
    artifact = tmp_path / "fake_predict.pkl"
    artifact.write_bytes(b"not-a-real-model")
    return PromotionResult(
        artifact_path=artifact,
        manifest_path=tmp_path / "export.json",
        run_id=_RID,
        family="brb1-lgbm-v6",
        tier4_gate_passed=False,
        override_used=True,
        scope="full",
    )


def test_rehearse_missing_feature_cols_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The post-promotion feature_cols check (promote.py:870-872) — the promotion
    itself is stubbed out so the test never fits a model."""
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data), feature_cols=[]
    )
    fake = _fake_promotion_result(tmp_path)
    monkeypatch.setattr("nmr.promote.promote_full_version", lambda *a, **k: fake)
    with pytest.raises(ValueError, match="no feature_cols"):
        rehearse_promotion(
            _RID,
            "brb1-lgbm-v6",
            rehearsal_data_root=tmp_path / "rehearsal",
            train_eras=6,
            validation_eras=6,
        )


def test_rehearse_acceptance_failure_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The Phase-D acceptance criterion is NOT overridable: a failed raw-contract
    validation is logged at ERROR and re-raised (promote.py:883-894)."""
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data))
    fake = _fake_promotion_result(tmp_path)
    monkeypatch.setattr("nmr.promote.promote_full_version", lambda *a, **k: fake)

    def _boom(*args, **kwargs) -> None:
        raise ValueError("boom")

    monkeypatch.setattr("nmr.submission.accept_promoted_artifact", _boom)
    with (
        caplog.at_level(logging.ERROR, logger="nmr.promote"),
        pytest.raises(ValueError, match="boom"),
    ):
        rehearse_promotion(
            _RID,
            "brb1-lgbm-v6",
            rehearsal_data_root=tmp_path / "rehearsal",
            train_eras=6,
            validation_eras=6,
        )
    assert "acceptance FAILED" in caplog.text


# --- Task 8: scope, fit-phase isolation, cross-check, atomic staging ---------

def test_train_only_scope_fits_train_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train_only promotes into exports/partial/<run_id>/ with persisted
    training_scope "partial", and the FIT phase never opens validation. The
    fit-phase spy records IngestionAgent.load calls only for the duration of
    the fit — the post-fit cross-check legitimately opens validation later."""
    data = _make_data(tmp_path / "data", validation_eras=32)
    _write_registry(stored_config=_stored_config_dict(data))
    opened: list[str] = []
    original_load = IngestionAgent.load

    def _recording_load(self, split, **kwargs):
        opened.append(split)
        return original_load(self, split, **kwargs)

    def _spy_frame(config, feature_cols, target_cols, orchestrator, *, scope):
        monkeypatch.setattr(IngestionAgent, "load", _recording_load)
        try:
            return _orig_full_history_frame(
                config, feature_cols, target_cols, orchestrator, scope=scope
            )
        finally:
            monkeypatch.setattr(IngestionAgent, "load", original_load)

    monkeypatch.setattr("nmr.promote._full_history_frame", _spy_frame)
    result = promote_full_version(
        _RID,
        "brb1-lgbm-v6",
        override_gate=True,
        scope="train_only",
    )
    assert result.scope == "train_only"
    assert result.cross_check_path is not None
    export = lifecycle.valid_export("brb1-lgbm-v6", "partial", _RID)
    assert export is not None
    assert export.training_scope == "partial"
    assert opened == ["train"]  # the fit-phase spy saw only train.parquet


def test_full_history_frame_train_only_returns_train_only_rows(
    tmp_path: Path,
) -> None:
    """_full_history_frame with scope='train_only' returns exactly the train
    split — validation rows never enter the fit frame."""
    from nmr.models import ModelOrchestrator
    from nmr.promote import _full_history_frame

    data = _make_data(tmp_path / "data")
    config = _config(data)
    orch = ModelOrchestrator(config.model, seed=config.run.seed)
    frame = _full_history_frame(
        config, ["f1", "f2"], ["target"], orch, scope="train_only"
    )
    eras = sorted({int(e) for e in frame.get_column("era").unique().to_list()})
    assert eras == list(range(1, 9))  # train eras 1..8 only


def test_train_only_spawn_spec_excludes_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spawned-worker spec carries include_validation=False for train_only
    (fit-phase isolation on the fresh-process path)."""
    from nmr.models import ModelOrchestrator

    captured: dict[str, object] = {}
    original = ModelOrchestrator._fit_full_history_subprocess

    def _spy(self, train_df, *, feature_cols, target_col, era_col, data, include_validation=False):
        captured["include_validation"] = include_validation
        return original(
            self, train_df, feature_cols=feature_cols, target_col=target_col,
            era_col=era_col, data=data, include_validation=include_validation,
        )

    monkeypatch.setattr(ModelOrchestrator, "_fit_full_history_subprocess", _spy)
    monkeypatch.setenv("NMR_FULL_HISTORY_SPAWN_MIN_BYTES", "1")
    data = _make_data(tmp_path / "data", validation_eras=32)
    _write_registry(stored_config=_stored_config_dict(data))
    promote_full_version(
        _RID,
        "brb1-lgbm-v6",
        override_gate=True,
        scope="train_only",
    )
    assert captured["include_validation"] is False


def test_train_only_writes_cross_check_scorecard(tmp_path: Path) -> None:
    """The partial export ships a versioned scorecard.json (official backend,
    fixed replay constants, window eras + per-era series + raw Sharpe)."""
    data = _make_data(tmp_path / "data", validation_eras=32)
    _write_registry(stored_config=_stored_config_dict(data))
    result = promote_full_version(
        _RID,
        "brb1-lgbm-v6",
        override_gate=True,
        scope="train_only",
    )
    slot = paths.export_dir("brb1-lgbm-v6", "partial", _RID)
    sc = json.loads((slot / "scorecard.json").read_text(encoding="utf-8"))
    assert result.cross_check_path == slot / "scorecard.json"
    assert sc["schema_version"] == 3
    assert sc["replay"]["backend"] == "official"
    assert sc["replay"]["n_trials"] == CROSSCHECK_N_TRIALS
    assert sc["scope"] == "partial"
    assert sc["window"]["eras"] == sorted(sc["window"]["eras"], key=int)
    # 32 validation eras minus the 8-era 20D-target purge.
    assert len(sc["window"]["eras"]) == 24
    assert sc["per_era"]["corr"] and all(
        {"era", "value"} <= set(entry) for entry in sc["per_era"]["corr"]
    )
    assert sc["per_era"]["mmc"] and sc["per_era"]["fnc"]
    assert isinstance(sc["raw_sharpe"], float)
    assert sc["generated_at"]
    assert set(sc["scorecard"]) >= {"corr", "mmc", "corr_sharpe_ac", "fnc", "n_eras"}


def test_full_scope_requires_current_pointer(tmp_path: Path) -> None:
    """A full-scope promotion repoints current.json atomically."""
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data))
    result = promote_full_version(
        _RID,
        "brb1-lgbm-v6",
        override_gate=True,
        scope="full",
    )
    assert result.scope == "full"
    assert result.cross_check_path is None
    pointer = json.loads(
        paths.current_pointer_path("brb1-lgbm-v6").read_text(encoding="utf-8")
    )
    assert pointer["run_id"] == _RID


def test_repromotion_rejected_without_force(tmp_path: Path) -> None:
    """Exports are immutable: promoting an existing slot raises with
    force=False; force=True enters the pointer-repair recovery (repoints
    current.json at the existing valid slot — never overwrites the slot)."""
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data))
    promote_full_version(
        _RID,
        "brb1-lgbm-v6",
        override_gate=True,
        force=True,
        scope="full",
    )
    with pytest.raises(ValueError, match="already exists"):
        promote_full_version(
            _RID,
            "brb1-lgbm-v6",
            override_gate=True,
            force=False,
            scope="full",
        )
    # force=True against the same existing valid slot is a no-op repair.
    result = promote_full_version(
        _RID,
        "brb1-lgbm-v6",
        override_gate=True,
        force=True,
        scope="full",
    )
    assert result.artifact_path.is_file()
    assert lifecycle.current_full_status("brb1-lgbm-v6") == "full"


def test_partial_scoring_failure_discards_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cross-check failure discards the staging dir — no half-written slot
    and no .tmp- residue (publication atomicity)."""
    data = _make_data(tmp_path / "data", validation_eras=32)
    _write_registry(stored_config=_stored_config_dict(data))

    def _boom(*args, **kwargs):
        raise RuntimeError("scoring exploded")

    monkeypatch.setattr("nmr.promote._run_cross_check", _boom)
    with pytest.raises(RuntimeError, match="scoring exploded"):
        promote_full_version(
            _RID,
            "brb1-lgbm-v6",
            override_gate=True,
            scope="train_only",
        )
    slot = paths.export_dir("brb1-lgbm-v6", "partial", _RID)
    assert not slot.exists()
    assert not (slot.parent / f".tmp-{_RID}").exists()
    assert lifecycle.scan_valid_exports("brb1-lgbm-v6", "partial") == []


def test_promote_invalid_scope_refused(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    _write_registry(stored_config=_stored_config_dict(data))
    with pytest.raises(ValueError, match="scope"):
        promote_full_version(
            _RID,
            "brb1-lgbm-v6",
            override_gate=True,
            scope="bogus",
        )


def test_ram_guard_train_only_scans_train_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A train_only guard call derives rows from train.parquet alone —
    validation.parquet is never scanned."""
    from nmr.promote import _ram_guard

    data = _make_data(tmp_path / "data")
    config = _config(data)
    _write_estimate(
        paths.shared_reports_dir(config.run.artifacts_dir),
        {
            "peak_commit_bytes": 1,
            "peak_bytes": 1,
            "parent_peak_commit_bytes": 0,
            "parent_peak_bytes": 0,
            "train_validation_rows": 1,
        },
    )
    scanned: list[str] = []
    original_scan = pl.scan_parquet

    def _spy_scan(path, *args, **kwargs):
        scanned.append(Path(path).name)
        return original_scan(path, *args, **kwargs)

    monkeypatch.setattr(pl, "scan_parquet", _spy_scan)
    _ram_guard(config, scope="train_only")
    assert scanned == ["train.parquet"]
