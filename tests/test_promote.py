"""Tests for the promotion writer (nmr.promote) — the money-path terminus.

Covers: gate enforcement + override recording, config normalization recording,
supplemental feature-set identity, slot/pointer overwrite guards, the
spawned include_validation path, and manifest/artifact validity (load_predict
round-trip with (0,1) output).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

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
from nmr.deployment import load_predict
from nmr.families import CURRENT_POINTER_NAME, available_slots, load_full_version
from nmr.promote import promote_full_version, rehearse_promotion
from nmr.runner import ExperimentRunner

_RID = "a" * 64


def _make_data(root: Path) -> Path:
    version_dir = root / "vtest"
    version_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for era in range(1, 17):
        for idx in range(8):
            f1 = idx * 0.05
            f2 = (idx % 3) * 0.1
            rows.append(
                {
                    "era": f"{era:04d}",
                    "id": f"{era}_{idx}",
                    "f1": f1,
                    "f2": f2,
                    "target": 0.4 * f1 - 0.2 * f2 + 0.01 * era,
                }
            )
    pl.DataFrame([r for r in rows if int(r["era"]) <= 8]).write_parquet(
        version_dir / "train.parquet"
    )
    pl.DataFrame([r for r in rows if int(r["era"]) > 8]).write_parquet(
        version_dir / "validation.parquet"
    )
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
    registry: Path,
    *,
    run_id: str = _RID,
    stored_config: dict | None = None,
    scorecard: dict | None = None,
    feature_cols: list[str] | None = None,
    weights: list[float] | None = None,
    supplemental_sha: str | None = None,
) -> Path:
    run_dir = registry / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": stored_config if stored_config is not None else _stored_config_dict(
            registry.parent / "data"
        ),
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
    path = run_dir / "run.json"
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    return path


def _promote(tmp_path: Path, **kwargs) -> object:
    data = _make_data(tmp_path / "data")
    registry = tmp_path / "registry"
    _write_registry(registry, stored_config=_stored_config_dict(data))
    return promote_full_version(
        _RID,
        "brb1-lgbm-v6",
        models_dir=tmp_path / "models",
        registry_dir=registry,
        **kwargs,
    )


def test_promote_happy_path_manifest_and_artifact(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    registry = tmp_path / "registry"
    _write_registry(registry, stored_config=_stored_config_dict(data))
    result = promote_full_version(
        _RID, "brb1-lgbm-v6", models_dir=tmp_path / "models", registry_dir=registry
    )

    assert result.artifact_path == (
        tmp_path / "models" / "brb1-lgbm-v6" / "full" / _RID / "predict.pkl"
    )
    assert result.artifact_path.is_file()
    assert result.tier4_gate_passed is True
    assert result.override_used is False

    # families.py read-side validates the published manifest (pointer + slot).
    version = load_full_version(tmp_path / "models", "brb1-lgbm-v6")
    assert version is not None
    assert version.family == "brb1-lgbm-v6"
    assert version.promoted_from_run_id == _RID
    assert version.artifact_path == "predict.pkl"
    assert available_slots(tmp_path / "models", "brb1-lgbm-v6") == [_RID]
    pointer = tmp_path / "models" / "brb1-lgbm-v6" / "full" / CURRENT_POINTER_NAME
    assert json.loads(pointer.read_text(encoding="utf-8"))["run_id"] == _RID

    # Manifest records the promotion verdict block + config normalizations.
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["tier4_gate_passed"] is True
    assert manifest["override_used"] is False
    normalizations = {n["field"]: n for n in manifest["config_normalizations"]}
    assert normalizations["split.embargo_eras"] == {"field": "split.embargo_eras", "from": 4, "to": 0}
    assert normalizations["data.horizon"] == {"field": "data.horizon", "from": None, "to": "20D"}
    # First-class training provenance (review directive 2026-08-18): a genuine
    # full version is explicitly NOT a rehearsal and states what it trained on.
    assert manifest["rehearsal"] is False
    assert manifest["training_rows"] > 0
    assert len(manifest["training_era_range"]) == 2

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
    registry = tmp_path / "registry"
    failing = _passing_scorecard()
    failing["corr"] = 0.01  # below corr_min 0.0286
    _write_registry(registry, stored_config=_stored_config_dict(data), scorecard=failing)

    with pytest.raises(ValueError, match="tier-4 promotion gate"):
        promote_full_version(
            _RID, "brb1-lgbm-v6", models_dir=tmp_path / "models", registry_dir=registry
        )
    # The rehearsal path: override records the failure in the artifact's own
    # manifest — a failed-gate artifact carries its verdict with it.
    result = promote_full_version(
        _RID,
        "brb1-lgbm-v6",
        models_dir=tmp_path / "models",
        registry_dir=registry,
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
    registry = tmp_path / "registry"
    _write_registry(
        registry, stored_config=_stored_config_dict(data), scorecard={}
    )
    with pytest.raises(ValueError, match="no validation scorecard"):
        promote_full_version(
            _RID, "brb1-lgbm-v6", models_dir=tmp_path / "models", registry_dir=registry
        )


def test_promote_supplemental_identity_mismatch_refused(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    registry = tmp_path / "registry"
    # Stored run carries a supplemental SHA but the config has no path.
    _write_registry(
        registry,
        stored_config=_stored_config_dict(data),
        supplemental_sha="f" * 64,
    )
    with pytest.raises(ValueError, match="supplemental feature-set identity"):
        promote_full_version(
            _RID, "brb1-lgbm-v6", models_dir=tmp_path / "models", registry_dir=registry
        )


def test_promote_supplemental_identity_match(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    supp = data / "vtest" / "derived_feature_sets.json"
    supp.write_text(json.dumps({"sets": {"medium": ["f1", "f2"]}}), encoding="utf-8")
    stored = _stored_config_dict(data)
    stored["data"]["supplemental_feature_sets"] = str(supp)
    sha = ExperimentRunner._supplemental_fingerprint(supp)
    registry = tmp_path / "registry"
    _write_registry(
        registry, stored_config=stored, supplemental_sha=sha
    )
    result = promote_full_version(
        _RID, "brb1-lgbm-v6", models_dir=tmp_path / "models", registry_dir=registry
    )
    assert result.artifact_path.is_file()


def test_promote_overwrite_and_repoint_guards(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    registry = tmp_path / "registry"
    _write_registry(registry, stored_config=_stored_config_dict(data), run_id=_RID)
    promote_full_version(
        _RID, "brb1-lgbm-v6", models_dir=tmp_path / "models", registry_dir=registry
    )
    # Same slot again: immutable, refused without force.
    with pytest.raises(ValueError, match="already exists"):
        promote_full_version(
            _RID, "brb1-lgbm-v6", models_dir=tmp_path / "models", registry_dir=registry
        )
    # A second run repointing current.json away: refused without force.
    other = "b" * 64
    _write_registry(registry, stored_config=_stored_config_dict(data), run_id=other)
    with pytest.raises(ValueError, match="repointing requires force"):
        promote_full_version(
            other, "brb1-lgbm-v6", models_dir=tmp_path / "models", registry_dir=registry
        )
    promote_full_version(
        other, "brb1-lgbm-v6", models_dir=tmp_path / "models",
        registry_dir=registry, force=True,
    )
    assert available_slots(tmp_path / "models", "brb1-lgbm-v6") == [_RID, other]
    assert load_full_version(tmp_path / "models", "brb1-lgbm-v6").promoted_from_run_id == other


def test_promote_rejects_invalid_run_id_and_family(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    registry = tmp_path / "registry"
    _write_registry(registry, stored_config=_stored_config_dict(data))
    with pytest.raises(ValueError, match="64-char"):
        promote_full_version(
            "not-a-run-id", "brb1-lgbm-v6", models_dir=tmp_path / "models",
            registry_dir=registry,
        )
    with pytest.raises(ValueError, match="invalid family name"):
        promote_full_version(
            _RID, "../evil", models_dir=tmp_path / "models", registry_dir=registry
        )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        promote_full_version(
            "c" * 64, "brb1-lgbm-v6", models_dir=tmp_path / "models",
            registry_dir=registry,
        )


def test_promote_spawn_path_trains_on_train_plus_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the fresh-process full-history path at small scale (the D7
    rehearsal forces the same path via NMR_FULL_HISTORY_SPAWN_MIN_BYTES) and
    prove include_validation wiring: the child re-reads train+validation."""
    monkeypatch.setenv("NMR_FULL_HISTORY_SPAWN_MIN_BYTES", "1")
    data = _make_data(tmp_path / "data")
    registry = tmp_path / "registry"
    _write_registry(registry, stored_config=_stored_config_dict(data))
    result = promote_full_version(
        _RID, "brb1-lgbm-v6", models_dir=tmp_path / "models", registry_dir=registry,
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
    registry = tmp_path / "registry"
    _write_registry(registry, stored_config=_stored_config_dict(data))
    result = rehearse_promotion(
        _RID,
        "brb1-lgbm-v6",
        models_dir=tmp_path / "models",
        registry_dir=registry,
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
        (result.artifact_path.parent / "manifest.json").read_text(encoding="utf-8")
    )
    assert slot_manifest["rehearsal"] is True
    assert slot_manifest["training_rows"] == result.train_validation_rows
    assert not (tmp_path / "models" / "brb1-lgbm-v6" / "full" / CURRENT_POINTER_NAME).exists()
