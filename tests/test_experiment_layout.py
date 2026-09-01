"""Runner round-trip against the experiment layout (Task 7).

The runner writes every output under ``experiments/<slug>/runs/<run_id>/``
(spec §3) and persists the rebuild-identity manifest fields (spec §3.1).

Mid-plan compat (Task 7): run.json RECORDING still happens via the control
plane scripts into the legacy registry layout until Task 11 — the runner
itself writes no run.json, so these tests assert the runner's outputs only.
"""

from __future__ import annotations

import json

import polars as pl

from nmr import paths
from nmr.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    RunConfig,
    SplitConfig,
)
from nmr.runner import ExperimentRunner, _compute_code_fingerprint, _data_fingerprint


def _build_train_frame() -> pl.DataFrame:
    rows: list[dict[str, float | str]] = []
    for era in range(1, 13):
        for idx in range(6):
            f1 = idx * 0.02
            f2 = (idx % 3) * 0.01
            rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f1": f1,
                    "f2": f2,
                    "target": (0.6 + 0.03 * era) * f1
                    - (0.3 + 0.01 * era) * f2
                    + 0.3 * f1 * f1
                    + 0.05 * era,
                    "target_alt": (0.2 + 0.02 * era) * f1
                    + (0.7 - 0.01 * era) * f2
                    - 0.2 * f2 * f2
                    - 0.04 * era,
                }
            )
    return pl.DataFrame(rows)


def _write_synthetic_data(root) -> None:
    version_dir = root / "vtest"
    version_dir.mkdir(parents=True, exist_ok=True)
    features = {
        "feature_sets": {
            "small": ["f1", "f2"],
            "medium": ["f1", "f2"],
            "all": ["f1", "f2"],
        },
        "targets": ["target", "target_alt"],
    }
    (version_dir / "features.json").write_text(json.dumps(features), encoding="utf-8")
    _build_train_frame().write_parquet(version_dir / "train.parquet")

    val_rows = []
    for era in range(13, 19):
        for idx in range(6):
            f1 = idx * 0.02
            f2 = (idx % 3) * 0.01
            val_rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f1": f1,
                    "f2": f2,
                    "target": (0.6 + 0.03 * era) * f1
                    - (0.3 + 0.01 * era) * f2
                    + 0.3 * f1 * f1
                    + 0.05 * era,
                    "target_alt": (0.2 + 0.02 * era) * f1
                    + (0.7 - 0.01 * era) * f2
                    - 0.2 * f2 * f2
                    - 0.04 * era,
                }
            )
    val = pl.DataFrame(val_rows)
    val.write_parquet(version_dir / "validation.parquet")
    val.select(["era", "id"]).with_columns(
        pl.lit(0.35).alias("numerai_meta_model")
    ).write_parquet(version_dir / "meta_model.parquet")
    val.select(["era", "id"]).with_columns(
        pl.lit(0.2).alias("bench_cyrusd_20")
    ).write_parquet(version_dir / "validation_benchmark_models.parquet")


def _config(tmp_path) -> ExperimentConfig:
    data_root = tmp_path / "data"
    _write_synthetic_data(data_root)
    return ExperimentConfig(
        data=DataConfig(
            version="vtest",
            feature_set="small",
            targets=("target", "target_alt"),
            data_dir=data_root,
        ),
        split=SplitConfig(
            scheme="walk_forward", purge_eras=1, embargo_eras=0, n_folds=2
        ),
        model=ModelConfig(
            backend="lightgbm",
            preset="fast",
            params={"n_estimators": 10, "learning_rate": 0.05, "min_data_in_leaf": 2},
        ),
        evaluation=EvalConfig(
            backend="custom",
            main_target="target",
            payout_policy="classic_legacy_075_225_clip005_v1",
            validation_scorecard=True,
        ),
        run=RunConfig(
            seed=17, artifacts_dir=tmp_path / "artifacts", name="layout-test"
        ),
    )


def test_runner_outputs_under_experiment(tmp_path, monkeypatch) -> None:
    """Every runner output lands under experiments/<slug>/runs/<run_id>/."""
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    cfg = _config(tmp_path)
    result = ExperimentRunner(cfg).run(deploy=True)

    slug = paths.validate_slug(cfg.run.name)
    run_dir = paths.run_dir(slug, result.run_id)
    # Per-target identity manifests (2026-08-26 review SECONDARY 1): the OOF
    # manifest sits next to its folds, the deploy manifests next to their pkls,
    # the validation manifest stays at the root.
    assert (run_dir / "oof_checkpoints" / "target" / "manifest.json").is_file()
    assert (run_dir / "deploy_checkpoints" / "target.manifest.json").is_file()
    assert (run_dir / "validation_checkpoints" / "manifest.json").is_file()
    assert (run_dir / "predict.pkl").is_file()
    assert (run_dir / "predict.pkl.manifest.json").is_file()
    # Mid-plan compat: the runner records no run.json — the scripts do (Task 11).
    assert not paths.run_json_path(slug, result.run_id).exists()
    # Nothing writes to the legacy artifacts/runs home anymore.
    assert not (tmp_path / "artifacts" / "runs").exists()


def test_run_manifest_persists_rebuild_identity(tmp_path, monkeypatch) -> None:
    """Spec §3.1: run.json manifest records the five rebuild-identity fields."""
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    cfg = _config(tmp_path)
    result = ExperimentRunner(cfg).run(deploy=False)
    manifest = result.manifest

    for field in (
        "data_fingerprint",
        "code_fingerprint",
        "environment",
        "pipeline_device",
        "oof_device",
    ):
        assert isinstance(manifest.get(field), str)
        assert manifest[field], f"rebuild identity field {field!r} is empty"

    # data_fingerprint is the persisted run-id term — the same value the
    # runner hashed into the run_id, not a recomputation.
    assert manifest["data_fingerprint"] == _data_fingerprint(cfg)
    assert len(manifest["data_fingerprint"]) == 64
    # code_fingerprint is the portable full-package hash (matches run-id term).
    assert manifest["code_fingerprint"] == _compute_code_fingerprint()
    assert len(manifest["code_fingerprint"]) == 64
    # pipeline_device is the config knob; oof_device the actual fit device.
    assert manifest["pipeline_device"] == str(cfg.model.device)
    assert manifest["oof_device"] == "cpu"
    # environment is a normalized name==version list — no paths.
    assert "numpy==" in manifest["environment"]
    assert "cloudpickle==" in manifest["environment"]
    assert "," in manifest["environment"]
    assert "/" not in manifest["environment"]
    assert "\\" not in manifest["environment"]


def test_portable_environment_is_sorted_and_path_free() -> None:
    from nmr.runner import _portable_environment

    parts = _portable_environment().split(",")
    assert parts == sorted(parts)
    assert len(parts) == 8  # the pinned deps: numpy, polars, lightgbm, xgboost,
    # catboost, scipy, numerai-tools, cloudpickle
    for part in parts:
        name, version = part.split("==")
        assert name and version
        assert "/" not in part and "\\" not in part
