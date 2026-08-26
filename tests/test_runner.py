"""Tests for deterministic experiment orchestration."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nmr import paths
from nmr.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    RunConfig,
    SplitConfig,
)
from nmr.data import IngestionAgent
from nmr.deployment import load_predict
from nmr.models import ModelOrchestrator
from nmr.runner import ExperimentRunner
from nmr.splitter import PurgedEraSplitter


@pytest.fixture(autouse=True)
def _isolated_experiments_root(tmp_path, monkeypatch) -> None:
    """Route runner outputs into a per-test experiments dir (never the repo root).

    Task 7: the runner writes under ``paths.run_dir(slug, run_id)`` = the module
    EXPERIMENTS_ROOT; tests must never touch (or pollute) the real repo root.
    """
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")


def _build_train_frame() -> pl.DataFrame:
    rows: list[dict[str, float | str]] = []
    for era in range(1, 13):
        for idx in range(6):
            f1 = (idx * 0.02)
            f2 = ((idx % 3) * 0.01)
            rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f1": f1,
                    "f2": f2,
                    "target": (0.6 + 0.03 * era) * f1 - (0.3 + 0.01 * era) * f2 + 0.3 * f1 * f1 + 0.05 * era,
                    "target_alt": (0.2 + 0.02 * era) * f1 + (0.7 - 0.01 * era) * f2 - 0.2 * f2 * f2 - 0.04 * era,
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
            # NOTE: features are shifted back into the training envelope
            # ((era - 11) => train-era-2..7 feature values). The brief's raw
            # `era * 0.03` formula puts every validation row outside the model's
            # training range, so the (correct) full-history model extrapolates to
            # a single constant leaf per era and full neutralization maps that to
            # exactly 0.0 — making the F-019 fidelity test's Spearman rho
            # undefined. Shifted features keep eras/assertions unchanged while
            # yielding non-degenerate predictions.
            f1 = (idx * 0.02)
            f2 = ((idx % 3) * 0.01)
            val_rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f1": f1,
                    "f2": f2,
                    "target": (0.6 + 0.03 * era) * f1 - (0.3 + 0.01 * era) * f2 + 0.3 * f1 * f1 + 0.05 * era,
                    "target_alt": (0.2 + 0.02 * era) * f1 + (0.7 - 0.01 * era) * f2 - 0.2 * f2 * f2 - 0.04 * era,
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
            backend="custom", main_target="target",
            metrics=("corr", "fnc", "sharpe"),  # mmc is validation-only; guard rejects it without the validation stage
            validation_scorecard=False,
        ),
        run=RunConfig(
            seed=17, artifacts_dir=tmp_path / "artifacts", name="runner-test"
        ),
    )


def test_runner_is_deterministic_and_leakage_safe(tmp_path) -> None:
    cfg = _config(tmp_path)
    runner = ExperimentRunner(cfg)
    first = runner.run(deploy=False)
    second = runner.run(deploy=False)

    assert first.run_id == second.run_id
    assert first.oof.equals(second.oof)
    assert first.metrics == second.metrics
    assert first.artifact is None and second.artifact is None

    eras = _build_train_frame().get_column("era").to_list()
    folds = PurgedEraSplitter(cfg.split).split(eras)
    expected_val_eras = {era for fold in folds for era in fold.val_eras}
    assert set(first.oof.get_column("era").to_list()) == expected_val_eras


def test_runner_deploy_serializes_reloadable_predict(tmp_path) -> None:
    cfg = _config(tmp_path)
    result = ExperimentRunner(cfg).run(deploy=True)

    assert result.artifact is not None
    loaded_predict = load_predict(result.artifact.path)
    live_features = pd.DataFrame(
        # in-envelope live features (train support: f1 in [0, 0.1], f2 in [0, 0.02])
        {"f1": [0.0, 0.02, 0.04, 0.06, 0.08, 0.1], "f2": [0.0, 0.01, 0.02, 0.0, 0.01, 0.02]},
        index=[f"id_{i}" for i in range(6)],
    )
    prediction = loaded_predict(live_features)
    assert list(prediction.columns) == ["prediction"]
    assert prediction.index.tolist() == [f"id_{i}" for i in range(6)]
    assert prediction["prediction"].notna().all()
    assert prediction["prediction"].nunique() > 1  # non-constant pipeline output
    # Submission-contract range: raw deploy output must be strictly in (0,1)
    # (numerai_tools validate_values hard-asserts between 0 and 1) — the
    # closure's per-era tie_kept_rank step guarantees (0.5/n, (n-0.5)/n).
    pred_values = prediction["prediction"]
    assert ((pred_values > 0) & (pred_values < 1)).all()


def test_deploy_predict_ranks_per_era_not_whole_frame(tmp_path) -> None:
    """Discriminates per-era from whole-frame ranking in the deploy closure.

    The final (0,1) step must rank WITHIN each era: an era of ``n`` rows must
    independently span ``(0.5/n, (n-0.5)/n)``. A whole-frame rank is globally
    monotone (per-era metric-invariance proofs and the (0,1) range check both
    pass under it), so only this per-era bound catches the regression.
    """
    cfg = _config(tmp_path)
    result = ExperimentRunner(cfg).run(deploy=True)
    assert result.artifact is not None
    loaded_predict = load_predict(result.artifact.path)
    live = pd.DataFrame(
        {
            "f1": [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.0, 0.03, 0.07],
            "f2": [0.0, 0.01, 0.02, 0.0, 0.01, 0.02, 0.02, 0.0, 0.01],
            "era": ["0001", "0001", "0001", "0002", "0002", "0002",
                    "0003", "0003", "0003"],
        },
        index=[f"id_{i}" for i in range(9)],
    )
    prediction = loaded_predict(live)
    values = prediction["prediction"].to_numpy()
    era_col = live["era"].to_numpy()
    for era in np.unique(era_col):
        block = values[era_col == era]
        n = block.size
        assert block.min() == pytest.approx(0.5 / n, rel=1e-12)
        assert block.max() == pytest.approx((n - 0.5) / n, rel=1e-12)


def test_run_id_is_path_independent_and_seed_sensitive(tmp_path) -> None:
    cfg_a = _config(tmp_path / "a")
    cfg_b = _config(tmp_path / "b")

    runner_a = ExperimentRunner(cfg_a)
    runner_b = ExperimentRunner(cfg_b)
    assert runner_a._run_id == runner_b._run_id

    cfg_seed_flip = ExperimentConfig(
        data=cfg_a.data,
        split=cfg_a.split,
        model=cfg_a.model,
        evaluation=cfg_a.evaluation,
        run=RunConfig(
            seed=cfg_a.run.seed + 1,
            artifacts_dir=cfg_a.run.artifacts_dir,
            name=cfg_a.run.name,
        ),
    )
    runner_seed_flip = ExperimentRunner(cfg_seed_flip)
    assert runner_seed_flip._run_id != runner_a._run_id


def _validation_config(tmp_path) -> ExperimentConfig:
    cfg = _config(tmp_path)
    return ExperimentConfig(
        data=cfg.data,
        split=cfg.split,
        model=cfg.model,
        evaluation=EvalConfig(
            backend="custom", main_target="target", validation_scorecard=True
        ),
        run=cfg.run,
    )


def test_validation_stage_produces_scorecard_and_purges_first_eras(tmp_path) -> None:
    cfg = _validation_config(tmp_path)
    result = ExperimentRunner(cfg).run(deploy=True)

    assert result.scorecard is not None
    assert result.scorecard.model_id == result.run_id
    assert result.manifest["validation_purge_dropped_first_eras"] == cfg.split.purge_eras
    assert result.validation_predictions is not None
    scored_eras = set(result.validation_predictions.get_column("era").to_list())
    assert "13" not in scored_eras  # purge_eras=1 -> era 13 dropped
    assert "14" in scored_eras
    assert result.artifact is not None


def test_deployed_artifact_matches_validation_stage_predictions(tmp_path) -> None:
    """F-019 fidelity: load_predict reproduces the scored validation pipeline."""
    from scipy.stats import spearmanr

    cfg = _validation_config(tmp_path)
    result = ExperimentRunner(cfg).run(deploy=True)
    assert result.artifact is not None and result.validation_predictions is not None

    val = pl.read_parquet(cfg.data.path("validation.parquet")).filter(
        pl.col("era") != "13"
    )
    features_pd = val.select(["id", "era", "f1", "f2"]).to_pandas().set_index("id")
    loaded = load_predict(result.artifact.path)
    out = loaded(features_pd)

    expected = result.validation_predictions.sort("id")
    actual = pl.from_pandas(out.reset_index().rename(columns={"index": "id"})).sort("id")
    rho, _ = spearmanr(expected.get_column("prediction").to_numpy(),
                       actual.get_column("prediction").to_numpy())
    assert rho > 0.999
    assert np.allclose(
        expected.get_column("prediction").to_numpy(),
        actual.get_column("prediction").to_numpy(),
        atol=1e-12,
    )


def test_single_fold_falls_back_to_uniform_weights(tmp_path, caplog) -> None:
    cfg = _config(tmp_path)
    single_fold = ExperimentConfig(
        data=cfg.data, split=SplitConfig(scheme="anchor", purge_eras=1, n_folds=1),
        model=cfg.model,
        evaluation=EvalConfig(
            backend="custom", main_target="target",
            metrics=("corr", "fnc", "sharpe"),  # mmc is validation-only; guard rejects it without the validation stage
            validation_scorecard=False,
        ),
        run=cfg.run,
    )
    import logging
    with caplog.at_level(logging.WARNING, logger="nmr.runner"):
        result = ExperimentRunner(single_fold).run(deploy=False)
    assert result.manifest["weights"] == [0.5, 0.5]  # 2 components, uniform
    assert any("uniform" in record.message for record in caplog.records)


def test_mmc_metric_requires_validation_scorecard(tmp_path) -> None:
    cfg = _config(tmp_path)
    bad = ExperimentConfig(
        data=cfg.data, split=cfg.split, model=cfg.model,
        evaluation=EvalConfig(
            backend="custom", main_target="target",
            validation_scorecard=False, metrics=("corr", "mmc", "sharpe"),
        ),
        run=cfg.run,
    )
    import pytest as _pytest
    with _pytest.raises(ValueError, match="mmc"):
        ExperimentRunner(bad).run(deploy=False)


def test_fold_held_out_weight_learning_and_scoring_eras(tmp_path) -> None:
    cfg = _config(tmp_path)
    result = ExperimentRunner(cfg).run(deploy=False)
    folds = PurgedEraSplitter(cfg.split).split(
        _build_train_frame().get_column("era").to_list()
    )
    assert set(result.manifest["weight_learning_eras"]) == {
        era for fold in folds[:-1] for era in fold.val_eras
    }
    assert set(result.manifest["scoring_eras"]) == set(folds[-1].val_eras)


def test_feature_subset_changes_run_id_and_uses_subset_features(tmp_path) -> None:
    """feature_subset must change the run fingerprint and reach the data layer."""
    import json as _json

    cfg = _config(tmp_path)
    # vtest features.json has small == medium == all == [f1, f2]; add a family
    # set via the data dir used by _config and re-run with feature_subset.
    version_dir = cfg.data.data_dir / "vtest"
    features = _json.loads((version_dir / "features.json").read_text(encoding="utf-8"))
    features["feature_sets"]["sunshine"] = ["f1", "f2"]
    (version_dir / "features.json").write_text(_json.dumps(features), encoding="utf-8")

    plain = ExperimentRunner(cfg)
    subset_cfg = ExperimentConfig(
        data=DataConfig(
            version=cfg.data.version, feature_set=cfg.data.feature_set,
            feature_subset="sunshine", targets=cfg.data.targets,
            data_dir=cfg.data.data_dir,
        ),
        split=cfg.split, model=cfg.model, evaluation=cfg.evaluation, run=cfg.run,
    )
    subset = ExperimentRunner(subset_cfg)
    assert plain._run_id != subset._run_id
    assert subset.run(deploy=False).manifest["feature_cols"] == ["f1", "f2"]


def test_compute_run_id_public_accessor_matches_private(tmp_path) -> None:
    cfg = _config(tmp_path)
    assert ExperimentRunner.compute_run_id(cfg) == ExperimentRunner(cfg)._run_id
    assert (
        ExperimentRunner.compute_run_id(cfg)
        == ExperimentRunner.compute_run_id(cfg)
    )


def test_runner_catboost_end_to_end(tmp_path) -> None:
    """CatBoost end-to-end: deterministic run, deploy roundtrip, oof_device.

    NOTE: the synthetic features are era-independent (f1 in [0, 0.1],
    f2 in [0, 0.02]), so any train window covers the live envelope and
    out-of-envelope constant-leaf collapse is structurally impossible.
    The era variation lives in the target coefficients instead, keeping
    per-era CORR series non-degenerate.
    """
    cfg = _config(tmp_path)
    catboost_cfg = ExperimentConfig(
        data=cfg.data, split=cfg.split,
        model=ModelConfig(backend="catboost", preset="fast",
                          params={"n_estimators": 40, "learning_rate": 0.05}),
        evaluation=cfg.evaluation, run=cfg.run,
    )
    runner = ExperimentRunner(catboost_cfg)
    first = runner.run(deploy=True)
    second = ExperimentRunner(catboost_cfg).run(deploy=False)
    assert first.run_id == second.run_id
    assert first.oof.equals(second.oof)
    assert first.manifest["oof_device"] == "cpu"
    assert first.artifact is not None
    loaded = load_predict(first.artifact.path)
    live_features = pd.DataFrame(
        {"f1": [0.0, 0.02, 0.04], "f2": [0.0, 0.01, 0.02]}, index=["a", "b", "c"]
    )
    pred = loaded(live_features)
    assert pred["prediction"].notna().all()
    assert pred["prediction"].nunique() > 1  # non-constant pipeline output


def test_environment_fingerprint_lightgbm_shape(tmp_path) -> None:
    """Backend-aware fingerprint: lightgbm configs fingerprint the base
    packages plus optuna (B2, 2026-08-18) but never catboost.
    """
    cfg = _config(tmp_path)  # lightgbm backend
    assert cfg.model.backend == "lightgbm"
    packages = ExperimentRunner._environment_fingerprint(cfg.model.backend)["packages"]
    assert set(packages) == {
        "numpy", "polars", "pandas", "lightgbm", "xgboost", "optuna"
    }
    assert "catboost" not in packages
    # The no-arg call must be byte-identical to the backend-specific one.
    assert packages == ExperimentRunner._environment_fingerprint()["packages"]


def test_environment_fingerprint_catboost_includes_version(tmp_path) -> None:
    """CatBoost-backend configs must fingerprint the catboost package version.

    The catboost-backend run_id must flag the installed catboost version, the
    same stability marker lightgbm/xgboost get from their package versions.
    """
    cfg = _config(tmp_path)
    catboost_cfg = ExperimentConfig(
        data=cfg.data, split=cfg.split,
        model=ModelConfig(backend="catboost", preset="fast",
                          params={"n_estimators": 40, "learning_rate": 0.05}),
        evaluation=cfg.evaluation, run=cfg.run,
    )
    packages = ExperimentRunner._environment_fingerprint(
        catboost_cfg.model.backend
    )["packages"]
    assert "catboost" in packages
    assert isinstance(packages["catboost"], str) and packages["catboost"]


def test_predict_in_era_batches_matches_full_frame_predict(tmp_path) -> None:
    """Chunked validation predict is bit-identical to the full-frame path."""
    from nmr.models import ModelOrchestrator, coerce_float32_features
    from nmr.runner import _predict_in_era_batches

    data_root = tmp_path / "data"
    _write_synthetic_data(data_root)
    cfg = _config(tmp_path)
    agent = IngestionAgent(cfg.data)
    val_df = agent.load("validation", columns=["era", "id", "f1", "f2"])
    train_df = agent.load("train", columns=["era", "id", "f1", "f2", "target"])

    orchestrator = ModelOrchestrator(cfg.model, seed=cfg.run.seed)
    model = orchestrator.train_full_history(
        train_df, feature_cols=["f1", "f2"], target_col="target"
    )

    def predict_fn(features_pd):
        # mirror the deploy closure: select only feature columns (era is meta)
        frame = features_pd.loc[:, ["f1", "f2"]]
        return features_pd.assign(prediction=model.predict(frame))[["prediction"]]

    batched = _predict_in_era_batches(val_df, ["f1", "f2"], predict_fn, batch_eras=2)
    full_feats = coerce_float32_features(val_df, ["f1", "f2"])
    full_pd = (
        pl.concat([val_df.select(["id"]), full_feats], how="horizontal")
        .to_pandas()
        .set_index("id")
    )
    full = val_df.select(["era", "id"]).with_columns(
        pl.Series("prediction", predict_fn(full_pd)["prediction"].to_numpy())
    )
    assert batched.equals(full)
    assert batched.height == val_df.height


def test_validation_purge_keeps_zero_padded_eras(tmp_path) -> None:
    """Regression (2026-08-11): the validation purge compared str(int) era
    indices against zero-padded era labels, silently truncating the scored
    window to eras >= 1000 (e.g. 232 of 649 eras). With padded fixture eras,
    the purge must keep the numeric window: 6 eras, purge 1 -> 0014..0018."""
    # _config writes the data; pad the files AFTER it (it rewrites them)
    cfg = _validation_config(tmp_path)
    vd = tmp_path / "data" / "vtest"
    for name in (
        "validation.parquet",
        "meta_model.parquet",
        "validation_benchmark_models.parquet",
    ):
        path = vd / name
        df = pl.read_parquet(path)
        df = df.with_columns(
            pl.col("era").cast(pl.Int32).cast(pl.String).str.zfill(4)
        )
        df.write_parquet(path)

    result = ExperimentRunner(cfg).run(deploy=True)
    assert result.validation_predictions is not None
    scored = sorted(
        result.validation_predictions.get_column("era").unique().to_list()
    )
    assert scored == ["0014", "0015", "0016", "0017", "0018"]


def test_predict_in_era_batches_empty_frame() -> None:
    from nmr.runner import _predict_in_era_batches

    empty = pl.DataFrame({"era": [], "id": []}, schema={"era": pl.String, "id": pl.String})
    out = _predict_in_era_batches(empty, ["f1"], lambda pdf: pdf, batch_eras=40)
    assert out.height == 0
    assert out.columns == ["era", "id", "prediction"]


def test_code_fingerprint_normalizes_line_endings(tmp_path) -> None:
    """Windows (CRLF) and POSIX (LF) checkouts of the same commit must hash
    identically — otherwise run_ids diverge across machines."""
    from nmr.runner import ExperimentRunner

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "a.py").write_bytes(b"x = 1\r\n")
    (pkg / "b.py").write_bytes(b"y = 2\r\n")
    lf = ExperimentRunner._code_fingerprint(pkg)

    (pkg / "a.py").write_bytes(b"x = 1\n")
    (pkg / "b.py").write_bytes(b"y = 2\n")
    assert ExperimentRunner._code_fingerprint(pkg) == lf

    (pkg / "a.py").write_bytes(b"x = 99\n")
    assert ExperimentRunner._code_fingerprint(pkg) != lf


def _with_supplemental(cfg: ExperimentConfig, supp_path) -> ExperimentConfig:
    """Clone ``cfg`` with a ``supplemental_feature_sets`` path attached."""
    return ExperimentConfig(
        data=DataConfig(
            version=cfg.data.version,
            feature_set=cfg.data.feature_set,
            feature_subset=cfg.data.feature_subset,
            supplemental_feature_sets=supp_path,
            targets=cfg.data.targets,
            data_dir=cfg.data.data_dir,
        ),
        split=cfg.split,
        model=cfg.model,
        evaluation=cfg.evaluation,
        run=cfg.run,
    )


def test_run_id_is_path_independent_with_supplemental_feature_sets(tmp_path) -> None:
    """Identical derived-set files at different absolute paths must produce
    the same run_id — the path must never leak into the canonical hash."""
    supp_a = tmp_path / "a" / "derived_feature_sets.json"
    supp_b = tmp_path / "b" / "derived_feature_sets.json"
    supp_a.parent.mkdir(parents=True)
    supp_b.parent.mkdir(parents=True)
    payload = json.dumps({"feature_sets": {"screen_stable": ["f1"]}})
    supp_a.write_text(payload, encoding="utf-8")
    supp_b.write_text(payload, encoding="utf-8")

    cfg_a = _with_supplemental(_config(tmp_path), supp_a)
    cfg_b = _with_supplemental(_config(tmp_path), supp_b)

    assert ExperimentRunner(cfg_a)._run_id == ExperimentRunner(cfg_b)._run_id


def test_run_id_changes_when_supplemental_file_contents_change(tmp_path) -> None:
    """Editing the derived-sets file in place must change run identity —
    the canonical hash must cover file contents, not the path string."""
    supp = tmp_path / "derived_feature_sets.json"
    supp.write_text(
        json.dumps({"feature_sets": {"screen_stable": ["f1"]}}), encoding="utf-8"
    )
    cfg = _with_supplemental(_config(tmp_path), supp)
    first = ExperimentRunner(cfg)._run_id

    supp.write_text(
        json.dumps({"feature_sets": {"screen_stable": ["f1", "f2"]}}),
        encoding="utf-8",
    )
    assert ExperimentRunner(cfg)._run_id != first


def test_run_id_supplemental_hash_is_line_ending_insensitive(tmp_path) -> None:
    """CRLF and LF checkouts of the same derived-sets file must hash
    identically (mirrors the code-fingerprint normalization)."""
    supp_a = tmp_path / "a" / "derived_feature_sets.json"
    supp_b = tmp_path / "b" / "derived_feature_sets.json"
    supp_a.parent.mkdir(parents=True)
    supp_b.parent.mkdir(parents=True)
    supp_a.write_bytes(b'{"feature_sets": {"screen_stable": ["f1"]}}\n')
    supp_b.write_bytes(b'{"feature_sets": {"screen_stable": ["f1"]}}\r\n')

    cfg_a = _with_supplemental(_config(tmp_path), supp_a)
    cfg_b = _with_supplemental(_config(tmp_path), supp_b)

    assert ExperimentRunner(cfg_a)._run_id == ExperimentRunner(cfg_b)._run_id


def test_runner_cross_process_determinism(tmp_path) -> None:
    """Two fresh interpreters running the full ExperimentRunner pipeline over
    identical synthetic data must produce the same run_id, OOF bytes, and
    metrics — the entry-point determinism guarantee. In-process re-runs cannot
    catch import-order, global-RNG, or hash-seed coupling."""
    import subprocess
    import sys

    _write_synthetic_data(tmp_path / "data")
    (tmp_path / "config.yaml").write_text(
        f"""
data:
  version: vtest
  feature_set: small
  targets: [target, target_alt]
  data_dir: {str(tmp_path / "data").replace(chr(92), "/")}
split:
  scheme: walk_forward
  purge_eras: 1
  embargo_eras: 0
  n_folds: 2
model:
  backend: lightgbm
  preset: fast
  params:
    n_estimators: 10
    learning_rate: 0.05
    min_data_in_leaf: 2
evaluation:
  backend: custom
  main_target: target
  metrics: [corr, fnc, sharpe]
  validation_scorecard: false
run:
  seed: 17
  name: cross-process-determinism
""",
        encoding="utf-8",
    )

    def _run_script(artifacts_dir, experiments_root) -> str:
        code = f"""
import dataclasses
import hashlib
import json
from pathlib import Path

import nmr.paths as nmr_paths
from nmr.config import ExperimentConfig, RunConfig, load_config
from nmr.runner import ExperimentRunner

nmr_paths.EXPERIMENTS_ROOT = Path({str(experiments_root)!r})

cfg = load_config({str(tmp_path / "config.yaml")!r})
cfg = ExperimentConfig(
    data=cfg.data,
    split=cfg.split,
    model=cfg.model,
    evaluation=cfg.evaluation,
    risk=cfg.risk,
    ensemble=cfg.ensemble,
    run=RunConfig(
        seed=cfg.run.seed,
        name=cfg.run.name,
        artifacts_dir={str(artifacts_dir)!r},
    ),
)
result = ExperimentRunner(cfg).run(deploy=False)
oof_bytes = json.dumps(
    result.oof.sort("id").to_dicts(), sort_keys=True, default=str
).encode("utf-8")
print(
    json.dumps(
        {{
            "run_id": result.run_id,
            "oof_sha256": hashlib.sha256(oof_bytes).hexdigest(),
            "metrics": dataclasses.asdict(result.metrics),
        }},
        sort_keys=True,
        default=str,
    )
)
"""
        return subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()

    # Distinct artifact dirs AND distinct experiments roots per process
    # (both are stripped from run_id; path-independence is covered by the
    # dedicated run_id tests above). Shared checkpoints would resume instead
    # of fit, and the [fit] progress prints would differ across stdout.
    run1 = _run_script(tmp_path / "artifacts_a", tmp_path / "exp_a")
    run2 = _run_script(tmp_path / "artifacts_b", tmp_path / "exp_b")
    assert run1 == run2


def test_validation_stage_enables_horizon_diagnostics(tmp_path) -> None:
    """Runner validation scorecards must enable horizon stability whenever the
    validation schema carries horizon target columns — not silently disable the
    flagship diagnostic (regression: the runner loaded only config targets, so
    every runner-produced scorecard reported 'horizon target columns
    unavailable')."""
    import json as _json

    data_root = tmp_path / "data" / "vtest"
    data_root.mkdir(parents=True)
    (_json_features := {
        "feature_sets": {
            "small": ["f1", "f2"],
            "medium": ["f1", "f2"],
            "all": ["f1", "f2"],
        },
        "targets": ["target", "target_ender_20", "target_ender_60"],
    }) and (data_root / "features.json").write_text(
        _json.dumps(_json_features), encoding="utf-8"
    )

    train_rows = []
    for era in range(1, 13):
        for idx in range(6):
            train_rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f1": idx * 0.02,
                    "f2": (idx % 3) * 0.01,
                    "target": 0.6 * idx * 0.02 + 0.05 * era,
                }
            )
    pl.DataFrame(train_rows).write_parquet(data_root / "train.parquet")

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
                    "target": (0.6 + 0.03 * era) * f1 - (0.3 + 0.01 * era) * f2,
                    "target_ender_20": (0.5 + 0.02 * era) * f1,
                    "target_ender_60": (0.4 + 0.01 * era) * f1,
                }
            )
    val = pl.DataFrame(val_rows)
    val.write_parquet(data_root / "validation.parquet")
    val.select(["era", "id"]).with_columns(
        pl.lit(0.35).alias("numerai_meta_model")
    ).write_parquet(data_root / "meta_model.parquet")
    val.select(["era", "id"]).with_columns(
        pl.lit(0.2).alias("v53_lgbm_ender20")
    ).write_parquet(data_root / "validation_benchmark_models.parquet")

    cfg = ExperimentConfig(
        data=DataConfig(
            version="vtest", feature_set="small", targets=("target",),
            data_dir=tmp_path / "data",
        ),
        split=SplitConfig(scheme="walk_forward", purge_eras=1, embargo_eras=0, n_folds=2),
        model=ModelConfig(
            backend="lightgbm", preset="fast",
            params={"n_estimators": 10, "learning_rate": 0.05, "min_data_in_leaf": 2},
        ),
        evaluation=EvalConfig(
            backend="custom", main_target="target", validation_scorecard=True
        ),
        run=RunConfig(seed=17, artifacts_dir=tmp_path / "artifacts", name="horizon-test"),
    )
    result = ExperimentRunner(cfg).run(deploy=True)
    reason = result.scorecard.horizon_reason
    assert reason != "horizon target columns unavailable", (
        "horizon target columns were present in the validation schema but the "
        "runner did not load them"
    )
    assert reason != "benchmark unavailable"


def test_run_id_changes_when_data_changes(tmp_path) -> None:
    """B1 (audit SEV-1 #3): the data term enters run identity as a snapshot
    fingerprint — same config/code/env with a changed data snapshot must
    produce a different run_id (validation grows weekly by design)."""
    cfg = _config(tmp_path)
    id_a = ExperimentRunner.compute_run_id(cfg)

    data_root = tmp_path / "data" / "vtest"
    val = pl.read_parquet(data_root / "validation.parquet")
    last_era = str(max(int(e) for e in val.get_column("era").unique()))
    val.filter(pl.col("era") != last_era).write_parquet(
        data_root / "validation.parquet"
    )
    id_b = ExperimentRunner.compute_run_id(cfg)
    assert id_a != id_b


def test_run_id_requires_data_snapshot(tmp_path) -> None:
    """B1: run_id fails loud without the data snapshot (dry-run needs data)."""
    cfg = _config(tmp_path)
    (tmp_path / "data" / "vtest" / "validation.parquet").unlink()
    with pytest.raises(ValueError, match="data fingerprint"):
        ExperimentRunner.compute_run_id(cfg)


def test_run_id_sensitive_to_optuna_version(tmp_path, monkeypatch) -> None:
    """B2: optuna entered the env fingerprint — a pin bump must flag drift."""
    import nmr.runner as runner_mod

    cfg = _config(tmp_path)
    versions = {
        "numpy": "1.0", "polars": "1.0", "pandas": "1.0",
        "lightgbm": "1.0", "xgboost": "1.0", "optuna": "3.0",
    }
    monkeypatch.setattr(runner_mod, "_package_version", lambda name: versions.get(name))
    id_a = runner_mod.ExperimentRunner.compute_run_id(cfg)
    versions["optuna"] = "4.0"
    id_b = runner_mod.ExperimentRunner.compute_run_id(cfg)
    assert id_a != id_b


def test_runner_and_research_share_oof_implementation(tmp_path) -> None:
    """C10 (audit SEV-2 #5): the runner's OOF delegate, research's alias, and
    the shared helper must produce identical output — one implementation."""
    from nmr._oof import train_multi_target_oof
    from nmr.research import _train_multi_target_oof as research_oof

    cfg = _config(tmp_path)
    runner = ExperimentRunner(cfg)
    train_df = _build_train_frame()
    splitter = PurgedEraSplitter(cfg.split)
    orch = ModelOrchestrator(cfg.model, seed=cfg.run.seed)
    targets = list(cfg.data.targets)

    shared = train_multi_target_oof(
        orch, train_df, feature_cols=["f1", "f2"], splitter=splitter, targets=targets
    )
    via_runner = runner._train_multi_target_oof(
        train_df, feature_cols=["f1", "f2"], splitter=splitter,
        model_orchestrator=orch,
    )
    via_research = research_oof(
        orch, train_df, feature_cols=["f1", "f2"], splitter=splitter, targets=targets
    )
    assert shared.equals(via_runner)
    assert shared.equals(via_research)


def test_runner_writes_and_reuses_oof_checkpoints(tmp_path, caplog) -> None:
    """Runner wiring: OOF folds are checkpointed under the run dir and a
    second run over the same config+data (same run_id) resumes them
    bit-for-bit instead of refitting (spec 2026-08-20-oof-checkpoint-resume)."""
    cfg = _config(tmp_path)
    result1 = ExperimentRunner(cfg).run(deploy=False)
    ckpt_root = paths.run_dir(cfg.run.name, result1.run_id) / "oof_checkpoints"
    assert (ckpt_root / "manifest.json").exists()
    assert sorted(p.name for p in (ckpt_root / "target").glob("fold_*.parquet"))
    caplog.clear()
    with caplog.at_level("INFO", logger="nmr.models"):
        result2 = ExperimentRunner(_config(tmp_path)).run(deploy=False)
    assert result2.run_id == result1.run_id
    assert result2.oof.equals(result1.oof)
    assert "loaded from checkpoint" in caplog.text


def test_deploy_checkpoints_written_and_mixed_resume_bit_for_bit(tmp_path, caplog) -> None:
    """Deploy mixed resume (spec 2026-08-23-checkpoint-coverage-extension):
    per-target pickled models persist under deploy_checkpoints/; deleting one
    .pkl and resuming refits only that target, and the deploy artifact's
    predictions are byte-identical to the uninterrupted run's."""
    live = pd.DataFrame(
        {"f1": [0.0, 0.02, 0.04, 0.06, 0.08, 0.1],
         "f2": [0.0, 0.01, 0.02, 0.0, 0.01, 0.02]},
        index=[f"id_{i}" for i in range(6)],
    )

    cfg = _config(tmp_path)
    first = ExperimentRunner(cfg).run(deploy=True)
    ckpt_root = paths.run_dir(cfg.run.name, first.run_id) / "deploy_checkpoints"
    assert (ckpt_root / "manifest.json").exists()
    assert sorted(p.name for p in ckpt_root.glob("*.pkl")) == [
        "target.pkl",
        "target_alt.pkl",
    ]
    manifest = json.loads((ckpt_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["device"] == "cpu"  # train_full_history is CPU-only by design
    assert len(manifest["code_sha256"]) == 64
    # Rebuild-identity terms (spec §3.1) mirror run.json's data_fingerprint
    # and environment fields.
    assert len(manifest["data_fingerprint"]) == 64
    assert manifest["environment"]
    expected = load_predict(first.artifact.path)(live)

    (ckpt_root / "target_alt.pkl").unlink()  # delete exactly ONE target
    caplog.clear()
    with caplog.at_level("INFO"):
        resumed = ExperimentRunner(_config(tmp_path)).run(deploy=True)
    assert resumed.run_id == first.run_id
    actual = load_predict(resumed.artifact.path)(live)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert caplog.text.count("train_full_history") == 1  # the refit happened
    assert caplog.text.count("loaded deploy checkpoint") == 1  # the other loaded


def test_deploy_checkpoint_code_mismatch_raises(tmp_path) -> None:
    """Deploy manifest identity: a changed fitting-code sha must fail loudly."""
    cfg = _config(tmp_path)
    first = ExperimentRunner(cfg).run(deploy=True)
    ckpt_root = paths.run_dir(cfg.run.name, first.run_id) / "deploy_checkpoints"
    manifest_path = ckpt_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["code_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="code_sha256"):
        ExperimentRunner(_config(tmp_path)).run(deploy=True)


def test_deploy_checkpoint_unknown_device_raises(tmp_path) -> None:
    """Deploy manifest identity: a device outside the known fit devices is
    rejected loudly even while the device is unresolved at resume entry."""
    cfg = _config(tmp_path)
    first = ExperimentRunner(cfg).run(deploy=True)
    ckpt_root = paths.run_dir(cfg.run.name, first.run_id) / "deploy_checkpoints"
    manifest_path = ckpt_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["device"] = "totally_different_device"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="device"):
        ExperimentRunner(_config(tmp_path)).run(deploy=True)


def test_deploy_checkpoint_device_exact_mismatch_on_refit_raises(tmp_path) -> None:
    """The authoritative device compare runs at the first refitted target
    (the device is only known post-fit): a stored 'gpu' device (deploy fits
    are CPU-only) must fail the exact compare when a missing .pkl forces a
    refit, instead of passing vacuously on the all-loaded path."""
    cfg = _config(tmp_path)
    first = ExperimentRunner(cfg).run(deploy=True)
    ckpt_root = paths.run_dir(cfg.run.name, first.run_id) / "deploy_checkpoints"
    manifest_path = ckpt_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["device"] = "gpu"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (ckpt_root / "target.pkl").unlink()
    with pytest.raises(ValueError, match="device"):
        ExperimentRunner(_config(tmp_path)).run(deploy=True)


def test_deploy_checkpoint_data_fingerprint_mismatch_raises(tmp_path) -> None:
    """Rebuild identity (spec §3.1): a deploy manifest recording a different
    data snapshot refuses resume at entry — never silently reuses stale pkls."""
    cfg = _config(tmp_path)
    first = ExperimentRunner(cfg).run(deploy=True)
    ckpt_root = paths.run_dir(cfg.run.name, first.run_id) / "deploy_checkpoints"
    manifest_path = ckpt_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data_fingerprint"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="data_fingerprint"):
        ExperimentRunner(_config(tmp_path)).run(deploy=True)


def test_deploy_checkpoint_torn_tree_raises(tmp_path) -> None:
    """Pickled models without a manifest.json are an inconsistent state."""
    cfg = _config(tmp_path)
    first = ExperimentRunner(cfg).run(deploy=True)
    ckpt_root = paths.run_dir(cfg.run.name, first.run_id) / "deploy_checkpoints"
    (ckpt_root / "manifest.json").unlink()
    with pytest.raises(ValueError, match="no manifest.json"):
        ExperimentRunner(_config(tmp_path)).run(deploy=True)


def test_deploy_checkpoint_corrupt_pkl_raises(tmp_path) -> None:
    """A corrupted pickled model must fail loudly with the path, never
    silently refit or produce garbage predictions."""
    cfg = _config(tmp_path)
    first = ExperimentRunner(cfg).run(deploy=True)
    ckpt_root = paths.run_dir(cfg.run.name, first.run_id) / "deploy_checkpoints"
    (ckpt_root / "target.pkl").write_bytes(b"garbage")
    with pytest.raises(ValueError, match="corrupt deploy checkpoint"):
        ExperimentRunner(_config(tmp_path)).run(deploy=True)


def test_deploy_checkpoint_dir_only_created_when_pipeline_built(tmp_path) -> None:
    """The deploy checkpoint dir is created only when the deploy pipeline is
    built (deploy=True or validation_scorecard=True)."""
    cfg = _config(tmp_path)
    result = ExperimentRunner(cfg).run(deploy=False)
    run_dir = paths.run_dir(cfg.run.name, result.run_id)
    assert (run_dir / "oof_checkpoints").exists()
    assert not (run_dir / "deploy_checkpoints").exists()


def _write_synthetic_data_multi_batch(root, n_val_eras: int = 42) -> None:
    """Synthetic data whose scored validation window spans two era-batches
    (_VAL_PREDICT_ERA_BATCH=40): 42 validation eras -> 41 scored -> 40+1."""
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
    for era in range(13, 13 + n_val_eras):
        for idx in range(6):
            f1 = idx * 0.02
            f2 = (idx % 3) * 0.01
            val_rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f1": f1,
                    "f2": f2,
                    "target": (0.6 + 0.03 * era) * f1 - (0.3 + 0.01 * era) * f2 + 0.3 * f1 * f1 + 0.05 * era,
                    "target_alt": (0.2 + 0.02 * era) * f1 + (0.7 - 0.01 * era) * f2 - 0.2 * f2 * f2 - 0.04 * era,
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


def _validation_checkpoint_config(tmp_path, *, device: str = "cpu") -> ExperimentConfig:
    """Validation config with the CV device pinned — the resume refit device
    must be box-independent so the device exact-compare tests are deterministic."""
    cfg = _validation_config(tmp_path)
    return ExperimentConfig(
        data=cfg.data,
        split=cfg.split,
        model=ModelConfig(
            backend=cfg.model.backend,
            preset=cfg.model.preset,
            params=cfg.model.params,
            device=device,
        ),
        evaluation=cfg.evaluation,
        run=cfg.run,
    )


def test_validation_checkpoints_mixed_resume_bit_for_bit(tmp_path, caplog) -> None:
    """Validation mixed resume (spec 2026-08-23-checkpoint-coverage-extension):
    per-era-batch prediction frames persist under validation_checkpoints/;
    deleting one batch and resuming predicts only that batch, and the stage's
    predictions are byte-identical to the uninterrupted run's."""
    data_root = tmp_path / "data"
    _write_synthetic_data_multi_batch(data_root)
    cfg = ExperimentConfig(
        data=DataConfig(
            version="vtest", feature_set="small",
            targets=("target", "target_alt"), data_dir=data_root,
        ),
        split=SplitConfig(scheme="walk_forward", purge_eras=1, embargo_eras=0, n_folds=2),
        model=ModelConfig(
            backend="lightgbm", preset="fast", device="cpu",
            params={"n_estimators": 10, "learning_rate": 0.05, "min_data_in_leaf": 2},
        ),
        evaluation=EvalConfig(
            backend="custom", main_target="target", validation_scorecard=True
        ),
        run=RunConfig(seed=17, artifacts_dir=tmp_path / "artifacts", name="val-ckpt-test"),
    )

    first = ExperimentRunner(cfg).run(deploy=False)
    ckpt_root = paths.run_dir(cfg.run.name, first.run_id) / "validation_checkpoints"
    assert (ckpt_root / "manifest.json").exists()
    assert sorted(p.name for p in ckpt_root.glob("preds_batch_*.parquet")) == [
        "preds_batch_00.parquet",
        "preds_batch_01.parquet",
    ]
    manifest = json.loads((ckpt_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["device"] == "cpu"  # full-history deploy fits are CPU-only
    assert len(manifest["code_sha256"]) == 64
    assert len(manifest["data_fingerprint"]) == 64  # rebuild identity (spec §3.1)
    assert manifest["environment"]
    expected = first.validation_predictions

    (ckpt_root / "preds_batch_00.parquet").unlink()  # delete exactly ONE batch
    caplog.clear()
    with caplog.at_level("INFO", logger="nmr.runner"):
        resumed = ExperimentRunner(cfg).run(deploy=False)
    assert resumed.run_id == first.run_id
    assert resumed.validation_predictions is not None
    assert resumed.validation_predictions.equals(expected)
    assert caplog.text.count("loaded validation checkpoint") == 1
    assert caplog.text.count("predicted and wrote validation checkpoint") == 1


def test_validation_checkpoint_code_mismatch_raises(tmp_path) -> None:
    """Validation manifest identity: a changed fitting-code sha must fail loudly."""
    cfg = _validation_checkpoint_config(tmp_path)
    first = ExperimentRunner(cfg).run(deploy=False)
    ckpt_root = paths.run_dir(cfg.run.name, first.run_id) / "validation_checkpoints"
    manifest_path = ckpt_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["code_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="code_sha256"):
        ExperimentRunner(cfg).run(deploy=False)


def test_validation_checkpoint_unknown_device_raises(tmp_path) -> None:
    """Validation manifest identity: a device outside the known fit devices is
    rejected loudly even while the device is unresolved at resume entry."""
    cfg = _validation_checkpoint_config(tmp_path)
    first = ExperimentRunner(cfg).run(deploy=False)
    ckpt_root = paths.run_dir(cfg.run.name, first.run_id) / "validation_checkpoints"
    manifest_path = ckpt_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["device"] = "totally_different_device"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="device"):
        ExperimentRunner(cfg).run(deploy=False)


def test_validation_checkpoint_device_exact_mismatch_on_resume_raises(tmp_path) -> None:
    """The authoritative device compare runs when a resume knows the current
    device (a CV refit resolves it): a stored 'gpu' device against a 'cpu'
    resume must fail the exact compare instead of passing vacuously."""
    cfg = _validation_checkpoint_config(tmp_path)  # device pinned to "cpu"
    first = ExperimentRunner(cfg).run(deploy=False)
    run_dir = paths.run_dir(cfg.run.name, first.run_id)
    manifest_path = run_dir / "validation_checkpoints" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["device"] = "gpu"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    # Force a CV refit so the resume's resolved device is known ("cpu");
    # the deploy fits and validation batches stay checkpointed.
    fold_path = next((run_dir / "oof_checkpoints" / "target").glob("fold_*.parquet"))
    fold_path.unlink()
    with pytest.raises(ValueError, match="device"):
        ExperimentRunner(cfg).run(deploy=False)


def test_validation_checkpoint_data_fingerprint_mismatch_raises(tmp_path) -> None:
    """Rebuild identity (spec §3.1): a validation manifest recording a
    different data snapshot refuses resume — never silently replays batches
    from a different data snapshot."""
    cfg = _validation_checkpoint_config(tmp_path)
    first = ExperimentRunner(cfg).run(deploy=False)
    ckpt_root = paths.run_dir(cfg.run.name, first.run_id) / "validation_checkpoints"
    manifest_path = ckpt_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data_fingerprint"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="data_fingerprint"):
        ExperimentRunner(cfg).run(deploy=False)


def test_validation_checkpoint_torn_tree_raises(tmp_path) -> None:
    """Prediction batches without a manifest.json are an inconsistent state."""
    cfg = _validation_checkpoint_config(tmp_path)
    first = ExperimentRunner(cfg).run(deploy=False)
    ckpt_root = paths.run_dir(cfg.run.name, first.run_id) / "validation_checkpoints"
    (ckpt_root / "manifest.json").unlink()
    with pytest.raises(ValueError, match="no manifest.json"):
        ExperimentRunner(cfg).run(deploy=False)


def test_validation_checkpoint_corrupt_batch_raises(tmp_path) -> None:
    """A corrupted batch parquet must fail loudly with the path, never
    silently repredict or produce garbage."""
    cfg = _validation_checkpoint_config(tmp_path)
    first = ExperimentRunner(cfg).run(deploy=False)
    ckpt_root = paths.run_dir(cfg.run.name, first.run_id) / "validation_checkpoints"
    (ckpt_root / "preds_batch_00.parquet").write_bytes(b"garbage")
    with pytest.raises(ValueError, match="corrupt validation checkpoint"):
        ExperimentRunner(cfg).run(deploy=False)


def test_validation_checkpoint_dir_only_created_when_stage_runs(tmp_path) -> None:
    """The validation checkpoint dir is created only when the validation
    scorecard stage runs (evaluation.validation_scorecard=true)."""
    cfg = _config(tmp_path)
    result = ExperimentRunner(cfg).run(deploy=False)
    run_dir = paths.run_dir(cfg.run.name, result.run_id)
    assert (run_dir / "oof_checkpoints").exists()
    assert not (run_dir / "validation_checkpoints").exists()
