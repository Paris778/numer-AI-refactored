"""Tests for research enablement helpers."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from nmr.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    RunConfig,
    SplitConfig,
)
from nmr.evaluation import EvaluationEngine
from nmr.inference import ac_adjusted_sharpe
from nmr.research import (
    HyperparameterSweep,
    _held_out_metric,
    _held_out_partition,
    _per_era_ac_sharpe,
    feature_exposure_report,
    neutralization_frontier,
)
from nmr.risk import NeutralizationEngine


def _train_frame() -> pl.DataFrame:
    # 25 eras -> held-out = round(0.2*25) = 5 eras (20D AC bandwidth floor is 4,
    # cap n-1, so >= 5 held-out eras are required). Features are bounded periodic
    # functions of era so held-out eras sit inside the train feature envelope; a
    # monotone-in-era feature would drive every held-out row into one tree leaf ->
    # per-era-constant predictions -> zero CORR, which vacuously passes the
    # corr_sharpe_ac end-to-end tests. The (era % 3)-scaled sin(idx) term makes the
    # per-era CORR genuinely vary across held-out eras (verified non-constant).
    rows: list[dict[str, float | str]] = []
    for era in range(1, 26):
        for idx in range(8):
            f1 = 0.35 + 0.12 * np.sin(0.3 * era) + 0.02 * idx
            f2 = 0.10 + 0.08 * np.cos(0.25 * era) + 0.015 * idx
            target = 0.7 * f1 - 0.4 * f2 + (0.01 + 0.02 * (era % 3)) * np.sin(idx)
            target_alt = 0.4 * f1 + 0.6 * f2 - 0.03 * np.cos(idx)
            rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f1": float(f1),
                    "f2": float(f2),
                    "target": float(target),
                    "target_alt": float(target_alt),
                }
            )
    return pl.DataFrame(rows)


def _write_data(tmp_path) -> ExperimentConfig:
    data_root = tmp_path / "data"
    vdir = data_root / "vresearch"
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "features.json").write_text(
        json.dumps(
            {
                "feature_sets": {
                    "small": ["f1", "f2"],
                    "medium": ["f1", "f2"],
                    "all": ["f1", "f2"],
                },
                "targets": ["target", "target_alt"],
            }
        ),
        encoding="utf-8",
    )
    _train_frame().write_parquet(vdir / "train.parquet")
    return ExperimentConfig(
        data=DataConfig(
            version="vresearch",
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
            params={"n_estimators": 1, "learning_rate": 0.05},
        ),
        evaluation=EvalConfig(backend="custom", main_target="target"),
        run=RunConfig(name="research", seed=19, artifacts_dir=tmp_path / "artifacts"),
    )


def test_sweep_is_deterministic_and_held_out(tmp_path) -> None:
    cfg = _write_data(tmp_path)
    space = {"n_estimators": [6], "learning_rate": [0.03, 0.07]}
    first = HyperparameterSweep(cfg, metric="sharpe").run(space, n_trials=2, seed=123)
    second = HyperparameterSweep(cfg, metric="sharpe").run(space, n_trials=2, seed=123)

    assert first.trials.equals(second.trials)
    assert first.best_params == second.best_params
    assert first.best_value == second.best_value


def test_held_out_partition_enforces_purge_gap(tmp_path) -> None:
    cfg = _write_data(tmp_path)
    eras = _train_frame().get_column("era").to_list()
    train_eras, _, held_out_eras = _held_out_partition(
        eras,
        frac=0.2,
        purge_eras=cfg.split.purge_eras,
    )

    train_max = max(map(int, train_eras))
    held_out_min = min(map(int, held_out_eras))
    assert held_out_min - train_max - 1 >= cfg.split.purge_eras


def test_neutralization_frontier_matches_endpoints(tmp_path) -> None:
    cfg = _write_data(tmp_path)
    df = _train_frame()
    proportions = [0.0, 1.0]
    frontier = neutralization_frontier(
        df,
        feature_cols=["f1", "f2"],
        proportions=proportions,
        target_col="target",
        pred_col="target_alt",
    )

    evaluator = EvaluationEngine("custom")
    risk = NeutralizationEngine()

    raw_scores = evaluator.per_era_corr(df, pred_col="target_alt", target_col="target")
    full_neutral = risk.neutralize(
        df,
        pred_col="target_alt",
        feature_cols=["f1", "f2"],
        proportion=1.0,
    )
    full_scores = evaluator.per_era_corr(
        full_neutral, pred_col="target_alt", target_col="target"
    )

    assert frontier.proportions == proportions
    assert frontier.metrics[0] == evaluator.summarize(raw_scores)
    assert frontier.metrics[1] == evaluator.summarize(full_scores)


def test_feature_exposure_report_is_deterministic_and_sorted() -> None:
    df = _train_frame().with_columns(pl.col("target_alt").alias("prediction"))
    first = feature_exposure_report(
        df, feature_cols=["f1", "f2"], pred_col="prediction"
    )
    second = feature_exposure_report(
        df, feature_cols=["f1", "f2"], pred_col="prediction"
    )

    assert first.equals(second)
    assert first.columns == ["feature", "mean_abs_exposure", "max_abs_exposure"]
    values = first.get_column("max_abs_exposure").to_list()
    assert values == sorted(values, reverse=True)


def test_feature_exposure_report_empty_oof_returns_zero_exposures() -> None:
    empty = pl.DataFrame({"era": [], "prediction": [], "f1": []})
    report = feature_exposure_report(empty, feature_cols=["f1"])

    assert report.height == 1
    assert report.get_column("feature").to_list() == ["f1"]
    assert report.get_column("mean_abs_exposure").to_list() == [0.0]
    assert report.get_column("max_abs_exposure").to_list() == [0.0]


def test_per_era_ac_sharpe_sorts_eras_chronologically() -> None:
    # Eras 1..12: insertion order deliberately shuffled; the numeric sort must
    # recover the chronological series ("1","2",...,"12", NOT lexicographic
    # "1","10","11","12","2",...).
    values = {str(era): 0.05 * (era % 7) - 0.1 for era in range(1, 13)}
    items = list(values.items())
    rng = np.random.default_rng(3)
    idx = rng.permutation(len(items))
    shuffled = dict(items[int(i)] for i in idx)

    got = _per_era_ac_sharpe(shuffled, horizon="20D")
    chronological = [values[str(era)] for era in sorted(range(1, 13))]
    expected = ac_adjusted_sharpe(chronological, horizon="20D")
    assert got == expected


def test_per_era_ac_sharpe_differs_from_lexicographic_order() -> None:
    # Prove the sort matters: a lexicographic series gives a different AC value.
    values = {str(era): 0.05 * (era % 7) - 0.1 for era in range(1, 13)}
    lexicographic = [values[k] for k in sorted(values)]  # "1","10","11","12","2",...
    assert _per_era_ac_sharpe(values, horizon="20D") != ac_adjusted_sharpe(
        lexicographic, horizon="20D"
    )


def test_per_era_ac_sharpe_requires_two_eras() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        _per_era_ac_sharpe({"1": 0.1}, horizon="20D")


def test_held_out_metric_supports_corr_sharpe_ac(tmp_path, monkeypatch) -> None:
    from nmr import research

    captured: dict[str, dict[str, float]] = {}
    orig = research._per_era_ac_sharpe

    def recording(per_era, *, horizon="20D"):
        captured["per_era"] = per_era
        return orig(per_era, horizon=horizon)

    monkeypatch.setattr(research, "_per_era_ac_sharpe", recording)
    cfg = _write_data(tmp_path)
    value = _held_out_metric(cfg, metric_name="corr_sharpe_ac")
    series = np.asarray(list(captured["per_era"].values()), dtype=float)
    assert np.isfinite(value)
    assert value != 0.0            # real AC path ran, not the std==0 short-circuit
    assert series.size >= 5        # 20D bandwidth floor: >= 5 held-out eras
    assert np.std(series) > 0.0    # per-era corr genuinely varies


def test_held_out_metric_still_rejects_unknown_metric(tmp_path) -> None:
    cfg = _write_data(tmp_path)
    with pytest.raises(ValueError, match="Unknown metric"):
        _held_out_metric(cfg, metric_name="bogus_metric")


def test_held_out_partition_preserves_zero_padded_labels() -> None:
    """Regression (2026-08-11): the partition returned str(int) era labels
    ("575"), which match nothing in is_in() filters on zero-padded data —
    the HPO held-out evaluation silently dropped every era below 1000."""
    eras = [f"{e:04d}" for e in range(1, 21)]  # "0001".."0020"
    train_eras, purge_eras, held_out_eras = _held_out_partition(
        eras, frac=0.2, purge_eras=2
    )
    # labels must round-trip through the padded data (not str(int))
    assert all(e in eras for e in train_eras + purge_eras + held_out_eras)
    assert set(train_eras) | set(purge_eras) | set(held_out_eras) == set(eras)
    # held-out = last 20% (4 eras: 0017-0020); purge = 2 eras before (0015-0016)
    assert held_out_eras == ["0017", "0018", "0019", "0020"]
    assert purge_eras == ["0015", "0016"]
    assert max(map(int, train_eras)) == 14
