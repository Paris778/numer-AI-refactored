"""Untiered benchmark fleet: config schema, generators, runner, placement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nmr.benchmark import generate_canonical_predictions
from nmr.benchmark_fleet import (
    _era_sharpe,
    _select_riskiest_features,
    _stack_partitions,
    generate_fleet_lightgbm_predictions,
    generate_fleet_xgb_predictions,
    generate_lagged_target_predictions,
    generate_mlp_predictions,
    generate_ridge_stack_predictions,
    load_fleet_config,
    load_fleet_suite_config,
)
from nmr.risk import NeutralizationEngine


def _write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "fleet.yaml"
    path.write_text(text, encoding="utf-8")
    return path


MINIMAL = """
cells:
  - benchmark_id: silly_target_lag_mean
    source: docs/05-notebooks
    input_space: none
    model_kind: target_lag_mean
    targets: [target]
    params: {window: 1}
"""


def test_minimal_cell_roundtrip(tmp_path):
    cfg = load_fleet_config(_write_yaml(tmp_path, MINIMAL))
    assert len(cfg.cells) == 1
    cell = cfg.cells[0]
    assert cell.benchmark_id == "silly_target_lag_mean"
    assert cell.input_space == "none"
    assert cell.model_kind == "target_lag_mean"
    assert cell.targets == ("target",)
    assert cell.params["window"] == 1
    assert cell.seed == 42
    assert cell.neutralization is None


def test_unknown_cell_key_rejected(tmp_path):
    text = MINIMAL.replace("    params: {window: 1}",
                           "    params: {window: 1}\n    bogus_key: 1")
    with pytest.raises(ValueError, match="Unknown FleetCellConfig keys"):
        load_fleet_config(_write_yaml(tmp_path, text))


def test_tier_key_is_rejected(tmp_path):
    text = MINIMAL.replace("    model_kind: target_lag_mean",
                           "    tier: 3\n    model_kind: target_lag_mean")
    with pytest.raises(ValueError, match="Unknown FleetCellConfig keys"):
        load_fleet_config(_write_yaml(tmp_path, text))


def test_invalid_kind_rejected(tmp_path):
    text = MINIMAL.replace("target_lag_mean", "null_constant_05")
    with pytest.raises(ValueError, match="model_kind"):
        load_fleet_config(_write_yaml(tmp_path, text))


def test_invalid_neutralization_rejected(tmp_path):
    text = MINIMAL.replace(
        "    params: {window: 1}",
        "    params: {window: 1}\n    neutralization: 0.6",
    )
    with pytest.raises(ValueError, match="neutralization"):
        load_fleet_config(_write_yaml(tmp_path, text))


def test_neutralization_none_parses(tmp_path):
    text = MINIMAL.replace(
        "    params: {window: 1}",
        "    params: {window: 1}\n    neutralization: none",
    )
    cell = load_fleet_config(_write_yaml(tmp_path, text)).cells[0]
    assert cell.neutralization is None


def test_lag_mean_requires_none_input_space(tmp_path):
    text = MINIMAL.replace("input_space: none", "input_space: small")
    with pytest.raises(ValueError, match="target_lag_mean requires"):
        load_fleet_config(_write_yaml(tmp_path, text))


def test_lag_mean_requires_single_target(tmp_path):
    text = MINIMAL.replace("targets: [target]", "targets: [target, target_ender_20]")
    with pytest.raises(ValueError, match="single target"):
        load_fleet_config(_write_yaml(tmp_path, text))


def test_ridge_stack_requires_main_target_and_specialists(tmp_path):
    text = """
cells:
  - benchmark_id: rs
    source: legacy
    input_space: small
    model_kind: ridge_stack
    params: {mode: fixed, alpha: 1e-6}
"""
    with pytest.raises(ValueError, match="ridge_stack requires"):
        load_fleet_config(_write_yaml(tmp_path, text))


def test_suite_config_aggregates_and_dedupes(tmp_path):
    (tmp_path / "a.yaml").write_text(MINIMAL, encoding="utf-8")
    (tmp_path / "b.yaml").write_text(
        MINIMAL.replace("silly_target_lag_mean", "other_cell"), encoding="utf-8"
    )
    cells = load_fleet_suite_config(tmp_path)
    assert [c.benchmark_id for c in cells] == ["other_cell", "silly_target_lag_mean"]
    (tmp_path / "b.yaml").write_text(MINIMAL, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate benchmark ids"):
        load_fleet_suite_config(tmp_path)


def _lag_train() -> pl.DataFrame:
    rows = []
    for era in (1, 2, 3, 4):
        for i in range(3):
            rows.append({"era": f"{era:04d}", "target": float(era)})
    return pl.DataFrame(rows)


def _val_index() -> pl.DataFrame:
    return pl.DataFrame(
        [{"era": f"{era:04d}", "id": f"{era}_{i}"}
         for era in (10, 11) for i in range(4)]
    )


def test_lag_mean_window_one_uses_last_train_era():
    out = generate_lagged_target_predictions(
        _lag_train(), _val_index(), target="target", window=1
    )
    assert out.columns == ["era", "id", "prediction"]
    assert out.height == 8
    # last train era is "0004" with target mean 4.0 -> constant for every val row
    assert (out.get_column("prediction") == 4.0).all()


def test_lag_mean_window_two_pools_rows():
    out = generate_lagged_target_predictions(
        _lag_train(), _val_index(), target="target", window=2
    )
    # trailing two train eras: 3.0 and 4.0 -> pooled mean 3.5
    assert (out.get_column("prediction") == 3.5).all()


def test_lag_mean_never_reads_val_targets():
    val = _val_index()  # deliberately has no target column at all
    out = generate_lagged_target_predictions(_lag_train(), val, target="target")
    assert out.height == 8


def test_lag_mean_rejects_bad_window():
    with pytest.raises(ValueError, match="window"):
        generate_lagged_target_predictions(_lag_train(), _val_index(), target="target", window=0)
    with pytest.raises(ValueError, match="window"):
        generate_lagged_target_predictions(_lag_train(), _val_index(), target="target", window=5)


def test_lag_mean_rejects_val_era_overlap():
    bad_train = _lag_train().with_columns(pl.lit("0010").alias("era"))
    with pytest.raises(ValueError, match="strictly earlier"):
        generate_lagged_target_predictions(bad_train, _val_index(), target="target")


def _tiny_train_val(eras: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)):
    """Synthetic train (with target) + val (no targets) with 3 features.

    Val eras derive from ``max(eras) + 1`` and ``max(eras) + 2`` so the
    purge gap between the trimmed train tail and the first val era is
    exactly 8 (the default ``purge_eras``) for any window length. Val
    ``f1``/``f3`` are deterministic per-row drifts (``row``, ``row % 2``)
    so shallow tree fits produce distinct predictions within each era —
    rank-gaussianization of tied predictions is not mean-zero/unit-variance.
    """
    rng = np.random.default_rng(11)
    train_rows, val_rows = [], []
    for i, era in enumerate(eras):
        for row in range(5):
            train_rows.append({
                "era": f"{era:04d}", "id": f"t{era}_{row}",
                "f1": float(i) + rng.normal(0, 0.01),
                "f2": rng.normal(0, 1),
                "f3": float(era % 2),
                "target": float(i % 3),
            })
    for era in (max(eras) + 1, max(eras) + 2):
        for row in range(5):
            val_rows.append({
                "era": f"{era:04d}", "id": f"v{era}_{row}",
                "f1": float(row), "f2": rng.normal(0, 1), "f3": float(row % 2),
            })
    return pl.DataFrame(train_rows), pl.DataFrame(val_rows)


def test_select_riskiest_ranks_by_cross_regime_variance():
    train, _ = _tiny_train_val()
    # f1 drifts monotonically across eras -> highest cross_regime_variance
    out = _select_riskiest_features(
        train, feature_cols=["f1", "f2", "f3"], target_col="target", count=1
    )
    assert out == ["f1"]


def test_fleet_lgbm_selection_none_matches_canonical():
    train, val = _tiny_train_val()
    params = {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2, "num_leaves": 4}
    fleet_out = generate_fleet_lightgbm_predictions(
        train, val, targets=["target"], feature_cols=["f1", "f2", "f3"],
        params=params, seed=42, neutralization=0.5, neutralizer_selection="none",
    )
    canonical = generate_canonical_predictions(
        train, val, targets=["target"], feature_cols=["f1", "f2", "f3"],
        params=params, seed=42, neutralization=0.5,
    )
    assert fleet_out.equals(canonical)


def test_fleet_lgbm_riskiest_50_matches_manual_pipeline():
    train, val = _tiny_train_val()
    params = {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2, "num_leaves": 4}
    neutralizers = _select_riskiest_features(
        train, feature_cols=["f1", "f2", "f3"], target_col="target", count=2
    )
    fleet_out = generate_fleet_lightgbm_predictions(
        train, val, targets=["target"], feature_cols=["f1", "f2", "f3"],
        params=params, seed=42, neutralization=1.0,
        neutralizer_selection="riskiest_50", neutralizer_count=2,
    )
    raw = generate_canonical_predictions(
        train, val, targets=["target"], feature_cols=["f1", "f2", "f3"],
        params=params, seed=42, neutralization=0.0,
    )
    with_features = raw.join(
        val.select(["era", "id", *neutralizers]), on=["era", "id"], how="inner"
    )
    manual = NeutralizationEngine().neutralize(
        with_features, pred_col="prediction",
        feature_cols=neutralizers, era_col="era", proportion=1.0,
    ).select(["era", "id", "prediction"]).sort(["era", "id"])
    assert fleet_out.equals(manual)


def test_fleet_lgbm_rejects_unknown_selection():
    train, val = _tiny_train_val()
    with pytest.raises(ValueError, match="neutralizer_selection"):
        generate_fleet_lightgbm_predictions(
            train, val, targets=["target"], feature_cols=["f1"],
            params={"n_estimators": 5}, seed=42,
            neutralizer_selection="bogus",
        )


def test_xgb_weighted_blend_normalizes_weights_and_orders_ranks():
    train, val = _tiny_train_val()
    params = {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1}
    train = train.with_columns(pl.col("target").alias("target_ender_20"))
    out = generate_fleet_xgb_predictions(
        train, val, targets=["target", "target_ender_20"],
        feature_cols=["f1", "f2", "f3"], params=params, seed=42,
        target_weights={"target": 0.35, "target_ender_20": 0.65},
    )
    assert out.columns == ["era", "id", "prediction"]
    # rank-gaussianized per era: mean ~0, std ~1 within each era
    for era in out.get_column("era").unique().to_list():
        vals = out.filter(pl.col("era") == era).get_column("prediction").to_numpy()
        assert abs(float(vals.mean())) < 1e-6
        assert abs(float(vals.std(ddof=0)) - 1.0) < 1e-6


def test_xgb_weights_default_to_equal():
    train, val = _tiny_train_val()
    params = {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1}
    a = generate_fleet_xgb_predictions(
        train, val, targets=["target"], feature_cols=["f1"],
        params=params, seed=42,
    )
    b = generate_fleet_xgb_predictions(
        train, val, targets=["target"], feature_cols=["f1"],
        params=params, seed=42, target_weights={"target": 2.0},
    )
    assert a.equals(b)  # single target: weight value is irrelevant


def test_xgb_early_stopping_engages_on_holdout():
    train, val = _tiny_train_val(eras=tuple(range(1, 25)))
    params = {
        "n_estimators": 50, "max_depth": 2, "learning_rate": 0.1,
        "early_stopping_rounds": 5, "holdout_era_frac": 0.25,
    }
    out = generate_fleet_xgb_predictions(
        train, val, targets=["target"], feature_cols=["f1", "f2", "f3"],
        params=params, seed=42,
    )
    assert out.height == 10


def test_xgb_rejects_weight_for_unknown_target():
    train, val = _tiny_train_val()
    with pytest.raises(ValueError, match="target_weights"):
        generate_fleet_xgb_predictions(
            train, val, targets=["target"], feature_cols=["f1"],
            params={"n_estimators": 5}, seed=42,
            target_weights={"bogus": 1.0},
        )


_MLP_KEYS = (
    "hidden_layer_sizes", "activation", "solver", "alpha", "learning_rate_init",
    "batch_size", "max_iter", "early_stopping", "n_iter_no_change",
    "validation_fraction",
)


def test_mlp_same_seed_is_deterministic():
    train, val = _tiny_train_val()
    params = {"hidden_layer_sizes": (8, 4), "max_iter": 30, "batch_size": 4,
              "learning_rate_init": 0.01}
    a = generate_mlp_predictions(
        train, val, target="target", feature_cols=["f1", "f2", "f3"],
        params=params, seed=42,
    )
    b = generate_mlp_predictions(
        train, val, target="target", feature_cols=["f1", "f2", "f3"],
        params=params, seed=42,
    )
    assert a.equals(b)


def test_mlp_constant_feature_stays_finite():
    train, val = _tiny_train_val()
    train = train.with_columns(pl.lit(1.0).alias("f4"))
    val = val.with_columns(pl.lit(1.0).alias("f4"))
    out = generate_mlp_predictions(
        train, val, target="target", feature_cols=["f1", "f2", "f3", "f4"],
        params={"hidden_layer_sizes": (8, 4), "max_iter": 30, "batch_size": 4},
        seed=42,
    )
    assert out.get_column("prediction").is_finite().all()


def test_mlp_rejects_unknown_param_key():
    train, val = _tiny_train_val()
    with pytest.raises(ValueError, match="unknown mlp param"):
        generate_mlp_predictions(
            train, val, target="target", feature_cols=["f1"],
            params={"hidden_layer_sizes": (4,), "bogus": 1}, seed=42,
        )


def test_stack_partitions_tail_and_purge_8():
    eras = [f"{e:04d}" for e in range(1, 31)]
    spec, meta = _stack_partitions(eras, meta_tail_pct=0.10, specialists=["target_x_20"])
    assert meta == [f"{e:04d}" for e in range(28, 31)]   # trailing 3
    assert spec[-1] == "0019"                             # 8-era purge (0020..0027)
    assert set(spec) & set(meta) == set()


def test_stack_partitions_purge_16_when_60d_present():
    eras = [f"{e:04d}" for e in range(1, 40)]
    spec, meta = _stack_partitions(eras, meta_tail_pct=0.10, specialists=["target_ender_60"])
    assert spec[-1] == "0019"  # meta = 0036..0039, 16-era purge = 0020..0035
    assert set(spec) & set(meta) == set()


def test_stack_partitions_raises_when_not_enough_eras():
    eras = [f"{e:04d}" for e in range(1, 8)]
    with pytest.raises(ValueError, match="stack split"):
        _stack_partitions(eras, meta_tail_pct=0.5, specialists=["target_x_20"])


def _stack_train_val():
    rng = np.random.default_rng(23)
    train_rows = []
    for era in range(1, 41):
        for row in range(6):
            train_rows.append({
                "era": f"{era:04d}", "id": f"t{era}_{row}",
                "f1": rng.normal(0, 1), "f2": rng.normal(0, 1),
                "main": rng.normal(0, 1),
                "aux1": rng.normal(0, 1),
                "aux2": rng.normal(0, 1),
            })
    for i, row in enumerate(train_rows):  # NaN some aux2 rows: per-target masking
        if i % 7 == 0:
            row["aux2"] = None
    val_rows = [
        {"era": f"{era:04d}", "id": f"v{era}_{row}", "f1": rng.normal(0, 1), "f2": rng.normal(0, 1)}
        for era in (41, 42) for row in range(6)
    ]
    return pl.DataFrame(train_rows), pl.DataFrame(val_rows)


def test_ridge_stack_fixed_never_reads_val_targets_and_is_ranked():
    train, val = _stack_train_val()
    out = generate_ridge_stack_predictions(  # val has no target cols — purity contract
        train, val, main_target="main", specialists=["aux1", "aux2"],
        feature_cols=["f1", "f2"],
        params={"mode": "fixed", "alpha": 1e-6, "meta_alpha": 1e-6, "meta_tail_pct": 0.10},
        seed=42,
    )
    assert out.columns == ["era", "id", "prediction"]
    for era in out.get_column("era").unique().to_list():
        vals = out.filter(pl.col("era") == era).get_column("prediction").to_numpy()
        assert abs(float(vals.mean())) < 1e-6


def test_ridge_stack_fixed_is_seed_deterministic():
    train, val = _stack_train_val()
    params = {"mode": "fixed", "alpha": 1e-6, "meta_alpha": 1e-6, "meta_tail_pct": 0.10}
    a = generate_ridge_stack_predictions(
        train, val, main_target="main", specialists=["aux1", "aux2"],
        feature_cols=["f1", "f2"], params=params, seed=42,
    )
    b = generate_ridge_stack_predictions(
        train, val, main_target="main", specialists=["aux1", "aux2"],
        feature_cols=["f1", "f2"], params=params, seed=42,
    )
    assert a.equals(b)


def test_ridge_stack_rejects_unknown_mode():
    train, val = _stack_train_val()
    with pytest.raises(ValueError, match="mode"):
        generate_ridge_stack_predictions(
            train, val, main_target="main", specialists=["aux1"],
            feature_cols=["f1"], params={"mode": "bogus", "alpha": 1.0}, seed=42,
        )


def test_era_sharpe_matches_manual_mean_over_std():
    eras = ["0001", "0001", "0002", "0002"]
    preds = np.array([0.1, 0.2, 0.9, 0.4])
    target = np.array([0.2, 0.4, 0.8, 0.3])
    corr1 = float(np.corrcoef([0.1, 0.2], [0.2, 0.4])[0, 1])
    corr2 = float(np.corrcoef([0.9, 0.4], [0.8, 0.3])[0, 1])
    manual = (corr1 + corr2) / 2.0 / (np.std([corr1, corr2], ddof=0) + 1e-12)
    assert abs(_era_sharpe(preds, eras, target) - manual) < 1e-9


def _search_train_val():
    rng = np.random.default_rng(31)
    train_rows = []
    for era in range(1, 61):
        signal = float(era % 5)
        for row in range(8):
            train_rows.append({
                "era": f"{era:04d}", "id": f"t{era}_{row}",
                "f1": rng.normal(0, 1), "f2": rng.normal(0, 1),
                "main": signal + rng.normal(0, 0.5),
                "aux1": signal + rng.normal(0, 0.5),
                "aux2": rng.normal(0, 1),  # useless specialist
            })
    val_rows = [
        {"era": f"{era:04d}", "id": f"v{era}_{row}",
         "f1": rng.normal(0, 1), "f2": rng.normal(0, 1),
         "main": float(era % 5) + rng.normal(0, 0.5)}
        for era in (61, 62) for row in range(8)
    ]
    bench_rows = [
        {"era": f"{era:04d}", "id": f"v{era}_{row}", "v53_lgbm_ender20": rng.normal(0, 1)}
        for era in (61, 62) for row in range(8)
    ]
    return pl.DataFrame(train_rows), pl.DataFrame(val_rows), pl.DataFrame(bench_rows)


_SEARCH_PARAMS = {
    "mode": "search",
    "meta_tail_pct": 0.10,
    "snnr_weights": {"aux1": 0.9, "aux2": 0.1},
    "top_k": 2,
    "min_coverage": 0.5,
    "min_abs_main_corr": 0.01,
    "priority_hints": [],
    "specialist_alpha_grid": [0.01, 1.0, 100.0],
    "specialist_sharpe_floor": -10.0,   # permissive on synthetic noise
    "min_specialists": 1,
    "meta_alpha_grid": [0.01, 1.0],
    "meta_lgbm_params": {
        "max_depth": 2, "n_estimators": 10, "learning_rate": 0.1,
        "colsample_bytree": 0.8, "subsample": 0.8, "reg_lambda": 1.0,
        "early_stopping_rounds": 5, "valid_tail_pct": 0.2, "min_valid_eras": 1,
    },
    "decorr_grid": [0.0],
    "neutralization_grid": [0.0],
    "benchmark_col": "v53_lgbm_ender20",
    "nan_fill": True,
}


def test_ridge_stack_search_runs_and_ranks():
    train, val, bench = _search_train_val()
    out = generate_ridge_stack_predictions(
        train, val, main_target="main", specialists=["aux1", "aux2"],
        feature_cols=["f1", "f2"], params=_SEARCH_PARAMS, seed=42,
        val_targets=val.select(["era", "id", "main"]),
        benchmarks=bench,
    )
    assert out.columns == ["era", "id", "prediction"]
    assert out.height == 16
    for era in out.get_column("era").unique().to_list():
        vals = out.filter(pl.col("era") == era).get_column("prediction").to_numpy()
        assert abs(float(vals.mean())) < 1e-6


def test_ridge_stack_search_requires_val_targets_and_benchmarks():
    train, val, bench = _search_train_val()
    with pytest.raises(ValueError, match="val_targets"):
        generate_ridge_stack_predictions(
            train, val, main_target="main", specialists=["aux1"],
            feature_cols=["f1"], params=_SEARCH_PARAMS, seed=42,
            benchmarks=bench,
        )
    with pytest.raises(ValueError, match="benchmarks"):
        generate_ridge_stack_predictions(
            train, val, main_target="main", specialists=["aux1"],
            feature_cols=["f1"], params=_SEARCH_PARAMS, seed=42,
            val_targets=val.select(["era", "id", "main"]),
        )


def test_ridge_stack_search_pruning_floor_raises():
    train, val, bench = _search_train_val()
    params = dict(_SEARCH_PARAMS)
    params["specialist_sharpe_floor"] = 1000.0
    with pytest.raises(ValueError, match="min_specialists"):
        generate_ridge_stack_predictions(
            train, val, main_target="main", specialists=["aux1", "aux2"],
            feature_cols=["f1", "f2"], params=params, seed=42,
            val_targets=val.select(["era", "id", "main"]),
            benchmarks=bench,
        )


def test_ridge_stack_search_is_seed_deterministic():
    train, val, bench = _search_train_val()
    a = generate_ridge_stack_predictions(
        train, val, main_target="main", specialists=["aux1", "aux2"],
        feature_cols=["f1", "f2"], params=_SEARCH_PARAMS, seed=42,
        val_targets=val.select(["era", "id", "main"]), benchmarks=bench,
    )
    b = generate_ridge_stack_predictions(
        train, val, main_target="main", specialists=["aux1", "aux2"],
        feature_cols=["f1", "f2"], params=_SEARCH_PARAMS, seed=42,
        val_targets=val.select(["era", "id", "main"]), benchmarks=bench,
    )
    assert a.equals(b)
