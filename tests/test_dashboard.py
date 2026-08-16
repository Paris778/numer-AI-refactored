from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import nmr.dashboard as dash
import nmr.evaluation as nmr_evaluation
import nmr.payout as payout
from nmr.config import REPO_ROOT


def test_resolve_benchmark_path_prefers_given_existing(tmp_path: Path) -> None:
    given = tmp_path / "reports" / "benchmark_hierarchy_scorecard.csv"
    given.parent.mkdir(parents=True)
    given.write_text("x", encoding="utf-8")
    assert dash.resolve_benchmark_path(given) == given


def test_resolve_benchmark_path_chain_falls_back(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    smoke = reports / "benchmark_hierarchy_scorecard_smoke.csv"
    smoke.write_text("x", encoding="utf-8")
    legacy = tmp_path / "benchmark_scores.csv"
    legacy.write_text("x", encoding="utf-8")
    # given path missing -> chain: full (missing) -> smoke (hit)
    assert dash.resolve_benchmark_path(reports / "nope.csv", reports_dir=reports, legacy_path=legacy) == smoke
    smoke.unlink()
    assert dash.resolve_benchmark_path(reports / "nope.csv", reports_dir=reports, legacy_path=legacy) == legacy
    legacy.unlink()
    assert dash.resolve_benchmark_path(reports / "nope.csv", reports_dir=reports, legacy_path=legacy) is None


def test_resolve_benchmark_path_false_disables_chain(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "benchmark_hierarchy_scorecard_smoke.csv").write_text("x", encoding="utf-8")
    assert dash.resolve_benchmark_path(False, reports_dir=reports) is None


def test_unified_schema_contains_leaderboard_projection() -> None:
    # dashboard_app._LEADERBOARD_SCHEMA is a subset the app wrapper projects onto.
    required = {
        "model_id", "source", "run_name", "backend", "preset", "feature_set",
        "feature_subset", "n_targets", "targets", "neutralization_proportion",
        "oof_device", "corr", "corr_ci_low", "corr_ci_high",
        "corr_sharpe_ac", "corr_sharpe_ac_ci_low", "corr_sharpe_ac_ci_high",
        "max_drawdown", "std_corr", "deflated_sharpe", "max_feature_exposure",
        "has_bmc", "has_horizon", "has_perturb", "has_regime", "run_dir",
    }
    assert required <= set(dash.UNIFIED_SCHEMA.names())
    # capital cells required by the executive table
    for col in ("cagr_1y", "gain_to_pain_ratio", "kelly_fraction", "mmc_down",
                "fnc", "mmc", "mean_payout", "n_eras", "tier", "turnover_mean"):
        assert col in dash.UNIFIED_SCHEMA.names()


def _write_benchmark_csv(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "bench.csv"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_benchmark_frame_full_and_minimal(tmp_path: Path) -> None:
    full = _write_benchmark_csv(
        tmp_path,
        "model_id,corr,corr_ci_low,corr_ci_high,corr_sharpe_ac,"
        "corr_sharpe_ac_ci_low,corr_sharpe_ac_ci_high,std_corr,max_drawdown,"
        "deflated_sharpe,fnc,cagr_1y,gain_to_pain_ratio,kelly_fraction,"
        "mmc_down,strategy_group,tier\n"
        "v53_lgbm_ender60,0.029,0.022,0.036,0.78,0.61,0.95,0.02,0.04,"
        "1.0,0.027,4.88,44.28,1.0,0.009,ref,4\n",
    )
    frame = dash.load_benchmark_frame(full)
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["model_id"] == "v53_lgbm_ender60"
    assert row["source"] == "benchmark"
    assert row["run_name"] == "ref"
    assert row["tier"] == 4
    assert row["cagr_1y"] == pytest.approx(4.88)
    assert row["gain_to_pain_ratio"] == pytest.approx(44.28)
    assert row["corr_sharpe_ac_ci_low"] == pytest.approx(0.61)

    minimal = _write_benchmark_csv(
        tmp_path,
        "model_id,corr,corr_sharpe_ac,std_corr,max_drawdown,strategy_group\n"
        "bench_a,0.05,0.5,0.3,0.2,linear\n",
    )
    row = dash.load_benchmark_frame(minimal).row(0, named=True)
    assert row["corr_sharpe_ac_ci_low"] is None
    assert row["fnc"] is None
    assert row["has_bmc"] is False


def test_load_benchmark_frame_missing_file_returns_empty_schema_frame(tmp_path: Path) -> None:
    frame = dash.load_benchmark_frame(tmp_path / "missing.csv")
    assert frame.height == 0
    assert frame.schema == dash.UNIFIED_SCHEMA


def _registry_entry(run_id: str, *, scorecard: bool = True) -> dict:
    entry = {
        "run_id": run_id,
        "metrics": {"mean": 0.1, "std": 0.2, "sharpe": 0.5, "max_drawdown": 0.05},
        "manifest": {
            "oof_device": "cpu",
            "config": {
                "data": {"feature_set": "small", "feature_subset": None,
                         "targets": ["target"]},
                "model": {"backend": "lightgbm", "preset": "fast"},
                "risk": {"neutralization_proportion": 1.0},
                "run": {"name": "sample-run"},
            },
        },
        "scorecard": None if not scorecard else {
            "corr": 0.12, "corr_ci_low": 0.05, "corr_ci_high": 0.19, "corr_n_eras": 30,
            "corr_sharpe_ac": 0.8, "corr_sharpe_ac_ci_low": 0.6, "corr_sharpe_ac_ci_high": 1.0,
            "max_drawdown": 0.1, "std_corr": 0.2, "deflated_sharpe": 0.97,
            "max_feature_exposure": 0.3, "bmc": 0.02, "fnc": 0.05, "n_eras": 30,
            "cagr_1y": 1.5, "gain_to_pain_ratio": 2.0, "kelly_fraction": 0.4,
            "mmc_down": 0.01, "mmc_down_reason": None,
        },
    }
    return entry


def _write_registry(tmp_path: Path, entries: list[dict]) -> None:
    for entry in entries:
        run_dir = tmp_path / entry["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps(entry), encoding="utf-8")


def test_load_unified_leaderboard_registry_only(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64), _registry_entry("b" * 64, scorecard=False)])
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    assert frame.height == 2
    rows = {r["model_id"]: r for r in frame.to_dicts()}
    assert rows["a" * 64]["source"] == "trained"
    assert rows["a" * 64]["corr"] == 0.12
    assert rows["b" * 64]["source"] == "trained_legacy"
    assert rows["b" * 64]["corr"] == 0.1  # legacy falls back to metrics.mean
    assert rows["a" * 64]["cagr_1y"] == 1.5  # stored capital block carried through


def test_load_unified_leaderboard_zero_scorecard_value_not_legacy(tmp_path: Path) -> None:
    entry = _registry_entry("c" * 64)
    entry["scorecard"]["corr"] = 0.0  # legitimate 0.0 must NOT fall through
    _write_registry(tmp_path, [entry])
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    assert frame.row(0, named=True)["corr"] == 0.0
    assert frame.row(0, named=True)["source"] == "trained"


def test_load_unified_leaderboard_corrupt_run_json_skipped(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("d" * 64)])
    bad_dir = tmp_path / ("e" * 64)
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "run.json").write_text("{not json", encoding="utf-8")
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    assert frame.height == 1


def test_load_unified_leaderboard_merges_benchmarks(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("f" * 64)])
    bench = _write_benchmark_csv(
        tmp_path,
        "model_id,corr,corr_sharpe_ac,std_corr,max_drawdown,strategy_group,tier\n"
        "bench_a,0.05,0.5,0.3,0.2,linear,1\n",
    )
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=bench)
    assert frame.height == 2
    assert set(frame.get_column("source").to_list()) == {"trained", "benchmark"}


def test_load_unified_leaderboard_empty_registry_returns_schema_frame(tmp_path: Path) -> None:
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    assert frame.height == 0
    assert frame.schema == dash.UNIFIED_SCHEMA


_GATE_YAML = REPO_ROOT / "configs" / "benchmarks" / "tier4_gate.yaml"


def _status_frame(tmp_path: Path, rows: list[dict]) -> pl.DataFrame:
    frame = pl.DataFrame(rows, schema=dash.UNIFIED_SCHEMA, strict=False)
    return dash.evaluate_gate_status(frame, _GATE_YAML, tmp_path / "champion.json")


def test_gate_status_research_and_capital_ready(tmp_path: Path) -> None:
    base = {
        "model_id": "r1", "source": "trained", "corr": 0.01,
        "corr_sharpe_ac": 0.2, "fnc": 0.001, "deflated_sharpe": 0.5,
        "gain_to_pain_ratio": 1.0, "cagr_1y": 0.1, "turnover_mean": None,
    }
    frame = _status_frame(tmp_path, [base])
    assert frame.row(0, named=True)["status"] == "RESEARCH"
    assert frame.row(0, named=True)["gate_corr"] is False

    passing = dict(base)
    passing.update({"model_id": "r2", "corr": 0.03, "corr_sharpe_ac": 0.8,
                    "fnc": 0.03, "deflated_sharpe": 0.96,
                    "gain_to_pain_ratio": 1.6, "cagr_1y": 0.01})
    frame = _status_frame(tmp_path, [passing])
    assert frame.row(0, named=True)["status"] == "CAPITAL READY"
    assert frame.row(0, named=True)["gate_corr"] is True
    assert frame.row(0, named=True)["gate_cagr_1y"] is True  # strict > 0.0


def test_gate_status_champion_via_pointer(tmp_path: Path) -> None:
    (tmp_path / "champion.json").write_text(
        json.dumps({"run_id": "ch" * 32}), encoding="utf-8"
    )
    frame = _status_frame(tmp_path, [{"model_id": "ch" * 32, "source": "trained",
                                      "corr": 0.0, "corr_sharpe_ac": 0.0,
                                      "fnc": 0.0, "deflated_sharpe": 0.0,
                                      "gain_to_pain_ratio": 0.0,
                                      "cagr_1y": 0.0, "turnover_mean": None}])
    assert frame.row(0, named=True)["status"] == "CHAMPION"


def test_gate_status_benchmark_rows_never_capital_ready(tmp_path: Path) -> None:
    ref = {"model_id": "v53_lgbm_ender60", "source": "benchmark", "corr": 0.029,
           "corr_sharpe_ac": 0.78, "fnc": 0.027, "deflated_sharpe": 1.0,
           "gain_to_pain_ratio": 44.0, "cagr_1y": 4.88, "turnover_mean": None}
    other = dict(ref, model_id="null_constant_05", corr=0.0, corr_sharpe_ac=0.0)
    frame = _status_frame(tmp_path, [ref, other])
    statuses = {r["model_id"]: r["status"] for r in frame.to_dicts()}
    assert statuses["v53_lgbm_ender60"] == "GATE HURDLE"
    assert statuses["null_constant_05"] == "BENCHMARK"
    ref_row = frame.filter(pl.col("model_id") == "v53_lgbm_ender60").row(0, named=True)
    assert ref_row["gate_corr_sharpe_ac"] is True
    assert ref_row["gate_turnover_mean"] is None  # turnover absent -> exempt


def test_gate_status_turnover_violation_when_present(tmp_path: Path) -> None:
    row = {"model_id": "r3", "source": "trained", "corr": 0.03,
           "corr_sharpe_ac": 0.8, "fnc": 0.03, "deflated_sharpe": 0.96,
           "gain_to_pain_ratio": 1.6, "cagr_1y": 0.01, "turnover_mean": 0.9}
    frame = _status_frame(tmp_path, [row])
    out = frame.row(0, named=True)
    assert out["gate_turnover_mean"] is False  # 0.9 > 0.35
    assert out["status"] == "RESEARCH"


def _synthetic_data_dir(tmp_path: Path) -> Path:
    """era/id/target + meta parquets over 3 eras with a perfectly
    correlated predictor so recomputed values are exactly known."""
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    for era in ("0001", "0002", "0003"):
        for i in range(10):
            t = float(i)
            rows.append({"era": era, "id": f"{era}_{i:03d}", "target": t})
    targets = pl.DataFrame(rows)
    meta = targets.select(
        ["era", "id", pl.col("target").alias("numerai_meta_model")]
    )
    targets.write_parquet(data / "validation.parquet")
    meta.write_parquet(data / "meta_model.parquet")
    return data


def _write_preds(run_dir: Path, scale: float) -> None:
    preds = [
        {"era": era, "id": f"{era}_{i:03d}", "prediction": scale * float(i)}
        for era in ("0001", "0002", "0003")
        for i in range(10)
    ]
    pl.DataFrame(preds).write_parquet(run_dir / "validation_preds.parquet")


def test_reconcile_capital_metrics_recomputes_missing_block(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64, scorecard=True)])
    entry = json.loads((tmp_path / ("a" * 64) / "run.json").read_text(encoding="utf-8"))
    del entry["scorecard"]["cagr_1y"]
    del entry["scorecard"]["gain_to_pain_ratio"]
    del entry["scorecard"]["kelly_fraction"]
    del entry["scorecard"]["mmc_down"]
    (tmp_path / ("a" * 64) / "run.json").write_text(json.dumps(entry), encoding="utf-8")
    _write_preds(tmp_path / ("a" * 64), scale=1.0)
    data = _synthetic_data_dir(tmp_path)

    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    out = dash.reconcile_capital_metrics(frame, tmp_path, data)
    row = out.row(0, named=True)
    assert row["cagr_1y"] is not None
    assert row["gain_to_pain_ratio"] is not None
    assert row["kelly_fraction"] is not None
    # perfect corr with target and meta -> payout == 0.05 clipped every era
    assert row["cagr_1y"] == pytest.approx((1.05 ** 52) - 1.0, abs=1e-6)
    assert row["gain_to_pain_ratio"] == float("inf")  # no losing eras


def test_reconcile_capital_metrics_stored_block_wins(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("b" * 64)])
    _write_preds(tmp_path / ("b" * 64), scale=0.0)  # junk preds must be ignored
    data = _synthetic_data_dir(tmp_path)
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    out = dash.reconcile_capital_metrics(frame, tmp_path, data)
    row = out.row(0, named=True)
    assert row["cagr_1y"] == 1.5       # stored value untouched
    assert row["gain_to_pain_ratio"] == 2.0
    assert row["kelly_fraction"] == 0.4
    assert row["mmc_down"] == 0.01


def test_reconcile_capital_metrics_missing_preds_degrades(tmp_path: Path) -> None:
    entry = _registry_entry("c" * 64)
    del entry["scorecard"]["cagr_1y"]
    del entry["scorecard"]["gain_to_pain_ratio"]
    del entry["scorecard"]["kelly_fraction"]
    _write_registry(tmp_path, [entry])
    data = _synthetic_data_dir(tmp_path)  # no validation_preds.parquet written
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    out = dash.reconcile_capital_metrics(frame, tmp_path, data)
    row = out.row(0, named=True)
    assert row["cagr_1y"] is None
    assert row["kelly_fraction"] is None


def test_reconcile_capital_metrics_missing_data_assets_noop(tmp_path: Path) -> None:
    entry = _registry_entry("d" * 64)
    del entry["scorecard"]["cagr_1y"]
    del entry["scorecard"]["gain_to_pain_ratio"]
    del entry["scorecard"]["kelly_fraction"]
    _write_registry(tmp_path, [entry])
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    out = dash.reconcile_capital_metrics(frame, tmp_path, tmp_path / "no-data")
    assert out.row(0, named=True)["cagr_1y"] is None


def test_extract_payout_timeseries_shape_and_determinism(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64), _registry_entry("b" * 64)])
    for run_id, scale in (("a" * 64, 1.0), ("b" * 64, -0.5)):
        _write_preds(tmp_path / run_id, scale=scale)
    data = _synthetic_data_dir(tmp_path)

    payload = dash.extract_payout_timeseries(
        tmp_path, data, run_ids=["b" * 64, "a" * 64], include_tier4_ref=False
    )
    assert payload["eras"] == ["0001", "0002", "0003"]  # numeric order
    assert len(payload["meta_downside_mask"]) == 3
    assert set(payload["series"]) == {"a" * 64, "b" * 64}
    for series in payload["series"].values():
        assert len(series["cumulative_wealth"]) == 3
        assert len(series["drawdown"]) == 3
        assert series["mdd"] <= 0.0
        assert isinstance(series["cagr"], float)
        assert series["label"]

    # determinism: identical payload hash across repeated runs and insertion orders
    again = dash.extract_payout_timeseries(
        tmp_path, data, run_ids=["a" * 64, "b" * 64], include_tier4_ref=False
    )
    assert json.dumps(again, sort_keys=True) == json.dumps(payload, sort_keys=True)

    # perfect-correlation series: wealth compounds at +5% per era, drawdown == 0
    perfect = payload["series"]["a" * 64]
    assert perfect["cumulative_wealth"][-1] == pytest.approx(1.05**3, abs=1e-9)
    assert perfect["drawdown"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert perfect["mdd"] == pytest.approx(0.0, abs=1e-12)


def test_extract_payout_timeseries_missing_run_skipped(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64)])
    _write_preds(tmp_path / ("a" * 64), scale=1.0)
    data = _synthetic_data_dir(tmp_path)
    payload = dash.extract_payout_timeseries(
        tmp_path, data, run_ids=["a" * 64, "9" * 64], include_tier4_ref=False
    )
    assert set(payload["series"]) == {"a" * 64}


_REAL_VALIDATION = Path("data/v5.3/validation.parquet")
_REAL_META = Path("data/v5.3/meta_model.parquet")
_REAL_BENCH = Path("data/v5.3/validation_benchmark_models.parquet")
_REAL_REGISTRY = Path("artifacts/registry")
_SMOKE_CSV = Path("artifacts/reports/benchmark_hierarchy_scorecard_smoke.csv")
_HAS_REAL = (
    _REAL_VALIDATION.exists()
    and _REAL_META.exists()
    and _REAL_BENCH.exists()
    and _REAL_REGISTRY.exists()
    and any(_REAL_REGISTRY.glob("*/run.json"))
)


@pytest.mark.skipif(not _HAS_REAL, reason="real registry/v5.3 data absent; skipped in CI")
def test_real_recompute_matches_stored_corr() -> None:
    frame = dash.load_unified_leaderboard(_REAL_REGISTRY, benchmark_path=False)
    row = frame.sort("corr_sharpe_ac", descending=True, nulls_last=True).row(0, named=True)
    lookups = dash._load_shared_lookups(Path("data/v5.3"))
    assert lookups is not None
    targets_86, meta, _ = lookups
    preds_path = Path(row["run_dir"]) / "validation_preds.parquet"
    corr, _, _ = dash._per_era_metrics(preds_path, targets_86, meta)
    assert len(corr) == row["corr_n_eras"]
    assert float(np.mean(list(corr.values()))) == pytest.approx(row["corr"], abs=1e-4)


@pytest.mark.skipif(not (_HAS_REAL and _SMOKE_CSV.exists()),
                    reason="real v5.3 data + smoke benchmark CSV absent; skipped in CI")
def test_real_tier4_cagr_matches_smoke_csv() -> None:
    lookups = dash._load_shared_lookups(Path("data/v5.3"))
    assert lookups is not None
    targets_86, meta, _ = lookups
    bench = pl.read_parquet(
        _REAL_BENCH, columns=["era", "id", "v53_lgbm_ender60"]
    )
    axis = sorted(
        meta.get_column("era").unique().to_list(), key=int
    )
    joined = (
        bench.filter(pl.col("era").is_in(axis))
        .join(targets_86, on=["era", "id"], how="inner")
        .join(meta, on=["era", "id"], how="inner")
    )
    engine = nmr_evaluation.EvaluationEngine()  # import nmr.evaluation as nmr_evaluation at top
    corr = engine.per_era_corr(joined, pred_col="v53_lgbm_ender60", target_col="target")
    mmc = engine.per_era_mmc(
        joined, pred_col="v53_lgbm_ender60",
        meta_col="numerai_meta_model", target_col="target",
    )
    pay = payout.payout_series(corr, mmc)  # import nmr.payout as payout at top
    recomputed = payout.annual_compounded_return(pay.clipped)
    stored = dash.load_benchmark_frame(_SMOKE_CSV).filter(
        pl.col("model_id") == "v53_lgbm_ender60"
    ).row(0, named=True)["cagr_1y"]
    assert float(recomputed) == pytest.approx(float(stored), abs=1e-6)


@pytest.mark.skipif(not _HAS_REAL, reason="real registry/v5.3 data absent; skipped in CI")
def test_real_reconcile_populates_all_capital_columns() -> None:
    frame = dash.load_unified_leaderboard(_REAL_REGISTRY, benchmark_path=False)
    out = dash.reconcile_capital_metrics(frame, _REAL_REGISTRY, Path("data/v5.3"))
    trained = out.filter(pl.col("source").is_in(["trained", "trained_legacy"]))
    assert trained.height > 0
    for row in trained.to_dicts():
        assert row["cagr_1y"] is not None
        assert row["gain_to_pain_ratio"] is not None
        assert row["kelly_fraction"] is not None


def test_dashboard_symbols_exported_from_package() -> None:
    import nmr

    for name in (
        "UNIFIED_SCHEMA",
        "evaluate_gate_status",
        "extract_payout_timeseries",
        "load_benchmark_frame",
        "load_unified_leaderboard",
        "reconcile_capital_metrics",
        "resolve_benchmark_path",
    ):
        assert getattr(nmr, name) is not None, f"nmr.{name} not exported"
        assert name in nmr.__all__
