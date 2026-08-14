from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from nmr.evaluation import MIN_OVERLAP_ERAS, NonVacuityError
from nmr.meta import fleet_summary, paired_era_comparison, promotion_verdict


def _frame(n_eras: int = 24) -> pl.DataFrame:
    rows = []
    for era in range(1, n_eras + 1):
        for idx in range(10):
            rows.append({"era": str(era), "id": f"{era}_{idx}", "prediction": idx * 0.1})
    return pl.DataFrame(rows)


def _era_index_metric(frame: pl.DataFrame) -> dict[str, float]:
    """Deterministic per-era metric: the era number itself."""
    return {
        str(era): float(era)
        for era in frame.get_column("era").unique().sort().to_list()
    }


def test_paired_comparison_estimates_mean_difference_with_ci() -> None:
    a = _frame()
    b = _frame()
    result = paired_era_comparison(
        a, b, metric_fn=_era_index_metric, seed=7, n_boot=50,
    )
    assert result.mean_diff == pytest.approx(0.0, abs=1e-9)
    assert result.n_eras == 24
    assert result.device_mismatch is False
    assert result.ci_low <= result.mean_diff <= result.ci_high
    assert result.alpha == 0.05 and result.n_boot == 50


def test_paired_comparison_sign_using_prediction_means() -> None:
    def mean_pred(frame: pl.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        for era in frame.get_column("era").unique().to_list():
            out[str(era)] = float(
                frame.filter(pl.col("era") == era).get_column("prediction").mean()
            )
        return out

    a = _frame()  # prediction = idx * 0.1 -> era mean 0.45
    b = a.with_columns((pl.col("prediction") + 1.0).alias("prediction"))  # era mean 1.45
    result = paired_era_comparison(a, b, metric_fn=mean_pred, seed=7, n_boot=50)
    assert result.mean_diff == pytest.approx(-1.0, abs=1e-9)  # a - b == -1.0


def test_paired_comparison_bootstrap_deterministic_under_seed() -> None:
    def mean_pred(frame: pl.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        for era in frame.get_column("era").unique().to_list():
            out[str(era)] = float(
                frame.filter(pl.col("era") == era).get_column("prediction").mean()
            )
        return out

    a = _frame()  # per-era prediction mean 0.45
    # Era-dependent shift -> per-era diffs (a - b) VARY across eras (-1, -2, ...),
    # so the block-bootstrap resample genuinely exercises the seeded RNG.
    b = a.with_columns(
        (pl.col("prediction") + pl.col("era").cast(pl.Int64)).alias("prediction")
    )
    r1 = paired_era_comparison(a, b, metric_fn=mean_pred, seed=11, n_boot=200)
    r2 = paired_era_comparison(a, b, metric_fn=mean_pred, seed=11, n_boot=200)
    assert r1 == r2  # same seed -> identical CI (cross-process determinism)


def test_paired_comparison_intersects_eras_and_raises_below_overlap_floor() -> None:
    a = _frame(n_eras=24)
    b = _frame(n_eras=10)  # overlap = 10 < MIN_OVERLAP_ERAS
    with pytest.raises(NonVacuityError):
        paired_era_comparison(a, b, metric_fn=_era_index_metric, seed=7)


def test_paired_comparison_device_mismatch_flag() -> None:
    result = paired_era_comparison(
        _frame(), _frame(), metric_fn=_era_index_metric, seed=7,
        device_a="gpu", device_b="cpu",
    )
    assert result.device_mismatch is True
    same = paired_era_comparison(
        _frame(), _frame(), metric_fn=_era_index_metric, seed=7,
        device_a="cpu", device_b="cpu",
    )
    assert same.device_mismatch is False


def test_paired_comparison_raises_when_era_col_missing() -> None:
    a = _frame()
    renamed = _frame().rename({"era": "epoch"})  # lacks default era_col "era"
    with pytest.raises(ValueError, match="oof_b"):
        paired_era_comparison(a, renamed, metric_fn=_era_index_metric, seed=7)
    with pytest.raises(ValueError, match="oof_a"):
        paired_era_comparison(renamed, a, metric_fn=_era_index_metric, seed=7)
    with pytest.raises(ValueError, match="oof_a.*oof_b"):
        paired_era_comparison(renamed, renamed, metric_fn=_era_index_metric, seed=7)


def test_paired_comparison_honors_renamed_era_col() -> None:
    def metric_on_epoch(frame: pl.DataFrame) -> dict[str, float]:
        return {
            str(era): float(era)
            for era in frame.get_column("epoch").unique().sort().to_list()
        }

    a = _frame().rename({"era": "epoch"})
    b = _frame().rename({"era": "epoch"})
    result = paired_era_comparison(
        a, b, metric_fn=metric_on_epoch, era_col="epoch", seed=7, n_boot=50,
    )
    assert result.n_eras == 24
    assert result.mean_diff == pytest.approx(0.0, abs=1e-9)


def _entry(run_id: str, metric: str = "corr_sharpe_ac", *, value: float | None = None,
           lo: float | None = None, hi: float | None = None) -> dict:
    scorecard: dict = {}
    if value is not None:
        scorecard[metric] = value
    if lo is not None:
        scorecard[f"{metric}_ci_low"] = lo
    if hi is not None:
        scorecard[f"{metric}_ci_high"] = hi
    return {"run_id": run_id, "scorecard": scorecard}


def test_verdict_promotes_when_candidate_ci_clears_champion() -> None:
    champion = _entry("c" * 64, value=0.10, lo=0.05, hi=0.15)
    candidate = _entry("d" * 64, value=0.25, lo=0.20, hi=0.30)
    assert promotion_verdict(candidate, champion) == "promote"


def test_verdict_holds_when_candidate_ci_below_champion() -> None:
    champion = _entry("c" * 64, value=0.25, lo=0.20, hi=0.30)
    candidate = _entry("d" * 64, value=0.10, lo=0.05, hi=0.15)
    assert promotion_verdict(candidate, champion) == "hold"


def test_verdict_cautions_on_ci_overlap() -> None:
    champion = _entry("c" * 64, value=0.18, lo=0.10, hi=0.26)
    candidate = _entry("d" * 64, value=0.20, lo=0.14, hi=0.27)
    assert promotion_verdict(candidate, champion) == "caution"


def test_verdict_cautions_when_ci_unavailable() -> None:
    champion = _entry("c" * 64, value=0.10)
    candidate = _entry("d" * 64, value=0.25)
    assert promotion_verdict(candidate, champion) == "caution"


def test_verdict_promotes_without_champion() -> None:
    candidate = _entry("d" * 64, value=0.25, lo=0.20, hi=0.30)
    assert promotion_verdict(candidate, None) == "promote"


def test_verdict_promotes_when_champion_lacks_scorecard() -> None:
    candidate = _entry("d" * 64, value=0.25, lo=0.20, hi=0.30)
    no_scorecard = {"run_id": "c" * 64}  # champion entry with no scorecard at all
    empty_scorecard = _entry("c" * 64)  # scorecard present but metric missing
    assert promotion_verdict(candidate, no_scorecard) == "promote"
    assert promotion_verdict(candidate, empty_scorecard) == "promote"


def test_verdict_lower_is_better_for_max_drawdown() -> None:
    champion = _entry("c" * 64, metric="max_drawdown", value=0.20, lo=0.18, hi=0.22)
    candidate = _entry("d" * 64, metric="max_drawdown", value=0.10, lo=0.08, hi=0.12)
    assert promotion_verdict(candidate, champion, metric="max_drawdown") == "promote"


def test_verdict_directions_match_registry_semantics() -> None:
    from nmr.meta import _VERDICT_DIRECTIONS
    from nmr.registry import _SCORECARD_METRIC_DIRECTION

    assert set(_VERDICT_DIRECTIONS) <= set(_SCORECARD_METRIC_DIRECTION)
    for metric, higher_is_better in _VERDICT_DIRECTIONS.items():
        assert _SCORECARD_METRIC_DIRECTION[metric] == higher_is_better


def test_verdict_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="metric"):
        promotion_verdict(_entry("d" * 64), None, metric="bogus")


def _full_entry(run_id: str, sharpe_ac: float) -> dict:
    return {
        "run_id": run_id,
        "manifest": {
            "oof_device": "cpu",
            "config": {
                "data": {"feature_set": "small", "feature_subset": None},
                "model": {"preset": "fast"},
                "risk": {"neutralization_proportion": 1.0},
            },
        },
        "scorecard": {
            "corr_sharpe_ac": sharpe_ac,
            "corr_sharpe_ac_ci_low": sharpe_ac - 0.05,
            "corr_sharpe_ac_ci_high": sharpe_ac + 0.05,
            "corr_sharpe_ac_n_eras": 30,
            "deflated_sharpe": 0.98,
            "max_feature_exposure": 0.3,
            "bmc": 0.02,
            "horizon_model_sharpe_20": 0.5,
            "perturb_ceiling_stability": 0.9,
            "regime_count": 3,
        },
    }


def test_fleet_summary_columns_and_flags() -> None:
    runs = [_full_entry("a" * 64, 0.12), _full_entry("b" * 64, 0.05)]
    frame = fleet_summary(runs, n_trials=2)
    assert frame.height == 2
    assert set(frame.columns) >= {
        "run_id", "metric", "metric_ci_low", "metric_ci_high", "metric_n_eras",
        "deflated_sharpe", "dsr_pass", "max_feature_exposure", "oof_device",
        "preset", "feature_set", "feature_subset", "neutralization_proportion",
        "has_bmc", "has_horizon", "has_perturb", "has_regime",
        "policy_n_trials", "policy_dsr_confidence",
    }
    first = frame.filter(pl.col("run_id") == "a" * 64).row(0, named=True)
    assert first["dsr_pass"] is True
    assert first["oof_device"] == "cpu"
    assert first["preset"] == "fast"
    assert first["feature_set"] == "small"
    assert first["has_bmc"] is True and first["has_horizon"] is True
    assert first["has_perturb"] is True and first["has_regime"] is True
    assert first["policy_n_trials"] == 2
    # sorted by metric desc, run_id tiebreak
    assert frame.get_column("run_id").to_list() == ["a" * 64, "b" * 64]


def test_fleet_summary_flags_legacy_runs_without_scorecard() -> None:
    legacy = {
        "run_id": "c" * 64,
        "manifest": {"oof_device": "cpu", "config": {
            "data": {"feature_set": "all", "feature_subset": None},
            "model": {"preset": "deep"},
            "risk": {"neutralization_proportion": 0.5},
        }},
        "scorecard": None,
    }
    frame = fleet_summary([legacy], n_trials=1)
    row = frame.row(0, named=True)
    assert row["metric"] is None
    assert row["dsr_pass"] is False
    assert row["has_bmc"] is False
    assert row["preset"] == "deep" and row["neutralization_proportion"] == 0.5


def test_fleet_summary_validates_policy_arguments() -> None:
    with pytest.raises(ValueError, match="n_trials"):
        fleet_summary([], n_trials=0)
    with pytest.raises(ValueError, match="dsr_confidence"):
        fleet_summary([], n_trials=1, dsr_confidence=1.5)


def test_neutralized_ic_series_parity_with_profile() -> None:
    import numpy as np

    from nmr.analysis import neutralized_ic_profile, neutralized_ic_series

    rng = np.random.default_rng(11)
    rows = []
    for e in range(6):
        era = f"{e + 1:04d}"
        for i in range(20):
            f1, f2 = float(rng.normal()), float(rng.normal())
            y = 0.6 * f1 + 0.4 * f2 + float(rng.normal(scale=0.3))
            rows.append(
                {"era": era, "id": f"{era}_{i}", "f1": f1, "f2": f2,
                 "prediction": y, "target": y}
            )
    frame = pl.DataFrame(rows)
    chunks = frame.partition_by("era", maintain_order=True)
    profile = neutralized_ic_profile(
        chunks, ["prediction"], ["f1", "f2"], "target", proportions=[1.0]
    )
    series = neutralized_ic_series(
        chunks, ["prediction"], ["f1", "f2"], "target", proportion=1.0
    )
    row = profile.filter(pl.col("signal") == "prediction").row(0, named=True)
    assert series.height == row["n_eras"]
    assert series["ic"].mean() == pytest.approx(row["mean_ic"], abs=1e-12)
    # proportion=0.0 keeps the raw signal: IC vs itself is ~1
    raw = neutralized_ic_series(
        chunks, ["prediction"], ["f1", "f2"], "target", proportion=0.0
    )
    assert raw["ic"].mean() > 0.99


def _evidence_environment(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Synthetic v0test data dir + registry with two recorded campaign runs."""
    import numpy as np

    d = tmp_path / "data" / "v0test"
    d.mkdir(parents=True)
    (d / "features.json").write_text(
        json.dumps(
            {
                "feature_sets": {
                    "small": ["f1"],
                    "medium": ["f1", "f2"],
                    "all": ["f1", "f2"],
                },
                "targets": ["target"],
            }
        ),
        encoding="utf-8",
    )
    rng = np.random.default_rng(12)
    rows = []
    for e in range(12):
        era = f"{e + 1:04d}"
        for i in range(15):
            f1, f2 = float(rng.normal()), float(rng.normal())
            rows.append(
                {"era": era, "id": f"{era}_{i}", "f1": f1, "f2": f2,
                 "target": 0.7 * f1 + 0.3 * f2 + float(rng.normal(scale=0.5))}
            )
    pl.DataFrame(rows).write_parquet(d / "validation.parquet")

    reg = tmp_path / "registry"
    for run_id, pred_fn in (
        ("a" * 64, lambda r: 0.7 * r["f1"]),                    # v2: linear only
        ("b" * 64, lambda r: 0.7 * r["f1"] + 0.25 * r["f2"]),   # v3: + nonlinear-ish
    ):
        rd = reg / run_id
        rd.mkdir(parents=True)
        preds = pl.DataFrame(
            [
                {"era": r["era"], "id": r["id"],
                 "prediction": float(pred_fn(r))}
                for r in rows
            ]
        )
        preds.write_parquet(rd / "validation_preds.parquet")
        (rd / "run.json").write_text(
            json.dumps(
                {
                    "scorecard": {
                        "corr": 0.05,
                        "corr_ci_low": 0.01,
                        "corr_ci_high": 0.09,
                        "corr_sharpe_ac": 0.6,
                        "max_drawdown": -0.12,
                        "n_eras": 6,
                    },
                    "manifest": {
                        "config": {
                            "model": {"backend": "lightgbm", "device": "cpu"}
                        },
                        "feature_cols": ["f1"],
                    },
                }
            ),
            encoding="utf-8",
        )

    log = tmp_path / "campaign.json"
    log.write_text(
        json.dumps(
            {
                "campaign_id": "x" * 64,
                "name": "t",
                "configs": [],
                "runs": [
                    {
                        "config_path": "configs/campaigns/benchmark-rebuild-v1/lgbm_v2.yaml",
                        "run_id": "a" * 64,
                        "status": "recorded",
                    },
                    {
                        "config_path": "configs/campaigns/benchmark-rebuild-v1/lgbm_v3.yaml",
                        "run_id": "b" * 64,
                        "status": "recorded",
                    },
                    {
                        "config_path": "configs/campaigns/benchmark-rebuild-v1/lgbm_v4.yaml",
                        "run_id": None,
                        "status": "error",
                        "error": "boom",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return d.parent, reg, log


def test_campaign_evidence_assembles_variants_and_pairwise(
    tmp_path: Path,
) -> None:
    import numpy as np

    from nmr.config import DataConfig
    from nmr.meta import campaign_evidence

    data_root, reg, log = _evidence_environment(tmp_path)
    evidence = campaign_evidence(
        log, reg, data=DataConfig(version="v0test", data_dir=data_root),
        min_overlap_eras=2,
    )

    variants = evidence.variants
    assert variants.height == 3  # two recorded + one error row
    recorded = variants.filter(pl.col("status") == "recorded")
    assert recorded.height == 2
    v2 = recorded.filter(pl.col("variant") == "lgbm_v2").row(0, named=True)
    # headline metrics now come from the FULL-window per-era IC series
    # (numeric-ordered, own bootstrap CI); the scorecard's 86-era cells are
    # kept as explicit secondary columns
    assert v2["mean_ic"] > 0.5  # fixture pred = 0.7*f1 vs target incl. 0.7*f1
    assert v2["ic_ci_lo"] <= v2["mean_ic"] <= v2["ic_ci_hi"]
    assert v2["ic_sharpe"] is not None and v2["ic_sharpe"] > 0
    assert v2["max_drawdown"] is not None and v2["max_drawdown"] <= 0.0
    assert v2["n_eras"] == 12
    assert v2["scorecard_ic_86era"] == pytest.approx(0.05)
    assert v2["scorecard_sharpe_ac_86era"] == pytest.approx(0.6)
    assert v2["n_features"] == 1
    assert v2["backend"] == "lightgbm"
    assert v2["device"] == "cpu"
    assert v2["fne100"] is not None and np.isfinite(v2["fne100"])
    err = variants.filter(pl.col("variant") == "lgbm_v4").row(0, named=True)
    assert "boom" in err["error"]

    pairwise = evidence.pairwise
    assert pairwise.height == 1
    row = pairwise.row(0, named=True)
    assert row["pair"] == "lgbm_v2 vs lgbm_v3"
    assert row["backend"] == "lightgbm"
    assert row["n_eras"] == 12
    # v3 carries strictly more signal -> v2 - v3 < 0 with CI excluding zero
    assert row["mean_diff"] < 0.0
    assert row["ci_high"] < 0.0
    assert row["ci_low"] <= row["mean_diff"] <= row["ci_high"]

def _variant_row(label, sharpe, skew, kurt, n_eras, std=0.1, status="recorded"):
    return {
        "variant": label, "status": status,
        "ic_sharpe": sharpe, "ic_std": std, "ic_skew": skew,
        "ic_kurt": kurt, "n_eras": n_eras,
    }

def test_attach_campaign_dsr_computes_fleet_deflation() -> None:
    import numpy as np
    from nmr.inference import deflated_sharpe
    from nmr.meta import _attach_campaign_dsr

    rows = [
        _variant_row("a", 0.4, 0.0, 3.0, 600),
        _variant_row("b", 0.6, 0.1, 3.2, 649),
        _variant_row("c", 0.5, -0.1, 2.9, 620),
    ]
    out = {r["variant"]: r for r in _attach_campaign_dsr(rows)}
    var = np.var([0.4, 0.6, 0.5], ddof=1)
    expected = deflated_sharpe(
        0.4, n_trials=3, n_obs=600, skew=0.0, kurt=3.0, trials_sr_var=var
    )
    assert out["a"]["dsr_campaign_aware"] == pytest.approx(expected)
    assert out["a"]["dsr_pass_campaign"] is (out["a"]["dsr_campaign_aware"] >= 0.95)
    assert out["a"]["dsr_reason"] is None
    assert out["a"]["dsr_n_trials"] == 3
    assert out["a"]["dsr_trials_sr_var"] == pytest.approx(var)

def test_attach_campaign_dsr_zero_variance_guard() -> None:
    from nmr.meta import _attach_campaign_dsr

    rows = [
        _variant_row("a", 0.5, 0.0, 3.0, 600),
        _variant_row("b", 0.5, 0.0, 3.0, 600),
    ]
    for r in _attach_campaign_dsr(rows):
        assert r["dsr_campaign_aware"] is None
        assert r["dsr_pass_campaign"] is False
        assert r["dsr_reason"] == "zero_cross_trial_sharpe_variance"

def test_attach_campaign_dsr_degenerate_and_error_rows() -> None:
    from nmr.meta import _attach_campaign_dsr

    rows = [
        _variant_row("good1", 0.4, 0.0, 3.0, 600),
        _variant_row("good2", 0.6, 0.1, 3.2, 649),
        _variant_row("const", 0.0, 0.0, 3.0, 600, std=0.0),
        _variant_row("short", 0.5, 0.0, 3.0, 3),
        {"variant": "err", "status": "error", "error": "boom"},
    ]
    out = {r["variant"]: r for r in _attach_campaign_dsr(rows)}
    assert out["good1"]["dsr_campaign_aware"] is not None
    assert out["const"]["dsr_campaign_aware"] is None
    assert out["const"]["dsr_reason"] == "degenerate_series"
    assert out["short"]["dsr_reason"] == "degenerate_series"
    assert out["err"]["dsr_campaign_aware"] is None


def test_pair_backend_tolerates_v2_error_rows() -> None:
    from nmr.meta import _pair_backend

    rows = [
        {"variant": "lgbm_v2", "status": "error", "error": "empty subset"},
        {"variant": "lgbm_v3", "backend": "lightgbm", "status": "recorded"},
        {"variant": "xgb_v2", "status": "error", "error": "empty subset"},
        {"variant": "xgb_v3", "backend": "xgboost", "status": "recorded"},
    ]
    assert _pair_backend(rows, "lgbm") == "lightgbm"
    assert _pair_backend(rows, "xgb") == "xgboost"


def test_pair_backend_falls_back_to_prefix_without_recorded_cell() -> None:
    from nmr.meta import _pair_backend

    rows = [{"variant": "lgbm_v2", "status": "error", "error": "empty subset"}]
    assert _pair_backend(rows, "lgbm") == "lgbm"
