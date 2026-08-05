"""Contract tests for control-plane scripts (F-018)."""

from __future__ import annotations

import pandas as pd
import polars as pl

import benchmark_runner
import generate_dashboard
import train_first_model  # noqa: F401  (import-time smoke)


class _StubSuite:
    """Public-surface stub: only iter_baseline_predictions exists."""

    def __init__(self) -> None:
        self.frame = pl.DataFrame(
            {"era": ["1", "1"], "id": ["a", "b"], "prediction": [0.1, 0.2]}
        )

    def iter_baseline_predictions(self, *, include_classical, min_train_eras):
        yield ("constant-0.5", "null", self.frame, 77)
        if include_classical:
            yield ("linear", "classical", self.frame, 81)


def test_candidate_strategies_consumes_only_public_api() -> None:
    suite = _StubSuite()
    benchmarks = pl.DataFrame(
        {"era": ["1", "1"], "id": ["a", "b"], "bench_a": [0.3, 0.4]}
    )
    contexts = list(
        benchmark_runner._candidate_strategies(suite, benchmarks, seed=77, min_train_eras=2, fast_mode=False)
    )
    assert [ctx.model_id for ctx in contexts] == ["constant-0.5", "linear", "bench_a"]
    assert contexts[0].seed == 77
    assert contexts[1].seed == 81


def test_dashboard_escapes_html_interpolation() -> None:
    df = pd.DataFrame(
        [
            {
                "model_id": "<script>alert(1)</script>",
                "source": "trained",
                "run_name": '"><img src=x onerror=alert(2)>',
                "feature_set": "small",
                "backend": "lightgbm",
                "preset": "fast",
                "n_targets": 1,
                "targets": "target",
                "mean": 0.1,
                "std": 0.2,
                "sharpe": 0.5,
                "max_drawdown": 0.05,
                "rank": 1,
            }
        ]
    )
    html = generate_dashboard._build_html(
        df,
        benchmark_path=__import__("pathlib").Path("benchmark_scores.csv"),
        registry_dir=__import__("pathlib").Path("registry"),
        legacy=pd.DataFrame(),
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_dashboard_ranks_trained_and_benchmark_on_same_sharpe() -> None:
    trained = pd.DataFrame(
        [
            {
                "model_id": "trained_a", "source": "trained", "run_name": "t",
                "feature_set": "small", "backend": "lgbm", "preset": "fast",
                "n_targets": 1, "targets": "target", "mean": 0.1, "std": 0.1,
                "sharpe": 1.5, "max_drawdown": 0.1,
            }
        ]
    )
    benchmark = pd.DataFrame(
        [
            {
                "model_id": "bench_a", "source": "benchmark", "run_name": "b",
                "feature_set": "all", "backend": "benchmark", "preset": "benchmark",
                "n_targets": 1, "targets": "target", "mean": 0.05, "std": 0.1,
                "sharpe": 0.5, "max_drawdown": 0.2,
            }
        ]
    )
    ranked = generate_dashboard._rank_models(pd.concat([trained, benchmark], ignore_index=True))
    assert ranked.iloc[0]["model_id"] == "trained_a"
