"""Tests for the public baseline-prediction generator (F-001/F-016)."""

from __future__ import annotations

import polars as pl
import pytest

from nmr.benchmark import NULL_BASELINES, BenchmarkSuite


def _suite(seed: int = 7) -> BenchmarkSuite:
    rows = []
    for era in range(1, 13):
        for idx in range(4):
            rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f1": float(idx) / 10.0,
                    "f2": float((idx % 2)) / 10.0,
                    "target": float(era) / 100.0,
                }
            )
    frame = pl.DataFrame(rows)
    return BenchmarkSuite(
        meta_model=frame.select(["era", "id"]).with_columns(
            pl.lit(0.1).alias("numerai_meta_model")
        ),
        benchmarks=pl.DataFrame(
            {"era": [], "id": [], "bench": []},
            schema={"era": pl.String, "id": pl.String, "bench": pl.Float64},
        ),
        features=frame.select(["era", "id", "f1", "f2"]),
        targets=frame.select(["era", "id", "target"]),
        n_trials=1,
        seed=seed,
        horizon="20D",
        n_boot=1,
        min_overlap_eras=2,
    )


def test_generator_yields_null_trivial_ordering_and_seed_convention() -> None:
    suite = _suite(seed=7)
    items = list(suite.iter_baseline_predictions(include_classical=False))
    ids = [model_id for model_id, _, _, _ in items]
    assert ids == [*NULL_BASELINES, "trivial"]
    assert [seed for _, _, _, seed in items] == [7, 8, 9, 10]


def test_generator_includes_classical_with_min_train_eras() -> None:
    suite = _suite(seed=7)
    items = list(suite.iter_baseline_predictions(include_classical=True, min_train_eras=2))
    ids = [model_id for model_id, _, _, _ in items]
    assert ids == [*NULL_BASELINES, "trivial", "linear", "tree"]
    assert [seed for _, _, _, seed in items] == [7, 8, 9, 10, 11, 12]
    for _, _, raw_preds, _ in items:
        assert {"era", "id", "prediction"} <= set(raw_preds.columns)


def test_walk_forward_uses_lightgbm_tree_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    suite = _suite(seed=7)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("simulated missing lightgbm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        suite._build_classical_model("tree")
