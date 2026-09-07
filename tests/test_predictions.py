"""Prediction artifact contract: frame invariants, provenance, composition, foreign scoring."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nmr.ensemble import Ensembler
from nmr.payout import CLASSIC_LEGACY_V1
from nmr.predictions import (
    PREDICTION_STAGES,
    PredictionProvenance,
    PredictionSet,
    compose_predictions,
    evaluate_prediction_set,
    prediction_set_from_frame,
    read_prediction_set,
)
from nmr.risk import NeutralizationEngine
from nmr.scorecard import evaluate_model as direct_evaluate_model


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "era": ["2", "1", "1"],
            "id": ["b", "a", "c"],
            "prediction": [0.2, 0.1, 0.3],
        }
    )


def _prov(**overrides) -> PredictionProvenance:
    payload = dict(
        stage="raw",
        trained_targets=("target",),
        ensemble_target="target",
        scoring_target=None,
        training_horizon="20D",
        scoring_horizon=None,
        era_partition="oof",
        data_fingerprint="abc",
        feature_fingerprint="def",
        fit_role="foreign",
        device="cpu",
        source_run_id=None,
        selection_bias=False,
        split_estimand=False,
    )
    payload.update(overrides)
    return PredictionProvenance(**payload)


def test_prediction_stages_are_closed() -> None:
    assert PREDICTION_STAGES == (
        "raw",
        "blended",
        "neutralized",
        "validation",
        "submission",
    )


def test_from_frame_sorts_and_renames() -> None:
    raw = pl.DataFrame({"e": ["2", "1"], "i": ["b", "a"], "p": [0.2, 0.1]})
    ps = prediction_set_from_frame(raw, _prov(), era_col="e", id_col="i", pred_col="p")
    assert isinstance(ps, PredictionSet)
    assert ps.frame.columns == ["era", "id", "prediction"]
    assert ps.frame.get_column("era").to_list() == ["1", "2"]
    assert ps.frame.get_column("id").to_list() == ["a", "b"]


def test_duplicate_keys_raise() -> None:
    dup = pl.DataFrame({"era": ["1", "1"], "id": ["a", "a"], "prediction": [0.1, 0.2]})
    with pytest.raises(ValueError, match="unique"):
        prediction_set_from_frame(dup, _prov())


def test_nonfinite_predictions_raise() -> None:
    bad = pl.DataFrame({"era": ["1"], "id": ["a"], "prediction": [float("nan")]})
    with pytest.raises(ValueError, match="finite"):
        prediction_set_from_frame(bad, _prov())


def test_submission_stage_requires_open_unit_interval() -> None:
    frame = pl.DataFrame({"era": ["1"], "id": ["a"], "prediction": [0.0]})
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        prediction_set_from_frame(frame, _prov(stage="submission"))
    ok = pl.DataFrame({"era": ["1"], "id": ["a"], "prediction": [0.5]})
    ps = prediction_set_from_frame(ok, _prov(stage="submission"))
    assert ps.provenance.stage == "submission"


def test_unknown_stage_raises() -> None:
    with pytest.raises(ValueError, match="stage"):
        _prov(stage="scored")


def test_read_prediction_set_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "preds.parquet"
    _frame().write_parquet(path)
    ps = read_prediction_set(path, _prov(stage="validation"))
    assert ps.frame.height == 3
    assert ps.frame.columns == ["era", "id", "prediction"]
    assert ps.provenance.stage == "validation"


def _component_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "era": ["1", "1", "1", "2", "2", "2"],
            "id": ["a", "b", "c", "a", "b", "c"],
            "pred_t1": [0.1, 0.5, 0.9, 0.2, 0.4, 0.8],
            "pred_t2": [0.9, 0.5, 0.1, 0.8, 0.4, 0.2],
            "f1": [0.0, 1.0, 2.0, 0.5, 1.5, 2.5],
        }
    )


def test_compose_matches_ensembler_blend() -> None:
    frame = _component_frame()
    weights = (0.75, 0.25)
    expected = Ensembler().blend(
        frame, pred_cols=["pred_t1", "pred_t2"], weights=weights, era_col="era"
    )
    composed = compose_predictions(
        frame,
        pred_cols=("pred_t1", "pred_t2"),
        weights=weights,
        provenance=_prov(stage="raw"),
    )
    assert composed.provenance.stage == "blended"
    assert composed.frame.get_column("prediction").to_list() == pytest.approx(
        expected.get_column("prediction").to_list()
    )


def test_compose_as_submission_is_open_unit_interval() -> None:
    frame = pl.DataFrame(
        {
            "era": ["1", "1"],
            "id": ["a", "b"],
            "pred_t1": [0.1, 0.9],
        }
    )
    composed = compose_predictions(
        frame,
        pred_cols=("pred_t1",),
        as_submission=True,
        provenance=_prov(stage="raw"),
    )
    preds = composed.frame.get_column("prediction").to_list()
    assert composed.provenance.stage == "submission"
    assert all(0.0 < x < 1.0 for x in preds)


def test_compose_empty_pred_cols_raises() -> None:
    with pytest.raises(ValueError, match="pred_cols"):
        compose_predictions(_frame(), pred_cols=(), provenance=_prov())


def test_compose_neutralize_matches_engine(tmp_path: Path) -> None:
    frame = _component_frame()
    weights = (0.5, 0.5)
    blended = Ensembler().blend(
        frame, pred_cols=["pred_t1", "pred_t2"], weights=weights, era_col="era"
    )
    engine = NeutralizationEngine(cache_dir=tmp_path / "ncache", max_cache_bytes=0)
    expected = engine.neutralize(
        blended,
        pred_col="prediction",
        feature_cols=["f1"],
        era_col="era",
        proportion=1.0,
    )
    composed = compose_predictions(
        frame,
        pred_cols=("pred_t1", "pred_t2"),
        weights=weights,
        feature_cols=("f1",),
        neutralization_proportion=1.0,
        neutralization_cache_dir=tmp_path / "ncache2",
        provenance=_prov(stage="raw"),
    )
    assert composed.provenance.stage == "neutralized"
    assert composed.frame.get_column("prediction").to_list() == pytest.approx(
        expected.get_column("prediction").to_list()
    )


def test_compose_zero_proportion_stays_blended(tmp_path: Path) -> None:
    frame = _component_frame()
    composed = compose_predictions(
        frame,
        pred_cols=("pred_t1", "pred_t2"),
        feature_cols=("f1",),
        neutralization_proportion=0.0,
        neutralization_cache_dir=tmp_path / "ncache",
        provenance=_prov(stage="raw"),
    )
    assert composed.provenance.stage == "blended"


def _tiny_like_scorecard() -> (
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
):
    rows: list[dict[str, float | str]] = []
    bench: list[dict[str, float | str]] = []
    for i in range(1, 21):
        era = f"{i:04d}"
        for j in range(3):
            pred = (0.2 * i) + (0.03 * j)
            meta = (0.15 * i) - (0.02 * j)
            rows.append(
                {
                    "era": era,
                    "id": f"{era}_{j:03d}",
                    "prediction": float(pred),
                    "target": float((i + j) % 5) / 4.0,
                    "f1": float((i + j) % 5),
                    "numerai_meta_model": float(meta),
                }
            )
            bench.append(
                {
                    "era": era,
                    "id": f"{era}_{j:03d}",
                    "v52_lgbm_cyrusd20": float(meta),
                }
            )
    full = pl.DataFrame(rows)
    predictions = full.select(["era", "id", "prediction"])
    meta_model = full.select(["era", "id", "numerai_meta_model"])
    targets = full.select(["era", "id", "target"])
    features = full.select(["era", "id", "f1"])
    benchmarks = pl.DataFrame(bench)
    return predictions, meta_model, benchmarks, features, targets


def test_evaluate_prediction_set_scores_foreign_parquet(tmp_path: Path) -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    path = tmp_path / "foreign.parquet"
    predictions.write_parquet(path)
    ps = read_prediction_set(path, _prov(stage="validation", fit_role="foreign"))
    kwargs = dict(
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=11,
        payout_policy=CLASSIC_LEGACY_V1,
        n_boot=5,
        min_overlap_eras=20,
    )
    scored = evaluate_prediction_set(ps, **kwargs)
    expected = direct_evaluate_model(ps.frame, **kwargs)
    assert scored.corr.value == expected.corr.value
    source = Path("nmr/predictions.py").read_text(encoding="utf-8")
    assert "from nmr.models" not in source
    assert "import nmr.models" not in source


def test_evaluate_prediction_set_refuses_raw_stage() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    ps = prediction_set_from_frame(predictions, _prov(stage="raw"))
    with pytest.raises(ValueError, match="stage"):
        evaluate_prediction_set(
            ps,
            meta_model=meta_model,
            benchmarks=benchmarks,
            features=features,
            targets=targets,
            n_trials=1,
            seed=0,
            payout_policy=CLASSIC_LEGACY_V1,
            n_boot=5,
            min_overlap_eras=20,
        )


def test_evaluate_prediction_set_allows_research_stage_with_flag() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    ps = prediction_set_from_frame(predictions, _prov(stage="neutralized"))
    scored = evaluate_prediction_set(
        ps,
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=11,
        payout_policy=CLASSIC_LEGACY_V1,
        n_boot=5,
        min_overlap_eras=20,
        allow_research_stage=True,
    )
    assert scored.corr.n_eras == 20
