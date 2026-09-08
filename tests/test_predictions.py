"""Prediction artifact contract: frame invariants, provenance, composition, foreign scoring."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nmr.ensemble import Ensembler
from nmr.payout import CLASSIC_LEGACY_V1
from nmr.predictions import (
    PREDICTION_STAGES,
    CapitalContext,
    PredictionProvenance,
    PredictionSet,
    compose_predictions,
    evaluate_prediction_set,
    prediction_set_from_frame,
    read_prediction_set,
    validation_key_fingerprint,
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


def _capital_prov(**overrides) -> PredictionProvenance:
    eras = tuple(f"{i:04d}" for i in range(1, 21))
    payload = dict(
        stage="validation",
        trained_targets=("target",),
        ensemble_target="target",
        scoring_target="target",
        training_horizon="20D",
        scoring_horizon="20D",
        payout_policy_id="classic_legacy_075_225_clip005_v1",
        era_partition="validation",
        data_fingerprint="abc",
        feature_fingerprint="def",
        fit_role="foreign",
        device="cpu",
        source_run_id=None,
        selection_bias=False,
        split_estimand=False,
        validation_window=eras,
    )
    payload.update(overrides)
    return PredictionProvenance(**payload)


def _capital_ctx(**overrides) -> CapitalContext:
    eras = tuple(f"{i:04d}" for i in range(1, 21))
    keys = [(f"{i:04d}", f"{i:04d}_{j:03d}") for i in range(1, 21) for j in range(3)]
    key_frame = pl.DataFrame({"era": [k[0] for k in keys], "id": [k[1] for k in keys]})
    payload = dict(
        validation_window=eras,
        scoring_target="target",
        scoring_horizon="20D",
        payout_policy_id="classic_legacy_075_225_clip005_v1",
        data_fingerprint="abc",
        feature_fingerprint="def",
        validation_key_fingerprint=validation_key_fingerprint(key_frame),
        validation_row_count=len(keys),
        trained_targets=("target",),
        ensemble_target="target",
        training_horizon="20D",
        split_estimand=False,
    )
    payload.update(overrides)
    return CapitalContext(**payload)


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
        provenance=_prov(stage="raw", era_partition="live"),
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
    ps = read_prediction_set(path, _capital_prov(fit_role="foreign"))
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
    scored = evaluate_prediction_set(ps, capital_context=_capital_ctx(), **kwargs)
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
            capital_context=_capital_ctx(),
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
    from nmr.predictions import ResearchEvaluation

    assert isinstance(scored, ResearchEvaluation)
    assert scored.is_capital is False
    assert scored.scorecard.corr.n_eras == 20


def test_oof_as_submission_cannot_enter_capital_scorecard() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    ranked = predictions.with_columns(
        (pl.col("prediction").rank() / (pl.len() + 1)).alias("prediction")
    )
    relabelled = prediction_set_from_frame(
        ranked,
        _prov(stage="submission", era_partition="oof"),
    )
    assert relabelled.provenance.stage == "submission"
    assert relabelled.provenance.era_partition == "oof"
    with pytest.raises(ValueError, match="capital"):
        evaluate_prediction_set(
            relabelled,
            meta_model=meta_model,
            benchmarks=benchmarks,
            features=features,
            targets=targets,
            n_trials=1,
            seed=11,
            payout_policy=CLASSIC_LEGACY_V1,
            n_boot=5,
            capital_context=_capital_ctx(),
            min_overlap_eras=20,
        )


def test_as_submission_rejects_oof_partition() -> None:
    frame = pl.DataFrame({"era": ["1", "1"], "id": ["a", "b"], "pred_t1": [0.1, 0.9]})
    with pytest.raises(ValueError, match="era_partition"):
        compose_predictions(
            frame,
            pred_cols=("pred_t1",),
            as_submission=True,
            provenance=_prov(stage="raw", era_partition="oof"),
        )


def test_submission_stage_is_not_capital_eligible() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    ps = prediction_set_from_frame(
        predictions.with_columns(
            (pl.col("prediction") % 1 * 0.5 + 0.25).alias("prediction")
        ),
        _capital_prov(stage="submission"),
    )
    with pytest.raises(ValueError, match="capital"):
        evaluate_prediction_set(
            ps,
            meta_model=meta_model,
            benchmarks=benchmarks,
            features=features,
            targets=targets,
            n_trials=1,
            seed=11,
            payout_policy=CLASSIC_LEGACY_V1,
            n_boot=5,
            capital_context=_capital_ctx(),
            min_overlap_eras=20,
        )


def test_capital_eval_rejects_selection_bias() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    ps = prediction_set_from_frame(predictions, _capital_prov(selection_bias=True))
    with pytest.raises(ValueError, match="selection_bias"):
        evaluate_prediction_set(
            ps,
            meta_model=meta_model,
            benchmarks=benchmarks,
            features=features,
            targets=targets,
            n_trials=1,
            seed=11,
            payout_policy=CLASSIC_LEGACY_V1,
            n_boot=5,
            capital_context=_capital_ctx(),
            min_overlap_eras=20,
        )


def test_capital_eval_requires_matching_target_horizon() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    ps = prediction_set_from_frame(
        predictions,
        _capital_prov(scoring_target="target_ender_60", scoring_horizon="60D"),
    )
    with pytest.raises(ValueError, match="scoring"):
        evaluate_prediction_set(
            ps,
            meta_model=meta_model,
            benchmarks=benchmarks,
            features=features,
            targets=targets,
            n_trials=1,
            seed=11,
            payout_policy=CLASSIC_LEGACY_V1,
            horizon="20D",
            main_target="target",
            capital_context=_capital_ctx(),
            n_boot=5,
            min_overlap_eras=20,
        )


def test_unknown_era_partition_raises() -> None:
    with pytest.raises(ValueError, match="era_partition"):
        _prov(era_partition="train")


def test_unknown_fit_role_raises() -> None:
    with pytest.raises(ValueError, match="fit_role"):
        _prov(fit_role="hpo")


def test_compose_rejects_nonfinite_and_negative_proportion() -> None:
    frame = _component_frame()
    with pytest.raises(ValueError, match="proportion"):
        compose_predictions(
            frame,
            pred_cols=("pred_t1",),
            neutralization_proportion=-1.0,
            provenance=_prov(),
        )
    with pytest.raises(ValueError, match="proportion"):
        compose_predictions(
            frame,
            pred_cols=("pred_t1",),
            neutralization_proportion=float("nan"),
            provenance=_prov(),
        )


def test_compose_joins_features_for_foreign_parquet(tmp_path: Path) -> None:
    frame = _component_frame()
    preds = frame.select(["era", "id", "pred_t1", "pred_t2"])
    feats = frame.select(["era", "id", "f1"])
    weights = (0.5, 0.5)
    expected = compose_predictions(
        frame,
        pred_cols=("pred_t1", "pred_t2"),
        weights=weights,
        feature_cols=("f1",),
        neutralization_proportion=1.0,
        neutralization_cache_dir=tmp_path / "n1",
        provenance=_prov(),
    )
    joined = compose_predictions(
        preds,
        pred_cols=("pred_t1", "pred_t2"),
        weights=weights,
        feature_cols=("f1",),
        features=feats,
        neutralization_proportion=1.0,
        neutralization_cache_dir=tmp_path / "n2",
        provenance=_prov(),
    )
    assert joined.provenance.stage == "neutralized"
    assert joined.frame.get_column("prediction").to_list() == pytest.approx(
        expected.frame.get_column("prediction").to_list()
    )


def test_capital_eval_requires_authoritative_context() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    ps = prediction_set_from_frame(predictions, _capital_prov())
    with pytest.raises(ValueError, match="CapitalContext"):
        evaluate_prediction_set(
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
        )


def test_capital_eval_rejects_missing_scoring_identity() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    ps = prediction_set_from_frame(
        predictions,
        _capital_prov(scoring_target=None, scoring_horizon=None),
    )
    with pytest.raises(ValueError, match="scoring"):
        evaluate_prediction_set(
            ps,
            meta_model=meta_model,
            benchmarks=benchmarks,
            features=features,
            targets=targets,
            n_trials=1,
            seed=11,
            payout_policy=CLASSIC_LEGACY_V1,
            capital_context=_capital_ctx(),
            n_boot=5,
            min_overlap_eras=20,
        )


def test_capital_eval_rejects_missing_fingerprints() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    ps = prediction_set_from_frame(
        predictions,
        _capital_prov(data_fingerprint=None, feature_fingerprint=None),
    )
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate_prediction_set(
            ps,
            meta_model=meta_model,
            benchmarks=benchmarks,
            features=features,
            targets=targets,
            n_trials=1,
            seed=11,
            capital_context=_capital_ctx(),
            payout_policy=CLASSIC_LEGACY_V1,
            n_boot=5,
            min_overlap_eras=20,
        )


@pytest.mark.parametrize(
    "frame_name", ["meta_model", "features", "targets", "benchmarks"]
)
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_capital_eval_rejects_incomplete_auxiliary_key_universe(
    frame_name: str, mutation: str
) -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    frames = {
        "meta_model": meta_model,
        "features": features,
        "targets": targets,
        "benchmarks": benchmarks,
    }
    source = frames[frame_name]
    if mutation == "missing":
        frames[frame_name] = source.slice(0, source.height - 1)
    else:
        frames[frame_name] = pl.concat([source, source.head(1)])
    ps = prediction_set_from_frame(predictions, _capital_prov())
    with pytest.raises(ValueError, match="key"):
        evaluate_prediction_set(
            ps,
            meta_model=frames["meta_model"],
            benchmarks=frames["benchmarks"],
            features=frames["features"],
            targets=frames["targets"],
            n_trials=1,
            seed=11,
            capital_context=_capital_ctx(),
            payout_policy=CLASSIC_LEGACY_V1,
            n_boot=5,
            min_overlap_eras=20,
        )


def test_capital_eval_rejects_duplicate_benchmark_key() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    duplicate_benchmarks = pl.concat([benchmarks, benchmarks.head(1)])
    ps = prediction_set_from_frame(predictions, _capital_prov())
    with pytest.raises(ValueError, match="key"):
        evaluate_prediction_set(
            ps,
            meta_model=meta_model,
            benchmarks=duplicate_benchmarks,
            features=features,
            targets=targets,
            n_trials=1,
            seed=11,
            capital_context=_capital_ctx(),
            payout_policy=CLASSIC_LEGACY_V1,
            n_boot=5,
            min_overlap_eras=20,
        )


def _capital_kwargs(meta_model, benchmarks, features, targets) -> dict:
    return dict(
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


def test_capital_eval_rejects_sparse_prediction_keys() -> None:
    """One id per era against a 3-id universe: exact key coverage refuses."""
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    sparse = predictions.group_by("era").head(1)
    ps = prediction_set_from_frame(sparse, _capital_prov())
    with pytest.raises(ValueError, match="key"):
        evaluate_prediction_set(
            ps,
            **_capital_kwargs(meta_model, benchmarks, features, targets),
            capital_context=_capital_ctx(),
        )


def test_capital_eval_rejects_extra_prediction_keys() -> None:
    """One extra (era, id) row refuses, even with the full window present."""
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    extra = pl.DataFrame({"era": ["0001"], "id": ["0001_zzz"], "prediction": [0.1]})
    ps = prediction_set_from_frame(pl.concat([predictions, extra]), _capital_prov())
    with pytest.raises(ValueError, match="key"):
        evaluate_prediction_set(
            ps,
            **_capital_kwargs(meta_model, benchmarks, features, targets),
            capital_context=_capital_ctx(),
        )


def test_capital_eval_rejects_forged_context_keys() -> None:
    """A context whose key universe does not describe the frame is refused."""
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    ps = prediction_set_from_frame(predictions, _capital_prov())
    with pytest.raises(ValueError, match="key"):
        evaluate_prediction_set(
            ps,
            **_capital_kwargs(meta_model, benchmarks, features, targets),
            capital_context=_capital_ctx(
                validation_key_fingerprint="0" * 64, validation_row_count=1
            ),
        )


def test_capital_eval_requires_payout_policy_match() -> None:
    """The evaluation's payout policy must equal the context's policy id."""
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    ps = prediction_set_from_frame(predictions, _capital_prov())
    with pytest.raises(ValueError, match="payout"):
        evaluate_prediction_set(
            ps,
            **_capital_kwargs(meta_model, benchmarks, features, targets),
            capital_context=_capital_ctx(
                payout_policy_id="classic_atomic_ender60_r1343_v1"
            ),
        )


def test_capital_eval_rejects_provenance_policy_mismatch() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    ps = prediction_set_from_frame(
        predictions,
        _capital_prov(payout_policy_id="classic_atomic_ender60_r1343_v1"),
    )
    with pytest.raises(ValueError, match="payout"):
        evaluate_prediction_set(
            ps,
            **_capital_kwargs(meta_model, benchmarks, features, targets),
            capital_context=_capital_ctx(),
        )


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"trained_targets": ("other",)}, "trained_targets"),
        ({"ensemble_target": "other"}, "ensemble_target"),
        ({"training_horizon": "60D"}, "training_horizon"),
        ({"split_estimand": True}, "split_estimand"),
    ],
)
def test_capital_eval_rejects_contradictory_training_estimand(
    override: dict, expected: str
) -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_like_scorecard()
    ps = prediction_set_from_frame(predictions, _capital_prov(**override))
    with pytest.raises(ValueError, match=expected):
        evaluate_prediction_set(
            ps,
            **_capital_kwargs(meta_model, benchmarks, features, targets),
            capital_context=_capital_ctx(),
        )


def test_validation_key_fingerprint_is_canonical() -> None:
    a = pl.DataFrame(
        {"era": ["1", "2", "1"], "id": ["a", "b", "c"], "prediction": [0.1, 0.2, 0.3]}
    )
    reordered = a.sort("id")
    extra_cols = a.with_columns(pl.lit(1).alias("x"))
    assert validation_key_fingerprint(a) == validation_key_fingerprint(reordered)
    assert validation_key_fingerprint(a) == validation_key_fingerprint(extra_cols)
    subset = pl.DataFrame({"era": ["1", "2"], "id": ["a", "b"]})
    assert validation_key_fingerprint(a) != validation_key_fingerprint(subset)


def test_compose_rejects_partial_feature_join(tmp_path: Path) -> None:
    frame = _component_frame()
    preds = frame.select(["era", "id", "pred_t1", "pred_t2"])
    feats = frame.head(1).select(["era", "id", "f1"])
    with pytest.raises(ValueError, match="join"):
        compose_predictions(
            preds,
            pred_cols=("pred_t1", "pred_t2"),
            feature_cols=("f1",),
            features=feats,
            neutralization_proportion=1.0,
            neutralization_cache_dir=tmp_path / "n-partial",
            provenance=_prov(),
        )


def test_prediction_column_is_float64() -> None:
    raw = pl.DataFrame({"era": ["1"], "id": ["a"], "prediction": ["0.5"]})
    ps = prediction_set_from_frame(raw, _prov())
    assert ps.frame.get_column("prediction").dtype == pl.Float64
    assert ps.frame.get_column("prediction").to_list() == [0.5]
