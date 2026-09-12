"""Gate mechanics for the 5-tier hierarchy (synthetic scorecards)."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nmr.benchmark import (
    NULL_FLOOR_KINDS,
    NULL_KINDS,
    BenchmarkHierarchyResult,
    NullFloorCalibration,
    Tier4GateConfig,
    assert_hierarchy_monotone,
    assert_tier0_null_floor,
    assert_tier4_gate,
    gate_report_frame,
    load_benchmark_file,
    load_null_floor_calibration,
    score_benchmark_column,
    tier4_gate_verdict,
    tier_max_corrs,
    verify_null_floor_window,
)
from nmr.payout import CLASSIC_LEGACY_V1
from nmr.scorecard import MetricScorecard, evaluate_model

GATE = Tier4GateConfig(
    payout_policy_id="classic_legacy_075_225_clip005_v1",
    scoring_target="target",
    scoring_horizon="20D",
    corr_min=0.0286,
    corr_sharpe_ac_min=1.50,
    fnc_min=0.020,
    gain_to_pain_min=1.50,
)


def _synthetic_inputs(n_eras: int = 60, rows_per_era: int = 16, seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = []
    for era_num in range(1, n_eras + 1):
        era = f"{era_num:04d}"
        for idx in range(rows_per_era):
            f1 = float(rng.normal())
            latent = 0.8 * f1 + float(rng.normal(0.0, 0.7))
            target = float(np.clip(0.5 + 0.2 * latent, 0.0, 1.0))
            rows.append(
                {
                    "era": era,
                    "id": f"{era}_{idx}",
                    "prediction": float(rng.random()),
                    "numerai_meta_model": float(0.55 * target + 0.45 * rng.random()),
                    "target": target,
                    "f1": f1,
                    "bench": float(0.6 * target + 0.4 * rng.random()),
                }
            )
    full = pl.DataFrame(rows)
    return (
        full.select(["era", "id", "prediction"]),
        full.select(["era", "id", "numerai_meta_model"]),
        full.select(["era", "id", "bench"]),
        full.select(["era", "id", "f1"]),
        full.select(["era", "id", "target"]),
    )


def _make_scorecard(**overrides: float) -> MetricScorecard:
    predictions, meta_model, benchmarks, features, targets = _synthetic_inputs()
    scorecard = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=77,
        payout_policy=CLASSIC_LEGACY_V1,
        benchmark_col="bench",
        n_boot=50,
        min_overlap_eras=20,
        model_id="probe",
    )
    return dataclasses.replace(scorecard, **overrides)


def _null_scorecards() -> dict[str, MetricScorecard]:
    out = {}
    for kind in NULL_KINDS:
        score = _make_scorecard(model_id=kind)
        # Synthetic degeneracy: the fixture's noise corr (|corr| ~ 0.013 on
        # 60x16 rows) is not at the null floor, and the 0.005 audit
        # tolerance is calibrated for real-data null baselines.
        # Floor-normalize corr; corr_sharpe_ac (~ -0.044) stays well inside
        # any sane calibrated envelope.
        out[kind] = dataclasses.replace(
            score,
            corr=dataclasses.replace(score.corr, value=0.0),
        )
    return out


def _null_scorecards_unzeroed() -> dict[str, MetricScorecard]:
    """Raw synthetic null scorecards (no floor normalization).

    The fixture's noise corr (~ 0.013 on 60x16 rows) is realistic for
    small-sample noise but exceeds the strict 0.005 audit default, so
    callers must pass an explicit corr tolerance.
    """
    return {kind: _make_scorecard(model_id=kind) for kind in NULL_KINDS}


_FIXTURE_ERAS = tuple(f"{era:04d}" for era in range(1, 61))
_FIXTURE_IDENTITY = {
    "payout_policy_id": CLASSIC_LEGACY_V1.policy_id,
    "scoring_target": CLASSIC_LEGACY_V1.target or "target",
    "scoring_horizon": CLASSIC_LEGACY_V1.scoring_horizon or "20D",
    "scoring_backend": "custom",
}


def _canonical_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _calibration_payload(
    cards: dict[str, MetricScorecard],
    *,
    threshold: float = 0.45,
    quantile: float = 0.99,
    seed_count: int = 1000,
) -> dict[str, object]:
    """Digest-correct calibration payload anchored to `cards` seed-42 values."""
    expected = {
        kind: {
            "corr": float(cards[kind].corr.value),
            "corr_sharpe_ac": float(cards[kind].corr_sharpe_ac.value),
        }
        for kind in NULL_FLOOR_KINDS
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "method": "family_wise_quantile",
        "data_version": "synthetic-fixture",
        "window_key_fingerprint": "f" * 64,
        "validation_eras": list(_FIXTURE_ERAS),
        "validation_rows": 60 * 16,
        "scoring_identity": dict(_FIXTURE_IDENTITY),
        "null_kinds": list(NULL_FLOOR_KINDS),
        "seed_set": {"seeds": [42, 43], "count": seed_count, "binding_seeds": [42]},
        "selected_quantile": quantile,
        "selected_threshold": threshold,
        "seed42_expected": expected,
        "seed42_family_max": max(
            abs(row["corr_sharpe_ac"]) for row in expected.values()
        ),
        "study_code_fingerprint": "synthetic",
        "generated_at": "2026-09-12T00:00:00Z",
    }
    payload["digest"] = _canonical_digest(payload)
    return payload


def _calibration(
    cards: dict[str, MetricScorecard] | None = None,
    *,
    threshold: float = 0.45,
    quantile: float = 0.99,
    seed_count: int = 1000,
) -> NullFloorCalibration:
    """In-memory calibration matching the fixture's seed-42 values."""
    cards = _null_scorecards() if cards is None else cards
    payload = _calibration_payload(
        cards, threshold=threshold, quantile=quantile, seed_count=seed_count
    )
    return NullFloorCalibration(
        schema_version=1,
        method=str(payload["method"]),
        data_version=str(payload["data_version"]),
        window_key_fingerprint=str(payload["window_key_fingerprint"]),
        validation_eras=tuple(payload["validation_eras"]),
        validation_rows=int(payload["validation_rows"]),
        scoring_identity=dict(_FIXTURE_IDENTITY),
        null_kinds=tuple(payload["null_kinds"]),
        seed_count=seed_count,
        selected_quantile=quantile,
        selected_threshold=threshold,
        seed42_expected=dict(payload["seed42_expected"]),
        seed42_family_max=float(payload["seed42_family_max"]),
        study_code_fingerprint=str(payload["study_code_fingerprint"]),
        digest=str(payload["digest"]),
    )


def _write_calibration(
    path: Path,
    cards: dict[str, MetricScorecard] | None = None,
    *,
    threshold: float = 0.45,
    mutate=None,
) -> Path:
    """Digest-correct calibration file; `mutate(payload)` runs pre-digest."""
    cards = _null_scorecards() if cards is None else cards
    payload = _calibration_payload(cards, threshold=threshold)
    if mutate is not None:
        mutate(payload)
        body = {k: v for k, v in payload.items() if k != "digest"}
        payload["digest"] = _canonical_digest(body)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_score_benchmark_column_wraps_predictions() -> None:
    _, _, benchmarks, _, _ = _synthetic_inputs()
    out = score_benchmark_column(benchmarks, column="bench")
    assert out.columns == ["era", "id", "prediction"]
    assert out.height == benchmarks.height


def test_score_benchmark_column_unknown_column_raises() -> None:
    _, _, benchmarks, _, _ = _synthetic_inputs()
    with pytest.raises(ValueError, match="nope"):
        score_benchmark_column(benchmarks, column="nope")


def test_tier0_null_floor_passes_and_reports_calibration_identity() -> None:
    cards = _null_scorecards()
    calibration = _calibration(cards)
    summary = assert_tier0_null_floor(cards, calibration=calibration)
    assert summary.method == "family_wise_quantile"
    assert summary.quantile == 0.99
    assert summary.seed_count == 1000
    assert summary.threshold == 0.45
    assert summary.reference_fingerprint == calibration.digest
    assert set(summary.observed) == set(NULL_FLOOR_KINDS)
    expected_family_max = max(
        abs(float(cards[kind].corr_sharpe_ac.value)) for kind in NULL_FLOOR_KINDS
    )
    assert summary.family_max_abs_ac_sharpe == pytest.approx(expected_family_max)


def test_tier0_null_floor_honors_calibrated_threshold() -> None:
    cards = _null_scorecards()
    cards["null_constant_05"] = dataclasses.replace(
        cards["null_constant_05"],
        corr_sharpe_ac=dataclasses.replace(
            cards["null_constant_05"].corr_sharpe_ac, value=0.19
        ),
    )
    inside = _calibration(cards, threshold=0.20)
    summary = assert_tier0_null_floor(cards, calibration=inside)
    assert summary.family_max_abs_ac_sharpe == pytest.approx(0.19)

    cards_high = {
        **cards,
        "null_constant_05": dataclasses.replace(
            cards["null_constant_05"],
            corr_sharpe_ac=dataclasses.replace(
                cards["null_constant_05"].corr_sharpe_ac, value=0.21
            ),
        ),
    }
    high_calibration = _calibration(cards_high, threshold=0.20)
    with pytest.raises(ValueError, match="calibrated family-wise"):
        assert_tier0_null_floor(cards_high, calibration=high_calibration)


def test_tier0_null_floor_rejects_high_corr() -> None:
    cards = _null_scorecards()
    cards["null_constant_05"] = dataclasses.replace(
        cards["null_constant_05"],
        corr=dataclasses.replace(cards["null_constant_05"].corr, value=0.05),
    )
    # Anchor matches the mutated card so the CORR floor (not the
    # determinism anchor) is the check under test.
    calibration = _calibration(cards)
    with pytest.raises(ValueError, match="null_constant_05"):
        assert_tier0_null_floor(cards, calibration=calibration)


def test_tier0_null_floor_rejects_determinism_anchor_drift() -> None:
    calibration = _calibration()  # anchored to the stock zeroed cards
    cards = _null_scorecards()
    cards["null_gaussian_rand"] = dataclasses.replace(
        cards["null_gaussian_rand"],
        corr_sharpe_ac=dataclasses.replace(
            cards["null_gaussian_rand"].corr_sharpe_ac, value=0.1
        ),
    )
    with pytest.raises(ValueError, match="determinism anchor"):
        assert_tier0_null_floor(cards, calibration=calibration)


def test_tier0_null_floor_requires_three_structural_kinds() -> None:
    cards = _null_scorecards()
    del cards["null_gaussian_rand"]
    with pytest.raises(ValueError, match="null_gaussian_rand"):
        assert_tier0_null_floor(cards, calibration=_calibration())


def test_tier0_null_floor_refuses_nonstructural_calibration_kinds() -> None:
    cards = _null_scorecards()
    calibration = dataclasses.replace(
        _calibration(cards),
        null_kinds=(*NULL_FLOOR_KINDS, "null_feature_mean"),
    )
    with pytest.raises(ValueError, match="structural"):
        assert_tier0_null_floor(cards, calibration=calibration)


def test_tier0_null_floor_ignores_null_feature_mean() -> None:
    # null_feature_mean is not structural noise (v5.3 corr 0.00294,
    # sharpe 0.257): its absence must not raise.
    cards = _null_scorecards()
    del cards["null_feature_mean"]
    assert_tier0_null_floor(cards, calibration=_calibration())


def test_tier0_null_floor_signature_is_pinned() -> None:
    params = inspect.signature(assert_tier0_null_floor).parameters
    assert "calibration" in params
    assert params["calibration"].default is inspect.Parameter.empty
    assert params["calibration"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["corr_tol"].default == 0.005
    assert "sharpe_tol" not in params
    assert "dsr_tol" not in params


def test_hierarchy_monotone_defaults_are_pinned() -> None:
    params = inspect.signature(assert_hierarchy_monotone).parameters
    assert params["metric"].default == "corr"
    assert params["atol"].default == 1e-5


def test_tier0_null_floor_passes_unzeroed_cards_at_explicit_tolerances() -> None:
    # Realistic synthetic values: corr ~ -0.0126, corr_sharpe_ac ~ -0.044
    # (fixed seed). The corr tolerance is explicit because small-sample
    # noise exceeds the 0.005 real-data audit floor; the AC envelope comes
    # from a calibration anchored to these exact values.
    cards = _null_scorecards_unzeroed()
    summary = assert_tier0_null_floor(
        cards, calibration=_calibration(cards), corr_tol=0.02
    )
    assert summary.corr_tol == 0.02


def test_tier0_null_floor_rejects_ac_sharpe_beyond_calibrated_envelope() -> None:
    cards = _null_scorecards()
    cards["null_constant_05"] = dataclasses.replace(
        cards["null_constant_05"],
        corr_sharpe_ac=dataclasses.replace(
            cards["null_constant_05"].corr_sharpe_ac, value=0.9
        ),
    )
    calibration = _calibration(cards, threshold=0.45)
    with pytest.raises(ValueError, match="calibrated family-wise"):
        assert_tier0_null_floor(cards, calibration=calibration)


def test_tier0_null_floor_ignores_high_deflated_sharpe() -> None:
    # DSR has no constant null value on v5.3 (measured null DSRs span
    # 0.11-1.0), so it is excluded from the floor: a high DSR alone passes.
    cards = _null_scorecards()
    cards["null_constant_05"] = dataclasses.replace(
        cards["null_constant_05"], deflated_sharpe=1.0
    )
    assert_tier0_null_floor(cards, calibration=_calibration(cards))


def test_null_floor_calibration_loader_round_trips(tmp_path: Path) -> None:
    path = _write_calibration(tmp_path / "calibration.json")
    calibration = load_null_floor_calibration(path)
    assert calibration.schema_version == 1
    assert calibration.method == "family_wise_quantile"
    assert calibration.selected_quantile == 0.99
    assert calibration.selected_threshold == 0.45
    assert calibration.seed_count == 1000
    assert calibration.null_kinds == NULL_FLOOR_KINDS
    assert calibration.validation_eras == _FIXTURE_ERAS
    assert calibration.scoring_identity["scoring_backend"] == "custom"
    assert set(calibration.seed42_expected) == set(NULL_FLOOR_KINDS)
    assert calibration.seed42_family_max == pytest.approx(
        max(abs(row["corr_sharpe_ac"]) for row in calibration.seed42_expected.values())
    )
    assert len(calibration.digest) == 64
    assert "quantile=0.99" in calibration.describe()


def test_null_floor_calibration_loader_rejects_tampered_payload(
    tmp_path: Path,
) -> None:
    path = _write_calibration(tmp_path / "calibration.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected_threshold"] = 9.99  # digest left untouched
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_null_floor_calibration(path)


def test_null_floor_calibration_loader_rejects_unknown_schema(
    tmp_path: Path,
) -> None:
    def bump(payload: dict[str, object]) -> None:
        payload["schema_version"] = 99

    path = _write_calibration(tmp_path / "calibration.json", mutate=bump)
    with pytest.raises(ValueError, match="schema_version"):
        load_null_floor_calibration(path)


def test_null_floor_calibration_loader_rejects_missing_fields(
    tmp_path: Path,
) -> None:
    def drop(payload: dict[str, object]) -> None:
        payload.pop("study_code_fingerprint")

    path = _write_calibration(tmp_path / "calibration.json", mutate=drop)
    with pytest.raises(ValueError, match="missing fields"):
        load_null_floor_calibration(path)


def test_null_floor_calibration_loader_rejects_feature_mean_kind(
    tmp_path: Path,
) -> None:
    def add_kind(payload: dict[str, object]) -> None:
        payload["null_kinds"] = [*NULL_FLOOR_KINDS, "null_feature_mean"]

    path = _write_calibration(tmp_path / "calibration.json", mutate=add_kind)
    with pytest.raises(ValueError, match="structural"):
        load_null_floor_calibration(path)


def test_verify_null_floor_window_accepts_matching_identity() -> None:
    calibration = _calibration()
    verify_null_floor_window(
        calibration,
        window_key_fingerprint=calibration.window_key_fingerprint,
        validation_eras=_FIXTURE_ERAS,
        scoring_identity=dict(_FIXTURE_IDENTITY),
    )


def test_verify_null_floor_window_rejects_stale_window() -> None:
    calibration = _calibration()
    with pytest.raises(ValueError, match="different data window"):
        verify_null_floor_window(
            calibration,
            window_key_fingerprint="0" * 64,
            validation_eras=_FIXTURE_ERAS,
            scoring_identity=dict(_FIXTURE_IDENTITY),
        )


def test_verify_null_floor_window_rejects_era_list_change() -> None:
    calibration = _calibration()
    with pytest.raises(ValueError, match="era window does not match"):
        verify_null_floor_window(
            calibration,
            window_key_fingerprint=calibration.window_key_fingerprint,
            validation_eras=[*_FIXTURE_ERAS, "0061"],
            scoring_identity=dict(_FIXTURE_IDENTITY),
        )


def test_verify_null_floor_window_rejects_scoring_identity_change() -> None:
    calibration = _calibration()
    drifted = {**_FIXTURE_IDENTITY, "scoring_target": "target_ender_60"}
    with pytest.raises(ValueError, match="scoring identity mismatch"):
        verify_null_floor_window(
            calibration,
            window_key_fingerprint=calibration.window_key_fingerprint,
            validation_eras=_FIXTURE_ERAS,
            scoring_identity=drifted,
        )


def test_gate_report_frame_emits_null_floor_rows_with_identity() -> None:
    cards = _null_scorecards()
    calibration = _calibration(cards)
    summary = assert_tier0_null_floor(cards, calibration=calibration)
    result = BenchmarkHierarchyResult(
        scorecards=cards,
        tier_of={kind: 0 for kind in NULL_FLOOR_KINDS},
        gate=None,
        null_floor_ok=True,
        null_floor_errors=(),
        tier4_violations=(),
        monotone_ok=True,
        monotone_error=None,
        gated_reference_id=None,
        null_floor_summary=summary,
    )
    report = gate_report_frame(result)
    assert report.height == 2 * len(NULL_FLOOR_KINDS)
    assert set(report.get_column("model_id")) == set(NULL_FLOOR_KINDS)
    assert set(report.get_column("field")) == {"corr", "corr_sharpe_ac"}
    assert report.get_column("pass").all()
    assert report.get_column("null_floor_method").unique().to_list() == [
        "family_wise_quantile"
    ]
    assert report.get_column("null_floor_quantile").unique().to_list() == [0.99]
    assert report.get_column("null_floor_seed_count").unique().to_list() == [1000]
    assert set(report.get_column("null_floor_reference_fingerprint").to_list()) == {
        calibration.digest
    }
    thresholds = {
        row["field"]: row["threshold"] for row in report.iter_rows(named=True)
    }
    assert thresholds["corr"] == 0.005
    assert thresholds["corr_sharpe_ac"] == 0.45


_REPO_ROOT = Path(__file__).resolve().parents[1]
_COMMITTED_CALIBRATION = (
    _REPO_ROOT / "configs" / "benchmarks" / "null_floor_calibration.json"
)
_REAL_META_MODEL = _REPO_ROOT / "data" / "v5.3" / "meta_model.parquet"

_COMMITTED_PRESENT = _COMMITTED_CALIBRATION.exists()
_CALIBRATION_BOUND = _COMMITTED_PRESENT and _REAL_META_MODEL.exists()


@pytest.mark.skipif(
    not _COMMITTED_PRESENT,
    reason="tier-0 calibration artifact not generated yet",
)
def test_committed_calibration_artifact_is_well_formed() -> None:
    calibration = load_null_floor_calibration(_COMMITTED_CALIBRATION)
    assert calibration.method == "family_wise_quantile"
    assert calibration.seed_count >= 1000
    assert 0.0 < calibration.selected_quantile <= 1.0
    assert calibration.selected_threshold > 0.0
    assert tuple(calibration.null_kinds) == NULL_FLOOR_KINDS
    assert len(calibration.digest) == 64
    # the stored seed-42 anchor is exactly the family max of its own values
    assert calibration.seed42_family_max == pytest.approx(
        max(abs(row["corr_sharpe_ac"]) for row in calibration.seed42_expected.values()),
        abs=1e-12,
    )


@pytest.mark.skipif(
    not _CALIBRATION_BOUND,
    reason="v5.3 dataset or calibration not on disk (git-ignored); skipped in CI",
)
def test_committed_calibration_is_bound_to_current_data_window() -> None:
    calibration = load_null_floor_calibration(_COMMITTED_CALIBRATION)
    from nmr.predictions import validation_key_fingerprint

    gate_config = load_benchmark_file(
        _REPO_ROOT / "configs" / "benchmarks" / "tier4_gate.yaml"
    )
    assert gate_config.gate is not None
    gate = gate_config.gate
    meta_model = pl.read_parquet(_REAL_META_MODEL, columns=["era", "id"])
    verify_null_floor_window(
        calibration,
        window_key_fingerprint=validation_key_fingerprint(meta_model),
        validation_eras=meta_model.get_column("era").unique().to_list(),
        scoring_identity={
            "payout_policy_id": gate.payout_policy_id,
            "scoring_target": gate.scoring_target,
            "scoring_horizon": gate.scoring_horizon,
            "scoring_backend": "custom",
        },
    )


def test_tier4_gate_passes_on_strong_scorecard() -> None:
    card = _make_scorecard(
        corr=dataclasses.replace(_make_scorecard().corr, value=0.04),
        corr_sharpe_ac=dataclasses.replace(_make_scorecard().corr_sharpe_ac, value=1.8),
        fnc=0.03,
        deflated_sharpe=1.2,
        gain_to_pain_ratio=2.0,
        cagr_1y=0.1,
        turnover_mean=0.1,
    )
    assert_tier4_gate(card, GATE)


def test_tier4_gate_reports_every_violation() -> None:
    card = _make_scorecard(
        corr=dataclasses.replace(_make_scorecard().corr, value=0.01),
        fnc=0.001,
        turnover_mean=0.9,
    )
    with pytest.raises(ValueError) as excinfo:
        assert_tier4_gate(card, GATE)
    message = str(excinfo.value)
    assert "corr" in message and "fnc" in message


def test_tier4_gate_allows_unavailable_turnover() -> None:
    # Turnover is structurally unavailable on v5.3 (consecutive validation
    # eras share zero ids); an unavailable turnover is reported by
    # gate_report_frame but is not a hard failure.
    card = _make_scorecard(
        corr=dataclasses.replace(_make_scorecard().corr, value=0.04),
        corr_sharpe_ac=dataclasses.replace(_make_scorecard().corr_sharpe_ac, value=1.8),
        fnc=0.03,
        deflated_sharpe=1.2,
        gain_to_pain_ratio=2.0,
        cagr_1y=0.1,
        turnover_mean=None,
        turnover_reason="no id column",
    )
    assert_tier4_gate(card, GATE)


def _corr_ladder(
    scalars: list[tuple[int, float]],
) -> tuple[dict[str, MetricScorecard], dict[str, int]]:
    cards: dict[str, MetricScorecard] = {}
    tier_of: dict[str, int] = {}
    for tier, scalar in scalars:
        model_id = f"t{tier}_probe"
        card = _make_scorecard(model_id=model_id)
        cards[model_id] = dataclasses.replace(
            card, corr=dataclasses.replace(card.corr, value=scalar)
        )
        tier_of[model_id] = tier
    return cards, tier_of


def test_monotone_ordering_passes_on_escalating_tiers() -> None:
    cards, tier_of = _corr_ladder([(0, 0.0), (1, 0.2), (2, 0.4), (3, 0.6), (4, 0.7)])
    assert_hierarchy_monotone(cards, tier_of=tier_of)


def test_monotone_rejects_inverted_tiers() -> None:
    cards, tier_of = _corr_ladder([(0, 0.5), (1, 0.4), (2, 0.3), (3, 0.2), (4, 0.1)])
    with pytest.raises(ValueError, match="monotone|ordering|tier"):
        assert_hierarchy_monotone(cards, tier_of=tier_of)


def _monotone_fixture() -> tuple[dict[str, MetricScorecard], dict[str, int]]:
    cards: dict[str, MetricScorecard] = {}
    tier_of: dict[str, int] = {}
    for tier, scalar in [(0, 0.0), (1, 0.2), (2, 0.4), (3, 0.6), (4, 0.7)]:
        model_id = f"t{tier}_probe"
        cards[model_id] = _make_scorecard(model_id=model_id, rank_scalar=scalar)
        tier_of[model_id] = tier
    return cards, tier_of


def test_monotone_ordering_passes_on_rank_scalar_metric() -> None:
    cards, tier_of = _monotone_fixture()
    assert_hierarchy_monotone(cards, tier_of=tier_of, metric="rank_scalar")


def test_monotone_missing_tier_raises() -> None:
    cards, tier_of = _monotone_fixture()
    del tier_of["t2_probe"]
    with pytest.raises(ValueError, match=r"0\.\.4"):
        assert_hierarchy_monotone(cards, tier_of=tier_of)


def test_monotone_missing_scorecard_raises() -> None:
    cards, tier_of = _monotone_fixture()
    del cards["t3_probe"]
    with pytest.raises(ValueError, match="t3_probe"):
        assert_hierarchy_monotone(cards, tier_of=tier_of)


def make_scorecard(
    *,
    corr: float = 0.03,
    sharpe: float = 0.8,
    fnc: float = 0.02,
    gpr: float = 1.5,
    cagr: float = 0.01,
    turnover: float | None = 0.1,
) -> MetricScorecard:
    """Build a tier-4-gate probe scorecard over the synthetic fixture.

    ``corr``/``sharpe`` replace the ``MetricCell.value`` of ``corr`` and
    ``corr_sharpe_ac``; the remaining kwargs map one-to-one onto
    ``MetricScorecard`` fields (``gpr`` -> ``gain_to_pain_ratio``,
    ``cagr`` -> ``cagr_1y``, ``turnover`` -> ``turnover_mean``).
    """
    card = _make_scorecard()
    return dataclasses.replace(
        card,
        corr=dataclasses.replace(card.corr, value=float(corr)),
        corr_sharpe_ac=dataclasses.replace(card.corr_sharpe_ac, value=float(sharpe)),
        fnc=float(fnc),
        gain_to_pain_ratio=float(gpr),
        cagr_1y=float(cagr),
        turnover_mean=turnover,
    )


def _gate() -> Tier4GateConfig:
    return Tier4GateConfig(
        payout_policy_id="classic_legacy_075_225_clip005_v1",
        scoring_target="target",
        scoring_horizon="20D",
        corr_min=0.0286,
        corr_sharpe_ac_min=0.78,
        fnc_min=0.020,
        gain_to_pain_min=1.50,
    )


def test_tier4_gate_verdict_shape_and_pass() -> None:
    card = make_scorecard(
        corr=0.03, sharpe=0.8, fnc=0.02, gpr=1.5, cagr=0.01, turnover=0.1
    )
    verdict = tier4_gate_verdict(card, _gate())
    assert verdict == {
        "corr": True,
        "corr_sharpe_ac": True,
        "fnc": True,
        "gain_to_pain_ratio": True,
    }


def test_tier4_gate_rejects_payout_policy_mismatch() -> None:
    card = dataclasses.replace(
        make_scorecard(
            corr=0.03,
            sharpe=0.8,
            fnc=0.02,
            gpr=1.5,
            cagr=0.01,
            turnover=0.1,
        ),
        payout_policy_id="classic_atomic_ender60_r1343_v1",
        scoring_target="target_ender_60",
        scoring_horizon="60D",
    )
    with pytest.raises(ValueError, match="policy mismatch"):
        tier4_gate_verdict(card, _gate())


def test_tier_max_corrs_orders_by_tier() -> None:
    cards = {
        "t0": make_scorecard(corr=0.002),
        "t1": make_scorecard(corr=0.005),
        "t4": make_scorecard(corr=0.029),
    }
    rungs = tier_max_corrs(cards, {"t0": 0, "t1": 1, "t4": 4})
    assert rungs == {0: 0.002, 1: 0.005, 4: 0.029}
