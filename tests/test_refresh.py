"""Pure logic for the round-aware Numerai data refresh (nmr/refresh.py)."""

from __future__ import annotations

import pytest

from nmr.config import REPO_ROOT, load_config
from nmr.refresh import (
    CURRENT_DATA_VERSION,
    EXPANDING_FILES,
    LIVE_FRESH_FILES,
    STATIC_FILES,
    _parse_version,
    build_era_manifest,
    classify_refresh_plan,
    detect_newer_version,
    needs_live_refresh,
)


def test_current_version_is_v5_3() -> None:
    assert CURRENT_DATA_VERSION == "v5.3"


def test_parse_version_valid() -> None:
    assert _parse_version("v5.2") == (5, 2)
    assert _parse_version("v5.10") == (5, 10)  # multi-digit minor
    assert _parse_version("v6.0") == (6, 0)
    assert _parse_version("v0.0") == (0, 0)


@pytest.mark.parametrize(
    "bad",
    ["5.2", "v5", "v5.2.1", "vX.2", "v5.a", "", "v5.2 "],
)
def test_parse_version_malformed_raises(bad: str) -> None:
    with pytest.raises(ValueError):
        _parse_version(bad)


def test_detect_newer_version_none_cases() -> None:
    assert detect_newer_version([], "v5.2") is None
    assert detect_newer_version(["v5.2"], "v5.2") is None
    assert detect_newer_version(["v5.0", "v5.1"], "v5.2") is None


def test_detect_newer_version_finds_newest() -> None:
    assert detect_newer_version(["v5.3"], "v5.2") == "v5.3"
    assert detect_newer_version(["v6.0"], "v5.2") == "v6.0"
    # multi-digit regression: v5.10 > v5.3 numerically, not lexicographically
    assert detect_newer_version(["v5.10"], "v5.3") == "v5.10"
    assert (
        detect_newer_version(["v4.9", "v5.3", "v5.2", "v5.10"], "v5.2")
        == "v5.10"
    )


def test_detect_newer_version_malformed_raises() -> None:
    with pytest.raises(ValueError):
        detect_newer_version(["v5.2", "garbage"], "v5.2")


def test_drift_guard_current_version_matches_canonical_config() -> None:
    cfg_path = REPO_ROOT / "configs" / "first_model.yaml"
    if not cfg_path.exists():
        pytest.skip("configs/first_model.yaml absent in this checkout")
    cfg = load_config(cfg_path)
    assert CURRENT_DATA_VERSION == cfg.data.version


def test_needs_live_refresh_truth_table() -> None:
    assert needs_live_refresh(1295, 1294, True) is True   # round advanced
    assert needs_live_refresh(1294, 1294, True) is False  # up to date
    assert needs_live_refresh(1294, 1294, False) is True  # file missing
    assert needs_live_refresh(1294, None, True) is True   # no ledger record
    assert needs_live_refresh(1295, 1296, True) is True   # ahead-of-remote: reconcile


def test_build_era_manifest_columns_and_values() -> None:
    records = build_era_manifest(
        {
            "train": ("0001", "0574"),
            "validation": ("0575", "1208"),
            "live": (None, None),
        },
        round_id=1294,
        today="2026-08-08",
    )
    assert [r["dataset"] for r in records] == ["train", "validation", "live"]
    assert records[0] == {
        "date": "2026-08-08",
        "dataset": "train",
        "start_era": "0001",
        "end_era": "0574",
        "round_id": None,
    }
    assert records[2] == {
        "date": "2026-08-08",
        "dataset": "live",
        "start_era": None,  # unlabeled round; script serializes to "X"
        "end_era": None,
        "round_id": 1294,
    }


def test_build_era_manifest_live_x_strings_pass_through() -> None:
    records = build_era_manifest(
        {"train": ("0001", "0574"), "validation": ("0575", "1208"), "live": ("X", "X")},
        round_id=1300,
        today="2026-08-08",
    )
    assert records[2]["start_era"] == "X"
    assert records[2]["end_era"] == "X"


def test_build_era_manifest_nonlive_empty_raises() -> None:
    with pytest.raises(ValueError):
        build_era_manifest(
            {"train": (None, None), "validation": ("0575", "1208"), "live": ("X", "X")},
            round_id=1300,
            today="2026-08-08",
        )


def test_build_era_manifest_deterministic() -> None:
    kwargs = {
        "era_ranges": {
            "train": ("0001", "0574"),
            "validation": ("0575", "1208"),
            "live": (None, None),
        },
        "round_id": 1294,
        "today": "2026-08-08",
    }
    assert build_era_manifest(**kwargs) == build_era_manifest(**kwargs)


def test_classify_refresh_plan_round_advanced() -> None:
    existing = {"features.json", "train.parquet", "live.parquet"}
    plan = classify_refresh_plan(round_advanced=True, existing=existing)
    for name in STATIC_FILES:
        assert plan[name] == "ensure"
    for name in LIVE_FRESH_FILES:
        assert plan[name] == "refresh"
    for name in EXPANDING_FILES:
        assert plan[name] == "refresh"


def test_classify_refresh_plan_no_advance() -> None:
    existing = set(STATIC_FILES) | set(LIVE_FRESH_FILES) | set(EXPANDING_FILES)
    plan = classify_refresh_plan(round_advanced=False, existing=existing)
    assert plan["live.parquet"] == "skip"
    assert plan["validation.parquet"] == "skip"
    assert plan["features.json"] == "ensure"


def test_classify_refresh_plan_missing_live_file() -> None:
    plan = classify_refresh_plan(
        round_advanced=False, existing=set(STATIC_FILES)
    )
    assert plan["live.parquet"] == "refresh"
    assert plan["live_benchmark_models.parquet"] == "refresh"


def test_classify_refresh_plan_live_only_skips_expanding() -> None:
    plan = classify_refresh_plan(
        round_advanced=True, existing=set(), live_only=True
    )
    for name in EXPANDING_FILES:
        assert plan[name] == "skip"
    assert plan["live.parquet"] == "refresh"
