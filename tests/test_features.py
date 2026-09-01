from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from nmr.config import DataConfig
from nmr.data import IngestionAgent
from nmr.features import (
    DEFAULT_MAX_ABS_DECAY,
    DEFAULT_MIN_MEAN_CORR,
    feature_stability_screen,
    resolve_feature_sets,
    resolve_small_feature_set,
    select_stable_features,
)


def _write_features(tmp_path, *, sets: dict[str, list[str]]) -> None:
    (tmp_path / "features.json").write_text(
        json.dumps({"feature_sets": sets, "targets": ["target"]}), encoding="utf-8"
    )


def test_resolve_feature_sets_returns_all_named_sets_in_sorted_order(tmp_path) -> None:
    _write_features(
        tmp_path,
        sets={"all": ["f3", "f1"], "small": ["f1"], "zulu": ["f4"], "alpha": ["f2"]},
    )
    resolved = resolve_feature_sets(tmp_path / "features.json")
    assert set(resolved) == {"all", "small", "zulu", "alpha"}
    assert list(resolved) == sorted(resolved)  # deterministic key order
    assert resolved["all"] == ["f3", "f1"]  # values preserved verbatim (copy)


def test_resolve_feature_sets_is_deterministic_across_calls(tmp_path) -> None:
    _write_features(tmp_path, sets={"small": ["f1"], "medium": ["f1", "f2"]})
    path = tmp_path / "features.json"
    assert resolve_feature_sets(path) == resolve_feature_sets(path)


def test_resolve_feature_sets_defensive_copy(tmp_path) -> None:
    _write_features(tmp_path, sets={"small": ["f1"]})
    resolved = resolve_feature_sets(tmp_path / "features.json")
    resolved["small"].append("corrupt_me")
    again = resolve_feature_sets(tmp_path / "features.json")
    assert again["small"] == ["f1"]


def test_resolve_feature_sets_rejects_missing_or_empty_feature_sets(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    (tmp_path / "empty.json").write_text(
        json.dumps({"feature_sets": {}, "targets": []}), encoding="utf-8"
    )
    (tmp_path / "notmap.json").write_text(
        json.dumps({"feature_sets": ["f1"], "targets": []}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        resolve_feature_sets(missing)
    with pytest.raises(ValueError, match="feature_sets"):
        resolve_feature_sets(tmp_path / "empty.json")
    with pytest.raises(ValueError, match="feature_sets"):
        resolve_feature_sets(tmp_path / "notmap.json")


def test_resolve_feature_sets_rejects_non_dict_top_level(tmp_path) -> None:
    (tmp_path / "list.json").write_text(json.dumps(["f1", "f2"]), encoding="utf-8")
    (tmp_path / "string.json").write_text(json.dumps("oops"), encoding="utf-8")
    with pytest.raises(ValueError, match="feature_sets"):
        resolve_feature_sets(tmp_path / "list.json")
    with pytest.raises(ValueError, match="feature_sets"):
        resolve_feature_sets(tmp_path / "string.json")


def _screen_frame() -> pl.DataFrame:
    """f_good: per-era CORR ~ +1 with zero decay; f_bad: CORR ~ -1 decaying to 0."""
    rows: list[dict] = []
    for era in range(1, 21):
        for idx in range(50):
            rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f_good": idx * 0.02,  # CORR +1 all eras
                    "f_bad": -idx * (0.02 - era * 0.001),  # CORR ~ -1 -> 0 (decay)
                    "target": idx * 0.02 + 0.5,
                }
            )
    return pl.DataFrame(rows)


def test_screen_reports_corr_and_decay_per_feature() -> None:
    frame = _screen_frame()
    screen = feature_stability_screen(
        frame, feature_cols=["f_good", "f_bad"], target_col="target", era_col="era"
    )
    assert set(screen.get_column("feature").to_list()) == {"f_good", "f_bad"}
    assert screen.height == 2
    good = screen.filter(pl.col("feature") == "f_good").row(0, named=True)
    bad = screen.filter(pl.col("feature") == "f_bad").row(0, named=True)
    assert good["mean_corr"] > 0.9
    assert bad["mean_corr"] < -0.9
    assert abs(good["decay_slope"]) < abs(bad["decay_slope"])
    assert good["n_eras"] == 20 and bad["n_eras"] == 20


def test_screen_flags_stability_by_default_thresholds() -> None:
    screen = feature_stability_screen(
        _screen_frame(), feature_cols=["f_good", "f_bad"], target_col="target"
    )
    good = screen.filter(pl.col("feature") == "f_good").get_column("stable")[0]
    bad = screen.filter(pl.col("feature") == "f_bad").get_column("stable")[0]
    assert good is True
    assert bad is False
    # default constants are positive and sane
    assert DEFAULT_MIN_MEAN_CORR > 0.0 and DEFAULT_MAX_ABS_DECAY > 0.0


def test_screen_all_degenerate_eras_yield_null_stats() -> None:
    rows = [
        {"era": "1", "id": "a", "f": 1.0, "target": 0.5},  # 1 row: degenerate
        {"era": "1", "id": "b", "f": 1.0, "target": 0.5},  # zero variance
        {"era": "2", "id": "c", "f": float("nan"), "target": 0.5},  # non-finite
        {"era": "3", "id": "d", "f": 0.2, "target": 0.9},
    ]
    frame = pl.DataFrame(rows)
    screen = feature_stability_screen(frame, feature_cols=["f"], target_col="target")
    assert screen.height == 1
    row = screen.row(0, named=True)
    assert row["n_eras"] == 0
    assert row["mean_corr"] is None
    assert row["decay_slope"] is None
    assert screen.get_column("stable").to_list() == [False]


def test_screen_excludes_degenerate_eras_from_aggregates() -> None:
    # era 0002 has an all-NaN target (degenerate): its forced zero-IC vector
    # must not dilute mean_corr, distort the decay slope, or count in n_eras.
    # Valid eras carry a perfect linear signal (corr = 1.0, slope = 0).
    rng = np.random.default_rng(7)
    rows = []
    for e in ["0001", "0002", "0003"]:
        for i in range(20):
            x = float(rng.normal())
            rows.append(
                {
                    "era": e,
                    "id": f"{e}_{i}",
                    "f": x,
                    "target": float("nan") if e == "0002" else x,
                }
            )
    screen = feature_stability_screen(
        pl.DataFrame(rows), feature_cols=["f"], target_col="target"
    )
    row = screen.row(0, named=True)
    assert row["n_eras"] == 2  # degenerate era excluded
    assert row["mean_corr"] > 0.9  # not diluted by the zero vector
    assert abs(row["decay_slope"]) <= 0.001  # no artificial decay from the zero
    assert screen.get_column("stable").to_list() == [True]


def test_select_stable_features_filters_on_thresholds() -> None:
    screen = feature_stability_screen(
        _screen_frame(), feature_cols=["f_good", "f_bad"], target_col="target"
    )
    kept = select_stable_features(screen, min_mean_corr=-1.0, max_abs_decay=1.0)
    assert kept == ["f_bad", "f_good"]  # sorted; both pass loose thresholds
    strict = select_stable_features(screen, min_mean_corr=0.9, max_abs_decay=0.01)
    assert strict == ["f_good"]


def test_stable_inverse_feature_is_retained_with_signed_correlation() -> None:
    rows = [
        {
            "era": str(era),
            "id": f"{era}_{index}",
            "inverse": -float(index),
            "target": float(index),
        }
        for era in range(1, 6)
        for index in range(20)
    ]
    screen = feature_stability_screen(
        pl.DataFrame(rows),
        feature_cols=["inverse"],
        target_col="target",
        min_mean_corr=0.1,
        max_abs_decay=0.01,
    )
    row = screen.row(0, named=True)

    assert row["mean_corr"] == pytest.approx(-1.0)
    assert row["stable"] is True
    assert select_stable_features(screen, min_mean_corr=0.1, max_abs_decay=0.01) == [
        "inverse"
    ]


def test_select_stable_features_rejects_screen_without_required_columns() -> None:
    bad = pl.DataFrame({"feature": ["f1"], "mean_corr": [0.5]})
    with pytest.raises(ValueError, match="decay_slope"):
        select_stable_features(bad, min_mean_corr=0.0, max_abs_decay=1.0)


def _write_features_with_sunshine(tmp_path) -> None:
    version_dir = tmp_path / "vtest"
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "features.json").write_text(
        json.dumps(
            {
                "feature_sets": {
                    "small": ["f1", "f2"],
                    "sunshine": ["f1", "f2", "f3"],
                },
                "targets": ["target"],
            }
        ),
        encoding="utf-8",
    )
    pl.DataFrame(
        {
            "era": ["1", "1"],
            "id": ["a", "b"],
            "f1": [0.1, 0.2],
            "f2": [0.3, 0.4],
            "f3": [0.5, 0.6],
            "target": [0.2, 0.3],
        }
    ).write_parquet(version_dir / "train.parquet")


def test_ingestion_resolves_feature_subset_from_features_json(tmp_path) -> None:
    _write_features_with_sunshine(tmp_path)
    cfg = DataConfig(
        version="vtest",
        feature_set="small",
        feature_subset="sunshine",
        data_dir=tmp_path,
    )
    agent = IngestionAgent(cfg)
    assert agent.features() == ["f1", "f2", "f3"]  # resolved_feature_set threaded


def test_ingestion_rejects_unknown_feature_subset_with_valid_options(tmp_path) -> None:
    _write_features_with_sunshine(tmp_path)
    cfg = DataConfig(
        version="vtest",
        feature_set="small",
        feature_subset="nope",
        data_dir=tmp_path,
    )
    agent = IngestionAgent(cfg)
    with pytest.raises(ValueError, match="sunshine"):
        agent.features()


def test_resolve_small_feature_set_intersects_with_available_columns(tmp_path) -> None:
    _write_features(
        tmp_path,
        sets={"small": ["feature_a", "feature_b", "feature_extra"]},
    )
    resolved = resolve_small_feature_set(
        tmp_path / "features.json", ["feature_b", "feature_a"]
    )
    assert resolved == ["feature_a", "feature_b"]  # declared order, filtered


def test_resolve_small_feature_set_raises_on_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_small_feature_set(tmp_path / "features.json", ["feature_a"])


def test_resolve_small_feature_set_raises_on_corrupt_json(tmp_path) -> None:
    (tmp_path / "features.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        resolve_small_feature_set(tmp_path / "features.json", ["feature_a"])


def test_resolve_small_feature_set_raises_when_small_absent(tmp_path) -> None:
    _write_features(tmp_path, sets={"medium": ["feature_a"]})
    with pytest.raises(ValueError, match="small"):
        resolve_small_feature_set(tmp_path / "features.json", ["feature_a"])


def test_resolve_small_feature_set_raises_on_empty_intersection(tmp_path) -> None:
    _write_features(tmp_path, sets={"small": ["feature_a"]})
    with pytest.raises(ValueError, match="no overlap"):
        resolve_small_feature_set(tmp_path / "features.json", ["feature_zzz"])
