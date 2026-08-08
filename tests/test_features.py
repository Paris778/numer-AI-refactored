from __future__ import annotations

import json

import pytest

from nmr.features import resolve_feature_sets


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
