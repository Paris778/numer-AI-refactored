# tests/test_opt.py
from __future__ import annotations

import pytest

from nmr.opt import _SpaceParam, _parse_space


def test_parse_space_accepts_all_kinds() -> None:
    space = {
        "learning_rate": {"kind": "float", "low": 0.005, "high": 0.05, "log": True},
        "n_estimators": {"kind": "int", "low": 100, "high": 10000, "log": True},
        "num_leaves": {"kind": "int", "low": 16, "high": 256},
        "boosting": {"kind": "categorical", "choices": ["gbdt", "dart"]},
    }
    parsed = {p.name: p for p in _parse_space(space)}
    assert parsed["learning_rate"].kind == "float"
    assert parsed["learning_rate"].log is True
    assert parsed["n_estimators"].log is True
    assert parsed["n_estimators"].step is None
    assert parsed["num_leaves"].step is None
    assert parsed["boosting"].choices == ["gbdt", "dart"]


def test_parse_space_int_step() -> None:
    parsed = {p.name: p for p in _parse_space(
        {"n_estimators": {"kind": "int", "low": 100, "high": 10000, "step": 100}})}
    assert parsed["n_estimators"].step == 100


@pytest.mark.parametrize(
    "space, match",
    [
        ({}, "empty"),
        ({"a": {"kind": "bogus", "low": 0, "high": 1}}, "kind"),
        ({"a": {"kind": "float", "low": 1.0, "high": 0.5}}, "low"),
        ({"a": {"kind": "float", "low": 0.0, "high": 0.1, "log": True}}, "low"),
        ({"a": {"kind": "int", "low": 1, "high": 10, "log": True, "step": 2}}, "step"),
        ({"a": {"kind": "int", "low": 1, "high": 10, "step": 0}}, "step"),
        ({"a": {"kind": "categorical", "choices": []}}, "choices"),
        ({"a": {"kind": "categorical", "choices": [(1, 2)]}}, "choices"),
        ({"a": {"kind": "categorical", "choices": [None]}}, "choices"),
        ({"a": {"kind": "float", "low": 1.0, "high": 2.0, "bogus": 1}}, "unknown"),
        ({"a": "not-a-dict"}, "spec"),
        ({"a": {"kind": "float", "high": 1.0}}, "low"),
        ({"a": {"kind": "int", "low": 1}}, "low"),
        ({"a": {"kind": "int", "low": 0, "high": 10, "log": True}}, "low"),
        ({"a": {"kind": "int", "low": 1, "high": 10, "step": 2.5}}, "step"),
    ],
)
def test_parse_space_validation_errors(space, match) -> None:
    with pytest.raises(ValueError, match=match):
        _parse_space(space)
