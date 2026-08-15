"""Purged train->validation split invariants for benchmark fits."""

from __future__ import annotations

import pytest

from nmr.benchmark import train_validation_purged_split


def _eras(start: int, stop: int) -> list[str]:
    return [f"{i:04d}" for i in range(start, stop)]


def test_valid_split_returns_trimmed_train_and_val() -> None:
    train, val = train_validation_purged_split(
        _eras(1, 100), _eras(100, 200), purge_eras=8
    )
    assert train == tuple(_eras(1, 92))
    assert val == tuple(_eras(100, 200))


def test_zero_purge_keeps_all_train_eras() -> None:
    train, val = train_validation_purged_split(
        _eras(1, 100), _eras(100, 200), purge_eras=0
    )
    assert train == tuple(_eras(1, 100))


def test_gap_too_small_raises() -> None:
    with pytest.raises(ValueError, match="purge"):
        train_validation_purged_split(_eras(1, 100), _eras(102, 200), purge_eras=8)


def test_gap_too_large_raises() -> None:
    with pytest.raises(ValueError, match="purge"):
        train_validation_purged_split(_eras(1, 100), _eras(120, 200), purge_eras=8)


def test_overlap_raises() -> None:
    with pytest.raises(ValueError, match="overlap"):
        train_validation_purged_split(_eras(1, 100), _eras(50, 200), purge_eras=8)


def test_non_numeric_label_raises() -> None:
    with pytest.raises(ValueError, match="[Nn]umeric"):
        train_validation_purged_split(["era_1", "0002"], _eras(10, 20), purge_eras=8)


def test_zero_padding_inconsistency_raises() -> None:
    with pytest.raises(ValueError, match="[Zz]ero-padding"):
        train_validation_purged_split(["1", "02", "0003"], _eras(11, 20), purge_eras=8)


def test_degenerate_train_raises() -> None:
    with pytest.raises(ValueError, match="purge|train"):
        train_validation_purged_split(_eras(1, 5), _eras(20, 30), purge_eras=8)


def test_empty_eras_raise() -> None:
    with pytest.raises(ValueError, match="empty"):
        train_validation_purged_split([], _eras(20, 30), purge_eras=8)
