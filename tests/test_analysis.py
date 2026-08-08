"""Unit tests for nmr.analysis — synthetic frames, seeded where random."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nmr.analysis import SplitStats, describe_splits, era_structure


def _frame(n_eras: int = 4, rows_per_era: int = 8) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for e in range(n_eras):
        era = f"{e + 1:04d}"
        for i in range(rows_per_era):
            rows.append({"era": era, "id": f"n{e:03d}{i:03d}", "x": float(i)})
    return pl.DataFrame(rows)


def test_describe_splits_counts() -> None:
    splits = {"train": _frame(4, 8), "validation": _frame(3, 10)}
    out = describe_splits(splits)
    assert set(out) == {"train", "validation"}
    train = out["train"]
    assert isinstance(train, SplitStats)
    assert train.n_rows == 32
    assert train.n_eras == 4
    assert train.min_era == "0001" and train.max_era == "0004"
    assert train.rows_per_era_min == 8
    assert train.rows_per_era_max == 8
    assert train.rows_per_era_mean == 8.0
    assert train.n_ids == 32


def test_describe_splits_rows_per_era_stats() -> None:
    frame = pl.DataFrame(
        {
            "era": ["0001", "0001", "0002", "0002", "0002"],
            "id": ["a", "b", "c", "d", "e"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    out = describe_splits({"s": frame})["s"]
    assert out.rows_per_era_min == 2
    assert out.rows_per_era_max == 3
    assert out.rows_per_era_mean == 2.5
    assert out.n_eras == 2


def test_describe_splits_requires_id() -> None:
    frame = pl.DataFrame({"era": ["0001"], "x": [1.0]})
    with pytest.raises(ValueError):
        describe_splits({"s": frame})


def test_era_structure_gap_detection() -> None:
    frame = pl.DataFrame(
        {
            "era": ["0001", "0002", "0002", "0004"],
            "id": ["a", "b", "c", "d"],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = era_structure(frame)
    assert out["era"].to_list() == ["0001", "0002", "0004"]
    assert out["gap"].to_list() == [False, False, True]  # 0004 jumps from 0002
    assert out["n_rows"].to_list() == [1, 2, 1]


def test_era_structure_empty_raises() -> None:
    with pytest.raises(ValueError):
        era_structure(pl.DataFrame({"era": [], "id": [], "x": []}))
