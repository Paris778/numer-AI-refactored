"""Tests for nmr.splitter.PurgedEraSplitter."""

from __future__ import annotations

import pytest

from nmr.config import SplitConfig
from nmr.splitter import Fold, PurgedEraSplitter


def _eras(start: int, stop: int) -> list[str]:
    return [str(value) for value in range(start, stop + 1)]


class TestValidation:
    def test_non_numeric_era_raises(self) -> None:
        splitter = PurgedEraSplitter(SplitConfig())
        with pytest.raises(ValueError, match="Non-numeric era label"):
            splitter.split(["1", "2", "era3"])

    def test_empty_era_universe_raises(self) -> None:
        splitter = PurgedEraSplitter(SplitConfig())
        with pytest.raises(ValueError, match="Era universe is empty"):
            splitter.split([])

    def test_duplicate_eras_collapse_to_unique(self) -> None:
        splitter = PurgedEraSplitter(SplitConfig(scheme="anchor", purge_eras=1))
        folds = splitter.split(["1", "2", "2", "3", "4", "5", "6"])
        assert folds[0].train_eras == ("1", "2")
        assert folds[0].val_eras == ("4", "5", "6")


class TestWalkForward:
    def test_walk_forward_yields_requested_fold_count(self) -> None:
        splitter = PurgedEraSplitter(
            SplitConfig(n_folds=4, purge_eras=1, embargo_eras=2)
        )
        folds = splitter.split(_eras(1, 25))
        assert len(folds) == 4
        assert all(isinstance(fold, Fold) for fold in folds)

    def test_walk_forward_is_expanding_and_strictly_past_only(self) -> None:
        splitter = PurgedEraSplitter(
            SplitConfig(n_folds=4, purge_eras=1, embargo_eras=2)
        )
        folds = splitter.split(_eras(1, 25))

        train_lengths = [len(fold.train_eras) for fold in folds]
        assert train_lengths == sorted(train_lengths)
        for fold in folds:
            assert max(map(int, fold.train_eras)) < min(map(int, fold.val_eras))

    def test_walk_forward_honors_purge_gap_and_excludes_buffer(self) -> None:
        splitter = PurgedEraSplitter(
            SplitConfig(n_folds=4, purge_eras=2, embargo_eras=1)
        )
        folds = splitter.split(_eras(1, 30))

        for fold in folds:
            train_max = max(map(int, fold.train_eras))
            val_min = min(map(int, fold.val_eras))
            assert val_min - train_max > 2
            purge_buffer = {train_max + 1, train_max + 2}
            assert purge_buffer.isdisjoint(set(map(int, fold.train_eras)))
            assert purge_buffer.isdisjoint(set(map(int, fold.val_eras)))

    def test_walk_forward_is_invariant_to_embargo_eras(self) -> None:
        eras = _eras(1, 25)
        base = PurgedEraSplitter(SplitConfig(n_folds=4, purge_eras=1, embargo_eras=0))
        widened = PurgedEraSplitter(
            SplitConfig(n_folds=4, purge_eras=1, embargo_eras=12)
        )

        assert base.split(eras) == widened.split(eras)

    def test_walk_forward_is_deterministic(self) -> None:
        splitter = PurgedEraSplitter(
            SplitConfig(n_folds=4, purge_eras=1, embargo_eras=2)
        )
        eras = [
            "10",
            "2",
            "3",
            "2",
            "9",
            "1",
            "8",
            "7",
            "6",
            "5",
            "4",
            "11",
            "12",
            "13",
            "14",
        ]
        assert splitter.split(eras) == splitter.split(eras)

    def test_walk_forward_matches_benchmark_geometry(self) -> None:
        splitter = PurgedEraSplitter(
            SplitConfig(n_folds=4, purge_eras=8, embargo_eras=4)
        )
        folds = splitter.split(_eras(1, 780))
        starts = [int(fold.val_eras[0]) for fold in folds]
        assert starts == [157, 313, 469, 625]
        train_ends = [int(fold.train_eras[-1]) for fold in folds]
        assert train_ends == [148, 304, 460, 616]


class TestAnchor:
    def test_anchor_structure_is_correct(self) -> None:
        splitter = PurgedEraSplitter(
            SplitConfig(scheme="anchor", purge_eras=1, embargo_eras=2)
        )
        folds = splitter.split(_eras(1, 10))

        assert len(folds) == 1
        fold = folds[0]
        assert fold.train_eras == ("1", "2", "3", "4")
        assert fold.val_eras == ("6", "7", "8", "9", "10")
        assert set(fold.train_eras).isdisjoint(set(fold.val_eras))

    def test_anchor_is_invariant_to_embargo_eras(self) -> None:
        eras = _eras(1, 10)
        base = PurgedEraSplitter(
            SplitConfig(scheme="anchor", purge_eras=1, embargo_eras=0)
        )
        widened = PurgedEraSplitter(
            SplitConfig(scheme="anchor", purge_eras=1, embargo_eras=12)
        )

        assert base.split(eras) == widened.split(eras)


class TestInfeasibility:
    def test_infeasible_walk_forward_raises(self) -> None:
        splitter = PurgedEraSplitter(
            SplitConfig(n_folds=4, purge_eras=8, embargo_eras=4)
        )
        with pytest.raises(ValueError, match="too small"):
            splitter.split(_eras(1, 12))

    def test_infeasible_anchor_raises(self) -> None:
        splitter = PurgedEraSplitter(
            SplitConfig(scheme="anchor", purge_eras=5, embargo_eras=1)
        )
        with pytest.raises(ValueError, match="too small"):
            splitter.split(_eras(1, 6))
