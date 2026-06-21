"""Era-purged validation splitting.

``PurgedEraSplitter`` is a pure, deterministic firewall between an era universe
and every downstream training/evaluation component. It operates only on era
labels and emits leakage-safe folds with these invariants:

- unique eras, ordered numerically
- validation eras are contiguous blocks
- train eras are strictly earlier than validation eras
- the purge buffer immediately before validation is excluded from both sets

For the forward-only schemes implemented here (``walk_forward`` and ``anchor``),
training is structurally restricted to eras strictly earlier than validation.
That makes ``embargo_eras`` inert by design; it is reserved for a future
two-sided scheme where post-validation training eras can exist.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from nmr.config import SplitConfig

__all__ = ["Fold", "PurgedEraSplitter"]


@dataclass(frozen=True)
class Fold:
    index: int
    train_eras: tuple[str, ...]
    val_eras: tuple[str, ...]


class PurgedEraSplitter:
    """Pure era-grouped splitter driven by :class:`SplitConfig`.

    ``embargo_eras`` is accepted for API continuity with the broader roadmap,
    but it has no effect for the forward-only schemes currently implemented.
    """

    def __init__(self, split: SplitConfig) -> None:
        self._split = split

    def split(self, eras: Iterable[str]) -> list[Fold]:
        """Return leakage-safe folds for ``eras``.

        The era universe is deduplicated and sorted numerically. Non-numeric era
        labels are rejected because chronology cannot be inferred safely.

        Notes
        -----
        ``embargo_eras`` is structurally inert here because train eras are always
        strictly earlier than validation eras.
        """
        ordered_eras = self._normalize_eras(eras)
        if self._split.scheme == "walk_forward":
            return self._walk_forward(ordered_eras)
        if self._split.scheme == "anchor":
            return self._anchor(ordered_eras)
        raise ValueError(f"Unsupported split scheme: {self._split.scheme!r}")

    def _normalize_eras(self, eras: Iterable[str]) -> list[str]:
        numeric_to_label: dict[int, str] = {}
        for era in eras:
            if not isinstance(era, str):
                raise ValueError(
                    f"Era labels must be strings, got {type(era).__name__}"
                )
            try:
                era_num = int(era)
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric era label {era!r}; splitter requires numeric chronology"
                ) from exc
            numeric_to_label.setdefault(era_num, era)

        if not numeric_to_label:
            raise ValueError("Era universe is empty")

        return [numeric_to_label[num] for num in sorted(numeric_to_label)]

    def _window_geometry(self, era_count: int, fold_count: int) -> tuple[int, int]:
        val_size = era_count // (fold_count + 1)
        if val_size < 1:
            raise ValueError(
                "Era universe is too small for the requested fold count: "
                f"eras={era_count}, folds={fold_count}"
            )

        prefix_size = era_count - fold_count * val_size
        min_train_size = prefix_size - self._split.purge_eras
        if min_train_size < 1:
            raise ValueError(
                "Era universe is too small after applying purge: "
                f"eras={era_count}, folds={fold_count}, purge={self._split.purge_eras}"
            )

        return val_size, prefix_size

    def _walk_forward(self, ordered_eras: list[str]) -> list[Fold]:
        val_size, prefix_size = self._window_geometry(
            era_count=len(ordered_eras), fold_count=self._split.n_folds
        )
        folds: list[Fold] = []

        for index in range(self._split.n_folds):
            val_start = prefix_size + index * val_size
            val_stop = val_start + val_size
            train_stop = val_start - self._split.purge_eras

            train_eras = tuple(ordered_eras[:train_stop])
            val_eras = tuple(ordered_eras[val_start:val_stop])
            fold = Fold(index=index, train_eras=train_eras, val_eras=val_eras)
            self._validate_fold(fold, ordered_eras)
            folds.append(fold)

        return folds

    def _anchor(self, ordered_eras: list[str]) -> list[Fold]:
        val_size, prefix_size = self._window_geometry(
            era_count=len(ordered_eras), fold_count=1
        )

        val_start = prefix_size
        train_stop = val_start - self._split.purge_eras
        fold = Fold(
            index=0,
            train_eras=tuple(ordered_eras[:train_stop]),
            val_eras=tuple(ordered_eras[val_start : val_start + val_size]),
        )
        self._validate_fold(fold, ordered_eras)
        return [fold]

    def _validate_fold(self, fold: Fold, ordered_eras: list[str]) -> None:
        if not fold.train_eras or not fold.val_eras:
            raise ValueError(f"Degenerate fold produced: {fold}")

        train_nums = [int(era) for era in fold.train_eras]
        val_nums = [int(era) for era in fold.val_eras]

        if set(fold.train_eras) & set(fold.val_eras):
            raise ValueError(f"Fold {fold.index} reuses eras across train/val")

        if max(train_nums) >= min(val_nums):
            raise ValueError(f"Fold {fold.index} is not strictly time-ordered")

        if min(val_nums) - max(train_nums) <= self._split.purge_eras:
            raise ValueError(f"Fold {fold.index} violates purge invariant")

        ordered_nums = [int(era) for era in ordered_eras]
        train_set = set(train_nums)
        val_set = set(val_nums)

        purge_buffer = {
            era_num
            for era_num in ordered_nums
            if max(train_nums) < era_num < min(val_nums)
        }
        if len(purge_buffer) != self._split.purge_eras:
            raise ValueError(
                f"Fold {fold.index} has incorrect purge buffer size: "
                f"expected {self._split.purge_eras}, got {len(purge_buffer)}"
            )
        if purge_buffer & train_set or purge_buffer & val_set:
            raise ValueError(f"Fold {fold.index} leaked purge eras into a split")
