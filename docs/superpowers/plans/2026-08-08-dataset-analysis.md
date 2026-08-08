# Dataset Refresh & Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a round-aware Numerai data refresh (`refresh_data.py` + `nmr/refresh.py`), a deterministic full-universe dataset analysis (`nmr/analysis.py` + `analyze_dataset.py`), and an LLM-optimized Markdown report renderer (`render_dataset_report.py`), per the approved design spec.

**Architecture:** Pure decision/policy logic lives in tested `nmr/` modules; three root scripts are thin control planes (argparse + wiring + I/O). Analysis statistics are era-partitioned and equal-era-weighted, matching the repo's per-era-first scoring convention. The renderer formats machine dumps (`artifacts/reports/dataset_analysis/`) into a dense, schema-annotated Markdown report under `docs/04-research/`.

**Tech Stack:** Python 3.11+ (`.venv`), Polars (data layer), NumPy/SciPy (moments, rank-gaussianization, Spearman), `numerapi` (refresh script only), stdlib `argparse`/`csv`/`re`.

**Spec:** `docs/superpowers/specs/2026-08-08-dataset-analysis-design.md` (read it — it is the contract; this plan is the how).

## Global Constraints

- **Tested boundary:** all business logic in `nmr/`; root scripts (`refresh_data.py`, `analyze_dataset.py`, `render_dataset_report.py`) contain only argument parsing, wiring, and printing. Never put a formula or validation rule in a script.
- **Determinism:** no wall-clock timestamps, no absolute paths in any output/dump/manifest that could feed a hash. `generated_at` in the report manifest is informational only and never enters `run_id` / `canonical_scorecards_bytes()`.
- **No new dependencies:** stdlib + NumPy/SciPy/Polars only in `nmr/`; `numerapi` is imported **only** in `refresh_data.py`, never inside `nmr/`. `scipy.stats` (rankdata, norm.ppf, skew/kurtosis) is an existing dependency.
- **Atomic writes:** all file writes that replace existing artifacts go through temp-file + `os.replace`. `nmr/_atomicio.atomic_write_text(path, text)` exists for text; parquet/JSON dumps in scripts use temp + `os.replace` helpers defined in the script (thin, no business logic).
- **No magic values:** thresholds as module-level named constants (`REGIME_LOW_PCT = 10.0`, `REGIME_HIGH_PCT = 90.0`, `IC_VOL_WINDOW = 20` in `nmr/analysis.py`; `DEFAULT_MIN_MEAN_CORR`/`DEFAULT_MAX_ABS_DECAY` already exist in `nmr/features.py`).
- **TDD:** write the failing test first, run it to confirm failure, implement, run to confirm pass, then commit. Run the full suite after every task: `./.venv/Scripts/python -m pytest -q` (Windows Git Bash; 413 tests must stay green).
- **Exports discipline:** every new public function is added to `nmr/__init__.py` **imports AND `__all__`**. Private helpers start with `_`.
- **Never:** import from or modify `../numer-AI/`; read or print `.env`; add third-party deps; commit `artifacts/` machine outputs unless the task says so (report Markdown under `docs/04-research/` IS committed).
- **Commit style** (from repo history): `feat:`, `fix:`, `refactor:`, `docs:`, `test:` prefixes, lowercase, imperative.
- **Windows:** the Bash tool runs Git Bash. Venv python is `./.venv/Scripts/python`. `data/` files are multi-GB — never delete them.
- **Era convention:** era labels are zero-padded strings (`"0001"`, `"1208"`); ordering/`min`/`max` by string is wrong for cross-era math — parse `int(era)` when ordering matters. `live.parquet` era column value is `"X"`.

---

## Phase 0 — `_per_era_pearson` extraction (micro-commit)

### Task 1: Extract per-era Pearson helper from `feature_stability_screen`

**Files:**
- Modify: `nmr/features.py:92-103` (the `partition_by` loop inside `feature_stability_screen`)
- Test: `tests/test_features_extraction.py` (new)

**Interfaces:**
- Consumes: existing `_feature_target_pearson(features: np.ndarray, target: np.ndarray) -> np.ndarray` (features.py:162, unchanged).
- Produces: `_per_era_pearson(frame: pl.DataFrame, feature_cols: Sequence[str], target_col: str, era_col: str) -> tuple[dict[str, np.ndarray], set[str]]` — returns `(corrs_by_era, degenerate_eras)` where `corrs_by_era[era]` is a 1-D array of Pearson CORR for each feature in `feature_cols`, and `degenerate_eras` is the set of era labels where the zero-vector branch fired (<2 non-null rows OR zero-variance target). **Behavior of `feature_stability_screen` must not change.**

- [ ] **Step 1: Write the failing test**

Create `tests/test_features_extraction.py`:

```python
"""Phase 0: _per_era_pearson extraction — single source of truth for per-era IC."""

from __future__ import annotations

import numpy as np
import polars as pl

from nmr.features import _per_era_pearson, feature_stability_screen


def _frame() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    eras = ["0001", "0002", "0003", "0004"]
    rows: list[dict[str, float | str]] = []
    for e in eras:
        for _ in range(10):
            rows.append(
                {
                    "era": e,
                    "feature_alpha": float(rng.normal()),
                    "feature_beta": float(rng.normal()),
                    "target": float(rng.normal()),
                }
            )
    return pl.DataFrame(rows)


def test_per_era_pearson_shapes_and_values() -> None:
    frame = _frame()
    corrs, degenerate = _per_era_pearson(
        frame, ["feature_alpha", "feature_beta"], "target", "era"
    )
    assert set(corrs) == {"0001", "0002", "0003", "0004"}
    assert degenerate == set()
    for era, vec in corrs.items():
        assert vec.shape == (2,)
        part = frame.filter(pl.col("era") == era)
        a = part["feature_alpha"].cast(pl.Float64).to_numpy()
        t = part["target"].cast(pl.Float64).to_numpy()
        assert np.isclose(vec[0], np.corrcoef(a, t)[0, 1], atol=1e-12)


def test_per_era_pearson_degenerate_eras() -> None:
    frame = pl.DataFrame(
        {
            "era": ["0001", "0002", "0002", "0002", "0002"],
            "feature_alpha": [1.0, 2.0, 1.0, 1.0, 1.0],
            "feature_beta": [3.0, 4.0, 5.0, 5.0, 5.0],
            "target": [0.1, 1.0, 1.0, 1.0, 1.0],
        }
    )
    corrs, degenerate = _per_era_pearson(
        frame, ["feature_alpha", "feature_beta"], "target", "era"
    )
    assert "0001" in degenerate  # single row -> <2 rows
    assert np.array_equal(corrs["0001"], np.zeros(2))
    assert "0002" in degenerate  # constant target -> zero variance
    assert np.array_equal(corrs["0002"], np.zeros(2))


def test_screen_uses_extracted_helper_single_source_of_truth() -> None:
    frame = _frame()
    screen = feature_stability_screen(
        frame, feature_cols=["feature_alpha", "feature_beta"], target_col="target"
    )
    corrs, _ = _per_era_pearson(
        frame, ["feature_alpha", "feature_beta"], "target", "era"
    )
    eras = sorted(corrs, key=int)
    matrix = np.column_stack([corrs[e] for e in eras])
    for i, feature in enumerate(["feature_alpha", "feature_beta"]):
        row = screen.filter(pl.col("feature") == feature)
        assert np.isclose(
            float(row["mean_corr"][0]), float(np.mean(matrix[i])), atol=1e-12
        )
        assert float(row["n_eras"][0]) == len(eras)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_features_extraction.py -q`
Expected: FAIL with `ImportError: cannot import name '_per_era_pearson'`.

- [ ] **Step 3: Implement the extraction**

Edit `nmr/features.py` — add this private function (place it right before `feature_stability_screen` or after `_feature_target_pearson`; module-private):

```python
def _per_era_pearson(
    frame: pl.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    era_col: str,
) -> tuple[dict[str, np.ndarray], set[str]]:
    """Per-era Pearson CORR of each feature vs ``target_col``, keyed by era label.

    Returns ``(corrs_by_era, degenerate_eras)``. A degenerate era (fewer than
    2 non-null rows, or a constant target) contributes a zero vector and its
    label is recorded in ``degenerate_eras``. This is the single implementation
    of per-era feature-target correlation: both ``feature_stability_screen``
    and ``nmr.analysis.feature_ic_by_era`` route through it.
    """
    feature_list = list(feature_cols)
    per_era: dict[str, np.ndarray] = {}
    degenerate: set[str] = set()
    for part in frame.select([era_col, target_col, *feature_list]).partition_by(
        era_col, maintain_order=True
    ):
        era = str(part.get_column(era_col).to_list()[0])
        clean = part.drop_nulls()
        target = clean.get_column(target_col).cast(pl.Float64).to_numpy()
        zero_target_variance = target.size > 0 and bool(np.all(target == target[0]))
        if clean.height < 2 or target.size == 0 or zero_target_variance:
            per_era[era] = np.zeros(len(feature_list), dtype=float)
            degenerate.add(era)
            continue
        features = clean.select(feature_list).cast(pl.Float64).to_numpy()
        per_era[era] = _feature_target_pearson(features, target)
    return per_era, degenerate
```

Then replace the loop body inside `feature_stability_screen` (features.py:92-103) with:

```python
    per_era, _ = _per_era_pearson(frame, feature_list, target_col, era_col)
```

Delete the now-unused inline loop. **Do not touch any other line of the screen function** — the aggregation below (`eras = sorted(per_era, key=int)`, rows building) stays identical.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_features_extraction.py tests/test_features.py -q`
Expected: PASS (new tests + all existing feature tests).

- [ ] **Step 5: Full suite + commit**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: 413+ tests PASS.
Then:
```bash
git add nmr/features.py tests/test_features_extraction.py
git commit -m "refactor(features): extract _per_era_pearson single source of truth"
```

---

## Phase 1 — Data refresh

### Task 2: `nmr/refresh.py` — version constants and comparison

**Files:**
- Create: `nmr/refresh.py`
- Test: `tests/test_refresh.py` (new)

**Interfaces:**
- Consumes: `nmr.config.REPO_ROOT` and `nmr.config.load_config` (drift-guard test).
- Produces:
  - `CURRENT_DATA_VERSION: str = "v5.2"` (module constant)
  - `_parse_version(v: str) -> tuple[int, int]` (private; raises `ValueError` on non-`v<major>.<minor>`)
  - `detect_newer_version(available: Sequence[str], current: str) -> str | None`
- Later tasks consume: `CURRENT_DATA_VERSION` (Task 4 script default, Task 15 renderer validation), `detect_newer_version` (Task 4).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_refresh.py`:

```python
"""Pure logic for the round-aware Numerai data refresh (nmr/refresh.py)."""

from __future__ import annotations

import pytest

from nmr.config import REPO_ROOT, load_config
from nmr.refresh import (
    CURRENT_DATA_VERSION,
    _parse_version,
    detect_newer_version,
)


def test_current_version_is_v5_2() -> None:
    assert CURRENT_DATA_VERSION == "v5.2"


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_refresh.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'nmr.refresh'`.

- [ ] **Step 3: Implement the module**

Create `nmr/refresh.py`:

```python
"""Round-aware Numerai dataset refresh policy — pure logic, no I/O, no numerapi.

``refresh_data.py`` performs all downloads and file I/O; this module only
decides *what* to do, given facts the script has already gathered (round
numbers, era ranges, available versions). Deterministic: same inputs, same
outputs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

__all__ = [
    "CURRENT_DATA_VERSION",
    "STATIC_FILES",
    "LIVE_FRESH_FILES",
    "EXPANDING_FILES",
    "detect_newer_version",
    "needs_live_refresh",
    "build_era_manifest",
    "classify_refresh_plan",
]

# The data version this repo's pipeline targets. Drift-guarded by a test
# asserting equality with configs/first_model.yaml's data.version.
CURRENT_DATA_VERSION = "v5.2"

# Files that change only when Numerai ships a new data version.
STATIC_FILES = ("features.json", "train.parquet", "train_benchmark_models.parquet")

# Files that change with every tournament round.
LIVE_FRESH_FILES = (
    "live.parquet",
    "live_benchmark_models.parquet",
    "live_example_preds.parquet",
    "live_example_preds.csv",
)

# Files that expand weekly as new validation eras are published.
EXPANDING_FILES = (
    "validation.parquet",
    "validation_benchmark_models.parquet",
    "validation_example_preds.parquet",
    "validation_example_preds.csv",
    "meta_model.parquet",
)

_VERSION_RE = re.compile(r"^v(\d+)\.(\d+)$")


def _parse_version(v: str) -> tuple[int, int]:
    """Parse ``v<major>.<minor>`` into integers, or raise ``ValueError``."""
    match = _VERSION_RE.match(v)
    if match is None:
        raise ValueError(
            f"Unrecognized dataset version {v!r}: expected 'v<major>.<minor>' "
            "(e.g. 'v5.2'); patch components are not supported"
        )
    return int(match.group(1)), int(match.group(2))


def detect_newer_version(available: Sequence[str], current: str) -> str | None:
    """Return the numerically-greatest version in ``available`` that strictly
    exceeds ``current``, or ``None``. Malformed entries raise (fail loudly —
    a strange filename in the API listing is a real signal)."""
    current_parsed = _parse_version(current)
    newest: tuple[tuple[int, int], str] | None = None
    for item in available:
        parsed = _parse_version(item)
        if parsed > current_parsed and (newest is None or parsed > newest[0]):
            newest = (parsed, item)
    return newest[1] if newest is not None else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_refresh.py -q`
Expected: PASS (all 8 tests).

- [ ] **Step 5: Add exports to package root**

Edit `nmr/__init__.py` — add only the names that exist in this task (Task 3 extends both blocks):

```python
from .refresh import CURRENT_DATA_VERSION, detect_newer_version
```

and in `__all__` (alphabetical position): `"CURRENT_DATA_VERSION"`, `"detect_newer_version"`.

- [ ] **Step 6: Full suite + commit**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS.
```bash
git add nmr/refresh.py nmr/__init__.py tests/test_refresh.py
git commit -m "feat(refresh): version detection and drift guard"
```

### Task 3: `nmr/refresh.py` — refresh decision functions

**Files:**
- Modify: `nmr/refresh.py`, `nmr/__init__.py`, `tests/test_refresh.py`

**Interfaces:**
- Consumes: `STATIC_FILES`, `LIVE_FRESH_FILES`, `EXPANDING_FILES` (Task 2).
- Produces:
  - `needs_live_refresh(current_round: int, last_recorded: int | None, live_exists: bool) -> bool`
  - `build_era_manifest(era_ranges: Mapping[str, tuple[str | None, str | None]], round_id: int, today: str) -> list[dict[str, str | int | None]]` — rows with keys `date, dataset, start_era, end_era, round_id`; raises `ValueError` if a **non-live** dataset has `(None, None)`; live `(None, None)` is valid.
  - `classify_refresh_plan(round_advanced: bool, existing: set[str], live_only: bool = False) -> dict[str, str]` — per-file `"refresh" | "ensure" | "skip"`.
- Later tasks consume: all three in Task 4.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_refresh.py`:

```python
from nmr.refresh import (
    EXPANDING_FILES,
    LIVE_FRESH_FILES,
    STATIC_FILES,
    build_era_manifest,
    classify_refresh_plan,
    needs_live_refresh,
)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_refresh.py -q`
Expected: FAIL with `ImportError: cannot import name 'needs_live_refresh'`.

- [ ] **Step 3: Implement the three functions**

Append to `nmr/refresh.py`:

```python
def needs_live_refresh(
    current_round: int, last_recorded: int | None, live_exists: bool
) -> bool:
    """True when live.parquet must be re-downloaded.

    Reconciles on any mismatch (stale *or* ahead-of-remote marker); the file
    must exist *and* match the current round.
    """
    return not live_exists or last_recorded is None or last_recorded != current_round


def build_era_manifest(
    era_ranges: Mapping[str, tuple[str | None, str | None]],
    round_id: int,
    today: str,
) -> list[dict[str, str | int | None]]:
    """Build refresh-ledger rows for ``numerai_era_data.csv``.

    ``era_ranges`` maps dataset name to ``(min_era, max_era)`` as read from the
    parquet ``era`` column. Live rounds are unlabeled: ``(None, None)`` is
    valid for ``live`` (the script serializes it to ``"X"``); any other dataset
    with an empty range raises — an empty parquet is a real error.
    """
    rows: list[dict[str, str | int | None]] = []
    for dataset in ("train", "validation", "live"):
        start, end = era_ranges[dataset]
        if start is None or end is None:
            if dataset != "live":
                raise ValueError(
                    f"{dataset} parquet has no era range (empty file): {(start, end)!r}"
                )
        rows.append(
            {
                "date": today,
                "dataset": dataset,
                "start_era": start,
                "end_era": end,
                "round_id": round_id if dataset == "live" else None,
            }
        )
    return rows


def classify_refresh_plan(
    round_advanced: bool,
    existing: set[str],
    live_only: bool = False,
) -> dict[str, str]:
    """Per-file refresh decision.

    Returns ``{filename: "refresh" | "ensure" | "skip"}``:
    - ``refresh`` — download now (round advanced, or the file is missing);
    - ``ensure``  — download only if missing (static files);
    - ``skip``    — already present and no trigger (or skipped by ``--live-only``).
    """
    plan: dict[str, str] = {}
    for name in STATIC_FILES:
        plan[name] = "ensure"
    for name in LIVE_FRESH_FILES:
        plan[name] = "refresh" if (round_advanced or name not in existing) else "skip"
    for name in EXPANDING_FILES:
        if live_only:
            plan[name] = "skip"
        else:
            plan[name] = "refresh" if (round_advanced or name not in existing) else "skip"
    return plan
```

- [ ] **Step 4: Extend package exports**

Edit `nmr/__init__.py`: change the refresh import to the full set and add the names to `__all__`:

```python
from .refresh import (
    CURRENT_DATA_VERSION,
    EXPANDING_FILES,
    LIVE_FRESH_FILES,
    STATIC_FILES,
    build_era_manifest,
    classify_refresh_plan,
    detect_newer_version,
    needs_live_refresh,
)
```

Add to `__all__` (sorted): `"CURRENT_DATA_VERSION"`, `"EXPANDING_FILES"`, `"LIVE_FRESH_FILES"`, `"STATIC_FILES"`, `"build_era_manifest"`, `"classify_refresh_plan"`, `"detect_newer_version"`, `"needs_live_refresh"`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_refresh.py -q`
Expected: PASS (all tests).

- [ ] **Step 6: Full suite + commit**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS.
```bash
git add nmr/refresh.py nmr/__init__.py tests/test_refresh.py
git commit -m "feat(refresh): round/era manifest and file refresh policy"
```

### Task 4: `refresh_data.py` — thin control-plane script

**Files:**
- Create: `refresh_data.py` (repo root)
- Test: `tests/test_refresh_script.py` (new)

**Interfaces:**
- Consumes: `nmr.refresh` (`CURRENT_DATA_VERSION`, `detect_newer_version`, `needs_live_refresh`, `build_era_manifest`, `classify_refresh_plan`), `nmr._atomicio.atomic_write_text`, `nmr.features.resolve_feature_sets`, `numerapi.NumerAPI`.
- Produces: `main(argv: list[str] | None = None) -> int` (exit code contract below) and the era CSV update. Testable via monkeypatched `NumerAPI` + `tmp_path`.

**Exit-code contract** (from the spec §3.4):

| Mode | Newer version | Files need refresh | Behavior / exit |
|---|---|---|---|
| default | yes | — | `[WARNING]` banner, proceed, exit 0 |
| `--check-only` | yes | — | no writes, exit 3 |
| `--check-only` | no | yes | no writes, print plan, exit 3 |
| `--check-only` | no | no | "everything current", exit 0 |
| `--strict` | yes | — | abort before downloads, exit 3 |
| `--dry-run` | any | — | print plan + banner, no writes, exit 0 |
| network/parse/integrity failure | — | — | loud message, exit 1 |

- [ ] **Step 1: Write the failing tests**

Create `tests/test_refresh_script.py`:

```python
"""Integration tests for refresh_data.py with a mocked NumerAPI."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import refresh_data


class FakeNapi:
    """Records download_dataset calls; serves fixed API responses."""

    def __init__(
        self,
        *,
        round_num: int | None = 1294,
        datasets: list[str] | None = None,
        fail_on: str | None = None,
    ) -> None:
        self.round_num = round_num
        self.datasets = datasets or [
            f"v5.2/{name}"
            for name in (
                "features.json",
                "train.parquet",
                "validation.parquet",
                "live.parquet",
                "train_benchmark_models.parquet",
                "validation_benchmark_models.parquet",
                "live_benchmark_models.parquet",
                "meta_model.parquet",
                "live_example_preds.parquet",
                "live_example_preds.csv",
                "validation_example_preds.parquet",
                "validation_example_preds.csv",
            )
        ]
        self.fail_on = fail_on
        self.downloads: list[tuple[str, str]] = []

    def get_current_round(self) -> int | None:
        return self.round_num

    def list_datasets(self) -> list[str]:
        return list(self.datasets)

    def download_dataset(self, filename: str, dest_path: str | Path) -> None:
        self.downloads.append((filename, str(dest_path)))
        if self.fail_on is not None and self.fail_on in filename:
            raise ConnectionError(f"simulated failure on {filename}")
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith("features.json"):
            dest.write_text(
                json.dumps(
                    {
                        "feature_sets": {"small": ["f1", "f2"], "medium": ["f1", "f2"]},
                        "targets": ["target"],
                    }
                ),
                encoding="utf-8",
            )
        elif filename.endswith(".csv"):
            dest.write_text("id,era\nn1,0001\n", encoding="utf-8")
        else:
            is_live = filename.endswith("live.parquet")
            pl.DataFrame(
                {
                    "era": ["X", "X"] if is_live else ["0001", "0002"],
                    "id": ["n1", "n2"],
                    "target": [0.0, 1.0],
                }
            ).write_parquet(dest)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _fake_napi(monkeypatch: pytest.MonkeyPatch) -> FakeNapi:
    fake = FakeNapi()

    class _Napi:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def get_current_round(self) -> int | None:
            return fake.get_current_round()

        def list_datasets(self) -> list[str]:
            return fake.list_datasets()

        def download_dataset(self, filename: str, dest_path: str | Path) -> None:
            fake.download_dataset(filename, dest_path)

    monkeypatch.setattr(refresh_data.numerapi, "NumerAPI", _Napi)
    return fake


def _era_csv(data_dir: Path) -> Path:
    return data_dir / "numerai_era_data.csv"


def test_dry_run_writes_nothing(data_dir: Path, _fake_napi: FakeNapi) -> None:
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir)), "--dry-run"]
    )
    assert rc == 0
    assert _fake_napi.downloads == []
    assert not _era_csv(data_dir).exists()


def test_fresh_refresh_writes_manifest(data_dir: Path, _fake_napi: FakeNapi) -> None:
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir))]
    )
    assert rc == 0
    downloaded = {Path(f).name for f, _ in _fake_napi.downloads}
    assert "live.parquet" in downloaded
    assert "validation.parquet" in downloaded
    assert "features.json" in downloaded
    csv_text = _era_csv(data_dir).read_text(encoding="utf-8")
    assert "live" in csv_text and "1294" in csv_text
    assert "X" in csv_text  # live era serialization


def test_no_refresh_when_up_to_date(data_dir: Path, _fake_napi: FakeNapi) -> None:
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir))]
    )
    assert rc == 0
    n_downloads = len(_fake_napi.downloads)
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir))]
    )
    assert rc == 0
    # second run: round unchanged, all files present -> no new downloads
    assert len(_fake_napi.downloads) == n_downloads


def test_failed_download_does_not_write_csv(
    data_dir: Path, _fake_napi: FakeNapi
) -> None:
    _fake_napi.fail_on = "validation.parquet"
    with pytest.raises(ConnectionError):
        refresh_data.main(
            ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir))]
        )
    assert not _era_csv(data_dir).exists()


def test_none_round_aborts(data_dir: Path, _fake_napi: FakeNapi) -> None:
    _fake_napi.round_num = None
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir))]
    )
    assert rc == 1
    assert _fake_napi.downloads == []


def test_check_only_newer_version_exit_3(data_dir: Path, _fake_napi: FakeNapi) -> None:
    _fake_napi.datasets.append("v5.3/live.parquet")
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir)), "--check-only"]
    )
    assert rc == 3
    assert _fake_napi.downloads == []


def test_check_only_all_current_exit_0(data_dir: Path, _fake_napi: FakeNapi) -> None:
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir))]
    )
    assert rc == 0
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir)), "--check-only"]
    )
    assert rc == 0


def test_strict_newer_version_aborts(data_dir: Path, _fake_napi: FakeNapi) -> None:
    _fake_napi.datasets.append("v5.3/live.parquet")
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir)), "--strict"]
    )
    assert rc == 3
    assert _fake_napi.downloads == []


def test_live_only_skips_expanding(data_dir: Path, _fake_napi: FakeNapi) -> None:
    rc = refresh_data.main(
        [
            "--data-dir", str(data_dir),
            "--era-csv", str(_era_csv(data_dir)),
            "--live-only",
        ]
    )
    assert rc == 0
    downloaded = {Path(f).name for f, _ in _fake_napi.downloads}
    assert "live.parquet" in downloaded
    assert "validation.parquet" not in downloaded


def test_csv_round_trip_matches_legacy_format(data_dir: Path, _fake_napi: FakeNapi) -> None:
    import pandas as pd

    refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir))]
    )
    df = pd.read_csv(_era_csv(data_dir))
    live = df[df["dataset"] == "live"].iloc[0]
    assert live["round_id"] == 1294.0  # legacy float format
    assert live["start_era"] == "X" and live["end_era"] == "X"
    train = df[df["dataset"] == "train"].iloc[0]
    assert pd.isna(train["round_id"])  # empty serialization
    assert train["start_era"] == "0001" and train["end_era"] == "0002"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_refresh_script.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'refresh_data'`.

- [ ] **Step 3: Implement the script**

Create `refresh_data.py` (repo root):

```python
"""Round-aware Numerai dataset refresh — thin control plane.

Downloads/updates data/v5.2 assets via the public Numerai API and maintains
data/numerai_era_data.csv (the refresh ledger). All decision logic lives in
``nmr/refresh.py``; this script only wires numerapi calls, file I/O, and
argument parsing.

Exit codes: 0 = ok (or advisory-only warning), 1 = hard failure,
3 = gate tripped (--check-only/--strict). See docs/superpowers/specs/
2026-08-08-dataset-analysis-design.md §3.4.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import polars as pl

from nmr._atomicio import atomic_write_text
from nmr.features import resolve_feature_sets
from nmr.refresh import (
    CURRENT_DATA_VERSION,
    build_era_manifest,
    classify_refresh_plan,
    detect_newer_version,
    needs_live_refresh,
)

import numerapi

_VERSION_ALERT = (
    "[WARNING] New Numerai data version detected: {newer} is available. "
    "This repo's pipeline targets {current}. Consider migrating before the "
    "next campaign; continuing with {current}."
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh Numerai v5.2 datasets and the era ledger."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--version", default=CURRENT_DATA_VERSION)
    parser.add_argument(
        "--era-csv",
        type=Path,
        default=Path("data") / "numerai_era_data.csv",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live-only", action="store_true")
    return parser.parse_args(argv)


def _read_last_live_round(era_csv: Path) -> int | None:
    if not era_csv.exists():
        return None
    df = pl.read_csv(era_csv, try_parse_dates=False)
    live = df.filter(pl.col("dataset") == "live")
    if live.is_empty() or "round_id" not in live.columns:
        return None
    rounds = live.get_column("round_id").drop_nulls().cast(pl.Int64)
    return None if rounds.is_empty() else int(rounds.max())


def _era_range(path: Path) -> tuple[str | None, str | None]:
    """Read (min_era, max_era) from a parquet file; None when unreadable."""
    if not path.exists():
        return None, None
    try:
        agg = (
            pl.scan_parquet(path)
            .select(
                pl.col("era").min().alias("min_era"),
                pl.col("era").max().alias("max_era"),
            )
            .collect()
        )
        return agg.row(0)
    except Exception:
        return None, None


def _validate_and_swap(name: str, tmp: Path, target: Path) -> None:
    """Integrity-check a downloaded temp file, then atomically swap it in."""
    if name == "features.json":
        sets = resolve_feature_sets(tmp)  # raises on malformed/empty feature_sets
        if not sets:
            raise ValueError(f"{name}: feature_sets is empty after validation")
        raw = json.loads(tmp.read_text(encoding="utf-8"))
        if not raw.get("targets"):
            raise ValueError(f"{name}: 'targets' list is missing or empty")
    elif name.endswith(".parquet"):
        pl.scan_parquet(tmp).collect_schema()  # raises on truncated/corrupt parquet
    else:  # example-pred CSV
        if tmp.stat().st_size == 0:
            raise ValueError(f"{name}: downloaded file is empty")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp.replace(target)


def _manifest_to_csv(records: Sequence[dict[str, str | int | None]]) -> str:
    """Serialize manifest rows to the legacy ledger format (CRLF)."""
    out = ["date,dataset,start_era,end_era,round_id"]
    for rec in records:
        start = "X" if rec["start_era"] is None else rec["start_era"]
        end = "X" if rec["end_era"] is None else rec["end_era"]
        round_id = rec["round_id"]
        rid = "" if round_id is None else str(float(round_id))
        out.append(f"{rec['date']},{rec['dataset']},{start},{end},{rid}")
    return "\r\n".join(out) + "\r\n"


def _load_existing_records(
    era_csv: Path, today: str
) -> list[dict[str, str | int | None]]:
    """Existing ledger rows minus today's (they are rebuilt fresh)."""
    if not era_csv.exists():
        return []
    df = pl.read_csv(era_csv, try_parse_dates=False)
    df = df.filter(pl.col("date") != today)
    return [
        {
            "date": str(row["date"]),
            "dataset": str(row["dataset"]),
            "start_era": None if row["start_era"] == "X" else str(row["start_era"]),
            "end_era": None if row["end_era"] == "X" else str(row["end_era"]),
            "round_id": (
                int(float(row["round_id"]))
                if row["round_id"] not in (None, "")
                else None
            ),
        }
        for row in df.iter_rows(named=True)
    ]


def _refresh(
    napi: object,
    args: argparse.Namespace,
    version: str,
    round_num: int,
    plan: dict[str, str],
) -> None:
    version_dir = args.data_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    # clean stale .part files from crashed runs
    for stale in version_dir.glob("*.part"):
        stale.unlink()

    for name, decision in plan.items():
        target = version_dir / name
        if decision == "ensure" and target.exists():
            continue
        if decision == "skip":
            continue
        print(f"downloading {version}/{name} ...")
        fd, tmp_name = tempfile.mkstemp(
            dir=version_dir, prefix=f"{name}.tmp.", suffix=".part"
        )
        os.close(fd)  # Windows: release the handle so os.replace/unlink can work
        tmp = Path(tmp_name)
        try:
            napi.download_dataset(f"{version}/{name}", dest_path=tmp)  # type: ignore[attr-defined]
            _validate_and_swap(name, tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink()

    # Ledger: read era ranges for every parquet present on disk (downloaded
    # this run or already present). Write the ledger only when all three
    # exist — a partial checkout (e.g. --live-only) simply skips the write.
    era_ranges: dict[str, tuple[str | None, str | None]] = {}
    for dataset in ("train", "validation", "live"):
        target = version_dir / f"{dataset}.parquet"
        if target.exists():
            era_ranges[dataset] = _era_range(target)
    if set(era_ranges) == {"train", "validation", "live"}:
        records = build_era_manifest(era_ranges, round_num, str(date.today()))
        existing_records = _load_existing_records(args.era_csv, str(date.today()))
        atomic_write_text(args.era_csv, _manifest_to_csv(existing_records + records))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    version = args.version
    napi = numerapi.NumerAPI()

    if args.dry_run:
        print("dry-run: no downloads or writes will be performed")

    # 1. current round (None -> abort)
    round_num = napi.get_current_round()
    if round_num is None:
        print("ERROR: could not determine the current tournament round", file=sys.stderr)
        return 1

    # 2. version alert
    available = napi.list_datasets()
    prefixes = sorted({f.split("/", 1)[0] for f in available})
    newer = detect_newer_version(prefixes, version)
    if newer is not None:
        print(_VERSION_ALERT.format(newer=newer, current=version))
        if args.strict:
            return 3

    # 3. plan
    last_round = _read_last_live_round(args.era_csv)
    live_exists = (args.data_dir / version / "live.parquet").exists()
    round_advanced = needs_live_refresh(round_num, last_round, live_exists)
    version_dir = args.data_dir / version
    if version_dir.exists():
        files = {p.name for p in version_dir.glob("*") if p.is_file()}
        existing = files - {p.name for p in version_dir.glob("*.part")}
    else:
        existing = set()
    plan = classify_refresh_plan(round_advanced, existing, live_only=args.live_only)

    if args.check_only:
        if newer is not None or any(v == "refresh" for v in plan.values()):
            print("check-only: refresh needed (newer version or stale files)")
            return 3
        print("everything current")
        return 0

    if args.dry_run:
        for name, decision in sorted(plan.items()):
            print(f"  {decision:>8}  {name}")
        print("dry-run complete (exit 0)")
        return 0

    # 4. execute
    _refresh(napi, args, version, round_num, plan)
    print(f"refresh complete for round {round_num}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: `existing` must be a plain set of filenames with no `.part` entries — the expression above is correct for both existing and missing version dirs.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_refresh_script.py -q`
Expected: PASS (all 10 tests).

- [ ] **Step 5: Full suite + commit**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS.
```bash
git add refresh_data.py tests/test_refresh_script.py
git commit -m "feat(refresh): round-aware dataset refresh script"
```

### Task 5: Refresh documentation (SSOT updates)

**Files:**
- Modify: `README.md`, `ARCHITECTURE.md`, `AGENTS.md`

**Interfaces:** none — pure documentation. Follow the spec §6: README owns data-asset requirements (refresh command), ARCHITECTURE.md owns the era-CSV schema, AGENTS.md toolkit table gets the new scripts.

- [ ] **Step 1: README — add "Refreshing data" subsection**

In `README.md`, in the data-assets area, add:

```markdown
### Refreshing data

Round-aware refresh of the Numerai datasets (thin script; policy in `nmr/refresh.py`):

```bash
python refresh_data.py            # round-based refresh + era ledger update
python refresh_data.py --dry-run  # print the plan, download nothing
python refresh_data.py --check-only   # exit 3 if a newer data version or stale files
```

Behavior: `live.parquet` (and live benchmarks/example preds) re-download every time the
tournament round advances; weekly-expanding files (`validation.parquet`,
`validation_benchmark_models.parquet`, `meta_model.parquet`, ...) re-download on round
advance; truly static files download only when missing. A prominent `[WARNING]` is
printed when the API lists a newer data version than the pipeline's target
(`--strict` turns it into exit code 3; `--check-only` into a status check). The era
ledger `data/numerai_era_data.csv` records per-dataset refresh dates, era ranges, and
the live round. See `docs/superpowers/specs/2026-08-08-dataset-analysis-design.md` §3.
```

- [ ] **Step 2: ARCHITECTURE.md — era-CSV schema + module entries**

In `ARCHITECTURE.md`:
- Add a short "Refresh ledger (`data/numerai_era_data.csv`)" subsection documenting the 5 columns (`date`, `dataset` ∈ {train, validation, live}, `start_era`, `end_era` — zero-padded strings, `"X"` for live — , `round_id` — float when present, empty for train/validation) and the round-based refresh triggers.
- In §3 dependency graph, add `nmr/refresh.py` (pure policy; depends on nothing inside nmr) and the `refresh_data.py` root script (depends on `nmr.refresh`, `nmr._atomicio`, `nmr.features.resolve_feature_sets`, `numerapi`).

- [ ] **Step 3: AGENTS.md — toolkit table row**

In the Agent Toolkit table, add one row:

```
| Refresh the Numerai datasets / era ledger | `nmr/refresh.py` + `refresh_data.py` |
```

- [ ] **Step 4: Verify no doc contradictions**

Run: `./.venv/Scripts/python -m pytest -q` — expected PASS (docs only).
Grep the four SSOT docs for "v5.2" and confirm the refresh doc references `CURRENT_DATA_VERSION` rather than duplicating the version string where it matters.

- [ ] **Step 5: Commit**

```bash
git add README.md ARCHITECTURE.md AGENTS.md
git commit -m "docs: document round-aware data refresh"
```

---

## Phase 2 — Analysis module (`nmr/analysis.py`)

### Task 6: `describe_splits` + `era_structure` (module scaffold)

**Files:**
- Create: `nmr/analysis.py`
- Test: `tests/test_analysis.py` (new)

**Interfaces:**
- Consumes: nothing (stdlib + polars + numpy only).
- Produces:
  - `SplitStats` frozen dataclass: `n_rows: int`, `n_eras: int`, `min_era: str`, `max_era: str`, `rows_per_era_min: int`, `rows_per_era_median: float`, `rows_per_era_max: int`, `rows_per_era_mean: float`, `rows_per_era_std: float`, `n_ids: int`
  - `describe_splits(splits: Mapping[str, pl.DataFrame]) -> dict[str, SplitStats]`
  - `era_structure(frame: pl.DataFrame, era_col: str = "era") -> pl.DataFrame` — columns `era, era_index, n_rows, n_ids, gap` (sorted by `era_index`; `gap=True` when `era_index != prev + 1`; first row `gap=False`; `n_ids` null when no `id` column)
- Later tasks append to this module; `nmr/__init__.py` exports are added in Task 13.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_analysis.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_analysis.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'nmr.analysis'`.

- [ ] **Step 3: Implement the module scaffold**

Create `nmr/analysis.py`:

```python
"""Deterministic dataset analysis for research reports.

Era-aware statistics over train/validation frames: split shapes, era
structure, target profiles, feature-target IC, feature moments, feature
correlation structure, regimes, and benchmark context. Pure functions: frames
in, frames/dicts out — no I/O, no wall-clock, no stochastic operations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.stats

__all__ = [
    "SplitStats",
    "describe_splits",
    "era_structure",
    "target_profile",
    "target_correlation_matrix",
    "feature_ic_screen",
    "feature_ic_by_era",
    "feature_summary",
    "FeatureCorrResult",
    "feature_correlation_structure",
    "within_set_redundancy",
    "cross_set_membership",
    "regime_analysis",
    "benchmark_era_corr",
]

REGIME_LOW_PCT = 10.0
REGIME_HIGH_PCT = 90.0
IC_VOL_WINDOW = 20


@dataclass(frozen=True)
class SplitStats:
    """Shape statistics for one dataset split."""

    n_rows: int
    n_eras: int
    min_era: str
    max_era: str
    rows_per_era_min: int
    rows_per_era_median: float
    rows_per_era_max: int
    rows_per_era_mean: float
    rows_per_era_std: float
    n_ids: int


def describe_splits(splits: Mapping[str, pl.DataFrame]) -> dict[str, SplitStats]:
    """Per-split shape statistics. Requires an ``id`` column in each frame."""
    out: dict[str, SplitStats] = {}
    for name, frame in splits.items():
        if "id" not in frame.columns:
            raise ValueError(f"split {name!r} missing required column 'id'")
        per_era = frame.group_by("era").len()
        counts = per_era.get_column("len").to_numpy()
        eras = sorted(per_era.get_column("era").to_list(), key=int)
        out[name] = SplitStats(
            n_rows=frame.height,
            n_eras=len(eras),
            min_era=eras[0],
            max_era=eras[-1],
            rows_per_era_min=int(counts.min()),
            rows_per_era_median=float(np.median(counts)),
            rows_per_era_max=int(counts.max()),
            rows_per_era_mean=float(counts.mean()),
            rows_per_era_std=float(counts.std(ddof=0)),
            n_ids=int(frame.get_column("id").n_unique()),
        )
    return out


def era_structure(frame: pl.DataFrame, era_col: str = "era") -> pl.DataFrame:
    """Per-era row/id counts with era-index gap detection (sorted by int era)."""
    if era_col not in frame.columns:
        raise ValueError(f"frame missing required column {era_col!r}")
    if frame.is_empty():
        raise ValueError("frame is empty: cannot compute era structure")
    n_ids = (
        pl.col("id").n_unique().alias("n_ids")
        if "id" in frame.columns
        else pl.lit(None, dtype=pl.Int64).alias("n_ids")
    )
    per = frame.group_by(era_col).agg(pl.len().alias("n_rows"), n_ids)
    per = per.with_columns(
        pl.col(era_col).cast(pl.Int64).alias("era_index")
    ).sort("era_index")
    gap = (
        per.select(
            (pl.col("era_index") != pl.col("era_index").shift(1) + 1)
            .fill_null(False)
            .alias("gap")
        ).get_column("gap")
    )
    return per.select(
        pl.col(era_col).alias("era"), "era_index", "n_rows", "n_ids", gap
    )
```

Note: `era` labels are zero-padded strings that parse as ints (`"0001"` → 1). If a label is not int-parseable (e.g. `"X"`), `cast(pl.Int64)` raises — this function is for train/validation only.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_analysis.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Full suite + commit**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS.
```bash
git add nmr/analysis.py tests/test_analysis.py
git commit -m "feat(analysis): split descriptors and era structure"
```

### Task 7: `target_profile` + `target_correlation_matrix`

**Files:**
- Modify: `nmr/analysis.py`, `tests/test_analysis.py`

**Interfaces:**
- Consumes: `scipy.stats.rankdata` (existing dep).
- Produces:
  - `target_profile(frame, target_cols: Sequence[str], era_col: str = "era") -> pl.DataFrame` — one row per target: `target, n_eras_present, missing_rate, era_mean_mean, era_mean_std, pooled_mean, pooled_std, pooled_skew, pooled_kurtosis, min, max, zero_variance_era_count`
  - `target_correlation_matrix(frame, target_cols, era_col="era") -> pl.DataFrame` — long-form `target_a, target_b, mean_corr, n_eras` for sorted pairs `a < b`; per-era Spearman, equal-era-weighted mean over observed eras only.
- Semantics: non-finite target rows dropped before moments; `missing_rate = 1 - n_finite/n_total`; an era counts toward `n_eras_present` if ≥1 valid value; `zero_variance_era_count` counts eras with ≥2 valid values and zero variance; correlation pairs skip eras with <2 valid rows or zero variance (recorded via `n_eras`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analysis.py`:

```python
from nmr.analysis import target_correlation_matrix, target_profile


def test_target_profile_moments() -> None:
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    for e in range(3):
        era = f"{e + 1:04d}"
        for v in rng.normal(size=100):
            rows.append({"era": era, "target": float(v)})
    frame = pl.DataFrame(rows)
    out = target_profile(frame, ["target"])
    assert len(out) == 1
    row = out.row(0, named=True)
    series = frame["target"].to_numpy()
    assert row["n_eras_present"] == 3
    assert np.isclose(row["pooled_mean"], float(series.mean()), atol=1e-12)
    assert np.isclose(row["pooled_std"], float(series.std(ddof=0)), atol=1e-12)
    assert np.isclose(
        row["pooled_skew"], float(scipy.stats.skew(series)), atol=1e-12
    )
    assert np.isclose(
        row["pooled_kurtosis"],
        float(scipy.stats.kurtosis(series, fisher=True)),
        atol=1e-12,
    )
    assert row["missing_rate"] == 0.0
    assert row["zero_variance_era_count"] == 0


def test_target_profile_non_finite_dropped() -> None:
    frame = pl.DataFrame(
        {
            "era": ["0001", "0001", "0001", "0002", "0002", "0002"],
            "target": [1.0, float("nan"), 3.0, None, 5.0, 6.0],
        }
    )
    out = target_profile(frame, ["target"])
    row = out.row(0, named=True)
    assert row["n_eras_present"] == 2
    assert np.isclose(row["missing_rate"], 2 / 6)
    assert np.isclose(row["pooled_mean"], (1.0 + 3.0 + 5.0 + 6.0) / 4)


def test_target_profile_zero_variance_era_counted() -> None:
    frame = pl.DataFrame(
        {
            "era": ["0001", "0001", "0002", "0002", "0002"],
            "target": [5.0, 5.0, 1.0, 2.0, 3.0],
        }
    )
    out = target_profile(frame, ["target"])
    assert out.row(0, named=True)["zero_variance_era_count"] == 1


def test_target_correlation_matrix_hand_computed() -> None:
    rng = np.random.default_rng(3)
    rows: list[dict[str, object]] = []
    for e in range(4):
        era = f"{e + 1:04d}"
        for i in range(20):
            a = float(rng.normal())
            b = 2.0 * a + float(rng.normal(scale=0.5))
            rows.append({"era": era, "target_alpha": a, "target_beta": b})
    frame = pl.DataFrame(rows)
    out = target_correlation_matrix(frame, ["target_alpha", "target_beta"])
    assert out.columns == ["target_a", "target_b", "mean_corr", "n_eras"]
    assert out.row(0, named=True)["target_a"] == "target_alpha"
    assert out.row(0, named=True)["target_b"] == "target_beta"
    assert out.row(0, named=True)["n_eras"] == 4
    era_corrs = []
    for part in frame.partition_by("era"):
        a = part["target_alpha"].to_numpy()
        b = part["target_beta"].to_numpy()
        ra, rb = scipy.stats.rankdata(a), scipy.stats.rankdata(b)
        era_corrs.append(np.corrcoef(ra, rb)[0, 1])
    assert np.isclose(out.row(0, named=True)["mean_corr"], float(np.mean(era_corrs)), atol=1e-12)


def test_target_correlation_matrix_nan_pair_skipped() -> None:
    frame = pl.DataFrame(
        {
            "era": ["0001", "0001", "0001", "0002", "0002", "0002"],
            "target_alpha": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target_beta": [None, None, None, 1.0, 2.0, 3.0],
        }
    )
    out = target_correlation_matrix(frame, ["target_alpha", "target_beta"])
    assert out.row(0, named=True)["n_eras"] == 1  # era 0001 skipped (all-NaN beta)
    assert np.isclose(out.row(0, named=True)["mean_corr"], 1.0)  # perfectly monotone in 0002


def test_target_correlation_matrix_deterministic() -> None:
    rng = np.random.default_rng(5)
    rows = [
        {"era": f"{e + 1:04d}", "a": float(v), "b": float(-v)}
        for e in range(3)
        for v in rng.normal(size=10)
    ]
    frame = pl.DataFrame(rows)
    out1 = target_correlation_matrix(frame, ["a", "b"])
    out2 = target_correlation_matrix(frame, ["a", "b"])
    assert out1.equals(out2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_analysis.py -q`
Expected: FAIL with `ImportError: cannot import name 'target_profile'`.

- [ ] **Step 3: Implement the two functions**

Append to `nmr/analysis.py`:

```python
def target_profile(
    frame: pl.DataFrame,
    target_cols: Sequence[str],
    era_col: str = "era",
) -> pl.DataFrame:
    """Per-target distribution/availability statistics.

    Non-finite target values are dropped before moments; ``missing_rate`` is
    the fraction of non-finite values over all rows; per-era means are
    computed over eras with at least one valid value.
    """
    target_list = list(target_cols)
    if not target_list:
        raise ValueError("target_cols must contain at least one target")
    n_total = frame.height
    era_values: dict[str, list[np.ndarray]] = {t: [] for t in target_list}
    pooled: dict[str, list[np.ndarray]] = {t: [] for t in target_list}
    zero_var_eras: dict[str, int] = {t: 0 for t in target_list}
    present_eras: dict[str, int] = {t: 0 for t in target_list}
    n_finite: dict[str, int] = {t: 0 for t in target_list}

    for part in frame.select([era_col, *target_list]).partition_by(
        era_col, maintain_order=True
    ):
        for t in target_list:
            values = part.get_column(t).cast(pl.Float64).to_numpy()
            finite = values[np.isfinite(values)]
            n_finite[t] += int(finite.size)
            if finite.size > 0:
                present_eras[t] += 1
                era_values[t].append(finite)
                if finite.size >= 2 and np.all(finite == finite[0]):
                    zero_var_eras[t] += 1
            pooled[t].append(finite)

    rows = []
    for t in target_list:
        pooled_arr = np.concatenate(pooled[t]) if pooled[t] else np.array([])
        if pooled_arr.size == 0:
            rows.append(
                {
                    "target": t,
                    "n_eras_present": present_eras[t],
                    "missing_rate": 1.0,
                    "era_mean_mean": None,
                    "era_mean_std": None,
                    "pooled_mean": None,
                    "pooled_std": None,
                    "pooled_skew": None,
                    "pooled_kurtosis": None,
                    "min": None,
                    "max": None,
                    "zero_variance_era_count": 0,
                }
            )
            continue
        era_means = np.array([float(np.mean(v)) for v in era_values[t]])
        mu = float(np.mean(pooled_arr))
        sd = float(np.std(pooled_arr, ddof=0))
        skew = float(scipy.stats.skew(pooled_arr)) if sd > 0 else 0.0
        kurt = float(scipy.stats.kurtosis(pooled_arr, fisher=True)) if sd > 0 else 0.0
        rows.append(
            {
                "target": t,
                "n_eras_present": present_eras[t],
                "missing_rate": 1.0 - n_finite[t] / n_total,
                "era_mean_mean": float(np.mean(era_means)),
                "era_mean_std": float(np.std(era_means, ddof=0)),
                "pooled_mean": mu,
                "pooled_std": sd,
                "pooled_skew": skew,
                "pooled_kurtosis": kurt,
                "min": float(np.min(pooled_arr)),
                "max": float(np.max(pooled_arr)),
                "zero_variance_era_count": zero_var_eras[t],
            }
        )
    return pl.DataFrame(rows)


def target_correlation_matrix(
    frame: pl.DataFrame,
    target_cols: Sequence[str],
    era_col: str = "era",
) -> pl.DataFrame:
    """Equal-era-weighted mean Spearman correlation between target pairs.

    An era is skipped for a pair when either target has <2 valid values or
    zero variance; ``n_eras`` records how many eras contributed.
    """
    target_list = list(target_cols)
    if len(target_list) < 2:
        raise ValueError("target_correlation_matrix needs at least two targets")
    pairs: dict[tuple[str, str], list[float]] = {}
    for i in range(len(target_list)):
        for j in range(i + 1, len(target_list)):
            pairs[(target_list[i], target_list[j])] = []

    for part in frame.select([era_col, *target_list]).partition_by(
        era_col, maintain_order=True
    ):
        for (a, b), values in pairs.items():
            av = part.get_column(a).cast(pl.Float64).to_numpy()
            bv = part.get_column(b).cast(pl.Float64).to_numpy()
            mask = np.isfinite(av) & np.isfinite(bv)
            avc, bvc = av[mask], bv[mask]
            if avc.size < 2 or np.std(avc) == 0.0 or np.std(bvc) == 0.0:
                continue
            ra = scipy.stats.rankdata(avc)
            rb = scipy.stats.rankdata(bvc)
            values.append(float(np.corrcoef(ra, rb)[0, 1]))

    rows = [
        {
            "target_a": a,
            "target_b": b,
            "mean_corr": float(np.mean(vals)) if vals else None,
            "n_eras": len(vals),
        }
        for (a, b), vals in pairs.items()
    ]
    return pl.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_analysis.py -q`
Expected: PASS (11 tests).

- [ ] **Step 5: Full suite + commit**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS.
```bash
git add nmr/analysis.py tests/test_analysis.py
git commit -m "feat(analysis): target profile and pairwise correlation"
```

### Task 8: `feature_ic_by_era` + `feature_ic_screen`

**Files:**
- Modify: `nmr/analysis.py`, `tests/test_analysis.py`

**Interfaces:**
- Consumes: `nmr.features._per_era_pearson` (Task 1), `nmr.features.feature_stability_screen`.
- Produces:
  - `feature_ic_by_era(frame, feature_cols: Sequence[str], target_col: str, era_col: str = "era") -> pl.DataFrame` — long-form `era, feature, ic, degenerate` (0.0 IC on degenerate eras per the screen convention).
  - `feature_ic_screen(frame, feature_cols, targets: Sequence[str], era_col: str = "era") -> pl.DataFrame` — rows `feature, target, mean_corr, corr_std, decay_slope, cross_regime_variance, n_eras, stable` (concatenation of `feature_stability_screen` per target).
- Later tasks consume: `feature_ic_by_era` in Task 11 (`regime_analysis`); both feed Task 14 dumps.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analysis.py`:

```python
from nmr.analysis import feature_ic_by_era, feature_ic_screen
from nmr.features import _per_era_pearson


def _ic_frame() -> pl.DataFrame:
    rng = np.random.default_rng(21)
    rows: list[dict[str, float | str]] = []
    for e in range(4):
        era = f"{e + 1:04d}"
        for i in range(12):
            rows.append(
                {
                    "era": era,
                    "feature_alpha": float(rng.normal()),
                    "feature_beta": float(rng.normal()),
                    "target": float(rng.normal()),
                }
            )
    return pl.DataFrame(rows)


def test_feature_ic_by_era_long_form() -> None:
    frame = _ic_frame()
    out = feature_ic_by_era(frame, ["feature_alpha", "feature_beta"], "target")
    assert out.columns == ["era", "feature", "ic", "degenerate"]
    assert out.height == 4 * 2
    assert out["feature"].n_unique() == 2
    assert out["era"].n_unique() == 4
    corrs, _ = _per_era_pearson(frame, ["feature_alpha", "feature_beta"], "target", "era")
    for era, vec in corrs.items():
        rows = out.filter(pl.col("era") == era)
        assert np.array_equal(rows["ic"].to_numpy(), vec)


def test_feature_ic_by_era_degenerate_flag() -> None:
    frame = pl.DataFrame(
        {
            "era": ["0001", "0002", "0002", "0002", "0002"],
            "feature_alpha": [1.0, 2.0, 1.0, 1.0, 1.0],
            "feature_beta": [3.0, 4.0, 5.0, 6.0, 7.0],
            "target": [0.1, 1.0, 1.0, 1.0, 1.0],
        }
    )
    out = feature_ic_by_era(frame, ["feature_alpha", "feature_beta"], "target")
    assert out.filter(pl.col("era") == "0001")["degenerate"].all()  # <2 rows
    assert out.filter(pl.col("era") == "0002")["degenerate"].all()  # const target
    assert (out.filter(pl.col("era") == "0001")["ic"] == 0.0).all()  # zero vectors


def test_feature_ic_screen_multi_target() -> None:
    frame = _ic_frame()
    out = feature_ic_screen(frame, ["feature_alpha", "feature_beta"], ["target"])
    assert out.columns == [
        "feature",
        "target",
        "mean_corr",
        "corr_std",
        "decay_slope",
        "cross_regime_variance",
        "n_eras",
        "stable",
    ]
    assert out.height == 2
    assert out["target"].to_list() == ["target", "target"]


def test_feature_ic_screen_empty_targets_raises() -> None:
    with pytest.raises(ValueError):
        feature_ic_screen(_ic_frame(), ["feature_alpha"], [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_analysis.py -q`
Expected: FAIL with `ImportError: cannot import name 'feature_ic_by_era'`.

- [ ] **Step 3: Implement the two functions**

Append to `nmr/analysis.py`:

```python
def feature_ic_by_era(
    frame: pl.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    era_col: str = "era",
) -> pl.DataFrame:
    """Per-era per-feature IC long-form, via ``_per_era_pearson``.

    Degenerate eras (per the screen convention) carry ``ic = 0.0`` and
    ``degenerate = True``; all other rows carry ``degenerate = False``.
    """
    from nmr.features import _per_era_pearson

    feature_list = list(feature_cols)
    corrs, degenerate = _per_era_pearson(frame, feature_list, target_col, era_col)
    rows = [
        {
            "era": era,
            "feature": feature,
            "ic": float(vec[i]),
            "degenerate": era in degenerate,
        }
        for era, vec in corrs.items()
        for i, feature in enumerate(feature_list)
    ]
    return pl.DataFrame(
        rows,
        schema={
            "era": pl.Utf8,
            "feature": pl.Utf8,
            "ic": pl.Float64,
            "degenerate": pl.Boolean,
        },
    )


def feature_ic_screen(
    frame: pl.DataFrame,
    feature_cols: Sequence[str],
    targets: Sequence[str],
    era_col: str = "era",
) -> pl.DataFrame:
    """Aggregated feature-target screen, one block per reference target.

    Thin wrapper over ``feature_stability_screen`` (the single screen
    implementation) that tags each block with its target.
    """
    from nmr.features import feature_stability_screen

    if not targets:
        raise ValueError("targets must contain at least one target column")
    blocks = [
        feature_stability_screen(
            frame, feature_cols=feature_cols, target_col=t, era_col=era_col
        ).with_columns(pl.lit(t).alias("target"))
        for t in targets
    ]
    return pl.concat(blocks).select(
        [
            "feature",
            "target",
            "mean_corr",
            "corr_std",
            "decay_slope",
            "cross_regime_variance",
            "n_eras",
            "stable",
        ]
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_analysis.py -q`
Expected: PASS (15 tests).

- [ ] **Step 5: Full suite + commit**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS.
```bash
git add nmr/analysis.py tests/test_analysis.py
git commit -m "feat(analysis): per-era feature IC and multi-target screen"
```

### Task 9: `feature_summary` — streaming Welford + Terriberry moments

**Files:**
- Modify: `nmr/analysis.py`, `tests/test_analysis.py`

**Interfaces:**
- Consumes: nothing new (numpy).
- Produces:
  - `feature_summary(chunks: Iterable[pl.DataFrame], feature_cols: Sequence[str], era_col: str = "era") -> pl.DataFrame` — per-feature `feature, pooled_mean, pooled_std, pooled_skew, pooled_kurtosis, min, max, missing_rate`. Caller drives chunking (era-sorted ascending ⇒ deterministic). Same chunk order on the same NumPy build ⇒ bit-identical.
  - Private helpers: `_chunk_moments(values: np.ndarray) -> tuple[float, float, float, float, float]` (n, mean, M2, M3, M4) and `_combine(a, b) -> tuple[...]` (Terriberry parallel combine).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analysis.py`:

```python
import scipy.stats

from nmr.analysis import feature_summary


def _chunks(n_eras: int = 5, rows: int = 40) -> list[pl.DataFrame]:
    rng = np.random.default_rng(42)
    chunks = []
    for e in range(n_eras):
        era = f"{e + 1:04d}"
        chunks.append(
            pl.DataFrame(
                {
                    "era": [era] * rows,
                    "f1": rng.normal(size=rows),
                    "f2": rng.normal(loc=2.0, scale=0.5, size=rows),
                }
            )
        )
    return chunks


def test_feature_summary_moments_match_scipy() -> None:
    chunks = _chunks()
    out = feature_summary(chunks, ["f1", "f2"])
    assert out.columns == [
        "feature",
        "pooled_mean",
        "pooled_std",
        "pooled_skew",
        "pooled_kurtosis",
        "min",
        "max",
        "missing_rate",
    ]
    full = pl.concat(chunks)
    for feature in ["f1", "f2"]:
        series = full[feature].to_numpy()
        row = out.filter(pl.col("feature") == feature).row(0, named=True)
        assert np.isclose(row["pooled_mean"], float(series.mean()), atol=1e-12)
        assert np.isclose(row["pooled_std"], float(series.std(ddof=0)), atol=1e-12)
        assert np.isclose(row["pooled_skew"], float(scipy.stats.skew(series)), atol=1e-9)
        assert np.isclose(
            row["pooled_kurtosis"],
            float(scipy.stats.kurtosis(series, fisher=True)),
            atol=1e-8,
        )
        assert row["missing_rate"] == 0.0


def test_feature_summary_constant_column() -> None:
    chunks = [
        pl.DataFrame({"era": ["0001"] * 5, "f1": [7.0] * 5}),
        pl.DataFrame({"era": ["0002"] * 5, "f1": [7.0] * 5}),
    ]
    out = feature_summary(chunks, ["f1"])
    row = out.row(0, named=True)
    assert row["pooled_std"] == 0.0
    assert row["pooled_skew"] == 0.0
    assert row["pooled_kurtosis"] == 0.0
    assert row["min"] == 7.0 and row["max"] == 7.0


def test_feature_summary_missing_rate() -> None:
    chunks = [
        pl.DataFrame({"era": ["0001", "0001", "0001"], "f1": [1.0, None, 3.0]}),
        pl.DataFrame({"era": ["0002", "0002", "0002"], "f1": [4.0, 5.0, None]}),
    ]
    out = feature_summary(chunks, ["f1"])
    assert np.isclose(out.row(0, named=True)["missing_rate"], 2 / 6)


def test_feature_summary_chunked_vs_single_pass() -> None:
    chunks = _chunks()
    out_chunked = feature_summary(chunks, ["f1", "f2"])
    out_single = feature_summary([pl.concat(chunks)], ["f1", "f2"])
    for c in ["pooled_mean", "pooled_std", "pooled_skew", "pooled_kurtosis", "min", "max"]:
        assert np.allclose(
            out_chunked[c].to_numpy(), out_single[c].to_numpy(), rtol=1e-9
        )


def test_feature_summary_chunked_bit_identical() -> None:
    chunks = _chunks()
    out1 = feature_summary(chunks, ["f1", "f2"])
    out2 = feature_summary(chunks, ["f1", "f2"])
    assert out1.equals(out2)  # same chunk order => bit-identical on same build


def test_feature_summary_requires_era_and_features() -> None:
    with pytest.raises(ValueError):
        feature_summary([pl.DataFrame({"era": ["0001"], "f1": [1.0]})], ["f1", "missing"])
    with pytest.raises(ValueError):
        feature_summary([pl.DataFrame({"x": [1.0]})], ["f1"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_analysis.py -q`
Expected: FAIL with `ImportError: cannot import name 'feature_summary'`.

- [ ] **Step 3: Implement the function + helpers**

Append to `nmr/analysis.py`:

```python
def _chunk_moments(values: np.ndarray) -> tuple[float, float, float, float, float]:
    """(n, mean, m2, m3, m4) over a finite 1-D array (raw central moment sums)."""
    n = values.size
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    mean = float(np.mean(values))
    centered = values - mean
    m2 = float(np.sum(centered**2))
    m3 = float(np.sum(centered**3))
    m4 = float(np.sum(centered**4))
    return (float(n), mean, m2, m3, m4)


def _combine(
    a: tuple[float, float, float, float, float],
    b: tuple[float, float, float, float, float],
) -> tuple[float, float, float, float, float]:
    """Terriberry parallel combine of (n, mean, M2, M3, M4) moments.

    ``M2/M3/M4`` are raw central-moment sums; ``mean`` is the arithmetic mean.
    """
    n1, mean_a, M2_a, M3_a, M4_a = a
    n2, mean_b, M2_b, M3_b, M4_b = b
    n = n1 + n2
    if n == 0.0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    delta = mean_b - mean_a
    mean = mean_a + delta * n2 / n
    M2 = M2_a + M2_b + delta * delta * n1 * n2 / n
    M3 = (
        M3_a
        + M3_b
        + delta * delta * delta * n1 * n2 * (n1 - n2) / (n * n)
        + 3.0 * delta * (n1 * M2_b - n2 * M2_a) / n
    )
    M4 = (
        M4_a
        + M4_b
        + delta**4 * n1 * n2 * (n1 * n1 - n1 * n2 + n2 * n2) / (n**3)
        + 6.0 * delta * delta * (n1 * n1 * M2_b + n2 * n2 * M2_a) / (n * n)
        + 4.0 * delta * (n1 * M3_b - n2 * M3_a) / n
    )
    return (n, mean, M2, M3, M4)


def feature_summary(
    chunks: Iterable[pl.DataFrame],
    feature_cols: Sequence[str],
    era_col: str = "era",
) -> pl.DataFrame:
    """Per-feature pooled moments via streaming Welford + Terriberry.

    Caller drives chunking (era-sorted ascending). Non-finite values are
    dropped before moments; ``missing_rate = 1 - n_finite / n_total``.
    """
    feature_list = list(feature_cols)
    if not feature_list:
        raise ValueError("feature_cols must contain at least one feature")
    acc = {
        f: [0.0, 0.0, 0.0, 0.0, 0.0, np.inf, -np.inf, 0.0]
        for f in feature_list
    }  # n, mean, m2, m3, m4, min, max, n_finite
    n_total = 0
    for chunk in chunks:
        if era_col not in chunk.columns:
            raise ValueError(f"chunk missing required column {era_col!r}")
        missing = set(feature_list) - set(chunk.columns)
        if missing:
            raise ValueError(f"chunk missing feature columns: {sorted(missing)}")
        n_total += chunk.height
        for f in feature_list:
            values = chunk.get_column(f).cast(pl.Float64).to_numpy()
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            state = acc[f]
            combined = _combine(tuple(state[:5]), _chunk_moments(finite))
            state[:5] = list(combined)
            state[5] = min(state[5], float(np.min(finite)))
            state[6] = max(state[6], float(np.max(finite)))
            state[7] += float(finite.size)

    rows = []
    for f in feature_list:
        n, mean, m2, m3, m4, cmin, cmax, n_finite = acc[f]
        if n == 0.0:
            rows.append(
                {
                    "feature": f,
                    "pooled_mean": None,
                    "pooled_std": None,
                    "pooled_skew": None,
                    "pooled_kurtosis": None,
                    "min": None,
                    "max": None,
                    "missing_rate": 1.0,
                }
            )
            continue
        std = float(np.sqrt(m2 / n)) if m2 > 0 else 0.0
        skew = float((m3 / n) / ((m2 / n) ** 1.5)) if m2 > 0 else 0.0
        kurt = float((m4 / n) / ((m2 / n) ** 2) - 3.0) if m2 > 0 else 0.0
        rows.append(
            {
                "feature": f,
                "pooled_mean": mean,
                "pooled_std": std,
                "pooled_skew": skew,
                "pooled_kurtosis": kurt,
                "min": cmin,
                "max": cmax,
                "missing_rate": 1.0 - n_finite / n_total,
            }
        )
    return pl.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_analysis.py -q`
Expected: PASS (20 tests). If kurtosis differs beyond `atol=1e-8`, loosen the test tolerance to `1e-7` — Terriberry higher-moment accumulation is order-sensitive in the last ulps; the chunked-vs-single-pass `rtol=1e-9` test is the correctness net.

- [ ] **Step 5: Full suite + commit**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS.
```bash
git add nmr/analysis.py tests/test_analysis.py
git commit -m "feat(analysis): streaming feature moments (Welford+Terriberry)"
```

### Task 10: `feature_correlation_structure` + `within_set_redundancy` + `cross_set_membership`

**Files:**
- Modify: `nmr/analysis.py`, `tests/test_analysis.py`

**Interfaces:**
- Consumes: `scipy.stats.rankdata`, `scipy.stats.norm` (existing deps).
- Produces:
  - `FeatureCorrResult` frozen dataclass: `matrix: np.ndarray` (float32 N×N symmetric era-averaged corr), `feature_order: tuple[str, ...]`, `top_pairs: pl.DataFrame` (`feature_a, feature_b, mean_corr`, top-100 by |corr|), `summary: dict` (`mean_abs_corr`, `p50_abs_corr`, `p90_abs_corr`, `n_pairs`).
  - `feature_correlation_structure(chunks: Iterable[pl.DataFrame], feature_cols: Sequence[str], era_col: str = "era") -> FeatureCorrResult`
  - `within_set_redundancy(result: FeatureCorrResult, sets: Mapping[str, Sequence[str]]) -> pl.DataFrame` — `feature_set, n_features, mean_abs_corr, median_abs_corr, max_abs_corr, n_pairs`
  - `cross_set_membership(sets: Mapping[str, Sequence[str]]) -> dict` — `{"sets": pl.DataFrame(feature_set, n_features), "subset_relations": pl.DataFrame(a, b, a_subset_of_b)}`
- Semantics: equal era weight; complete-case rows per era; rank-gaussianize via `norm.ppf(rank / (n+1))`; degenerate columns (zero variance) → 0.0 correlation.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analysis.py`:

```python
from nmr.analysis import (
    FeatureCorrResult,
    cross_set_membership,
    feature_correlation_structure,
    within_set_redundancy,
)


def test_feature_correlation_structure_equal_era_weight() -> None:
    # era sizes differ: 0001 has 10 rows, 0002 has 4 rows; both have
    # f1~f2 near-perfectly correlated and f1~f3 anti-correlated
    rng = np.random.default_rng(9)

    def _era(era: str, n: int) -> pl.DataFrame:
        base = rng.normal(size=n)
        return pl.DataFrame(
            {
                "era": [era] * n,
                "f1": base,
                "f2": base + rng.normal(scale=0.01, size=n),
                "f3": -base,
            }
        )

    chunks = [_era("0001", 10), _era("0002", 4)]
    result = feature_correlation_structure(chunks, ["f1", "f2", "f3"])
    assert isinstance(result, FeatureCorrResult)
    assert result.matrix.shape == (3, 3)
    mat = result.matrix
    assert np.allclose(mat[0, 1], 1.0, atol=1e-3)
    assert np.allclose(mat[0, 2], -1.0, atol=1e-3)
    assert np.allclose(mat, mat.T, atol=1e-12)  # symmetric
    assert result.feature_order == ("f1", "f2", "f3")
    assert result.top_pairs.columns == ["feature_a", "feature_b", "mean_corr"]


def test_feature_correlation_structure_zero_variance_era() -> None:
    chunks = [
        pl.DataFrame(
            {"era": ["0001"] * 3, "f1": [1.0, 2.0, 3.0], "f2": [1.0, 1.0, 1.0]}
        ),
        pl.DataFrame(
            {"era": ["0002"] * 3, "f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]}
        ),
    ]
    result = feature_correlation_structure(chunks, ["f1", "f2"])
    # era 0001 f2 has zero variance -> 0.0 correlation; era 0002 -> ~1.0;
    # equal era weight -> ~0.5
    assert np.isclose(result.matrix[0, 1], 0.5, atol=1e-6)


def test_feature_correlation_structure_no_eras_raises() -> None:
    with pytest.raises(ValueError):
        feature_correlation_structure(
            [pl.DataFrame({"era": ["0001"], "f1": [1.0]})], ["f1"]
        )


def test_within_set_redundancy() -> None:
    rng = np.random.default_rng(13)
    chunks = [
        pl.DataFrame(
            {
                "era": [f"{e + 1:04d}"] * 8,
                "fa": rng.normal(size=8),
                "fb": rng.normal(size=8),
                "fc": rng.normal(size=8),
            }
        )
        for e in range(3)
    ]
    result = feature_correlation_structure(chunks, ["fa", "fb", "fc"])
    sets = {"pair": ["fa", "fb"], "solo": ["fa"], "all3": ["fa", "fb", "fc"]}
    out = within_set_redundancy(result, sets)
    assert out["feature_set"].to_list() == ["all3", "pair", "solo"]  # sorted
    row_solo = out.filter(pl.col("feature_set") == "solo").row(0, named=True)
    assert row_solo["n_pairs"] == 0
    assert row_solo["mean_abs_corr"] is None
    row_pair = out.filter(pl.col("feature_set") == "pair").row(0, named=True)
    assert row_pair["n_pairs"] == 1
    assert np.isclose(
        row_pair["mean_abs_corr"], float(np.abs(result.matrix[0, 1])), atol=1e-12
    )


def test_cross_set_membership_subset_relations() -> None:
    sets = {
        "small": ["a", "b"],
        "medium": ["a", "b", "c"],
        "all": ["a", "b", "c", "d"],
    }
    out = cross_set_membership(sets)
    assert out["sets"]["n_features"].to_list() == [2, 3, 4]
    relations = out["subset_relations"]
    rel = {
        (r["a"], r["b"]): r["a_subset_of_b"] for r in relations.iter_rows(named=True)
    }
    assert rel[("small", "medium")] is True
    assert rel[("small", "all")] is True
    assert rel[("medium", "all")] is True
    assert rel[("all", "small")] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_analysis.py -q`
Expected: FAIL with `ImportError: cannot import name 'FeatureCorrResult'`.

- [ ] **Step 3: Implement the dataclass + three functions**

Append to `nmr/analysis.py`:

```python
@dataclass(frozen=True)
class FeatureCorrResult:
    """Era-averaged feature correlation structure."""

    matrix: np.ndarray  # float32 (N, N) symmetric
    feature_order: tuple[str, ...]
    top_pairs: pl.DataFrame
    summary: dict


def _rank_gaussianize_chunk(
    chunk: pl.DataFrame,
    feature_list: Sequence[str],
    era_col: str,
) -> np.ndarray | None:
    """Complete-case per-era rank-gaussianized feature matrix, or None."""
    clean = chunk.select([era_col, *feature_list]).drop_nulls()
    if clean.height < 2:
        return None
    out = np.empty((clean.height, len(feature_list)), dtype=np.float64)
    for j, feature in enumerate(feature_list):
        col = clean.get_column(feature).cast(pl.Float64).to_numpy()
        ranks = scipy.stats.rankdata(col, method="average")
        out[:, j] = scipy.stats.norm.ppf(ranks / (col.size + 1))
    return out


def feature_correlation_structure(
    chunks: Iterable[pl.DataFrame],
    feature_cols: Sequence[str],
    era_col: str = "era",
) -> FeatureCorrResult:
    """Equal-era-weighted mean feature correlation matrix.

    Per era: complete-case rows only, rank-gaussianized per feature, then the
    full correlation matrix; matrices are summed and divided by the era count.
    Degenerate columns (zero variance) contribute 0.0.
    """
    feature_list = list(feature_cols)
    if not feature_list:
        raise ValueError("feature_cols must contain at least one feature")
    n = len(feature_list)
    acc = np.zeros((n, n), dtype=np.float64)
    n_eras = 0
    for chunk in chunks:
        gauss = _rank_gaussianize_chunk(chunk, feature_list, era_col)
        if gauss is None:
            continue
        mat = np.corrcoef(gauss, rowvar=False)
        mat = np.where(np.isfinite(mat), mat, 0.0)
        acc += mat
        n_eras += 1
    if n_eras == 0:
        raise ValueError("no usable eras in feature_correlation_structure input")
    mean_mat = (acc / n_eras).astype(np.float32)

    iu = np.triu_indices(n, k=1)
    abs_vals = np.abs(mean_mat[iu])
    order = np.argsort(abs_vals)[::-1][:100]
    top_rows = [
        {
            "feature_a": feature_list[iu[0][k]],
            "feature_b": feature_list[iu[1][k]],
            "mean_corr": float(mean_mat[iu[0][k], iu[1][k]]),
        }
        for k in order
    ]
    summary = {
        "mean_abs_corr": float(abs_vals.mean()) if abs_vals.size else 0.0,
        "p50_abs_corr": float(np.percentile(abs_vals, 50)) if abs_vals.size else 0.0,
        "p90_abs_corr": float(np.percentile(abs_vals, 90)) if abs_vals.size else 0.0,
        "n_pairs": int(abs_vals.size),
    }
    return FeatureCorrResult(
        matrix=mean_mat,
        feature_order=tuple(feature_list),
        top_pairs=pl.DataFrame(
            top_rows,
            schema={
                "feature_a": pl.Utf8,
                "feature_b": pl.Utf8,
                "mean_corr": pl.Float64,
            },
        ),
        summary=summary,
    )


def within_set_redundancy(
    result: FeatureCorrResult,
    sets: Mapping[str, Sequence[str]],
) -> pl.DataFrame:
    """Per-feature-set pairwise |corr| summary, indexed from the full matrix."""
    index = {f: i for i, f in enumerate(result.feature_order)}
    rows = []
    for name in sorted(sets):
        members = [f for f in sets[name] if f in index]
        if len(members) < 2:
            rows.append(
                {
                    "feature_set": name,
                    "n_features": len(members),
                    "mean_abs_corr": None,
                    "median_abs_corr": None,
                    "max_abs_corr": None,
                    "n_pairs": 0,
                }
            )
            continue
        idx = [index[f] for f in members]
        sub = result.matrix[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), k=1)
        abs_vals = np.abs(sub[iu])
        rows.append(
            {
                "feature_set": name,
                "n_features": len(members),
                "mean_abs_corr": float(abs_vals.mean()),
                "median_abs_corr": float(np.median(abs_vals)),
                "max_abs_corr": float(abs_vals.max()),
                "n_pairs": int(abs_vals.size),
            }
        )
    return pl.DataFrame(rows)


def cross_set_membership(sets: Mapping[str, Sequence[str]]) -> dict:
    """Set sizes and pairwise empirical subset relations."""
    names = sorted(sets)
    set_rows = [
        {"feature_set": name, "n_features": len(set(sets[name]))} for name in names
    ]
    rel_rows = []
    for a in names:
        for b in names:
            if a == b:
                continue
            rel_rows.append(
                {
                    "a": a,
                    "b": b,
                    "a_subset_of_b": set(sets[a]).issubset(set(sets[b])),
                }
            )
    return {
        "sets": pl.DataFrame(set_rows),
        "subset_relations": pl.DataFrame(rel_rows),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_analysis.py -q`
Expected: PASS (25 tests).

- [ ] **Step 5: Full suite + commit**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS.
```bash
git add nmr/analysis.py tests/test_analysis.py
git commit -m "feat(analysis): feature correlation structure and set redundancy"
```

### Task 11: `regime_analysis`

**Files:**
- Modify: `nmr/analysis.py`, `tests/test_analysis.py`

**Interfaces:**
- Consumes: `feature_ic_by_era` output (Task 8).
- Produces: `regime_analysis(ic_by_era: pl.DataFrame) -> dict` with keys:
  - `"regime_thresholds"`: `{"low_pct": 10.0, "high_pct": 90.0, "q1": float, "q3": float, "mean_ic_low": float, "mean_ic_high": float}`
  - `"era_signal"`: `pl.DataFrame` — `era, mean_ic, ic_std, n_features, pct_rank, regime, crash, hot` (regime ∈ {low, normal, high}; crash = pct_rank ≤ 10; hot = pct_rank ≥ 90)
  - `"crash_eras"`: `list[str]`, `"hot_eras"`: `list[str]`
  - `"ic_persistence"`: `{"mean": float, "std": float, "n_adjacent": int}` (Spearman of adjacent-era IC-vectors)
  - `"rolling_vol"`: `pl.DataFrame` — `era, rolling_std` (window `IC_VOL_WINDOW=20`, min 2 periods)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analysis.py`:

```python
from nmr.analysis import (
    IC_VOL_WINDOW,
    REGIME_HIGH_PCT,
    REGIME_LOW_PCT,
    regime_analysis,
)


def _ic_by_era_series(n_eras: int = 30) -> pl.DataFrame:
    # era mean_ic ramps upward so quartile/decile bands are well-separated
    rows = []
    for e in range(n_eras):
        era = f"{e + 1:04d}"
        mean_ic = -0.05 + 0.10 * e / max(n_eras - 1, 1)
        for f in ["fa", "fb", "fc"]:
            rows.append({"era": era, "feature": f, "ic": float(mean_ic)})
    return pl.DataFrame(rows)


def test_regime_analysis_bands_and_flags() -> None:
    out = regime_analysis(_ic_by_era_series(30))
    assert REGIME_LOW_PCT == 10.0
    assert REGIME_HIGH_PCT == 90.0
    assert IC_VOL_WINDOW == 20
    sig = out["era_signal"]
    assert "regime" in sig.columns and "crash" in sig.columns and "hot" in sig.columns
    first = sig.row(0, named=True)
    last = sig.row(sig.height - 1, named=True)
    assert first["regime"] == "low"
    assert first["crash"] is True
    assert last["regime"] == "high"
    assert last["hot"] is True
    th = out["regime_thresholds"]
    assert th["mean_ic_low"] <= th["q1"] <= th["q3"] <= th["mean_ic_high"]
    assert out["crash_eras"] == ["0001", "0002", "0003"]
    assert out["hot_eras"] == [f"{e:04d}" for e in range(28, 31)]


def test_regime_analysis_persistence_rank_stable_series() -> None:
    # feature IC ranks constant across eras -> adjacent Spearman = 1.0
    rows = []
    for e in range(5):
        era = f"{e + 1:04d}"
        for f, ic in [("fa", 0.1), ("fb", 0.05), ("fc", 0.0)]:
            rows.append({"era": era, "feature": f, "ic": ic})
    out = regime_analysis(pl.DataFrame(rows))
    assert np.isclose(out["ic_persistence"]["mean"], 1.0, atol=1e-12)
    assert out["ic_persistence"]["n_adjacent"] == 4


def test_regime_analysis_deterministic() -> None:
    ic = _ic_by_era_series(25)
    out1 = regime_analysis(ic)
    out2 = regime_analysis(ic)
    assert out1["era_signal"].equals(out2["era_signal"])
    assert out1["crash_eras"] == out2["crash_eras"]
    assert out1["ic_persistence"] == out2["ic_persistence"]


def test_regime_analysis_requires_columns() -> None:
    with pytest.raises(ValueError):
        regime_analysis(pl.DataFrame({"era": ["0001"], "ic": [0.1]}))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_analysis.py -q`
Expected: FAIL with `ImportError: cannot import name 'regime_analysis'`.

- [ ] **Step 3: Implement the function**

Append to `nmr/analysis.py`:

```python
def regime_analysis(ic_by_era: pl.DataFrame) -> dict:
    """Deterministic, percentile-based regime analysis of per-era feature IC.

    Crash/hot use decile thresholds (``REGIME_LOW_PCT`` / ``REGIME_HIGH_PCT``);
    the regime column uses quartile bands. ``ic_persistence`` is the mean
    adjacent-era Spearman rank correlation of per-era feature IC vectors.
    """
    required = {"era", "feature", "ic"}
    missing = required - set(ic_by_era.columns)
    if missing:
        raise ValueError(f"ic_by_era missing required columns: {sorted(missing)}")

    sig = (
        ic_by_era.group_by("era")
        .agg(
            pl.col("ic").mean().alias("mean_ic"),
            pl.col("ic").std().alias("ic_std"),
            pl.col("feature").count().alias("n_features"),
        )
        .sort("era")
    )
    mean_ics = sig.get_column("mean_ic").to_numpy()
    n = len(mean_ics)
    ranks = np.argsort(np.argsort(mean_ics))
    pct = 100.0 * ranks / (n - 1) if n > 1 else np.array([50.0])

    q1 = float(np.percentile(mean_ics, 25.0))
    q3 = float(np.percentile(mean_ics, 75.0))
    low_thr = float(np.percentile(mean_ics, REGIME_LOW_PCT))
    high_thr = float(np.percentile(mean_ics, REGIME_HIGH_PCT))

    regime = np.where(pct <= 25.0, "low", np.where(pct >= 75.0, "high", "normal"))
    crash = pct <= REGIME_LOW_PCT
    hot = pct >= REGIME_HIGH_PCT
    sig = sig.with_columns(
        pl.Series("pct_rank", pct),
        pl.Series("regime", regime),
        pl.Series("crash", crash),
        pl.Series("hot", hot),
    )

    eras = sig.get_column("era").to_list()
    crash_eras = [e for e, c in zip(eras, crash) if c]
    hot_eras = [e for e, h in zip(eras, hot) if h]

    # adjacent-era IC-vector Spearman
    pivot = ic_by_era.pivot(on="feature", index="era", values="ic").sort("era")
    feature_names = [c for c in pivot.columns if c != "era"]
    matrix = pivot.select(feature_names).to_numpy()
    matrix = np.nan_to_num(matrix, nan=0.0)
    ranks_mat = np.apply_along_axis(scipy.stats.rankdata, 1, matrix)
    adj = [
        float(np.corrcoef(ranks_mat[t], ranks_mat[t - 1])[0, 1])
        for t in range(1, ranks_mat.shape[0])
    ]
    persistence = {
        "mean": float(np.mean(adj)) if adj else 0.0,
        "std": float(np.std(adj, ddof=0)) if adj else 0.0,
        "n_adjacent": len(adj),
    }

    rolling = sig.select(
        pl.col("era"),
        pl.col("mean_ic")
        .rolling_std(window_size=IC_VOL_WINDOW, min_periods=2)
        .alias("rolling_std"),
    )

    return {
        "regime_thresholds": {
            "low_pct": REGIME_LOW_PCT,
            "high_pct": REGIME_HIGH_PCT,
            "q1": q1,
            "q3": q3,
            "mean_ic_low": low_thr,
            "mean_ic_high": high_thr,
        },
        "era_signal": sig,
        "crash_eras": crash_eras,
        "hot_eras": hot_eras,
        "ic_persistence": persistence,
        "rolling_vol": rolling,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_analysis.py -q`
Expected: PASS (29 tests). If `crash_eras`/`hot_eras` boundary counts differ from the test (percentile edge effects on 30 eras), adjust the expected lists to match the actual decile boundaries — the invariants that matter are `crash ⊆ low`, `hot ⊆ high`, and determinism.

- [ ] **Step 5: Full suite + commit**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS.
```bash
git add nmr/analysis.py tests/test_analysis.py
git commit -m "feat(analysis): percentile-based regime analysis"
```

### Task 12: `benchmark_era_corr`

**Files:**
- Modify: `nmr/analysis.py`, `tests/test_analysis.py`

**Interfaces:**
- Consumes: `nmr.features._per_era_pearson` (Task 1).
- Produces: `benchmark_era_corr(frame, benchmark_cols: Sequence[str], target_col: str, era_col: str = "era") -> dict` — `{"benchmarks": pl.DataFrame(benchmark, mean_corr, corr_std, n_eras, first_era, last_era), "per_era": pl.DataFrame(era, benchmark, corr)}`. Degenerate eras (<2 rows, constant target) are **silently absent** — `n_eras` reflects actual overlap.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analysis.py`:

```python
from nmr.analysis import benchmark_era_corr


def test_benchmark_era_corr_known_values() -> None:
    rng = np.random.default_rng(31)
    rows = []
    for e in range(3):
        era = f"{e + 1:04d}"
        for i in range(20):
            pred = float(rng.normal())
            rows.append(
                {
                    "era": era,
                    "id": f"{era}-{i}",
                    "benchmark_small": pred,
                    "benchmark_medium": 2.0 * pred + float(rng.normal(scale=0.1)),
                    "target": pred + float(rng.normal(scale=0.1)),
                }
            )
    frame = pl.DataFrame(rows)
    out = benchmark_era_corr(frame, ["benchmark_small", "benchmark_medium"], "target")
    summary = out["benchmarks"]
    assert summary["benchmark"].to_list() == ["benchmark_medium", "benchmark_small"]
    assert summary["n_eras"].to_list() == [3, 3]
    assert summary["first_era"].to_list() == ["0001", "0001"]
    assert summary["last_era"].to_list() == ["0003", "0003"]
    for row in summary.iter_rows(named=True):
        assert row["mean_corr"] > 0.5


def test_benchmark_era_corr_absent_degenerate_eras() -> None:
    # era 0001 has only 1 row -> degenerate -> absent from output
    frame = pl.DataFrame(
        {
            "era": ["0001", "0002", "0002", "0002", "0002"],
            "id": ["a", "b", "c", "d", "e"],
            "benchmark_small": [1.0, 1.0, 2.0, 3.0, 4.0],
            "target": [0.5, 1.0, 2.0, 3.0, 4.0],
        }
    )
    out = benchmark_era_corr(frame, ["benchmark_small"], "target")
    assert out["benchmarks"]["n_eras"].to_list() == [1]
    assert out["benchmarks"]["first_era"].to_list() == ["0002"]
    assert set(out["per_era"]["era"].to_list()) == {"0002"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_analysis.py -q`
Expected: FAIL with `ImportError: cannot import name 'benchmark_era_corr'`.

- [ ] **Step 3: Implement the function**

Append to `nmr/analysis.py`:

```python
def benchmark_era_corr(
    frame: pl.DataFrame,
    benchmark_cols: Sequence[str],
    target_col: str,
    era_col: str = "era",
) -> dict:
    """Per-era CORR of benchmark models vs target.

    Lightweight context for the report (floors/ceilings), distinct from the
    full ``BenchmarkSuite`` harness. Degenerate eras (fewer than 2 non-null
    rows or constant target) are silently absent — ``n_eras`` reflects the
    actual era overlap.
    """
    from nmr.features import _per_era_pearson

    benchmark_list = list(benchmark_cols)
    if not benchmark_list:
        raise ValueError("benchmark_cols must contain at least one benchmark")
    corrs, degenerate = _per_era_pearson(frame, benchmark_list, target_col, era_col)
    rows = [
        {"era": era, "benchmark": b, "corr": float(vec[i])}
        for era, vec in corrs.items()
        if era not in degenerate
        for i, b in enumerate(benchmark_list)
    ]
    per_era = pl.DataFrame(
        rows,
        schema={"era": pl.Utf8, "benchmark": pl.Utf8, "corr": pl.Float64},
    )
    summary = (
        per_era.group_by("benchmark")
        .agg(
            pl.col("corr").mean().alias("mean_corr"),
            pl.col("corr").std().alias("corr_std"),
            pl.col("era").count().alias("n_eras"),
            pl.col("era").min().alias("first_era"),
            pl.col("era").max().alias("last_era"),
        )
        .sort("benchmark")
    )
    return {"benchmarks": summary, "per_era": per_era}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_analysis.py -q`
Expected: PASS (31 tests).

- [ ] **Step 5: Full suite + commit**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS.
```bash
git add nmr/analysis.py tests/test_analysis.py
git commit -m "feat(analysis): benchmark era correlation profile"
```

### Task 13: Analysis exports + module-level integration check

**Files:**
- Modify: `nmr/__init__.py`
- Test: `tests/test_analysis.py` (one real-data-cheap test)

**Interfaces:**
- Consumes: all Task 6–12 functions.
- Produces: package-root exports so scripts and notebooks use `nmr.*` symbols.

- [ ] **Step 1: Write the failing test (exports + real subset relation)**

Append to `tests/test_analysis.py`:

```python
import json

import nmr
from nmr.config import REPO_ROOT


def test_analysis_symbols_exported() -> None:
    for name in [
        "SplitStats",
        "describe_splits",
        "era_structure",
        "target_profile",
        "target_correlation_matrix",
        "feature_ic_screen",
        "feature_ic_by_era",
        "feature_summary",
        "FeatureCorrResult",
        "feature_correlation_structure",
        "within_set_redundancy",
        "cross_set_membership",
        "regime_analysis",
        "benchmark_era_corr",
    ]:
        assert name in nmr.__all__, name
        assert hasattr(nmr, name), name


def test_real_feature_sets_small_subset_medium_subset_all() -> None:
    """Cheap real-data guard: canonical sets nest (reads features.json only)."""
    features_json = REPO_ROOT / "data" / "v5.2" / "features.json"
    if not features_json.exists():
        pytest.skip("data/v5.2/features.json absent in this checkout")
    raw = json.loads(features_json.read_text(encoding="utf-8"))
    sets = raw["feature_sets"]
    small = set(sets["small"])
    medium = set(sets["medium"])
    all_ = set(sets["all"])
    assert small <= medium <= all_
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_analysis.py -q`
Expected: FAIL on `test_analysis_symbols_exported` (missing attributes).

- [ ] **Step 3: Add exports to package root**

Edit `nmr/__init__.py`: the file's imports are alphabetical by module — `.analysis` goes **first**:

```python
from .analysis import (
    FeatureCorrResult,
    SplitStats,
    benchmark_era_corr,
    cross_set_membership,
    describe_splits,
    era_structure,
    feature_correlation_structure,
    feature_ic_by_era,
    feature_ic_screen,
    feature_summary,
    regime_analysis,
    target_correlation_matrix,
    target_profile,
    within_set_redundancy,
)
```

Add all 14 names to `__all__` (sorted): `"FeatureCorrResult"`, `"SplitStats"`, `"benchmark_era_corr"`, `"cross_set_membership"`, `"describe_splits"`, `"era_structure"`, `"feature_correlation_structure"`, `"feature_ic_by_era"`, `"feature_ic_screen"`, `"feature_summary"`, `"regime_analysis"`, `"target_correlation_matrix"`, `"target_profile"`, `"within_set_redundancy"`.

Also verify the refresh exports from Tasks 2–3 are present in `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_analysis.py -q`
Expected: PASS (34 tests).

- [ ] **Step 5: Full suite + commit**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS.
```bash
git add nmr/__init__.py tests/test_analysis.py
git commit -m "feat(analysis): export analysis API at package root"
```

---

## Phase 3 — Scripts (`analyze_dataset.py`, `render_dataset_report.py`)

### Task 14: `analyze_dataset.py` — dumps + phase-boundary gate

**Files:**
- Create: `analyze_dataset.py` (repo root)
- Test: `tests/test_analyze_dataset.py` (new)

**Interfaces:**
- Consumes: `nmr.analysis` (all), `nmr.features.resolve_feature_sets`, `nmr.config.DataConfig`, `nmr.data.IngestionAgent`, `nmr.refresh.CURRENT_DATA_VERSION`, `nmr._atomicio.atomic_write_text`.
- Produces: `main(argv) -> int`; dumps under `artifacts/reports/dataset_analysis/`:
  `overview.json`, `era_structure.parquet`, `targets.json`, `target_corr.parquet`, `feature_summary.parquet`, `feature_ic_screen.parquet`, `feature_ic_by_era.parquet`, `feature_corr_medium.parquet`, `feature_corr_all_summary.json`, `set_membership.json`, `regimes.json`, `era_signal.parquet`, `benchmarks.json`, `manifest.json`.
- CLI: `--data-dir` (default `data`), `--version` (default `CURRENT_DATA_VERSION`), `--output-dir` (default `artifacts/reports/dataset_analysis`), `--features {small,medium,all}` (default `all`), `--max-eras` (default `None`), `--targets` (repeatable; default = primary 20D + primary 60D resolved from actual columns), `--all-targets` (flag), `--full-all-matrix` (flag: also persist the full matrix).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_analyze_dataset.py`:

```python
"""Integration tests for analyze_dataset.py on tiny synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import analyze_dataset


@pytest.fixture
def fake_data(tmp_path: Path) -> Path:
    """A minimal v5.2 data dir: features.json + tiny train/validation parquets."""
    d = tmp_path / "data" / "v5.2"
    d.mkdir(parents=True)
    (d / "features.json").write_text(
        json.dumps(
            {
                "feature_sets": {
                    "small": ["f_alpha", "f_beta"],
                    "medium": ["f_alpha", "f_beta", "f_gamma"],
                    "all": ["f_alpha", "f_beta", "f_gamma"],
                },
                "targets": ["target_alpha_20", "target_beta_60", "target"],
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for e in range(4):
        era = f"{e + 1:04d}"
        for i in range(10):
            rows.append(
                {
                    "era": era,
                    "id": f"{era}_{i}",
                    "f_alpha": float(i),
                    "f_beta": float(i % 3),
                    "f_gamma": float(10 - i),
                    "target": float(i),
                    "target_alpha_20": float(i),
                    "target_beta_60": float(9 - i),
                }
            )
    train = pl.DataFrame(rows[:20])
    valid = pl.DataFrame(rows[20:])
    train.write_parquet(d / "train.parquet")
    valid.write_parquet(d / "validation.parquet")
    (tmp_path / "data" / "numerai_era_data.csv").write_text(
        "date,dataset,start_era,end_era,round_id\n"
        "2026-08-08,train,0001,0002,\n"
        "2026-08-08,validation,0003,0004,\n"
        "2026-08-08,live,X,X,1300.0\n",
        encoding="utf-8",
    )
    return d.parent


def test_analyze_writes_all_dumps(tmp_path: Path, fake_data: Path) -> None:
    out = tmp_path / "dumps"
    rc = analyze_dataset.main(
        [
            "--data-dir", str(fake_data),
            "--output-dir", str(out),
            "--features", "small",
            "--max-eras", "3",
        ]
    )
    assert rc == 0
    expected = [
        "overview.json",
        "era_structure.parquet",
        "targets.json",
        "target_corr.parquet",
        "feature_summary.parquet",
        "feature_ic_screen.parquet",
        "feature_ic_by_era.parquet",
        "feature_corr_medium.parquet",
        "feature_corr_all_summary.json",
        "set_membership.json",
        "regimes.json",
        "era_signal.parquet",
        "benchmarks.json",
        "manifest.json",
    ]
    for name in expected:
        assert (out / name).exists(), name
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_version"] == "v5.2"
    assert manifest["feature_count"] == 2
    assert "generated_at" in manifest
    overview = json.loads((out / "overview.json").read_text(encoding="utf-8"))
    assert set(overview["splits"]) == {"train", "validation"}
    # benchmarks.json exists even without a benchmark parquet
    assert "benchmarks" in json.loads(
        (out / "benchmarks.json").read_text(encoding="utf-8")
    )


def test_analyze_deterministic_dumps(tmp_path: Path, fake_data: Path) -> None:
    out1 = tmp_path / "d1"
    out2 = tmp_path / "d2"
    analyze_dataset.main(
        ["--data-dir", str(fake_data), "--output-dir", str(out1), "--features", "small"]
    )
    analyze_dataset.main(
        ["--data-dir", str(fake_data), "--output-dir", str(out2), "--features", "small"]
    )
    for name in [
        "era_structure.parquet",
        "feature_summary.parquet",
        "feature_ic_screen.parquet",
        "feature_ic_by_era.parquet",
        "feature_corr_medium.parquet",
        "era_signal.parquet",
    ]:
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes(), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_analyze_dataset.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'analyze_dataset'`.

- [ ] **Step 3: Implement the script**

Create `analyze_dataset.py` (repo root):

```python
"""Deterministic dataset analysis -> machine-readable dumps.

Thin control plane: wires ``nmr.analysis`` functions over train+validation
and writes JSON/parquet dumps under ``artifacts/reports/dataset_analysis/``
for the report renderer. See docs/superpowers/specs/2026-08-08-dataset-analysis-design.md §4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import polars as pl

from nmr import analysis
from nmr._atomicio import atomic_write_text
from nmr.config import DataConfig
from nmr.data import IngestionAgent
from nmr.features import resolve_feature_sets
from nmr.refresh import CURRENT_DATA_VERSION


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics and write analysis dumps."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--version", default=CURRENT_DATA_VERSION)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "reports" / "dataset_analysis",
    )
    parser.add_argument("--features", choices=("small", "medium", "all"), default="all")
    parser.add_argument("--max-eras", type=int, default=None)
    parser.add_argument("--targets", action="append", default=None)
    parser.add_argument("--all-targets", action="store_true")
    parser.add_argument("--full-all-matrix", action="store_true")
    return parser.parse_args(argv)


def _atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".tmp.", suffix=".part"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.write_parquet(tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_write_json(payload: object, path: Path) -> None:
    """Atomic JSON write via the shared _atomicio text helper."""
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _era_chunks(
    agent: IngestionAgent,
    splits: Sequence[str],
    columns: Sequence[str],
    max_eras: int | None,
) -> list[pl.DataFrame]:
    """Collect era-partitioned chunks from the requested splits (Design A)."""
    frames = [agent.scan(split, columns=columns).collect() for split in splits]
    if max_eras is not None:
        frames = [
            f.filter(pl.col("era").cast(pl.Int64) <= max_eras) for f in frames
        ]
    if not frames:
        raise ValueError("no split frames to analyze")
    combined = pl.concat(frames)
    return combined.partition_by("era", maintain_order=True)


def _resolve_reference_targets(
    all_targets: list[str], explicit: list[str] | None, all_targets_flag: bool
) -> list[str]:
    if explicit:
        return explicit
    if all_targets_flag:
        return all_targets
    primary_20 = "target" if "target" in all_targets else all_targets[0]
    primary_60 = next(
        (t for t in all_targets if t.endswith("_60") and t != primary_20), primary_20
    )
    return [primary_20, primary_60]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    version_dir = args.data_dir / args.version
    features_path = version_dir / "features.json"
    if not features_path.exists():
        print(f"ERROR: {features_path} missing — run refresh_data.py first", file=sys.stderr)
        return 1

    feature_sets = resolve_feature_sets(features_path)
    all_targets = json.loads(features_path.read_text(encoding="utf-8"))["targets"]
    feature_cols = feature_sets[args.features]
    medium_cols = feature_sets["medium"]
    targets = _resolve_reference_targets(all_targets, args.targets, args.all_targets)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    config = DataConfig(version=args.version, feature_set=args.features, data_dir=args.data_dir)
    agent = IngestionAgent(config)
    splits = ("train", "validation")
    target_columns = [c for c in targets if c in agent.schema("train").names()]

    # overview + era structure
    overview_frames = {s: agent.scan(s, columns=["era", "id"]).collect() for s in splits}
    split_stats = analysis.describe_splits(overview_frames)
    _atomic_write_json(
        {
            "splits": {s: split_stats[s].__dict__ for s in splits},
            "feature_set": args.features,
            "n_features": len(feature_cols),
            "targets": all_targets,
            "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        },
        out / "overview.json",
    )
    _atomic_write_parquet(
        analysis.era_structure(pl.concat(list(overview_frames.values()))),
        out / "era_structure.parquet",
    )

    # full frame (Design A, single collection) for screen + IC + regimes
    full = _era_chunks(agent, splits, ["era", "id", *feature_cols, *target_columns], args.max_eras)
    full_frame = pl.concat(full)

    _atomic_write_json(
        {
            t: analysis.target_profile(full_frame, [t]).row(0, named=True)
            for t in target_columns
        },
        out / "targets.json",
    )
    _atomic_write_parquet(
        analysis.target_correlation_matrix(full_frame, target_columns),
        out / "target_corr.parquet",
    )
    ic_by_era = analysis.feature_ic_by_era(full_frame, feature_cols, target_columns[0])
    _atomic_write_parquet(ic_by_era, out / "feature_ic_by_era.parquet")
    _atomic_write_parquet(
        analysis.feature_ic_screen(full_frame, feature_cols, target_columns),
        out / "feature_ic_screen.parquet",
    )
    _atomic_write_parquet(
        analysis.feature_summary(full, feature_cols),
        out / "feature_summary.parquet",
    )

    # correlation structure: medium full matrix + selected-set summary
    if set(medium_cols) <= set(feature_cols):
        medium_chunks = full
    else:
        medium_chunks = _era_chunks(agent, splits, ["era", *medium_cols], args.max_eras)
    medium_result = analysis.feature_correlation_structure(medium_chunks, medium_cols)
    _atomic_write_parquet(medium_result.top_pairs, out / "feature_corr_medium.parquet")
    selected_result = analysis.feature_correlation_structure(full, feature_cols)
    selected_summary = dict(selected_result.summary)
    selected_summary["top_pairs"] = selected_result.top_pairs.to_dicts()
    _atomic_write_json(selected_summary, out / "feature_corr_all_summary.json")
    if args.full_all_matrix:
        _atomic_write_parquet(
            pl.DataFrame(selected_result.matrix),
            out / "feature_corr_all_matrix.parquet",
        )

    _atomic_write_json(
        {
            "sets": {k: {"n_features": len(v)} for k, v in feature_sets.items()},
            "subset_relations": analysis.cross_set_membership(feature_sets)[
                "subset_relations"
            ].to_dicts(),
        },
        out / "set_membership.json",
    )

    # regimes (reuses the ic_by_era computed above)
    regimes = analysis.regime_analysis(ic_by_era)
    _atomic_write_json(
        {
            "regime_thresholds": regimes["regime_thresholds"],
            "crash_eras": regimes["crash_eras"],
            "hot_eras": regimes["hot_eras"],
            "ic_persistence": regimes["ic_persistence"],
        },
        out / "regimes.json",
    )
    _atomic_write_parquet(regimes["era_signal"], out / "era_signal.parquet")

    # benchmarks + meta model (validation coverage; empty list when absent)
    bench_rows: list[dict] = []
    sources: list[pl.LazyFrame] = []
    bench_path = version_dir / "validation_benchmark_models.parquet"
    meta_path = version_dir / "meta_model.parquet"
    if bench_path.exists():
        sources.append(pl.scan_parquet(bench_path))
    if meta_path.exists():
        sources.append(pl.scan_parquet(meta_path))
    if sources:
        target_side = agent.scan(
            "validation", columns=["era", "id", *target_columns]
        ).collect()
        bench_frame = (
            pl.concat([s.collect() for s in sources], how="align")
            .join(target_side, on=["era", "id"], how="inner")
        )
        bench_cols = [
            c
            for c in bench_frame.columns
            if c.startswith("benchmark_") or "meta" in c.lower()
        ]
        if bench_cols:
            bench_rows = analysis.benchmark_era_corr(
                bench_frame, bench_cols, target_columns[0]
            )["benchmarks"].to_dicts()
    _atomic_write_json({"benchmarks": bench_rows}, out / "benchmarks.json")

    # manifest (generated_at informational only — never hashed)
    era_csv = args.data_dir / "numerai_era_data.csv"
    refresh_date = None
    if era_csv.exists():
        try:
            df = pl.read_csv(era_csv, try_parse_dates=False)
            refresh_date = str(df["date"].max())
        except Exception:
            refresh_date = None
    _atomic_write_json(
        {
            "data_version": args.version,
            "feature_set": args.features,
            "feature_count": len(feature_cols),
            "target_count": len(target_columns),
            "era_ranges": {
                s: overview_frames[s]["era"].min() + ".." + overview_frames[s]["era"].max()
                for s in splits
            },
            "refresh_date": refresh_date,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        },
        out / "manifest.json",
    )
    print(f"wrote analysis dumps to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_analyze_dataset.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Phase-boundary gate — real-data smoke**

Run (the spec's Phase 2→3 gate; must pass before the renderer task):

```bash
./.venv/Scripts/python analyze_dataset.py --features small --max-eras 5 --output-dir artifacts/reports/dataset_analysis_smoke
```

Expected: exit 0; all 14 dump files exist under `artifacts/reports/dataset_analysis_smoke/`; `manifest.json` parses; numeric dumps non-empty. Optionally repeat with `--features medium --max-eras 5` to validate the medium correlation path cheaply.

- [ ] **Step 6: Full suite + commit**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS.
```bash
git add analyze_dataset.py tests/test_analyze_dataset.py
git commit -m "feat(analysis): dataset analysis script with dumps"
```

### Task 15: `render_dataset_report.py` + LLM-optimized report

**Files:**
- Create: `render_dataset_report.py` (repo root)
- Test: `tests/test_report_render.py` (new)

**Interfaces:**
- Consumes: the Task 14 dumps; `nmr.refresh.CURRENT_DATA_VERSION`.
- Produces:
  - `render_report(manifest, overview, era_structure_rows, targets, target_corr_rows, feature_summary_rows, ic_screen_rows, regime, era_signal_rows, benchmark_rows, corr_summary, set_membership) -> str` — pure Markdown renderer (no file I/O), deterministic.
  - `main(argv) -> int` — loads dumps from the output dir, validates `manifest["data_version"] == CURRENT_DATA_VERSION`, writes `docs/04-research/dataset-analysis-YYYY-MM.md`.
- Report sections match the spec §5.2: front matter, `## 1. Dataset Overview` (incl. feature-set table), `## 2. Era Structure`, `## 3. Targets`, `## 4. Features`, `## 5. Regimes & Signal Dynamics`, `## 6. Benchmarks & Meta-Model`, `## 7. Modeling Implications`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_report_render.py`:

```python
"""Renderer tests: deterministic Markdown from synthetic dump content."""

from __future__ import annotations

import json

import render_dataset_report


def _fixture() -> dict:
    return {
        "manifest": {
            "data_version": "v5.2",
            "feature_set": "small",
            "feature_count": 2,
            "target_count": 1,
            "era_ranges": {"train": "0001..0002", "validation": "0003..0004"},
            "refresh_date": "2026-08-08",
        },
        "overview": {
            "splits": {
                "train": {"n_rows": 20, "n_eras": 2, "min_era": "0001", "max_era": "0002"},
                "validation": {"n_rows": 20, "n_eras": 2, "min_era": "0003", "max_era": "0004"},
            },
            "n_features": 2,
            "targets": ["target"],
            "feature_sets": {"small": 2, "medium": 3, "all": 3},
        },
        "era_structure_rows": [
            {"era": "0001", "n_rows": 10, "n_ids": 10, "gap": False},
            {"era": "0002", "n_rows": 10, "n_ids": 10, "gap": False},
        ],
        "targets": {
            "target": {
                "n_eras_present": 4,
                "missing_rate": 0.0,
                "pooled_mean": 0.0,
                "pooled_std": 1.0,
            }
        },
        "target_corr_rows": [
            {"target_a": "target", "target_b": "target", "mean_corr": 1.0, "n_eras": 4}
        ],
        "feature_summary_rows": [
            {"feature": "f1", "pooled_mean": 0.1, "pooled_std": 1.0, "missing_rate": 0.0}
        ],
        "ic_screen_rows": [
            {"feature": "f1", "target": "target", "mean_corr": 0.05, "n_eras": 4, "stable": True}
        ],
        "regime": {
            "regime_thresholds": {"q1": -0.01, "q3": 0.01},
            "crash_eras": ["0001"],
            "hot_eras": ["0004"],
            "ic_persistence": {"mean": 0.5, "std": 0.1, "n_adjacent": 3},
        },
        "era_signal_rows": [
            {"era": "0001", "mean_ic": -0.02, "regime": "low", "crash": True, "hot": False},
            {"era": "0002", "mean_ic": 0.0, "regime": "normal", "crash": False, "hot": False},
        ],
        "benchmark_rows": [
            {"benchmark": "benchmark_small", "mean_corr": 0.03, "n_eras": 4}
        ],
        "corr_summary": {"mean_abs_corr": 0.2, "top_pairs": []},
        "set_membership": {"sets": {"small": {"n_features": 2}}},
    }


def test_render_report_deterministic() -> None:
    md1 = render_dataset_report.render_report(**{**_fixture()})
    md2 = render_dataset_report.render_report(**{**_fixture()})
    assert md1 == md2


def test_render_report_structure() -> None:
    md = render_dataset_report.render_report(**{**_fixture()})
    assert md.startswith("# Dataset Analysis")
    for header in ["## 1. Dataset Overview", "## 2. Era Structure", "## 3. Targets",
                   "## 4. Features", "## 5. Regimes & Signal Dynamics",
                   "## 6. Benchmarks & Meta-Model", "## 7. Modeling Implications"]:
        assert header in md
    # schema blocks precede tables; takeaways present
    assert "| Column |" not in md  # schema is prose, tables use plain headers
    assert "Key takeaways" in md


def test_render_report_escapes_markdown_special_chars() -> None:
    fx = _fixture()
    fx["feature_summary_rows"] = [
        {"feature": "feat|name", "pooled_mean": 0.1, "pooled_std": 1.0, "missing_rate": 0.0}
    ]
    md = render_dataset_report.render_report(**fx)
    assert "feat\\|name" in md  # escaped pipe inside table cell


def test_main_rejects_version_mismatch(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"data_version": "v5.3", "feature_count": 2}), encoding="utf-8"
    )
    rc = render_dataset_report.main(
        ["--dumps-dir", str(tmp_path), "--output", str(tmp_path / "out.md")]
    )
    assert rc == 1
    assert not (tmp_path / "out.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_report_render.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'render_dataset_report'`.

- [ ] **Step 3: Implement the renderer**

Create `render_dataset_report.py` (repo root):

```python
"""LLM-optimized dataset analysis report renderer.

Reads the dumps from analyze_dataset.py and renders a dense, schema-annotated
Markdown report under docs/04-research/. Pure formatting: every number comes
from the dumps; the same dumps produce byte-identical Markdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from nmr.refresh import CURRENT_DATA_VERSION


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(columns: list[str], rows: list[dict]) -> str:
    """Dense pipe table; pipes inside cell values are escaped."""
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for c in columns:
            value = str(_fmt(row.get(c))).replace("|", "\\|")
            cells.append(value)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _schema_block(text: str) -> str:
    return f"**Schema:** {text}"


def render_report(
    manifest: dict,
    overview: dict,
    era_structure_rows: list[dict],
    targets: dict,
    target_corr_rows: list[dict],
    feature_summary_rows: list[dict],
    ic_screen_rows: list[dict],
    regime: dict,
    era_signal_rows: list[dict],
    benchmark_rows: list[dict],
    corr_summary: dict,
    set_membership: dict,
) -> str:
    """Render the full report. Deterministic given identical inputs."""
    out: list[str] = []
    out.append("# Dataset Analysis — Numerai " + manifest["data_version"])
    out.append("")
    out.append("> Generated from `artifacts/reports/dataset_analysis/` dumps. "
               "All numbers have full precision in the dumps; tables are display-rounded. "
               "Schema lines precede every table.")
    out.append("")
    out.append(f"- Data version: `{manifest['data_version']}`")
    out.append(f"- Feature set: `{manifest['feature_set']}` ({manifest['feature_count']} features)")
    out.append(f"- Refresh date: `{manifest.get('refresh_date')}`")
    out.append(f"- Era ranges: train `{manifest['era_ranges'].get('train')}`, "
               f"validation `{manifest['era_ranges'].get('validation')}`")
    out.append("")

    out.append("## 1. Dataset Overview")
    out.append("")
    out.append(_schema_block("split | n_rows | n_eras | min_era | max_era"))
    out.append("")
    out.append(_table(
        ["split", "n_rows", "n_eras", "min_era", "max_era"],
        [{"split": k, **v} for k, v in overview["splits"].items()],
    ))
    out.append("")
    out.append(_schema_block("feature_set | n_features"))
    out.append("")
    out.append(_table(
        ["feature_set", "n_features"],
        [{"feature_set": k, "n_features": v} for k, v in overview.get("feature_sets", {}).items()],
    ))
    out.append("")
    out.append("- **Key takeaways:** the tournament is a per-era cross-section of obfuscated "
               "equities; eras are the unit of evaluation. Never pool rows across eras for "
               "metrics.")
    out.append("")

    out.append("## 2. Era Structure")
    out.append("")
    out.append(_schema_block("era | era_index | n_rows | n_ids | gap (non-consecutive era label)"))
    out.append("")
    out.append(_table(["era", "era_index", "n_rows", "n_ids", "gap"], era_structure_rows))
    out.append("")
    out.append("- **Key takeaways:** gaps in the era index are data anomalies; the "
               "train→validation boundary is a distribution-shift checkpoint.")
    out.append("")

    out.append("## 3. Targets")
    out.append("")
    out.append(_schema_block("target | n_eras_present | missing_rate | pooled_mean | pooled_std"))
    out.append("")
    out.append(_table(
        ["target", "n_eras_present", "missing_rate", "pooled_mean", "pooled_std"],
        [{"target": k, **v} for k, v in targets.items()],
    ))
    out.append("")
    out.append(_schema_block("target_a | target_b | mean_corr | n_eras (per-era Spearman, equal-era-weighted)"))
    out.append("")
    out.append(_table(["target_a", "target_b", "mean_corr", "n_eras"], target_corr_rows))
    out.append("")
    out.append("- **Key takeaways:** targets are integer ranks 0..5; auxiliary targets have "
               "staggered era availability — check `n_eras_present` before training on them.")
    out.append("")

    out.append("## 4. Features")
    out.append("")
    out.append(_schema_block("feature | pooled_mean | pooled_std | missing_rate"))
    out.append("")
    out.append(_table(
        ["feature", "pooled_mean", "pooled_std", "missing_rate"], feature_summary_rows
    ))
    out.append("")
    out.append(_schema_block("feature | target | mean_corr | n_eras | stable (per-era Pearson IC)"))
    out.append("")
    out.append(_table(
        ["feature", "target", "mean_corr", "n_eras", "stable"], ic_screen_rows
    ))
    out.append("")
    out.append(f"- Mean |pairwise corr|: `{_fmt(corr_summary.get('mean_abs_corr'))}`; "
               "top pairs in `feature_corr_medium.parquet` / `feature_corr_all_summary.json`.")
    out.append("- **Key takeaways:** prefer features passing the stability screen "
               "(`stable=True`); avoid highly redundant families.")
    out.append("")

    out.append("## 5. Regimes & Signal Dynamics")
    out.append("")
    out.append(_schema_block("era | mean_ic | regime | crash | hot"))
    out.append("")
    out.append(_table(["era", "mean_ic", "regime", "crash", "hot"], era_signal_rows))
    out.append("")
    out.append(f"- Crash eras (bottom decile): `{regime.get('crash_eras')}`")
    out.append(f"- Hot eras (top decile): `{regime.get('hot_eras')}`")
    out.append(f"- Adjacent-era IC persistence: mean "
               f"`{_fmt(regime.get('ic_persistence', {}).get('mean'))}`, "
               f"n `{regime.get('ic_persistence', {}).get('n_adjacent')}`")
    out.append("- **Key takeaways:** signal is regime-dependent; expect IC mean-reversion — "
               "never tune on crash eras in-sample.")
    out.append("")

    out.append("## 6. Benchmarks & Meta-Model")
    out.append("")
    out.append(_schema_block("benchmark | mean_corr | n_eras"))
    out.append("")
    out.append(_table(["benchmark", "mean_corr", "n_eras"], benchmark_rows))
    out.append("")
    out.append("- **Key takeaways:** benchmark models define the achievable floor; the meta "
               "model is the upper reference. Stay above the best benchmark after neutralization.")
    out.append("")

    out.append("## 7. Modeling Implications")
    out.append("")
    out.append("- Validate **era-grouped with purge** (8 eras for 20D, 16 for 60D); "
               "random row-level CV is leakage.")
    out.append("- Rank-gaussianize per era before ensembling; never blend raw outputs.")
    out.append("- Select features from the stability screen, not pooled correlation.")
    out.append("- Watch auxiliary-target era coverage before including them.")
    out.append("")
    return "\n".join(out)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the dataset analysis report.")
    parser.add_argument(
        "--dumps-dir",
        type=Path,
        default=Path("artifacts") / "reports" / "dataset_analysis",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    d = args.dumps_dir

    manifest = _load_json(d / "manifest.json")
    if manifest["data_version"] != CURRENT_DATA_VERSION:
        print(
            f"ERROR: dumps data_version {manifest['data_version']} != "
            f"CURRENT_DATA_VERSION {CURRENT_DATA_VERSION}",
            file=sys.stderr,
        )
        return 1
    required_dumps = (
        "overview.json", "era_structure.parquet", "targets.json",
        "target_corr.parquet", "feature_summary.parquet",
        "feature_ic_screen.parquet", "regimes.json", "era_signal.parquet",
        "benchmarks.json", "feature_corr_all_summary.json", "set_membership.json",
    )
    for name in required_dumps:
        if not (d / name).exists():
            print(f"ERROR: missing dump {d / name}", file=sys.stderr)
            return 1

    md = render_report(
        manifest=manifest,
        overview=_load_json(d / "overview.json"),
        era_structure_rows=pl.read_parquet(d / "era_structure.parquet").to_dicts(),
        targets=_load_json(d / "targets.json"),
        target_corr_rows=pl.read_parquet(d / "target_corr.parquet").to_dicts(),
        feature_summary_rows=pl.read_parquet(d / "feature_summary.parquet").to_dicts(),
        ic_screen_rows=pl.read_parquet(d / "feature_ic_screen.parquet").to_dicts(),
        regime=_load_json(d / "regimes.json"),
        era_signal_rows=pl.read_parquet(d / "era_signal.parquet").to_dicts(),
        benchmark_rows=_load_json(d / "benchmarks.json").get("benchmarks", []),
        corr_summary=_load_json(d / "feature_corr_all_summary.json"),
        set_membership=_load_json(d / "set_membership.json"),
    )
    refresh_date = manifest.get("refresh_date") or "0000-00"
    output = args.output or (
        Path("docs") / "04-research" / f"dataset-analysis-{refresh_date[:7]}.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_report_render.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Full suite + commit**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS.
```bash
git add render_dataset_report.py tests/test_report_render.py
git commit -m "feat(analysis): LLM-optimized report renderer"
```

---

## Phase 4 — Production run & deliverable

### Task 16: Refresh, full analysis, render, commit the report

**Files:**
- Create: `docs/04-research/dataset-analysis-2026-08.md` (the deliverable)
- Modify: `docs/DOCS_README.md` (register the report in the master map)

**Interfaces:** none — execution + documentation.

- [ ] **Step 1: Real data refresh**

Run: `./.venv/Scripts/python refresh_data.py --dry-run` first (network check, no writes).
Then run the real refresh: `./.venv/Scripts/python refresh_data.py`
Note: this downloads `live.parquet` (~20 MB) plus weekly-expanding files — `validation.parquet` is ~4 GB. If bandwidth is a concern, use `--live-only` and record the limitation in the report's provenance. **This is the only step needing network; confirm with the user before the heavy download.**

- [ ] **Step 2: Full analysis run**

Run:
```bash
./.venv/Scripts/python analyze_dataset.py --features all --output-dir artifacts/reports/dataset_analysis
```
Expected: exit 0; runtime minutes-to-1h (document the actual wall time truthfully in the report's provenance). All 14 dumps present.

- [ ] **Step 3: Render + verify**

Run: `./.venv/Scripts/python render_dataset_report.py`
Expected: `docs/04-research/dataset-analysis-2026-08.md` written. Read it end-to-end: every table has a schema line, every section has key takeaways, no `null` where a number is expected, the manifest block matches the actual dumps.

- [ ] **Step 4: Register in the docs map**

In `docs/DOCS_README.md`, add the report to the appropriate research tier with a one-line description and its refresh cadence (re-run `analyze_dataset.py` + `render_dataset_report.py` after each `refresh_data.py`).

- [ ] **Step 5: Commit the deliverable**

```bash
git add docs/04-research/dataset-analysis-2026-08.md docs/DOCS_README.md
git commit -m "docs: add dataset analysis report (2026-08)"
```

---

## Phase 5 — Full verification

### Task 17: Complete verification gate

**Files:** none (verification only).

- [ ] **Step 1: Full test suite**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS — existing 413 + new refresh/analysis/render/script tests.

- [ ] **Step 2: Refresh smoke (offline)**

Run: `./.venv/Scripts/python refresh_data.py --dry-run --check-only`
Expected: exit 0 or 3 per the contract, no writes (verify `data/numerai_era_data.csv` mtime unchanged).

- [ ] **Step 3: SSOT doc sweep**

Run: `git diff --stat` and grep the four SSOT docs for stale references. Confirm:
- `README.md` documents `refresh_data.py`.
- `ARCHITECTURE.md` documents the era CSV schema + new modules.
- `AGENTS.md` toolkit table lists refresh/analyze/render.
- `docs/DOCS_README.md` registers the report.
- No duplicated version strings: `CURRENT_DATA_VERSION` is the single source (drift-guard test enforces it).

- [ ] **Step 4: Final report**

Write the review-format summary (Task Summary, Affected Files, Architecture, Tests, Execution Verification, Risks & Follow-ups) in the completion message. Do not claim any test passed without running it.

---

## Self-Review (completed at plan-writing time)

- **Spec coverage:** every spec section maps to a task — refresh policy (§3 → Tasks 2–4), era CSV + integrity + version alert (§3.2–3.4 → Tasks 2–4), analysis functions (§4.1 → Tasks 6–12), semantics (§4.2 → embedded in Tasks 7, 9, 10, 11), dumps (§4.3 → Task 14), renderer + report (§5 → Task 15), docs (§6 → Tasks 5, 16), gates (§7 → Tasks 14 step 5, 17). The `--max-eras`/`--all-targets`/`--full-all-matrix` flags and the `benchmark_era_corr` overlap semantics are in Tasks 14/12.
- **Placeholder scan:** no TBD/TODO; every code step contains full code. The Task 4 `--live-only` ledger edge is resolved in the code itself (ranges read from disk, ledger written only when all three parquets exist) — no fix-note needed.
- **Type consistency:** `_per_era_pearson` returns `(dict, set)` everywhere it is consumed (Tasks 1, 8, 12); `FeatureCorrResult` fields used consistently (Task 10 ↔ Task 14); `feature_ic_by_era` schema `(era, feature, ic, degenerate)` consumed by Task 11; dump filenames in Task 14 match the Task 15 loader list; `render_report`'s 12 keyword parameters match the `main()` loader and the test fixture exactly.
- **Known deltas from spec (intentional):** `_per_era_pearson` returns a tuple (spec said dict) to carry the degenerate-era set; `era_structure` sorts by int-cast era labels; Task 4 default refresh re-downloads expanding files on round advance (the spec's `--live-only` is the escape hatch); the report's `## 5/6` section content is assembled from the spec's §5.2 with feature sets folded into §1.
