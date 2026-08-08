# Agentic Research Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the agentic-research harness to `nmr`: two new tested modules (`nmr/features.py`, `nmr/meta.py`), one campaign-orchestration module (`nmr/campaign.py`) plus a thin `run_campaign.py` CLI, one deliberate config-schema addition (`DataConfig.feature_subset`), four Kimi skills (S1–S4), and same-commit doc updates (AGENTS/ARCHITECTURE/README/example.yaml).

**Architecture:** Everything lives in the tested `nmr/` boundary; scripts and skills are thin control planes/protocols over existing deterministic machinery. T1 (`features.py`) makes feature-subset campaigns executable (the closed `VALID_FEATURE_SETS` enum otherwise blocks them). T2 (`meta.py`) adds the missing statistical decision layer (paired era comparison, promotion verdict, fleet summary) on top of `block_bootstrap_ci`/`MIN_OVERLAP_ERAS`. T3 (`campaign.py` + `run_campaign.py`) gives agents a deterministic, auditable batch entrypoint with atomic trial-lineage logs. No metric math, no fold geometry, no determinism-core changes.

**Tech Stack:** Python 3.11+, Polars, NumPy/SciPy; pytest (TDD). No new dependencies.

## Global Constraints

- `nmr/` is the only tested boundary; scripts and skills contain zero business logic (AGENTS §2.1).
- TDD: no production code without a failing test first (AGENTS §7). Every new function has a test that was watched fail.
- No new third-party dependencies (AGENTS §3); no NN/torch backend (deployment decision).
- No metric-formula changes, no purge-geometry changes; purge floor (8/16 eras) is protocol-enforced in skills, not code.
- Canonical hashes exclude wall-clock and absolute paths; nothing added here enters `canonical_scorecards_bytes`.
- Doc SSOT: any change to a documented fact updates its owner file in the same change set (AGENTS Self-Update Directive; the four files: AGENTS.md / ARCHITECTURE.md / CONTRIBUTING.md / README.md). No fact duplicated across files — cross-reference instead.
- **Git flow (user-authorized 2026-08-08):** implementation happens on the `agentic-harness` branch; each task's final step is a commit on that branch with a conventional message. `main` is never written to; merging/pushing requires explicit user approval. The plan's "Record" steps below are realized as commits.
- Real-data tests require `data/v5.2/` assets (present in this repo); synthetic-data tests follow the `tests/test_runner.py` pattern (tmp `vtest` version dir, tiny frames, `fast` preset, few trees).
- Full-suite gate is `pytest -q`; benchmark smoke is `benchmark_runner.py --fast-mode` (only for data/evaluation/scorecard-touching changes).
- Verification honesty: run the commands, read the output, report truthfully.

## File Structure

| File | Responsibility |
|---|---|
| `nmr/features.py` (new) | Feature-set resolution (pure fn of `features.json`) + stability screen + stable-feature selection |
| `nmr/meta.py` (new) | Paired era comparison (block bootstrap), promotion verdict, fleet summary |
| `nmr/campaign.py` (new) | Campaign id hashing, campaign log schema, atomic log writer |
| `run_campaign.py` (new, root) | Thin CLI over `nmr/campaign.py` + `ExperimentRunner` + `RunRegistry` |
| `nmr/config.py` (modify) | `DataConfig.feature_subset: str \| None = None` + `resolved_feature_set` property |
| `nmr/runner.py` (modify) | Public `compute_run_id` accessor; thread `resolved_feature_set` |
| `nmr/research.py` (modify) | Thread `resolved_feature_set` in `_held_out_metric` |
| `nmr/__init__.py` (modify) | Export new public symbols (imports + `__all__`) |
| `tests/test_features.py` (new) | T1 tests |
| `tests/test_meta.py` (new) | T2 tests |
| `tests/test_campaign.py` (new) | T3 logic tests |
| `tests/test_runner.py`, `tests/test_config.py`, `tests/test_scripts.py` (modify) | Additive cases |
| `configs/example.yaml` (modify) | Document `feature_subset` |
| `ARCHITECTURE.md`, `AGENTS.md`, `README.md` (modify) | SSOT updates |
| `.superpowers/skills/feature-campaign/SKILL.md` etc. (new, location per writing-skills conventions) | S1–S4 skills |

---

### Task 1: `nmr/features.py` — `resolve_feature_sets`

**Files:**
- Create: `nmr/features.py` (only `resolve_feature_sets` for now)
- Test: `tests/test_features.py` (new file)

**Interfaces:**
- Consumes: nothing from this plan; stdlib `json` + `pathlib`.
- Produces: `resolve_feature_sets(features_json: Path) -> dict[str, list[str]]` — consumed by Task 9 (exports) and the S1 skill.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_features.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'nmr.features'`.

- [ ] **Step 3: Write minimal implementation**

```python
# nmr/features.py
"""Feature-set resolution and stability screening for research campaigns.

Pure functions over ``features.json`` and the train frame; no model logic and
no file state beyond the explicit ``features_json`` argument. Derived subsets
must remain pure functions of their inputs so the run_id fingerprint (config +
data_version + ``nmr/*.py`` + env) is unchanged by subset selection.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["resolve_feature_sets"]


def resolve_feature_sets(features_json: Path) -> dict[str, list[str]]:
    """Return every named feature set in ``features.json``, deterministically ordered.

    Includes the canonical sets (small/medium/all) and the obfuscated family
    sets (intelligence, charisma, sunshine, ...) exactly as declared. Pure
    function of the file contents; values are defensive copies.
    """
    path = Path(features_json)
    raw = json.loads(path.read_text(encoding="utf-8"))
    sets = raw.get("feature_sets")
    if not isinstance(sets, dict) or not sets:
        raise ValueError(f"{path}: 'feature_sets' must be a non-empty mapping")
    result: dict[str, list[str]] = {}
    for name, values in sorted(sets.items()):
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(
                f"{path}: feature set {name!r} must be a list of strings"
            )
        result[name] = list(values)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python -m pytest tests/test_features.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Record** — task deliverable: `nmr/features.py` + `tests/test_features.py` on disk.

---

### Task 2: `nmr/features.py` — `feature_stability_screen` + `select_stable_features`

**Files:**
- Modify: `nmr/features.py` (append two functions + threshold constants)
- Test: `tests/test_features.py` (append)

**Interfaces:**
- Consumes: `resolve_feature_sets` (Task 1, not called by these functions — screen takes an explicit `frame`).
- Produces:
  - `feature_stability_screen(frame: pl.DataFrame, *, feature_cols: Sequence[str], target_col: str, era_col: str = "era", min_mean_corr: float = DEFAULT_MIN_MEAN_CORR, max_abs_decay: float = DEFAULT_MAX_ABS_DECAY) -> pl.DataFrame` — rows per feature: `feature, mean_corr, corr_std, decay_slope, cross_regime_variance, n_eras, stable`.
  - `select_stable_features(screen: pl.DataFrame, *, min_mean_corr: float, max_abs_decay: float) -> list[str]`
  - `DEFAULT_MIN_MEAN_CORR = 0.01`, `DEFAULT_MAX_ABS_DECAY = 0.001` (module-level named constants — AGENTS §6).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_features.py
import polars as pl

from nmr.features import (
    DEFAULT_MAX_ABS_DECAY,
    DEFAULT_MIN_MEAN_CORR,
    feature_stability_screen,
    select_stable_features,
)


def _screen_frame() -> pl.DataFrame:
    """f_good: per-era CORR ~ +1 with zero decay; f_bad: CORR ~ -1 decaying to 0."""
    rows: list[dict] = []
    for era in range(1, 21):
        for idx in range(50):
            rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f_good": idx * 0.02,                      # CORR +1 all eras
                    "f_bad": -idx * (0.02 - era * 0.001),      # CORR ~ -1 -> 0 (decay)
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


def test_screen_handles_degenerate_eras_without_raising() -> None:
    rows = [
        {"era": "1", "id": "a", "f": 1.0, "target": 0.5},   # 1 row: degenerate
        {"era": "1", "id": "b", "f": 1.0, "target": 0.5},   # zero variance
        {"era": "2", "id": "c", "f": float("nan"), "target": 0.5},  # non-finite
        {"era": "3", "id": "d", "f": 0.2, "target": 0.9},
    ]
    frame = pl.DataFrame(rows)
    screen = feature_stability_screen(frame, feature_cols=["f"], target_col="target")
    assert screen.height == 1
    row = screen.row(0, named=True)
    assert row["n_eras"] == 3
    assert screen.get_column("stable").to_list() == [False]


def test_select_stable_features_filters_on_thresholds() -> None:
    screen = feature_stability_screen(
        _screen_frame(), feature_cols=["f_good", "f_bad"], target_col="target"
    )
    kept = select_stable_features(screen, min_mean_corr=-1.0, max_abs_decay=1.0)
    assert kept == ["f_bad", "f_good"]  # sorted; both pass loose thresholds
    strict = select_stable_features(screen, min_mean_corr=0.9, max_abs_decay=0.01)
    assert strict == ["f_good"]


def test_select_stable_features_rejects_screen_without_required_columns() -> None:
    bad = pl.DataFrame({"feature": ["f1"], "mean_corr": [0.5]})
    with pytest.raises(ValueError, match="decay_slope"):
        select_stable_features(bad, min_mean_corr=0.0, max_abs_decay=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_features.py -q`
Expected: FAIL with `ImportError: cannot import name 'feature_stability_screen'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to nmr/features.py
from collections.abc import Sequence

import numpy as np
import polars as pl

DEFAULT_MIN_MEAN_CORR = 0.01
DEFAULT_MAX_ABS_DECAY = 0.001

_SCREEN_COLUMNS = (
    "feature", "mean_corr", "corr_std", "decay_slope",
    "cross_regime_variance", "n_eras", "stable",
)


def feature_stability_screen(
    frame: pl.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    era_col: str = "era",
    min_mean_corr: float = DEFAULT_MIN_MEAN_CORR,
    max_abs_decay: float = DEFAULT_MAX_ABS_DECAY,
) -> pl.DataFrame:
    """Per-feature era-window CORR, decay, and cross-regime drift statistics.

    Definition (ARCHITECTURE.md §P): per-era Pearson CORR(feature, target)
    using the same vectorized per-era pattern as ``feature_exposure_report``;
    degenerate eras (zero variance, <2 usable rows, non-finite values)
    contribute 0.0. Aggregates across eras: ``mean_corr`` (mean), ``corr_std``
    (population std), ``decay_slope`` (linear slope of CORR vs era index),
    ``cross_regime_variance`` (variance of first-half vs second-half era-window
    mean CORR — a regime-drift proxy). ``stable`` is True when
    ``mean_corr >= min_mean_corr`` and ``|decay_slope| <= max_abs_decay`` and
    ``n_eras >= 2``.
    """
    feature_list = list(feature_cols)
    if not feature_list:
        raise ValueError("feature_cols must contain at least one feature")
    required = {era_col, target_col, *feature_list}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing required columns: {sorted(missing)}")

    per_era: dict[str, np.ndarray] = {}
    for part in frame.select([era_col, target_col, *feature_list]).partition_by(
        era_col, maintain_order=True
    ):
        era = str(part.get_column(era_col).to_list()[0])
        clean = part.drop_nulls()
        if clean.height < 2:
            per_era[era] = np.zeros(len(feature_list), dtype=float)
            continue
        target = clean.get_column(target_col).cast(pl.Float64).to_numpy()
        features = clean.select(feature_list).cast(pl.Float64).to_numpy()
        per_era[era] = _feature_target_pearson(features, target)

    if not per_era:
        return pl.DataFrame(
            {name: [] for name in _SCREEN_COLUMNS}
        )

    eras = sorted(per_era, key=int)
    matrix = np.column_stack([per_era[era] for era in eras])
    rows = []
    for i, feature in enumerate(feature_list):
        series = matrix[i]
        era_index = np.arange(len(eras), dtype=float)
        slope = (
            float(np.polyfit(era_index, series, 1)[0]) if len(series) >= 2 else 0.0
        )
        mid = len(series) // 2
        first = float(np.mean(series[:mid])) if mid > 0 else 0.0
        second = float(np.mean(series[mid:])) if len(series) - mid > 0 else 0.0
        cross_regime = 0.25 * (first - second) ** 2
        mean_corr = float(np.mean(series))
        stable = (
            mean_corr >= min_mean_corr
            and abs(slope) <= max_abs_decay
            and len(series) >= 2
        )
        rows.append(
            {
                "feature": feature,
                "mean_corr": mean_corr,
                "corr_std": float(np.std(series, ddof=0)),
                "decay_slope": slope,
                "cross_regime_variance": cross_regime,
                "n_eras": int(len(series)),
                "stable": stable,
            }
        )
    return pl.DataFrame(rows, schema=_SCREEN_COLUMNS)


def select_stable_features(
    screen: pl.DataFrame,
    *,
    min_mean_corr: float,
    max_abs_decay: float,
) -> list[str]:
    """Return the sorted stable feature names passing both thresholds."""
    required = {"feature", "mean_corr", "decay_slope", "stable", "n_eras"}
    missing = required - set(screen.columns)
    if missing:
        raise ValueError(f"screen missing required columns: {sorted(missing)}")
    kept = screen.filter(
        (pl.col("mean_corr") >= min_mean_corr)
        & (pl.col("decay_slope").abs() <= max_abs_decay)
        & (pl.col("n_eras") >= 2)
        & (pl.col("stable") == True)  # noqa: E712  (explicit boolean literal)
    )
    return sorted(kept.get_column("feature").to_list())


def _feature_target_pearson(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_centered = target - np.mean(target)
    target_norm = float(np.linalg.norm(target_centered))
    if target_norm == 0.0:
        return np.zeros(features.shape[1], dtype=float)
    feature_centered = features - np.mean(features, axis=0)
    denoms = np.linalg.norm(feature_centered, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrs = (feature_centered.T @ target_centered) / (denoms * target_norm)
    return np.where(np.isfinite(corrs), corrs, 0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python -m pytest tests/test_features.py -q`
Expected: PASS (all tests in file).

- [ ] **Step 5: Record** — deliverable on disk.

---

### Task 3: `DataConfig.feature_subset` + threading

**Files:**
- Modify: `nmr/config.py:50-71` (DataConfig), `nmr/runner.py:71`, `nmr/research.py:207`
- Test: `tests/test_config.py` (append), `tests/test_features.py` (append), `tests/test_runner.py` (append)

**Interfaces:**
- Consumes: `resolve_feature_sets` result names are the valid `feature_subset` values (validated at ingestion, not load).
- Produces: `DataConfig.feature_subset: str | None = None`; `DataConfig.resolved_feature_set -> str` (subset wins over `feature_set`). Consumed by S1 skill and Tasks 4+ (manifest reads).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_config.py
def test_feature_subset_overrides_feature_set_in_resolution():
    from nmr.config import DataConfig

    cfg = DataConfig(feature_set="small", feature_subset="sunshine")
    assert cfg.resolved_feature_set == "sunshine"
    plain = DataConfig(feature_set="small")
    assert plain.resolved_feature_set == "small"


def test_feature_subset_must_be_non_empty_when_provided():
    from nmr.config import DataConfig

    import pytest as _pytest

    with _pytest.raises(ValueError, match="feature_subset"):
        DataConfig(feature_subset="")
```

```python
# append to tests/test_features.py
import polars as pl

from nmr.config import DataConfig, ExperimentConfig
from nmr.data import IngestionAgent


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
        {"era": ["1", "1"], "id": ["a", "b"], "f1": [0.1, 0.2], "f2": [0.3, 0.4],
         "f3": [0.5, 0.6], "target": [0.2, 0.3]}
    ).write_parquet(version_dir / "train.parquet")


def test_ingestion_resolves_feature_subset_from_features_json(tmp_path) -> None:
    _write_features_with_sunshine(tmp_path)
    cfg = DataConfig(
        version="vtest", feature_set="small", feature_subset="sunshine",
        data_dir=tmp_path,
    )
    agent = IngestionAgent(cfg)
    assert agent.features() == ["f1", "f2", "f3"]  # resolved_feature_set threaded


def test_ingestion_rejects_unknown_feature_subset_with_valid_options(tmp_path) -> None:
    _write_features_with_sunshine(tmp_path)
    cfg = DataConfig(
        version="vtest", feature_set="small", feature_subset="nope", data_dir=tmp_path,
    )
    agent = IngestionAgent(cfg)
    with pytest.raises(ValueError, match="sunshine"):
        agent.features()
```

```python
# append to tests/test_runner.py
def test_feature_subset_changes_run_id_and_uses_subset_features(tmp_path) -> None:
    """feature_subset must change the run fingerprint and reach the data layer."""
    import json as _json

    cfg = _config(tmp_path)
    # vtest features.json has small == medium == all == [f1, f2]; add a family
    # set via the data dir used by _config and re-run with feature_subset.
    version_dir = cfg.data.data_dir / "vtest"
    features = _json.loads((version_dir / "features.json").read_text(encoding="utf-8"))
    features["feature_sets"]["sunshine"] = ["f1", "f2"]
    (version_dir / "features.json").write_text(_json.dumps(features), encoding="utf-8")

    plain = ExperimentRunner(cfg)
    subset_cfg = ExperimentConfig(
        data=DataConfig(
            version=cfg.data.version, feature_set=cfg.data.feature_set,
            feature_subset="sunshine", targets=cfg.data.targets,
            data_dir=cfg.data.data_dir,
        ),
        split=cfg.split, model=cfg.model, evaluation=cfg.evaluation, run=cfg.run,
    )
    subset = ExperimentRunner(subset_cfg)
    assert plain._run_id != subset._run_id
    assert subset.run(deploy=False).manifest["feature_cols"] == ["f1", "f2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_config.py tests/test_features.py tests/test_runner.py -q`
Expected: FAIL — `feature_subset` is not a known `DataConfig` field (TypeError from dataclass construction).

- [ ] **Step 3: Write minimal implementation**

```python
# nmr/config.py — DataConfig
    feature_set: str = "small"
    feature_subset: str | None = None
    targets: tuple[str, ...] = ("target",)
    data_dir: Path = REPO_ROOT / "data"

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "data_dir", _resolve_path(self.data_dir))
        if self.feature_set not in VALID_FEATURE_SETS:
            raise ValueError(
                f"feature_set={self.feature_set!r} not in {VALID_FEATURE_SETS}"
            )
        if self.feature_subset is not None and not self.feature_subset:
            raise ValueError(
                "data.feature_subset must be a non-empty string when provided"
            )
        if not self.targets:
            raise ValueError("data.targets must contain at least one target")

    @property
    def resolved_feature_set(self) -> str:
        """Feature set actually used: explicit ``feature_subset`` wins over ``feature_set``.

        ``feature_subset`` names are validated against ``features.json`` at
        ingestion time (fail loud, fail late — ``IngestionAgent.features``).
        """
        return self.feature_subset if self.feature_subset is not None else self.feature_set
```

```python
# nmr/runner.py:71 — replace
        feature_cols = agent.features(self._config.data.feature_set)
# with
        feature_cols = agent.features(self._config.data.resolved_feature_set)
```

```python
# nmr/research.py:207 — replace
    feature_cols = agent.features(config.data.feature_set)
# with
    feature_cols = agent.features(config.data.resolved_feature_set)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python -m pytest tests/test_config.py tests/test_features.py tests/test_runner.py -q`
Expected: PASS (additive; existing tests untouched and green).

- [ ] **Step 5: Record** — deliverable on disk.

---

### Task 4: `nmr/meta.py` — `paired_era_comparison`

**Files:**
- Create: `nmr/meta.py` (only `PairedResult` + `paired_era_comparison` for now)
- Test: `tests/test_meta.py` (new file)

**Interfaces:**
- Consumes: `nmr.evaluation.MIN_OVERLAP_ERAS`, `NonVacuityError`; `nmr.inference.block_bootstrap_ci`, `resolve_block_len`, `Horizon`.
- Produces: `PairedResult(mean_diff, ci_low, ci_high, n_eras, device_mismatch, alpha, n_boot, block_len)`; `paired_era_comparison(oof_a, oof_b, *, metric_fn, era_col="era", horizon="20D", n_boot=1000, seed, alpha=0.05, min_overlap_eras=MIN_OVERLAP_ERAS, block_len=None, device_a=None, device_b=None) -> PairedResult`. Consumed by S3 skill and `promotion_verdict` (Task 5, indirectly).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_meta.py
from __future__ import annotations

import polars as pl
import pytest

from nmr.evaluation import MIN_OVERLAP_ERAS, NonVacuityError
from nmr.meta import paired_era_comparison


def _frame(n_eras: int = 24) -> pl.DataFrame:
    rows = []
    for era in range(1, n_eras + 1):
        for idx in range(10):
            rows.append({"era": str(era), "id": f"{era}_{idx}", "prediction": idx * 0.1})
    return pl.DataFrame(rows)


def _era_index_metric(frame: pl.DataFrame) -> dict[str, float]:
    """Deterministic per-era metric: the era number itself."""
    return {
        str(era): float(era)
        for era in frame.get_column("era").unique().sort().to_list()
    }


def test_paired_comparison_estimates_mean_difference_with_ci() -> None:
    a = _frame()
    b = _frame()
    result = paired_era_comparison(
        a, b, metric_fn=_era_index_metric, seed=7, n_boot=50,
    )
    assert result.mean_diff == pytest.approx(0.0, abs=1e-9)
    assert result.n_eras == 24
    assert result.device_mismatch is False
    assert result.ci_low <= result.mean_diff <= result.ci_high
    assert result.alpha == 0.05 and result.n_boot == 50


def test_paired_comparison_sign_using_prediction_means() -> None:
    def mean_pred(frame: pl.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        for era in frame.get_column("era").unique().to_list():
            out[str(era)] = float(
                frame.filter(pl.col("era") == era).get_column("prediction").mean()
            )
        return out

    a = _frame()  # prediction = idx * 0.1 -> era mean 0.45
    b = a.with_columns((pl.col("prediction") + 1.0).alias("prediction"))  # era mean 1.45
    result = paired_era_comparison(a, b, metric_fn=mean_pred, seed=7, n_boot=50)
    assert result.mean_diff == pytest.approx(-1.0, abs=1e-9)  # a - b == -1.0


def test_paired_comparison_bootstrap_deterministic_under_seed() -> None:
    a, b = _frame(), _frame()
    r1 = paired_era_comparison(a, b, metric_fn=_era_index_metric, seed=11, n_boot=200)
    r2 = paired_era_comparison(a, b, metric_fn=_era_index_metric, seed=11, n_boot=200)
    assert r1 == r2  # same seed -> identical CI (cross-process determinism)


def test_paired_comparison_intersects_eras_and_raises_below_overlap_floor() -> None:
    a = _frame(n_eras=24)
    b = _frame(n_eras=10)  # overlap = 10 < MIN_OVERLAP_ERAS
    with pytest.raises(NonVacuityError):
        paired_era_comparison(a, b, metric_fn=_era_index_metric, seed=7)


def test_paired_comparison_device_mismatch_flag() -> None:
    result = paired_era_comparison(
        _frame(), _frame(), metric_fn=_era_index_metric, seed=7,
        device_a="gpu", device_b="cpu",
    )
    assert result.device_mismatch is True
    same = paired_era_comparison(
        _frame(), _frame(), metric_fn=_era_index_metric, seed=7,
        device_a="cpu", device_b="cpu",
    )
    assert same.device_mismatch is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_meta.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'nmr.meta'`.

- [ ] **Step 3: Write minimal implementation**

```python
# nmr/meta.py
"""Cross-run meta-analysis: paired era comparison and promotion decisions.

Decision layer on top of ``nmr.inference`` and ``nmr.evaluation``. All
statistics reuse the repo's seeded block-bootstrap machinery; nothing here
mutates the registry.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from nmr.evaluation import MIN_OVERLAP_ERAS, NonVacuityError
from nmr.inference import Horizon, block_bootstrap_ci, resolve_block_len

__all__ = ["PairedResult", "paired_era_comparison"]


@dataclass(frozen=True)
class PairedResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    n_eras: int
    device_mismatch: bool
    alpha: float
    n_boot: int
    block_len: int


def paired_era_comparison(
    oof_a: pl.DataFrame,
    oof_b: pl.DataFrame,
    *,
    metric_fn: Callable[[pl.DataFrame], dict[str, float]],
    era_col: str = "era",
    horizon: Horizon = "20D",
    n_boot: int = 1000,
    seed: int,
    alpha: float = 0.05,
    min_overlap_eras: int = MIN_OVERLAP_ERAS,
    block_len: int | None = None,
    device_a: str | None = None,
    device_b: str | None = None,
) -> PairedResult:
    """Compare two runs on per-era metric differences via block bootstrap.

    ``metric_fn`` maps an OOF frame to ``{era: metric}`` (e.g. a closure over
    ``EvaluationEngine().per_era_corr`` with explicit pred/target/era columns).
    Positive ``mean_diff`` means A is better. Eras are intersected on the
    numeric era index; fewer than ``min_overlap_eras`` overlapping eras raises
    :class:`NonVacuityError`. A device mismatch is reported (GPU vs CPU OOF
    values are not comparable — see AGENTS.md operational hazards), never
    silently corrected.
    """
    per_era_a = metric_fn(oof_a)
    per_era_b = metric_fn(oof_b)
    overlap = sorted(set(per_era_a) & set(per_era_b), key=int)
    if len(overlap) < min_overlap_eras:
        raise NonVacuityError(
            f"paired overlap {len(overlap)} eras < MIN_OVERLAP_ERAS "
            f"{min_overlap_eras}"
        )
    diffs = np.asarray(
        [float(per_era_a[era]) - float(per_era_b[era]) for era in overlap],
        dtype=float,
    )
    blen = (
        block_len
        if block_len is not None
        else resolve_block_len(int(diffs.size), horizon)
    )
    ci = block_bootstrap_ci(
        diffs,
        lambda arr: float(np.mean(arr)),
        block_len=blen,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )
    return PairedResult(
        mean_diff=float(np.mean(diffs)),
        ci_low=ci.lo,
        ci_high=ci.hi,
        n_eras=int(diffs.size),
        device_mismatch=(
            device_a is not None
            and device_b is not None
            and device_a != device_b
        ),
        alpha=float(alpha),
        n_boot=int(n_boot),
        block_len=int(blen),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python -m pytest tests/test_meta.py -q`
Expected: PASS.

- [ ] **Step 5: Record** — deliverable on disk.

---

### Task 5: `nmr/meta.py` — `promotion_verdict`

**Files:**
- Modify: `nmr/meta.py` (append)
- Test: `tests/test_meta.py` (append)

**Interfaces:**
- Consumes: registry entry dicts shaped like `RunRegistry.list()` output (with a `"scorecard"` block containing `<metric>`, `<metric>_ci_low`, `<metric>_ci_high`).
- Produces: `promotion_verdict(candidate: dict, champion: dict | None, *, metric: str = "corr_sharpe_ac", alpha: float = 0.05) -> str` returning `"promote" | "hold" | "caution"`. Consumed by S3/S2 skills.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_meta.py
from nmr.meta import promotion_verdict
from nmr.registry import RunRegistry


def _entry(run_id: str, metric: str = "corr_sharpe_ac", *, value: float | None = None,
           lo: float | None = None, hi: float | None = None) -> dict:
    scorecard: dict = {}
    if value is not None:
        scorecard[metric] = value
    if lo is not None:
        scorecard[f"{metric}_ci_low"] = lo
    if hi is not None:
        scorecard[f"{metric}_ci_high"] = hi
    return {"run_id": run_id, "scorecard": scorecard}


def test_verdict_promotes_when_candidate_ci_clears_champion() -> None:
    champion = _entry("c" * 64, value=0.10, lo=0.05, hi=0.15)
    candidate = _entry("d" * 64, value=0.25, lo=0.20, hi=0.30)
    assert promotion_verdict(candidate, champion) == "promote"


def test_verdict_holds_when_candidate_ci_below_champion() -> None:
    champion = _entry("c" * 64, value=0.25, lo=0.20, hi=0.30)
    candidate = _entry("d" * 64, value=0.10, lo=0.05, hi=0.15)
    assert promotion_verdict(candidate, champion) == "hold"


def test_verdict_cautions_on_ci_overlap() -> None:
    champion = _entry("c" * 64, value=0.18, lo=0.10, hi=0.26)
    candidate = _entry("d" * 64, value=0.20, lo=0.14, hi=0.27)
    assert promotion_verdict(candidate, champion) == "caution"


def test_verdict_cautions_when_ci_unavailable() -> None:
    champion = _entry("c" * 64, value=0.10)
    candidate = _entry("d" * 64, value=0.25)
    assert promotion_verdict(candidate, champion) == "caution"


def test_verdict_promotes_without_champion() -> None:
    candidate = _entry("d" * 64, value=0.25, lo=0.20, hi=0.30)
    assert promotion_verdict(candidate, None) == "promote"


def test_verdict_lower_is_better_for_max_drawdown() -> None:
    champion = _entry("c" * 64, metric="max_drawdown", value=0.20, lo=0.18, hi=0.22)
    candidate = _entry("d" * 64, metric="max_drawdown", value=0.10, lo=0.08, hi=0.12)
    assert promotion_verdict(candidate, champion, metric="max_drawdown") == "promote"


def test_verdict_directions_match_registry_semantics() -> None:
    from nmr.meta import _VERDICT_DIRECTIONS

    assert set(_VERDICT_DIRECTIONS) <= set(
        RunRegistry._SCORECARD_METRIC_DIRECTION
    )
    for metric, higher_is_better in _VERDICT_DIRECTIONS.items():
        assert RunRegistry._SCORECARD_METRIC_DIRECTION[metric] == higher_is_better


def test_verdict_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="metric"):
        promotion_verdict(_entry("d" * 64), None, metric="bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_meta.py -q`
Expected: FAIL with `ImportError: cannot import name 'promotion_verdict'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to nmr/meta.py
from typing import Literal

# Directions aligned with RunRegistry._SCORECARD_METRIC_DIRECTION (parity-tested
# in test_meta.py). True = higher-is-better.
_VERDICT_DIRECTIONS: dict[str, bool] = {
    "corr": True,
    "mmc": True,
    "fnc": True,
    "corr_sharpe_ac": True,
    "mean_payout": True,
    "deflated_sharpe": True,
    "std_corr": False,
    "max_drawdown": False,
}


def promotion_verdict(
    candidate: dict,
    champion: dict | None,
    *,
    metric: str = "corr_sharpe_ac",
    alpha: float = 0.05,
) -> Literal["promote", "hold", "caution"]:
    """Significance-aware promotion decision on registry entries.

    Compares CI-bearing scorecard cells: candidate ``ci_low > champion
    ci_high`` (higher-is-better) -> ``"promote"``; the mirror -> ``"hold"``;
    any overlap, missing CI, or missing champion scorecard -> ``"caution"``.
    This is an advisory verdict only — it never writes the registry.
    """
    if metric not in _VERDICT_DIRECTIONS:
        raise ValueError(
            f"metric={metric!r} not in {sorted(_VERDICT_DIRECTIONS)}"
        )
    higher_is_better = _VERDICT_DIRECTIONS[metric]

    def _cell(entry: dict) -> tuple[float | None, float | None, float | None]:
        scorecard = entry.get("scorecard") or {}
        value = scorecard.get(metric)
        if value is None:
            return None, None, None
        return (
            float(value),
            scorecard.get(f"{metric}_ci_low"),
            scorecard.get(f"{metric}_ci_high"),
        )

    cand_value, cand_lo, cand_hi = _cell(candidate)
    if cand_value is None:
        raise ValueError(
            f"candidate run lacks scorecard metric {metric!r}; "
            "cannot issue a significance-aware verdict"
        )
    if champion is None:
        return "promote"
    champ_value, champ_lo, champ_hi = _cell(champion)
    if champ_value is None:
        return "promote"
    if None in (cand_lo, cand_hi, champ_lo, champ_hi):
        return "caution"

    if higher_is_better:
        if cand_lo > champ_hi:
            return "promote"
        if cand_hi < champ_lo:
            return "hold"
    else:
        if cand_hi < champ_lo:
            return "promote"
        if cand_lo > champ_hi:
            return "hold"
    return "caution"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python -m pytest tests/test_meta.py -q`
Expected: PASS.

- [ ] **Step 5: Record** — deliverable on disk.

---

### Task 6: `nmr/meta.py` — `fleet_summary`

**Files:**
- Modify: `nmr/meta.py` (append)
- Test: `tests/test_meta.py` (append)

**Interfaces:**
- Consumes: registry entry dicts (`RunRegistry.list()` output) with `manifest.config` and scorecard block.
- Produces: `fleet_summary(runs: Sequence[dict], *, metric: str = "corr_sharpe_ac", n_trials: int, dsr_confidence: float = 0.95) -> pl.DataFrame` with per-run cells + DSR pass/fail + robustness/device/grouping flags. Consumed by S3 skill.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_meta.py
import polars as pl

from nmr.meta import fleet_summary


def _full_entry(run_id: str, sharpe_ac: float) -> dict:
    return {
        "run_id": run_id,
        "manifest": {
            "oof_device": "cpu",
            "config": {
                "data": {"feature_set": "small", "feature_subset": None},
                "model": {"preset": "fast"},
                "risk": {"neutralization_proportion": 1.0},
            },
        },
        "scorecard": {
            "corr_sharpe_ac": sharpe_ac,
            "corr_sharpe_ac_ci_low": sharpe_ac - 0.05,
            "corr_sharpe_ac_ci_high": sharpe_ac + 0.05,
            "corr_sharpe_ac_n_eras": 30,
            "deflated_sharpe": 0.98,
            "max_feature_exposure": 0.3,
            "bmc": 0.02,
            "horizon_model_sharpe_20": 0.5,
            "perturb_ceiling_stability": 0.9,
            "regime_count": 3,
        },
    }


def test_fleet_summary_columns_and_flags() -> None:
    runs = [_full_entry("a" * 64, 0.12), _full_entry("b" * 64, 0.05)]
    frame = fleet_summary(runs, n_trials=2)
    assert frame.height == 2
    assert set(frame.columns) >= {
        "run_id", "metric", "metric_ci_low", "metric_ci_high", "metric_n_eras",
        "deflated_sharpe", "dsr_pass", "max_feature_exposure", "oof_device",
        "preset", "feature_set", "feature_subset", "neutralization_proportion",
        "has_bmc", "has_horizon", "has_perturb", "has_regime",
        "policy_n_trials", "policy_dsr_confidence",
    }
    first = frame.filter(pl.col("run_id") == "a" * 64).row(0, named=True)
    assert first["dsr_pass"] is True
    assert first["oof_device"] == "cpu"
    assert first["preset"] == "fast"
    assert first["feature_set"] == "small"
    assert first["has_bmc"] is True and first["has_horizon"] is True
    assert first["has_perturb"] is True and first["has_regime"] is True
    assert first["policy_n_trials"] == 2
    # sorted by metric desc, run_id tiebreak
    assert frame.get_column("run_id").to_list() == ["a" * 64, "b" * 64]


def test_fleet_summary_flags_legacy_runs_without_scorecard() -> None:
    legacy = {
        "run_id": "c" * 64,
        "manifest": {"oof_device": "cpu", "config": {
            "data": {"feature_set": "all", "feature_subset": None},
            "model": {"preset": "deep"},
            "risk": {"neutralization_proportion": 0.5},
        }},
        "scorecard": None,
    }
    frame = fleet_summary([legacy], n_trials=1)
    row = frame.row(0, named=True)
    assert row["metric"] is None
    assert row["dsr_pass"] is False
    assert row["has_bmc"] is False
    assert row["preset"] == "deep" and row["neutralization_proportion"] == 0.5


def test_fleet_summary_validates_policy_arguments() -> None:
    with pytest.raises(ValueError, match="n_trials"):
        fleet_summary([], n_trials=0)
    with pytest.raises(ValueError, match="dsr_confidence"):
        fleet_summary([], n_trials=1, dsr_confidence=1.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_meta.py -q`
Expected: FAIL with `ImportError: cannot import name 'fleet_summary'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to nmr/meta.py
def fleet_summary(
    runs: Sequence[dict],
    *,
    metric: str = "corr_sharpe_ac",
    n_trials: int,
    dsr_confidence: float = 0.95,
) -> pl.DataFrame:
    """Flatten registry entries into a per-run fleet table.

    Per-run cells: the requested scorecard metric (value + CI + n_eras), the
    stored ``deflated_sharpe`` with a pass/fail flag against
    ``dsr_confidence``, max feature exposure, ``oof_device``, and grouping
    attributes from the manifest config (preset, feature_set, feature_subset,
    neutralization_proportion) plus robustness presence flags (bmc, horizon,
    perturbation, regime). ``n_trials`` and ``dsr_confidence`` are recorded as
    policy context columns; the stored DSR itself was computed with
    ``n_trials=1`` at scorecard time — campaign-aware DSR requires era-level
    recompute via :func:`paired_era_comparison` tooling and is out of scope
    here. Runs without a scorecard are flagged (legacy), never silently
    dropped. Deterministic: sorted by metric desc, run_id tiebreak.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if not (0.0 < dsr_confidence < 1.0):
        raise ValueError("dsr_confidence must satisfy 0 < dsr_confidence < 1")

    rows: list[dict] = []
    for entry in runs:
        run_id = entry["run_id"]
        manifest = entry.get("manifest") or {}
        config = manifest.get("config") or {}
        data_cfg = config.get("data") or {}
        model_cfg = config.get("model") or {}
        risk_cfg = config.get("risk") or {}
        scorecard = entry.get("scorecard") or {}
        metric_value = scorecard.get(metric)
        rows.append(
            {
                "run_id": run_id,
                "metric": float(metric_value) if metric_value is not None else None,
                "metric_ci_low": scorecard.get(f"{metric}_ci_low"),
                "metric_ci_high": scorecard.get(f"{metric}_ci_high"),
                "metric_n_eras": scorecard.get(f"{metric}_n_eras"),
                "deflated_sharpe": scorecard.get("deflated_sharpe"),
                "dsr_pass": bool(
                    scorecard.get("deflated_sharpe") is not None
                    and float(scorecard["deflated_sharpe"]) >= dsr_confidence
                ),
                "max_feature_exposure": scorecard.get("max_feature_exposure"),
                "oof_device": manifest.get("oof_device"),
                "preset": model_cfg.get("preset"),
                "feature_set": data_cfg.get("feature_set"),
                "feature_subset": data_cfg.get("feature_subset"),
                "neutralization_proportion": risk_cfg.get("neutralization_proportion"),
                "has_bmc": scorecard.get("bmc") is not None,
                "has_horizon": scorecard.get("horizon_model_sharpe_20") is not None,
                "has_perturb": scorecard.get("perturb_ceiling_stability") is not None,
                "has_regime": scorecard.get("regime_count") is not None,
                "policy_n_trials": n_trials,
                "policy_dsr_confidence": dsr_confidence,
            }
        )
    frame = pl.DataFrame(rows)
    if frame.height > 0:
        frame = frame.sort(
            ["metric", "run_id"], descending=[True, False], nulls_last=True
        )
    return frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python -m pytest tests/test_meta.py -q`
Expected: PASS.

- [ ] **Step 5: Record** — deliverable on disk.

---

### Task 7: `nmr/campaign.py` — campaign identity + trial-lineage log

**Files:**
- Create: `nmr/campaign.py`
- Test: `tests/test_campaign.py` (new file)

**Interfaces:**
- Consumes: `nmr._atomicio.atomic_write_text`.
- Produces:
  - `campaign_id(name: str, config_paths: Sequence[str | Path]) -> str` (64-hex, deterministic, path-independent: hashes name + sorted per-file content SHA256).
  - `@dataclass(frozen=True) CampaignConfig(path: str, sha256: str)`
  - `@dataclass(frozen=True) CampaignRun(config_path: str, run_id: str | None, status: str, error: str | None = None)` with `status in {"recorded", "skipped", "error"}`.
  - `@dataclass(frozen=True) CampaignLog(campaign_id, name, configs: tuple[CampaignConfig, ...], runs: tuple[CampaignRun, ...])`
  - `build_campaign_log(name: str, config_paths: Sequence[str | Path], runs: Sequence[CampaignRun]) -> CampaignLog`
  - `write_campaign_log(log: CampaignLog, campaigns_dir: str | Path) -> Path` (atomic; returns written path).
  Consumed by Task 8 CLI and the S3 skill.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_campaign.py
from __future__ import annotations

import json

import polars as pl
import pytest

from nmr.campaign import (
    CampaignConfig,
    CampaignLog,
    CampaignRun,
    build_campaign_log,
    campaign_id,
    write_campaign_log,
)


def _write_config(tmp_path, name: str, content: str) -> None:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_campaign_id_is_deterministic_and_path_independent(tmp_path) -> None:
    a = _write_config(tmp_path, "a.yaml", "run:\n  name: x\n")
    b = _write_config(tmp_path, "b.yaml", "run:\n  name: y\n")
    c = _write_config(tmp_path, "c.yaml", "run:\n  name: x\n")  # same content as a

    assert campaign_id("camp", [a, b]) == campaign_id("camp", [a, b])
    assert campaign_id("camp", [a, b]) != campaign_id("camp", [b, a])  # order matters
    assert campaign_id("camp", [a, b]) != campaign_id("other", [a, b])
    # identical content, different file name -> identical id (path-independent)
    assert campaign_id("camp", [a, b]) == campaign_id("camp", [c, b])
    assert len(campaign_id("camp", [a, b])) == 64
    assert campaign_id("camp", [a, b]).isalnum()


def test_build_campaign_log_validates_inputs(tmp_path) -> None:
    cfg = _write_config(tmp_path, "a.yaml", "run:\n  name: x\n")
    with pytest.raises(ValueError, match="name"):
        build_campaign_log("", [cfg], runs=())
    with pytest.raises(ValueError, match="config_paths"):
        build_campaign_log("camp", [], runs=())
    with pytest.raises(FileNotFoundError):
        build_campaign_log("camp", [tmp_path / "missing.yaml"], runs=())
    with pytest.raises(ValueError, match="status"):
        build_campaign_log(
            "camp", [cfg], runs=[CampaignRun(str(cfg), run_id=None, status="bogus")]
        )


def test_write_campaign_log_atomic_and_schema(tmp_path) -> None:
    cfg = _write_config(tmp_path, "a.yaml", "run:\n  name: x\n")
    log = build_campaign_log(
        "camp",
        [cfg],
        runs=[
            CampaignRun(str(cfg), run_id="a" * 64, status="recorded"),
            CampaignRun(str(cfg), run_id=None, status="error", error="boom"),
        ],
    )
    out_dir = tmp_path / "campaigns"
    written = write_campaign_log(log, out_dir)
    assert written == out_dir / f"{log.campaign_id}.json"
    assert written.exists()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["campaign_id"] == log.campaign_id
    assert payload["name"] == "camp"
    assert payload["configs"][0]["path"] == str(cfg)
    assert len(payload["configs"][0]["sha256"]) == 64
    assert payload["runs"][0]["status"] == "recorded"
    assert payload["runs"][1]["error"] == "boom"
    assert set(payload) == {"campaign_id", "name", "configs", "runs"}


def test_write_campaign_log_is_idempotent(tmp_path) -> None:
    cfg = _write_config(tmp_path, "a.yaml", "run:\n  name: x\n")
    log = build_campaign_log("camp", [cfg], runs=())
    p1 = write_campaign_log(log, tmp_path / "out")
    p2 = write_campaign_log(log, tmp_path / "out")
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_campaign.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'nmr.campaign'`.

- [ ] **Step 3: Write minimal implementation**

```python
# nmr/campaign.py
"""Campaign orchestration: deterministic trial-lineage logs for research fleets.

A campaign is a named batch of experiment configs whose runs share a
hypothesis. The registry stores per-run state but not per-hypothesis lineage;
this module provides that attribution schema. All writes are atomic
(temp + fsync + os.replace) per AGENTS.md §9. No wall-clock fields are stored
in the log (canonical-determinism friendly; file mtime carries chronology).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from nmr._atomicio import atomic_write_text

__all__ = [
    "CampaignConfig",
    "CampaignRun",
    "CampaignLog",
    "campaign_id",
    "build_campaign_log",
    "write_campaign_log",
]

_VALID_STATUSES = ("recorded", "skipped", "error")


@dataclass(frozen=True)
class CampaignConfig:
    path: str
    sha256: str


@dataclass(frozen=True)
class CampaignRun:
    config_path: str
    run_id: str | None
    status: str
    error: str | None = None


@dataclass(frozen=True)
class CampaignLog:
    campaign_id: str
    name: str
    configs: tuple[CampaignConfig, ...]
    runs: tuple[CampaignRun, ...]

    def to_payload(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "configs": [
                {"path": c.path, "sha256": c.sha256} for c in self.configs
            ],
            "runs": [
                {
                    "config_path": r.config_path,
                    "run_id": r.run_id,
                    "status": r.status,
                    "error": r.error,
                }
                for r in self.runs
            ],
        }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def campaign_id(name: str, config_paths: Sequence[str | Path]) -> str:
    """Deterministic, path-independent campaign id (64-char hex).

    Hashes the name plus the sorted per-file content SHA256 digests, so moving
    or renaming config files does not change the campaign identity.
    """
    if not name:
        raise ValueError("campaign name must be non-empty")
    if not config_paths:
        raise ValueError("campaign requires at least one config path")
    digests = [
        _file_sha256(Path(path)) for path in config_paths
    ]
    payload = json.dumps(
        {"name": name, "configs": sorted(digests)}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_campaign_log(
    name: str,
    config_paths: Sequence[str | Path],
    runs: Sequence[CampaignRun],
) -> CampaignLog:
    """Validate and assemble a :class:`CampaignLog`."""
    if not name:
        raise ValueError("campaign name must be non-empty")
    if not config_paths:
        raise ValueError("config_paths must contain at least one path")
    configs: list[CampaignConfig] = []
    for path in config_paths:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"config file not found: {resolved}")
        configs.append(
            CampaignConfig(path=str(path), sha256=_file_sha256(resolved))
        )
    for run in runs:
        if run.status not in _VALID_STATUSES:
            raise ValueError(
                f"run status {run.status!r} not in {_VALID_STATUSES}"
            )
        if run.status != "error" and run.run_id is None:
            raise ValueError(
                "non-error campaign runs must carry a run_id"
            )
    return CampaignLog(
        campaign_id=campaign_id(name, config_paths),
        name=name,
        configs=tuple(configs),
        runs=tuple(runs),
    )


def write_campaign_log(
    log: CampaignLog, campaigns_dir: str | Path
) -> Path:
    """Write ``log`` atomically to ``campaigns_dir/{campaign_id}.json``."""
    out_dir = Path(campaigns_dir)
    target = out_dir / f"{log.campaign_id}.json"
    atomic_write_text(
        target,
        json.dumps(log.to_payload(), indent=2, sort_keys=True),
    )
    return target
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python -m pytest tests/test_campaign.py -q`
Expected: PASS.

- [ ] **Step 5: Record** — deliverable on disk.

---

### Task 8: `run_campaign.py` CLI + public `compute_run_id`

**Files:**
- Modify: `nmr/runner.py` (add `compute_run_id` static method near `_compute_run_id`)
- Create: `run_campaign.py` (root)
- Test: `tests/test_runner.py` (append), `tests/test_campaign.py` (append, CLI wiring), `tests/test_scripts.py` (append import smoke)

**Interfaces:**
- Consumes: `nmr.campaign` (Task 7), `ExperimentRunner.compute_run_id`, `RunRegistry`.
- Produces: `run_campaign.py` CLI: `--config PATH` (repeatable), `--name NAME`, `--registry DIR` (default `artifacts/registry`), `--campaigns-dir DIR` (default `artifacts/campaigns`), `--deploy`, `--dry-run`. Exit 0 on success, 1 if any trial errored.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_runner.py
def test_compute_run_id_public_accessor_matches_private(tmp_path) -> None:
    cfg = _config(tmp_path)
    assert ExperimentRunner.compute_run_id(cfg) == ExperimentRunner(cfg)._run_id
    assert (
        ExperimentRunner.compute_run_id(cfg)
        == ExperimentRunner.compute_run_id(cfg)
    )
```

```python
# append to tests/test_campaign.py
import subprocess
import sys

import pytest

import run_campaign
from nmr.runner import ExperimentRunner, RunResult
from nmr.evaluation import MetricSummary


def _stub_run(tmp_path, monkeypatch) -> None:
    import polars as pl

    def fake_run(self, *, deploy: bool = False) -> RunResult:
        return RunResult(
            run_id="a" * 64,
            oof=pl.DataFrame({"id": ["x"], "era": ["1"], "prediction": [0.5]}),
            metrics=MetricSummary(mean=0.1, std=0.2, sharpe=0.5, max_drawdown=0.05),
            artifact=None,
            manifest={"run_id": "a" * 64, "oof_device": "cpu"},
        )

    monkeypatch.setattr(ExperimentRunner, "run", fake_run)
    monkeypatch.setattr(
        ExperimentRunner,
        "compute_run_id",
        staticmethod(lambda config: "a" * 64),
    )


def test_run_campaign_main_records_and_writes_log(tmp_path, monkeypatch) -> None:
    cfg = _write_config(tmp_path, "a.yaml", "run:\n  name: x\n")
    _stub_run(tmp_path, monkeypatch)
    registry_dir = tmp_path / "registry"
    campaigns_dir = tmp_path / "campaigns"
    rc = run_campaign.main([
        "--config", str(cfg), "--name", "camp",
        "--registry", str(registry_dir), "--campaigns-dir", str(campaigns_dir),
    ])
    assert rc == 0
    assert (registry_dir / ("a" * 64) / "run.json").exists()
    logs = list(campaigns_dir.glob("*.json"))
    assert len(logs) == 1
    payload = json.loads(logs[0].read_text(encoding="utf-8"))
    assert payload["runs"][0]["status"] == "recorded"
    assert payload["runs"][0]["run_id"] == "a" * 64


def test_run_campaign_dry_run_writes_nothing(tmp_path, monkeypatch) -> None:
    cfg = _write_config(tmp_path, "a.yaml", "run:\n  name: x\n")
    _stub_run(tmp_path, monkeypatch)
    rc = run_campaign.main([
        "--config", str(cfg), "--name", "camp",
        "--registry", str(tmp_path / "registry"),
        "--campaigns-dir", str(tmp_path / "campaigns"),
        "--dry-run",
    ])
    assert rc == 0
    assert not (tmp_path / "registry").exists()
    assert not (tmp_path / "campaigns").exists()


def test_run_campaign_error_records_and_returns_1(tmp_path, monkeypatch) -> None:
    cfg = _write_config(tmp_path, "a.yaml", "run:\n  name: x\n")
    _stub_run(tmp_path, monkeypatch)

    def boom(self, *, deploy: bool = False):
        raise RuntimeError("training failed")

    monkeypatch.setattr(ExperimentRunner, "run", boom)
    rc = run_campaign.main([
        "--config", str(cfg), "--name", "camp",
        "--registry", str(tmp_path / "registry"),
        "--campaigns-dir", str(tmp_path / "campaigns"),
    ])
    assert rc == 1
    logs = list((tmp_path / "campaigns").glob("*.json"))
    assert len(logs) == 1
    payload = json.loads(logs[0].read_text(encoding="utf-8"))
    assert payload["runs"][0]["status"] == "error"
    assert "training failed" in payload["runs"][0]["error"]


def test_run_campaign_rejects_no_configs(tmp_path, monkeypatch, capsys) -> None:
    _stub_run(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        run_campaign.main(["--name", "camp"])
```

```python
# append to tests/test_scripts.py
def test_run_campaign_imports_as_control_plane() -> None:
    import run_campaign  # noqa: F401  (import-time smoke)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_runner.py tests/test_campaign.py tests/test_scripts.py -q`
Expected: FAIL — `compute_run_id` missing; `run_campaign` module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# nmr/runner.py — add after _compute_run_id's class body (inside ExperimentRunner)
    @staticmethod
    def compute_run_id(config: ExperimentConfig) -> str:
        """Public accessor for the canonical run id (used by campaign tooling)."""
        return ExperimentRunner._compute_run_id(config)
```

```python
# run_campaign.py (repo root) — thin control plane
"""Run a named batch of experiment configs and record trial lineage.

Thin control plane: argument parsing, wiring, and printing only. All logic
lives in ``nmr.campaign`` / ``nmr.runner`` / ``nmr.registry``.

Usage:
    python run_campaign.py --config configs/a.yaml --config configs/b.yaml \
        --name my-campaign [--registry artifacts/registry] \
        [--campaigns-dir artifacts/campaigns] [--deploy] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from nmr import ExperimentRunner, RunRegistry, load_config
from nmr.campaign import CampaignRun, build_campaign_log, write_campaign_log

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_campaign")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True,
                        help="path to an experiment config YAML (repeatable)")
    parser.add_argument("--name", required=True, help="campaign name")
    parser.add_argument("--registry", default="artifacts/registry",
                        help="registry root directory")
    parser.add_argument("--campaigns-dir", default="artifacts/campaigns",
                        help="campaign log output directory")
    parser.add_argument("--deploy", action="store_true",
                        help="pass deploy=True to ExperimentRunner.run")
    parser.add_argument("--dry-run", action="store_true",
                        help="print run ids without training or writing")
    args = parser.parse_args(argv)

    config_paths = [Path(p) for p in args.config]
    registry = RunRegistry(args.registry)
    existing = {entry["run_id"] for entry in registry.list()}

    runs: list[CampaignRun] = []
    failed = 0
    for path in config_paths:
        try:
            cfg = load_config(path)
            run_id = ExperimentRunner.compute_run_id(cfg)
        except Exception as exc:  # validation/config failures are campaign-level
            logger.error("[campaign] config %s invalid: %s", path, exc)
            runs.append(CampaignRun(str(path), run_id=None, status="error", error=str(exc)))
            failed += 1
            continue

        if args.dry_run:
            logger.info("[campaign] dry-run: %s -> %s", path, run_id)
            runs.append(CampaignRun(str(path), run_id=run_id, status="skipped"))
            continue

        if run_id in existing:
            logger.info("[campaign] %s already recorded; skipping", run_id)
            runs.append(CampaignRun(str(path), run_id=run_id, status="skipped"))
            continue

        try:
            result = ExperimentRunner(cfg).run(deploy=args.deploy)
            registry.record(result)
            runs.append(CampaignRun(str(path), run_id=result.run_id, status="recorded"))
            logger.info("[campaign] recorded %s -> %s", path, result.run_id)
        except Exception as exc:
            logger.exception("[campaign] run failed for %s", path)
            runs.append(CampaignRun(str(path), run_id=None, status="error", error=str(exc)))
            failed += 1

    if args.dry_run:
        for run in runs:
            print(f"dry-run\t{run.config_path}\t{run.run_id}")
        return 0

    log = build_campaign_log(args.name, config_paths, runs)
    log_path = write_campaign_log(log, args.campaigns_dir)
    logger.info("[campaign] log written to %s", log_path)
    for run in runs:
        print(f"{run.status}\t{run.config_path}\t{run.run_id}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python -m pytest tests/test_runner.py tests/test_campaign.py tests/test_scripts.py -q`
Expected: PASS.

- [ ] **Step 5: Record** — deliverable on disk.

---

### Task 9: Public exports + SSOT docs (same change set)

**Files:**
- Modify: `nmr/__init__.py` (imports + `__all__`)
- Modify: `configs/example.yaml`
- Modify: `ARCHITECTURE.md`, `AGENTS.md`, `README.md`

- [ ] **Step 1: Write the failing test (exports contract)**

```python
# tests/test_contribution.py — append a public-API surface test
def test_public_api_includes_harness_symbols():
    import nmr

    for name in (
        "resolve_feature_sets",
        "feature_stability_screen",
        "select_stable_features",
        "PairedResult",
        "paired_era_comparison",
        "promotion_verdict",
        "fleet_summary",
        "CampaignLog",
        "campaign_id",
        "build_campaign_log",
        "write_campaign_log",
    ):
        assert name in nmr.__all__
        assert getattr(nmr, name) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_contribution.py -q`
Expected: FAIL (symbols missing from `__all__`/imports).

- [ ] **Step 3: Implement exports**

```python
# nmr/__init__.py — add imports (alphabetical placement near existing peers)
from .campaign import (
    CampaignLog,
    build_campaign_log,
    campaign_id,
    write_campaign_log,
)
from .features import (
    feature_stability_screen,
    resolve_feature_sets,
    select_stable_features,
)
from .meta import (
    PairedResult,
    fleet_summary,
    paired_era_comparison,
    promotion_verdict,
)
# and add the same names to __all__ (alphabetical, before the closing bracket)
```

- [ ] **Step 4: Docs (same change set, SSOT — one fact, one home)**

- `configs/example.yaml` — under `data:`, add:
  ```yaml
  feature_subset: null    # optional: any name in features.json (e.g. 'sunshine');
                          # overrides feature_set for this experiment
  ```
- `ARCHITECTURE.md` — add three component sections under §2, lettered after the last (currently M/N/O; use P/Q/R):
  - **§P `nmr/features.py`** — subset resolution (`resolve_feature_sets`: all named sets from `features.json`, deterministic order), stability screen (definition: per-era Pearson CORR(feature, target), degenerate era → 0.0; aggregates mean/std/decay slope/cross-regime variance; `stable` predicate and its `DEFAULT_*` thresholds), `select_stable_features`; `DataConfig.feature_subset` semantics (validated at ingestion, `resolved_feature_set` precedence).
  - **§Q `nmr/meta.py`** — `paired_era_comparison` (era intersection, `MIN_OVERLAP_ERAS` floor → `NonVacuityError`, block bootstrap on era-level diffs, `device_mismatch` flag), `promotion_verdict` ("promote"/"hold"/"caution", CI-separability, direction table parity with registry), `fleet_summary` (columns, DSR policy note: stored DSR used `n_trials=1`).
  - **§R `nmr/campaign.py` + `run_campaign.py`** — campaign id (name + sorted content hashes, path-independent), `artifacts/campaigns/{id}.json` schema (campaign_id, name, configs, runs), atomic write, CLI contract.
  - Update **§4** config registry table with `feature_subset`; update **§3** dependency graph (features/meta/campaign depend on evaluation/inference/config/_atomicio; no new cycles); update **§6** known-gaps: "feature engineering deferred" → "feature subset resolution + stability screening now supported (`nmr/features.py`); expression transforms still deferred".
- `AGENTS.md` — toolkit table additions (within the existing 32 KB budget, currently 20.8 KB):
  - `Change feature-set resolution / stability screening | nmr/features.py — resolve_feature_sets, feature_stability_screen, select_stable_features`
  - `Change cross-run meta-analysis / promotion verdicts | nmr/meta.py — paired_era_comparison, promotion_verdict, fleet_summary`
  - `Change campaign orchestration | nmr/campaign.py + run_campaign.py`
- `README.md` — annotated project tree: add `nmr/features.py`, `nmr/meta.py`, `nmr/campaign.py`, `run_campaign.py`, and `artifacts/campaigns/`; no commands duplicated (cross-reference CONTRIBUTING for verification).

- [ ] **Step 5: Run the exports + config tests, then full suite**

Run: `.\.venv\Scripts\python -m pytest tests/test_contribution.py tests/test_config.py tests/test_features.py tests/test_meta.py tests/test_campaign.py -q`
Expected: PASS. Then the full suite (Task 11) before sign-off.

- [ ] **Step 6: Record** — deliverable on disk; doc SSOT scan via `doc-ssot-hygiene` at sign-off.

---

### Task 10: Kimi skills S1–S4

**Files:**
- Create: project skill files (location per `writing-skills` conventions — invoke `superpowers:writing-skills` before authoring; expected `.superpowers/skills/<name>/SKILL.md` or the platform's project-skill location):
  - `.superpowers/skills/feature-campaign/SKILL.md`
  - `.superpowers/skills/hpo-narrowing/SKILL.md`
  - `.superpowers/skills/run-meta-analysis/SKILL.md`
  - `.superpowers/skills/verification-before-claim/SKILL.md`

**Interfaces:**
- Consumes: Tasks 1–9 APIs (`resolve_feature_sets`, `feature_stability_screen`, `select_stable_features`, `paired_era_comparison`, `promotion_verdict`, `fleet_summary`, `campaign_id`, `write_campaign_log`, `HyperparameterSweep`, `RunRegistry`).
- Produces: four skill protocols executable by Kimi sub-agents. No production code.

- [ ] **Step 1: Invoke `superpowers:writing-skills` and follow its conventions for project-scope skills.**

- [ ] **Step 2: Author `feature-campaign` (S1)** — protocol: discover (resolve all named sets), screen (`feature_stability_screen`), generate candidate subsets, materialize configs with `data.feature_subset`, run via `run_campaign.py`, Pareto-select. Hard rules: purge 8/16 protocol-enforced (code does not block weakening — `splitter.py` checks only the configured gap), no transforms in control plane, subsets are pure functions of `features.json`.

- [ ] **Step 3: Author `hpo-narrowing` (S2)** — three stages: coarse `HyperparameterSweep` at `fast` preset with `MetricSummary`-only metrics (`mean/std/sharpe/max_drawdown` — `_held_out_metric` does `getattr`; `corr_sharpe_ac` raises), narrow around top-k, confirm with full `ExperimentRunner` + `promotion_verdict` proposal (human commits champion). Budget rule: `fast` only for routine sweeps.

- [ ] **Step 4: Author `run-meta-analysis` (S3)** — ingest registry entries, pair only on aligned era windows + matching `oof_device`, apply `paired_era_comparison`, `fleet_summary` with explicit `n_trials` policy, group by config attrs, emit robust families. Never mutates the registry.

- [ ] **Step 5: Author `verification-before-claim` (S4)** — checklist: full `pytest -q`; canonical-hash purity triage for any new scorecard field; parity tests for metric changes; purge gate ≥8/16; seed threading + `deterministic=True`/`force_col_wise=True`; no business logic in scripts; doc SSOT same-change-set; artifact trust (cloudpickle only from `artifacts/`); `oof_device` logged before cross-run comparison.

- [ ] **Step 6: Verify each skill file renders (read it back) and cross-reference the exact APIs it names.**

---

### Task 11: Full verification gate

**Files:** none (verification only).

- [ ] **Step 1: Full suite**

Run: `.\.venv\Scripts\python -m pytest -q`
Expected: PASS, all tests (307 baseline + new: ~35–40). Report the exact count.

- [ ] **Step 2: Benchmark smoke (data-layer touched via T1 threading)**

Run: `.\.venv\Scripts\python benchmark_runner.py --fast-mode --output artifacts/benchmark_scores_smoke.csv --labels-output artifacts/benchmark_test_era_labels_smoke.csv`
Expected: exits 0; smoke CSV written. (Overwrites the checked-in smoke artifacts as documented.)

- [ ] **Step 3: Doc SSOT scan (`doc-ssot-hygiene`)**

Run the skill's checklist across AGENTS.md / ARCHITECTURE.md / README.md / CONTRIBUTING.md: no duplicated facts, no contradictions, AGENTS ≤ 32 KB, `docs/test_docs_hygiene.py` still green.

- [ ] **Step 4: Record** — report results truthfully; do not claim anything not run.

---

## Self-Review

**Spec coverage** (build list → task):
- T1 `nmr/features.py` + `DataConfig.feature_subset` → Tasks 1–3. ✔
- T2 `nmr/meta.py` (paired comparison, promotion verdict, fleet summary) → Tasks 4–6. ✔
- T3 `run_campaign.py` + `artifacts/campaigns/` log → Tasks 7–8 (campaign logic in `nmr/campaign.py` — control-plane boundary, deliberate refinement of the build list's "tests/test_scripts.py" note: logic tests live in `tests/test_campaign.py`, script gets an import smoke). ✔
- S1–S4 skills → Task 10. ✔
- Exports + SSOT docs → Task 9. ✔
- Verification gate → Task 11. ✔
- "NOT adding" (no NN, no BO lib, no metric changes, no invariants-guardian persona, no transforms in control plane) → respected throughout; `promotion_verdict` is advisory and never writes the registry. ✔

**Placeholder scan:** all code steps carry real code; no TBD/“similar to” references.

**Type consistency:** `resolved_feature_set` (Task 3) is used in Task 3 tests and by runner/research; `PairedResult` fields (Task 4) match the Task 4 tests; `promotion_verdict` output literals match Task 5 tests; `fleet_summary` column names match Task 6 tests; `campaign_id`/`build_campaign_log`/`write_campaign_log`/`CampaignRun` signatures match Tasks 7–8 tests; export names in Task 9 match the module `__all__`s.
