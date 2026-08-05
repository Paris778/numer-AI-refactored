# Codebase Sanity Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 28 audit findings (F-001..F-028) in `numer-AI-refactored` on branch `sanity-check`, so the deployed artifact equals the evaluated strategy, the benchmark runner works, promotion is guarded, and the dashboard ranks comparable metrics.

**Architecture:** Sequential, test-first clusters. P0 first (benchmark runner, deploy artifact, validation scorecard, unified dashboard), then P1 (registry/promotion, honest evaluation, model guards, risk cache), then P2 (vectorization, benchmark polish, CI), then P3 + docs re-sync. One commit per task; every intermediate state keeps the full pytest suite green.

**Tech Stack:** Python 3.12 (venv `.venv\Scripts\python`), Polars, pandas, NumPy/SciPy, LightGBM/XGBoost, cloudpickle 3.1.1, pytest (203 tests at baseline).

**Spec:** `docs/superpowers/specs/2026-08-05-codebase-sanity-check-design.md` (authoritative — read it before starting).

## Global Constraints

- **Test gate:** run the full suite after every task: `.\.venv\Scripts\python -m pytest -q` (repo root). Never claim green without executing.
- **Test-first:** write the failing test, run it to see it fail, then implement, then run to green.
- **No new third-party deps.** `nmr/` is the only tested boundary; scripts/notebooks contain zero business logic.
- **Determinism:** canonical hashes (`run_id`, `canonical_scorecards_bytes`, cache keys) exclude wall-clock fields and absolute paths. `dataclasses.asdict(config)` feeds the run_id — adding config fields intentionally changes run_ids (expected, not a regression).
- **Leakage:** era-purged validation only; never weaken `PurgedEraSplitter` invariants. New validation scorecard stage must drop the first `split.purge_eras` validation eras.
- **Atomic writes:** registry JSON, artifact payload + manifest, OOF parquet, and the risk-cache pair all write via temp file + fsync + `os.replace`.
- **SSOT same-commit:** any change that makes `AGENTS.md`/`ARCHITECTURE.md`/`CONTRIBUTING.md`/`README.md` stale requires updating that file in the same commit.
- **Fail loud:** no silent `except Exception`; catch specific types; no silent substitutions.
- **Test fixtures:** plan snippets reference existing helpers (`_risk_frame()`, `_model_frame(n_eras=...)`, `_tiny_model_params()`, `_anchor_splitter()`, `_result(run_id, sharpe=...)`, `_scorecard(...)`, `_train_frame()`). VERIFY each helper exists and matches the snippet's signature before writing a test — adapt the snippet to the file's actual helper if it differs. Never invent parallel helpers when an equivalent exists (e.g., `test_registry.py::_result` exists; `_scorecard` does not and must be created).
- **Commit style:** one commit per task on branch `sanity-check`; message prefix matches repo convention (`fix:`, `feat:`, `docs:`).
- **Python:** venv interpreter is `.venv\Scripts\python`; CI pins Python 3.12.

---

## Task 1: Benchmark runner integrity — public baseline generator (F-001, F-016)

**Files:**
- Modify: `nmr/benchmark.py` (add `iter_baseline_predictions`; refactor `run_classical_baselines` at lines 155-176 to consume it)
- Modify: `benchmark_runner.py:86-122` (`_candidate_strategies` migrates off private methods)
- Create: `tests/test_benchmark_baselines.py`

**Interfaces:**
- Produces: `BenchmarkSuite.iter_baseline_predictions(*, include_classical: bool = False, min_train_eras: int = 10) -> Iterator[tuple[str, str, pl.DataFrame, int]]` yielding `(model_id, group, raw_preds, seed)` in order: `NULL_BASELINES` (seed `base+idx`), `("trivial", "classical", …, base+3)`, and when `include_classical` also `("linear", …, base+4)` and `("tree", …, base+5)`; `base = self._eval_cfg.seed`.
- Consumes: existing `null_prediction_frame`, `_trivial_prediction_frame`, `_walk_forward_model_predictions` (private methods remain, but only the generator and internal callers touch them).
- Downstream: Task 10's script smoke test relies on `_candidate_strategies` calling only `iter_baseline_predictions`.

**Seed-convention guardrails (do not regress):**
- `run_classical_baselines` must preserve its historical behavior of scoring with the suite's default seed — call `self.evaluate_predictions(raw_preds, model_id=model_id)` **without** `seed=`; do NOT pass the generator's yielded seed (old code passed no seed → `run_seed = self._eval_cfg.seed`).
- `benchmark_runner` benchmark-model rows keep the historical `seed + 6` convention (they feed bootstrap CIs — changing them silently shifts every benchmark-model scorecard).

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmark_baselines.py`:

```python
"""Tests for the public baseline-prediction generator (F-001/F-016)."""

from __future__ import annotations

import polars as pl

from nmr.benchmark import NULL_BASELINES, BenchmarkSuite


def _suite(seed: int = 7) -> BenchmarkSuite:
    rows = []
    for era in range(1, 13):
        for idx in range(4):
            rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f1": float(idx) / 10.0,
                    "f2": float((idx % 2)) / 10.0,
                    "target": float(era) / 100.0,
                }
            )
    frame = pl.DataFrame(rows)
    return BenchmarkSuite(
        meta_model=frame.select(["era", "id"]).with_columns(
            pl.lit(0.1).alias("numerai_meta_model")
        ),
        benchmarks=pl.DataFrame(
            {"era": [], "id": [], "bench": []},
            schema={"era": pl.String, "id": pl.String, "bench": pl.Float64},
        ),
        features=frame.select(["era", "id", "f1", "f2"]),
        targets=frame.select(["era", "id", "target"]),
        n_trials=1,
        seed=seed,
        horizon="20D",
        n_boot=1,
        min_overlap_eras=2,
    )


def test_generator_yields_null_trivial_ordering_and_seed_convention() -> None:
    suite = _suite(seed=7)
    items = list(suite.iter_baseline_predictions(include_classical=False))
    ids = [model_id for model_id, _, _, _ in items]
    assert ids == [*NULL_BASELINES, "trivial"]
    assert [seed for _, _, _, seed in items] == [7, 8, 9, 10]


def test_generator_includes_classical_with_min_train_eras() -> None:
    suite = _suite(seed=7)
    items = list(suite.iter_baseline_predictions(include_classical=True, min_train_eras=2))
    ids = [model_id for model_id, _, _, _ in items]
    assert ids == [*NULL_BASELINES, "trivial", "linear", "tree"]
    assert [seed for _, _, _, seed in items] == [7, 8, 9, 10, 11, 12]
    for _, _, raw_preds, _ in items:
        assert {"era", "id", "prediction"} <= set(raw_preds.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_benchmark_baselines.py -v`
Expected: FAIL — `AttributeError: 'BenchmarkSuite' object has no attribute 'iter_baseline_predictions'`.

- [ ] **Step 3: Implement**

In `nmr/benchmark.py`, add the generator right after `run_classical_baselines` and refactor `run_classical_baselines` to consume it:

```python
    def iter_baseline_predictions(
        self,
        *,
        include_classical: bool = False,
        min_train_eras: int = 10,
    ) -> Iterator[tuple[str, str, pl.DataFrame, int]]:
        """Yield (model_id, group, raw_preds, seed) for null/trivial/classical baselines."""
        base = self._eval_cfg.seed
        for idx, baseline in enumerate(NULL_BASELINES):
            r_seed = base + idx
            yield (
                baseline,
                "null",
                self.null_prediction_frame(baseline, seed=r_seed),
                r_seed,
            )
        yield ("trivial", "classical", self._trivial_prediction_frame(), base + 3)
        if include_classical:
            yield (
                "linear",
                "classical",
                self._walk_forward_model_predictions(
                    model_name="linear", min_train_eras=min_train_eras
                ),
                base + 4,
            )
            yield (
                "tree",
                "classical",
                self._walk_forward_model_predictions(
                    model_name="tree", min_train_eras=min_train_eras
                ),
                base + 5,
            )

    def run_classical_baselines(
        self,
        *,
        min_train_eras: int = 10,
    ) -> dict[str, MetricScorecard]:
        """Generate and score S11 classical rungs: trivial, linear, and tree."""
        out: dict[str, MetricScorecard] = {}
        for model_id, group, raw_preds, _seed in self.iter_baseline_predictions(
            include_classical=True, min_train_eras=min_train_eras
        ):
            if group != "classical":
                continue
            # Historical behavior: score with the suite's default seed (no seed= arg).
            out[model_id] = self.evaluate_predictions(raw_preds, model_id=model_id)
        return out
```

Add `Iterator` to the `collections.abc` import at the top of `nmr/benchmark.py` (check current imports; `typing` usage exists).

In `benchmark_runner.py`, replace the body of `_candidate_strategies` (lines 86-122) with:

```python
def _candidate_strategies(
    suite: BenchmarkSuite,
    benchmarks: pl.DataFrame,
    seed: int,
    min_train_eras: int,
    fast_mode: bool,
) -> Iterator[StrategyContext]:
    for model_id, group, raw_preds, r_seed in suite.iter_baseline_predictions(
        include_classical=not fast_mode, min_train_eras=min_train_eras
    ):
        yield StrategyContext(model_id, group, raw_preds, r_seed)

    benchmark_cols = sorted([c for c in benchmarks.columns if c not in {"era", "id"}])
    for col in benchmark_cols:
        preds = benchmarks.select(["era", "id", pl.col(col).alias("prediction")])
        yield StrategyContext(col, "benchmark_model", preds, seed + 6)
```

The `seed` parameter stays live — benchmark-model rows keep the historical `seed + 6` (it feeds `evaluate_normalized_predictions(seed=...)` → bootstrap CIs). The suite was constructed with `seed=args.seed`, so the generator's base seed equals the old convention for null/trivial/classical rows.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_benchmark_baselines.py tests/test_benchmark_slice2.py -q`
Expected: PASS (slice2 regression: classical baselines still produce `{"trivial", "linear", "tree"}`).
Then full suite: `.\.venv\Scripts\python -m pytest -q` — expect 203+ passing, 0 failing.

- [ ] **Step 5: Commit**

```bash
git add nmr/benchmark.py benchmark_runner.py tests/test_benchmark_baselines.py
git commit -m "fix: restore benchmark runner integrity via public baseline generator (F-001, F-016)"
```

---

## Task 2: Deployed artifact = evaluated strategy (F-002, F-026, F-013-deployment, all-eras anchor, `neutralize_array`)

**Files:**
- Create: `nmr/_atomicio.py` (shared atomic write helpers — the field-tested `RunRegistry._atomic_json_write` pattern extracted to one implementation)
- Modify: `nmr/_transforms.py` (add `neutralize_array`; move design-matrix construction here)
- Modify: `nmr/risk.py` (`_neutralize_era` uses `neutralize_array` with cached pinv; underdetermined-era warning)
- Modify: `nmr/config.py` (new `RiskConfig` section with `neutralization_proportion`)
- Modify: `nmr/models.py` (add `train_full_history`, CPU-only, with null-target filter)
- Modify: `nmr/deployment.py` (atomic payload + manifest writes via `nmr._atomicio`)
- Modify: `nmr/registry.py` (`_atomic_json_write` delegates to `nmr._atomicio.atomic_write_text`)
- Modify: `nmr/runner.py` (`_build_deploy_pipeline` + `_serialize_predict_artifact(predict_fn, model_meta, artifact_path)` that only serializes; pipeline built exactly once in `run()`; `register_pickle_by_value`; manifest enrichment; run() uses configured proportion)
- Modify: `tests/test_runner.py` (deploy test updated — 6-row fixture), `tests/test_risk.py` (add neutralize_array parity test), `tests/test_models.py` (add train_full_history test), `tests/test_config.py` (RiskConfig test)
- Modify: `ARCHITECTURE.md` §2D/§2F/§2G/§2N, `AGENTS.md` §8 hazard note (same commit)

**Interfaces:**
- Produces:
  - `nmr._atomicio.atomic_write_bytes(path, data: bytes) -> None` and `nmr._atomicio.atomic_write_text(path, text, *, encoding="utf-8") -> None` — temp file in the target directory, single write handle (write → flush → fsync → close → `os.replace`). Used by deployment (Task 2), registry (Task 4), risk cache (Task 7).
  - `nmr._transforms.neutralize_array(pred: np.ndarray, features: np.ndarray, proportion: float = 1.0, *, pseudo_inverse: np.ndarray | None = None) -> np.ndarray` — validates finite, zero-std returns `pred.copy()` unchanged, builds design `[features | 1]`, pinv via `np.linalg.pinv(design, rcond=1e-6)` when `pseudo_inverse is None`, returns `pred - proportion * (design @ coeffs)`. **No logging inside** (it is embedded by value in the deployment closure — the engine logs instead).
  - `ModelOrchestrator.train_full_history(df, *, feature_cols, target_col, era_col="era") -> object` — fits one CPU-only model on all eras; null/non-finite targets filtered with logged drop count; `ValueError` if nothing remains.
  - `RiskConfig` (`neutralization_proportion: float = 1.0`, validated `0.0 <= p <= 1.0`) registered in `_SECTIONS` and `__all__`.
  - `ExperimentRunner._serialize_predict_artifact(predict_fn, model_meta, artifact_path) -> DeploymentArtifact` — **serializes only**; the pipeline (models + closure) is built once in `run()` via `_build_deploy_pipeline(...)` and shared with the validation stage.
- Consumes: Task 1's green state; existing `serialize_predict`, `rank_gaussianize`, `rank_gaussianize_unit_variance`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_risk.py`:

```python
from nmr._transforms import neutralize_array

def test_neutralize_array_cached_matches_uncached(tmp_path) -> None:
    df = _risk_frame().filter(pl.col("era") == "1")
    engine = NeutralizationEngine(cache_dir=tmp_path)
    # First call populates the cache; second hits it.
    engine.neutralize(df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0)
    result = engine.neutralize(
        df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0
    )
    pred = df.get_column("pred").to_numpy()
    features = df.select(["f1", "f2"]).to_numpy()
    direct = neutralize_array(pred, features, 1.0, pseudo_inverse=None)
    assert np.allclose(result.get_column("pred").to_numpy(), direct, atol=1e-12)


def test_neutralize_array_zero_variance_returns_unchanged() -> None:
    pred = np.full(5, 0.5)
    features = np.arange(10, dtype=float).reshape(5, 2)
    out = neutralize_array(pred, features, 1.0)
    assert np.array_equal(out, pred)
```

Append to `tests/test_models.py`:

```python
def test_train_full_history_covers_all_eras_and_is_cpu_only() -> None:
    df = _model_frame()
    orchestrator = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=5,
    )
    model = orchestrator.train_full_history(
        df, feature_cols=["f1", "f2", "f3"], target_col="target"
    )
    assert model is not None
    assert model.get_params()["device_type"] == "cpu"


def test_train_full_history_drops_null_targets() -> None:
    df = _model_frame(n_eras=4)
    df = df.with_columns(
        pl.when(pl.col("id") == "1_0").then(None).otherwise(pl.col("target")).alias("target")
    )
    orchestrator = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=5,
    )
    model = orchestrator.train_full_history(
        df, feature_cols=["f1", "f2", "f3"], target_col="target"
    )
    assert model is not None
```

Add to `tests/test_config.py`:

```python
def test_risk_section_validates_proportion() -> None:
    from nmr.config import RiskConfig
    assert RiskConfig().neutralization_proportion == 1.0
    assert RiskConfig(neutralization_proportion=0.0).neutralization_proportion == 0.0
    with pytest.raises(ValueError):
        RiskConfig(neutralization_proportion=1.5)
```

Update `tests/test_runner.py::test_runner_deploy_serializes_reloadable_predict` — the fixture must have **more rows than features+intercept** (2 features + intercept = 3 design columns; with ≤ 3 rows full neutralization fits exactly and the output is constant — `nunique() > 1` would fail forever):

```python
def test_runner_deploy_serializes_reloadable_predict(tmp_path) -> None:
    cfg = _config(tmp_path)
    result = ExperimentRunner(cfg).run(deploy=True)

    assert result.artifact is not None
    loaded_predict = load_predict(result.artifact.path)
    live_features = pd.DataFrame(
        {"f1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "f2": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
        index=[f"id_{i}" for i in range(6)],
    )
    prediction = loaded_predict(live_features)
    assert list(prediction.columns) == ["prediction"]
    assert prediction.index.tolist() == [f"id_{i}" for i in range(6)]
    assert prediction["prediction"].notna().all()
    assert prediction["prediction"].nunique() > 1  # non-constant pipeline output
```

Plan note (document, don't test): full neutralization annihilates predictions whenever an era has `n_rows <= n_features + 1` (the least-squares fit is exact) — the engine logs a warning in that case (implemented below).

Update `tests/test_registry.py::test_atomic_write_failure_keeps_previous_run_json` — because `_atomic_json_write` now delegates to `nmr._atomicio.atomic_write_text`, the failure-injection point moves to `nmr._atomicio.os.replace`:

```python
def test_atomic_write_failure_keeps_previous_run_json(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = RunRegistry(tmp_path)
    result = _result("run-a", sharpe=0.7)
    run_dir = registry.record(result)
    stable_json = (run_dir / "run.json").read_text(encoding="utf-8")

    import nmr._atomicio as atomicio_module

    def fail_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(atomicio_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        registry.record(result)

    assert (run_dir / "run.json").read_text(encoding="utf-8") == stable_json
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_risk.py tests/test_models.py tests/test_config.py tests/test_runner.py -q`
Expected: FAIL — `ImportError: cannot import name 'neutralize_array'`, `AttributeError: 'ModelOrchestrator' object has no attribute 'train_full_history'`, `ImportError: cannot import name 'RiskConfig'`.

- [ ] **Step 3: Implement**

**3a. `nmr/_transforms.py`** — append:

```python
def neutralize_array(
    pred: np.ndarray,
    features: np.ndarray,
    proportion: float = 1.0,
    *,
    pseudo_inverse: np.ndarray | None = None,
) -> np.ndarray:
    """Per-era intercept-aware linear neutralization (single source of truth).

    The engine passes its cached per-era pseudo-inverse; the deployment closure
    passes ``None`` and the design pseudo-inverse is computed here so both paths
    share identical geometry and ``rcond``. Zero-variance predictions are
    returned unchanged (the era keeps its rows; callers decide on logging).
    """
    pred_array = np.asarray(pred, dtype=float).reshape(-1)
    feature_matrix = np.asarray(features, dtype=float)
    if pred_array.shape[0] != feature_matrix.shape[0]:
        raise ValueError("pred and features must have the same number of rows")
    if not np.all(np.isfinite(pred_array)) or not np.all(np.isfinite(feature_matrix)):
        raise ValueError("pred and features must contain only finite values")
    if np.std(pred_array) == 0.0:
        return pred_array.copy()

    design = np.hstack(
        (feature_matrix, np.ones((feature_matrix.shape[0], 1), dtype=float))
    )
    if pseudo_inverse is None:
        pseudo_inverse = np.asarray(np.linalg.pinv(design, rcond=1e-6), dtype=float)
    coeffs = pseudo_inverse.dot(pred_array)
    adjustment = design.dot(coeffs)
    return pred_array - (proportion * adjustment)
```

Add `"neutralize_array"` to `_transforms.__all__`.

**3b. `nmr/risk.py`** — replace the body of `_neutralize_era` (lines 92-116) to delegate the solve:

```python
    def _neutralize_era(
        self,
        era_df: pl.DataFrame,
        *,
        era_label: str,
        pred_col: str,
        feature_cols: Sequence[str],
        proportion: float,
    ) -> np.ndarray:
        pred = self._column_values(era_df, pred_col)
        features = self._feature_matrix(era_df, feature_cols)
        if np.std(pred) == 0.0:
            logger.warning(
                "[neutralize] era %s has zero-variance predictions; returning unchanged",
                era_label,
            )
            return np.asarray(pred, dtype=float).copy()
        if era_df.height <= len(list(feature_cols)) + 1:
            logger.warning(
                "[neutralize] era %s has %d rows <= %d features+intercept; "
                "neutralization fits exactly and the output may be near-zero",
                era_label, era_df.height, len(feature_list),
            )

        design = _design_matrix(features)
        pseudo_inverse = self._load_or_compute_pseudo_inverse(
            era_df,
            era_label=era_label,
            feature_cols=feature_cols,
            design=design,
        )
        return neutralize_array(
            pred, features, proportion, pseudo_inverse=pseudo_inverse
        )
```

Delete `_design_matrix` and `_compute_pseudo_inverse` from the class; add module-level helpers:

```python
def _design_matrix(features: np.ndarray) -> np.ndarray:
    return np.hstack((features, np.ones((features.shape[0], 1), dtype=float)))


def _compute_pseudo_inverse(design: np.ndarray) -> np.ndarray:
    return np.asarray(np.linalg.pinv(design, rcond=1e-6), dtype=float)
```

Update `_load_or_compute_pseudo_inverse` to call `_compute_pseudo_inverse(design)` (module-level) instead of the deleted method. Update imports: `from nmr._transforms import neutralize_array`. (The zero-variance warning log lands here — the B4 contract, pinned by a Task 7 test.)

**3c. `nmr/config.py`** — add after `EvalConfig`:

```python
@dataclass(frozen=True)
class RiskConfig:
    """Risk transforms: neutralization strength and cache budget."""

    neutralization_proportion: float = 1.0
    cache_max_bytes: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.neutralization_proportion <= 1.0:
            raise ValueError("risk.neutralization_proportion must be in [0, 1]")
        if self.cache_max_bytes is not None and self.cache_max_bytes < 0:
            raise ValueError("risk.cache_max_bytes must be >= 0 or None")
```

Add `risk: RiskConfig = field(default_factory=RiskConfig)` to `ExperimentConfig`, `"risk": RiskConfig` to `_SECTIONS`, and `"RiskConfig"` to `__all__`.

**3d. `nmr/models.py`** — add:

```python
    def train_full_history(
        self,
        df: pl.DataFrame,
        *,
        feature_cols: Sequence[str],
        target_col: str,
        era_col: str = "era",
    ) -> object:
        """Fit a single CPU-only model on every era (deployment/validation artifact).

        CPU-only by design: determinism is per-device and the deployed model must
        reproduce identically on any hosted runtime (which may lack a GPU).
        """
        train_df = df.filter(pl.col(era_col).is_not_null())
        train_df = train_df.filter(
            pl.col(target_col).is_not_null() & pl.col(target_col).is_finite()
        )
        dropped = df.height - train_df.height
        if dropped:
            logger.warning(
                "[train_full_history] dropped %d rows with null/non-finite %s targets",
                dropped,
                target_col,
            )
        if train_df.is_empty():
            raise ValueError("No usable training rows after null-target filtering")
        model = self._fit_model(
            features=self._feature_frame(train_df, feature_cols=feature_cols),
            target=train_df.get_column(target_col).to_numpy(),
            use_gpu=False,
        )
        logger.info(
            "[train_full_history] %s: fitted on %d rows (all eras)", target_col, train_df.height
        )
        return model
```

Then make `_fit_model` CPU-only-aware: `_fit_model` currently tries GPU first. `train_full_history` must not. Add a private flag:

```python
    def _fit_model(
        self, *, features: pd.DataFrame, target: np.ndarray, use_gpu: bool = True
    ) -> object:
```

Call sites: `_fit_predict_fold` passes `use_gpu=True` (current behavior preserved); `train_full_history` passes `use_gpu=False`. `_device_candidate_params(use_gpu: bool) -> list[dict[str, Any]]` returns `[self._resolved_params(use_gpu=use_gpu)]` when `use_gpu=False`. (F-009 logging is Task 6 — keep `_fit_model`'s loop shape unchanged here beyond the flag.)

**3e. Create `nmr/_atomicio.py`** — one shared atomic-write implementation (the field-tested `RunRegistry._atomic_json_write` pattern), reused by deployment (this task), registry (Task 4), and the risk cache (Task 7). Single write handle (write → flush → fsync → close → `os.replace`) — a read-only reopen for fsync is unreliable on Windows:

```python
"""Atomic file writes: temp file in the target directory + fsync + os.replace.

This is the single implementation of the repo's atomic-write contract
(AGENTS.md §9). Every registry JSON write, artifact payload/manifest write,
OOF parquet write, and neutralization-cache write goes through these helpers.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp + fsync + os.replace)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f"{target.name}.tmp.", suffix=".part"
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def atomic_write_text(
    path: str | Path, text: str, *, encoding: str = "utf-8"
) -> None:
    """Write ``text`` to ``path`` atomically (UTF-8 by default)."""
    atomic_write_bytes(path, text.encode(encoding))
```

**3f. `nmr/deployment.py`** — make `serialize_predict` atomic via the shared helper (payload first, manifest last):

```python
    from nmr._atomicio import atomic_write_bytes, atomic_write_text

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"predict_fn": predict_fn, "models": models}
    payload_bytes = cloudpickle.dumps(payload)
    atomic_write_bytes(artifact_path, payload_bytes)
    logger.info(
        "[serialize_predict] artifact written to %s (%d bytes)",
        artifact_path,
        len(payload_bytes),
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(feature_names),
        "sha256": _sha256_bytes(payload_bytes),
        "environment": _environment_fingerprint(),
    }
    atomic_write_text(
        _manifest_path(artifact_path),
        json.dumps(manifest, sort_keys=True, indent=2),
    )
    return DeploymentArtifact(path=artifact_path, manifest=manifest)
```

**3g. `nmr/registry.py`** — make `_atomic_json_write` delegate to the shared helper (keeps the existing test `test_atomic_write_failure_keeps_previous_run_json` green — it monkeypatches `nmr.registry.os.replace`):

```python
    def _atomic_json_write(self, path: Path, payload: dict[str, Any]) -> None:
        atomic_write_text(path, json.dumps(payload, sort_keys=True, indent=2))
```

with `from nmr._atomicio import atomic_write_text` at the top of `nmr/registry.py` (the existing `tempfile`/`os` imports may become unused — remove them if so).

**3h. `nmr/runner.py`** — add module-level imports `import cloudpickle`, `from nmr import _transforms`, `from nmr._transforms import neutralize_array, rank_gaussianize, rank_gaussianize_unit_variance`. `_build_deploy_pipeline` builds the closure (unchanged from the original design); `_serialize_predict_artifact` now **serializes only** — it takes the prebuilt pipeline and must never retrain models:

```python
    def _build_deploy_pipeline(
        self,
        *,
        orchestrator: ModelOrchestrator,
        train_df: pl.DataFrame,
        feature_cols: Sequence[str],
        target_cols: Sequence[str],
        weights: Sequence[float],
        proportion: float,
    ) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], dict[str, object]]:
        """Train per-target full-history models ONCE and return (predict, model_meta).

        The closure's code path references only numpy/pandas plus the shared
        transform helpers; cloudpickle.register_pickle_by_value(nmr._transforms)
        embeds those helpers by value so the artifact loads without `nmr`.
        """
        logger.info("[build_deploy_pipeline] training full-history models (CPU-only)")
        trained: dict[str, object] = {}
        for target in target_cols:
            trained[target] = orchestrator.train_full_history(
                train_df,
                feature_cols=feature_cols,
                target_col=target,
                era_col="era",
            )
        ordered_features = list(feature_cols)
        target_order = list(target_cols)
        weight_array = np.asarray(list(weights), dtype=float)

        def predict(
            live_features: pd.DataFrame,
            live_benchmark_models: pd.DataFrame = None,
        ) -> pd.DataFrame:
            del live_benchmark_models
            frame = live_features.loc[:, ordered_features]
            components = [
                np.asarray(trained[t].predict(frame), dtype=float)
                for t in target_order
            ]
            design = np.column_stack(components)
            if "era" in live_features.columns:
                era_values = live_features["era"].astype(str).to_numpy()
            else:
                era_values = np.full(len(live_features), "1")
            feature_matrix = frame.to_numpy(dtype=float)
            blended = np.empty(len(live_features), dtype=float)
            for era in np.unique(era_values):
                mask = era_values == era
                block = design[mask]
                normalized = np.column_stack(
                    [
                        rank_gaussianize_unit_variance(block[:, i])
                        for i in range(block.shape[1])
                    ]
                )
                combined = rank_gaussianize(normalized.dot(weight_array))
                blended[mask] = neutralize_array(
                    combined, feature_matrix[mask], proportion
                )
            return pd.DataFrame({"prediction": blended}, index=live_features.index)

        meta = {
            "targets": target_order,
            "weights": [float(w) for w in weights],
            "proportion": float(proportion),
            "geometry": "all_eras",
            "device": "cpu",
        }
        return predict, meta

    def _serialize_predict_artifact(
        self,
        *,
        predict_fn: Callable[[pd.DataFrame], pd.DataFrame],
        model_meta: dict[str, object],
        artifact_path: Path,
    ) -> DeploymentArtifact:
        """Serialize the prebuilt pipeline closure. Does NOT retrain models."""
        cloudpickle.register_pickle_by_value(_transforms)
        return serialize_predict(
            predict_fn,
            path=artifact_path,
            feature_names=list(model_meta["feature_names"]),
            models=model_meta,
        )
```

(`model_meta` carries `feature_names` too — add `"feature_names": ordered_features` to the `meta` dict in `_build_deploy_pipeline`.)

In `run()`, change the proportion source and the deploy block — the pipeline is built **exactly once** when deploy is requested (Task 3 will extend this to `deploy or validation_scorecard`):

```python
        neutralization_proportion = self._config.risk.neutralization_proportion
        neutralized = NeutralizationEngine(
            max_cache_bytes=self._config.risk.cache_max_bytes
        ).neutralize(
            blended,
            pred_col="prediction",
            feature_cols=feature_cols,
            era_col="era",
            proportion=neutralization_proportion,
        )
        ...
        artifact = None
        if deploy:
            logger.info("[run] serializing deploy artifact")
            pipeline = self._build_deploy_pipeline(
                orchestrator=model_orchestrator,
                train_df=train_df,
                feature_cols=feature_cols,
                target_cols=target_cols,
                weights=weights,
                proportion=neutralization_proportion,
            )
            artifact = self._serialize_predict_artifact(
                predict_fn=pipeline[0],
                model_meta=pipeline[1],
                artifact_path=(
                    self._config.run.artifacts_dir / "runs" / self._run_id / "predict.pkl"
                ),
            )
            logger.info("[run] artifact written to %s", artifact.path)
```

Add to the manifest dict: `"pipeline_device": "cpu"` (the full-history models are CPU-only; the OOF-CV device is recorded separately in Task 6 as `"oof_device"`), and keep `"weights": list(weights)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_risk.py tests/test_models.py tests/test_config.py tests/test_runner.py tests/test_deployment.py -q`
Expected: PASS.
Then full suite: `.\.venv\Scripts\python -m pytest -q` — green.

- [ ] **Step 5: Docs + commit**

- `ARCHITECTURE.md` §2D: add `neutralize_array` row to the transforms table. §2F: note the engine delegates the solve to `neutralize_array` and zero-variance eras return predictions unchanged (logged). §2G: add `train_full_history` (CPU-only, all eras, null-target filter) to the ModelOrchestrator list; note the anchor fold is no longer used for deployment. §2N: rewrite the deploy sentence — artifact wraps the full pipeline (per-target all-eras models + rank-gaussianize + weights + neutralize), no `splitter` param, manifest carries `targets/weights/proportion/geometry/device`.
- `AGENTS.md` §8: add a hazard entry: "Deployment closure embeds `nmr._transforms` helpers by value via `cloudpickle.register_pickle_by_value`; the artifact's predict path depends only on numpy/scipy/pandas at load time (no `nmr` import). The fidelity test (`tests/test_runner.py`) is the drift guard — never hand-duplicate the transform math inside the closure."
- `AGENTS.md` §9: reword the atomicity bullet to: "Registry JSON, artifact payload + manifest, OOF parquet, and the neutralization-cache pair all write via temp + fsync + `os.replace`."

```bash
git add nmr/_transforms.py nmr/risk.py nmr/config.py nmr/models.py nmr/deployment.py nmr/runner.py tests/ ARCHITECTURE.md AGENTS.md
git commit -m "fix: deploy full evaluated strategy (per-target all-eras pipeline) instead of raw anchor model (F-002, F-026, F-013)"
```

---

## Task 3: Validation scorecard stage + unified dashboard + vectorized exposure (FEAT-002, F-004, F-023, F-005, F-019)

**Files:**
- Modify: `nmr/config.py` (`EvalConfig.validation_scorecard: bool = True`)
- Modify: `nmr/scorecard.py` (`evaluate_model` gains `backend: str = "custom"`)
- Modify: `nmr/research.py` (vectorized `feature_exposure_report`, per-era Pearson)
- Modify: `nmr/runner.py` (validation stage; `RunResult.scorecard` + `RunResult.validation_predictions`; load validation/meta/benchmarks; purge-drop; share full-history models)
- Modify: `generate_dashboard.py` (scorecard-driven trained rows, legacy section, `html.escape`)
- Modify: `tests/test_runner.py` (fixture validation data; `validation_scorecard=False` in `_config`; validation-stage tests incl. purge-drop + fidelity)
- Modify: `ARCHITECTURE.md` §L/§2A/§2N, `README.md` (dashboard description) — same commit

**Interfaces:**
- Produces: `RunResult` gains `scorecard: MetricScorecard | None = None` and `validation_predictions: pl.DataFrame | None = None` (frozen dataclass, defaulted — existing constructions unchanged). `feature_exposure_report` keeps its signature and output columns but computes per-era **Pearson** correlation. `evaluate_model(..., backend="custom")`.
- Consumes: Task 2's `_build_deploy_pipeline` closure (validation stage calls the same closure so the scorecard and artifact share the identical code path).

- [ ] **Step 1: Write the failing tests**

Update `tests/test_runner.py`:

1. `_config()` — set `validation_scorecard=False`:

```python
        evaluation=EvalConfig(
            backend="custom", main_target="target", validation_scorecard=False
        ),
```

2. Extend `_write_synthetic_data` to also write validation-era assets:

```python
def _write_synthetic_data(root) -> None:
    version_dir = root / "vtest"
    version_dir.mkdir(parents=True, exist_ok=True)
    features = {
        "feature_sets": {"small": ["f1", "f2"], "medium": ["f1", "f2"], "all": ["f1", "f2"]},
        "targets": ["target", "target_alt"],
    }
    (version_dir / "features.json").write_text(json.dumps(features), encoding="utf-8")
    _build_train_frame().write_parquet(version_dir / "train.parquet")

    val_rows = []
    for era in range(13, 19):
        for idx in range(6):
            f1 = (era * 0.03) + (idx * 0.02)
            f2 = (era * -0.02) + (idx * 0.01)
            val_rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f1": f1,
                    "f2": f2,
                    "target": 0.6 * f1 - 0.3 * f2 + 0.05 * era,
                    "target_alt": 0.2 * f1 + 0.7 * f2 - 0.04 * era,
                }
            )
    val = pl.DataFrame(val_rows)
    val.write_parquet(version_dir / "validation.parquet")
    val.select(["era", "id"]).with_columns(
        pl.lit(0.35).alias("numerai_meta_model")
    ).write_parquet(version_dir / "meta_model.parquet")
    val.select(["era", "id"]).with_columns(
        pl.lit(0.2).alias("bench_cyrusd_20")
    ).write_parquet(version_dir / "validation_benchmark_models.parquet")
```

3. New tests:

```python
def _validation_config(tmp_path) -> ExperimentConfig:
    cfg = _config(tmp_path)
    return ExperimentConfig(
        data=cfg.data,
        split=cfg.split,
        model=cfg.model,
        evaluation=EvalConfig(
            backend="custom", main_target="target", validation_scorecard=True
        ),
        run=cfg.run,
    )


def test_validation_stage_produces_scorecard_and_purges_first_eras(tmp_path) -> None:
    cfg = _validation_config(tmp_path)
    result = ExperimentRunner(cfg).run(deploy=True)

    assert result.scorecard is not None
    assert result.scorecard.model_id == result.run_id
    assert result.manifest["validation_purge_dropped_first_eras"] == cfg.split.purge_eras
    assert result.validation_predictions is not None
    scored_eras = set(result.validation_predictions.get_column("era").to_list())
    assert "13" not in scored_eras  # purge_eras=1 -> era 13 dropped
    assert "14" in scored_eras
    assert result.artifact is not None


def test_deployed_artifact_matches_validation_stage_predictions(tmp_path) -> None:
    """F-019 fidelity: load_predict reproduces the scored validation pipeline."""
    import pandas as pd
    from scipy.stats import spearmanr

    cfg = _validation_config(tmp_path)
    result = ExperimentRunner(cfg).run(deploy=True)
    assert result.artifact is not None and result.validation_predictions is not None

    val = pl.read_parquet(cfg.data.path("validation.parquet")).filter(
        pl.col("era") != "13"
    )
    features_pd = val.select(["id", "era", "f1", "f2"]).to_pandas().set_index("id")
    loaded = load_predict(result.artifact.path)
    out = loaded(features_pd)

    expected = result.validation_predictions.sort("id")
    actual = pl.from_pandas(out.reset_index().rename(columns={"index": "id"})).sort("id")
    rho, _ = spearmanr(expected.get_column("prediction").to_numpy(),
                       actual.get_column("prediction").to_numpy())
    assert rho > 0.999
    assert np.allclose(
        expected.get_column("prediction").to_numpy(),
        actual.get_column("prediction").to_numpy(),
        atol=1e-12,
    )
```

Add `import numpy as np` to test_runner.py if missing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_runner.py -q`
Expected: FAIL — `EvalConfig.__init__() got an unexpected keyword argument 'validation_scorecard'`, then missing `scorecard` attribute.

- [ ] **Step 3: Implement**

**3a. `nmr/config.py`** — add to `EvalConfig`:

```python
    validation_scorecard: bool = True
```

**3b. `nmr/scorecard.py`** — add the `backend` parameter:

```python
def evaluate_model(
    predictions: pl.DataFrame,
    *,
    ...
    backend: str = "custom",
    model_id: str = "model",
    ...
```

and replace the hardcoded engine construction (line 393) with:

```python
    evaluator = EvaluationEngine(backend)
```

**3c. `nmr/research.py`** — replace `feature_exposure_report` (lines 136-178) with the vectorized Pearson version:

```python
def feature_exposure_report(
    oof: pl.DataFrame,
    *,
    feature_cols: Sequence[str],
    era_col: str = "era",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Per-era Pearson correlation of predictions vs each feature (vectorized).

    Definition (documented in ARCHITECTURE.md §L): plain Pearson correlation of
    the raw prediction and feature columns per era, then aggregated. This is the
    community-standard exposure definition (it is NOT the power-1.5 Numerai
    CORR used by per_era_corr). Values changed vs the pre-2026-08-05
    implementation; recorded exposure numbers are not comparable across that
    boundary.
    """
    feature_list = list(feature_cols)
    if not feature_list:
        raise ValueError("feature_cols must contain at least one feature")

    per_era: dict[str, np.ndarray] = {}
    parts = oof.select([era_col, pred_col, *feature_list]).partition_by(
        era_col, maintain_order=True
    )
    for part in parts:
        era = str(part.get_column(era_col).to_list()[0])
        clean = part.drop_nulls()
        if clean.is_empty():
            per_era[era] = np.zeros(len(feature_list), dtype=float)
            continue
        pred = clean.get_column(pred_col).cast(pl.Float64).to_numpy()
        features = clean.select(feature_list).cast(pl.Float64).to_numpy()
        per_era[era] = _pred_feature_pearson(pred, features)

    eras = sorted(per_era, key=int)
    matrix = np.column_stack([per_era[era] for era in eras])
    rows = [
        {
            "feature": feature,
            "mean_abs_exposure": float(np.mean(np.abs(matrix[i]))),
            "max_abs_exposure": float(np.max(np.abs(matrix[i]))),
        }
        for i, feature in enumerate(feature_list)
    ]
    return pl.DataFrame(rows).sort("max_abs_exposure", descending=True)


def _pred_feature_pearson(pred: np.ndarray, features: np.ndarray) -> np.ndarray:
    pred_centered = pred - np.mean(pred)
    pred_norm = float(np.linalg.norm(pred_centered))
    if pred_norm == 0.0:
        return np.zeros(features.shape[1], dtype=float)
    feature_centered = features - np.mean(features, axis=0)
    denoms = np.linalg.norm(feature_centered, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrs = (feature_centered.T @ pred_centered) / (denoms * pred_norm)
    return np.where(np.isfinite(corrs), corrs, 0.0)
```

**3d. `nmr/runner.py`** — add fields to `RunResult`, add the validation stage, and wire `scorecard`/`validation_predictions`:

```python
@dataclass(frozen=True)
class RunResult:
    run_id: str
    oof: pl.DataFrame
    metrics: MetricSummary
    artifact: DeploymentArtifact | None
    manifest: dict[str, Any]
    scorecard: MetricScorecard | None = None
    validation_predictions: pl.DataFrame | None = None
```

In `run()` (after the OOF metric computation, before the deploy block), build the shared pipeline and run the validation stage:

```python
        pipeline = None
        if deploy or self._config.evaluation.validation_scorecard:
            pipeline = self._build_deploy_pipeline(
                orchestrator=model_orchestrator,
                train_df=train_df,
                feature_cols=feature_cols,
                target_cols=target_cols,
                weights=weights,
                proportion=neutralization_proportion,
            )

        scorecard = None
        validation_predictions = None
        validation_purge = None
        if self._config.evaluation.validation_scorecard:
            scorecard, validation_predictions, validation_purge = (
                self._run_validation_stage(
                    predict_fn=pipeline[0], feature_cols=feature_cols
                )
            )
            logger.info("[run] validation scorecard ready: corr_sharpe_ac=%.5f",
                        scorecard.corr_sharpe_ac.value)

        artifact = None
        if deploy:
            logger.info("[run] serializing deploy artifact")
            assert pipeline is not None  # built once above when deploy=True
            artifact = self._serialize_predict_artifact(
                predict_fn=pipeline[0],
                model_meta=pipeline[1],
                artifact_path=(
                    self._config.run.artifacts_dir / "runs" / self._run_id / "predict.pkl"
                ),
            )
            logger.info("[run] artifact written to %s", artifact.path)
```

(`pipeline` is built once and reused by both stages — the design's "trained once and shared".)

Add the stage method and update the manifest:

```python
    def _run_validation_stage(
        self, *, predict_fn, feature_cols: Sequence[str]
    ) -> tuple[MetricScorecard, pl.DataFrame, int]:
        data = self._config.data
        agent = IngestionAgent(data)
        target_cols = list(
            dict.fromkeys([*self._config.data.targets, self._config.evaluation.main_target])
        )
        val_df = agent.load(
            "validation", columns=["era", "id", *feature_cols, *target_cols]
        )
        meta_path = data.path("meta_model.parquet")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"validation_scorecard=true requires {meta_path}; disable the "
                "validation stage or provide the meta model"
            )
        meta_model = pl.read_parquet(meta_path).select(["era", "id", "numerai_meta_model"])

        bench_path = data.path("validation_benchmark_models.parquet")
        benchmarks = (
            pl.read_parquet(bench_path) if bench_path.exists() else None
        )
        if benchmarks is None:
            logger.warning("[validation] benchmark models missing; BMC/horizon disabled")

        purge = self._config.split.purge_eras
        all_eras = sorted({int(e) for e in val_df.get_column("era").unique().to_list()})
        if purge > 0:
            keep = {str(e) for e in all_eras[purge:]}
            val_df = val_df.filter(pl.col("era").is_in(keep))
        logger.info(
            "[validation] dropping first %d validation eras (20D-target overlap); "
            "%d eras scored", purge, val_df.select(pl.col("era").n_unique()).item()
        )

        features_pd = val_df.select(["id", "era", *feature_cols]).to_pandas().set_index("id")
        prediction_frame = predict_fn(features_pd)
        preds = val_df.select(["era", "id"]).with_columns(
            pl.Series("prediction", prediction_frame["prediction"].to_numpy())
        )
        scorecard = evaluate_model(
            preds,
            meta_model=meta_model,
            benchmarks=benchmarks,
            features=val_df.select(["era", "id", *feature_cols]),
            targets=val_df.select(["era", "id", *target_cols]),
            n_trials=1,
            seed=self._config.run.seed,
            horizon="20D",
            main_target=self._config.evaluation.main_target,
            benchmark_col=(
                # First non-join column (same convention as benchmark_runner) —
                # never positional index 2, which assumes column order.
                next(
                    (
                        col
                        for col in benchmarks.columns
                        if col not in {"era", "id"}
                    ),
                    None,
                )
                if benchmarks is not None
                else None
            ),
            backend=self._config.evaluation.backend,
            model_id=self._run_id,
        )
        return scorecard, preds, purge
```

`benchmark_col` selection: benchmark models frame columns are `[era, id, <model cols...>]`; pick the first non-join column like `benchmark_runner` does. Update the manifest dict with:

```python
            "validation_purge_dropped_first_eras": validation_purge,
```

(only when the stage ran — include the key always, value `None` when skipped) and add imports: `from nmr.scorecard import MetricScorecard, evaluate_model`.

Update the final `return RunResult(...)` in `run()` to pass the new fields explicitly (a frozen dataclass with defaults will silently omit them otherwise):

```python
        return RunResult(
            run_id=self._run_id,
            oof=oof,
            metrics=metrics,
            artifact=artifact,
            manifest=manifest,
            scorecard=scorecard,
            validation_predictions=validation_predictions,
        )
```

**3e. `generate_dashboard.py`** — trained rows read `run.json["scorecard"]`; legacy rows excluded from ranking; escape all interpolations. In `_load_registry_runs`, change the metrics source:

```python
        scorecard = payload.get("scorecard") or {}
        sc_mean = scorecard.get("corr")
        sc_std = scorecard.get("std_corr")
        sc_sharpe = scorecard.get("corr_sharpe_ac")
        sc_dd = scorecard.get("max_drawdown")
        rows.append(
            {
                "model_id": payload.get("run_id", run_file.parent.name),
                "source": "trained" if scorecard else "trained_legacy",
                "run_name": run_cfg.get("name", "unknown"),
                "feature_set": data_cfg.get("feature_set", "unknown"),
                "backend": model_cfg.get("backend", "unknown"),
                "preset": model_cfg.get("preset", "unknown"),
                "n_targets": len(data_cfg.get("targets", [])),
                "targets": ", ".join(data_cfg.get("targets", [])),
                # Explicit None checks: a legitimate scorecard value of 0.0 must
                # NOT fall through to the legacy OOF metric.
                "mean": float(sc_mean if sc_mean is not None else metrics.get("mean", 0.0)),
                "std": float(sc_std if sc_std is not None else metrics.get("std", 0.0)),
                "sharpe": float(sc_sharpe if sc_sharpe is not None else metrics.get("sharpe", 0.0)),
                "max_drawdown": float(
                    sc_dd if sc_dd is not None else metrics.get("max_drawdown", 0.0)
                ),
                "artifact_path": payload.get("artifact_path"),
                "run_dir": str(run_file.parent),
            }
        )
```

In `generate_dashboard`, split legacy rows into a secondary table and rank only comparable rows:

```python
    trained = _load_registry_runs(registry_dir)
    benchmarks = _load_benchmarks(benchmark_path)
    combined = pd.concat([trained, benchmarks], ignore_index=True)
    comparable = combined[combined["source"] != "trained_legacy"].copy()
    legacy = combined[combined["source"] == "trained_legacy"].copy()
    ranked = _rank_models(comparable) if not comparable.empty else comparable
```

Update `_build_html` to render the legacy section (a second `<table>` below the ranked one) and wrap every interpolated cell with `html.escape(...)`. Add `import html` at the top. Specifically: `title="{html.escape(str(row['model_id']))}"`, `{html.escape(str(row['model_id']))[:16]}`, and each of `row['run_name']`, `row['feature_set']`, `row['backend']`, `row['preset']`, `row['targets']`, and the header/footer interpolations (`registry_dir`, `benchmark_path.name`, `REPO_ROOT`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_runner.py tests/test_research.py tests/test_scorecard.py -q`
Expected: PASS (scorecard's `backend` default keeps existing tests green; exposure test only asserts determinism/columns/sort).
Then full suite: `.\.venv\Scripts\python -m pytest -q` — green.

- [ ] **Step 5: Docs + commit**

**Determinism-fixture discipline:** the exposure-definition change (F-005) alters `max_feature_exposure` and therefore any scorecard-derived hash/value fixture that pins exposure numbers. If `tests/test_benchmark_slice1.py`/`slice3.py`/`test_scorecard.py`/`test_research.py` pin such values, update them **deliberately** — the commit message must state the old→new hash/value pair (a silently regenerated fixture is indistinguishable from a determinism regression). The cross-process tests in slice1/slice3 compare two subprocess outputs against each other, so they survive the change untouched.

- `ARCHITECTURE.md` §2A: add `validation_scorecard=True` row to the evaluation table; note `metrics` semantics (corr/fnc/sharpe on train OOF; mmc/bmc/cwmm validation-only). §L: replace the exposure bullet with "per-era Pearson correlation (vectorized; definition change dated 2026-08-05 — numbers not comparable with earlier runs)". §K: note `evaluate_model(..., backend="custom")`. §N: validation stage + `RunResult.scorecard`/`validation_predictions` + purge-drop + shared pipeline.
- `README.md`: dashboard now ranks trained runs and benchmarks on the same validation-scorecard definitions; legacy runs shown separately.

```bash
git add nmr/config.py nmr/scorecard.py nmr/research.py nmr/runner.py generate_dashboard.py tests/test_runner.py ARCHITECTURE.md README.md
git commit -m "feat: validation scorecard stage, unified dashboard, vectorized Pearson exposure (F-004, F-005, F-019, F-023)"
```

---

## Task 4: Guarded promotion & registry hardening (F-003, F-021, F-022, F-013-registry)

**Files:**
- Modify: `nmr/registry.py` (`promote_if_better`, direction-aware metric validation, deterministic `best()`, stable `list()`, run-id regex, atomic OOF parquet, scorecard block)
- Modify: `train_first_model.py` (use `promote_if_better`)
- Modify: `tests/test_registry.py` (64-hex run_ids; new promotion tests)
- Modify: `ARCHITECTURE.md` §N (registry API) — same commit

**Interfaces:**
- Produces: `RunRegistry.promote_if_better(run_id, metric="corr_sharpe_ac") -> tuple[Path, bool]`; `_SCORECARD_METRIC_FIELDS` + `_SCORECARD_METRIC_DIRECTION` module constants; `best(metric="sharpe")` validates metric against `MetricSummary` fields; `promote(run_id)` and `promote_if_better` regex-validate `run_id`.
- Consumes: Task 3's `RunResult.scorecard`.

- [ ] **Step 1: Write the failing tests**

Rewrite `tests/test_registry.py` — replace `"run-a"`/`"run-b"` with 64-hex ids (`"a" * 64`, `"b" * 64`) in the existing tests, and append:

```python
from nmr.scorecard import MetricCell, MetricScorecard


def _scorecard(sharpe_ac: float, *, max_drawdown: float = 0.1) -> MetricScorecard:
    cell = lambda v: MetricCell(value=v, ci_low=None, ci_high=None, n_eras=10)
    return MetricScorecard(
        model_id="m", n_eras=10, rank_scalar=0.0, deflated_sharpe=0.0,
        mean_payout=cell(0.0), corr=cell(0.0), mmc=cell(0.0), fnc=0.0,
        corr_sharpe_ac=cell(sharpe_ac), cvar5=0.0, max_drawdown=max_drawdown,
        burn_rate=0.0, mmc_sharpe_ac=0.0, sortino=0.0, calmar=0.0,
        std_corr=0.1, max_burn_streak=0, time_to_recovery=0,
        horizon_stability=None, horizon_reason=None, regime_corr=None,
        regime_reason=None, perturbation=None, max_feature_exposure=0.0,
        bmc=None, bmc_reason=None, cwmm=None, cwmm_reason=None,
        book_correlation=None, metric_timing_seconds=None, eval_total_seconds=0.0,
    )


def _result_with_scorecard(
    run_id: str, sharpe_ac: float, *, max_drawdown: float = 0.1
) -> RunResult:
    result = _result(run_id, sharpe=0.5)
    return RunResult(
        run_id=result.run_id, oof=result.oof, metrics=result.metrics,
        artifact=result.artifact, manifest=result.manifest,
        scorecard=_scorecard(sharpe_ac, max_drawdown=max_drawdown),
    )


def test_promote_if_better_promotes_only_strictly_better(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    registry.promote("a" * 64)

    registry.record(_result_with_scorecard("b" * 64, sharpe_ac=0.9))
    path, promoted = registry.promote_if_better("b" * 64)
    assert promoted is True
    assert json.loads(path.read_text(encoding="utf-8")) == {"run_id": "b" * 64}

    registry.record(_result_with_scorecard("c" * 64, sharpe_ac=0.85))
    _, promoted = registry.promote_if_better("c" * 64)
    assert promoted is False
    assert json.loads(path.read_text(encoding="utf-8")) == {"run_id": "b" * 64}


def test_promote_if_better_direction_lower_is_better_for_drawdown(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.5, max_drawdown=0.2))
    registry.promote("a" * 64)

    # Higher drawdown is WORSE on max_drawdown -> must not promote.
    registry.record(_result_with_scorecard("b" * 64, sharpe_ac=0.5, max_drawdown=0.4))
    _, promoted = registry.promote_if_better("b" * 64, metric="max_drawdown")
    assert promoted is False

    registry.record(_result_with_scorecard("c" * 64, sharpe_ac=0.5, max_drawdown=0.1))
    _, promoted = registry.promote_if_better("c" * 64, metric="max_drawdown")
    assert promoted is True


def test_promote_if_better_legacy_champion_is_displaced(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    registry.record(_result("a" * 64, sharpe=0.9))  # no scorecard
    registry.promote("a" * 64)

    registry.record(_result_with_scorecard("b" * 64, sharpe_ac=0.4))
    _, promoted = registry.promote_if_better("b" * 64)
    assert promoted is True


def test_promote_if_better_refuses_legacy_candidate(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    registry.promote("a" * 64)
    registry.record(_result("b" * 64, sharpe=9.9))  # legacy candidate, no scorecard
    with pytest.raises(ValueError, match="scorecard"):
        registry.promote_if_better("b" * 64)


def test_promote_rejects_non_hex_run_id(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    with pytest.raises(ValueError, match="run_id"):
        registry.promote("../../etc/passwd")


def test_promote_if_better_unknown_metric_raises(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    with pytest.raises(ValueError, match="metric"):
        registry.promote_if_better("a" * 64, metric="nope")


def test_best_validates_metric_name(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    registry.record(_result("a" * 64, sharpe=0.5))
    with pytest.raises(ValueError, match="metric"):
        registry.best("nope")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_registry.py -q`
Expected: FAIL — `promote` no longer accepts `"run-a"` (regex), `promote_if_better` missing.

- [ ] **Step 3: Implement**

In `nmr/registry.py`:

```python
import re

_RUN_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
_SCORECARD_METRIC_FIELDS = (
    "corr_sharpe_ac",
    "rank_scalar",
    "corr",
    "mmc",
    "fnc",
    "deflated_sharpe",
    "std_corr",
    "max_drawdown",
)
# True when a larger value is better for that metric.
_SCORECARD_METRIC_DIRECTION = {
    "corr_sharpe_ac": True,
    "rank_scalar": True,
    "corr": True,
    "mmc": True,
    "fnc": True,
    "deflated_sharpe": True,
    "std_corr": False,
    "max_drawdown": False,
}
_METRIC_SUMMARY_FIELDS = ("mean", "std", "sharpe", "max_drawdown")
```

Rewrite `record` (atomic parquet + scorecard block):

```python
    def record(self, result: RunResult) -> Path:
        logger.info("[record] recording run %s", result.run_id)
        run_dir = self._root / result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        oof_path = run_dir / "oof.parquet"
        tmp_oof = run_dir / f"{oof_path.name}.tmp.{os.getpid()}"
        try:
            result.oof.write_parquet(tmp_oof)
            os.replace(tmp_oof, oof_path)
        finally:
            if tmp_oof.exists():
                tmp_oof.unlink()
        logger.info("[record] OOF written to %s", oof_path)

        scorecard_block = None
        if result.scorecard is not None:
            row = result.scorecard.to_frame().to_dicts()[0]
            scorecard_block = {
                key: value
                for key, value in row.items()
                if not key.startswith(("timing_", "quality_metric"))
            }

        run_payload = {
            "run_id": result.run_id,
            "metrics": dataclasses.asdict(result.metrics),
            "manifest": result.manifest,
            "scorecard": scorecard_block,
            "oof_path": oof_path.name,
            "artifact_path": str(result.artifact.path) if result.artifact else None,
            "artifact_manifest": result.artifact.manifest if result.artifact else None,
        }
        self._atomic_json_write(run_dir / "run.json", run_payload)
        logger.info("[record] run metadata written to %s/run.json", run_dir)
        return run_dir
```

Rewrite `list` (stable sort), `best` (validated, deterministic), `promote` (regex), and add `promote_if_better`:

```python
    def list(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for run_file in self._root.glob("*/run.json"):
            entries.append(json.loads(run_file.read_text(encoding="utf-8")))
        entries.sort(
            key=lambda entry: (
                (self._root / entry["run_id"] / "run.json").stat().st_mtime,
                entry["run_id"],
            ),
            reverse=True,
        )
        return entries

    def best(self, metric: str = "sharpe") -> dict[str, Any] | None:
        if metric not in _METRIC_SUMMARY_FIELDS:
            raise ValueError(
                f"metric={metric!r} not in {sorted(_METRIC_SUMMARY_FIELDS)}"
            )
        runs = self.list()
        if not runs:
            return None
        return max(
            runs,
            key=lambda run: (float(run["metrics"][metric]), run["run_id"]),
        )

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(
                f"run_id={run_id!r} is not a 64-char lowercase hex string"
            )

    def promote(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        logger.info("[promote] promoting run %s to champion", run_id)
        run_json = self._root / run_id / "run.json"
        if not run_json.exists():
            raise FileNotFoundError(f"Run {run_id!r} does not exist in registry")

        champion_path = self._root / "champion.json"
        self._atomic_json_write(champion_path, {"run_id": run_id})
        logger.info("[promote] champion pointer written to %s", champion_path)
        return champion_path

    def promote_if_better(self, run_id: str, metric: str = "corr_sharpe_ac") -> tuple[Path, bool]:
        """Promote ``run_id`` only if its scorecard metric beats the champion's.

        Direction-aware: ``max_drawdown``/``std_corr`` are lower-is-better.
        A scorecard-bearing candidate may displace a scorecard-less champion
        (legacy OOF metrics are in-sample-biased). Legacy candidates (no
        scorecard) are refused — use :meth:`promote` for explicit overrides.
        """
        self._validate_run_id(run_id)
        if metric not in _SCORECARD_METRIC_FIELDS:
            raise ValueError(
                f"metric={metric!r} not in {sorted(_SCORECARD_METRIC_FIELDS)}"
            )
        run_json = self._root / run_id / "run.json"
        if not run_json.exists():
            raise FileNotFoundError(f"Run {run_id!r} does not exist in registry")
        candidate = json.loads(run_json.read_text(encoding="utf-8"))
        candidate_scorecard = candidate.get("scorecard")
        if not candidate_scorecard or metric not in candidate_scorecard:
            raise ValueError(
                f"Run {run_id!r} has no scorecard metric {metric!r}; "
                "legacy runs require manual promote()"
            )

        champion_path = self._root / "champion.json"
        if not champion_path.exists():
            logger.info("[promote_if_better] no champion; promoting %s", run_id)
            self._atomic_json_write(champion_path, {"run_id": run_id})
            return champion_path, True

        champion_id = json.loads(champion_path.read_text(encoding="utf-8")).get("run_id")
        champion_json = self._root / champion_id / "run.json"
        if not champion_json.exists():
            logger.warning(
                "[promote_if_better] champion %s missing; treating as no champion",
                champion_id,
            )
            self._atomic_json_write(champion_path, {"run_id": run_id})
            return champion_path, True

        champion = json.loads(champion_json.read_text(encoding="utf-8"))
        champion_scorecard = champion.get("scorecard")
        if not champion_scorecard or metric not in champion_scorecard:
            logger.info(
                "[promote_if_better] champion %s has no scorecard; promoting on presence",
                champion_id,
            )
            self._atomic_json_write(champion_path, {"run_id": run_id})
            return champion_path, True

        higher_is_better = _SCORECARD_METRIC_DIRECTION[metric]
        candidate_value = float(candidate_scorecard[metric])
        champion_value = float(champion_scorecard[metric])
        if higher_is_better:
            better = candidate_value > champion_value
        else:
            better = candidate_value < champion_value
        if not better:
            logger.info(
                "[promote_if_better] %s (%.6f) not better than champion %s (%.6f) on %s; "
                "keeping champion",
                run_id, candidate_value, champion_id, champion_value, metric,
            )
            return champion_path, False

        logger.info("[promote_if_better] promoting %s over %s on %s", run_id, champion_id, metric)
        self._atomic_json_write(champion_path, {"run_id": run_id})
        return champion_path, True
```

In `train_first_model.py`, replace line 26:

```python
    champion_path, promoted = registry.promote_if_better(result.run_id)
    print(f"promoted:    {promoted} (champion: {champion_path})")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_registry.py -q`
Then full suite: `.\.venv\Scripts\python -m pytest -q` — green.

- [ ] **Step 5: Docs + commit**

`ARCHITECTURE.md` §N registry block: document `promote_if_better` (direction-aware, legacy-champion displacement, legacy-candidate refusal, regex-validated run ids), deterministic `best()`, and the `scorecard` block in `run.json`.

```bash
git add nmr/registry.py train_first_model.py tests/test_registry.py ARCHITECTURE.md
git commit -m "feat: guarded champion promotion and registry hardening (F-003, F-021, F-022, F-013)"
```

---

## Task 5: Honest evaluation — fold-held-out weights, config-driven ensemble, metrics wiring (F-006, F-015)

**Files:**
- Modify: `nmr/config.py` (new `EnsembleConfig` with `method`)
- Modify: `nmr/runner.py` (weights learned on folds `0..K-2`; score on final fold; uniform fallback `n_folds<2`; `ensemble.method` wiring; `evaluation.metrics` wiring + mmc guard)
- Modify: `tests/test_runner.py` (uniform-weights fallback test; metrics wiring test)
- Modify: `ARCHITECTURE.md` §2A/§H/§N — same commit

**Interfaces:**
- Produces: `EnsembleConfig(method: str = "ridge")` validated against `("ridge", "non_negative")`; registered in `_SECTIONS`/`__all__`/`ExperimentConfig`.
- Consumes: Task 2's config/closure plumbing; Task 3's validation stage.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_ensemble_section_validates_method() -> None:
    from nmr.config import EnsembleConfig
    assert EnsembleConfig().method == "ridge"
    assert EnsembleConfig(method="non_negative").method == "non_negative"
    with pytest.raises(ValueError):
        EnsembleConfig(method="svm")
```

Append to `tests/test_runner.py`:

```python
def test_single_fold_falls_back_to_uniform_weights(tmp_path, caplog) -> None:
    cfg = _config(tmp_path)
    single_fold = ExperimentConfig(
        data=cfg.data, split=SplitConfig(scheme="anchor", purge_eras=1, n_folds=1),
        model=cfg.model,
        evaluation=EvalConfig(
            backend="custom", main_target="target", validation_scorecard=False
        ),
        run=cfg.run,
    )
    import logging
    with caplog.at_level(logging.WARNING, logger="nmr.runner"):
        result = ExperimentRunner(single_fold).run(deploy=False)
    assert result.manifest["weights"] == [0.5, 0.5]  # 2 components, uniform
    assert any("uniform" in record.message for record in caplog.records)


def test_mmc_metric_requires_validation_scorecard(tmp_path) -> None:
    cfg = _config(tmp_path)
    bad = ExperimentConfig(
        data=cfg.data, split=cfg.split, model=cfg.model,
        evaluation=EvalConfig(
            backend="custom", main_target="target",
            validation_scorecard=False, metrics=("corr", "mmc", "sharpe"),
        ),
        run=cfg.run,
    )
    import pytest as _pytest
    with _pytest.raises(ValueError, match="mmc"):
        ExperimentRunner(bad).run(deploy=False)
```

(Add `SplitConfig` to the test_runner imports if missing.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_config.py tests/test_runner.py -q`
Expected: FAIL — `ImportError: cannot import name 'EnsembleConfig'`; `AssertionError` on uniform weights; no mmc ValueError.

- [ ] **Step 3: Implement**

**3a. `nmr/config.py`:**

```python
VALID_ENSEMBLE_METHODS = ("ridge", "non_negative")

@dataclass(frozen=True)
class EnsembleConfig:
    """Ensemble weight-learning method (applies to the OOF blend)."""

    method: str = "ridge"

    def __post_init__(self) -> None:
        if self.method not in VALID_ENSEMBLE_METHODS:
            raise ValueError(
                f"ensemble.method={self.method!r} not in {VALID_ENSEMBLE_METHODS}"
            )
```

Add `ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)` to `ExperimentConfig`, `"ensemble": EnsembleConfig` to `_SECTIONS`, `"EnsembleConfig"` to `__all__`.

**3b. `nmr/runner.py`** — replace the weight-learning block in `run()` (lines 91-99):

```python
        folds = splitter.split(train_df.get_column("era").to_list())
        if len(folds) < 2:
            logger.warning(
                "[run] n_folds < 2; falling back to uniform ensemble weights"
            )
            weights = tuple(1.0 / len(pred_cols) for _ in pred_cols)
            weight_learning_eras: list[str] = []
        else:
            weight_learning_eras = [
                era for fold in folds[:-1] for era in fold.val_eras
            ]
            weight_df = joined.filter(pl.col("era").is_in(weight_learning_eras))
            weights = ensembler.learn_weights(
                weight_df.select(["era", *pred_cols, main_target]),
                pred_cols=pred_cols,
                target_col=main_target,
                era_col="era",
                method=self._config.ensemble.method,
            )
        scoring_eras = (
            [era for fold in folds for era in fold.val_eras]
            if len(folds) < 2
            else list(folds[-1].val_eras)
        )
        logger.info("[run] ensemble weights: %s (learned on %d eras, scored on %d)",
                    dict(zip(pred_cols, weights)), len(weight_learning_eras), len(scoring_eras))
```

After the neutralize step, compute metrics on the scoring eras only:

```python
        per_era_all = evaluator.per_era_corr(
            neutralized,
            pred_col="prediction",
            target_col=main_target,
            era_col="era",
        )
        per_era_corr = {
            era: value for era, value in per_era_all.items() if era in set(scoring_eras)
        }
        metrics = evaluator.summarize(per_era_corr)
```

Add metrics wiring right after (still on the full `neutralized` frame):

```python
        summary_metrics: dict[str, float] = {
            "corr": metrics.mean,
            "sharpe": metrics.sharpe,
        }
        if "fnc" in set(self._config.evaluation.metrics):
            fnc_by_era = evaluator.per_era_fnc(
                neutralized.filter(pl.col("era").is_in(set(scoring_eras))),
                pred_col="prediction",
                feature_cols=feature_cols,
                target_col=main_target,
                era_col="era",
            )
            summary_metrics["fnc"] = evaluator.summarize(fnc_by_era).mean
```

(`neutralized` derives from `blended`, which retains the feature columns — the FNC runs on the blended+neutralized frame restricted to scoring eras.)

The MMC guard lives at the **top of `run()`** (before any training — the design requires failing before potentially hours of multi-target CV), not here:

```python
    def run(self, *, deploy: bool = False) -> RunResult:
        requested_metrics = set(self._config.evaluation.metrics)
        if "mmc" in requested_metrics and not self._config.evaluation.validation_scorecard:
            raise ValueError(
                "evaluation.metrics includes 'mmc' but the validation scorecard stage "
                "is disabled (evaluation.validation_scorecard=false). MMC requires the "
                "meta model, which covers validation eras only."
            )
        logger.info("[run] starting experiment run_id=%s", self._run_id)
        ...
```

Update the manifest:

```python
            "weights": list(weights),
            "weight_learning_eras": weight_learning_eras,
            "scoring_eras": scoring_eras,
            "summary_metrics": summary_metrics,
```

**3c. `nmr/research.py` — wire the same strategy surface into HPO (F-006 completeness).** `_held_out_metric` currently hardcodes `method="ridge"` (line ~224) and `proportion=1.0` (line ~270), so the HPO sweep evaluates a *different strategy* than the runner deploys. Replace both with the configured values:

```python
    weights = ensembler.learn_weights(
        joined_train.select(["era", *pred_cols, main_target]),
        pred_cols=pred_cols,
        target_col=main_target,
        era_col="era",
        method=config.ensemble.method,
    )
    ...
    neutralized = NeutralizationEngine(
        max_cache_bytes=config.risk.cache_max_bytes
    ).neutralize(
        blended,
        pred_col="prediction",
        feature_cols=feature_cols,
        era_col="era",
        proportion=config.risk.neutralization_proportion,
    )
```

(`neutralization_frontier` sweeps proportions explicitly — it is intentionally NOT wired to `neutralization_proportion`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_config.py tests/test_runner.py tests/test_registry.py -q`
Then full suite: `.\.venv\Scripts\python -m pytest -q` — green.

- [ ] **Step 5: Docs + commit**

`ARCHITECTURE.md` §2A: add `ensemble.method` row; reword the `evaluation.metrics` row to its real semantics. §H: note method comes from `EnsembleConfig`. §N: weights learned on folds 0..K-2 and scored on the final fold; uniform fallback at `n_folds < 2`; `scoring_eras`/`weight_learning_eras` manifest fields.

```bash
git add nmr/config.py nmr/runner.py tests/test_config.py tests/test_runner.py ARCHITECTURE.md
git commit -m "feat: fold-held-out ensemble weights, config-driven method, metrics wiring (F-006, F-015)"
```

---

## Task 6: Model-layer guards (F-007, F-009, F-014)

**Files:**
- Modify: `nmr/splitter.py` (`purge_eras` property)
- Modify: `nmr/models.py` (null-target filter in `_fit_predict_fold`; `_fit_model` logged narrow-exception fallback + `resolved_device`; purge-width assertion in `_fit_predict_fold`)
- Modify: `tests/test_models.py` (violating-fold test replaces the tautology; null-target fold test; device-resolution test)
- Modify: `ARCHITECTURE.md` §C/§G — same commit

**Interfaces:**
- Produces: `PurgedEraSplitter.purge_eras -> int` property; `ModelOrchestrator.resolved_device: str | None` attribute.
- Consumes: Task 2's `train_full_history`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models.py`:

```python
def test_fit_predict_fold_rejects_zero_purge_gap() -> None:
    from nmr.splitter import Fold

    df = _model_frame(n_eras=8)
    orchestrator = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=3,
    )
    violating = Fold(
        index=0,
        train_eras=tuple(str(e) for e in range(1, 5)),
        val_eras=tuple(str(e) for e in range(5, 7)),  # gap = 1 <= purge_eras=1
    )
    with pytest.raises(ValueError, match="purge"):
        orchestrator._fit_predict_fold(
            df, fold=violating, feature_cols=["f1", "f2", "f3"],
            target_col="target", era_col="era", purge_eras=1,
        )


def test_fit_predict_fold_drops_null_target_rows() -> None:
    df = _model_frame(n_eras=6).with_columns(
        pl.when(pl.col("id") == "1_0")
        .then(None)
        .otherwise(pl.col("target"))
        .alias("target")
    )
    orchestrator = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=3,
    )
    model, prediction = orchestrator.train_anchor_fold(
        df,
        feature_cols=["f1", "f2", "f3"],
        target_col="target",
        splitter=_anchor_splitter(),
    )
    assert model is not None
    assert prediction.height > 0


def test_fit_model_records_resolved_device() -> None:
    orchestrator = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=3,
    )
    df = _model_frame(n_eras=4)
    orchestrator.train_full_history(df, feature_cols=["f1", "f2", "f3"], target_col="target")
    assert orchestrator.resolved_device == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_models.py -q`
Expected: FAIL — `TypeError: _fit_predict_fold() got an unexpected keyword argument 'purge_eras'`; `AttributeError: resolved_device`.

- [ ] **Step 3: Implement**

**3a. `nmr/splitter.py`** — add a public property:

```python
    @property
    def purge_eras(self) -> int:
        return self._split.purge_eras
```

**3b. `nmr/models.py`:**

- `__init__`: add `self.resolved_device: str | None = None`.
- `_fit_predict_fold` signature gains `purge_eras: int`; assert the purge width:

```python
    def _fit_predict_fold(
        self,
        df: pl.DataFrame,
        *,
        fold: Fold,
        feature_cols: Sequence[str],
        target_col: str,
        era_col: str,
        purge_eras: int,
    ) -> tuple[object, pl.DataFrame]:
        self._assert_fold_is_leakage_safe(fold, purge_eras=purge_eras)
        train_df = df.filter(pl.col(era_col).is_in(fold.train_eras))
        val_df = df.filter(pl.col(era_col).is_in(fold.val_eras))
        if train_df.is_empty() or val_df.is_empty():
            raise ValueError(f"Degenerate training slice for fold {fold.index}")

        train_df = train_df.filter(
            pl.col(target_col).is_not_null() & pl.col(target_col).is_finite()
        )
        dropped = df.filter(pl.col(era_col).is_in(fold.train_eras)).height - train_df.height
        if dropped:
            logger.warning(
                "[_fit_predict_fold] %s fold %d: dropped %d rows with null/non-finite targets",
                target_col, fold.index, dropped,
            )
        if train_df.is_empty():
            raise ValueError(f"No usable training rows for fold {fold.index} after null filtering")
        ...
```

- Update `_assert_fold_is_leakage_safe` to be purge-aware (replace the tautology):

```python
    def _assert_fold_is_leakage_safe(self, fold: Fold, *, purge_eras: int) -> None:
        train_eras = {int(era) for era in fold.train_eras}
        val_eras = {int(era) for era in fold.val_eras}
        if train_eras & val_eras:
            raise ValueError(f"Fold {fold.index} reuses eras across train/val")
        if not train_eras or not val_eras:
            raise ValueError(f"Fold {fold.index} is degenerate")

        train_max = max(train_eras)
        val_min = min(val_eras)
        if train_max >= val_min:
            raise ValueError(f"Fold {fold.index} is not strictly time-ordered")
        if val_min - train_max <= purge_eras:
            raise ValueError(
                f"Fold {fold.index} violates purge invariant: gap "
                f"{val_min - train_max} <= purge_eras={purge_eras}"
            )
```

- Update the two call sites of `_fit_predict_fold`: `train_cross_validation` and `train_anchor_fold` pass `purge_eras=splitter.purge_eras`.
- `_fit_model` — narrow exceptions (single clause; `lightgbm`/`xgboost` are already top-level imports in this module) and record the resolved device:

```python
    def _fit_model(
        self, *, features: pd.DataFrame, target: np.ndarray, use_gpu: bool = True
    ) -> object:
        candidate_params = self._device_candidate_params(use_gpu=use_gpu)
        last_error: Exception | None = None
        backend_errors = (
            (ValueError, TypeError, lgb.basic.LightGBMError)
            if self._config.backend == "lightgbm"
            else (ValueError, TypeError, xgb.core.XGBoostError)
        )

        for params in candidate_params:
            model = self._build_model(params)
            try:
                model.fit(features, target)
            except backend_errors as exc:
                logger.warning(
                    "[fit] %s fit failed (%s: %s); trying next candidate",
                    self._config.backend, type(exc).__name__, exc,
                )
                last_error = exc
                continue
            self.resolved_device = (
                "gpu"
                if params.get("device_type") == "gpu"
                or params.get("tree_method") == "gpu_hist"
                else "cpu"
            )
            return model

        assert last_error is not None
        raise last_error
```

- `_device_candidate_params(use_gpu: bool)`:

```python
    def _device_candidate_params(self, *, use_gpu: bool) -> list[dict[str, Any]]:
        if not use_gpu:
            return [self._resolved_params(use_gpu=False)]
        cpu_params = self._resolved_params(use_gpu=False)
        gpu_params = self._resolved_params(use_gpu=True)
        if gpu_params == cpu_params:
            return [cpu_params]
        return [gpu_params, cpu_params]
```

- Update `train_full_history` (Task 2) to call `self._fit_model(..., use_gpu=False)` — adjust its current call.
- Wire the resolved OOF device into the run manifest: in `nmr/runner.py`, after `run()`'s training completes (the OOF CV models), add `"oof_device": model_orchestrator.resolved_device` to the manifest dict (Task 2 already records `"pipeline_device": "cpu"` for the full-history models — this is the separate OOF-CV device, per the design's stated intent).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_models.py tests/test_splitter.py tests/test_runner.py -q`
Then full suite: `.\.venv\Scripts\python -m pytest -q` — green.

- [ ] **Step 5: Docs + commit**

`ARCHITECTURE.md` §C: `purge_eras` property. §G: `_assert_fold_is_leakage_safe` now enforces the purge gap at train time; `_fit_predict_fold` filters null targets; GPU→CPU fallback is logged with the exception type; `resolved_device` recorded.

```bash
git add nmr/splitter.py nmr/models.py tests/test_models.py ARCHITECTURE.md
git commit -m "fix: real purge-width assertion, null-target guards, logged device fallback (F-007, F-009, F-014)"
```

---

## Task 7: Risk cache budget, corruption resilience, zero-variance contract (F-008, F-011, F-012)

**Files:**
- Modify: `nmr/risk.py` (cache I/O hardening; `max_cache_bytes` + LRU eviction + `os.utime` on hit; cache-size log at init; zero-variance log already added in Task 2)
- Modify: `nmr/config.py` (wire `risk.cache_max_bytes` into the engine)
- Modify: `tests/test_risk.py` (eviction, corruption, zero-variance era preserved)
- Modify: `ARCHITECTURE.md` §2F — same commit

**Interfaces:**
- Produces: `NeutralizationEngine(cache_dir=None, max_cache_bytes: int | None = None)`; `DEFAULT_CACHE_MAX_BYTES = 2 * 2**30`; `cache_size_bytes() -> int`.
- Consumes: Task 2's `neutralize_array` + `risk` config section.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_risk.py`:

```python
def test_cache_corruption_recomputes(tmp_path) -> None:
    df = _risk_frame().filter(pl.col("era") == "1")
    engine = NeutralizationEngine(cache_dir=tmp_path)
    engine.neutralize(df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0)
    npy_files = list(tmp_path.glob("*.npy"))
    assert len(npy_files) == 1
    npy_files[0].write_bytes(b"\x00" * 16)  # truncate/corrupt
    result = engine.neutralize(
        df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0
    )
    assert np.all(np.isfinite(result.get_column("pred").to_numpy()))


def test_cache_eviction_respects_budget(tmp_path) -> None:
    """Two eras' cache entries; the budget fits only one -> the older is evicted."""
    full_df = _risk_frame()  # eras "1" and "2"
    engine = NeutralizationEngine(cache_dir=tmp_path, max_cache_bytes=900)
    engine.neutralize(
        full_df, pred_col="pred", feature_cols=["f1", "f2"], proportion=1.0
    )
    assert len(list(tmp_path.glob("*.npy"))) == 2  # one per era, pre-eviction
    assert engine.cache_size_bytes() <= 900
    survivors = [p.name for p in tmp_path.glob("*.npy")]
    assert len(survivors) == 1
    assert "era_2_" in survivors[0]  # mtime-oldest (era 1) evicted, era 2 survives


def test_zero_variance_era_keeps_rows_and_is_logged(tmp_path, caplog) -> None:
    df = pl.DataFrame(
        {
            "era": ["1", "1", "2", "2"],
            "id": ["a", "b", "c", "d"],
            "pred": [0.5, 0.5, 0.1, 0.9],
            "f1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    import logging
    engine = NeutralizationEngine(cache_dir=tmp_path)
    with caplog.at_level(logging.WARNING, logger="nmr.risk"):
        result = engine.neutralize(df, pred_col="pred", feature_cols=["f1"], proportion=1.0)
    assert result.height == 4
    era1 = result.filter(pl.col("era") == "1").get_column("pred").to_numpy()
    assert np.array_equal(era1, np.array([0.5, 0.5]))
    assert any("zero-variance" in record.message for record in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_risk.py -q`
Expected: FAIL — `TypeError: NeutralizationEngine.__init__() got an unexpected keyword argument 'max_cache_bytes'`; corruption test raises `EOFError` from `np.load`.

- [ ] **Step 3: Implement**

In `nmr/risk.py`:

```python
DEFAULT_CACHE_MAX_BYTES = 2 * 2**30  # 2 GiB
```

`__init__`:

```python
    def __init__(
        self, *, cache_dir: Path | None = None, max_cache_bytes: int | None = None
    ) -> None:
        self._cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else REPO_ROOT / "artifacts" / "cache" / "neutralization"
        )
        self._max_cache_bytes = (
            DEFAULT_CACHE_MAX_BYTES
            if max_cache_bytes is None
            else int(max_cache_bytes)
        )
        if self._max_cache_bytes < 0:
            raise ValueError("max_cache_bytes must be >= 0")
        total = self.cache_size_bytes()
        logger.info(
            "[neutralization] cache dir=%s max_bytes=%d current_bytes=%d",
            self._cache_dir, self._max_cache_bytes, total,
        )
```

`cache_size_bytes` + eviction + atomic store + hardened load + LRU touch:

```python
    def cache_size_bytes(self) -> int:
        if not self._cache_dir.exists():
            return 0
        return sum(
            path.stat().st_size for path in self._cache_dir.iterdir() if path.is_file()
        )

    def _evict_to_budget(self) -> None:
        if not self._cache_dir.exists():
            return
        files = sorted(
            (p for p in self._cache_dir.iterdir() if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        total = sum(p.stat().st_size for p in files)
        for path in files:
            if total <= self._max_cache_bytes:
                break
            try:
                size = path.stat().st_size
                path.unlink()
                total -= size
            except OSError:
                continue
        if total > self._max_cache_bytes:
            logger.warning(
                "[neutralization] cache still above budget (%d bytes); "
                "raise risk.cache_max_bytes or clear artifacts/cache/neutralization",
                total,
            )
```

`_load_cached_array`: broaden the exception set and touch mtime on hit:

```python
        try:
            array = np.load(array_path)
        except (OSError, ValueError, EOFError):
            return None
        try:
            os.utime(array_path)
            os.utime(metadata_path)
        except OSError:
            pass
        return np.asarray(array, dtype=float)
```

(Add `import os` to risk.py.) `_store_cached_array` — atomic (temp + `os.replace` for the array; the shared helper for metadata), then evict:

```python
    def _store_cached_array(
        self,
        array_path: Path,
        metadata_path: Path,
        *,
        metadata: dict[str, object],
        array: np.ndarray,
    ) -> None:
        from nmr._atomicio import atomic_write_text

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_array = array_path.with_name(f"{array_path.name}.tmp.{os.getpid()}")
        try:
            np.save(tmp_array, np.asarray(array, dtype=float))
            os.replace(tmp_array, array_path)
            atomic_write_text(
                metadata_path,
                json.dumps(metadata, sort_keys=True, indent=2),
            )
        finally:
            if tmp_array.exists():
                tmp_array.unlink()
        self._evict_to_budget()
```

(No fsync on the `.npy` temp — cache corruption self-heals by recompute; the atomicity contract is the temp + `os.replace` pattern. Cache entries are keyed by content hash, so a torn file can never be mistaken for another era's entry.)

In `nmr/config.py` — wire `cache_max_bytes` where the engine is constructed in the runner (`nmr/runner.py`):

```python
        neutralized = NeutralizationEngine(
            max_cache_bytes=self._config.risk.cache_max_bytes
        ).neutralize(...)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_risk.py tests/test_runner.py -q`
Then full suite: `.\.venv\Scripts\python -m pytest -q` — green.

- [ ] **Step 5: Docs + commit**

`ARCHITECTURE.md` §2F: cache key + write-age-LRU budget (`DEFAULT_CACHE_MAX_BYTES`), corruption→recompute, zero-variance eras return predictions unchanged (logged) — replaces the old "all values finite" phrasing where it conflicts.

```bash
git add nmr/risk.py nmr/config.py nmr/runner.py tests/test_risk.py ARCHITECTURE.md
git commit -m "feat: neutralization cache budget, corruption resilience, zero-variance contract (F-008, F-011, F-012)"
```

---

## Task 8: Vectorized per-era engines + public helpers (F-010, F-027, risk-loop)

**Files:**
- Modify: `nmr/evaluation.py` (partition-once `_per_era_metric`/`per_era_bmc`/`per_era_cwmm`/`_resolve_overlap_eras`; promote `sorted_era_labels`/`clean_frame` to module-level public)
- Modify: `nmr/robustness.py` (use the public helpers)
- Modify: `nmr/risk.py` (`neutralize` loop via `partition_by`)
- Modify: `nmr/__init__.py` (export `sorted_era_labels`, `clean_frame`)
- Modify: `tests/test_evaluation.py` (public helpers; appearance-order≠numeric-order era case)
- Modify: `ARCHITECTURE.md` §E — same commit

**Interfaces:**
- Produces: module-level `sorted_era_labels(labels: Sequence[str]) -> list[str]` and `clean_frame(df: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame` in `nmr/evaluation.py`, added to `__all__` and re-exported from `nmr/__init__.py`.
- Consumes: nothing new — outputs must be byte-identical to the pre-change contract (full suite is the oracle).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluation.py`:

```python
from nmr.evaluation import clean_frame, sorted_era_labels


def test_sorted_era_labels_sorts_numerically() -> None:
    assert sorted_era_labels(["10", "2", "1"]) == ["1", "2", "10"]
    with pytest.raises(ValueError):
        sorted_era_labels(["a", "b"])


def test_clean_frame_drops_nulls_and_nonfinite() -> None:
    df = pl.DataFrame(
        {"era": ["1", "1", "1"], "pred": [0.1, None, float("inf")], "target": [1.0, 2.0, 3.0]}
    )
    out = clean_frame(df, ["pred", "target"])
    assert out.to_dicts() == [{"era": "1", "pred": 0.1, "target": 1.0}]


def test_per_era_metric_handles_appearance_order_eras() -> None:
    df = pl.DataFrame(
        {
            "era": ["5", "5", "2", "2"],  # appearance order != numeric order
            "pred": [1.0, 2.0, 3.0, 4.0],
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )
    engine = EvaluationEngine("custom")
    out = engine.per_era_corr(df, pred_col="pred", target_col="target")
    assert list(out.keys()) == ["2", "5"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_evaluation.py -q`
Expected: FAIL — `ImportError: cannot import name 'clean_frame'` / `sorted_era_labels`.

- [ ] **Step 3: Implement**

**3a. `nmr/evaluation.py`** — add module-level functions (move the bodies of the methods):

```python
def sorted_era_labels(labels: Sequence[str]) -> list[str]:
    numeric_to_label: dict[int, str] = {}
    for label in labels:
        try:
            numeric_label = int(label)
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric era label {label!r}; evaluation requires chronological eras"
            ) from exc
        numeric_to_label.setdefault(numeric_label, label)
    return [numeric_to_label[num] for num in sorted(numeric_to_label)]


def clean_frame(df: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    clean_df = df.select(list(columns)).drop_nulls()
    if clean_df.is_empty():
        return clean_df
    mask = np.ones(clean_df.height, dtype=bool)
    for col in columns:
        values = clean_df.get_column(col).to_numpy()
        if np.issubdtype(values.dtype, np.number):
            mask &= np.isfinite(values)
    if mask.all():
        return clean_df
    return clean_df.filter(pl.Series("mask", mask))
```

Add both to `__all__`. Delete the methods `_sorted_labels` and `_clean_frame` and update **all** internal call sites (grep `_sorted_labels` / `_clean_frame` in `evaluation.py` and `robustness.py`; the engine's `_per_era_metric`, `summarize`, `per_era_bmc`, `per_era_cwmm`, `_resolve_overlap_eras`, and `robustness.py:320-327` all switch to the module-level names).

Rewrite `_per_era_metric` (partition-once):

```python
    def _per_era_metric(
        self,
        df: pl.DataFrame,
        *,
        era_col: str,
        required_cols: Sequence[str],
        score_fn,
    ) -> dict[str, float]:
        eras = sorted_era_labels(df.get_column(era_col).to_list())
        parts_by_era = {
            str(part.get_column(era_col).to_list()[0]): part
            for part in df.partition_by(era_col, maintain_order=True)
        }
        scores: dict[str, float] = {}
        for era in eras:
            clean_df = clean_frame(parts_by_era[era], required_cols)
            scores[era] = self._normalize_score(score_fn(clean_df))
        return scores
```

Rewrite `per_era_bmc` / `per_era_cwmm` loops to iterate `parts_by_era[era]` from a single `partition_by` (same pattern), and `_resolve_overlap_eras` to partition once and re-sort numerically:

```python
    def _resolve_overlap_eras(
        self,
        df: pl.DataFrame,
        *,
        era_col: str,
        coverage_col: str,
        min_overlap_eras: int,
    ) -> list[str]:
        if min_overlap_eras < 1:
            raise ValueError("min_overlap_eras must be >= 1")
        eras = sorted_era_labels(df.get_column(era_col).to_list())
        parts_by_era = {
            str(part.get_column(era_col).to_list()[0]): part
            for part in df.partition_by(era_col, maintain_order=True)
        }
        overlap_eras: list[str] = []
        for era in eras:
            era_values = (
                parts_by_era[era]
                .select(pl.col(coverage_col).cast(pl.Float64, strict=False))
                .drop_nulls()
                .get_column(coverage_col)
                .to_numpy()
            )
            if era_values.size == 0:
                continue
            if np.isfinite(era_values).any():
                overlap_eras.append(era)
        if len(overlap_eras) < min_overlap_eras:
            raise NonVacuityError(
                "Non-vacuity violation: intersection yielded only "
                f"{len(overlap_eras)} eras; minimum required {min_overlap_eras}."
            )
        return overlap_eras
```

(`partition_by` yields appearance order; iterating `sorted_era_labels` order and keying parts by era keeps output order identical to before.)

**3b. `nmr/robustness.py`** — replace `engine._sorted_labels(...)` and `engine._clean_frame(...)` with the module-level imports:

```python
from nmr.evaluation import clean_frame, sorted_era_labels
```

and update the four call sites (lines ~320-327).

**3c. `nmr/risk.py`** — convert the era loop in `neutralize` to `partition_by`:

```python
        parts: list[pl.DataFrame] = []
        era_parts = work_df.partition_by(era_col, maintain_order=True)
        total = len(era_parts)
        for idx, era_df in enumerate(era_parts, start=1):
            era_label = str(era_df.get_column(era_col).to_list()[0])
            if idx == 1 or idx == total or idx % 50 == 0:
                logger.info("[neutralize] era %d/%d: %s", idx, total, era_label)
            neutralized = self._neutralize_era(
                era_df,
                era_label=era_label,
                pred_col=pred_col,
                feature_cols=feature_list,
                proportion=proportion,
            )
            parts.append(
                era_df.with_columns(pl.Series(name=pred_col, values=neutralized))
            )
```

(Delete the now-unused `eras = work_df.get_column(...)...unique(...)` line.)

**3d. `nmr/__init__.py`** — add `clean_frame, sorted_era_labels` to the evaluation import list and to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_evaluation.py tests/test_risk.py tests/test_robustness.py tests/test_scorecard.py tests/test_benchmark_slice3.py -q`
Then full suite: `.\.venv\Scripts\python -m pytest -q` — green (this is the vectorization oracle gate).

- [ ] **Step 5: Docs + commit**

`ARCHITECTURE.md` §E: engines are partition-once; `sorted_era_labels`/`clean_frame` are public module-level helpers (no private cross-module access).

```bash
git add nmr/evaluation.py nmr/robustness.py nmr/risk.py nmr/__init__.py tests/test_evaluation.py ARCHITECTURE.md
git commit -m "perf: partition-once per-era engines and public evaluation helpers (F-010, F-027)"
```

---

## Task 9: Benchmark module polish (F-017, F-024, F-028)

**Files:**
- Modify: `nmr/benchmark.py` (module logger; walk-forward INFO progress; remove sklearn fallback; WARNING on id-column inference)
- Modify: `ARCHITECTURE.md` §M — same commit

**Interfaces:**
- Consumes: Task 1's green state. Produces: module logger `logging.getLogger("nmr.benchmark")`; `_build_classical_model("tree")` raises if lightgbm is unavailable.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_benchmark_baselines.py`:

```python
def test_walk_forward_uses_lightgbm_tree_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    suite = _suite(seed=7)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("simulated missing lightgbm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        suite._build_classical_model("tree")
```

Add `import pytest` to that file. (Run this test alone — it patches `builtins.__import__` globally and must not run concurrently with other imports.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_benchmark_baselines.py::test_walk_forward_uses_lightgbm_tree_without_fallback -q`
Expected: FAIL — the sklearn `GradientBoostingRegressor` fallback is returned instead of raising.

- [ ] **Step 3: Implement**

In `nmr/benchmark.py`:

```python
logger = logging.getLogger("nmr.benchmark")
```

(add `import logging` if absent — verify with grep; the audit says the module has no logging import).

- `_build_classical_model`, tree branch (lines 487-505): remove the `except ImportError` fallback:

```python
        if name == "tree":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.1,
                random_state=self._eval_cfg.seed,
                n_jobs=1,
                verbose=-1,
            )
```

- `_walk_forward_model_predictions` (lines 455-481): add progress logging inside the loop:

```python
        for idx in range(min_train_eras, len(eras)):
            logger.info(
                "[walk_forward] %s baseline: era %d/%d (train %d eras -> predict %s)",
                model_name, idx - min_train_eras + 1, len(eras) - min_train_eras,
                idx, eras[idx],
            )
```

- `_infer_id_column` (lines 937-956): log the fallback inference:

```python
    for alias in ("id", "index", "unnamed: 0", "column_1", ""):
        if alias in normalized:
            return normalized[alias]
    logger.warning(
        "[tutorial] no known id alias in columns %r; inferring first non-metric column %r",
        columns, non_metric[0],
    )
    return non_metric[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_benchmark_baselines.py tests/test_benchmark_slice1.py tests/test_benchmark_slice2.py -q`
Then full suite: `.\.venv\Scripts\python -m pytest -q` — green.

- [ ] **Step 5: Docs + commit**

`ARCHITECTURE.md` §M: tree baseline is lightgbm-only (no sklearn fallback); walk-forward logs progress; id inference logs at WARNING.

```bash
git add nmr/benchmark.py tests/test_benchmark_baselines.py ARCHITECTURE.md
git commit -m "fix: fail loudly on missing lightgbm, add benchmark module logging (F-017, F-024, F-028)"
```

---

## Task 10: CI workflow + script contract tests (F-018)

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `tests/test_scripts.py`
- Modify: `CONTRIBUTING.md` (CI reference) — same commit

**Interfaces:**
- Consumes: Task 1's public `iter_baseline_predictions` (the script contract test depends on it).

- [ ] **Step 1: Write the failing test**

Create `tests/test_scripts.py`:

```python
"""Contract tests for control-plane scripts (F-018)."""

from __future__ import annotations

import pandas as pd
import polars as pl

import benchmark_runner
import generate_dashboard
import train_first_model  # noqa: F401  (import-time smoke)


class _StubSuite:
    """Public-surface stub: only iter_baseline_predictions exists."""

    def __init__(self) -> None:
        self.frame = pl.DataFrame(
            {"era": ["1", "1"], "id": ["a", "b"], "prediction": [0.1, 0.2]}
        )

    def iter_baseline_predictions(self, *, include_classical, min_train_eras):
        yield ("constant-0.5", "null", self.frame, 77)
        if include_classical:
            yield ("linear", "classical", self.frame, 81)


def test_candidate_strategies_consumes_only_public_api() -> None:
    suite = _StubSuite()
    benchmarks = pl.DataFrame(
        {"era": ["1", "1"], "id": ["a", "b"], "bench_a": [0.3, 0.4]}
    )
    contexts = list(
        benchmark_runner._candidate_strategies(suite, benchmarks, seed=77, min_train_eras=2, fast_mode=False)
    )
    assert [ctx.model_id for ctx in contexts] == ["constant-0.5", "linear", "bench_a"]
    assert contexts[0].seed == 77
    assert contexts[1].seed == 81


def test_dashboard_escapes_html_interpolation() -> None:
    df = pd.DataFrame(
        [
            {
                "model_id": "<script>alert(1)</script>",
                "source": "trained",
                "run_name": '"><img src=x onerror=alert(2)>',
                "feature_set": "small",
                "backend": "lightgbm",
                "preset": "fast",
                "n_targets": 1,
                "targets": "target",
                "mean": 0.1,
                "std": 0.2,
                "sharpe": 0.5,
                "max_drawdown": 0.05,
                "rank": 1,
            }
        ]
    )
    html = generate_dashboard._build_html(df, benchmark_path=__import__("pathlib").Path("benchmark_scores.csv"), registry_dir=__import__("pathlib").Path("registry"))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_dashboard_ranks_trained_and_benchmark_on_same_sharpe() -> None:
    trained = pd.DataFrame(
        [
            {
                "model_id": "trained_a", "source": "trained", "run_name": "t",
                "feature_set": "small", "backend": "lgbm", "preset": "fast",
                "n_targets": 1, "targets": "target", "mean": 0.1, "std": 0.1,
                "sharpe": 1.5, "max_drawdown": 0.1, "rank": 0,
            }
        ]
    )
    benchmark = pd.DataFrame(
        [
            {
                "model_id": "bench_a", "source": "benchmark", "run_name": "b",
                "feature_set": "all", "backend": "benchmark", "preset": "benchmark",
                "n_targets": 1, "targets": "target", "mean": 0.05, "std": 0.1,
                "sharpe": 0.5, "max_drawdown": 0.2, "rank": 0,
            }
        ]
    )
    ranked = generate_dashboard._rank_models(pd.concat([trained, benchmark], ignore_index=True))
    assert ranked.iloc[0]["model_id"] == "trained_a"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_scripts.py -q`
Expected: FAIL — `_build_html` does not escape (the `<script>` string is present).

- [ ] **Step 3: Implement**

**3a. `generate_dashboard.py`** — Task 3 already added `html.escape`; if the Task 3 dashboard changes landed, this test passes already. Verify; if not, add the escaping now (wrap every `{...}` interpolation in `_build_html` with `html.escape(str(...))` and add `import html`).

**3b. Create `.github/workflows/ci.yml`:**

```yaml
name: ci

on:
  push:
    branches: [main, sanity-check]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run test suite
        run: python -m pytest -q
```

(Real-data tests self-skip without `data/v5.2/`, so CI is green without dataset assets — verified guards at tests/test_data.py, tests/test_parity.py, tests/test_scorecard.py, tests/test_robustness.py, tests/test_contribution.py.)

**3c. `CONTRIBUTING.md`** — add a line under the verification section: "CI (`.github/workflows/ci.yml`) runs `pytest -q` on Python 3.12 for every push/PR; real-data tests self-skip without `data/v5.2/`."

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_scripts.py -q`
Then full suite: `.\.venv\Scripts\python -m pytest -q` — green.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml tests/test_scripts.py CONTRIBUTING.md generate_dashboard.py
git commit -m "ci: add pytest workflow and script contract tests (F-018)"
```

---

## Task 11: P3 quick wins (F-020, F-025)

**Files:**
- Modify: `nmr/config.py` (drop runtime `PYTHONHASHSEED`; docstring)
- Modify: `requirements.txt` (remove `python-dotenv`)
- Modify: `tests/test_config.py` (env untouched assertion)
- Modify: `AGENTS.md` §9 (dotenv wording) — same commit

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_set_global_seeds_does_not_touch_hash_env() -> None:
    import os
    os.environ.pop("PYTHONHASHSEED", None)
    set_global_seeds(42)
    assert "PYTHONHASHSEED" not in os.environ
```

(Add `set_global_seeds` to the test_config imports.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_config.py::test_set_global_seeds_does_not_touch_hash_env -q`
Expected: FAIL — `PYTHONHASHSEED` is set by `set_global_seeds`.

- [ ] **Step 3: Implement**

In `nmr/config.py`, `set_global_seeds`:

```python
def set_global_seeds(seed: int) -> None:
    """Seed Python and NumPy for reproducible runs.

    Note: ``PYTHONHASHSEED`` is NOT set here — CPython fixes hash randomization
    at interpreter startup, so a runtime assignment affects only subprocesses
    (none are spawned). Model backends (LightGBM/XGBoost) receive their seed via
    model params, not here.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is a hard dependency, but stay resilient.
        pass
```

In `requirements.txt`, remove the `python-dotenv` line.

In `AGENTS.md` §9, replace the credential sentence with: "Numerai API credentials are used only in notebooks via `numerapi`; never hardcode or print them. `.env` is git-ignored and is never read by `nmr/`."

In `README.md:81`, replace the dotenv sentence with: "Numerai API credentials (for `numerapi` download/upload) are used only in notebooks and loaded from a git-ignored `.env`; no credentials are needed to run tests or train on already-downloaded data."

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_config.py -q`
Then full suite: `.\.venv\Scripts\python -m pytest -q` — green.

- [ ] **Step 5: Commit**

```bash
git add nmr/config.py requirements.txt tests/test_config.py AGENTS.md README.md
git commit -m "fix: drop inert PYTHONHASHSEED assignment and unused dotenv dep (F-020, F-025)"
```

---

## Task 12: Final docs re-sync, exports, test count, verification (SSOT sweep)

**Files:**
- Modify: `nmr/__init__.py` (verify export sweep — `sorted_era_labels`, `clean_frame` landed in Task 8; check nothing else new needs exporting)
- Modify: `AGENTS.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `README.md` (final sweep)
- Modify: `configs/example.yaml` (add `risk`/`ensemble`/`validation_scorecard`)
- Regenerate: `artifacts/benchmark_scores_smoke.csv`, `artifacts/benchmark_scores.csv`, `artifacts/dashboard.html` (fast-mode at minimum)

**Interfaces:** none new — documentation and verification only.

- [ ] **Step 1: Export sweep check**

Run: `.\.venv\Scripts\python -c "import nmr; print([n for n in ('sorted_era_labels','clean_frame','iter_baseline_predictions','train_full_history','promote_if_better') if n in nmr.__all__])"`
Expected: `['sorted_era_labels', 'clean_frame']` — the methods need no export; if any missing, add to `nmr/__init__.py` imports + `__all__`.

- [ ] **Step 2: Update configs/example.yaml**

Merge the new fields into the EXISTING sections — the file already has `evaluation:`; appending a second `evaluation:` block would silently shadow the first (`yaml.safe_load` keeps only the last duplicate key):

```yaml
# existing evaluation section, add one field:
evaluation:
  backend: custom           # custom (fast) | official (numerai_tools oracle)
  main_target: target
  metrics: [corr, mmc, fnc, sharpe]
  validation_scorecard: true
```

and append the two NEW sections after it:

```yaml
ensemble:
  method: ridge              # ridge | non_negative

risk:
  neutralization_proportion: 1.0   # float in [0, 1]
  cache_max_bytes: null             # null -> 2 GiB default
```

(Add a comment: `metrics` on train OOF supports `corr`/`fnc`/`sharpe`; `mmc` requires the validation scorecard stage.)

- [ ] **Step 3: AGENTS.md sweep**

- §1: "pytest is the sole automated gate" → "pytest, enforced by CI (`.github/workflows/ci.yml`)".
- §2.4: keep the statement — the purge assertion in `_assert_fold_is_leakage_safe` is now real (verify the sentence still reads true; adjust if it referenced the tautology).
- §7: add a line noting the CI workflow runs the same fast gate.
- §8: GPU hazard — reword to "GPU-first with CPU fallback; the fallback is logged with the exception type and the resolved device is recorded in the run manifest".
- §9: atomicity bullet already updated in Task 2; verify it enumerates registry JSON + artifact payload/manifest + OOF parquet + neutralization-cache pair.
- §8: add the deployment-closure hazard from Task 2 if not present.
- Test count: replace "203 tests" with the actual count from Step 5.

- [ ] **Step 4: ARCHITECTURE.md sweep**

Verify every §2A–§2O subsection matches the code after Tasks 1-11. Specifically: §2A config table (add `ensemble`, `risk` rows; `validation_scorecard`; metrics semantics), §2D (`neutralize_array`), §2E (partition-once + public helpers), §2F (NaN contract, cache budget + LRU), §2G (`train_full_history`, CPU-only, purge assertion, `resolved_device`), §2H (`EnsembleConfig.method`), §2M (generator + no sklearn fallback + logging), §2N (validation stage, closure, `RunResult.scorecard`/`validation_predictions`, `promote_if_better`, run.json `scorecard` block), §2O (dashboard unified). Fix any stale wording found.

- [ ] **Step 5: README.md + CONTRIBUTING.md + test count**

- README: update the dashboard description and the config surface mention; replace "203 tests" with the actual count.
- CONTRIBUTING: verify the CI line from Task 10; replace "203 tests" with the actual count.
- Get the count: `.\.venv\Scripts\python -m pytest --collect-only -q 2>&1 | tail -n 1` (Git Bash; in PowerShell use `| Select-Object -Last 1`). The last line reports `N tests collected` — use that number in all three files (edit once, then confirm `grep -rn "203" AGENTS.md README.md CONTRIBUTING.md` finds no stale count).

- [ ] **Step 6: Verification gates**

Run each and record results truthfully:

```bash
.\.venv\Scripts\python -m pytest -q                                        # full suite
.\.venv\Scripts\python benchmark_runner.py --fast-mode                     # real-data smoke; regenerates artifacts/benchmark_scores_smoke.csv
.\.venv\Scripts\python benchmark_runner.py --fast-mode --output artifacts/benchmark_scores.csv   # regenerate primary CSV under new exposure definition (fast-mode minimum; linear/tree rows documented absent)
.\.venv\Scripts\python generate_dashboard.py                               # regenerates artifacts/dashboard.html
```

Optional (only if wall-clock budget allows; walk-forward per-era training is long): a full non-fast run `.\.venv\Scripts\python benchmark_runner.py` to backfill `linear`/`tree` rows — if skipped, note it explicitly as a follow-up.
Optional: a real-data runner round-trip `.\train_first_model.py` (fast preset) to exercise validation stage + deploy + `load_predict` — time-boxed; if skipped, note it.

- [ ] **Step 7: Commit**

```bash
git add nmr/__init__.py configs/example.yaml AGENTS.md ARCHITECTURE.md README.md CONTRIBUTING.md
git commit -m "docs: SSOT re-sync and final verification sweep (F-013/14/15/25 secondary, counts)"
```

---

## Final review checklist

- [ ] All 28 findings covered (ledger in the design spec §7; no orphans).
- [ ] `RunResult.oof` keeps the full stacked OOF; metrics narrow to final-fold eras (`scoring_eras` in manifest).
- [ ] Single weight set: learned on folds 0..K-2, used by OOF scoring, validation stage, and closure.
- [ ] Deploy fidelity: `load_predict(artifact)(validation_features)` ≈ validation-stage predictions (Spearman > 0.999).
- [ ] Full suite green after every task; real-data smoke + dashboard regenerated; test count updated once at the end.
- [ ] `docs/reviews/` left untracked; design spec + this plan committed on `sanity-check`.
