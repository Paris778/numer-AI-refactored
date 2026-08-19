# Oracle Parity Depth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grow the load-bearing parity surface from 8 collected tests to ~25: every custom metric (CORR/MMC/FNC/BMC) and the neutralization engine must provably match `numerai_tools.scoring` on degenerate inputs — ties, constant columns, NaN, single-row eras, duplicate/constant features, partial proportions — and the one place the engines genuinely diverge (zero-variance neutralization: the oracle NaNs the column, we return it unchanged) is pinned and documented as an intentional, submission-safe divergence, never silently blessed.

**Architecture:** Test-driven parity. Each task probes both engines on a degenerate case, then codifies `custom == official` as a permanent test — with the oracle result computed in-test (never hardcoded). One task is a real fix: `NeutralizationEngine._neutralize_era` returns unchanged predictions for zero-variance eras (nmr/risk.py:127-132) while the oracle returns the projected residual (~0); the guard is removed so the least-squares path produces the oracle-matching result, and the one test pinning the old behavior (tests/test_risk.py:373) is updated with the docs in the same commit.

**Tech Stack:** Python 3.12, pytest, polars/numpy, `numerai_tools` 0.5.3 (the installed oracle), `nmr.evaluation.EvaluationEngine`, `nmr.risk.NeutralizationEngine`.

## Global Constraints

- **The assertion is always `custom == official`.** Every test computes the oracle's answer from the same synthetic frame at run time. Never hardcode expected numeric values — that would test the oracle version, not parity. `pytest.approx` tolerances: CORR/MMC/BMC `abs=1e-6`, FNC `abs=1e-5`, neutralization `np.allclose(atol=1e-8, rtol=0)`.
- **Probe before asserting.** For any case not pre-verified by this plan, run the provided probe snippet first. If both engines return: assert equality. If both raise: assert both raise the same exception type. If they disagree: STOP — it is a parity bug; convert the task into a fix (TDD: pin the oracle behavior in a failing test, fix `nmr/`, update affected tests/docs in the same commit). Never write a test that blesses a divergence.
- **Parity holds on the domain where both engines are well-defined; intentional divergence is pinned, never blessed.** `numerai_tools.scoring.neutralize` NaNs zero-variance prediction columns (`scoring.py:394`, `df[df.columns[df.std() == 0]] = np.nan` — pandas `std`, ddof=1). Our path returns those rows unchanged (`nmr/risk.py:127-132` and `nmr/_transforms.py:78-79` — two independent guards, numpy `std`, ddof=0). Matching the oracle would inject NaN into the deploy closure and break the (0,1) submission contract; "fixing" the guard is a no-op because the second guard in `_transforms.neutralize_array` catches it anyway. The divergence is deliberate: pin it in a test, document the reason, and move on (Task 6). Any OTHER `nmr/` divergence discovered while probing is a parity bug: fix it, do not bless it.
- **Synthetic data only, deterministic seeds, fast.** No v5.3 assets, no model fits. Frames are ≤ 3 eras × ≤ 300 rows.
- **Docs follow code in the same commit.** If neutralization behavior changes, grep `ARCHITECTURE.md` and `docs/06-evaluation/evaluation-suite-bible.md` for "zero-variance" / neutralization wording and update it (AGENTS.md rule 8).
- **Coverage measurement — working form only** (root cause of the dotted-spec bug documented in the CI-coverage-floor plan): `./.venv/Scripts/python -m pytest -q --no-header -p no:cacheprovider --cov=nmr --cov-report=term-missing`; never `--cov=nmr.<submodule>`.
- **Verify per task:** `ruff check tests/<file>.py` + the targeted pytest run; final task runs the full fast gate.

---

### Task 1: CORR parity on degenerate per-era inputs

**Files:**
- Modify: `tests/test_parity.py` (append)
- Test: `tests/test_parity.py`

**Interfaces:**
- Consumes: `EvaluationEngine(backend).per_era_corr(df, pred_col, target_col) -> dict[str, float]` (evaluation.py:173); existing helper `_synthetic_eval_frame` (test_parity.py:21).
- Produces: nothing used by other tasks.

**Pre-verified (2026-08-19 probe):** constant pred → both `0.0`; constant target → both `0.0`; NaN preds → both `0.0`; single-row era → both `0.0`; tied preds → agree to 1e-16.

- [ ] **Step 1: Add the parametrized tests**

```python
def _corr_frame(*, n_eras: int = 2, n_rows: int = 8, seed: int = 20260819) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for era in range(1, n_eras + 1):
        for idx in range(n_rows):
            rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "pred": float(np.clip(rng.normal(0.5 + 0.05 * era, 0.2), 0.0, 1.0)),
                    "target": float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])),
                }
            )
    return pl.DataFrame(rows)


@pytest.mark.parametrize(
    "mutate",
    [
        "constant_pred",
        "constant_target",
        "nan_pred",
        "ties_pred",
        "single_row",
        "two_rows",
    ],
)
def test_corr_degenerate_eras_match_oracle(mutate: str) -> None:
    """Degenerate inputs never drift the custom path from numerai_tools."""
    df = _corr_frame()
    if mutate == "constant_pred":
        df = df.with_columns(pl.lit(0.5).alias("pred"))
    elif mutate == "constant_target":
        df = df.with_columns(pl.lit(0.5).alias("target"))
    elif mutate == "nan_pred":
        df = df.with_columns(
            pl.when(pl.col("id").str.ends_with("_0")).then(None).otherwise(pl.col("pred")).alias("pred")
        )
    elif mutate == "ties_pred":
        df = df.with_columns(pl.Series("pred", [0.1, 0.1, 0.5, 0.5, 0.9, 0.9, 0.2, 0.2] * 2, dtype=pl.Float64))
    elif mutate == "single_row":
        df = pl.concat([df.group_by("era", maintain_order=True).head(1)])
    elif mutate == "two_rows":
        df = pl.concat([df.group_by("era", maintain_order=True).head(2)])

    custom = EvaluationEngine("custom").per_era_corr(df, pred_col="pred", target_col="target")
    official = EvaluationEngine("official").per_era_corr(df, pred_col="pred", target_col="target")
    assert list(custom) == list(official)
    for era in custom:
        assert custom[era] == pytest.approx(official[era], abs=1e-6, nan_ok=True)
```

- [ ] **Step 2: Run to verify**

Run: `./.venv/Scripts/python -m pytest -q tests/test_parity.py::test_corr_degenerate_eras_match_oracle -p no:cacheprovider`
Expected: `6 passed`

- [ ] **Step 3: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_parity.py
git add tests/test_parity.py
git commit -m "test: CORR parity on degenerate per-era inputs"
```

---

### Task 2: MMC parity on degenerate meta/pred columns

**Files:**
- Modify: `tests/test_parity.py` (append)
- Test: `tests/test_parity.py`

**Interfaces:**
- Consumes: `EvaluationEngine(backend).per_era_mmc(df, pred_col, meta_col, target_col)` (evaluation.py:190).

**Pre-verified:** constant meta → both `0.0`. **Probe first:** ties in meta, NaN in meta, NaN in pred.

- [ ] **Step 1: Probe the unverified cases**

```bash
./.venv/Scripts/python - <<'EOF'
import numpy as np, polars as pl
from nmr.evaluation import EvaluationEngine
rng = np.random.default_rng(7)
rows = []
for era in (1, 2):
    for i in range(10):
        rows.append({"era": str(era), "id": f"{era}_{i}",
                     "pred": float(rng.uniform(0, 1)),
                     "meta": float(rng.uniform(0, 1)),
                     "target": float(rng.choice([0.0, 0.5, 1.0]))})
df = pl.DataFrame(rows)
cases = {
    "ties_meta": df.with_columns(pl.Series("meta", [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 0.9, 0.9] * 2, dtype=pl.Float64)),
    "nan_meta": df.with_columns(pl.when(pl.col("id").str.ends_with("_0")).then(None).otherwise(pl.col("meta")).alias("meta")),
    "nan_pred": df.with_columns(pl.when(pl.col("id").str.ends_with("_0")).then(None).otherwise(pl.col("pred")).alias("pred")),
}
for name, frame in cases.items():
    for backend in ("custom", "official"):
        try:
            out = EvaluationEngine(backend).per_era_mmc(frame, pred_col="pred", meta_col="meta", target_col="target")
            print(name, backend, "OK", out)
        except Exception as exc:
            print(name, backend, "RAISE", type(exc).__name__, str(exc)[:60])
EOF
```

Expected: both engines behave identically in every row (same outputs, or same exception). If any row disagrees → parity bug: stop and fix `nmr/evaluation.py` per Global Constraints; do not bless the divergence.

- [ ] **Step 2: Add the parametrized test**

```python
@pytest.mark.parametrize("mutate", ["constant_meta", "ties_meta", "nan_meta", "nan_pred"])
def test_mmc_degenerate_columns_match_oracle(mutate: str) -> None:
    df = _corr_frame().with_columns(
        pl.Series("meta", np.linspace(0.1, 0.9, df.height), dtype=pl.Float64)
    )
    if mutate == "constant_meta":
        df = df.with_columns(pl.lit(0.5).alias("meta"))
    elif mutate == "ties_meta":
        df = df.with_columns(pl.Series("meta", [0.2, 0.2, 0.5, 0.5, 0.9, 0.9, 0.4, 0.4] * 2, dtype=pl.Float64))
    elif mutate == "nan_meta":
        df = df.with_columns(
            pl.when(pl.col("id").str.ends_with("_0")).then(None).otherwise(pl.col("meta")).alias("meta")
        )
    elif mutate == "nan_pred":
        df = df.with_columns(
            pl.when(pl.col("id").str.ends_with("_0")).then(None).otherwise(pl.col("pred")).alias("pred")
        )

    custom = EvaluationEngine("custom").per_era_mmc(df, pred_col="pred", meta_col="meta", target_col="target")
    official = EvaluationEngine("official").per_era_mmc(df, pred_col="pred", meta_col="meta", target_col="target")
    assert list(custom) == list(official)
    for era in custom:
        assert custom[era] == pytest.approx(official[era], abs=1e-6, nan_ok=True)
```

Note: `pl.Series("meta", np.linspace(...), ...)` needs `df.height` — construct the series inline: `pl.Series("meta", np.linspace(0.1, 0.9, df.height), dtype=pl.Float64)`. If the probe showed both engines RAISE on a case, assert `pytest.raises(<ExcType>)` for both instead — the test must encode what the probe showed.

- [ ] **Step 3: Run to verify**

Run: `./.venv/Scripts/python -m pytest -q tests/test_parity.py::test_mmc_degenerate_columns_match_oracle -p no:cacheprovider`
Expected: `4 passed`

- [ ] **Step 4: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_parity.py
git add tests/test_parity.py
git commit -m "test: MMC parity on degenerate meta/pred columns"
```

---

### Task 3: FNC parity on degenerate feature matrices

**Files:**
- Modify: `tests/test_parity.py` (append)
- Test: `tests/test_parity.py`

**Interfaces:**
- Consumes: `EvaluationEngine(backend).per_era_fnc(df, pred_col, feature_cols, target_col)` (evaluation.py:211).

**Pre-verified:** duplicate feature columns → agree to 1e-8; constant feature column → agree to 1e-8. **Probe first:** wide matrix (features > rows), NaN feature.

- [ ] **Step 1: Probe the unverified cases**

```bash
./.venv/Scripts/python - <<'EOF'
import numpy as np, polars as pl
from nmr.evaluation import EvaluationEngine
rng = np.random.default_rng(9)
df = pl.DataFrame({
    "era": ["1"] * 6,
    "id": [f"1_{i}" for i in range(6)],
    "pred": rng.uniform(0, 1, 6),
    "target": rng.choice([0.0, 0.5, 1.0], 6),
    "f1": rng.normal(size=6), "f2": rng.normal(size=6),
})
cases = {
    "wide": df.with_columns([pl.Series(f"w{i}", rng.normal(size=6), dtype=pl.Float64) for i in range(8)]),
    "nan_feature": df.with_columns(pl.Series("f1", [1.0, np.nan, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float64)),
}
for name, frame in cases.items():
    feats = ["f1", "f2"] if name == "nan_feature" else [c for c in frame.columns if c.startswith("w")] + ["f1", "f2"]
    for backend in ("custom", "official"):
        try:
            out = EvaluationEngine(backend).per_era_fnc(frame, pred_col="pred", feature_cols=feats, target_col="target")
            print(name, backend, "OK", out)
        except Exception as exc:
            print(name, backend, "RAISE", type(exc).__name__, str(exc)[:60])
EOF
```

Expected: identical behavior per case. Divergence → parity bug, fix `nmr/evaluation.py`, do not bless.

- [ ] **Step 2: Add the tests**

```python
@pytest.mark.parametrize("mutate", ["duplicate_feature", "constant_feature", "wide_matrix", "nan_feature"])
def test_fnc_degenerate_features_match_oracle(mutate: str) -> None:
    rng = np.random.default_rng(20260819)
    df = _corr_frame().with_columns(
        pl.Series("f1", rng.normal(size=_corr_frame().height), dtype=pl.Float64),
        pl.Series("f2", rng.normal(size=_corr_frame().height), dtype=pl.Float64),
    )
    feats = ["f1", "f2"]
    if mutate == "duplicate_feature":
        df = df.with_columns(pl.col("f1").alias("f1_copy"))
        feats = ["f1", "f1_copy"]
    elif mutate == "constant_feature":
        df = df.with_columns(pl.lit(0.5).alias("fconst"))
        feats = ["f1", "fconst"]
    elif mutate == "wide_matrix":
        df = df.with_columns(
            [pl.Series(f"w{i}", rng.normal(size=df.height), dtype=pl.Float64) for i in range(12)]
        )
        feats = ["f1", "f2", *[f"w{i}" for i in range(12)]]
    elif mutate == "nan_feature":
        df = df.with_columns(
            pl.when(pl.col("id").str.ends_with("_0")).then(None).otherwise(pl.col("f1")).alias("f1")
        )

    custom = EvaluationEngine("custom").per_era_fnc(df, pred_col="pred", feature_cols=feats, target_col="target")
    official = EvaluationEngine("official").per_era_fnc(df, pred_col="pred", feature_cols=feats, target_col="target")
    assert list(custom) == list(official)
    for era in custom:
        assert custom[era] == pytest.approx(official[era], abs=1e-5, nan_ok=True)
```

Construct the base frame once (do not call `_corr_frame()` twice — heights must match):

```python
def test_fnc_degenerate_features_match_oracle(mutate: str) -> None:
    rng = np.random.default_rng(20260819)
    base = _corr_frame()
    df = base.with_columns(
        pl.Series("f1", rng.normal(size=base.height), dtype=pl.Float64),
        pl.Series("f2", rng.normal(size=base.height), dtype=pl.Float64),
    )
    ...
```

- [ ] **Step 3: Run to verify**

Run: `./.venv/Scripts/python -m pytest -q tests/test_parity.py::test_fnc_degenerate_features_match_oracle -p no:cacheprovider`
Expected: `4 passed`

- [ ] **Step 4: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_parity.py
git add tests/test_parity.py
git commit -m "test: FNC parity on degenerate feature matrices"
```

---

### Task 4: Neutralization parity across proportions

**Files:**
- Modify: `tests/test_risk_parity.py` (append)
- Test: `tests/test_risk_parity.py`

**Interfaces:**
- Consumes: `NeutralizationEngine(cache_dir=tmp_path).neutralize(df, pred_col, feature_cols, proportion) -> pl.DataFrame` (risk.py:60); existing `_synthetic_parity_frame` (test_risk_parity.py:27) and `_oracle_per_era` (test_risk_parity.py:49 — currently hardcodes `proportion=1.0`; add a `proportion: float = 1.0` parameter to it).

**Pre-verified:** proportion 0.0 and 0.5 are bit-identical to the oracle.

- [ ] **Step 1: Parameterize the existing oracle helper**

In `tests/test_risk_parity.py`, change `_oracle_per_era` to accept the proportion:

```python
def _oracle_per_era(
    df: pl.DataFrame, *, pred_col: str, feature_cols: list[str], proportion: float = 1.0
) -> np.ndarray:
    parts: list[pl.DataFrame] = []
    for era in df.get_column("era").unique(maintain_order=True).to_list():
        era_df = df.filter(pl.col("era") == era)
        pdf = era_df.to_pandas()
        neutralized = oracle_neutralize(
            pdf[[pred_col]], pdf[feature_cols], proportion=proportion
        )
        parts.append(
            era_df.with_columns(
                pl.Series(name=pred_col, values=neutralized[pred_col].to_numpy())
            )
        )
    return pl.concat(parts).sort(["era", "id"]).get_column(pred_col).to_numpy()
```

The existing call site (test_risk_parity.py:76) keeps working unchanged (default 1.0).

- [ ] **Step 2: Add the parametrized test**

```python
@pytest.mark.parametrize("proportion", [0.0, 0.25, 0.5, 0.75])
def test_neutralization_matches_oracle_across_proportions(
    tmp_path, proportion: float
) -> None:
    """proportion=1.0 is the existing test; the partial blends are equally
    part of the contract (risk configs sweep this knob)."""
    df = _synthetic_parity_frame()
    feature_cols = ["f1", "f2", "f3"]
    engine = NeutralizationEngine(cache_dir=tmp_path)

    result = engine.neutralize(
        df, pred_col="pred", feature_cols=feature_cols, proportion=proportion
    )
    actual = result.sort(["era", "id"]).get_column("pred").to_numpy()
    expected = _oracle_per_era(
        df, pred_col="pred", feature_cols=feature_cols, proportion=proportion
    )
    assert np.allclose(actual, expected, atol=NEUTRALIZE_ATOL, rtol=0.0, equal_nan=True)
```

- [ ] **Step 3: Run to verify**

Run: `./.venv/Scripts/python -m pytest -q tests/test_risk_parity.py::test_neutralization_matches_oracle_across_proportions -p no:cacheprovider`
Expected: `4 passed`

- [ ] **Step 4: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_risk_parity.py
git add tests/test_risk_parity.py
git commit -m "test: neutralization parity across proportions"
```

---

### Task 5: Neutralization parity on degenerate feature matrices

**Files:**
- Modify: `tests/test_risk_parity.py` (append)
- Test: `tests/test_risk_parity.py`

**Interfaces:**
- Consumes: same as Task 4.

**Pre-verified:** duplicate feature columns → bit-identical to oracle; constant feature column → bit-identical.

- [ ] **Step 1: Add the parametrized test**

```python
@pytest.mark.parametrize("mutate", ["duplicate_feature", "constant_feature"])
def test_neutralization_degenerate_features_match_oracle(tmp_path, mutate: str) -> None:
    df = _synthetic_parity_frame()
    feature_cols = ["f1", "f2", "f3"]
    if mutate == "duplicate_feature":
        df = df.with_columns(pl.col("f1").alias("f1_copy"))
        feature_cols = ["f1", "f1_copy", "f2", "f3"]
    elif mutate == "constant_feature":
        df = df.with_columns(pl.lit(0.5).alias("fconst"))
        feature_cols = ["f1", "fconst", "f2", "f3"]

    engine = NeutralizationEngine(cache_dir=tmp_path)
    result = engine.neutralize(df, pred_col="pred", feature_cols=feature_cols, proportion=1.0)
    actual = result.sort(["era", "id"]).get_column("pred").to_numpy()
    expected = _oracle_per_era(df, pred_col="pred", feature_cols=feature_cols)
    assert np.allclose(actual, expected, atol=NEUTRALIZE_ATOL, rtol=0.0, equal_nan=True)
```

- [ ] **Step 2: Run to verify**

Run: `./.venv/Scripts/python -m pytest -q tests/test_risk_parity.py::test_neutralization_degenerate_features_match_oracle -p no:cacheprovider`
Expected: `2 passed`

- [ ] **Step 3: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_risk_parity.py
git add tests/test_risk_parity.py
git commit -m "test: neutralization parity on degenerate feature matrices"
```

---

### Task 6: Pin the intentional zero-variance divergence (no production change)

**Files:**
- Modify: `tests/test_risk_parity.py` (append — the pin lives with the other parity tests)
- Docs: `ARCHITECTURE.md` only if its neutralization section does not already state the divergence (grep first; `nmr/_transforms.py:69-70` and `nmr/risk.py` already document it in code)

**Interfaces:**
- Consumes: `NeutralizationEngine(cache_dir=tmp_path).neutralize(df, pred_col, feature_cols, proportion)`; the installed oracle `numerai_tools.scoring.neutralize` (0.5.3).
- Produces: a permanent test that freezes the documented behavior on BOTH sides of the boundary, so a future change on either side (ours or an oracle upgrade) is caught and re-evaluated deliberately.

**Verified facts (2026-08-19, re-probed against the installed oracle):**
- Multi-row constant-pred era: oracle → `[nan nan nan nan]` (`scoring.py:394` NaNs zero-variance columns); custom → `[0.5, 0.5, 0.5, 0.5]` (guard in `risk.py:127-132`; a second guard in `_transforms.neutralize_array:78-79` would return unchanged even if the first were removed — the "fix" would be a no-op).
- Single-row era: oracle → `1.11e-16` (pandas `std` of one element is NaN under ddof=1, so the oracle's guard does not fire and the full solve produces float residue); custom → unchanged `[0.3]`. The apparent "divergence" is a ddof artifact, and our answer is the submission-safe one.
- Matching the oracle here would inject NaN into neutralized OOF and into the deploy closure — a (0,1) submission contract violation. Divergence is intentional.

- [ ] **Step 1: Add the divergence-pinning test**

Append to `tests/test_risk_parity.py`:

```python
def test_zero_variance_era_divergence_is_intentional(tmp_path) -> None:
    """The oracle NaNs zero-variance prediction columns (scoring.py:394);
    we return them unchanged. Pinned, documented divergence — matching the
    oracle would put NaN into the deploy closure and break the (0,1)
    submission contract. Re-evaluate only if the oracle changes behavior."""
    df = pl.DataFrame(
        {
            "era": ["1", "1", "2", "2", "2"],
            "id": ["a", "b", "c", "d", "e"],
            "pred": [0.5, 0.5, 0.1, 0.5, 0.9],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    engine = NeutralizationEngine(cache_dir=tmp_path)
    result = engine.neutralize(df, pred_col="pred", feature_cols=["f1"], proportion=1.0)
    era1 = result.filter(pl.col("era") == "1").get_column("pred").to_numpy()
    assert np.array_equal(era1, np.array([0.5, 0.5]))  # unchanged, not NaN

    # Pin the oracle's actual behavior too, so an upstream change is noticed.
    pdf = df.filter(pl.col("era") == "1").select(["pred", "f1"]).to_pandas()
    oracle_out = oracle_neutralize(pdf[["pred"]], pdf[["f1"]], proportion=1.0)["pred"].to_numpy()
    assert np.isnan(oracle_out).all()

    # The healthy era still matches the oracle exactly (parity holds off the
    # degenerate set).
    era2_custom = result.filter(pl.col("era") == "2").get_column("pred").to_numpy()
    era2_expected = _oracle_per_era(
        df.filter(pl.col("era") == "2"), pred_col="pred", feature_cols=["f1"]
    )
    assert np.allclose(era2_custom, era2_expected, atol=NEUTRALIZE_ATOL, rtol=0.0, equal_nan=True)
```

- [ ] **Step 2: Run to verify it passes**

Run: `./.venv/Scripts/python -m pytest -q tests/test_risk_parity.py::test_zero_variance_era_divergence_is_intentional -p no:cacheprovider`
Expected: `1 passed`. The existing pin in `tests/test_risk.py:373` stays untouched (its behavior is unchanged — no production code moves).

- [ ] **Step 3: Check the docs state the divergence**

```bash
grep -rn "zero-variance\|zero variance" ARCHITECTURE.md docs/06-evaluation/ | head
```

If `ARCHITECTURE.md`'s neutralization section lacks a one-liner, add: "Zero-variance prediction eras are returned unchanged (both `risk.py` and `_transforms.neutralize_array` guard this) — an intentional divergence from `numerai_tools.scoring.neutralize`, which NaNs zero-variance columns (`scoring.py:394`); matching it would inject NaN into the deploy closure and break the (0,1) submission contract. Pinned by `tests/test_risk_parity.py::test_zero_variance_era_divergence_is_intentional`."

- [ ] **Step 4: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_risk_parity.py
git add tests/test_risk_parity.py ARCHITECTURE.md
git commit -m "test: pin the intentional zero-variance neutralization divergence"
```

---

### Task 7: BMC parity depth — multi-era and missing benchmark rows

**Files:**
- Modify: `tests/test_parity.py` (append)
- Test: `tests/test_parity.py`

**Interfaces:**
- Consumes: `EvaluationEngine(backend).per_era_bmc(df, pred_col, benchmark_col, target_col, min_overlap_eras)` (evaluation.py:235); existing `_slice3_inputs` (test_parity.py:198) builds meta/benchmarks/features/targets frames. Oracle: `numerai_tools.scoring.correlation_contribution` per era (pattern already in `test_slice3_bmc_oracle_parity`).

**Probe first:** benchmark column containing NaN rows (left-join case) — verify both engines agree before codifying.

- [ ] **Step 1: Probe NaN-benchmark behavior**

```bash
./.venv/Scripts/python - <<'EOF'
import numpy as np, polars as pl
from numerai_tools.scoring import correlation_contribution
from nmr.evaluation import EvaluationEngine
rng = np.random.default_rng(11)
rows = []
for era in range(1, 22):
    for i in range(10):
        rows.append({"era": f"{era:04d}", "id": f"{era}_{i}",
                     "prediction": float(rng.uniform(0, 1)), "target": float(rng.uniform(0, 1)),
                     "benchmark": float(rng.uniform(0, 1)) if (era + i) % 5 else np.nan})
df = pl.DataFrame(rows)
out = EvaluationEngine("custom").per_era_bmc(df, pred_col="prediction", benchmark_col="benchmark", target_col="target", min_overlap_eras=20)
print("custom:", {k: round(v, 6) for k, v in sorted(out.items(), key=int)[:4]})
era = sorted(out, key=int)[0]
pdf = df.filter(pl.col("era") == era).to_pandas()
direct = float(correlation_contribution(pdf[["prediction"]], pdf["benchmark"].rename("benchmark"), pdf["target"].rename("target"))["prediction"])
print("direct era", era, ":", round(direct, 6))
EOF
```

Expected: the custom per-era value equals the direct `correlation_contribution` call on the same era (NaN handling included). If they disagree → parity bug in `per_era_bmc`; fix, do not bless.

- [ ] **Step 2: Add the multi-era BMC test**

```python
def test_bmc_multi_era_and_nan_benchmark_match_oracle() -> None:
    meta_model, benchmarks, features, targets = _slice3_inputs(n_eras=40, rows_per_era=20)
    cfg_pred = targets.select(["era", "id"]).with_columns(
        (0.7 * pl.col("id").cum_count()).cast(pl.Float64).alias("prediction")
    )
    base = (
        cfg_pred.join(targets.select(["era", "id", "target"]), on=["era", "id"], how="inner")
        .join(benchmarks, on=["era", "id"], how="left")  # nulls where unmatched
    )
    # inject explicit NaN benchmark rows too
    base = base.with_columns(
        pl.when(pl.col("v52_lgbm_cyrusd20").is_null()).then(np.nan).otherwise(pl.col("v52_lgbm_cyrusd20")).alias("v52_lgbm_cyrusd20")
    )

    evaluator = EvaluationEngine("custom")
    per_era = evaluator.per_era_bmc(
        base, pred_col="prediction", benchmark_col="v52_lgbm_cyrusd20",
        target_col="target", min_overlap_eras=20,
    )
    eras = sorted(per_era, key=int)
    assert len(eras) >= 20
    for one_era in eras[:3]:  # spot-check three eras against the direct oracle
        pdf = base.filter(pl.col("era") == one_era).select(
            ["prediction", "v52_lgbm_cyrusd20", "target"]
        ).to_pandas()
        direct = float(
            correlation_contribution(
                pdf[["prediction"]],
                pdf["v52_lgbm_cyrusd20"].rename("v52_lgbm_cyrusd20"),
                pdf["target"].rename("target"),
            )["prediction"]
        )
        assert per_era[one_era] == pytest.approx(direct, abs=1e-6)
```

If the Step-1 probe shows the engines disagree on NaN benchmarks, do NOT write this test as-is: report the divergence and (with the fix) pin the corrected behavior — the assertion stays `custom == direct oracle`.

- [ ] **Step 3: Run to verify**

Run: `./.venv/Scripts/python -m pytest -q tests/test_parity.py::test_bmc_multi_era_and_nan_benchmark_match_oracle -p no:cacheprovider`
Expected: `1 passed`

- [ ] **Step 4: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_parity.py
git add tests/test_parity.py
git commit -m "test: BMC parity multi-era with missing benchmark rows"
```

---

### Task 8: Verify the deepened parity surface and the full fast gate

**Files:** none (verification only)

- [ ] **Step 1: Count the parity surface**

Run: `./.venv/Scripts/python -m pytest tests/test_parity.py tests/test_risk_parity.py --collect-only -q -p no:cacheprovider 2>&1 | tail -1`
Expected: ≥ 25 collected (baseline was 8).

- [ ] **Step 2: Full fast gate**

```bash
./.venv/Scripts/python -m ruff check .
./.venv/Scripts/python -m pytest -q -p no:cacheprovider
```

Expected: ruff clean; ≥ 865 + new tests all passing; zero production-code changes — the only behavioral statements added are the divergence pins (oracle NaNs, ours unchanged).

---

## Self-Review Notes

- **Spec coverage:** every gap named in the remediation report for oracle parity ("the entire guarantee rests on 8 collected tests") maps to a task: CORR (T1), MMC (T2), FNC (T3), neutralization proportions (T4) + degenerate features (T5) + the zero-variance divergence pinned as intentional with the oracle's NaN behavior documented on both sides (T6), BMC (T7). Unverified cases carry a probe step that converts divergence into a fix task per the Global Constraints — with the one named exception (zero-variance) now pinned instead of "fixed".
- **Placeholder scan:** no TBDs; all probes and tests are literal code.
- **Type consistency:** `_oracle_per_era` gains `proportion: float = 1.0` in T4 and is reused by T5/T6 — the signature matches everywhere it is called. `_slice3_inputs(n_eras=40, rows_per_era=20)` matches its definition (test_parity.py:198-204). `per_era_*` method signatures match evaluation.py:173-235.
