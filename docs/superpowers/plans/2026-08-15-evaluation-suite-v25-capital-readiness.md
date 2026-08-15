# Quantitative Evaluation Suite v2.5 (Capital-Readiness Engine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add six capital-readiness metrics (CAGR 1Y, gain-to-pain, bounded Kelly, strict MMC-down, cross-era turnover, overlapping capital simulator) to the scorecard via `nmr/payout.py`, `nmr/evaluation.py`, and `nmr/scorecard.py`.

**Architecture:** Pure float64 helpers in `nmr/payout.py` (CAGR/GPR/Kelly/simulator) and pure frame/slicing helpers in `nmr/evaluation.py` (turnover, downside-era slicing); `evaluate_model` wires them into `MetricScorecard` with 12 new required fields, all derived from the already-joined `base` frame — zero new inputs, zero changes to runner/benchmark/campaign. All new fields are deterministic and participate in canonical scorecard hashes.

**Tech Stack:** Python 3.11+, NumPy (float64), SciPy (`spearmanr`), Polars. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-15-evaluation-suite-v25-capital-readiness-design.md` (approved by user 2026-08-15).

## Global Constraints

- Run Python via `./.venv/Scripts/python`; pytest from the repo root (`./.venv/Scripts/python -m pytest`). Never the pip shim.
- **No commit without the user's explicit go-ahead per task.** Each task ends with a commit step; pause there if the user has not pre-authorized.
- All new metric math in float64; degenerate inputs raise `ValueError` (no silent coercion); legitimately-absent metrics use `None` + a `*_reason` string.
- `MetricScorecard` new fields have **no defaults** (fail-loud convention). `PayoutResult.overlapping_sim` defaults to `None` only for backward-compatible direct constructions.
- No wall-clock, RNG, or absolute paths in any new metric. Canonical bytes keep the new fields; only `timing_*` / `quality_metric_*` columns are stripped (unchanged behavior — add no stripping logic).
- Kelly is computed on the **raw unclipped** series (`series.raw`) — director-locked.
- MMC-down is **strict**: `CORR_meta < 0`, `M_min = 5`, `None` + `"insufficient_downside_eras"` when vacuous — director-locked.
- No new dependencies (`numpy`, `scipy`, `polars` only).
- SSOT: `ARCHITECTURE.md`, `docs/06-evaluation/evaluation-suite-bible.md`, and AGENTS.md test counts are updated in the same changeset as the code that requires them (Task 10).

---

### Task 1: `annual_compounded_return` helper

**Files:**
- Create: (none)
- Modify: `nmr/payout.py` (add helper + `__all__`)
- Test: `tests/test_payout.py` (append tests + import)

**Interfaces:**
- Consumes: existing `_as_finite_1d` in `nmr/payout.py`.
- Produces: `annual_compounded_return(clipped: np.ndarray | list[float] | tuple[float, ...], *, eras_per_year: float = 52.0) -> float` — used by Task 5 and Task 8's tests.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_payout.py` and extend the `from nmr.payout import (...)` block with `annual_compounded_return`:

```python
def test_annual_cagr_math() -> None:
    series = np.full(52, 0.01)
    expected = (1.01) ** 52 - 1.0
    assert annual_compounded_return(series) == pytest.approx(expected, rel=1e-12)


def test_annual_cagr_ruin_and_short_series() -> None:
    # product <= 0 -> -1.0 (total loss)
    assert annual_compounded_return(np.array([-1.0, 0.05])) == -1.0
    assert annual_compounded_return(np.array([-1.5, 0.05])) == -1.0
    # fewer than 2 observations -> 0.0
    assert annual_compounded_return(np.array([0.01])) == 0.0


def test_annual_cagr_input_validation() -> None:
    with pytest.raises(ValueError):
        annual_compounded_return(np.array([]))
    with pytest.raises(ValueError):
        annual_compounded_return(np.array([0.01, np.nan]))
    with pytest.raises(ValueError):
        annual_compounded_return(np.zeros((2, 2)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_payout.py -q -k cagr`
Expected: FAIL — `ImportError: cannot import name 'annual_compounded_return'` (first), then collection errors for the other two.

- [ ] **Step 3: Write the implementation**

In `nmr/payout.py`, after `payout_series` (and add the name to `__all__`):

```python
def annual_compounded_return(
    clipped: np.ndarray | list[float] | tuple[float, ...],
    *,
    eras_per_year: float = 52.0,
) -> float:
    """Annualized geometric compounded return on stake.

    Computes (prod(1 + r_t)) ** (eras_per_year / n) - 1 in float64 over the
    clipped round-return series. Returns -1.0 when the wealth product is
    <= 0 (total loss) and 0.0 when fewer than 2 observations exist.
    """
    x = _as_finite_1d(clipped, name="clipped")
    n = int(x.size)
    if n < 2:
        return 0.0
    cum_growth = float(np.prod(1.0 + x))
    if cum_growth <= 0.0:
        return -1.0
    return float(cum_growth ** (float(eras_per_year) / float(n)) - 1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_payout.py -q`
Expected: PASS (all existing payout tests + the 3 new ones).

- [ ] **Step 5: Commit**

```bash
git add nmr/payout.py tests/test_payout.py
git commit -m "feat: annual compounded stake return (CAGR 1Y) helper"
```

---

### Task 2: `gain_to_pain_ratio` helper

**Files:**
- Modify: `nmr/payout.py`
- Test: `tests/test_payout.py`

**Interfaces:**
- Consumes: `_as_finite_1d`.
- Produces: `gain_to_pain_ratio(clipped: np.ndarray | list[float] | tuple[float, ...]) -> float` — used by Task 5.

- [ ] **Step 1: Write the failing tests**

Add `gain_to_pain_ratio` to the `nmr.payout` import in `tests/test_payout.py`, then append:

```python
def test_gain_to_pain_ratio() -> None:
    series = np.array([0.03, 0.03, 0.03, -0.01])
    assert gain_to_pain_ratio(series) == pytest.approx(9.0)


def test_gain_to_pain_zero_burn_states() -> None:
    # all positive -> +inf
    assert math.isinf(gain_to_pain_ratio(np.array([0.02, 0.01])))
    # all zero -> 0.0
    assert gain_to_pain_ratio(np.array([0.0, 0.0])) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_payout.py -q -k gain_to_pain`
Expected: FAIL — `ImportError: cannot import name 'gain_to_pain_ratio'`.

- [ ] **Step 3: Write the implementation**

In `nmr/payout.py`, after `annual_compounded_return` (add to `__all__`):

```python
def gain_to_pain_ratio(
    clipped: np.ndarray | list[float] | tuple[float, ...],
) -> float:
    """Gain-to-pain ratio: sum(positive returns) / sum(|negative returns|).

    Zero-loss states return +inf (precedented by ``calmar``; the canonical
    JSON sanitizer maps non-finite floats to strings) or 0.0 when the series
    is entirely flat.
    """
    x = _as_finite_1d(clipped, name="clipped")
    pos_sum = float(np.sum(np.maximum(x, 0.0)))
    neg_sum = float(np.sum(np.abs(np.minimum(x, 0.0))))
    if neg_sum == 0.0:
        return float("inf") if pos_sum > 0.0 else 0.0
    return float(pos_sum / neg_sum)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_payout.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nmr/payout.py tests/test_payout.py
git commit -m "feat: gain-to-pain ratio helper"
```

---

### Task 3: `kelly_fraction` helper (raw series)

**Files:**
- Modify: `nmr/payout.py`
- Test: `tests/test_payout.py`

**Interfaces:**
- Consumes: `_as_finite_1d`.
- Produces: `kelly_fraction(raw: np.ndarray | list[float] | tuple[float, ...]) -> float` — used by Task 5. Input is the **unclipped raw** payout series (director-locked; the caller passes `series.raw`).

- [ ] **Step 1: Write the failing tests**

Add `kelly_fraction` to the `nmr.payout` import in `tests/test_payout.py`, then append:

```python
def test_kelly_fraction_bounds_and_degenerate() -> None:
    # zero variance -> 0.0
    assert kelly_fraction(np.array([0.01, 0.01, 0.01])) == 0.0
    # non-positive mean -> 0.0
    assert kelly_fraction(np.array([-0.01, 0.01])) == 0.0
    # mid-range: mu=0.02, sigma=0.2 -> 0.5
    series = np.array([0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.1])
    mu = float(np.mean(series))
    var = float(np.var(series, ddof=0))
    assert kelly_fraction(series) == pytest.approx(min(1.0, mu / var))
    # saturation: mu=0.1, var=1e-6 -> mu/var = 100,000 -> capped at 1.0
    # (a constant array would have zero variance -> 0.0, so use a
    #  low-variance positive-drift sequence)
    assert kelly_fraction(np.array([0.101, 0.099, 0.101, 0.099])) == 1.0


def test_kelly_uses_raw_not_clipped() -> None:
    # Raw series: 19 x +0.03 and one -0.5. Raw Kelly = 0.0035 / 0.01334275
    # ~ 0.2623 (< 1). The clipped variant compresses variance so its Kelly
    # saturates at 1.0. Locks the director-approved raw-series contract.
    raw = np.array([0.03] * 19 + [-0.5])
    clipped = np.clip(raw, -0.05, 0.05)
    kelly_raw = kelly_fraction(raw)
    assert 0.0 < kelly_raw < 1.0
    assert kelly_fraction(clipped) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_payout.py -q -k kelly`
Expected: FAIL — `ImportError: cannot import name 'kelly_fraction'`.

- [ ] **Step 3: Write the implementation**

In `nmr/payout.py`, after `gain_to_pain_ratio` (add to `__all__`):

```python
def kelly_fraction(
    raw: np.ndarray | list[float] | tuple[float, ...],
) -> float:
    """Bounded discrete Kelly stake fraction: min(1.0, max(0.0, mu / var)).

    Computed on the RAW (unclipped) payout series. The clipped series has
    Popoviciu-bounded variance (sigma^2 <= 0.0025 under the +-5% clip), so
    mu/sigma^2 there saturates at 1.0 for every viable model and carries no
    discrimination. ``payout_report`` passes ``series.raw``.
    """
    x = _as_finite_1d(raw, name="raw")
    mu = float(np.mean(x))
    var = float(np.var(x, ddof=0))
    if var == 0.0 or mu <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, mu / var)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_payout.py -q`
Expected: PASS. (Check `test_kelly_uses_raw_not_clipped` passes — raw Kelly ≈ 0.2623 < 1, clipped Kelly = 1.0 exactly; verified numerically 2026-08-15.)

- [ ] **Step 5: Commit**

```bash
git add nmr/payout.py tests/test_payout.py
git commit -m "feat: bounded Kelly fraction on raw payout series"
```

---

### Task 4: Overlapping capital simulator

**Files:**
- Modify: `nmr/payout.py` (module docstring, `_HORIZON_ERAS` constant, `OverlappingSimulationResult`, `simulate_overlapping_portfolio`, `__all__`)
- Test: `tests/test_payout.py`

**Interfaces:**
- Consumes: `_as_finite_1d`.
- Produces:
  - `_HORIZON_ERAS: dict[str, int] = {"20D": 20, "60D": 60}`
  - `OverlappingSimulationResult(portfolio_cagr: float, portfolio_max_drawdown: float, avg_capital_utilization: float, final_equity: float)` (frozen dataclass)
  - `simulate_overlapping_portfolio(clipped, *, horizon_eras: int = 20, initial_capital: float = 1.0, eras_per_year: float = 52.0) -> OverlappingSimulationResult` — used by Task 5.

- [ ] **Step 1: Write the failing tests**

Add `simulate_overlapping_portfolio` and `OverlappingSimulationResult` to the `nmr.payout` import in `tests/test_payout.py`, then append:

```python
def test_overlapping_sim_zero_return_lockup() -> None:
    # K=20, n=100, all returns zero. Steady-state utilization (pre-deployment)
    # is (K-1)/K = 0.95; 20 warm-up eras average 0.475 -> overall 0.855 exactly.
    result = simulate_overlapping_portfolio(
        np.zeros(100), horizon_eras=20
    )
    assert isinstance(result, OverlappingSimulationResult)
    assert result.final_equity == pytest.approx(1.0)
    assert result.portfolio_cagr == 0.0
    assert result.portfolio_max_drawdown == pytest.approx(0.0, abs=1e-12)
    assert result.avg_capital_utilization == pytest.approx(0.855, abs=1e-9)


def test_overlapping_sim_short_series() -> None:
    result = simulate_overlapping_portfolio(np.full(5, 0.01), horizon_eras=20)
    assert result.portfolio_cagr == 0.0
    assert result.portfolio_max_drawdown == 0.0
    assert result.avg_capital_utilization == 0.0
    assert result.final_equity == 1.0


def test_overlapping_sim_drag() -> None:
    # Positive-drift volatile series: cash drag makes the tranched portfolio
    # CAGR strictly below the serial geometric product CAGR.
    # (Verified numerically: port_cagr ~ 0.0348 vs serial_cagr ~ 1.559.)
    series = np.array([0.08, -0.04] * 30)
    result = simulate_overlapping_portfolio(series, horizon_eras=20)
    serial_final = float(np.prod(1.0 + series))
    serial_cagr = serial_final ** (52.0 / 60.0) - 1.0
    assert result.portfolio_cagr == pytest.approx(
        result.final_equity ** (52.0 / 60.0) - 1.0
    )
    assert result.portfolio_cagr < serial_cagr
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_payout.py -q -k overlapping_sim`
Expected: FAIL — `ImportError: cannot import name 'simulate_overlapping_portfolio'`.

- [ ] **Step 3: Write the implementation**

In `nmr/payout.py`:

(a) Extend the module docstring with the accounting convention (panel-required fix):

```python
"""Payout proxy and downside diagnostics for Evaluation Suite v2.

...existing text...

Terminal-tranche accounting convention (simulator): tranches still locked at
the end of the series are carried at par principal in ``final_equity`` and the
equity curve — no mark-to-market and no unrealized payoff for the final
``horizon_eras`` tranches.
"""
```

(b) After `_as_finite_1d`, add the constant and dataclass (add `OverlappingSimulationResult` to `__all__`; `_HORIZON_ERAS` is private and stays out):

```python
_HORIZON_ERAS: dict[str, int] = {"20D": 20, "60D": 60}


@dataclass(frozen=True)
class OverlappingSimulationResult:
    portfolio_cagr: float
    portfolio_max_drawdown: float
    avg_capital_utilization: float
    final_equity: float
```

(c) After `kelly_fraction`, add:

```python
def simulate_overlapping_portfolio(
    clipped: np.ndarray | list[float] | tuple[float, ...],
    *,
    horizon_eras: int = 20,
    initial_capital: float = 1.0,
    eras_per_year: float = 52.0,
) -> OverlappingSimulationResult:
    """Simulate multi-round concurrent capital lockup and dynamic reinvestment.

    Each era deploys min(cash, total_equity / horizon_eras) into a new tranche
    that matures ``horizon_eras`` eras later with the initiating era's return
    (1 + r_{t - K}) — Numerai round semantics. Equity and utilization are
    recorded BEFORE the era's deployment. Tranches still locked at the end of
    the series are carried at par principal (no unrealized payoff). Series
    shorter than the horizon return a zeroed result.
    """
    x = _as_finite_1d(clipped, name="clipped")
    n = int(x.size)
    horizon = int(horizon_eras)
    if horizon < 1:
        raise ValueError("horizon_eras must be >= 1")
    if n < horizon:
        return OverlappingSimulationResult(
            portfolio_cagr=0.0,
            portfolio_max_drawdown=0.0,
            avg_capital_utilization=0.0,
            final_equity=float(initial_capital),
        )

    cash = float(initial_capital)
    active_stakes: list[tuple[int, float]] = []
    equity_curve = np.zeros(n, dtype=float)
    utilization = np.zeros(n, dtype=float)

    for t in range(n):
        still_active: list[tuple[int, float]] = []
        for maturity_t, principal in active_stakes:
            if maturity_t == t:
                cash += principal * (1.0 + x[maturity_t - horizon])
            else:
                still_active.append((maturity_t, principal))
        active_stakes = still_active

        locked_capital = sum(p for _, p in active_stakes)
        total_equity = cash + locked_capital
        equity_curve[t] = total_equity
        utilization[t] = (
            locked_capital / total_equity if total_equity > 0 else 0.0
        )

        allocated = min(cash, total_equity / float(horizon))
        cash -= allocated
        active_stakes.append((t + horizon, allocated))

    final_eq = float(equity_curve[-1])
    cagr = (
        float(final_eq / initial_capital) ** (float(eras_per_year) / float(n))
        - 1.0
        if final_eq > 0.0
        else -1.0
    )
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = np.where(
        running_max > 0, (running_max - equity_curve) / running_max, 0.0
    )
    return OverlappingSimulationResult(
        portfolio_cagr=cagr,
        portfolio_max_drawdown=float(np.max(drawdowns)),
        avg_capital_utilization=float(np.mean(utilization)),
        final_equity=final_eq,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_payout.py -q`
Expected: PASS. The utilization assertion is exact (0.855 = (20×0.475 + 80×0.95)/100). The zero-return max-drawdown assertion uses `pytest.approx(0.0, abs=1e-12)` — float64 roundoff makes equity dip to ~0.9999999999999997 (drawdown ~3.33e-16), which exact `== 0.0` would reject.

- [ ] **Step 5: Commit**

```bash
git add nmr/payout.py tests/test_payout.py
git commit -m "feat: multi-round overlapping capital velocity simulator"
```

---

### Task 5: `payout_report` integration

**Files:**
- Modify: `nmr/payout.py` (`PayoutResult` + `payout_report`)
- Test: `tests/test_payout.py`

**Interfaces:**
- Consumes: Task 1–4 helpers, `_HORIZON_ERAS`, `OverlappingSimulationResult`.
- Produces: extended `PayoutResult` with `cagr_1y: float`, `gain_to_pain_ratio: float`, `kelly_fraction: float`, `overlapping_sim: OverlappingSimulationResult | None = None` — consumed by Task 8.

- [ ] **Step 1: Write the failing test**

In `tests/test_payout.py`, add `annual_compounded_return`, `gain_to_pain_ratio`, `kelly_fraction`, `simulate_overlapping_portfolio` to the imports if not already imported (they are, from Tasks 1–4), then append:

```python
def test_payout_report_includes_capital_metrics() -> None:
    corr = {f"{i:04d}": 0.02 + 0.01 * ((i % 5) - 2) for i in range(1, 41)}
    mmc = {f"{i:04d}": 0.01 for i in range(1, 41)}
    report = payout_report(corr, mmc, horizon="20D", n_trials=1, seed=7)
    series = payout_series(corr, mmc)
    assert report.cagr_1y == pytest.approx(
        annual_compounded_return(series.clipped)
    )
    assert report.gain_to_pain_ratio == pytest.approx(
        gain_to_pain_ratio(series.clipped)
    )
    assert report.kelly_fraction == pytest.approx(kelly_fraction(series.raw))
    assert report.overlapping_sim is not None
    expected_sim = simulate_overlapping_portfolio(series.clipped, horizon_eras=20)
    assert report.overlapping_sim.final_equity == pytest.approx(
        expected_sim.final_equity
    )
    assert report.overlapping_sim.avg_capital_utilization == pytest.approx(
        expected_sim.avg_capital_utilization
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_payout.py -q -k capital_metrics`
Expected: FAIL — `AttributeError: 'PayoutResult' object has no attribute 'cagr_1y'`.

- [ ] **Step 3: Write the implementation**

In `nmr/payout.py`:

(a) Extend `PayoutResult` (fields after `time_to_recovery`):

```python
    cagr_1y: float
    gain_to_pain_ratio: float
    kelly_fraction: float
    overlapping_sim: OverlappingSimulationResult | None = None
```

(b) In `payout_report`, replace the `return PayoutResult(...)` block's tail:

```python
    clipped = series.clipped
    horizon_eras = _HORIZON_ERAS[horizon]
    return PayoutResult(
        n_eras=n,
        pf=float(pf),
        mean_payout=float(np.mean(clipped)),
        payout_ci=payout_ci,
        deflated_sharpe=float(dsr),
        burn_rate=burn_rate(clipped),
        cvar5=cvar(clipped, q=0.05),
        max_drawdown=max_drawdown(clipped),
        sortino=sortino(clipped),
        calmar=calmar(clipped),
        mmc_sharpe=ac_adjusted_sharpe(mmc_aligned, horizon=horizon),
        max_burn_streak=max_burn_streak(clipped),
        time_to_recovery=time_to_recovery(clipped),
        cagr_1y=annual_compounded_return(clipped),
        gain_to_pain_ratio=gain_to_pain_ratio(clipped),
        kelly_fraction=kelly_fraction(series.raw),
        overlapping_sim=simulate_overlapping_portfolio(
            clipped, horizon_eras=horizon_eras
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_payout.py -q`
Expected: PASS. (If any existing test constructs `PayoutResult` directly it will now fail with a missing-argument TypeError — grep confirmed none exist outside `payout_report`.)

- [ ] **Step 5: Commit**

```bash
git add nmr/payout.py tests/test_payout.py
git commit -m "feat: payout_report carries CAGR, GPR, Kelly, and simulator"
```

---

### Task 6: `downside_era_indices` helper

**Files:**
- Modify: `nmr/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: existing `sorted_era_labels` in `nmr/evaluation.py`.
- Produces: `downside_era_indices(meta_corr: Mapping[str, float], *, threshold: float = 0.0) -> list[str]` — chronological list of eras with meta CORR strictly below threshold; used by Task 8.

- [ ] **Step 1: Write the failing tests**

In `tests/test_evaluation.py`, add `downside_era_indices` to the `nmr.evaluation` import, then append:

```python
def test_downside_era_indices_strict() -> None:
    meta_corr = {"0002": 0.01, "0001": -0.02, "0003": 0.0, "0004": -0.01}
    assert downside_era_indices(meta_corr) == ["0001", "0004"]
    assert downside_era_indices(meta_corr, threshold=0.0) == ["0001", "0004"]


def test_downside_era_indices_rejects_non_numeric_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="Non-numeric era label"):
        downside_era_indices({"X": 0.1})
    with pytest.raises(ValueError, match="threshold"):
        downside_era_indices({"0001": -0.1}, threshold=np.nan)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_evaluation.py -q -k downside`
Expected: FAIL — `ImportError: cannot import name 'downside_era_indices'`.

- [ ] **Step 3: Write the implementation**

In `nmr/evaluation.py`, after `sorted_era_labels` (add to `__all__`):

```python
def downside_era_indices(
    meta_corr: Mapping[str, float],
    *,
    threshold: float = 0.0,
) -> list[str]:
    """Chronological eras where the meta model's CORR is strictly below threshold.

    Strict comparison (CORR_meta < threshold) per the director-locked
    MMC-down contract. Era labels must be numeric (fail-loud via
    ``sorted_era_labels``).
    """
    threshold_f = float(threshold)
    if not math.isfinite(threshold_f):
        raise ValueError("threshold must be finite")
    return [
        era
        for era in sorted_era_labels(list(meta_corr.keys()))
        if float(meta_corr[era]) < threshold_f
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_evaluation.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nmr/evaluation.py tests/test_evaluation.py
git commit -m "feat: strict downside-era slicing helper"
```

---

### Task 7: `per_era_turnover` helper

**Files:**
- Modify: `nmr/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `sorted_era_labels`; `scipy.stats.spearmanr` (new top-level import in `nmr/evaluation.py`).
- Produces: `per_era_turnover(df: pl.DataFrame, *, pred_col: str, era_col: str = "era", id_col: str = "id") -> dict[str, float]` — maps each target era to `1 - Spearman(pred_{t-1}, pred_t)` over shared IDs; transitions with < 10 shared IDs are skipped. Used by Task 8.

- [ ] **Step 1: Write the failing tests**

In `tests/test_evaluation.py`, add `per_era_turnover` to the `nmr.evaluation` import, then append:

```python
def _turnover_frame() -> pl.DataFrame:
    rows: list[dict[str, float | str]] = []
    for j in range(12):
        rows.append({"era": "0001", "id": f"id{j:03d}", "prediction": float(j)})
        rows.append({"era": "0002", "id": f"id{j:03d}", "prediction": float(j)})
        rows.append(
            {"era": "0003", "id": f"id{j:03d}", "prediction": float(11 - j)}
        )
    return pl.DataFrame(rows)


def test_per_era_turnover_identical_and_inverse() -> None:
    out = per_era_turnover(_turnover_frame(), pred_col="prediction")
    assert out["0002"] == pytest.approx(0.0)  # identical -> rho=1
    assert out["0003"] == pytest.approx(2.0)  # inverted -> rho=-1


def test_per_era_turnover_skips_small_intersection() -> None:
    rows: list[dict[str, float | str]] = []
    for j in range(12):
        rows.append({"era": "0001", "id": f"id{j:03d}", "prediction": float(j)})
    for j in range(12, 17):  # only 5 shared ids -> skipped
        rows.append({"era": "0002", "id": f"id{j:03d}", "prediction": float(j)})
    out = per_era_turnover(pl.DataFrame(rows), pred_col="prediction")
    assert out == {}


def test_per_era_turnover_missing_columns_raises() -> None:
    # pred_col is a required keyword-only argument; pass it so the missing
    # id column triggers the ValueError path (omitting pred_col would raise
    # TypeError for the wrong reason).
    with pytest.raises(ValueError, match="Missing required columns"):
        per_era_turnover(
            pl.DataFrame({"era": ["0001"], "prediction": [0.5]}),
            pred_col="prediction",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_evaluation.py -q -k turnover`
Expected: FAIL — `ImportError: cannot import name 'per_era_turnover'`.

- [ ] **Step 3: Write the implementation**

In `nmr/evaluation.py`:

(a) Add the top-level import next to the existing imports:

```python
from scipy.stats import spearmanr
```

(b) Add after `downside_era_indices` (add to `__all__`):

```python
def per_era_turnover(
    df: pl.DataFrame,
    *,
    pred_col: str,
    era_col: str = "era",
    id_col: str = "id",
) -> dict[str, float]:
    """Spearman prediction rank turnover: 1 - rho(pred_{t-1}, pred_t) on shared IDs.

    For each consecutive chronological era pair, ranks the predictions of the
    previous and current era over the intersection of stock IDs (>= 10 rows
    required) and returns 1 - Spearman rho, bounded in [0, 2]. Non-finite rho
    (degenerate era) maps to 0.0 -> turnover 1.0.
    """
    missing = [c for c in (era_col, id_col, pred_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    eras = sorted_era_labels(df.get_column(era_col).to_list())
    if len(eras) < 2:
        return {}

    parts = {
        str(part.get_column(era_col).to_list()[0]): part.select(
            [id_col, pred_col]
        )
        for part in df.partition_by(era_col, maintain_order=True)
    }

    turnovers: dict[str, float] = {}
    for prev_era, curr_era in zip(eras, eras[1:]):
        joined = parts[prev_era].join(
            parts[curr_era], on=id_col, how="inner", suffix="_curr"
        )
        if joined.height < 10:
            continue
        rho, _ = spearmanr(
            joined.get_column(pred_col).to_numpy(),
            joined.get_column(f"{pred_col}_curr").to_numpy(),
        )
        if not math.isfinite(rho):
            rho = 0.0
        turnovers[curr_era] = float(1.0 - rho)
    return turnovers
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_evaluation.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nmr/evaluation.py tests/test_evaluation.py
git commit -m "feat: cross-era prediction turnover (Spearman)"
```

---

### Task 8: Scorecard integration (12 new fields + wiring)

**Files:**
- Modify: `nmr/scorecard.py` (imports, `_MMC_DOWN_MIN_ERAS`, `MetricScorecard`, `to_frame`, `evaluate_model`)
- Modify: `tests/test_registry.py` (`_scorecard` helper)
- Modify: `tests/test_benchmark_gates.py` (`_scorecard` helper)
- Modify: `tests/test_scorecard.py` (imports, extend the `to_frame` column contract test, add 4 new tests)

**Interfaces:**
- Consumes: Task 5's `PayoutResult` fields; Task 6's `downside_era_indices`; Task 7's `per_era_turnover`; existing `EvaluationEngine`, `sorted_era_labels`, `_mark`.
- Produces: `MetricScorecard` with 12 new required fields and `to_frame()` columns `cagr_1y, gain_to_pain_ratio, kelly_fraction, mmc_down, mmc_down_n_eras, mmc_down_reason, turnover_mean, turnover_std, turnover_reason, sim_portfolio_cagr, sim_portfolio_mdd, sim_capital_utilization`; `evaluate_model` populates them.

- [ ] **Step 1: Write the failing tests**

In `tests/test_scorecard.py`, extend the `nmr.payout` / `nmr.evaluation` imports:

```python
from nmr.evaluation import EvaluationEngine, downside_era_indices
from nmr.payout import (
    annual_compounded_return,
    gain_to_pain_ratio,
    kelly_fraction,
    payout_report,
    payout_series,
    simulate_overlapping_portfolio,
)
```

Then append these four tests:

```python
def _mmc_down_frames(n_down: int) -> tuple[pl.DataFrame, ...]:
    """30 eras x 4 rows; meta CORR < 0 in exactly the LAST n_down eras.

    Sign guarantee (holds for ANY corr(pred, target) in [-1, 1]):
      up eras:   meta =  target + 0.5*pred -> corr(meta, target) >= 0.5 > 0
      down eras: meta = -target + 0.5*pred -> corr(meta, target) <= -0.5 < 0
    Target is era-varying ((i + j) % 5) so the per-era CORR series is
    non-degenerate — payout_report's deflated_sharpe requires finite skew/kurt,
    and an era-invariant target makes the series constant and raises ValueError.
    """
    rows: list[dict[str, float | str]] = []
    for i in range(1, 31):
        era = f"{i:04d}"
        downside = i > (30 - n_down)
        for j in range(4):
            pred = 0.1 + 0.1 * j
            target = float((i + j) % 5) / 4.0
            meta = -target + 0.5 * pred if downside else target + 0.5 * pred
            rows.append(
                {
                    "era": era,
                    "id": f"id{j}",
                    "prediction": pred,
                    "numerai_meta_model": meta,
                    "target": target,
                    "f1": float(j),
                }
            )
    full = pl.DataFrame(rows)
    return (
        full.select(["era", "id", "prediction"]),
        full.select(["era", "id", "numerai_meta_model"]),
        full.select(["era", "id", "f1"]),
        full.select(["era", "id", "target"]),
    )


def test_mmc_down_filtering() -> None:
    predictions, meta_model, features, targets = _mmc_down_frames(10)
    full = predictions.join(meta_model, on=["era", "id"]).join(
        targets, on=["era", "id"]
    )
    engine = EvaluationEngine("custom")
    mmc_by_era = engine.per_era_mmc(
        full, pred_col="prediction", meta_col="numerai_meta_model",
        target_col="target",
    )
    expected_down = [f"{i:04d}" for i in range(21, 31)]
    expected_value = float(np.mean([mmc_by_era[e] for e in expected_down]))

    score = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=None,
        features=features,
        targets=targets,
        n_trials=1,
        seed=7,
    )
    assert score.mmc_down == pytest.approx(expected_value)
    assert score.mmc_down_n_eras == 10
    assert score.mmc_down_reason is None


def test_mmc_down_insufficient() -> None:
    predictions, meta_model, features, targets = _mmc_down_frames(2)
    score = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=None,
        features=features,
        targets=targets,
        n_trials=1,
        seed=7,
    )
    assert score.mmc_down is None
    assert score.mmc_down_n_eras == 2
    assert score.mmc_down_reason == "insufficient_downside_eras"


def _turnover_scorecard_frames() -> tuple[pl.DataFrame, ...]:
    rows: list[dict[str, float | str]] = []
    for i in range(1, 26):
        era = f"{i:04d}"
        for j in range(12):
            pred = 0.1 * i + 0.01 * j
            rows.append(
                {
                    "era": era,
                    "id": f"id{j:03d}",
                    "prediction": pred,
                    "numerai_meta_model": pred * 0.5,
                    "target": float((i + j) % 5) / 4.0,
                    "f1": float(j % 5),
                }
            )
    full = pl.DataFrame(rows)
    return (
        full.select(["era", "id", "prediction"]),
        full.select(["era", "id", "numerai_meta_model"]),
        full.select(["era", "id", "f1"]),
        full.select(["era", "id", "target"]),
    )


def test_turnover_flows_into_scorecard() -> None:
    predictions, meta_model, features, targets = _turnover_scorecard_frames()
    score = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=None,
        features=features,
        targets=targets,
        n_trials=1,
        seed=7,
    )
    # pred is a constant per-era shift -> Spearman rho = 1 every transition
    assert score.turnover_mean == 0.0
    assert score.turnover_std == 0.0
    assert score.turnover_reason is None


def test_capital_metrics_flow_from_payout() -> None:
    predictions, meta_model, benchmarks, features, targets = _tiny_inputs()
    score = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=7,
    )
    engine = EvaluationEngine("custom")
    full = (
        predictions.join(meta_model, on=["era", "id"])
        .join(targets, on=["era", "id"])
    )
    corr = engine.per_era_corr(full, pred_col="prediction", target_col="target")
    mmc = engine.per_era_mmc(
        full, pred_col="prediction", meta_col="numerai_meta_model",
        target_col="target",
    )
    series = payout_series(corr, mmc)
    assert score.cagr_1y == pytest.approx(annual_compounded_return(series.clipped))
    assert score.gain_to_pain_ratio == pytest.approx(
        gain_to_pain_ratio(series.clipped)
    )
    assert score.kelly_fraction == pytest.approx(kelly_fraction(series.raw))
    expected_sim = simulate_overlapping_portfolio(series.clipped, horizon_eras=20)
    assert score.sim_portfolio_cagr == pytest.approx(expected_sim.portfolio_cagr)
    assert score.sim_portfolio_mdd == pytest.approx(
        expected_sim.portfolio_max_drawdown
    )
    assert score.sim_capital_utilization == pytest.approx(
        expected_sim.avg_capital_utilization
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_scorecard.py -q -k "mmc_down or turnover_flows or capital_metrics_flow"`
Expected: FAIL — `AttributeError: 'MetricScorecard' object has no attribute 'mmc_down'` (and `'cagr_1y'` / `'turnover_mean'` in the other tests) — the new fields don't exist yet, and the rewritten fixtures are non-degenerate so `evaluate_model` itself completes.

- [ ] **Step 3: Write the implementation**

In `nmr/scorecard.py`:

(a) Add imports (top of file):

```python
import numpy as np
from nmr.evaluation import (
    MIN_OVERLAP_ERAS,
    EvaluationEngine,
    NonVacuityError,
    downside_era_indices,
    per_era_turnover,
    sorted_era_labels,
)
```

Add the module-level constant near the imports:

```python
_MMC_DOWN_MIN_ERAS = 5
```

(b) Extend `MetricScorecard` — insert the 12 fields between `book_correlation` and `metric_timing_seconds`:

```python
    book_correlation: object | None

    cagr_1y: float
    gain_to_pain_ratio: float
    kelly_fraction: float
    mmc_down: float | None
    mmc_down_n_eras: int | None
    mmc_down_reason: str | None
    turnover_mean: float | None
    turnover_std: float | None
    turnover_reason: str | None
    sim_portfolio_cagr: float
    sim_portfolio_mdd: float
    sim_capital_utilization: float

    metric_timing_seconds: dict[str, float] | None
    eval_total_seconds: float
```

(c) Extend `to_frame()` — add after the `"book_correlation"` entry in `row`:

```python
            "cagr_1y": self.cagr_1y,
            "gain_to_pain_ratio": self.gain_to_pain_ratio,
            "kelly_fraction": self.kelly_fraction,
            "mmc_down": self.mmc_down,
            "mmc_down_n_eras": self.mmc_down_n_eras,
            "mmc_down_reason": self.mmc_down_reason,
            "turnover_mean": self.turnover_mean,
            "turnover_std": self.turnover_std,
            "turnover_reason": self.turnover_reason,
            "sim_portfolio_cagr": self.sim_portfolio_cagr,
            "sim_portfolio_mdd": self.sim_portfolio_mdd,
            "sim_capital_utilization": self.sim_capital_utilization,
```

(d) In `evaluate_model`, after the `mmc_by_era` block (right after `_mark("mmc_by_era", t0)`) insert:

```python
    t0 = time.perf_counter()
    meta_corr_by_era = evaluator.per_era_corr(
        base,
        pred_col=meta_col,
        target_col=main_target,
        era_col=era_col,
    )
    _mark("meta_corr_by_era", t0)

    t0 = time.perf_counter()
    downside_eras = downside_era_indices(meta_corr_by_era)
    mmc_down_n = len(downside_eras)
    if mmc_down_n >= _MMC_DOWN_MIN_ERAS:
        mmc_down_value = float(np.mean([mmc_by_era[e] for e in downside_eras]))
        mmc_down_reason = None
    else:
        mmc_down_value = None
        mmc_down_reason = "insufficient_downside_eras"
    _mark("mmc_down", t0)

    t0 = time.perf_counter()
    turnover_mean: float | None = None
    turnover_std: float | None = None
    if id_col in base.columns:
        turnover_by_era = per_era_turnover(
            base,
            pred_col=pred_col,
            era_col=era_col,
            id_col=id_col,
        )
        turnover_values = [
            turnover_by_era[e]
            for e in sorted_era_labels(list(turnover_by_era.keys()))
        ]
        if len(turnover_values) >= 2:
            turnover_mean = float(np.mean(turnover_values))
            turnover_std = float(np.std(turnover_values, ddof=0))
            turnover_reason = None
        else:
            turnover_reason = "insufficient_transitions"
    else:
        turnover_reason = "id column unavailable"
    _mark("turnover", t0)
```

(e) In the `return MetricScorecard(...)` block, after `book_correlation=None,` add:

```python
        cagr_1y=payout.cagr_1y,
        gain_to_pain_ratio=payout.gain_to_pain_ratio,
        kelly_fraction=payout.kelly_fraction,
        mmc_down=mmc_down_value,
        mmc_down_n_eras=mmc_down_n,
        mmc_down_reason=mmc_down_reason,
        turnover_mean=turnover_mean,
        turnover_std=turnover_std,
        turnover_reason=turnover_reason,
        sim_portfolio_cagr=payout.overlapping_sim.portfolio_cagr,
        sim_portfolio_mdd=payout.overlapping_sim.portfolio_max_drawdown,
        sim_capital_utilization=payout.overlapping_sim.avg_capital_utilization,
```

(`payout.overlapping_sim` is always non-None here because `payout_report` requires >= 2 eras; attribute access fails loudly otherwise.)

(f) Update the two test helpers — in `tests/test_registry.py` `_scorecard` and `tests/test_benchmark_gates.py` `_scorecard`, insert between `book_correlation=None,` and `metric_timing_seconds=None,`:

```python
        cagr_1y=0.0, gain_to_pain_ratio=0.0, kelly_fraction=0.0,
        mmc_down=None, mmc_down_n_eras=0, mmc_down_reason=None,
        turnover_mean=None, turnover_std=None, turnover_reason=None,
        sim_portfolio_cagr=0.0, sim_portfolio_mdd=0.0,
        sim_capital_utilization=0.0,
```

(g) Extend the existing column-contract test: Read `tests/test_scorecard.py` around `test_scorecard_to_frame_one_row_and_columns` (line ~176) and add the 12 names to its `required` set:

```python
        "cagr_1y",
        "gain_to_pain_ratio",
        "kelly_fraction",
        "mmc_down",
        "mmc_down_n_eras",
        "mmc_down_reason",
        "turnover_mean",
        "turnover_std",
        "turnover_reason",
        "sim_portfolio_cagr",
        "sim_portfolio_mdd",
        "sim_capital_utilization",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_scorecard.py tests/test_registry.py tests/test_benchmark_gates.py -q`
Expected: PASS.

Then run the determinism gates that exercise the scorecard cross-process:
Run: `./.venv/Scripts/python -m pytest tests/test_benchmark_slice1.py tests/test_benchmark_slice3.py -q`
Expected: PASS (two-run hash equality — new fields are deterministic, no pinned literals should break; if a pinned hash exists, surface it and report).

- [ ] **Step 5: Commit**

```bash
git add nmr/scorecard.py tests/test_scorecard.py tests/test_registry.py tests/test_benchmark_gates.py
git commit -m "feat: scorecard capital-readiness fields (MMC-down, turnover, sim)"
```

---

### Task 9: Public API exports

**Files:**
- Modify: `nmr/__init__.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: the seven new public symbols from Tasks 4–7.
- Produces: all seven importable from `nmr` and present in `nmr.__all__`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evaluation.py`:

```python
def test_public_api_exports_v25_capital_symbols() -> None:
    import nmr

    for name in [
        "OverlappingSimulationResult",
        "annual_compounded_return",
        "gain_to_pain_ratio",
        "kelly_fraction",
        "simulate_overlapping_portfolio",
        "downside_era_indices",
        "per_era_turnover",
    ]:
        assert name in nmr.__all__
        assert getattr(nmr, name) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_evaluation.py -q -k public_api`
Expected: FAIL — `AssertionError` (names missing from `nmr.__all__`).

- [ ] **Step 3: Write the implementation**

In `nmr/__init__.py`:

(a) In the `from .evaluation import (...)` block add `downside_era_indices, per_era_turnover`.
(b) In the `from .payout import (...)` block add `OverlappingSimulationResult, annual_compounded_return, gain_to_pain_ratio, kelly_fraction, simulate_overlapping_portfolio`.
(c) In `__all__`, add all seven names in their respective sections.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_evaluation.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nmr/__init__.py tests/test_evaluation.py
git commit -m "feat: export v2.5 capital metrics from nmr public API"
```

---

### Task 10: SSOT documentation & test counts

**Files:**
- Modify: `AGENTS.md` (2 test-count spots)
- Modify: `render_dataset_report.py` (test-count comment, line ~356)
- Modify: `ARCHITECTURE.md` (module registry + scorecard schema)
- Modify: `docs/06-evaluation/evaluation-suite-bible.md` (new v2.5 section)

**Interfaces:**
- Consumes: the new test count (measured, not guessed) and the finished v2.5 semantics from Tasks 1–9.
- Produces: docs consistent with the code (enforced by `tests/test_docs_hygiene.py`).

- [ ] **Step 1: Watch the hygiene gate fail (RED)**

Run: `./.venv/Scripts/python -m pytest tests/test_docs_hygiene.py -q`
Expected: FAIL — AGENTS.md claims 629 tests; the suite now collects more.

- [ ] **Step 2: Measure the real collection count**

Run: `./.venv/Scripts/python -m pytest --collect-only -q | tail -1`
Expected: e.g. `649 tests collected` (record the actual number `N`).

- [ ] **Step 3: Update the test-count claims**

In `AGENTS.md` (~line 33) and (~line 195), and in `render_dataset_report.py` (~line 356), replace `629` with the measured `N`:

```python
# AGENTS.md line 33: "Test: pytest (629 tests)."  ->  "Test: pytest (N tests)."
# AGENTS.md line 195: "full 629-test suite ..."    ->  "full N-test suite ..."
# render_dataset_report.py: "# 6. Tests (629-collection guard enforced by tests/test_docs_hygiene.py)"
#   -> "# 6. Tests (N-collection guard enforced by tests/test_docs_hygiene.py)"
```

- [ ] **Step 4: Update ARCHITECTURE.md**

`Grep -n "payout_report\|PayoutResult\|MetricScorecard" ARCHITECTURE.md` and read the surrounding sections, then update:

- The payout row of the module registry/table: note the four new helpers and the extended `PayoutResult` (CAGR/GPR/Kelly/sim).
- The evaluation row: note `downside_era_indices` and `per_era_turnover`.
- The scorecard schema section: add the 12 new columns and the `mmc_down`/`turnover` None-reason semantics.

- [ ] **Step 5: Update the evaluation bible**

Append to `docs/06-evaluation/evaluation-suite-bible.md` a section "v2.5 Capital-Readiness Metrics" covering, one paragraph each: CAGR 1Y (formula + ruin/-1.0 + n<2 contracts), GPR (formula + `+inf`/0.0 conventions), Kelly (raw-series rationale — Popoviciu compression), MMC-down (strict CORR_meta < 0, M_min=5, None+reason), turnover (Spearman on shared IDs, [0,2], skip <10, degenerate → 1.0), overlapping simulator (tranche semantics, pre-deployment recording, terminal at-cost convention), and the canonical-bytes inclusion rationale (deterministic fields are hashed; only timing stripped).

- [ ] **Step 6: Verify the gate passes (GREEN)**

Run: `./.venv/Scripts/python -m pytest tests/test_docs_hygiene.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add AGENTS.md render_dataset_report.py ARCHITECTURE.md docs/06-evaluation/evaluation-suite-bible.md
git commit -m "docs: SSOT updates for v2.5 capital-readiness metrics"
```

---

### Task 11: Pre-sign-off gate (verification only)

**Files:** none (verification task).

- [ ] **Step 1: Full suite**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: all pass, `0 failed` (the measured `N` tests from Task 10).

- [ ] **Step 2: Real-data benchmark smoke**

Run: `./.venv/Scripts/python benchmark_runner.py --fast-mode --output artifacts/benchmark_scores_smoke.csv --labels-output artifacts/benchmark_test_era_labels_smoke.csv`
Expected: clean exit; smoke CSVs regenerated; no regression in null-floor/gate assertions. If any benchmark scorecard row changed numerically, investigate before reporting green (only `timing_*`/`quality_metric_*` fields may differ; all other columns must match pre-change values).

- [ ] **Step 3: Report**

Report in the project review format: summary, affected files, approach, test evidence (both commands with real output), risks (Kelly saturation semantics, MMC-down vacuity on short windows, `+inf` GPR serialization).
