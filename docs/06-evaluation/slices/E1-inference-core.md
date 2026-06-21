# Work Slice E1 — Inference Core

> **Hand-off spec for the implementing engineer.** Parent spec: [../evaluation-suite-bible.md](../evaluation-suite-bible.md) §4 (and the API constraint it imposes from §7.4). This slice is self-contained; read it top to bottom. The agent specifies and grades; you implement. Grading is Green / Yellow / Red against the gates in §6 below — a single failed hard gate is Red.

## 0) Why this slice exists

Every Tier-1 decision scalar in the suite is wrapped by this layer before it is believed: block-bootstrap confidence intervals, an autocorrelation-adjusted Sharpe, and a multiple-testing-aware Deflated Sharpe. Nothing downstream (E2, E3, E4, E5, E6) can be built until these three primitives exist and are proven. There is no Numerai oracle for these — correctness is established by closed-form properties and seeded probes, not parity.

## 1) Scope

**In scope (this slice only):**
- New module `nmr/inference.py`.
- New test module `tests/test_inference.py`.
- Export the public surface from `nmr/__init__.py`.

**Out of scope (do not build here):** payout/downside (E2), BMC/CWMM (E3), robustness/perturbation (E4), scorecard (E5), book correlation (E6). No `statsmodels`, no `Optuna`, no new third-party dependency. Allowed imports: `numpy`, `scipy.stats` (`norm`, `skew`, `kurtosis`), Python stdlib. `polars` is **not** needed here — this layer operates on `numpy` arrays.

## 2) House conventions (match exactly)

- `from __future__ import annotations` at top; module docstring; explicit `__all__`.
- Result containers are `@dataclass(frozen=True)`.
- Validate at the boundary and raise `ValueError` with a precise message. No silent coercion.
- Determinism: use `numpy.random.default_rng(seed)` **only**. Never touch the global `numpy.random` state. No reliance on `set`/`dict` iteration order for any numeric path.
- Population standard deviation (`ddof=0`); when `std == 0`, any Sharpe is `0.0`.
- Must pass under `-W error` (the suite runs `pytest -q -W error`): no RuntimeWarnings from divide-by-zero, invalid sqrt, or NaN reductions. Guard them.

## 3) Public surface (exact signatures)

```python
@dataclass(frozen=True)
class SeriesStats:
    n: int
    mean: float
    std: float        # ddof=0
    sharpe: float     # mean/std, 0.0 if std==0
    skew: float       # sample skewness (bias-corrected); 0.0 if undefined
    kurt: float       # NON-excess kurtosis (normal == 3.0); 3.0 if undefined

@dataclass(frozen=True)
class BootstrapCI:
    point: float      # stat_fn on the original (un-resampled) data
    lo: float         # lower percentile bound
    hi: float         # upper percentile bound
    alpha: float
    n_boot: int
    block_len: int

def era_series_stats(series: ArrayLike1D) -> SeriesStats: ...

def resolve_block_len(n: int, horizon: Horizon, *, override: int | None = None) -> int: ...
def resolve_bandwidth(n: int, horizon: Horizon, *, override: int | None = None) -> int: ...

def block_bootstrap_ci(
    data: ArrayLikeND,                 # 1-D (n,) or 2-D (n, k); rows are eras
    stat_fn: Callable[[np.ndarray], float],
    *,
    block_len: int,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
    min_valid_frac: float = 0.5,
) -> BootstrapCI: ...

def ac_adjusted_sharpe(
    series: ArrayLike1D,
    *,
    horizon: Horizon | None = None,
    bandwidth: int | None = None,      # explicit K overrides horizon-resolved K
) -> float: ...

def deflated_sharpe(
    sharpe: float,
    *,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurt: float,                       # non-excess (normal == 3.0)
    trials_sr_var: float | None = None,
    sr0_benchmark: float = 0.0,
) -> float: ...
```

`Horizon = Literal["20D", "60D"]`. `ArrayLike1D` / `ArrayLikeND` = `Sequence[float] | np.ndarray` coerced via `np.asarray(..., dtype=float)`.

## 4) Function-by-function specification

### 4.1 `era_series_stats`

1. Coerce to a finite 1-D float array `x`. If any non-finite value is present, raise `ValueError`. If `x.size == 0`, raise `ValueError`.
2. `n = x.size`, `mean = x.mean()`, `std = x.std(ddof=0)`.
3. `sharpe = 0.0 if std == 0 else mean/std`.
4. `skew`: `0.0` if `n < 3` or `std == 0`, else `scipy.stats.skew(x, bias=False)`.
5. `kurt`: `3.0` if `n < 4` or `std == 0`, else `scipy.stats.kurtosis(x, fisher=False, bias=False)` (non-excess; normal ⇒ 3.0). The neutral defaults (skew 0, kurt 3) make a degenerate series collapse the DSR denominator to its Gaussian baseline.
6. Return `SeriesStats`. All fields native Python `float`/`int`.

### 4.2 `resolve_block_len` / `resolve_bandwidth` (horizon-floor clamp — bible §4.1–4.2)

- If `override is not None`: validate `1 <= override <= n` (block) / `1 <= override <= n-1` (bandwidth) and return it.
- Heuristics: block `h = round(n ** (1/3))`; bandwidth `h = floor(4 * (n/100) ** (2/9))`.
- **Floors (mandatory):** block — `5` for `"20D"`, `13` for `"60D"`; bandwidth — `4` for `"20D"`, `12` for `"60D"`.
- Return `min(max(h, floor), cap)` where `cap = n` (block) or `n-1` (bandwidth). **The floor wins over the heuristic.** If `n` is so small that the floor exceeds the cap, raise `ValueError` (the track is too short to bootstrap that horizon honestly).

### 4.3 `block_bootstrap_ci` (circular block, row-coherent — bible §4.1)

**Hard API constraint:** resampling is over the **era axis (rows)** of a 1-D **or** 2-D array; the *same* drawn block indices index **every column together**. This is what lets a conditional statistic (e.g. tail-conditional book correlation, §7.4 / E6) recompute its own mask **inside** `stat_fn` on each contiguous resampled replicate. Pre-slicing a non-contiguous conditional subset before bootstrapping is forbidden and is not this function's concern — it only guarantees coherent contiguous resampling.

Algorithm:
1. Coerce `data` to a finite float array of shape `(n,)` or `(n, k)`; `n = data.shape[0]`. Raise `ValueError` if `n == 0`, if `data` has non-finite values, or if `not (1 <= block_len <= n)`, or if `n_boot < 1`, or if `not (0 < alpha < 1)`.
2. `point = float(stat_fn(data))`.
3. `rng = np.random.default_rng(seed)`. `n_blocks = ceil(n / block_len)`.
4. For each of `n_boot` replicates, in order:
   - Draw `n_blocks` start indices `s ~ rng.integers(0, n)`.
   - Build the index vector by concatenating `(s + arange(block_len)) % n` for each start; truncate to length `n`.
   - `resampled = data[idx]` (works identically for 1-D and 2-D; rows stay aligned across columns).
   - `theta_b = float(stat_fn(resampled))`.
5. Collect `theta`; drop non-finite replicate values. If the valid fraction `< min_valid_frac`, raise `ValueError` (the statistic is too often degenerate to trust a CI). `point` itself must be finite or raise.
6. `lo, hi = np.percentile(valid, [100*alpha/2, 100*(1 - alpha/2)])`.
7. Return `BootstrapCI(point, lo, hi, alpha, n_boot, block_len)`.

Determinism: identical `(data, stat_fn, block_len, n_boot, seed, alpha)` ⇒ identical `BootstrapCI`, within a process **and across separate process invocations**.

### 4.4 `ac_adjusted_sharpe` (Lo 2002, Bartlett — bible §4.2)

1. Coerce to finite 1-D `x`; `n = x.size`. Raise `ValueError` if `n < 2` or non-finite.
2. `std = x.std(ddof=0)`; if `std == 0` return `0.0`. `sr = x.mean() / std`.
3. Resolve `K`: if `bandwidth is not None` use it (validate `1 <= K <= n-1`); elif `horizon is not None` use `resolve_bandwidth(n, horizon)`; else raise `ValueError` ("provide horizon or bandwidth").
4. Sample autocorrelations, biased normalization by the full centered sum of squares:
   $$\rho_k = \frac{\sum_{t=1}^{n-k}(x_t-\bar x)(x_{t+k}-\bar x)}{\sum_{t=1}^{n}(x_t-\bar x)^2},\quad k=1\dots K.$$
5. `D = 1 + 2 * sum_{k=1}^{K} (1 - k/(K+1)) * rho_k`. Clamp `D = max(D, 1e-12)` (no warning; Bartlett keeps `D>0` in theory, this guards finite-sample pathologies).
6. Return `sr / sqrt(D)`. By construction `D > 1` under net-positive autocorrelation ⇒ result `< sr` (hard gate §6).

### 4.5 `deflated_sharpe` (Bailey & López de Prado 2014 — bible §4.3, §5.0)

1. Validate `n_obs >= 2`, `n_trials >= 1`. Coerce numeric args to float.
2. **Cross-trial variance handling (no temporal fallback):**
   - `n_trials == 1` ⇒ `SR0 = sr0_benchmark` (no multiple-testing inflation; default 0.0). Ignore `trials_sr_var`.
   - `n_trials > 1` and `trials_sr_var is None` ⇒ **raise `ValueError`** (never substitute the single-series analytic variance — bible §4.3). Require `trials_sr_var > 0`.
   - `n_trials > 1` with valid `trials_sr_var` ⇒
     $$SR_0 = sr0\_benchmark + \sqrt{trials\_sr\_var}\,\Big[(1-\gamma_e)\,\Phi^{-1}\!\big(1-\tfrac{1}{N}\big) + \gamma_e\,\Phi^{-1}\!\big(1-\tfrac{1}{N e}\big)\Big]$$
     with `γe = np.euler_gamma`, `e = np.e`, `Φ⁻¹ = scipy.stats.norm.ppf`.
3. Radicand `r = 1 - skew*sharpe + (kurt - 1)/4 * sharpe**2`. If `r <= 0` raise `ValueError` (pathological moments for this Sharpe — surfaces a real data problem, do not clamp).
4. `DSR = norm.cdf((sharpe - SR0) * sqrt(n_obs - 1) / sqrt(r))`. Return `float` in `[0, 1]`.
5. **Caller contract (document in the docstring):** the `sharpe`, `skew`, `kurt` passed in MUST come from the **unclipped** raw series for the payout-proxy use case (bible §5.0); this function does not clip.

## 5) Reuse / boundaries

- Do **not** re-derive ranking/gaussianize math here — that lives in `nmr/_transforms.py` and is irrelevant to E1.
- `EvaluationEngine.summarize` already defines `mean/std/sharpe/max_drawdown` with the `ddof=0` and `std==0⇒sharpe=0` conventions; mirror those conventions exactly so the two modules agree.
- E1 has **no** I/O, no parquet, no config dependency. Pure functions over arrays.

## 6) Gates (acceptance — graded hardest first)

Hard gates (any failure ⇒ Red):

1. **G1 Bootstrap determinism.** Same `(data, stat_fn, block_len, n_boot, seed)` ⇒ bit-identical `BootstrapCI` on repeated calls. (Grader additionally verifies cross-process identity via a probe.)
2. **G2 AC direction.** On a seeded AR(1) series `x_t = 0.6 x_{t-1} + ε_t` (n≈600), `ac_adjusted_sharpe(x, horizon="20D") < era_series_stats(x).sharpe`. On a seeded i.i.d. series (φ=0), the two are equal within a small tolerance (e.g. `1e-2` relative).
3. **G3 Deflation monotonicity.** For fixed `sharpe, skew, kurt, trials_sr_var`, `deflated_sharpe` is **strictly decreasing** across `n_trials ∈ {1?→use ≥2}, 10, 100, 1000}` (compare `n_trials>1` values; also assert `n_trials=1` yields the largest, since SR0=benchmark).
4. **G4 Single-model guard.** `n_trials > 1` with `trials_sr_var=None` raises `ValueError`; `n_trials == 1` returns a value and ignores `trials_sr_var`.
5. **G5 Horizon-floor clamp.** `resolve_block_len(574, "60D") >= 13` and `resolve_bandwidth(574, "60D") >= 12`; `"20D"` floors `5` / `4`. `ac_adjusted_sharpe(..., horizon="60D")` uses `K ≥ 12`.
6. **G6 Row-coherent 2-D bootstrap (E6 enabler).** With `data` shape `(n, 2)` (candidate, book) and a `stat_fn` that masks to the book column's worst decile **inside** the function and returns the conditional correlation, `block_bootstrap_ci` returns a finite CI. Assert columns stay aligned: a `stat_fn` returning `corr(col0, col1)` on a perfectly correlated 2-col input returns ≈1.0 on every replicate (CI degenerate at 1.0), proving rows were not shuffled independently per column.
7. **G7 Boundary raises (non-vacuity).** Empty input ⇒ `ValueError`; `block_len` out of `[1, n]` ⇒ `ValueError`; non-finite input ⇒ `ValueError`; `deflated_sharpe` radicand ≤ 0 ⇒ `ValueError`. Each guard must be shown to actually fire (a test that triggers it), not merely exist.

Soft gate (Yellow if missing, not Red):

8. **G8 CI coverage sanity.** On i.i.d. `N(μ, σ²)`, across ≥200 seeds with bounded `n_boot` (e.g. 300), the 95% CI for the mean covers `μ` in roughly `[0.90, 0.99]` of trials. Keep runtime modest; this is a calibration check, not a proof.

Global:

- **Full suite stays green** under `.\.venv\Scripts\python -m pytest -q -W error` (141 existing + new).
- Public names exported from `nmr/__init__.py`; `__all__` updated in both modules.
- No new dependency; no global RNG; no warning under `-W error`.

## 7) Definition of done (Green)

All hard gates G1–G7 pass, G8 within band, full suite green under `-W error`, signatures exactly as in §3, frozen result dataclasses, precise `ValueError` boundaries, docstrings stating the unclipped-series caller contract for `deflated_sharpe` and the row-coherence guarantee for `block_bootstrap_ci`. The grader will additionally run an independent probe to `artifacts/_probe.py` (cross-process determinism, AR(1) direction on real-shaped n, and the 2-D row-coherence check) and will not accept the suite passing as sufficient evidence on its own.

## 8) Suggested implementation order (for the engineer)

1. `era_series_stats` + its edge tests (fastest feedback).
2. `resolve_block_len` / `resolve_bandwidth` + floor tests.
3. `ac_adjusted_sharpe` + AR(1) direction test (G2).
4. `deflated_sharpe` + monotonicity & guard tests (G3, G4).
5. `block_bootstrap_ci` + determinism, 2-D coherence, boundary, coverage tests (G1, G6, G7, G8).
6. Wire `__init__.py`, run the full suite under `-W error`, then hand back for grading.
