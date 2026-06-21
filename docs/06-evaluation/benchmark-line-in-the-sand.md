# Benchmark "Line in the Sand" — Null Baselines & the S11 Ladder

> **Purpose of this file:** a standing memory aid so we do not forget the *floor* every metric and every model must clear. This is the cheap, brutal sanity layer that sits **underneath** the whole evaluation suite. It is referenced by the evaluation bible §11 (E6 gate) and §15 (deferral ledger), and it couples to the S11 `BenchmarkSuite`. Build it alongside E5/E6 — not after.

## 1) The one idea

Before we trust *any* number the suite produces, we prove that **worthless predictions score worthless**. If a constant or random "model" can post a non-trivial CORR, payout, MMC, BMC, or FNC, then the **metric is broken** (or leaking), not impressive. The null baselines are the line in the sand: every real model must clear them, and every metric must collapse to its null value on them.

Two failure modes this catches:
- **A broken metric** — a scoring bug, a sign flip, a leak, a normalization error — surfaces as a null baseline scoring above its floor.
- **A worthless model dressed up** — anything that cannot beat constant/random is not a model.

## 2) Null / trivial baselines (the floor roster)

All seeded and deterministic (fixed by data-version + seed, independent of any model under test).

| Baseline | Definition | Expected result on every metric |
| --- | --- | --- |
| **constant-0.5** | every prediction = 0.5 (or any constant) | CORR = 0, MMC ≈ 0, BMC ≈ 0, FNC = 0, payout ≈ 0; std=0 ⇒ Sharpe = 0; rank stability **well-defined, not NaN** |
| **uniform-random** | i.i.d. `U(0,1)` per row, seeded | CORR ≈ 0, MMC ≈ 0, BMC ≈ 0, FNC ≈ 0, payout ≈ 0 (within bootstrap CI of 0) |
| **gaussian-random** | i.i.d. `N(0,1)` per row, seeded | same as uniform-random — ≈ 0 on every skill metric |

**Non-negotiable rule:** a degenerate/constant prediction must yield **0.0, never NaN**, under `-W error` (already enforced in E2 G6 and the engine short-circuit). The null baselines re-verify this end-to-end through the full scorecard.

## 3) The full S11 ladder (rungs, low → high)

Every rung emits a **complete scorecard** (Tier-1 + Tier-2 at minimum), so we can read the gradient, not just a pass/fail:

1. **Null** — constant-0.5, uniform-random, gaussian-random (§2). *Floor.*
2. **Trivial** — single-feature predictor, mean-of-features. Should be barely above null; exposes any metric that rewards trivial feature beta.
3. **Linear** — a plain linear/ridge model on the feature set. The "are you even trying" bar.
4. **Tree** — a basic GBDT (default-ish params). The realistic "did your fancy model beat a default tree" bar.
5. **Numerai benchmark models** — the real upper reference. **BMC is measured against these** (stake-weighted on the leaderboard; **highest-staked single benchmark in Diagnostics**, per canon `01-canon/scoring/02-mmc-bmc.md`). Beating these is the actual goal.

A candidate's scorecard is only interesting **relative to where it lands on this ladder**.

## 4) Hard gates (these are the "lines")

- **G — Null floor on every metric (incl. new ones).** constant-0.5 / uniform-random / gaussian-random ⇒ CORR≈0, payout≈0, FNC≈0, MMC≈0, BMC≈0, and **well-defined** (non-NaN) rank stability, max-drawdown, burn rate, CVaR, book correlation. Any null baseline scoring meaningfully above 0 on a skill metric ⇒ **Red** (broken metric).
- **G — Every ladder rung emits Tier-1 + Tier-2.** No rung is allowed to skip metrics; the ladder must be directly comparable row-for-row with a real candidate's scorecard.
- **G — Monotone sanity.** null ≤ trivial ≤ linear ≤ tree ≤ benchmark on the headline rank scalar (the Deflated Payout Proxy), within CI. A gross inversion (e.g. random > tree) ⇒ investigate before trusting the suite.
- **G — Determinism.** Same data-version + seed ⇒ identical baseline scorecards across process invocations.

## 5) Reference numbers (sanity anchors, not gates)

- **"Good" mean CORR ≈ 0.01–0.03** (archive bible). A null baseline near this is a red flag; a real model near/above this is plausible.
- **Purge convention: 8 eras (20D), 16 eras (60D)** — the benchmark walk-forward convention; also usable as a conservative bootstrap block-length floor.
- **Payout proxy** floors at ~0 for null because `0.75·CORR + 2.25·MMC` ⇒ 0 when both are 0.

## 6) Where this lives in the build

- **Specced, deferred to E5/E6** (bible §15 deferral ledger). The `BenchmarkSuite` (S11) is the home; E6 wires the full scorecard into every baseline and adds the null-floor gate.
- **Couples to E6** because the new Tier-3 metrics (book correlation, BMC, CWMM) must *also* floor on the null baselines — not just the Tier-1 metrics.
- **Build alongside E5/E6, not after** — the scorecard aggregator (E5) and the baselines (S11) are mutually validating: the baselines are the first real consumers of the scorecard, and they prove the scorecard composes correctly.

---

*Memory aid only. The authoritative spec is `docs/06-evaluation/evaluation-suite-bible.md` (E6 gate §11, deferral ledger §15). If a floor changes, change it in the bible first, then here.*
