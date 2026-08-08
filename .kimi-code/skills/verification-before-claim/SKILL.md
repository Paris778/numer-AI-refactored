---
name: verification-before-claim
description: Use when about to report any agent change to the nmr framework as done — before claiming tests pass, before committing, or when finishing any data, evaluation, or scorecard-touching work
type: prompt
disableModelInvocation: false
---

# Verification Before Claim (S4)

**Core principle:** no "done" without evidence. Run the gates, capture the output, then claim.

## When to Use
- Ending any task in this repo, before reporting success to the requester.
- Any change touching data loading, evaluation metrics, scorecards, or canonical hashes.

## Checklist (all items required)

| # | Gate | How | Fails if |
|---|---|---|---|
| 1 | Full suite green | `.\.venv\Scripts\python -m pytest -q` (repo root) | Any failure, error, or unexplained skip |
| 2 | Benchmark smoke | `benchmark_runner.py --fast-mode --output artifacts/benchmark_scores_smoke.csv --labels-output artifacts/benchmark_test_era_labels_smoke.csv` for any data/evaluation/scorecard change | Error exit or empty outputs |
| 3 | Canonical-hash purity | Any new scorecard/instrumentation field triaged into canonical-vs-excluded (`canonical_scorecards_bytes`) | Timing/path fields inside canonical bytes → `test_benchmark_slice1/3` flake |
| 4 | Parity | Metric changes update `tests/test_parity.py` + `tests/test_risk_parity.py` | Custom metric diverges from `numerai_tools.scoring` |
| 5 | Purge gate | `purge_eras >= 8` (20D targets) / `>= 16` (60D) in every config | Below threshold — protocol-enforced: code only checks the configured gap, it does not block weakening |
| 6 | Determinism | Seed threading via config (`run.seed` → `set_global_seeds`); `deterministic=True`/`force_col_wise=True` preserved | Unseeded stochastic ops; weakened flags |
| 7 | Boundary | No business logic in scripts/notebooks; logic in `nmr/` only | Formula/transform/validation rule in a control plane |
| 8 | Doc SSOT | Docs updated in the same change-set (AGENTS.md Self-Update Directive; ARCHITECTURE.md §P–§R for these skills) | AGENTS.md/ARCHITECTURE.md/README.md/CONTRIBUTING.md drift |
| 9 | Artifact trust | cloudpickle loads only from `artifacts/` | Loading untrusted `.pkl` |
| 10 | Device-aware comparison | `oof_device` logged and checked before any cross-run numeric comparison | GPU vs CPU OOF compared silently |

## Hard Rules
- [ ] Never claim tests passed without executing them and reading the output.
- [ ] Never suppress or hand-wave a failure — investigate (systematic-debugging) or report it.
- [ ] Canonical hashes exclude wall-clock timing and absolute paths.
- [ ] Purge thresholds are protocol constants, not negotiable config values.
- [ ] Never auto-promote; never fabricate results.

## Common Mistakes
| Mistake | Fix |
|---|---|
| "I ran the tests mentally" | Execute, capture, quote |
| New scorecard field without canonical triage | Triage first — determinism tests will catch it |
| Dropping `purge_eras` "for more data" | Leakage is a correctness bug (AGENTS.md §2.4) |
| Comparing runs across devices | Check `oof_device` first (see run-meta-analysis) |
| Deleting a field that was canonical | Re-check `canonical_scorecards_bytes` before removing anything |
