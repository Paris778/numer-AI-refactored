# Design Spec: Money-Making Program — Research OS to 90+/100

> Status: ACTIVE (director's program of record, 2026-09-12). Phase 0 close-out is
> the only authorized work until its exit criteria pass. This record holds the
> north star, risk envelope, defect register, and phase-gate structure. Per-phase
> step detail converts to task-level implementation plans at execution time, and
> those plan files are deleted on completion — `docs/superpowers/plans/` must stay
> empty (`tests/test_docs_hygiene.py`, T8).
>
> Authority: supersedes `2026-09-07-model-os-design.md` §2 #9 (effort split) and
> its "no dashboard expansion" decision only where the phases below explicitly
> authorize new work. All other Model OS decisions remain binding: no new serious
> model family without director disposition; do not rewrite
> `PurgedEraSplitter`, neutralization math, or oracle metrics; do not reopen the
> gate Sharpe pin (0.5358); composition math stays rank → blend → neutralize →
> final rank.
>
> Evidence baseline: verified 2026-09-12 14:48 local. Calibration digest
> `7f55db6032d3824340cb18bb64d82b4e0fbec822d039e10fc25da46f888dcdf9` (family-wise
> quantile, q=0.99, 1000 seeds, threshold 0.4188512614071038, generated 13:35);
> gate report written 14:23; fleet PID 18908 mid-run (started 13:38, ETA ≈16:20);
> stored receipt still the 2026-09-09 red copy; worktree 13 modified + 4 untracked.

## 0. Mission

Build the system that produces money-making Numerai Classic models: deterministic,
leakage-safe, gate-guarded, and fast to iterate. The absence of a competitive
model today is expected at this stage — it is not a weakness. The system is the
asset; the model is the output of the system. Terminal goal: **train models that
make money** and never stake one that has not repeatedly survived the gates.

## 1. Definition of 90+/100 (binding rubric)

Two scores per area (A–N of the audit scorecard): Engineering integrity and Money
readiness. **Money readiness ≥ 90 requires all of:** one pre-registered candidate
with capital evidence; passing unchanged Tier-4 gates with documented margin; an
accepted deployment artifact; an official-scorecard cross-check inside tolerance;
live round monitoring with a kill-switch. Infrastructure alone can never score 90
on money readiness.

Baseline: Eng 79.7 / Money 53.6. Targets at phase exit: P1 84/56, P3 88/62,
P5 90/70, P7 92/88, P8 93/90.

## 2. Verified baseline (2026-09-12)

| Fact | Status | Evidence |
|---|---|---|
| Calibration artifact exists | VERIFIED BY EXECUTION | `configs/benchmarks/null_floor_calibration.json` (untracked), digest `7f55db6032d3…` |
| Tier-0 nulls pass under the calibrated gate | VERIFIED BY EXECUTION | `artifacts/reports/benchmark_gate_report.csv` 14:23; all four null kinds pass, fingerprint-bound |
| Tier-4 row passes | VERIFIED BY EXECUTION — reference row only | `v53_lgbm_ender60`: corr 0.02927014311764319 vs 0.0286; AC-Sharpe 0.5358798441316628 vs 0.5358 (margin +0.0000798); FNC 0.02727794845999274 vs 0.02; GPR 47.76278056274705 vs 1.5 |
| No candidate clears the bar | VERIFIED BY STORED ARTIFACT | Best locked validation is CatBoost: corr 0.007742443210053531, AC-Sharpe 0.2089210197502235, FNC 0.007783069728297194, GPR 1.7314793708770426 (n_eras 86, target_ender_60, 60D) |
| Generalization collapse is real | VERIFIED BY STORED ARTIFACT | XGB ensemble: OOF corr 0.03291252222396539 → val 0.0006087025879073467; CatBoost: OOF 0.01956983474790033 → val 0.007742443210053531 |
| Stored receipt is stale and red | CONTRADICTED | `artifacts/reports/real_data_gate_receipt.json`, 2026-09-09, `all_passed=false`, benchmark exit 1 |
| No capital evidence, exports, or pointers | CONTRADICTED | Five run records; zero `capital_evidence`, exports, authorization receipts, champion/current pointers |
| One-off harness absent | VERIFIED ABSENT | no `run_one_off` in the repository |
| Numerai API/MCP integration absent in code | VERIFIED ABSENT | no MCP config; `nmr/` never calls numerapi; canon exists at `docs/01-canon/11-api-and-mcp.md`; `TODO-NOTES.md` requests MCP |
| Docs budget | VERIFIED | `AGENTS.md` is 32,767 B against a 32,768 B budget — trim before adding |

Diagnosis note (INFERRED): the 2026-09-09 `Null floor violation for
null_uniform_rand.corr` occurred under the retired single-seed floor. The same
null measures -0.001683333012217118 against the current 0.005 tolerance — a gate
miscalibration, not a pipeline defect.

## 3. Defect register (close-out items)

| ID | Defect | Location | Acceptance test |
|---|---|---|---|
| D1 | Export validity ignores the immutable authorization receipt | `nmr/lifecycle.py:294` | Deleting or tampering `promotion_authorization.json` makes `valid_export` return None and blocks pointer resolution |
| D2 | 60D HPO computes AC-Sharpe with 20D bandwidth; selected winner labelled `selection_bias=False` | `nmr/research.py:516`, `nmr/research.py:161` | 60D configs use 60D bandwidth; post-selection results carry `selection_bias=True` and are capital-ineligible |
| D3 | Calibration study hygiene: statistical logic in a top-level script, `generated_at` inside the digest, no resume checkpoints | `null_floor_study.py:393` | Cross-time digest equality; interruption/resume reproduces the uninterrupted output; logic moved into `nmr/` |
| D4 | Assurance gaps: full pytest never completed; coverage receipt dated 2026-08-19 with `branch_coverage=false`; mutation covers only `nmr/splitter.py` | `coverage.json`, `scripts/mutation_gate.py` | Full suite green with classified skips; branch coverage on the final state; mutation breadth across seven modules |
| D5 | CatBoost hosted-runtime availability unverified | `nmr/deployment.py`, `nmr/model_backend_catboost.py` | Hosted validation, or deployable adapters restricted to LightGBM/XGBoost |

## 4. Phase map

```text
PHASE 0 close-out ─► PHASE 1 Research OS ─► PHASE 2 training pipeline
                                   │                    │
                                   ▼                    ▼
                          PHASE 3 gates ◄──── PHASE 4 dashboard/monitoring
                                   │
                    ┌──────────────┴───────────────┐
                    ▼                              ▼
          PHASE 5 Numerai loop            PHASE 6 research program
                    └──────────────┬───────────────┘
                                   ▼
                          PHASE 7 capital path
                                   ▼
                          PHASE 8 certification
```

Phases 1 and 2 may run in parallel after Phase 0. Phase 6 is gated on Phases 3
and 5. Each phase closes only on receipt-backed evidence.

## 5. PHASE 0 — Charter, current-cycle close-out, defect register

**Objective:** convert today's calibration cycle into committed, receipt-backed
truth, and ratify the north star, risk envelope, and doctrine. No code changes
beyond close-out.

**Steps**

1. Freeze the calibration cycle: wait for the fleet to finish and
   `real_data_gate_receipt.json` to be rewritten with `all_passed=true`. Commit
   `configs/benchmarks/null_floor_calibration.json` (currently untracked — the
   gate cannot reproduce on another checkout without it), `null_floor_study.py`,
   `freeze_ridge_reference.py`, and the 13 modified tracked files. Acceptance:
   green receipt; clean tree; digest recorded in `ARCHITECTURE.md` §M.
2. Ratify the KPI stack. Validation: CORR, AC-Sharpe (60D), FNC, MMC, GPR, payout
   proxy, CVaR5, max drawdown, turnover. Ops: receipt success rate, gate pass
   rate, run reproducibility. Live: official round CORR/MMC, live-vs-local delta,
   rounds since upload.
3. Write the risk envelope: max drawdown, turnover band, per-model capital
   ceiling, crowd-correlation ceiling, kill-switch rules. Home: `docs/02-strategy/`.
4. Codify research doctrine: pre-registration before every candidate; gates guard
   promotion; no threshold edits without recorded evidence; no hero models; `nmr/`
   is the only tested boundary.
5. Trim `AGENTS.md` to ≤30 KB before any addition; move reference detail to
   `ARCHITECTURE.md`. Acceptance: `tests/test_docs_hygiene.py` green.
6. Open the defect register (§3) with owners and due phases.
7. Publish phase exit criteria as the program contract.

**Expected artifacts:** green real-data receipt; committed calibration; strategy
risk envelope; trimmed `AGENTS.md`; defect register; signed charter.

**Success criteria:** receipt `all_passed=true`; tree clean; calibration
reproducible from the committed study script; charter signed; zero threshold
changes during close-out.

**Guardrails:** do not rerun the fleet twice; never hand-edit the calibration
JSON; never touch `experiments/*/runs/`.

## 6. PHASE 1 — Research OS

**Objective:** any researcher can take an idea — including a model trained
outside this repository — to a persisted, reproducible, gate-labelled result
without touching core code.

**Steps**

1. Build the one-off harness: `nmr/oneoff.py` (pure logic) plus `run_one_off.py`
   (thin CLI). Foreign three-column parquet → `PredictionSet` with explicit
   provenance → optional `compose_predictions` → research-labelled scorecard →
   persisted run record. The capital stage is refused by construction (no
   `CapitalContext` exists outside the trusted loader). Acceptance: a synthetic
   external parquet is scored and re-read with zero edits to `nmr/` core.
2. Per-run gate receipts: extend `nmr/experiment_store.record_run` to persist a
   `gate_receipt` block (config hash, code identity, `data_fingerprint`,
   `promotion_data_fingerprint`, gate version, calibration digest, Tier-4 row and
   margins, verdict) via atomic writes. Acceptance: deleting the receipt
   invalidates gate status; tampering fails closed.
3. Experiment lifecycle states: extend `nmr/lifecycle.derive_stage` with
   `draft → running → completed → gated_passed/failed → archived/deployed`,
   consumed by registry and dashboard. A run cannot be `deployed` without
   `gated_passed`.
4. Orchestration object: encode hierarchy → fleet → hard-failure → receipt as one
   resumable workflow with stage markers, completed/total counts, elapsed time,
   and per-stage checkpoints. Map queues onto existing primitives: scout =
   `run.workflow="gpu_screen"`, confirm = `cpu_confirm`, production = CPU deploy
   fits. Acceptance: kill mid-run, resume, byte-identical outputs.
5. Config discipline: prove a run can be re-executed from its own stored manifest
   alone; ban flag-driven training in scripts.
6. Determinism standing tests: extend the seed-42 anchor into a committed
   determinism receipt covering `run_id`, OOF digest, and scorecard digest
   cross-process.
7. Close D1 and D2 with red-before/green-after tests.
8. Harden D3: move calibration statistics into `nmr/` (for example
   `nmr/null_floor.py`), remove `generated_at` from the digest, add batch
   checkpoints and progress markers, and a time-invariance test.
9. Runbook: fleet ETA expectations, `nohup` and log discipline, thread caps at
   process start (`apply_thread_limits`), Modern Standby settings, RAM guards,
   junction/worktree hazard — one page in `CONTRIBUTING.md`.

**Expected artifacts:** `nmr/oneoff.py`, `run_one_off.py`, receipt schema in
`run.json`, lifecycle states, orchestration module, D1–D3 closures, runbook.

**Success criteria:** external model scored end-to-end without core edits; every
persisted run carries a verifiable gate receipt; D1–D3 tests green.

**Guardrails:** no new dependencies; receipts carry no timing or absolute paths.

## 7. PHASE 2 — Model training pipeline

**Objective:** make training deterministic, provenance-complete, and modular — a
new model becomes a checklist, not a surgery.

**Steps**

1. Content identity for all runs: adopt `promotion_data_fingerprint` as the
   run-level snapshot marker with bounded, cached hashing on data write; keep the
   structural fingerprint for cheap identity. Acceptance: same-shape value edits
   change run identity; a refresh changes it exactly once.
2. Schema contracts and drift detection in `nmr/data.py`: dtype, column, and era
   coverage changes fail loudly before training.
3. Feature registry: extend `features.json` entries with definition, source,
   dependencies, neutralization rule, screen metrics, provenance hash;
   `derived_feature_sets.json` already provides derived-set provenance.
4. Transform contract: property-test `nmr/_transforms.py` for era-independence
   and keep it numpy/scipy-only (embedded by value into deploy artifacts).
5. Backend intake checklist: document and test the three-step path (adapter →
   registry entry → identity fingerprint) using `nmr/model_backend_registry.py`
   and `experiments/custom-registry-test`; template in `CONTRIBUTING.md`.
6. Scout → scale promotion policy: scout = `small`/`medium` features, `fast`
   preset, `gpu_screen`; scale = full train eras, `standard`; only scout-gated
   configs may scale. Encoded in `nmr/campaign.py` / `run_campaign.py`.
7. Training telemetry: feature importance, per-fold timing, per-era IC persisted
   under the run artifact directory — with timing excluded from every hashed
   payload.
8. Dependency upgrade protocol: one library at a time, exact pin, with oracle
   parity, determinism, and a fresh real-data receipt before merge (addresses the
   `TODO-NOTES.md` upgrade item).

**Expected artifacts:** content-fingerprint migration, schema contract tests,
feature registry, backend template, scout/scale policy, telemetry schema,
upgrade receipts.

**Success criteria:** a new backend lands in one PR touching two files plus tests;
a scout config cannot silently scale; refresh → fingerprint change → new `run_id`
is test-verified.

**Guardrails:** the `all` universe (3,555 features) stays prohibited for routine
iteration; memory guards untouched; CatBoost stays CPU-only.

## 8. PHASE 3 — Evaluation pipeline and gates

**Objective:** every number that authorizes money comes from a gate whose
threshold has provenance, whose margins are explicit, and whose failure modes are
diagnosed rather than argued away.

**Steps**

1. Gate versioning: stamp a gate version plus calibration digest into every gate
   report and run receipt; bump on any threshold-class change.
2. Candidate bar = reference + margin: implement `capital_margin` per metric
   (recommended: CORR ≥ ref + 0.002, AC-Sharpe ≥ ref + 0.05, FNC ≥ ref + 0.002,
   GPR ≥ ref + 5.0) with documented rationale — the reference row currently clears
   its own thresholds by +0.00067 CORR and +0.0000798 Sharpe, a refresh-fragile
   margin. Acceptance: a candidate equal to the reference is rejected.
3. Threshold provenance and refresh procedure: bind config thresholds to the
   stored reference receipt by test; document who may refresh, on what evidence,
   with a version bump.
4. Fleet as a reusable grid with selection rules: keep the 19 cells across the
   four fleet configs; add deterministic champion selection with robustness
   constraints; permanently exclude `fa_v151_ridge_ensemble` from naive
   comparisons; use `--only-fleet` / `--rungs-csv` for fleet-only iteration.
5. Null-floor integrity: keep the calibration-bound gate; add a dataless-CI-safe
   loader stub so refusal tests never silently skip in CI; keep the recalibration
   procedure in `docs/06-evaluation/benchmark-line-in-the-sand.md`.
6. Multiple-testing accounting: document the family-wise burden (19 fleet cells ×
   2 scored fields per tier, plus HPO trials); pre-register every candidate
   destined for promotion.
7. Financial-failure mode tests: strong CORR with negative payout/CVaR, or
   degenerate turnover, must be refused.
8. Audit command: `scripts/audit_run.py <run_id>` reconstructs data fingerprints,
   config, code identity, gate receipts, and official comparison from disk alone.

**Expected artifacts:** margin policy, gate-version field, refresh procedure, fleet
selection rules, `audit_run.py`, expanded gate tests.

**Success criteria:** every threshold traceable; margin policy test-enforced; a
stale threshold change impossible without a version bump; audit command
reproduces a verdict.

**Guardrails:** never change a threshold to make a candidate pass; the 0.5358 pin
stays closed.

## 9. PHASE 4 — Dashboard and monitoring layer

**Objective:** a director answers "what is running, what passed, what is stale,
what is broken" in under a minute without reading logs.

**Steps**

1. Unified-schema extension in `nmr/dashboard.py` and `dashboard_ui/`: gate
   receipt fields, lifecycle state, fleet placement, official-vs-local deltas.
   Keep the zero-dependency static report portable; Streamlit stays a thin host
   and never enters `nmr/`.
2. Run/gate status view: current phase, ETA, last completed stage, receipt
   freshness, red/green per gate — sourced from receipts, not log scraping.
3. Receipt table with filters and diff-versus-prior.
4. Model performance view: CORR by era, AC-Sharpe, drawdown, FNC/MMC, turnover;
   fleet side-by-side with robustness versus raw performance.
5. Infra health: runtime distributions (hierarchy ≈42 min, fleet ≈1.9 h), receipt
   success rate, RAM/disk guards, and process liveness — a stalled overnight job
   must surface as an alert-class state (the Modern Standby incident cost 9.4 h).
6. Staleness enforcement: extend the existing regeneration rule into a test that
   fails when the dashboard window is stale relative to the meta overlap.
7. Experiment explorer: index from `run.json` plus receipts, with a per-family
   narrative (hypothesis → method → results → decision → next) under
   `docs/04-research/`.

**Expected artifacts:** extended dashboard schema, status/receipts/health views,
staleness test, experiment index.

**Success criteria:** the dashboard alone answers current fleet state, last green
receipt, best candidate versus Tier-4 margins, and any stale or stalled job;
static report regenerates deterministically.

**Guardrails:** no new charting dependency.

## 10. PHASE 5 — Numerai integration and official-scorecard cross-checks

**Objective:** close the loop between local evaluation and the tournament's own
numbers, so the system cannot drift from the thing that pays.

**Steps**

1. Credential and scope hygiene: a dedicated API key scoped to upload submissions
   and pickled models, download previous submissions, view historical submission
   info, and view user info — never staking or deletion scopes. `.env` stays
   git-ignored; `nmr/` never reads credentials; thin CLIs read environment
   variables only; never print or log key material.
2. Install Numerai MCP for research agents per `docs/01-canon/11-api-and-mcp.md`:
   endpoint `https://api-tournament.numer.ai/mcp` (stateless MCP over Streamable
   HTTP), auth header `NUMERAI_MCP_AUTH` in the form
   `Token PUBLIC_ID` + escaped `$` + `SECRET_KEY`, installed via the official
   one-liner or manual client configuration. Canon lists Codex CLI (recommended),
   Cursor, and Claude Code; for VS Code, configure the MCP client with the same
   endpoint and header, then verify by querying the current round and model
   performance. This closes `TODO-NOTES.md` item 1.
3. Official-client module `nmr/official.py`: pure logic, no I/O and no
   credentials — normalize official scorecards into local metric names (CORR, MMC,
   FNC, AC-Sharpe, payout), align era windows, compute deltas. Network lives in
   thin CLIs: `pull_official_scorecard.py` (writes
   `artifacts/reports/official/<model>/<round>.json`) and `upload_submission.py`
   (submits the accepted artifact or prediction CSV; `--dry-run` by default).
4. Round-score configuration binding: bind `scoring_target`, `scoring_horizon`,
   and payout policy identity to the round's declared configuration; a
   horizon/multiplier change invalidates the comparison or forces explicit
   re-identification — never passes silently.
5. Tolerance bands: per-metric acceptance deltas (start: |ΔCORR| ≤ 0.002,
   |ΔMMC| ≤ 0.002, |ΔFNC| ≤ 0.002, |ΔSharpe| ≤ 0.05) with era-window alignment
   rules and a discrepancy report on breach. Document that local validation is an
   era-window estimate, not a live round series.
6. Live monitoring loop: after each upload round, pull the official scorecard,
   record it under the export's lineage, and plot live versus expected; suspend
   when the live delta breaches the band for N consecutive rounds.
7. Provenance closure: every exported model reconstructible end-to-end — data
   fingerprints → config → code identity → gate receipt → artifact digest →
   official round results.

**Expected artifacts:** MCP configured for the agent environment,
`nmr/official.py`, two CLIs, round-binding logic, tolerance-band config,
discrepancy report, live monitoring records.

**Success criteria:** an agent queries the current round via MCP; a completed
upload yields an official scorecard pulled back with a delta report against the
local capital scorecard; horizon/policy drift cannot pass silently.

**Guardrails:** no staking automation; no credentials in code, logs, tests, or
chat; uploads only for artifacts that passed the acceptance gate and only with
director approval.

## 11. PHASE 6 — Model research program

**Objective:** systematically close the generalization gap, then produce
pre-registered candidates that beat the reference with selection bias controlled
by construction.

**Steps**

1. Baseline consolidation: lock three reproducible canaries (LightGBM medium,
   CatBoost ender60, Ridge medium) with committed configs and full gate receipts.
2. Diagnose the OOF → validation collapse (highest-value investigation). Evidence:
   XGB OOF 0.03291252222396539 → val 0.0006087025879073467; CatBoost OOF
   0.01956983474790033 → val 0.007742443210053531. Hypothesis register, each
   dispositioned with a test: overfitting to train-era structure; residual
   target-overlap effects on the last folds; 60D distribution shift; OOF-guided
   hyperparameter selection; meta-model/benchmark drift across the validation
   window. Acceptance: a written cause ranking with evidence.
3. Pre-registration protocol: hypothesis, config, metrics, decision rule, and
   stopping rule declared before the run; registered in the experiment explorer.
4. Ensembling program: rank-domain blending across diverse GBMs plus linear
   components (`nmr/ensemble.py`), purged-fold weight learning, and the
   neutralization frontier (`nmr/research.py`) — selection-bias labels carry
   through to scorecards.
5. Feature program: campaigns via the feature skill (`.kimi-code/skills/`),
   train-only stability screens, drift/PSI/Wasserstein diagnostics, derived
   subsets — within the `medium`/`small` policy.
6. Target/horizon program: ender20 versus ender60 cross-checks and multi-target
   ensembles with horizon-stability diagnostics; horizon shopping forbidden by
   pre-registration and identity preservation.
7. Risk-aware objectives: payout-aware and neutralization-aware objectives,
   turnover and capacity constraints, drawdown-sensitive weighting — evaluated
   through the Phase 3 gates including the financial-failure tests.
8. Multiple-testing budget: cap candidates per cycle, track family-wise error,
   and allow at most one pre-registered candidate per cycle to reach capital
   evidence.
9. Escalation gate for new families: any new architecture class (stacking
   meta-learners, foundation-model approaches) requires a written
   dependency-exception request plus a GPU/RAM plan.

**Expected artifacts:** baseline receipts, collapse diagnosis report,
pre-registration register, campaign results, candidate budget policy.

**Success criteria:** at least one pre-registered candidate produces capital
evidence and clears Tier-4 with the Phase 3 margin under unchanged gates; the
collapse diagnosis documents mechanisms.

**Guardrails:** no standard/deep campaign before Phases 3 and 5 are green; OOF is
never capital evidence; selection-biased cells are never compared naively.

## 12. PHASE 7 — Capital path: promotion → deployment → live monitoring

**Objective:** make the money path boring — rehearsal-tested, receipt-bound, and
reversible.

**Steps**

1. Close D1: `valid_export` requires the immutable `promotion_authorization.json`;
   current/export pointers respect it.
2. Rehearse then promote: `promote_model.py --scope train_only` first, then
   `--scope full`. Promotion refuses stale data (structural and content
   fingerprints), identity mismatches, and selection-biased evidence.
3. Acceptance gate on a real candidate: `nmr/submission.accept_promoted_artifact`
   against the reloaded `predict.pkl` — finite, strictly in (0,1), full live
   universe coverage.
4. Hosted-runtime verification (D5): validate the CatBoost artifact in Numerai's
   hosted runtime, or restrict deployable adapters to LightGBM/XGBoost until
   verified; the closure stays numpy/scipy/pandas-only at load time.
5. Champion governance: single-writer pointers only; immutable promotion
   receipts; documented rollback via authorization-verified pointer repair.
6. Stake governance: capital sizing from the risk envelope; staking keys and
   scopes isolated from submission automation; kill-switch and suspension
   criteria pre-committed before the first stake.

**Expected artifacts:** lifecycle fix, rehearsal receipt, accepted artifact
manifest, hosted-runtime evidence or adapter restriction, rollback runbook,
staking policy.

**Success criteria:** one end-to-end rehearsal — run → evidence → promotion →
acceptance → dry-run upload — with every step receipt-bound and reversible.

**Guardrails:** `champion.json` and `current.json` are never hand-edited; no
uploads while gates are red.

## 13. PHASE 8 — Assurance, documentation, program governance

**Objective:** reach independently auditable engineering quality and keep it.

**Steps**

1. Full pytest to completion; classify and record every skip with its reason as
   part of the receipt.
2. Coverage gate binding: regenerate coverage on the final branch state with
   branch coverage enabled, then bind floors in `scripts/coverage_gate.py`.
3. Mutation breadth: extend `scripts/mutation_gate.py` (CI-only, Linux) beyond
   `nmr/splitter.py` to evaluation, risk, transforms, predictions, promotion,
   lifecycle, and deployment; ratchet floors from committed receipts only.
4. Dataless container gate (`scripts/ci_repro.Dockerfile`) before sign-off for
   anything touching data loading, campaigns, or fixtures.
5. Docs SSOT audit: no contradictions or duplication across the four owner
   files; keep `AGENTS.md` ≤30 KB; retire stale records to `docs/99-archive/`.
6. Program cadence: weekly receipt review, defect-register burn-down, and at most
   one pre-registered candidate; monthly threshold-provenance and calibration
   review.
7. Recertify against the §1 rubric; require Eng ≥90 and Money ≥90 with receipts.

**Expected artifacts:** completed test evidence, coverage receipt, mutation
receipts, container-gate runs, docs audit, certification report.

**Success criteria:** lint and full suite green with recorded skips; coverage
floors bound to the current branch; mutation floors across seven modules; no doc
contradictions; certification ≥90/90.

**Guardrails:** never weaken a gate to hit a score — the ratchet is one-way.

## 14. Director decisions required before Phase 1 execution

1. Risk envelope numbers: max drawdown, turnover band, per-model capital ceiling.
2. Margin policy: accept the recommended `capital_margin` values or set others —
   they define what "beat the benchmark" means.
3. Deploy adapter policy: verify CatBoost in the hosted runtime, or restrict
   deploys to LightGBM/XGBoost this cycle.
4. One-off candidate quota: how many pre-registered candidates per cycle may
   reach capital evidence (recommend one to two).
5. Compute escalation: stay on the current machine (`medium`/`small` only; the
   full version is RAM-deferred), or provision a larger box — this bounds Phase 6.

## 15. Non-goals

- No fourth boosting family, neural network, or stacking zoo without the §14.5 /
  Phase 6.9 disposition.
- No rewrite of `PurgedEraSplitter`, neutralization math, or oracle metrics.
- No weakening of checkpoint identity or purge assertions.
- No staking or live upload of a candidate that has not cleared every gate.
- No new third-party dependencies without a recorded director exception.

## 16. Change control

This record is absorbed phase by phase: durable facts move to their owner
documents (`docs/02-strategy/` charter, `ARCHITECTURE.md` mechanics,
`CONTRIBUTING.md` commands, `AGENTS.md` agent rules), phase receipts accumulate
under `artifacts/reports/`, and this record is condensed into
`docs/99-archive/` once Phase 8 certifies or the program is superseded. Update
this file in the same commit as any change that makes a phase, gate, or defect
statement stale.
