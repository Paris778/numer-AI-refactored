# Design Spec: Model-Development Operating System

> Status: APPROVED (director disposition 2026-09-07). The director's OS-first
> decision is the binding authority. Implementation is authorized for the
> prediction/composition/evaluation seam, feature-set derivation extraction,
> and research-proxy labelling. **No new serious model family is authorized.**
>
> Scope: a narrow, auditable operating system so training produces prediction
> artifacts, composition is reusable, and evaluation consumes those artifacts
> without knowing what a model is. Existing CatBoost/LightGBM paths remain
> canaries. Ridge as a protocol-test baseline is deferred until this seam is
> stable.

## 1. Mission

The current problem is not lack of model variety. LightGBM, XGBoost, and
CatBoost already exist, but their results are not comparable or reliably
profitable. Adding another model would increase search-space size,
multiple-testing burden, selection bias, checkpoint complexity, and the
temptation to explain away poor validation as insufficient capacity.

Evidence on 2026-09-07:

```text
OOF performance:        sometimes promising
validation performance: poor
capital gate:           not passed (best own val CORR 0.00774 vs 0.0286)
capital gate itself:    Sharpe re-pinned 2026-09-07 to 0.5358 (coherent)
deployment state:       absent (no champion, no exports)

XGB 4-target ensemble:
  OOF CORR  ~0.033
  val CORR  ~0.0006
  val payout negative
```

That is evidence that the research-to-capital path is not yet trustworthy
enough to guide model selection. This spec builds that path.

## 2. Approved Decisions Log

| # | Question | Decision |
|---|---|---|
| 1 | Another serious model now? | **No.** OS first. Existing CatBoost/LightGBM as canaries only. |
| 2 | First extra model after OS? | Simple deterministic diversity baseline (ridge), as protocol test + diversity — not a presumed alpha model. Deferred until go/no-go §10. |
| 3 | Rewrite composition math? | **No.** Keep rank → blend → neutralize → final rank. Make the stage explicit and reusable. |
| 4 | Force every HPO trial through `evaluate_model`? | **No.** Research proxy (cheap, labelled) vs capital scorecard (promotion-only). |
| 5 | Prohibit cross-horizon experiments? | **No.** Persist identity separately so they cannot be misunderstood. |
| 6 | Worktree isolation? | **No.** AGENTS.md junction/worktree deletion hazard: never junction `experiments/` or `artifacts/`. Work in the main checkout. |
| 7 | Dashboard expansion? | **No.** Provenance fields already on the unified schema; no new UI. |
| 8 | Estimator Protocol / open model registry? | Deferred. Closed GBM backends stay. Foreign prediction files enter through the prediction artifact, not through `_build_model`. |
| 9 | Effort split | 60% OS/eval contract, 20% data/feature provenance, 20% controlled CatBoost/LGBM reproduction. No standard/deep XGB campaign. |
| 10 | Gate Sharpe 0.86 vs 0.536 | Already resolved 2026-09-07: gate is **benchmark-relative**. Configured AC-Sharpe min is 0.5358. Do not re-open. |

## 3. Non-goals (this cycle)

- A fourth boosting family, neural net, or stacking zoo.
- Rewriting `PurgedEraSplitter`, neutralization math, or oracle metrics.
- Weakening checkpoint identity.
- Moving HPO onto the runner's last-fold geometry in this cycle (label the
  existing 80/20 proxy instead; geometry unification is a later cycle).
- Dashboard or data-platform rewrite.
- Staking, live upload, or champion promotion of a failing candidate.

## 4. Architecture

Three boxes, one seam:

```text
train / foreign producer
        │
        ▼
 PredictionSet  (era, id, prediction) + provenance
        │
        ▼
 compose (optional): rank components → blend → neutralize → [submission rank]
        │
        ▼
 research proxy  XOR  capital scorecard (evaluate_model)
        │
        ▼
 gate / promote / accept  (capital scorecard only)
```

`evaluate_model` stays a library over frames. Training stays a producer of
`PredictionSet`. Composition becomes a named function that both the runner and
a foreign parquet can call. Research proxies are labelled so they cannot be
read as capital.

## 5. Prediction artifact contract

Module: `nmr/predictions.py` (new). Public types exported from `nmr/__init__.py`.

### 5.1 Frame

Required columns, in this order after validation: `era`, `id`, `prediction`.

Invariants (fail loud with `ValueError`):

- unique `(era, id)`
- deterministic sort `["era", "id"]`
- finite `prediction` (no NaN/Inf)
- non-null `era` and `id`
- height ≥ 1

### 5.2 Stages

Closed set `PREDICTION_STAGES = ("raw", "blended", "neutralized", "validation", "submission")`.

Stage-specific range rules:

| Stage | Extra rule |
|---|---|
| `raw` | finite only |
| `blended` | finite only (rank-gaussianized blend output) |
| `neutralized` | finite only |
| `validation` | finite only; this is the capital-scorecard input |
| `submission` | every value strictly in `(0, 1)` (post `tie_kept_rank`) |

Mixing stages is a provenance error, not a silent coerce.

### 5.3 Provenance

Frozen dataclass `PredictionProvenance`:

```text
stage: str                         # ∈ PREDICTION_STAGES
trained_targets: tuple[str, ...]
ensemble_target: str | None
scoring_target: str | None
training_horizon: str | None       # "20D" | "60D" | None
scoring_horizon: str | None
era_partition: str | None          # "oof" | "validation" | "live" | "held_out" | None
data_fingerprint: str | None
feature_fingerprint: str | None
fit_role: str | None               # "cv_oof" | "full_history" | "foreign" | None
device: str | None
source_run_id: str | None
selection_bias: bool               # default False
split_estimand: bool               # default False
```

Canonical JSON for hashing/persistence uses `sort_keys=True`, excludes
wall-clock and absolute paths. `None` fields are present (not omitted) so
readers can see what was unknown.

`split_estimand` is true when scoring target/horizon differs from ensemble
target / training horizon — same predicate as `nmr.runner._estimand_block`.

### 5.4 `PredictionSet`

Frozen dataclass: `frame: pl.DataFrame`, `provenance: PredictionProvenance`.

Constructors:

- `prediction_set_from_frame(frame, provenance, *, era_col="era", id_col="id", pred_col="prediction") -> PredictionSet`
- validates frame, normalizes column names, sorts, freezes provenance.

Readers of a parquet: `read_prediction_set(path, provenance) -> PredictionSet`.
The parquet itself does not have to embed provenance (existing `oof.parquet`
and `validation_preds.parquet` stay three-column). Provenance lives on the
run manifest / caller. Optional sidecar is out of scope this cycle.

## 6. Composition contract

Do not rewrite `Ensembler` or `NeutralizationEngine`. Add
`compose_predictions(...)` in `nmr/predictions.py` that:

1. Validates component columns (`pred_*` or explicit `pred_cols`).
2. Calls `Ensembler.blend` (per-era rank-gaussianize → weighted blend →
   re-gaussianize).
3. If `neutralization_proportion > 0`, joins features on `(era, id)` when
   they are missing from the prediction frame (`features=` argument) and
   calls `NeutralizationEngine.neutralize`. Proportion must be finite in
   `[0, 1]` — negative/`NaN` raise.
4. Returns a `PredictionSet` at stage `blended` or `neutralized`.
5. Optional `as_submission=True` applies per-era `tie_kept_rank` to `(0, 1)`
   and returns stage `submission`. Research partitions (`oof`, `held_out`)
   cannot take this path.

The critical test: a prediction parquet produced **outside** `models.py` can
be composed and scored without modifying trainer code.

The runner may later call this helper; wiring the runner is a follow-on task
in the same cycle only if the helper's tests are green and the runner's
existing OOF bytes stay bit-identical. If wiring risks checkpoint identity,
leave the runner calling `Ensembler`/`NeutralizationEngine` directly and keep
the helper as the foreign-file path.

## 7. Evaluation levels

### 7.1 Capital scorecard

`evaluate_model` is unchanged as the capital function. A thin wrapper
`evaluate_prediction_set(prediction_set, *, meta_model, benchmarks, features, targets, **kwargs)`:

- capital path returns a `MetricScorecard` and requires a `CapitalContext`
  derived by a trusted validation loader (the runner computes every field
  from the loaded, purged `validation.parquet` and the run config) plus all
  of: `stage == "validation"`, `era_partition == "validation"`,
  `selection_bias is False`, required scoring/payout/fingerprint identity,
  training-estimand identity, and **exact `(era, id)` key coverage** —
  predictions, meta model, features, and targets must all equal the
  authoritative key universe; when benchmarks are supplied, they must also
  equal it, with no duplicate keys. Sparse or extra rows in any frame are
  rejected. The final joined base is checked after the benchmark join against
  the same row count and key fingerprint. Missing identity is
  rejected; at the evaluate level, self-attested window/target/horizon that
  internally matches both provenance and context is accepted (the predicate
  binds the frame to the context — it does not authenticate the context's
  producer); the authority comes from promotion's evidence check, which
  re-verifies the key universe against the data on disk and requires the
  runner-produced evidence block.
- The runner persists `manifest.capital_evidence` (`CAPITAL_EVIDENCE_VERSION`):
  window, key fingerprint, row count, payout/target/horizon, data/feature
  fingerprints, training estimand, and the stored scorecard's digest.
- Promotion requires that evidence block (internal check), independently
  derives the purged validation window from the current data and policy,
  binds `scorecard.n_eras` to that window, and recomputes the exact `(era, id)`
  universe. Fresh publication and pointer repair use the same authorization;
  repair additionally checks an immutable `promotion_authorization.json`
  receipt binding export activation flags, gate/acceptance fields, evidence,
  and scorecard digests to the run.
  Direct `evaluate_model` scorecards
  (benchmarks, fleet, HPO, cross-check replay) are research/report-only and
  can never enter promotion; legacy runs without evidence are refused.
- `submission` is a rank-domain artifact, **not** capital-eligible.
- `as_submission=True` cannot relabel `oof` / `held_out` partitions.
- research stages (`raw` / `blended` / `neutralized`, or any non-capital
  provenance) return `ResearchEvaluation(scorecard, is_capital=False)` only
  when `allow_research_stage=True`; otherwise they raise.
- does not import `nmr.models`.

Promotion authorization: only a scorecard bound to runner-produced capital
evidence (full `evaluate_prediction_set` on the locked validation window)
can authorize promotion. `ResearchEvaluation` is not a promotion input, and
neither is a bare `MetricScorecard` from a direct `evaluate_model` call.

### 7.2 Research proxy

Extend `SweepResult` (and the Optuna return that reuses it) with explicit
proxy labels. Defaults make existing tests keep passing **and** make the
proxy unmistakable:

```text
is_capital: bool = False
proxy_metric: str | None = None      # the metric that was optimized
proxy_split: str = "held_out_80_20"  # current HPO geometry
proxy_target: str | None = None
selection_bias: bool = False
```

`HyperparameterSweep.run` and `bayesian_sweep` populate these from the config
(`evaluation.main_target`, the metric name). They never set `is_capital=True`.

Do **not** change the 80/20 held-out geometry in this cycle. Labelling is the
contract; unifying HPO onto last-fold OOF is a later cycle.

## 8. Target / estimand identity

Already persisted on the runner manifest (`estimand` block, 2026-09-07).
`PredictionProvenance` is the frame-level twin. No change to
`_estimand_block` except that `compose_predictions` / `prediction_set_from_frame`
must be able to build provenance from that block without duplicating the
split predicate. Extract `_scoring_identity` / split predicate to a tiny
shared helper only if it avoids importing `runner` from `predictions`.
Prefer duplicating the two-line predicate over a circular import.

## 9. Data exploration / derived feature sets

Move the reusable derivation currently in `analyze_dataset.py::_stage_derived_sets`
into `nmr/features.py`:

```text
derive_feature_sets(screen: pl.DataFrame, drift: pl.DataFrame, *, primary_target: str | None = None) -> dict[str, list[str]]
```

Pure function of the two frames. Same four keys, same sort, same "no drift
row ⇒ keep" rule, same empty-set-is-valid semantics. The script becomes a
thin reader/writer. Tests in `tests/test_features.py` own the logic;
`tests/test_analyze_dataset.py` keeps the CLI integration.

Do not build a new analysis dashboard.

## 10. Go / no-go before another serious model

All of the following must be true:

1. `ruff check .` and the required pytest gate are clean.
2. Tier-4 gate and official receipt remain reconciled (already true; do not regress).
3. Existing CatBoost path still runs through `ExperimentRunner` (no trainer rewrite required this cycle).
4. A foreign prediction parquet can be evaluated without modifying trainer code.
5. Target and horizon identity are persisted (manifest estimand — done) and present on `PredictionSet`.
6. HPO proxy results cannot be mistaken for capital scorecards (`is_capital=False` on `SweepResult`).
7. Checkpoint reuse still fails after fitting-code changes (do not weaken; no required change unless composition moves into the runner).
8. Validation window remains untouched during HPO selection (unchanged 80/20-on-train; validation parquet is not loaded by `_held_out_metric`).
9. Artifact acceptance exists (already true). Reaching a passing economic gate is **not** required this cycle.

## 11. Testing

TDD. New tests:

- `tests/test_predictions.py` — frame invariants, stage rules, provenance frozen, compose of synthetic components, foreign parquet → `evaluate_model` without `nmr.models`.
- `tests/test_features.py` — `derive_feature_sets` success, empty sets, missing columns, primary-target fallback, drift-keep rule.
- `tests/test_research.py` / `tests/test_opt.py` — SweepResult proxy labels, `is_capital is False`.
- `tests/test_contribution.py` — public API symbols.
- Existing runner/scorecard/analyze tests must stay green; OOF bytes of the synthetic runner fixture must not change if the runner is unwired.

## 12. Documentation (same commit as code)

- This spec.
- `ARCHITECTURE.md` — new short section for `nmr/predictions.py`; §P notes `derive_feature_sets`; SweepResult fields.
- `AGENTS.md` toolkit row for prediction artifacts (keep under 32 KiB).
- `CODEBASE.md` routing: "score a foreign parquet" → `nmr/predictions.py`.
- `docs/superpowers/README.md` indexes this spec.
- `analyze_dataset.py` docstring points at the extracted function.

No aspirational text. Describe what exists after each task lands.
