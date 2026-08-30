# Design Spec: Untiered Benchmark Fleet (Community & Tutorial Model Recreation)

> Status: APPROVED (director disposition 2026-08-19). Implementation authorized across all sections.
> Scope: a new untiered layer of benchmark models in `nmr/` + `configs/benchmarks/fleet/`, recreating the three Numerai tutorial models (small + deep), the three community example scripts (shallow + deep), and the Finance Arena v0.2–v1.5.1 model series, all re-fit on v5.3 data through the framework's tested machinery. The existing 5-tier benchmark hierarchy is untouched.

## 1. Mission

Populate the benchmark system with a fleet of deterministic, re-runnable benchmark models spanning silly heuristics to competitive community architectures, without disturbing the existing 5-tier "line in the sand" hierarchy. Fleet models are **untiered**: they are scored through the identical evaluation pipeline and their measured scorecards place them against the tier ladder indirectly. This gives the hierarchy an open-ended supply of reference points (and, later, promotion candidates) without re-semanticizing tiers or weakening hard gates.

## 2. Approved Decisions Log

| # | Question | Decision |
|---|---|---|
| 1 | Integration target | Extend the benchmark system (configs + `nmr/benchmark.py` family), **not** standalone `ExperimentRunner` registry runs. |
| 2 | Tier assignment | **None.** New models are untiered fleet cells; measured performance places them against tier rungs indirectly (report-only). Existing tiers 0–4, gates, and configs are untouched. |
| 3 | Roster | Only the listed roster + missing variants. Skip already-existing cells (`null_constant_05`, `linear_ridge_*`, `canon_hello_numerai`) and skip legacy extras (`finance_arena_v0_1.ipynb`, `simple_lgbm_shallow.ipynb`). |
| 4 | Fidelity policy | Architecture/params/targets/neutralization proportions faithful to source notebooks; processing always via framework generators (purged trimmed-train fit → predict validation, per-era rank-Gaussianization, `NeutralizationEngine`). Where a notebook conflicts with an invariant (4-era embargoes, hand-rolled neutralize, CV loops, multi-seed retraining), the framework wins and the deviation is documented per cell (§6). |
| 5 | v5.2-era notebook sources | Re-fit on v5.3 (`data/v5.3`). v4/v4.1-only target names mapped by horizon/role (§6.3). |
| 6 | SNNR auxiliary targets | Pinned from the legacy artifact `../numer-AI/exploratory_notebooks/outputs/snnr_weights_vs_correlation_v5.2.csv` (17 targets, all present in v5.3). No runtime SNNR computation in cells. |
| 7 | v1.5.1 selection bias | Faithful: the sweep's candidate selection uses the validation set (as the notebook did). Report column `selection_bias: true` for search-mode cells. Guardrail-based filtering is simplified to best validation mean-CORR (documented deviation). |

## 3. Architecture — Untiered Benchmark Fleet

- **Existing hierarchy: zero changes.** Configs, `BenchmarkHierarchy`, gates (`assert_tier0_null_floor`, `assert_tier4_gate`, `assert_hierarchy_monotone`), anchors, tier-4 reference, and all tiered outputs keep their exact semantics.
- **New module `nmr/benchmark_fleet.py`** (one clear purpose: fleet config schema + fleet generators + fleet runner). Imports shared primitives from `nmr/benchmark.py` (`_train_validation_purged_split`, `_feature_cols` resolution, `_domain_frames` patterns, `tier4_gate_verdict` — see §5). `nmr/benchmark.py` keeps the hierarchy and gains exactly one thing: an extended `canonical_scorecards_bytes(fleet_scorecards=...)` for determinism coverage (§5.4). No logic moves out of `nmr/`.
- **Config layer:** `configs/benchmarks/fleet/` — 4 grouped YAML files (silly / tutorials / community / finance-arena). Same validation rigor as tiered configs (frozen dataclass, unknown keys rejected, closed sets, fail-fast).
- **Scoring:** each fleet cell → `(era, id, prediction)` frame via its generator → `evaluate_model` (the untouched evaluation core) with the same target/meta/benchmark overlap window and FNC@feature-universe definition as tiered cells.
- **Indirect tiering (report-only):** per fleet scorecard, compute `placement` against the per-tier max-corr rungs (same metric `assert_hierarchy_monotone` uses): e.g. `tier2..tier3`, `above tier 4`, `below tier 0`. Plus 7 informational tier-4 gate verdict columns. Nothing hard-fails from fleet results.
- **Determinism:** fleet scorecards join `canonical_scorecards_bytes` (sorted benchmark ids); cell seeds fixed; tree fits `n_jobs=1`; no wall-clock/paths.

## 4. Config Layer — `configs/benchmarks/fleet/`

| File | Cells |
|---|---|
| `fleet_silly.yaml` | `silly_target_lag_mean` |
| `fleet_tutorials.yaml` | `tutorial_hello_deep`, `tutorial_neutralized_small`, `tutorial_neutralized_deep`, `tutorial_ensemble_small`, `tutorial_ensemble_deep` |
| `fleet_community.yaml` | `community_example_shallow`, `community_example_deep`, `community_advanced_shallow`, `community_advanced_deep`, `community_sunshine_shallow`, `community_sunshine_deep` |
| `fleet_finance_arena.yaml` | `fa_v02_xgb`, `fa_v03_lgbm_mt`, `fa_v04_xgb_weighted`, `fa_v05_ridge_stack`, `fa_v060_mlp`, `fa_v150_ridge_stack_tail10`, `fa_v151_ridge_ensemble` |

### 4.1 Cell schema (frozen `FleetCellConfig`)

```yaml
cells:
  - benchmark_id: str            # unique across fleet (and disjoint from tiered ids)
    source: str                  # provenance: notebook/script path (report + audit only)
    input_space: none|small|medium
    model_kind: target_lag_mean|lightgbm|xgboost|mlp|ridge_stack
    targets: [target, ...]       # blend targets (or [primary] for single-target kinds)
    target_weights: {...}        # optional per-target blend weights (xgboost weighted blends)
    params: {...}                # kind-specific, exact-pinned (see §6)
    seed: 42
    neutralization: none|0.25|0.35|0.5|1.0   # via NeutralizationEngine (oracle parity)
    neutralizer_selection: none|riskiest_50  # optional: restrict neutralizers to a feature subset
    neutralizer_count: 50        # used when neutralizer_selection=riskiest_50
    fast_mode_params: {...}      # overrides when runner --fast-mode
    anchors: {...}               # optional, report-only; re-pinned after first measured run
```

- `VALID_FLEET_MODEL_KINDS = ("target_lag_mean", "lightgbm", "xgboost", "mlp", "ridge_stack")` — disjoint from tiered `VALID_BENCHMARK_MODEL_KINDS` except `lightgbm`/`xgboost` (shared generator entry points with fleet-only extensions).
- `VALID_FLEET_NEUTRALIZATION = (None, 0.25, 0.35, 0.5, 1.0)`.
- Validation rules: `target_lag_mean` requires `input_space: none` and a single primary target; `ridge_stack` requires `params.main_target` plus `params.specialists` (see §6.5); unknown keys/sections rejected; degenerate params raise `ValueError`. Seeds identical across processes (determinism invariant).

## 5. Engine

### 5.1 `nmr/benchmark_fleet.py`

- `load_fleet_config(path) -> FleetConfig` (frozen dataclass, mirrors `load_benchmark_config`).
- `BenchmarkFleet` class: `from_config_dir(config_dir)`, `run() -> FleetResult`, `fleet_frame()`, `write_fleet_csv(...)`.
- `run()` order: for each cell → generator predictions → `evaluate_model` → scorecard; then placement + tier-4 verdicts; one cell in flight (memory guard); lazy column projection.
- `fleet_placement(scorecard, tier_max_corrs) -> str`: rungs from per-tier max `corr.value` over the tiered scorecards (shared helper, computed once per run); returns the tightest rung interval containing the fleet corr, `above tier 4`, or `below tier 0`.
- `selection_bias` report column derived from `model_kind == "ridge_stack" and params.mode == "search"`.

### 5.2 Generators (all deterministic; output = `[era, id, prediction]` on validation)

**`target_lag_mean`** — no fit. For each validation era, prediction = mean of `target` over **all rows pooled** across the trailing `window` (default 1) **train** eras, constant within the validation era. Train targets only ⇒ leak-safe by construction. Raises if `window >` available train eras.

**`lightgbm`** (shared with hierarchy) — single fit on purged trimmed train, predict validation, optional `neutralization` + `neutralizer_selection`. `riskiest_50`: rank features by cross-era IC instability using `nmr/features.py` stability screening (complement ranking; documented deviation from the notebooks' `get_biggest_change_features` — same intent, framework-tested machinery), take top `neutralizer_count`, neutralize against those columns only.

**`xgboost`** — extended with: multi-target support (one fit per target, rank-Gaussian per component, weighted mean per `target_weights` — normalized — then re-Gaussianize; equal weights default); optional `early_stopping_rounds` + `holdout_era_frac` (deterministic tail-of-train era holdout, used only for early stopping — final predict is the best-iteration model). Single-target behavior unchanged (hierarchy compatibility).

**`mlp`** — sklearn `MLPRegressor` with `StandardScaler` fit on trimmed train only (stats persisted, applied to val; zero-variance → 0.0 safeguard reuses the existing standardization rule). Params validated against a closed set of allowed keys (`hidden_layer_sizes`, `activation`, `solver`, `alpha`, `learning_rate_init`, `batch_size`, `max_iter`, `early_stopping`, `n_iter_no_change`, `validation_fraction`). Fixed `random_state=seed`.

**`ridge_stack`** — two modes, both leak-safe by construction (meta features only from train):

- Stack split topology (both modes): trim the purged train, then reserve the trailing `params.meta_tail_pct` of train **eras** as the meta tail; the remainder is the specialist-train partition. The boundary between them gets a horizon-aware purge buffer: **16 eras if any specialist is a 60D target, else 8** (mirrors `PurgedEraSplitter`'s 20D/60D purge convention; 8 extra eras on a ~500-era train is negligible).
- `mode: fixed` (v0.5, v1.5.0): fit one Ridge per specialist target on specialist-train (NaN-masked per target independently, `alpha=params.alpha`); predict meta tail + validation; per-era rank each component (rank-Gaussian); fit meta Ridge (`alpha=params.meta_alpha`) on tail OOF preds vs `params.main_target`; predict validation meta-features; optional `neutralization`; final per-era rank-Gaussianize.
- `mode: search` (v1.5.1): deterministic config-driven selection inside the generator —
  1. **Target quality filter** on train: coverage ≥ `target_min_coverage` (0.50), |corr to main| ≥ `target_min_abs_main_corr` (0.01), priority hints first, then top-`target_top_k` (12) by pinned `snnr_weight`.
  2. **Specialist alpha search**: `np.logspace(-2, 4, 13)` per surviving target; Sharpe of meta-tail ranked preds vs main target; prune Sharpe < `specialist_sharpe_floor` (0.50); keep ≥ `min_specialists` (6), else raise.
  3. **Meta candidates**: non-negative Ridge over `np.logspace(-2, 4, 9)` (fallback to plain Ridge if `positive=True` unsupported) + LightGBM (`max_depth=3, n_estimators=500, learning_rate=0.03, colsample_bytree=0.8, subsample=0.8, reg_lambda=1.0`, early stopping 50 vs 20% meta-era tail, ≥5 eras).
  4. **Post-processing sweeps**: benchmark decorr `[0.00, 0.05, 0.10]` against `v53_lgbm_ender20` — the notebook's subtractive rank-space decorrelation (`base_raw - decorr_strength * per_era_rank(benchmark)` then re-rank) — × neutralization grid `[0.0, 0.1, ..., 0.5]` via `NeutralizationEngine`.
  5. **Selection**: best validation mean CORR20V2 (notebook's primary metric). **Documented deviation**: the notebook's guardrail filters (annual return, MMC floor, max drawdown) are not reproduced; selection is metric-only. `selection_bias: true` in the report.

### 5.3 `tier4_gate_verdict`

`nmr/benchmark.py` gains `tier4_gate_verdict(scorecard, gate_config) -> dict[str, bool|None]` returning per-threshold booleans (None where a field is structurally unavailable, e.g. turnover). `assert_tier4_gate` refactored to build on it (hard gate unchanged: raises on failure). Fleet rows display the verdict dict.

### 5.4 Determinism

`canonical_scorecards_bytes(scorecards, fleet_scorecards=None)` — fleet scorecards serialized in the same canonical frame (sorted ids, timing fields stripped). Cross-process hash test extended. No other change to hashing.

## 6. Roster — 19 cells

All cells ship without `anchors` (re-pinned after first measured run, decision #2 procedure from the hierarchy spec). In the tables below, `ender_20` is shorthand for `target_ender_20`, `jeremy_20` for `target_jeremy_20`, etc.

### 6.1 Wave 0 — silly (1)

| id | kind | input_space | targets | params | neutralization |
|---|---|---|---|---|---|
| `silly_target_lag_mean` | target_lag_mean | none | [target] | window: 1 | none |

### 6.2 Wave 1 — tutorials (5; `canon_hello_numerai` small already exists)

Sources: `docs/05-notebooks/1_hello_numerai.ipynb`, `2_feature_neutralization.ipynb`, `3_target_ensemble.ipynb`. Shallow = notebook defaults (2k trees); deep = the notebooks' commented "for better performance" params (30k trees). Notebook 3 applies **no neutralization** (verified: no neutralize code in the notebook).

| id | kind | input_space | targets | params (trees/lr/depth/leaves) | neutralization |
|---|---|---|---|---|---|
| `tutorial_hello_deep` | lightgbm | small | [target] | 30000/0.001/10/31 | none |
| `tutorial_neutralized_small` | lightgbm | small | [target] | 2000/0.01/5/15 | 0.5 |
| `tutorial_neutralized_deep` | lightgbm | small | [target] | 30000/0.001/10/15 | 0.5 |
| `tutorial_ensemble_small` | lightgbm | small | [ender_20, victor_20, xerxes_20, teager2b_20] | 2000/0.01/5/31 | none |
| `tutorial_ensemble_deep` | lightgbm | small | same 4 targets | 30000/0.001/10/31 | none |

All: colsample_bytree 0.1; seed 42; fast_mode_params `n_estimators: 50`.

### 6.3 Wave 2 — community scripts (6)

Sources: `docs/05-notebooks/community_notebooks/example_model.py` (`_example_`), `example_model_advanced.py` (`_advanced_`), `example_model_sunshine.py` (`_sunshine_`). Shallow = 2k trees / lr 0.01 / depth 5 (sunshine: leaves 32); deep = the scripts' recommended params 20k / lr 0.001 / depth 6 / leaves 64.

v4/v4.1 target mapping (decision #5): `nomi_v4_60 → target_ender_60` (canonical 60D aux), `jerome_v4_60 → target_ender_60`, `nomi_v4_20 → target_ender_20`, `jerome_v4_20 → target_jeremy_20` (closest v5.3 name, same horizon — flagged assumption, one-line config edits if wrong). Identical-name 20D targets (`ralph`, `tyler`, `victor`, `waldo`) map verbatim.

| id | kind | input_space | targets | params | neutralization | deviations |
|---|---|---|---|---|---|---|
| `community_example_shallow` | lightgbm | medium | [target] | 2000/0.01/5/31 | 1.0, riskiest_50 | screen replaces `get_biggest_change_features` |
| `community_example_deep` | lightgbm | medium | [target] | 20000/0.001/6/64 | 1.0, riskiest_50 | same |
| `community_advanced_shallow` | lightgbm | medium | [target, ender_60, jeremy_20] | 2000/0.01/5/31 | 1.0, riskiest_50 | 3-fold CV + model-selection loop collapsed to single purged fit |
| `community_advanced_deep` | lightgbm | medium | same 3 | 20000/0.001/6/64 | 1.0, riskiest_50 | same |
| `community_sunshine_shallow` | lightgbm | medium | [ender_20, ender_60, ralph_20, tyler_20, victor_20, waldo_20] | 2000/0.01/5/32 | 0.5 (all features) | equal-weight blend; notebook's `all_data` retrain maps to trimmed-train fit |
| `community_sunshine_deep` | lightgbm | medium | same 6 | 20000/0.001/6/64 | 0.5 (all features) | same |

All: colsample_bytree 0.1; seed 42; fast_mode_params `n_estimators: 50`.

### 6.4 Wave 3 — Finance Arena v0.x (5)

Sources: `../numer-AI/models/version_0/v0.{2..6}/finance_arena_v0*.ipynb` (mined read-only). Deviations: 4-era embargoes → purged trimmed-train fit; hand-rolled per-era neutralize → `NeutralizationEngine`; multi-seed [42,43,44] retraining → single seed 42; GPU flags → CPU `n_jobs=1`.

| id | kind | input_space | targets | params | neutralization |
|---|---|---|---|---|---|
| `fa_v02_xgb` | xgboost | small | [target] | 2000/0.01/6, subsample 0.8, colsample 0.1, early_stopping_rounds 50, holdout_era_frac 0.1 | none (rank only) |
| `fa_v03_lgbm_mt` | lightgbm | small | [target, ender_20, victor_20] | 20000/0.01/6/64 | 0.5 |
| `fa_v04_xgb_weighted` | xgboost | small | [target, jasper_20, teager2b_20, claudia_20] | 2000/0.01/6, subsample 0.8, colsample 0.1; weights 0.35/0.30/0.23/0.12 | 0.35 |
| `fa_v05_ridge_stack` | ridge_stack | small | (params.main_target = ender_20) | mode fixed, alpha 1e-6, meta_alpha 1e-6, meta_tail_pct 0.30, specialists = 17 SNNR | 0.5 |
| `fa_v060_mlp` | mlp | small | [ender_20] | (256,128,64)/relu/adam/alpha 1e-3/lr 0.001/batch 1024/max_iter 150/early_stopping 15/val_frac 0.05 | 0.5 |

fast_mode_params: trees → 50 (v0.2/v0.3/v0.4); mlp `max_iter: 20`; ridge_stack fixed-mode unchanged (closed-form, already fast); ridge_stack search-mode shrinks the specialist alpha grid to 3 points (`logspace(-2, 4, 3)`) and the decorr/neutralization sweeps to a single point each, keeping the smoke gate minutes-scale.

### 6.5 Wave 4 — Finance Arena v1.5.x (2)

Sources: `../numer-AI/models/version_1/v1.5/fa_v1.5.0_ridge_ensemble.ipynb` (exports `fa_v1.5.0_ridge_stacking_tail10`), `fa_v1.5.1_ridge_ensemble.ipynb`.

| id | kind | input_space | params | neutralization |
|---|---|---|---|---|
| `fa_v150_ridge_stack_tail10` | ridge_stack | small | mode fixed, alpha 1e-6, meta_alpha 1e-6, meta_tail_pct 0.10, main_target ender_20, specialists = 17 SNNR | none |
| `fa_v151_ridge_ensemble` | ridge_stack | small | mode search, main_target ender_20, aux candidates = 17 SNNR + pinned snnr_weights, top_k 12, min coverage 0.50, min |corr| 0.01, priority hints [victor_20, xerxes_20, teager_20], alpha grid logspace(-2,4,13), sharpe floor 0.50, min specialists 6, meta alpha grid logspace(-2,4,9), LGBM meta (3/500/0.03/0.8/0.8/λ1.0/es50), meta_tail_pct 0.10, decorr grid [0.00,0.05,0.10] vs v53_lgbm_ender20, neu grid [0.0..0.5 step 0.1], NaN fill 0.5, selection = best val mean CORR20V2 (guardrails simplified) | swept (selected value) |

The pinned 17 SNNR specialists (decision #6), all present in v5.3: `target_jasper_20, target_teager2b_20, target_claudia_20, target_rowan_20, target_waldo_20, target_ender_60, target_xerxes_20, target_jeremy_20, target_cyrusd_20, target_agnes_20, target_victor_20, target_ralph_20, target_caroline_20, target_delta_20, target_tyler_20, target_sam_20, target_echo_20`.

## 7. Runner & Reporting — `benchmark_runner.py`

- CLI gains `--fleet-configs configs/benchmarks/fleet` (default; `--no-fleet` disables). Thin control plane only.
- Fleet cells run after tiered cells. Outputs: `artifacts/reports/benchmark_fleet_scorecard.csv` (smoke suffix pattern preserved), columns: scorecard fields + `benchmark_id`, `source`, `placement`, `selection_bias`, and 7 `gate_*` verdict columns. Existing hierarchy CSV + gate report unchanged.
- Exit codes unchanged (tiered hard gates only). A crashing fleet cell fails loudly (non-zero), no swallow.
- `--fast-mode`: `n_boot=1` + `fast_mode_params` overrides apply to fleet cells identically.

## 8. Tests

New `tests/test_benchmark_fleet.py` (synthetic fixtures only; real-data smoke stays the pre-sign-off gate):

- fleet config loading: valid round-trip, unknown keys rejected, missing tier not required, invalid kind/neutralization rejected, `target_lag_mean` input_space rule, `ridge_stack` required params;
- `target_lag_mean`: per-era constant equals trailing-window train mean; window>1; degenerate window raises; never reads validation targets (fixture omits them);
- `lightgbm` + `neutralizer_selection=riskiest_50`: neutralizers restricted to selected columns; count honored;
- `xgboost` multi-target: weight normalization, equal-weight default, rank-blend then re-Gaussianize; early stopping picks best-iteration model on deterministic holdout;
- `mlp`: same-seed determinism (two fits), train-only scaler stats, zero-variance → 0.0, closed param-key set;
- `ridge_stack` fixed: tail-split purity (meta features only from tail train eras, never validation), meta fit on tail OOF, NaN-masked specialists, rank-domain blend, final re-Gaussianize;
- `ridge_stack` search: quality filter (coverage/corr/priority/top-k), alpha grid bounds, Sharpe pruning floor + min-kept raise, deterministic sweep ordering, `selection_bias` derived flag;
- `fleet_placement`: known rungs → correct interval; `above tier 4` / `below tier 0` edges; empty ladder raises;
- determinism: fleet scorecards in `canonical_scorecards_bytes` cross-process hash (subprocess).

Existing tests: `tests/test_benchmark_hierarchy.py` determinism test extended with fleet bytes; `tier4_gate_verdict` covered via the existing gate tests plus new verdict-shape tests. Parity tests untouched (`NeutralizationEngine` reused as-is).

## 9. Documentation & SSOT (implementation commit)

- `docs/06-evaluation/benchmark-line-in-the-sand.md`: new "Untiered Benchmark Fleet" section (schema, placement semantics, fidelity policy, anchor re-pin procedure).
- `docs/06-evaluation/benchmark-line-in-the-sand.md`: active hierarchy reference pointing at this fleet extension.
- `ARCHITECTURE.md`: `nmr/benchmark_fleet.py` module spec, fleet schema, generators, runner CLI, artifact paths.
- `AGENTS.md`: toolkit table row and operational hazards (deep 20k/30k-tree fits are multi-hour; fleet is exempt from hard gates; the v1.5.1 search cell is selection-biased by design).
- `CONTRIBUTING.md` / `README.md`: touched only if they name removed artifacts (none — check in plan).

## 10. Implementation Watchpoints

1. **Meta-feature purity**: every `ridge_stack` variant must guarantee meta features come only from tail train eras; validation is never touched by specialist or meta fits until final predict. Assert in code.
2. **Aux-target NaN masking**: each specialist fit masks its own target's NaN rows independently (hierarchy watchpoint #1, same rule).
3. **Standardization zero-variance safeguard**: reuse for `mlp` scaler (0.0 output, no divide-by-zero).
4. **Search determinism**: grid ordering fixed by config; tie-breaking by grid index; no dict-iteration-order dependence.
5. **Rank-domain invariant**: all multi-target blends rank-Gaussianize per component, blend, then re-Gaussianize (never blend raw regression outputs).
6. **Canonical hash safety**: fleet scorecard rows must exclude timing fields exactly like tiered rows.

## 11. Risks & Hazards

- **Deep-cell runtime**: 20k/30k-tree single fits on ~2.1M train rows (medium = 780 features) are multi-hour CPU jobs each; full fleet ≈ tens of CPU-hours across waves. `nohup` + log polling; fast-mode smoke keeps verification minutes-scale. GPU acceleration for fleet cells is a **deferred follow-up** (per-device determinism caveat).
- **v1.5.1 sweep runtime**: 156 ridge fits on 2.4M×42 (seconds-scale each) + tiny meta sweeps — expected minutes, not hours; but the *deep tree cells* dominate the wall clock.
- **`fa_v151_ridge_ensemble` is selection-biased**: its scorecard was selected on validation. Never compare it naively against unbiased cells; the `selection_bias` column exists for exactly this reason.
- **v4 target-name mappings** (`jerome_v4_20 → target_jeremy_20` etc.): name-adjacency assumptions, one-line config edits if incorrect.
- **Fleet scorecards vs monotonicity**: fleet results never feed the hard gates, so a fleet model beating the tier-4 reference cannot break the ladder. Re-pinning anchors is manual and evidence-driven.
- **Suite drift**: verification results belong in review output, not as hardcoded counts in maintained docs.

## 12. Verification & Execution Order

1. TDD implementation + tests per §8; `ruff check .` + `pytest -q` fast gate.
2. Real-data smoke: `benchmark_runner.py --fast-mode` (fleet on) → hierarchy + fleet CSVs written; placements and gate verdicts populated; hard gates still pass.
3. Wave-by-wave full runs (background, `nohup`, log polling): wave 0+1 (silly + tutorials) → wave 2 (community) → wave 3 (FA v0.x) → wave 4 (FA v1.5.x). Record measured scorecards; re-pin fleet anchors as a follow-up commit after each wave if adopted.
4. End-of-session gate: `ruff check .` + `pytest -q` on final state; truthful report including any skips.
