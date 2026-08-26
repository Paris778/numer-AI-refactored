# Design Spec: Model Lifecycle & Self-Contained Experiment Layout (v3 — revised, 2nd review round)

> Status: REVISED (2026-08-26, 2nd review round) — resolves the 4 blocking + 4 secondary findings of the second critical review (v2 resolved the first round's 9 blocking + 2 secondary). Supersedes the initial draft. Replaces the three global per-model pools (`artifacts/runs/`, `artifacts/registry/`, `artifacts/models/`) with self-contained per-family directories at repo root `experiments/`, introduces a human display-name naming layer over the machine slug, and defines the four-stage model lifecycle (research → partial → full → staked, plus the `degraded` state) with a new train-only (partial) promotion scope and a local cross-check scoring step.

## 1. Mission

Two problems motivate this change:

1. **Illegibility.** A model is known by `brb1-xgb-v6 · 2610a99d` — a cryptic slug plus a 64-hex hash, with several near-identical runs per family differing only by hash. There is no human-facing name, no visible lifecycle stage, and no per-family record of what was done and why.
2. **Fragility + scatter.** Per-model state is spread across three global pools with no versioning and no per-experiment documentation. The 2026-08-26 session's registry-loss incident (a junction/worktree deletion wiped all 29 run dirs; the registry was git-ignored and unrecoverable) — **session history, not repository evidence** — motivated versioning the small human-relevant record.

The change rides the clean slate: no legacy runs to migrate — the new layout is the baseline.

## 2. Domain model (resolves B1: identity & ownership)

Three concepts, used precisely throughout:

- **Family** (slug): a research lineage. One directory `experiments/<slug>/`. A family contains **many runs** and **many exports** (each export belongs to exactly one run). The family-level `meta.json`, `README.md`, `config.yaml`, display name, and lifecycle badge describe the **family**, not a single run.
- **Run**: one experiment execution, identified by `run_id`. Immutable once recorded at `runs/<run_id>/run.json`. Each run's **effective** config is stored inside its own `run.json` (existing behavior); the family `config.yaml` is the family's **base reference config** written at family creation — runs may diverge from it.
- **Export**: an immutable deploy artifact for a specific run at a specific scope (`partial` | `full`), at `exports/<scope>/<run_id>/`. Never mutated after publication; new promotion = new slot.

Ownership table:

| Item | Owner | Notes |
|---|---|---|
| display_name | family (`meta.json`) | human label for the family |
| lifecycle badge | family | derived from family export state (stage = family's highest valid stage) |
| staked record | family (`meta.json`) | **bound to one export** (B7) |
| config | run (`run.json`) + family base (`config.yaml`) | run.json is authoritative per run |
| scorecards | run (`run.json`) + partial export (`scorecard.json`) | run scorecard = research CV; export scorecard = cross-check |

## 3. Layout & persisted-file inventory (resolves B2)

```
experiments/<slug>/                          # family = research lineage
  README.md                    [git]         # human record: what was done, decisions, results
  config.yaml                  [git]         # family base config (reference; per-run effective configs in run.json)
  meta.json                    [git]         # {display_name, staked: {run_id, scope, numerai_model_id, staked_at, status}}
  runs/<run_id>/
    run.json                   [git]         # scorecard + provenance + effective config
    oof.parquet                [ignored]     # per-fold OOF preds; reconstructable by checkpoint-resume re-run
    validation_preds.parquet   [ignored]     # era-batched validation preds; reconstructable
    predict.pkl                [ignored]     # research deploy closure (runner-built)
    predict.pkl.manifest.json  [ignored]     # sibling hash manifest; travels with its pickle
    oof_checkpoints/           [ignored]     # resume state (code/device identity guarded)
    deploy_checkpoints/        [ignored]
    validation_checkpoints/    [ignored]
  exports/
    partial/<run_id>/
      manifest.json            [git]         # promotion record: config, provenance, tier-4 receipts, training_scope
      scorecard.json           [git]         # local cross-check reference (B5 contract)
      predict.pkl              [ignored]     # train-only artifact (rebuildable via promotion)
      predict.pkl.manifest.json [ignored]    # deploy sibling hash manifest
    full/<run_id>/
      manifest.json            [git]
      predict.pkl              [ignored]
      predict.pkl.manifest.json [ignored]
    full/current.json          [git]         # active full-version pointer (versioned: it IS the "what's live" record)
experiments/champion.json      [git]         # global best-run pointer {run_id, experiment_slug, promoted_at} (B3)
```

Every file produced by the system is classified: **git-tracked** (small, identity-bearing record), **ignored** (heavy or reconstructable), or absent-by-design. The git record preserves identity, provenance, and results even if all heavy files are lost. **Reproducibility claim (scoped):** ignored pickles/parquets are reproducible by re-running the recorded config **only while the original inputs remain available** — the code identity, the dependency environment, the device state, and the data snapshot at scoring time. The data fingerprint is a *snapshot marker*, not a content snapshot (documented detection limits: restated feature values within unchanged schema/row-count/era-stats are undetected; validation grows over time). Therefore the run record persists the exact data fingerprint + code identity (portable, environment-independent) so reproducibility conditions are *verifiable*; when the current fingerprint differs from the recorded one, a rebuild is **refused** (no silent stale reuse — delete-to-refit discipline, mirroring the checkpoint identity rule). Heavy files are never auto-loaded from outside the repo (existing trusted-source rule).

### 3.1 Git rules (exact, resolves B2)

```
experiments/**
!experiments/
!experiments/**/
!experiments/*/README.md
!experiments/*/config.yaml
!experiments/*/meta.json
!experiments/*/runs/*/run.json
!experiments/*/exports/**/manifest.json
!experiments/*/exports/**/scorecard.json
!experiments/*/exports/full/current.json
!experiments/champion.json
!experiments/**/.gitkeep
```

`experiments/` tree scaffolding is kept via `.gitkeep` (root + nested dirs, as `artifacts/` does) — explicitly negated above. Everything else (parquet, pkl, checkpoints) stays ignored by the blanket rule.

## 4. Naming (unchanged from draft)

**Display name layered over machine slug.** Slug = family dir + `run.name`, `^[a-z0-9_-]+$`, template `<theme>-<backend>-<vN>` going forward (e.g. `ender-xgb-v1`). Human `display_name` (e.g. `"Ender XGB v1 · medium"`) lives in `meta.json`, surfaces in dashboard labels, manifests, and docs; the hash appears only in tooltips/URIs. Lifecycle is a **badge** (`· partial` / `· full` / `· staked`), never a name mutation. `display_name` is not a run-config field (run-id determinism untouched).

## 5. Lifecycle (resolves B1/B6/B7 — total function)

Stages are **derived from filesystem state** (never a mutable counter). The derivation is a **total function**: every filesystem state maps to exactly one stage. Badge precedence `staked` > `full` > `degraded` > `partial` > `research`:

- `research` — ≥1 `runs/<run_id>/run.json`, no valid exports.
- `partial` — ≥1 **valid** `exports/partial/<run_id>/`, no valid full export.
- `degraded` — ≥1 **valid** full export, but `exports/full/current.json` is missing or points at an invalid slot. (A valid full exists; the family pointer is broken — surfaced explicitly, never silently read as `full` or `partial`.)
- `full` — `exports/full/current.json` points at a **valid** full slot.
- `staked` — `meta.json.staked.status == "active"` AND the referenced export is valid; otherwise the badge shows the underlying stage and the staked record is flagged `stale` in the payload.

The reviewer-flagged undefined state — valid full export + missing/dangling `current.json` + no valid partial — is exactly the `degraded` stage. Precedence applies to the highest stage the family qualifies for.

**Export validity predicate** (single definition, used by discovery and dashboard):

```
valid(export) ⇔ manifest.json present AND training_scope == dir scope
  AND predict.pkl present AND predict.pkl.manifest.json present
  AND sha256(predict.pkl) == sibling manifest value AND load_predict() succeeds
  AND manifest.family == slug AND manifest.promoted_from_run_id ∈ family runs/
  AND (scope == partial ⇒ scorecard.json present)
```

**Partial selection** (no single pointer): the dashboard renders one `family::partial::<run_id>` row per **valid** partial export. Ordering is deterministic: by `promoted_at` descending — `promoted_at` must parse as ISO-8601 (unparseable values sort last and are flagged in the payload) — tie-broken by `run_id` ascending. The family badge is `partial` iff ≥1 valid partial and no valid full.

**Staked binding** (B7): `meta.json.staked` is `{run_id, scope: "full", numerai_model_id, staked_at, status: "active" | "retired"}`, where `run_id` must reference a **valid full export** — the artifact actually uploaded to Numerai. Dashboard rule: `staked` badge only when the referenced full export is valid; otherwise `stale` flag. **Row binding:** when `staked` references a full export that is *not* the current one (staked older run, `current.json` repointed later), the dashboard renders the staked export as its own `family::full::<run_id>` row **bound to the staked record** (annotated `staked`), distinct from the current-full row — the badge and the row can never disagree about which artifact is staked.

**Validity vs upload acceptance** (SECONDARY 3): export validity is a *lifecycle* notion — it does **not** imply Numerai upload acceptance. `accept_promoted_artifact` (raw-output contract, official validator) remains the pre-upload gate for whatever is manually uploaded; the manual upload itself is out of scope (documented, §14).

## 6. Promotion (resolves B4/B6)

`promote_full_version(run_id, family, *, scope: "full" | "train_only", ...)` — single writer, one code path; scope controls fit data, target dir, and pointer handling:

| Aspect | `train_only` (partial) | `full` |
|---|---|---|
| Fit data | `train.parquet` only | train + validation |
| Target dir | `exports/partial/<run_id>/` | `exports/full/<run_id>/` |
| `current.json` | never touched | repointed atomically |
| Tier-4 gate | applies (same `override_gate` path) | same |
| RAM guard | scoped row count (below) | train+validation row count |
| Post-promotion cross-check | **required**: `scorecard.json` written before publication | n/a |

**Data access is scope-parameterized** (B4): `_full_history_frame` and the spawned-worker spec read only the scope's parquet — a `train_only` fit must never open `validation.parquet` (asserted by test). The RAM guard's `current_rows` uses the scope's file only.

**Memory claim corrected** (B4): measured sizes are train = 2,746,268 rows, validation = 4,107,040 rows (2026-08-26). A train-only fit is ~40% of the full window's rows — *smaller* than full, not "≈ full". The guard's single-point/curve extrapolation is applied per scope with the scope's row count; no new RAM ceiling is introduced.

**Publication atomicity** (B5): the export slot is staged at `exports/<scope>/.tmp-<run_id>/` (predict.pkl + sibling manifest + manifest.json + scorecard.json), then **atomically renamed into place** as `exports/<scope>/<run_id>/` — a single directory rename, and discovery ignores any `.tmp-`-prefixed entry. Lifecycle discovery never observes a half-written slot. On any failure (fit, gate, scoring), the temp dir is discarded and the promotion reports failure — no partial slot is ever published.

## 7. Cross-check scoring contract (resolves B5 — executable spec)

**Purpose:** the reference the user compares against Numerai platform diagnostics — same artifact, same validation eras, local vs platform.

- **Inputs** (four frames, joined on era + id): `predictions` (the staged partial artifact's output on validation eras); `targets` (validation targets; `main_target` from the run config); `meta_model` (the Numerai meta model — **required** for MMC); `features` (validation feature matrix — **required** for FNC). **No `benchmarks` frame** (`benchmarks=None` — benchmark-models are irrelevant to this check; this is the correct reading of "no benchmark join", which does *not* exclude meta_model/features).
- **Algorithm:** one existing call, no new scoring math —
  `evaluate_model(predictions, meta_model=…, benchmarks=None, features=…, targets=…, n_trials=<run's deflated-sharpe trials>, seed=<run seed>, horizon=<run horizon>, main_target=<run target>, backend="official")`.
  `backend="official"` forces the `numerai_tools` oracle (never the custom path). The oracle-parity contract already pins custom↔official agreement.
- **Metrics:** the resulting `MetricScorecard` — `corr`, `mmc`, `corr_sharpe_ac` as `MetricCell {value, ci_low, ci_high, n_eras}`, `fnc` as a scalar, plus `n_eras`, `deflated_sharpe`, `max_drawdown`, `burn_rate` — *exactly* the repo's data model (no invented shapes). Per-era series are taken from the scorecard's `era_series_stats` (corr/mmc/fnc per era). The platform's raw Sharpe is compared against `corr_sharpe_ac`'s AC-adjusted value with the raw per-era mean also reported — both are in the output.
- **Era policy:** reuse the existing scorecard rules verbatim — era-grouped only, degenerate-era handling (zero variance / <2 rows / non-finite) and `MIN_OVERLAP_ERAS` non-vacuity per `evaluate_model`; degenerate eras stay in the per-era series and count toward `n_eras` exactly as the scorecard counts them; no row-level CV (leakage-safety assertions unchanged).
- **Window persistence (recoverable without ignored files):** the **exact ordered era list** scored by this check is persisted in `scorecard.json` (which is git-tracked). The window is *not* derived from the ignored `validation_preds.parquet` at read time — the eras were recorded at scoring time, so the set survives heavy-file loss. `window` = `{first_era, last_era, eras: [ordered list]}`.
- **Output `scorecard.json` schema** (versioned, canonical):
  ```
  {schema_version: 2, run_id, family, scope: "train_only",
   window: {first_era, last_era, eras: ["0575", …]},
   scorecard: {corr: {value, ci_low, ci_high, n_eras},
               mmc: {value, ci_low, ci_high, n_eras},
               corr_sharpe_ac: {value, ci_low, ci_high, n_eras},
               fnc: <scalar>, n_eras, deflated_sharpe, max_drawdown, burn_rate},
   per_era: {corr: [{era, value}, …], mmc: […], fnc: […]},
   generated_at}
  ```
  Canonical serialization excludes `generated_at` (timing-strip discipline, mirroring the registry's canonical scorecard bytes). Per-era entries are labeled `{era, value}` objects — never positional.
- **Failure:** scoring failure aborts the partial publication (atomicity in §6) — an export without a `scorecard.json` is not a valid partial.

## 8. Dashboard

Unified schema gains `display_name` + `lifecycle_stage`; family rows show the lifecycle badge; detail rows: `family::full` (existing) and `family::partial::<run_id>` (new, one per valid partial export). Payload carries both new fields through the vanilla renderer; `staked` renders `stale` when the referenced export is invalid.

## 9. Storage abstraction & retargeting (resolves B8)

- New `nmr/paths.py`: `EXPERIMENTS_ROOT = REPO_ROOT / "experiments"` and pure helpers `experiment_dir(slug)`, `run_dir(slug, run_id)`, `export_dir(slug, scope, run_id)`, `champion_path()`. **Single place** for per-experiment path derivation; no module hardcodes `experiments/` strings.
- `run.artifacts_dir` is **repurposed, not overloaded**: it keeps its current meaning for the *shared machine cache* — default `REPO_ROOT / "artifacts"`. **Shared-path ownership (explicit, SECONDARY 4):** `artifacts/cache/` (data-fingerprint cache, neutralization cache), `artifacts/reports/` (RAM curve + RAM estimate, benchmark scorecards), and `artifacts/campaigns/` (campaign logs + evidence) all stay under `config.run.artifacts_dir` — `nmr/paths.py` provides `shared_cache_dir(...)` / `shared_reports_dir(...)` helpers so nothing derives them from an experiment dir. `artifacts_dir` **no longer determines run/export output location** — those derive from `EXPERIMENTS_ROOT` + family slug. Its stored value remains stripped from run-id payloads (existing rule).
- **Retarget list** (all path consumers move to `nmr/paths.py`): `nmr/registry.py`, `nmr/runner.py` (checkpoints, research `predict.pkl`, validation stage), `nmr/promote.py`, `nmr/families.py`, `nmr/dashboard.py`, `nmr/meta.py`, `nmr/submission.py`, `dashboard_ui/app.py`, `dashboard_ui/report.py`, `train_first_model.py`, `run_campaign.py`, `promote_model.py`, `rehearse_promotion.py`, `generate_dashboard.py`.
- **Campaigns spanning families** (`run_campaign.py`): the campaign runner keeps its own artifact root (`artifacts/campaigns/`) for campaign-level evidence; each constituent run writes to *its own* family's `experiments/<slug>/runs/<run_id>/`, resolved per run from that run's `run.name`. One campaign, many families — no shared per-model state.

## 10. Champion (resolves B3)

`experiments/champion.json` = `{run_id, experiment_slug, promoted_at}` — the slug is required because a bare run_id no longer locates the run. `RunRegistry.promote` / `promote_if_better` write this pointer and resolve candidates at `experiments/<slug>/runs/<run_id>/`. Dangling pointer → fail loud with the available families/runs (no mtime guessing — determinism discipline). Cross-family comparison is unchanged (scorecard values per run); only resolution changes. Concurrent writers: registry JSON writes stay temp + fsync + `os.replace` (existing atomic-write rule).

## 11. Determinism (resolves SECONDARY 1)

Precise acceptance claim: **under identical code identity, configurations differing only in artifact/experiment roots produce identical run_ids** — path fields are stripped from the run-id payload (existing `_strip_path_dependent_fields` behavior) and the data fingerprint excludes paths. Editing `nmr/*.py` legitimately changes run_ids (code identity is part of the hash) — that is not a violation.

## 12. Tests (resolves B9 — acceptance list, not implied coverage)

`tests/test_experiment_layout.py` (new) + retargeted existing:

1. **Layout round-trip**: synthetic run writes/reads `experiments/<slug>/runs/<run_id>/run.json` + preds; scan discovers runs; every file in §3 exists at its classified location.
2. **Lifecycle derivation is total** (B4-round-2): each filesystem state maps to exactly one stage — incl. the reviewer-flagged state (valid full export + missing/dangling `current.json` + no valid partial) ⇒ `degraded`; badge precedence `staked` > `full` > `degraded` > `partial` > `research`; `current.json` requirement for `full`; `staked` stale-flag when the referenced export is missing/invalid.
3. **`train_only` never opens validation** (B4): a train-only promotion fails if `validation.parquet` is accessed (spy on the data layer); `training_era_range`/`training_rows` equal the train file's; spawned worker runs with `include_validation=False`.
4. **RAM guard scope** (B4): guard uses the scope's row count; a `train_only` guard call does not scan `validation.parquet`.
5. **Partial publication atomicity** (B5): a forced scoring failure leaves no `exports/partial/<run_id>/` and no temp residue; discovery sees no slot.
6. **Cross-check scorecard** (B5): `scorecard.json` equals `evaluate_model(..., backend="official")` on the artifact's validation predictions (meta_model + features + targets joined on era/id; `benchmarks=None`); schema matches the repo's `MetricScorecard` shape (MetricCell fields, scalar `fnc`); per-era entries are labeled `{era, value}` and the exact ordered era list is persisted in `window.eras`; `generated_at` excluded from canonical bytes.
7. **Export validity** (B6): missing pickle / bad sibling hash / `load_predict` failure / scope-manifest mismatch / unknown run_id ⇒ invalid; multiple partial slots ⇒ one dashboard row each; ordering deterministic (ISO-8601 `promoted_at` descending, tie-break `run_id` ascending; unparseable timestamps sort last + flagged).
8. **Staked binding** (B7): staked referencing a different/older run ⇒ stale flag, never a `staked` badge; **staked-vs-current**: when the staked full export is not the current one, the staked export renders as its own row bound to the staked record, distinct from the current-full row.
9. **Display-name propagation + escaping**: `meta.json` display_name reaches the payload; special characters escape correctly in the vanilla renderer.
10. **Git retention** (B2/SECONDARY 1): the §3.1 rules track the classified files incl. root + nested `.gitkeep`, and ignore parquet/pkl/checkpoints (assert via `git check-ignore` on fixtures).
11. **Determinism** (SECONDARY 1): same code identity + different experiment roots ⇒ identical run_id.
12. **Rebuild refusal on fingerprint change** (B3-round-2): with a recorded fingerprint differing from the current data fingerprint, a rebuild/resume is refused (no silent stale reuse); with matching identity, checkpoint resume reproduces byte-identical outputs.
13. **Retarget**: every test referencing `artifacts/registry`, `artifacts/runs`, `artifacts/models` moves to the `experiments/` layout.
14. **Champion resolution** (B3): promote/promote_if_better write `{run_id, experiment_slug, promoted_at}`; dangling pointer fails loud.
15. **Shared paths** (SECONDARY 4): RAM curve/estimate, fingerprint cache, neutralization cache, campaign logs, benchmark reports resolve under `config.run.artifacts_dir`, never under an experiment dir.

## 13. Files

- Modify: `nmr/registry.py`, `nmr/runner.py`, `nmr/promote.py` (scope + cross-check + atomic staging), `nmr/families.py` (experiment scan, partial versions, validity predicate), `nmr/dashboard.py`, `nmr/meta.py`, `nmr/submission.py`, `nmr/config.py` (artifacts_dir comment/semantics), `dashboard_ui/app.py`, `dashboard_ui/report.py`, CLIs (`train_first_model.py`, `run_campaign.py`, `promote_model.py`, `rehearse_promotion.py`, `generate_dashboard.py`), `.gitignore`, `configs/example.yaml`
- New: `nmr/paths.py`, `experiments/` scaffold (`.gitkeep`), `docs/02-strategy/model-lifecycle.md`
- Tests: new `tests/test_experiment_layout.py`; retargeted existing path-dependent tests
- Docs (same commit, SSOT): `AGENTS.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `README.md`

## 14. Out of Scope

- Migrating legacy runs (registry wiped by the clean slate — by design).
- Renaming existing families (none remain; the convention applies going forward).
- Automating the Numerai upload/stake itself — the upload is a manual act; the stake is recorded post-hoc in `meta.json`, bound to a valid full export.
- Benchmark/fleet/campaign storage (stays in `artifacts/reports/` + `artifacts/campaigns/` — cross-family by nature).
- Re-creating the promising models from the 2026-08-26 roster (a separate campaign; the roster snapshot lives in `artifacts/dashboard.html` + `/tmp/model_roster.csv`).
- Changing run-id hashing, metric formulas, or the official-backend parity contract.
- External artifact store / cloud backup for heavy files — not in scope; reproducibility is scoped to "while the original inputs remain available" (§3), and the run record persists the fingerprint + identity to make that verifiable.
