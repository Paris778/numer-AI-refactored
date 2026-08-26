# Design Spec: Model Lifecycle & Self-Contained Experiment Layout (v4 — final contract)

> Status: REVISED (2026-08-26, 3rd review round consolidated) — resolves the 3 blocking + 4 secondary findings of the third-pass review plus the audit's additional blockers. Supersedes v1–v3. Replaces the three global per-model pools (`artifacts/runs/`, `artifacts/registry/`, `artifacts/models/`) with self-contained per-family directories at repo root `experiments/`, introduces a human display-name naming layer over the machine slug, and defines the six-state model lifecycle (uninitialized → research → partial → degraded → full → staked) with a new train-only (partial) promotion scope and a local cross-check scoring step.

## 1. Mission

Two problems motivate this change:

1. **Illegibility.** A model is known by `brb1-xgb-v6 · 2610a99d` — a cryptic slug plus a 64-hex hash, with several near-identical runs per family differing only by hash. There is no human-facing name, no visible lifecycle stage, and no per-family record of what was done and why.
2. **Fragility + scatter.** Per-model state is spread across three global pools with no versioning and no per-experiment documentation. The 2026-08-26 session's registry-loss incident (a junction/worktree deletion wiped all 29 run dirs; the registry was git-ignored and unrecoverable) — **session history, not repository evidence** — motivated versioning the small human-relevant record.

The change rides the clean slate: no legacy runs to migrate — the new layout is the baseline.

## 2. Domain model & vocabulary (resolves B1-round-1, audit vocabulary)

**Vocabulary is a single mapping — one term per concept, never interchangeable:**

| Concept | Term | Used as |
|---|---|---|
| Research lineage container | **family** | `experiments/<slug>/` |
| One experiment execution | **run** | `runs/<run_id>/` |
| Promotion request scope | **`train_only`** | `promote_full_version(scope="train_only")` — a *request* |
| Persisted lifecycle scope | **`partial`** | the export's persisted `training_scope: "partial"` (a *state*) |
| Full-history scope | **`full`** | persisted + request |
| Deploy artifact | **export** | `exports/<scope>/<run_id>/` |

`train_only` names the fit request; `partial` names the resulting artifact state. `training_scope` is persisted as `"partial"` or `"full"` — never `"train_only"`.

- **Family**: contains **many runs** and **many exports**. Family-level `meta.json`, `README.md`, `base_config.yaml`, display name, and lifecycle badge describe the family.
- **Run**: one experiment execution; immutable once recorded. Each run's **effective** config lives in its own `run.json`; the family `base_config.yaml` is a non-authoritative reference copy (explicitly labeled as such — the authoritative config is per-run).
- **Export**: an immutable deploy artifact for exactly one run at one scope; never mutated after publication; new promotion = new slot.

Ownership:

| Item | Owner | Notes |
|---|---|---|
| display_name | family (`meta.json`) | human label; ordinary editable metadata (audit: no hash impact; dashboard + README both derive from `meta.json` — single source) |
| lifecycle badge | family | derived; stage = family's highest valid stage |
| staked record | family (`meta.json`) | bound to exactly one export |
| config | run (`run.json`) authoritative; family `base_config.yaml` reference | labeled non-authoritative |
| scorecards | run (`run.json`) + partial export (`scorecard.json`) | research CV vs cross-check |

**Family creation** (audit): the scaffold (`base_config.yaml` copy + `meta.json` with `display_name` + an empty `README.md` placeholder) is created **atomically with the first run's `run.json`** — no family directory exists before its first run is recorded. A human may subsequently edit `README.md`/`display_name`. A directory present without `run.json` (e.g. hand-created) is the explicit `uninitialized` state (§5).

## 3. Layout & persisted-file inventory (resolves B2-round-1)

```
experiments/<slug>/                          # family = research lineage
  README.md                    [git]         # human record (placeholder at creation, human-authored after)
  base_config.yaml             [git]         # family base config — NON-authoritative reference copy
  meta.json                    [git]         # {display_name, staked: {run_id, scope, numerai_model_id, staked_at, status}}
  runs/<run_id>/
    run.json                   [git]         # scorecard + provenance + effective config + rebuild identity (§3.2)
    oof.parquet                [ignored]     # per-fold OOF preds
    validation_preds.parquet   [ignored]     # era-batched validation preds
    predict.pkl                [ignored]     # research deploy closure (runner-built)
    predict.pkl.manifest.json  [ignored]     # sibling hash manifest; travels with its pickle
    oof_checkpoints/           [ignored]     # resume state (code/device identity guarded)
    deploy_checkpoints/        [ignored]
    validation_checkpoints/    [ignored]
  exports/
    partial/<run_id>/
      export.json              [git]         # promotion record: config, provenance, tier-4 receipts, training_scope: "partial"
      scorecard.json           [git]         # local cross-check reference (§7)
      predict.pkl              [ignored]     # train-only artifact
      predict.pkl.manifest.json [ignored]    # deploy sibling hash manifest
    full/<run_id>/
      export.json              [git]         # training_scope: "full"
      predict.pkl              [ignored]
      predict.pkl.manifest.json [ignored]
    full/current.json          [git]         # active full-version pointer (versioned: it IS the "what's live" record)
experiments/champion.json      [git]         # global best-run pointer {run_id, experiment_slug, promoted_at}
```

Naming decisions (audit): the git-tracked promotion record is **`export.json`** (never confused with the ignored `predict.pkl.manifest.json`); the family reference config is **`base_config.yaml`** (non-authoritative, labeled).

Every file is classified: **git-tracked** (small, identity-bearing), **ignored** (heavy or reconstructable), or absent-by-design.

**Reproducibility claim (scoped, B3-round-2):** ignored artifacts are reproducible by re-running the recorded config **only while the original inputs remain available** — code identity, dependency environment, device state, and the data snapshot at scoring time. The data fingerprint is a *snapshot marker*, not a content snapshot (documented detection limits: restated feature values within unchanged schema/row-count/era-stats are undetected; validation grows over time). Therefore:

### 3.1 Rebuild identity (persisted — resolves B3-round-3, audit)

`run.json`'s manifest gains named fields (currently the fingerprint is hashed but not persisted; runner.py:635):

```
run.json:
  manifest:
    data_fingerprint: <sha256>      # persisted value used in run-id computation
    code_fingerprint: <sha256>      # portable code identity (no absolute paths)
    environment: <portable spec>    # dependency versions, normalized
    pipeline_device: <config device>
    oof_device: <actual fit device>
```

**Rebuild/resume refusal rule:** a rebuild or checkpoint-resume compares the current `data_fingerprint` + `code_fingerprint` + `environment` + device against the recorded values; any mismatch ⇒ refuse (no silent stale reuse — delete-to-refit discipline, mirroring the existing checkpoint identity rule). The refusal applies to OOF, validation predictions, and deploy artifacts alike. Acceptance tests assert "refuses when fingerprints differ" — not that every data mutation is detected.

### 3.2 Git rules (exact)

```
experiments/**
!experiments/
!experiments/**/
!experiments/*/README.md
!experiments/*/base_config.yaml
!experiments/*/meta.json
!experiments/*/runs/*/run.json
!experiments/*/exports/**/export.json
!experiments/*/exports/**/scorecard.json
!experiments/*/exports/full/current.json
!experiments/champion.json
!experiments/**/.gitkeep
```

Tree scaffolding via `.gitkeep` (root + nested). Everything else (parquet, pkl, checkpoints) stays ignored.

## 4. Naming

**Display name layered over machine slug.** Slug = family dir + `run.name`, `^[a-z0-9_-]+$`, template `<theme>-<backend>-<vN>` going forward (e.g. `ender-xgb-v1`). Human `display_name` (e.g. `"Ender XGB v1 · medium"`) lives in `meta.json`, surfaces in dashboard labels, manifests, and docs; the hash appears only in tooltips/URIs. Lifecycle is a **badge** (`· partial` / `· full` / `· staked`), never a name mutation. `display_name` is not a run-config field (run-id determinism untouched); editing it is an ordinary metadata edit, audited by git history, reflected everywhere from `meta.json` (single source).

## 5. Lifecycle (total function — resolves B4-round-2, audit empty-family)

Stages derived from filesystem state; the derivation is a **total function** — every state maps to exactly one stage. Badge precedence: `staked` > `full` > `degraded` > `partial` > `research` > `uninitialized`.

| State | Definition |
|---|---|
| `uninitialized` | family dir exists, **no** `runs/<run_id>/run.json` (hand-created scaffold; transient by §2 creation rule) |
| `research` | ≥1 `run.json`, no valid exports |
| `partial` | ≥1 **valid** `exports/partial/<run_id>/`, no valid full export |
| `degraded` | ≥1 **valid** full export, but `current.json` missing/dangling (valid full exists; pointer broken) |
| `full` | `current.json` points at a **valid** full slot |
| `staked` | `meta.json.staked.status == "active"` AND the referenced export is valid; else the underlying stage shows and the staked record is flagged `stale` |

**`staked` never hides a broken pointer** (audit): the payload carries **both** facts — `lifecycle_stage` (precedence result) **and** `current_full_status` (`full` / `degraded` / `none`). A `staked` + `degraded` combination is rendered explicitly.

**Export identity binding** (audit, blocker): a slot is valid only when **all** identities agree:

```
pointer.run_id == slot dir run_id  (current.json → exports/full/<run_id>/)
manifest.promoted_from_run_id == slot dir run_id
export.json.family == directory slug
run.json.run_id == slot dir run_id          (the promoted run, when present)
run.json manifest family == directory slug
```

A copied/mislabeled slot fails the predicate (no silent mismatch).

**Export validity predicate** (used by discovery and dashboard):

```
valid(export) ⇔ export.json present AND training_scope == dir scope ("partial"|"full")
  AND predict.pkl present AND predict.pkl.manifest.json present
  AND sha256(predict.pkl) == sibling manifest value AND load_predict() succeeds
  AND identity binding (§5 above) holds
  AND (scope == partial ⇒ scorecard.json present)
```

**Partial selection:** one `family::partial::<run_id>` row per **valid** partial export; ordering deterministic — `promoted_at` (must parse ISO-8601; unparseable sorts last + flagged) descending, tie-break `run_id` ascending.

**Staked binding:** `meta.json.staked = {run_id, scope: "full", numerai_model_id, staked_at, status: "active" | "retired"}` — `run_id` must reference a **valid full export** (the artifact actually uploaded). When the staked export is not the current one, it renders as its own `family::full::<run_id>` row **bound to the staked record**, distinct from the current-full row.

**Validity ≠ upload acceptance:** lifecycle validity does **not** imply Numerai upload acceptance; `accept_promoted_artifact` remains the pre-upload gate; the manual upload is out of scope (§14).

## 6. Promotion (resolves B4-round-1, audit re-promotion)

`promote_full_version(run_id, family, *, scope: "train_only" | "full", ...)` — single writer, one code path:

| Aspect | `train_only` (→ partial export) | `full` |
|---|---|---|
| Fit data | `train.parquet` only | train + validation |
| Persisted `training_scope` | `"partial"` | `"full"` |
| Target dir | `exports/partial/<run_id>/` | `exports/full/<run_id>/` |
| `current.json` | never touched | repointed atomically |
| Tier-4 gate | applies (same `override_gate` path) | same |
| RAM guard | scoped row count | train+validation row count |
| Post-fit cross-check | **required**: `scorecard.json` before publication | n/a |

**Fit-phase data isolation** (B1-round-3): the **fit phase** (`_full_history_frame`, spawned-worker spec) reads only the scope's parquet — a `train_only` fit must never open `validation.parquet`. The **post-fit cross-check phase** (a separate, later phase of the same promotion) is *permitted and required* to read validation data. The acceptance test spies **only the fit phase** — never the whole promotion.

**Memory:** measured sizes (2026-08-26): train = 2,746,268 rows, validation = 4,107,040 rows. A train-only fit is ~40% of the full window's rows — smaller. The guard's extrapolation uses the scope's row count.

**Publication atomicity:** the export slot is staged at `exports/<scope>/.tmp-<run_id>/` (predict.pkl + sibling manifest + export.json + scorecard.json), then **atomically renamed** into place as `exports/<scope>/<run_id>/` — one directory rename; discovery ignores `.tmp-`-prefixed entries. Failure at any point (fit, gate, scoring) discards the temp dir and reports failure — a half-written slot never appears.

**Re-promotion (audit, money-path blocker):** exports are immutable — promoting an already-present `exports/<scope>/<run_id>/` slot raises `ValueError` before any write. `force=True` is retained **only** for repointing `current.json` at a *different existing* full slot (or repairing a dangling pointer) — it never overwrites or replaces an immutable slot. Windows directory replacement (rename onto an existing dir fails) is therefore never exercised on slots: staging → rename targets an absent path.

## 7. Cross-check scoring contract (resolves B5, B2-round-3, B3-round-3 — executable)

**Purpose:** the reference the user compares against Numerai platform diagnostics — same artifact, same validation eras, local vs platform.

- **Inputs** (four frames, joined on era + id): `predictions` (staged partial artifact's output on validation eras); `targets` (validation targets; `main_target` from the run config); `meta_model` (required for MMC); `features` (required for FNC). **No `benchmarks` frame** (`benchmarks=None` — benchmark-models are irrelevant here).
- **Algorithm — new function, no signature breakage:** `nmr/scorecard.py` gains
  `evaluate_cross_check(predictions, *, meta_model, features, targets, horizon, main_target, seed) -> CrossCheckResult`.
  It reuses the **same internal per-era computation path** as `evaluate_model` (no duplicated math) and returns `{scorecard: MetricScorecard, per_era: {corr: [{era, value}], mmc: [...], fnc: [...]}, raw_sharpe: float}`. `evaluate_model` itself is **unchanged** (compat; its per-era locals stay local).
- **Replay parameters (fixed, B3-round-3):** cross-check scoring uses **fixed named module constants**, documented and pinned: `n_trials=1` (matches the runner's existing call), `n_boot`/`alpha`/`pf`/`clip`/`sr0_benchmark` = `evaluate_model`'s defaults. The **effective values** are persisted in `scorecard.json` (a replay record). No config-schema change.
- **Metrics:** the repo's real `MetricScorecard` — `corr`, `mmc`, `corr_sharpe_ac` as `MetricCell {value, ci_low, ci_high, n_eras}`, `fnc` scalar, `n_eras`, `deflated_sharpe`, `max_drawdown`, `burn_rate` — plus `raw_sharpe` (plain per-era Sharpe of corr) reported alongside AC-adjusted `corr_sharpe_ac` so the platform's raw Sharpe and the repo's adjusted Sharpe are both comparable.
- **Era policy:** existing scorecard rules verbatim — era-grouped, degenerate-era handling (zero variance / <2 rows / non-finite) and `MIN_OVERLAP_ERAS` non-vacuity; degenerate eras stay in the per-era series and count toward `n_eras` exactly as the scorecard counts them; no row-level CV.
- **Window persistence:** the **exact ordered era list** scored is persisted in the git-tracked `scorecard.json` — never derived from the ignored `validation_preds.parquet` at read time.
- **Output `scorecard.json` schema** (versioned, canonical):
  ```
  {schema_version: 3, run_id, family, scope: "partial",
   window: {first_era, last_era, eras: ["0575", …]},
   scorecard: {corr: {value, ci_low, ci_high, n_eras},
               mmc: {value, ci_low, ci_high, n_eras},
               corr_sharpe_ac: {value, ci_low, ci_high, n_eras},
               fnc: <scalar>, n_eras, deflated_sharpe, max_drawdown, burn_rate},
   raw_sharpe: <float>,
   replay: {n_trials, n_boot, alpha, pf, clip, sr0_benchmark, backend: "official"},
   per_era: {corr: [{era, value}, …], mmc: […], fnc: […]},
   generated_at}
  ```
  Per-era entries are labeled `{era, value}` objects — never positional. Canonical serialization excludes `generated_at` (timing-strip discipline). `backend: "official"` is forced (never the custom path).
- **Failure:** scoring failure aborts the partial publication (§6) — an export without `scorecard.json` is not a valid partial.

## 8. Dashboard

Unified schema gains `display_name` + `lifecycle_stage` + `current_full_status`; family rows show the lifecycle badge (with `stale` flag on broken staked refs and `degraded` surfaced under `current_full_status`). Row IDs are uniform: `family::full::<run_id>` and `family::partial::<run_id>` (no legacy `family::full` alias — `current.json` provides the "current" fact, not a row-id). **Partial rows are diagnostic-only (audit):** they do **not** participate in the cross-family leaderboard ranking — `EVALUABLE_ROWS` becomes `source not in ("full", "partial")`; their cross-check metrics render on the family detail. Staked rows bind to the staked record as defined in §5.

## 9. Storage abstraction & retargeting (resolves B8, audit cross-family)

- New modules (audit's cleaner structure, adopted):
  - `nmr/paths.py` — pure path derivation: `EXPERIMENTS_ROOT`, `experiment_dir(slug)`, `run_dir(slug, run_id)`, `export_dir(slug, scope, run_id)`, `champion_path()`, plus shared helpers `shared_cache_dir()` / `shared_reports_dir()` (always under `config.run.artifacts_dir`).
  - `nmr/lifecycle.py` — export validity, stage derivation (total function), staked binding, deterministic ordering.
  - `nmr/experiment_store.py` — run/export persistence, atomic directory publication, scaffold creation.
  - `nmr/registry.py` — **global run comparison + champion pointer only** (cross-family: `list()`, `best()`, `promote_if_better()` iterate `experiments/*/runs/*/run.json`, validating family slug against the directory; `promote_if_better` compares scorecards across families — comparison unchanged, resolution changed).
  - `nmr/families.py` — compatibility-facing discovery wrapper, delegating validity/lifecycle to `nmr/lifecycle.py`.
- `run.artifacts_dir` is **repurposed**: shared machine cache only — default `REPO_ROOT / "artifacts"`; `artifacts/cache/` (fingerprint cache, neutralization cache), `artifacts/reports/` (RAM curve/estimate, benchmark scorecards), `artifacts/campaigns/` stay under it. It **no longer determines run/export output location**. Stored value stays stripped from run-id payloads.
- **Retarget list:** `nmr/registry.py`, `nmr/runner.py`, `nmr/promote.py`, `nmr/families.py`, `nmr/dashboard.py`, `nmr/meta.py`, `nmr/submission.py`, `dashboard_ui/app.py`, `dashboard_ui/report.py`, `train_first_model.py`, `run_campaign.py`, `promote_model.py`, `rehearse_promotion.py`, `generate_dashboard.py`.
- **Champion concurrency (audit):** `promote_if_better`'s read-compare-write is **not** atomic; the spec adopts a **single-writer operational invariant** — registry pointer writes happen only from CLI/runner entry points, never from concurrent processes — documented in AGENTS.md. (A lock/CAS is out of scope; the invariant is the contract.)

## 10. Champion (resolves B3-round-1)

`experiments/champion.json` = `{run_id, experiment_slug, promoted_at}` — the slug is required because a bare run_id no longer locates the run. Resolution: `experiments/<slug>/runs/<run_id>/`. Dangling pointer → fail loud with available families/runs (no mtime guessing). Writes stay temp + fsync + `os.replace` (atomic-write rule), under the single-writer invariant (§9).

## 11. Determinism

Precise acceptance claim: **under identical code identity, configurations differing only in artifact/experiment roots produce identical run_ids** — path fields are stripped from the run-id payload and the data fingerprint excludes paths. Editing `nmr/*.py` legitimately changes run_ids (code identity is part of the hash) — not a violation. The persisted rebuild identity (§3.1) records the fingerprint + code/env/device used, making the reproducibility boundary verifiable.

## 12. Tests (acceptance list)

`tests/test_experiment_layout.py` (new) + retargeted existing:

1. **Layout round-trip**: synthetic run writes/reads `experiments/<slug>/runs/<run_id>/run.json` + preds; scan discovers runs; every §3 file at its classified location.
2. **Lifecycle total function**: every fs state maps to exactly one stage — incl. `uninitialized` (dir, no run.json) and `degraded` (valid full + dangling pointer); precedence `staked` > `full` > `degraded` > `partial` > `research` > `uninitialized`; `staked` + `degraded` exposed together (`lifecycle_stage` + `current_full_status`).
3. **Fit-phase validation isolation** (B1-round-3): the `train_only` **fit** fails if `validation.parquet` is opened (spy on the data layer during fit only); `training_era_range`/`training_rows` equal the train file's; spawned worker runs with `include_validation=False`; the **post-fit cross-check phase** is asserted to open validation (allowed).
4. **RAM guard scope**: guard uses the scope's row count; a `train_only` guard call does not scan validation.
5. **Publication atomicity**: a forced scoring failure leaves no `exports/partial/<run_id>/` and no `.tmp-` residue; discovery sees no slot.
6. **Cross-check scorecard** (B2/B3-round-3): `scorecard.json` equals `evaluate_cross_check(..., backend official)` on the artifact's validation predictions (meta_model + features + targets joined on era/id; `benchmarks` absent); schema matches the repo's `MetricScorecard` shapes + labeled `{era, value}` per-era entries + `raw_sharpe` + persisted `window.eras` + `replay` record; `generated_at` excluded from canonical bytes.
7. **Export validity + identity binding** (audit): missing pickle / bad sibling hash / `load_predict` failure / scope mismatch / **directory-vs-manifest-vs-pointer-vs-run.json identity disagreement** ⇒ invalid; multiple partials ⇒ one row each, deterministic ordering (ISO-8601 desc, `run_id` tie-break, unparseable last + flagged).
8. **Staked binding**: staked referencing a missing/invalid/different export ⇒ stale flag + underlying stage, never a `staked` badge; staked-vs-current renders two distinct rows.
9. **Display-name propagation + escaping**: `meta.json` display_name reaches the payload; special characters escape correctly in the vanilla renderer.
10. **Git retention**: §3.2 rules track the classified files incl. root + nested `.gitkeep`, ignore parquet/pkl/checkpoints (via `git check-ignore` on fixtures).
11. **Determinism**: same code identity + different experiment roots ⇒ identical run_id.
12. **Rebuild refusal** (B3-round-3): recorded fingerprint/code/env/device differing from current ⇒ resume/rebuild refused (OOF, validation, and deploy alike); with matching identity, checkpoint resume reproduces byte-identical outputs.
13. **Retarget**: every test referencing `artifacts/registry`, `artifacts/runs`, `artifacts/models` moves to the `experiments/` layout.
14. **Champion**: promote/promote_if_better write `{run_id, experiment_slug, promoted_at}`; cross-family iteration validates slug; dangling pointer fails loud.
15. **Partial non-ranking** (audit): partial rows carry cross-check metrics on the family detail but are excluded from leaderboard ranking (`EVALUABLE_ROWS`).
16. **Re-promotion rejection** (audit): promoting an existing slot raises before any write; `force=True` repoints `current.json` only, never replaces a slot.

## 13. Files

- Modify: `nmr/scorecard.py` (new `evaluate_cross_check` + `CrossCheckResult`; `evaluate_model` unchanged), `nmr/registry.py`, `nmr/runner.py`, `nmr/promote.py`, `nmr/families.py`, `nmr/dashboard.py`, `nmr/meta.py`, `nmr/submission.py`, `nmr/config.py` (artifacts_dir semantics comment), `dashboard_ui/app.py`, `dashboard_ui/report.py`, CLIs (`train_first_model.py`, `run_campaign.py`, `promote_model.py`, `rehearse_promotion.py`, `generate_dashboard.py`), `.gitignore`, `configs/example.yaml`
- New: `nmr/paths.py`, `nmr/lifecycle.py`, `nmr/experiment_store.py`, `experiments/` scaffold (`.gitkeep`), `docs/02-strategy/model-lifecycle.md`
- Tests: new `tests/test_experiment_layout.py` (+ cross-check tests in `tests/test_scorecard.py` / `tests/test_promote.py`); retargeted existing path-dependent tests
- Docs (same commit, SSOT): `AGENTS.md` (toolkit table, hazards — incl. the junction-deletion hazard and the single-writer champion invariant), `ARCHITECTURE.md` (incl. the stale artifact-layout section at `ARCHITECTURE.md:292`), `CONTRIBUTING.md`, `README.md`

## 14. Out of Scope

- Migrating legacy runs (registry wiped by the clean slate — by design).
- Renaming existing families (none remain; the convention applies going forward).
- Automating the Numerai upload/stake — manual act; stake recorded post-hoc bound to a valid full export.
- Benchmark/fleet/campaign storage (stays in `artifacts/reports/` + `artifacts/campaigns/` — cross-family by nature).
- Re-creating the promising models from the 2026-08-26 roster (a separate campaign; the roster snapshot lives in `artifacts/dashboard.html` + `/tmp/model_roster.csv`).
- Changing run-id hashing, metric formulas, or the official-backend parity contract.
- External artifact store / cloud backup for heavy files (reproducibility scoped to "while original inputs remain available"; §3.1 makes the boundary verifiable).
- Champion comparison locking/CAS (single-writer invariant is the contract, §9).
