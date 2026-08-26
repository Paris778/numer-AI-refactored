# Design Spec: Model Lifecycle & Self-Contained Experiment Layout (v2 — revised)

> Status: REVISED (2026-08-26) after critical review — resolves 9 blocking + 2 secondary findings. Supersedes the initial draft. Replaces the three global per-model pools (`artifacts/runs/`, `artifacts/registry/`, `artifacts/models/`) with self-contained per-family directories at repo root `experiments/`, introduces a human display-name naming layer over the machine slug, and defines the four-stage model lifecycle (research → partial → full → staked) with a new train-only (partial) promotion scope and a local cross-check scoring step.

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

Every file produced by the system is classified: **git-tracked** (small, identity-bearing record), **ignored** (heavy or reconstructable), or absent-by-design. Ignored pickles/parquets are rebuilt deterministically by re-running the recorded config (the checkpoint/resume discipline; identity-guarded so stale reuse is impossible); they are never auto-loaded from outside the repo (existing trusted-source rule). The git record preserves identity, provenance, and results even if all heavy files are lost.

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
```

`experiments/` tree scaffolding is kept via `.gitkeep` (as `artifacts/` does). Everything else (parquet, pkl, checkpoints) stays ignored by the blanket rule.

## 4. Naming (unchanged from draft)

**Display name layered over machine slug.** Slug = family dir + `run.name`, `^[a-z0-9_-]+$`, template `<theme>-<backend>-<vN>` going forward (e.g. `ender-xgb-v1`). Human `display_name` (e.g. `"Ender XGB v1 · medium"`) lives in `meta.json`, surfaces in dashboard labels, manifests, and docs; the hash appears only in tooltips/URIs. Lifecycle is a **badge** (`· partial` / `· full` / `· staked`), never a name mutation. `display_name` is not a run-config field (run-id determinism untouched).

## 5. Lifecycle (resolves B1/B6/B7)

Stages are **derived from filesystem state** (never a mutable counter), with badge precedence `staked` > `full` > `partial` > `research`:

- `research` — ≥1 `runs/<run_id>/run.json`, no valid exports.
- `partial` — ≥1 **valid** `exports/partial/<run_id>/` and no valid full.
- `full` — `exports/full/current.json` points at a **valid** full slot.
- `staked` — `meta.json.staked.status == "active"` AND the referenced export is valid; otherwise the badge shows the underlying stage and the staked record is flagged `stale` in the payload.

**Export validity predicate** (single definition, used by discovery and dashboard):

```
valid(export) ⇔ manifest.json present AND training_scope == dir scope
  AND predict.pkl present AND predict.pkl.manifest.json present
  AND sha256(predict.pkl) == sibling manifest value AND load_predict() succeeds
  AND manifest.family == slug AND manifest.promoted_from_run_id ∈ family runs/
  AND (scope == partial ⇒ scorecard.json present)
```

**Partial selection** (no single pointer): the dashboard renders one `family::partial::<run_id>` row per **valid** partial export; rows order by `promoted_at`. The family badge is `partial` iff ≥1 valid partial and no valid full.

**Staked binding** (B7): `meta.json.staked` is `{run_id, scope: "full", numerai_model_id, staked_at, status: "active" | "retired"}`, where `run_id` must reference a **valid full export** — the artifact actually uploaded to Numerai. Dashboard rule: `staked` badge only when the referenced full export is valid; otherwise `stale` flag.

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

## 7. Cross-check scoring contract (resolves B5)

**Purpose:** the reference the user compares against Numerai platform diagnostics — same artifact, same validation eras, local vs platform.

- **Inputs:** the published (staged) partial artifact's predictions on `validation.parquet` eras; target column from the run's config (`target`); era column `era`; no benchmark/meta join at this step.
- **Algorithm:** reuse the existing era-grouped scorecard machinery — `evaluate_model` / `MetricScorecard` with `backend="official"` forced (the `numerai_tools` oracle; never the custom path). No new scoring math — the oracle parity contract already pins the custom path to official.
- **Metrics:** per-era `corr`, `mmc`, `fnc` series + aggregates `corr.mean`, `corr.sharpe` (plain per-era Sharpe) and the repo's `corr_sharpe_ac` (AC-adjusted) — both reported, so the platform's raw Sharpe and the repo's adjusted Sharpe are both comparable.
- **Era policy:** reuse the existing scorecard rules verbatim — era-grouped only, degenerate-era handling (zero variance / <2 rows / non-finite), `MIN_OVERLAP_ERAS` non-vacuity, no row-level CV (leakage-safety assertions unchanged). Window = the validation eras covered by the run's `validation_preds.parquet`; recorded as `[first_era, last_era]` in the output.
- **Output `scorecard.json` schema** (versioned, canonical): `{schema_version: 1, run_id, family, scope: "train_only", window: [first, last], n_eras, metrics: {corr: {per_era: [...], mean, sharpe}, mmc: {...}, fnc: {...}, corr_sharpe_ac: {...}}, generated_at}`. Canonical serialization excludes `generated_at` (timing-strip discipline, mirroring the registry's canonical scorecard bytes).
- **Failure:** scoring failure aborts the partial publication (atomicity in §6) — an export without a scorecard is not a valid partial.

## 8. Dashboard

Unified schema gains `display_name` + `lifecycle_stage`; family rows show the lifecycle badge; detail rows: `family::full` (existing) and `family::partial::<run_id>` (new, one per valid partial export). Payload carries both new fields through the vanilla renderer; `staked` renders `stale` when the referenced export is invalid.

## 9. Storage abstraction & retargeting (resolves B8)

- New `nmr/paths.py`: `EXPERIMENTS_ROOT = REPO_ROOT / "experiments"` and pure helpers `experiment_dir(slug)`, `run_dir(slug, run_id)`, `export_dir(slug, scope, run_id)`, `champion_path()`. **Single place** for per-experiment path derivation; no module hardcodes `experiments/` strings.
- `run.artifacts_dir` is **repurposed, not overloaded**: it keeps its current meaning for the *shared machine cache* — default `REPO_ROOT / "artifacts"`, still home of `artifacts/cache/` (data-fingerprint cache, neutralization cache) and `artifacts/reports/`. It **no longer determines run/export output location** — those derive from `EXPERIMENTS_ROOT` + family slug. Its stored value remains stripped from run-id payloads (existing rule).
- **Retarget list** (all path consumers move to `nmr/paths.py`): `nmr/registry.py`, `nmr/runner.py` (checkpoints, research `predict.pkl`, validation stage), `nmr/promote.py`, `nmr/families.py`, `nmr/dashboard.py`, `nmr/meta.py`, `nmr/submission.py`, `dashboard_ui/app.py`, `dashboard_ui/report.py`, `train_first_model.py`, `run_campaign.py`, `promote_model.py`, `rehearse_promotion.py`, `generate_dashboard.py`.
- **Campaigns spanning families** (`run_campaign.py`): the campaign runner keeps its own artifact root (`artifacts/campaigns/`) for campaign-level evidence; each constituent run writes to *its own* family's `experiments/<slug>/runs/<run_id>/`, resolved per run from that run's `run.name`. One campaign, many families — no shared per-model state.

## 10. Champion (resolves B3)

`experiments/champion.json` = `{run_id, experiment_slug, promoted_at}` — the slug is required because a bare run_id no longer locates the run. `RunRegistry.promote` / `promote_if_better` write this pointer and resolve candidates at `experiments/<slug>/runs/<run_id>/`. Dangling pointer → fail loud with the available families/runs (no mtime guessing — determinism discipline). Cross-family comparison is unchanged (scorecard values per run); only resolution changes. Concurrent writers: registry JSON writes stay temp + fsync + `os.replace` (existing atomic-write rule).

## 11. Determinism (resolves SECONDARY 1)

Precise acceptance claim: **under identical code identity, configurations differing only in artifact/experiment roots produce identical run_ids** — path fields are stripped from the run-id payload (existing `_strip_path_dependent_fields` behavior) and the data fingerprint excludes paths. Editing `nmr/*.py` legitimately changes run_ids (code identity is part of the hash) — that is not a violation.

## 12. Tests (resolves B9 — acceptance list, not implied coverage)

`tests/test_experiment_layout.py` (new) + retargeted existing:

1. **Layout round-trip**: synthetic run writes/reads `experiments/<slug>/runs/<run_id>/run.json` + preds; scan discovers runs; every file in §3 exists at its classified location.
2. **Lifecycle derivation**: each stage derived correctly at each fs state; badge precedence; `current.json` requirement for `full`; `staked` stale-flag when the referenced export is missing/invalid.
3. **`train_only` never opens validation** (B4): a train-only promotion fails if `validation.parquet` is accessed (spy on the data layer); `training_era_range`/`training_rows` equal the train file's; spawned worker runs with `include_validation=False`.
4. **RAM guard scope** (B4): guard uses the scope's row count; a `train_only` guard call does not scan `validation.parquet`.
5. **Partial publication atomicity** (B5): a forced scoring failure leaves no `exports/partial/<run_id>/` and no temp residue; discovery sees no slot.
6. **Cross-check scorecard** (B5): `scorecard.json` equals a direct official-backend scoring of the artifact's validation predictions; schema fields present; `generated_at` excluded from canonical bytes.
7. **Export validity** (B6): missing pickle / bad sibling hash / `load_predict` failure / scope-manifest mismatch / unknown run_id ⇒ invalid; multiple partial slots ⇒ one dashboard row each, deterministic order.
8. **Staked binding** (B7): staked referencing a different/older run ⇒ stale flag, never a `staked` badge.
9. **Display-name propagation + escaping**: `meta.json` display_name reaches the payload; special characters escape correctly in the vanilla renderer.
10. **Git retention** (B2): the §3.1 rules track the classified files and ignore the heavy ones (assert via `git check-ignore` on fixtures).
11. **Determinism** (SECONDARY 1): same code identity + different experiment roots ⇒ identical run_id.
12. **Retarget**: every test referencing `artifacts/registry`, `artifacts/runs`, `artifacts/models` moves to the `experiments/` layout.
13. **Champion resolution** (B3): promote/promote_if_better write `{run_id, experiment_slug, promoted_at}`; dangling pointer fails loud.

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
- External artifact store / cloud backup for heavy files (reconstruction is by deterministic re-run; not in scope).
