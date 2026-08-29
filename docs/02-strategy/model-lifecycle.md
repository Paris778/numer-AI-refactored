# Model Lifecycle & Experiment Workflow

The operational guide for the self-contained experiment layout (`experiments/`)
and the six-state model lifecycle that drives it. Layout, schemas, and module
specs live in [`ARCHITECTURE.md`](../../ARCHITECTURE.md) (§N, §X–§Z, Model
Families); this document is the *workflow* — how a model moves from research to
staked, how the states are derived from disk, and how we operate the store.
Implementation of record: `nmr/paths.py` (layout), `nmr/lifecycle.py`
(derivation), `nmr/experiment_store.py` (persistence).

## 1. The lifecycle

Six stages, derived by `nmr.lifecycle.derive_stage(family, staked)` as a
**total function over filesystem state** — every possible on-disk state maps to
exactly one stage:

| Stage | Derived when |
|---|---|
| `uninitialized` | family directory exists but has **no** `runs/<run_id>/run.json` (a hand-created scaffold; transient — a family is normally created atomically with its first run, §5) |
| `research` | ≥ 1 `run.json`, no valid exports |
| `partial` | ≥ 1 **valid** `exports/partial/<run_id>/`, no valid full export |
| `degraded` | ≥ 1 **valid** full export, but `current.json` is missing or dangling (a valid full exists; the pointer is broken) |
| `full` | `current.json` points at a **valid** full slot |
| `staked` | `meta.json.staked.status == "active"` AND the staked `run_id` is a **valid** full export |

**Export validity** (the predicate behind every "valid" above,
`nmr.lifecycle.valid_export`): the slot has `export.json` whose `family` matches
the directory slug, `promoted_from_run_id` equals the slot-dir `run_id`, and
`training_scope` equals the directory scope (`"partial"`/`"full"`); `predict.pkl`
and its sibling `predict.pkl.manifest.json` are present and the SHA256 agrees;
`load_predict()` succeeds (hash-verified loadability — trusted-source rule);
and a `partial` slot additionally requires `scorecard.json`. Identity binding is
strict: slot-dir `run_id` == `promoted_from_run_id` == family slug, AND the
**run record is present and agrees** — `experiments/<family>/runs/<run_id>/
run.json` must exist, its payload `run_id` must equal the slot `run_id`, and its
manifest config `run.name` (when present) must equal the family (2026-08-26
review, BLOCKING 2). An export without a run record is an **orphan** — invalid,
never render-valid; a copied or mislabeled slot fails the predicate. Malformed
numeric metadata (`training_rows: NaN`, non-numeric strings) invalidates that
slot and is contained — one bad export never aborts a scan (SECONDARY 3).

**Badge precedence:** `staked` > `full` > `degraded` > `partial` > `research` >
`uninitialized`. The badge is the highest valid stage the filesystem supports;
the derivation is a total function, so no state is ambiguous.

**`staked` never hides a broken pointer.** `derive_stage` returns **two** facts
— `(lifecycle_stage, current_full_status)` — and the dashboard renders both: a
family whose stake references an invalid/missing export keeps the `stale` flag
while the underlying stage (e.g. `full`) shows. `current_full_status` is
`"full"` when the pointer resolves to a valid full slot, `"degraded"` when valid
full slots exist but the pointer is missing, dangling, or carries a non-hex
`run_id` (treated as corrupt — 2026-08-29 re-review), `"none"` otherwise.

**Surfacing in the dashboard:** `nmr/dashboard.UNIFIED_SCHEMA` carries
`display_name`, `lifecycle_stage`, `current_full_status`, and `stale` per
family. `stale = staked is not None and staked.status == "active" and
stage != "staked"` — a broken staked reference. `degraded` appears under
`current_full_status` (a valid full export whose `current.json` points nowhere).
Full and partial rows are **diagnostic-only**: one `family::<scope>::<run_id>`
row per VALID slot (partials carry their cross-check cells from the slot's
`scorecard.json`), excluded from leaderboard ranking (`EVALUABLE_ROWS` =
trained + benchmark) — never ranked, never charted as candidates.

## 2. The workflow

The money path is six steps. Steps 3 and 6 are manual acts (§5).

```
research → partial → upload → compare → full → stake
```

1. **Research** — `ExperimentRunner(cfg).run(deploy=...)` trains the CV OOF and
   (when the validation scorecard is enabled) the validation stage. The run is
   recorded under `experiments/<slug>/runs/<run_id>/` (`run.json` + `oof.parquet`
   + `validation_preds.parquet`), with the rebuild-identity fields (§4). The
   scorecard here is the *research* scorecard — CV OOF on the final fold's
   validation eras.
2. **Partial** — the promotion writer `promote_full_version(run_id, family, scope="train_only")` (a library call; `promote_model.py` exposes it as `--scope train_only`, default `full`): a **train-only** fit (never opens
   `validation.parquet` during the fit) published as the immutable slot
   `exports/partial/<run_id>/` plus a post-fit **cross-check** `scorecard.json`
   scored on validation eras through the official backend (`evaluate_cross_check`).
   `training_scope` persists as `"partial"` — never `"train_only"` (the request
   scope is `train_only`; the artifact state is `partial`).
3. **Upload** — manually upload the partial's `predict.pkl` to Numerai (Model
   Uploads). `accept_promoted_artifact` (raw output vs the official validator)
   remains the pre-upload gate; the upload itself is a manual act.
4. **Compare** — the platform reports live diagnostics for the uploaded model.
   Compare them against the local `scorecard.json`: same artifact, same
   validation-era window, local-vs-platform. **The partial is the honest
   cross-check instrument** — it never saw validation during training, so its
   local diagnostics are a fair expectation of what the platform should report.
5. **Full** — `promote_model.py` (the writer's default `scope="full"`) trains on
   train+validation and publishes the immutable slot `exports/full/<run_id>/`,
   repointing the atomic `current.json` pointer. The family moves to `full`.
6. **Stake** — a manual act: record the stake in `experiments/<slug>/meta.json`
   (`staked: {run_id, scope: "full", numerai_model_id, staked_at, status}`),
   bound to a **valid full export** (the artifact actually uploaded).

**The in-sample warning — never skip step 4's meaning.** A full-history model
was trained on train+validation, so the validation eras are **in-sample** for
it: the platform's *historical* diagnostics for the full model over those eras
are meaningless — they reflect eras the model memorized, not out-of-sample
skill. That is why the partial (train-only) exists: its `scorecard.json` is the
reference for historical diagnostics. The full model's **forward** diagnostics
(eras after its training cutoff) remain meaningful — that is what the platform
will actually report once it goes live.

## 3. Naming

- **Slug** — the family directory name and `config.run.name`; template
  `<theme>-<backend>-<vN>` going forward (e.g. `ender-xgb-v1`), validated by
  `nmr.paths.validate_slug` against `^[a-z0-9_-]+$` (lowercase-only — prevents
  case-collision overwrites on case-insensitive filesystems).
- **`display_name`** — the human label (e.g. `"Ender XGB v1 · medium"`), owned
  by `meta.json` (`{"display_name": ..., "staked": ...}`). It is the **single
  source**: the dashboard and family docs derive from it, never a parallel copy.
  Editing it is an ordinary metadata edit (audited by git), and it is not a
  run-config field — run-id determinism is untouched.
- **Hash only in tooltips** — the 64-hex `run_id` appears in tooltips/URIs and
  slot paths, never in the display name.
- **Lifecycle is a badge, never a name mutation** — `Ender XGB v1 · partial` /
  `· full` / `· staked` renders as a suffix badge; the display name itself never
  changes when the stage advances.

## 4. Layout

The full tree (spec §3; schemas in `ARCHITECTURE.md` §N):

```
experiments/
├── champion.json                    # {run_id, experiment_slug, promoted_at} — global best-run pointer [git]
└── <slug>/                          # family = research lineage
    ├── README.md                    # human record: what was done, decisions, results [git]
    ├── base_config.yaml             # family base config — NON-authoritative reference copy [git]
    ├── meta.json                    # {display_name, staked: {run_id, scope, numerai_model_id, staked_at, status}} [git]
    ├── runs/<run_id>/
    │   ├── run.json                 # scorecard + provenance + effective config + rebuild identity [git]
    │   ├── oof.parquet              # per-fold OOF preds [ignored]
    │   ├── validation_preds.parquet # era-batched validation preds [ignored]
    │   ├── predict.pkl              # research deploy closure (runner-built, deploy=True) [ignored]
    │   ├── predict.pkl.manifest.json# sibling hash manifest [ignored]
    │   ├── oof_checkpoints/         # resume state, code/device identity-guarded [ignored]
    │   ├── deploy_checkpoints/      # [ignored]
    │   └── validation_checkpoints/  # [ignored]
    └── exports/
        ├── partial/<run_id>/
        │   ├── export.json          # promotion record: config, provenance, tier-4 receipts, training_scope: "partial" [git]
        │   ├── scorecard.json       # local cross-check reference (the step-4 instrument) [git]
        │   ├── predict.pkl          # train-only artifact [ignored]
        │   └── predict.pkl.manifest.json [ignored]
        ├── full/<run_id>/
        │   ├── export.json          # training_scope: "full" [git]
        │   ├── predict.pkl          # [ignored]
        │   └── predict.pkl.manifest.json [ignored]
        └── full/current.json        # {"run_id", "promoted_at"} — active full-version pointer [git]
```

**Git-tracked vs ignored** (`.gitignore`): the **small record is versioned** —
`README.md`, `base_config.yaml`, `meta.json`, `runs/*/run.json`,
`exports/**/export.json`, `exports/**/scorecard.json`, `exports/full/current.json`,
`champion.json` (tree kept alive by `.gitkeep`). Everything heavy or
reconstructable is ignored — parquet, `predict.pkl` + sibling manifests,
checkpoints. A family's record survives registry wipes; its artifacts are
reproducible by re-running the recorded config **while the original inputs
remain available** (code identity, dependency environment, device, and the data
snapshot at scoring time).

**Rebuild-identity fields** (`run.json` → `manifest`, spec §3.1): each run
persists the exact values that entered its `run_id` —

| Field | Meaning |
|---|---|
| `data_fingerprint` | the data snapshot term (era stats, schema, row counts, `features.json` content) — the **same value** hashed into the run_id, computed once at runner construction |
| `code_fingerprint` | portable code identity over `nmr/*.py` (no absolute paths) |
| `environment` | normalized dependency versions |
| `pipeline_device` | the config device knob (`auto`/`gpu`/`cpu`) |
| `oof_device` | the actual fit device (post-fit `resolved_device`) |

**The fingerprint is a snapshot marker, not a content snapshot.** It detects
schema/row-count/era-stats/`features.json` changes; restated feature values
within an unchanged structure are NOT detected, and a growing
`validation.parquet` changes run identity by design (the data term is enforced).
**Rebuild-refusal rule:** a rebuild or checkpoint-resume compares the current
code identity and device against the recorded values — any mismatch raises
(`ValueError` with "delete the directory" guidance), never silently reusing
stale state. The persisted fingerprints make the reproducibility boundary
verifiable: you can always tell *what* a run was fitted on, and *whether* the
current tree can reproduce it.

## 5. How we operate

- **Family scaffold is created with the first run.** `record_run` writes the
  family (`meta.json` + `base_config.yaml` + `README.md`) atomically with the
  first `run.json`; no family directory exists before its first run. A
  hand-created directory without a `run.json` is the explicit `uninitialized`
  state, never an error.
- **Exports are immutable.** A slot is published by a single atomic directory
  rename from `exports/<scope>/.tmp-<run_id>/`; a half-written slot never
  appears. Promoting an already-present `exports/<scope>/<run_id>/` raises
  `ValueError` **before any write** with `force=False` — a new promotion
  means a new run, a new slot. Old slots remain for rollback; repointing
  `current.json` is a deliberate write.
- **`force=True` executes the pointer-repair recovery.** Against an existing
  VALID full slot it validates the slot (`lifecycle.valid_export`) and writes
  ONLY `current.json` — no refit, no republish, the slot is never overwritten
  (2026-08-26 review, BLOCKING 1). Against an INVALID existing slot it
  refuses; with `force=False` the existing-slot rejection stands.
- **Single-writer champion, enforced.** `experiments/champion.json` and
  `current.json` are written only from CLI/runner entry points (never
  hand-edited). The read-compare-write in `promote_if_better` is serialized by
  an inter-process advisory lock on `<root>/champion.json.lock`
  (`nmr/_filelock.py` — `msvcrt.locking`/`fcntl.flock`, 30 s timeout, clear
  error on expiry): concurrent writers serialize, so the final champion is the
  best value (2026-08-26 review, BLOCKING 3). All pointer writes are
  temp + fsync + `os.replace`.
- **Pointer-write failure is recoverable — executably.** Promotion is a
  two-write act: the full slot publishes first, then `current.json` is
  repointed. If the pointer write fails after the slot publishes, the family
  shows `degraded` (valid full export, dangling pointer); re-run promotion
  with `--force` and the writer validates the existing slot and repoints
  `current.json` at it — no refit, never delete the slot.
- **Upload and stake are manual acts.** Lifecycle validity (`valid_export`)
  does **not** imply Numerai upload acceptance — `accept_promoted_artifact`
  (raw output vs the official validator) remains the pre-upload gate, and the
  upload/stake themselves are human actions outside the codebase.
- **Clearing `experiments/*/runs/` destroys run history — ask first.** The
  versioned record is recoverable from git; the ignored artifacts are not
  recoverable from anywhere.
