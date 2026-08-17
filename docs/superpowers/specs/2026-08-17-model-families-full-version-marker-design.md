# Model Family Full-Version Marker — Design

**Date:** 2026-08-17
**Status:** Approved for implementation
**Branch:** `feature/model-full-version-marker` (based on `feature/executive-dashboard-v2` @ `904c61a`)
**Audience:** Portfolio Manager / Director of Investing & autonomous LLM coding agents

---

## 1. Problem Statement & Goal

Every model on the executive dashboard exists in a **research version** (trained on
`train.parquet` only, evaluated via era-purged walk-forward CV and recorded in
`artifacts/registry/{run_id}/`). A subset of families may additionally be **promoted to a
full version** (trained on `train.parquet + validation.parquet`, deployed for live
prediction). Not all families will earn that privilege.

Today there is **no way to tell, at a glance, whether a model family has a full version**:
the unified leaderboard knows nothing about promotion state. This spec adds:

1. A **discovery mechanism** — a family-directory convention under `artifacts/models/`
   whose presence marks a family as promoted (the user's "one directory per model").
2. A **dedicated visual marker** on the main dashboard table — a `FULL` chip on research
   rows whose family has a full version, plus the full version itself as a first-class
   table row.
3. **Documentation** of the convention and the new module in the golden SSOT files
   (`AGENTS.md`, `ARCHITECTURE.md`, `README.md`).

**Out of scope (deliberately):** the full-version **training pipeline** (fitting a model on
train+validation) and the promotion writer. This spec defines the storage contract and the
read/discovery side only; the writer is a future workstream that imports the same module.

---

## 2. Locked Decisions

| # | Decision | Choice |
|---|---|---|
| D1 | Discovery mechanism | **Family directory convention** — `artifacts/models/<family>/full/manifest.json`; manifest existence IS the marker |
| D2 | Scope | **Marker + discovery + docs only** — no full-training pipeline in this spec |
| D3 | Display | **Badge on research rows + full version as its own row** (marked in-sample, metrics "—") |
| D4 | Family key | **Directory name = `run.name`** — all registry runs sharing `manifest.config.run.name` are family members; duplicate reruns share the family. No config-schema change, so `run_id` hashing is untouched |
| D5 | Module placement | **New read-only `nmr/families.py`** — dedicated tested boundary; future promotion writer imports the same contract |
| D6 | Table grouping | Active Champion → Promoted Full Versions → Research Fleet → Benchmark Floor |
| D7 | Marker semantics | **Manifest existence**, not bare dir existence — a `full/` dir without a valid manifest is ignored (half-written promotion) |
| D8 | Family charset | **Lowercase-only** `^[a-z0-9_-]+$` — prevents case-collision overwrites on case-insensitive filesystems (Windows NTFS / macOS APFS); matches the existing `run.name` convention |
| D9 | Chart evaluability | **Single module-level predicate** (`EVALUABLE_ROWS = source != "full"`) owns chart inclusion — no ad-hoc per-chart filters |
| D10 | Promotion completeness | Valid manifest additionally requires a **non-empty relative `artifact_path` whose file exists** — no ghost/hollow promotions |
| D11 | Champion vs full | Champion pointer always names a registry `run_id`; a full row (`<family>::full`) can never be champion — the two table partitions are additive, not exclusive |

---

## 3. Storage Convention

New top-level artifacts directory, sibling of `registry/`, `runs/`, `reports/`:

```
artifacts/
└── models/
    └── <family>/                # family name == run.name of the research runs
        └── full/
            └── manifest.json    # THE MARKER: presence + validity == "has full version"
```

### 3.1 `manifest.json` schema (writer contract — writer is future work)

```json
{
  "family": "brb1-xgb-v6",
  "training_scope": "full",
  "promoted_from_run_id": "2610a99d833779ea78d14aa0b78d16a0a1d17706fc3aa6b9536ca7e9093445a5",
  "promoted_at": "2026-08-17T12:00:00Z",
  "artifact_path": "artifacts/models/brb1-xgb-v6/full/predict.pkl",
  "config": {
    "run": { "name": "brb1-xgb-v6", "seed": 20260810, "artifacts_dir": "artifacts" },
    "data": { "version": "v5.3", "feature_set": "all", "feature_subset": "medium" },
    "model": { "backend": "xgboost", "preset": "fast", "device": "auto" }
  }
}
```

### 3.2 Validation rules (read side)

A manifest is **valid** only when ALL hold; otherwise the family is treated as **not
promoted** and a warning is logged (skip-gracefully, matching the corrupt-`run.json`
precedent in `load_unified_leaderboard`):

1. `manifest.json` exists under `artifacts/models/<family>/full/`.
2. JSON parses as an object.
3. `family` equals the directory name exactly (dir names are lowercase, §4.3).
4. `training_scope == "full"`.
5. `promoted_from_run_id` is a non-empty string.
6. `artifact_path` is a non-empty **relative** path — no leading `/`, no drive
   letter, no `..` — resolved against the repo root, **and the referenced file
   exists on disk**. A manifest written before model serialization completes
   ("hollow" promotion) is treated as not promoted and warned, exactly like a
   corrupt `run.json`.

**Lineage warning (not invalidation):** if `promoted_from_run_id` has no directory under
`artifacts/registry/` (source run pruned), the consumer logs a warning but still renders
the full row — the deployed model exists; provenance is informational.

### 3.3 Determinism & integrity notes

- `promoted_at` is wall-clock but the manifest is **display metadata only** — it never
  enters `run_id`, `canonical_scorecards_bytes()`, neutralization cache keys, or any other
  canonical hash. Canonical-hash invariants are untouched.
- `artifacts/models/` is **machine-generated, not source of truth**. The read side is
  strictly read-only; the registry stays immutable.
- **Writer contract (future):** the promotion writer must serialize the model artifact
  BEFORE writing `manifest.json` (temp file + fsync + `os.replace`, matching the repo's
  atomic-write discipline), so a valid manifest always implies an existing artifact.

---

## 4. `nmr/families.py` — Read-Only Family Layer (tested boundary)

New module in `nmr/`; all logic covered by `tests/test_families.py`. Exported via
`nmr/__init__.py` (`import` + `__all__` updated in the same commit — package-API test
guard).

### 4.1 Constants

```python
FAMILY_DIR_NAME = "models"           # under artifacts root
FULL_DIR_NAME = "full"
FULL_MANIFEST_NAME = "manifest.json"
DEFAULT_MODELS_DIR = REPO_ROOT / "artifacts" / "models"   # mirrors REPORTS_DIR style
```

### 4.2 Public API

```python
@dataclass(frozen=True)
class FullVersion:
    family: str
    manifest_path: Path
    artifact_path: str | None
    promoted_from_run_id: str | None
    promoted_at: str | None
    config: dict[str, Any]          # raw snapshot

def full_manifest_path(models_dir: Path, family: str) -> Path
    # Resolve artifacts/models/<family>/full/manifest.json.

def load_full_version(models_dir: Path, family: str) -> FullVersion | None
    # None when: manifest missing, JSON invalid, or validation rules (§3.2) fail.

def scan_full_versions(models_dir: Path) -> dict[str, FullVersion]
    # family -> FullVersion for every valid manifest under models_dir.
    # Missing models_dir -> {}.

def family_has_full_version(models_dir: Path, family: str) -> bool
    # Convenience: load_full_version(...) is not None.
```

### 4.3 Path-traversal / naming guard (approval item 1)

Family names are validated against a strict **lowercase** charset before ANY path
construction: `^[a-z0-9_-]+$`. Lowercase-only prevents case-collision overwrites on
case-insensitive filesystems (Windows NTFS, macOS APFS) where `Model_A` and `model_a`
would silently collide and clobber each other; it also matches the existing `run.name`
convention (all 29 registry runs are lowercase). Enforcement points:

- `full_manifest_path()` and `load_full_version()` **raise** `ValueError` on an invalid
  family name (fail-fast — callers must not feed unvalidated names).
- `scan_full_versions()` **skips** directory entries that do not match the regex
  (defensive — the filesystem may contain arbitrary entries, including mixed-case dirs).
- `manifest.family` must equal the (validated) directory name (§3.2 rule 3).

This prevents path traversal (`..`, `/`, `\`), special filesystem characters, whitespace,
and mixed-case collisions from reaching filesystem calls.

---

## 5. Unified Schema & Leaderboard Integration

### 5.1 `UNIFIED_SCHEMA` additions (`nmr/dashboard.py`)

| Column | Type | Semantics |
|---|---|---|
| `family` | `pl.String` | Trained rows: `run.name`. Full rows: manifest `family`. Benchmarks: `null` |
| `training_scope` | `pl.String` | `"research"` (trained rows) / `"full"` (full rows) / `null` (benchmarks) |
| `has_full_version` | `pl.Boolean` | Trained rows: does the family have a valid full manifest? Full rows: `false`. Benchmarks: `false` |

### 5.2 `load_unified_leaderboard` changes

Signature gains `models_dir: Path | None = None` (default → `DEFAULT_MODELS_DIR`).

- **Scan once (review item 1):** `full_versions = scan_full_versions(models_dir)` runs a
  single top-of-function pass; `promoted_families = set(full_versions)` is built from it.
  Every trained row resolves `has_full_version` via **O(1) set membership** — never a
  per-row filesystem check. N runs across M families ⇒ 1 disk scan, not N.
- **Trained rows**: `family = run.name`, `training_scope = "research"`,
  `has_full_version = run.name in promoted_families`.
- **Full rows**: one appended per valid `FullVersion` from the single scan:
  - `model_id = f"{family}::full"`, `source = "full"`, `run_name = family`,
    `run_dir = str(manifest_path.parent)`.
  - **All metric cells null** — validation metrics for a train+validation model are
    in-sample and must never be shown as comparable OOF numbers (honest "—" rendering via
    existing `_fmt(None)`).
  - `family`/`training_scope` set; `has_full_version = false`.
  - **Lineage warning (review item 2b):** if `(registry_dir / promoted_from_run_id)` does
    not exist, log a warning — provenance is dangling after registry pruning. The row
    still renders; the warning is informational.
- **Status assignment**: `evaluate_gate_status` gains an explicit early branch —
  `source == "full"` rows are stamped `status = "FULL"` with all `gate_*` receipts `None`.
  This is mandatory: without it, all-null metric cells fall through the gate ladder and
  full rows would be mislabeled `RESEARCH`. Status stays owned by the gate function.
- **Benchmark rows**: `family`/`training_scope` null, `has_full_version` false.
- Missing `models_dir` → empty scan → all flags false, zero full rows. **No behavioral
  change for existing callers/tests.**

`_LEADERBOARD_SCHEMA` (`dashboard_app.py`) mirrors the three new columns.

### 5.3 Chart-inclusion predicate (review item 4)

```python
EVALUABLE_ROWS: pl.Expr = pl.col("source") != "full"
```

Module-level in `nmr/dashboard.py` — the **single** predicate every chart and candidate
selection path uses. Source-based, not `training_scope`-based, on purpose: benchmark rows
have null `training_scope` and must remain visible in the Sharpe leaderboard, while
`source` is never null. Full rows carry no validation metrics and are excluded everywhere.

---

## 6. Display Layer

### 6.1 Static HTML report (`generate_dashboard.py`)

- **Research rows with a full version**: a `<span class="badge full">FULL</span>` chip in
  the model cell (`_row_html`), next to the label. New CSS class `.badge.full` with a color
  distinct from status badges.
- **Full rows**: a dedicated table group **"Promoted Full Versions"** rendered **between
  the Active Champion group and the Research Fleet** (approval item 3 — D6). Status badge
  `FULL` (new `_STATUS_BADGE["FULL"] = "full"` entry). Metric cells render "—" via
  `_fmt(None)`; gate cells pass through `_td_gate(..., None)` without red tinting.
- **Champion vs full coexistence (review item 5b — D11)**: the champion pointer always
  names a registry `run_id` (`champion.json` → `{"run_id": ...}`), and a full row's
  `model_id = "<family>::full"` can never equal it — a full row can never itself be
  champion. If the champion *family* has a full version, the champion research row stays in
  the Active Champion group (carrying the FULL chip) while its full row renders in the
  Promoted group. The two partitions are additive, not exclusive.
- **`_table_rows`**: group order = champion rows + full rows + fleet rows + benchmark rows.
- **`_bar_input`** (Sharpe leaderboard chart): filter `EVALUABLE_ROWS` (§5.3) before
  ranking — full rows never reach the chart.

### 6.2 Streamlit app (`dashboard_app.py`)

- `_LEADERBOARD_SCHEMA` gains `family`, `training_scope`, `has_full_version`.
- `load_registry_frame()` broadens the trained-only filter to
  `pl.col("source").is_in(["trained", "trained_legacy", "full"])` (approval item 2) so the
  existing Source multiselect surfaces `full` and full rows render.
- **Deterministic default sort (review item 5a)**: `load_registry_frame()` pins full rows
  first (`source == "full"` ascending-key first), then `corr_sharpe_ac` descending with
  nulls last — no client-side Arrow/AgGrid null-placement ambiguity on column sorts.
- `has_full_version` rendered via `st.column_config.CheckboxColumn(label="Full", help=...)` —
  a checkbox column is the unambiguous boolean presentation; no extra text column.
- No other view-contract changes.

### 6.3 Charts

All chart and candidate-selection paths use the single `EVALUABLE_ROWS` predicate (§5.3) —
Sharpe leaderboard (`_bar_input`), similarity-matrix candidate selection, and
multimetric/drawdown candidates. No ad-hoc `source != "full"` checks scattered across call
sites. (Full rows also have no registry `validation_preds.parquet`, so the predicate
prevents missing-run warnings in the similarity matrix.)

---

## 7. Documentation Updates (SSOT — same commit as code)

| File | Change |
|---|---|
| `ARCHITECTURE.md` | New §: `nmr/families.py` module contract (API, validation rules, path guard); `artifacts/models/<family>/full/manifest.json` artifact schema; UNIFIED_SCHEMA column additions; updated artifact-layout diagram |
| `AGENTS.md` | Toolkit table row (model families / full-version discovery → `nmr/families.py`); artifact-layout mention; test-count bump (772 → new count, from `pytest --collect-only` in the same commit) |
| `README.md` | One annotated-tree line for `artifacts/models/` |
| `docs/superpowers/specs/2026-08-17-model-families-full-version-marker-design.md` | This spec |

No `CONTRIBUTING.md` change (no new commands). No config-schema change (D4).

---

## 8. Testing Plan

### 8.1 `tests/test_families.py` (new)

- `full_manifest_path()` resolution under a `tmp_path` models dir.
- **Path-guard**: `full_manifest_path`/`load_full_version` raise `ValueError` for invalid
  family names (`../evil`, `a/b`, `a b`, `a:b`, `""`); `scan_full_versions` skips
  non-matching directory entries.
- **Lowercase guard (review item 3)**: mixed-case family names (`ModelA`) are rejected by
  the regex — `load_full_version("ModelA")` raises, `scan_full_versions` skips a mixed-case
  dir, and a manifest whose `family` is mixed-case inside a lowercase dir is invalid.
- `load_full_version` happy path; missing manifest → `None`; corrupt JSON → `None`;
  `family != dir name` → `None`; `training_scope != "full"` → `None`; missing
  `promoted_from_run_id` → `None`.
- **Artifact-existence (review item 2a)**: empty / absolute / `..`-containing `artifact_path`
  → `None`; a non-empty relative `artifact_path` whose file does not exist → `None`
  (hollow promotion rejected); existing file → valid.
- `scan_full_versions` across multiple families (valid + invalid interleaved) → only valid
  ones, keyed by family.
- `family_has_full_version` true/false.

### 8.2 `tests/test_dashboard.py` additions

- `load_unified_leaderboard(..., models_dir=<tmp fixture>)`:
  - Trained rows carry `family` (= `run.name`), `training_scope="research"`, and
    `has_full_version` reflecting the fixture's manifests.
  - Full rows appended with `source="full"`, `model_id == "<family>::full"` (unique),
    `run_name == family`, null metric cells, `training_scope="full"`.
  - Benchmark rows (when `benchmark_path` provided) untouched by the family logic.
  - `benchmark_path=False` isolation intact; missing `models_dir` → zero full rows, all
    flags false.
- **Scan-once (review item 1)**: monkeypatch `scan_full_versions` with a call counter —
  N trained runs resolve `has_full_version` with exactly one scan invocation.
- **Lineage (review item 2b)**: a full manifest whose `promoted_from_run_id` has no
  registry dir still renders, with a warning logged (`caplog`).
- `evaluate_gate_status` with a full row present → that row's status is `"FULL"`, receipts
  all `None`, and no full row is ever mislabeled `RESEARCH`.
- **`EVALUABLE_ROWS` (review item 4)**: excludes full rows, keeps trained and benchmark
  rows.
- Existing determinism tests (`tests/test_benchmark_hierarchy.py`, `tests/test_scorecard.py`)
  are the guard that no canonical-hash payload changed — run unchanged.

---

## 9. Verification Gates

```bash
# 1. Lint gate
./.venv/Scripts/python -m ruff check .

# 2. Functional gate (full suite; test count claimed in AGENTS.md must match)
./.venv/Scripts/python -m pytest -q

# 3. Targeted subsets while iterating
./.venv/Scripts/python -m pytest -q tests/test_families.py tests/test_dashboard.py

# 4. Real-data acceptance: regenerate the report with a populated models dir
./.venv/Scripts/python generate_dashboard.py
#   -> artifacts/dashboard.html shows FULL chips and the Promoted Full Versions group
```

**Determinism gate:** `canonical_scorecards_bytes()` / `run_id` hashing must not change —
families.py is read-only display metadata; no hashed payload is touched.

---

## 10. Out of Scope / Follow-Ups

- **Full-version training pipeline** (fit on train+validation) — future workstream; its
  writer will produce `artifacts/models/<family>/full/manifest.json` per this contract and
  will run under the same "long jobs" operational guidance.
- **Promotion authorization / champion pointer interaction** — promotion remains a
  deliberate human/agent act; no auto-promotion rules in this spec.
- **Per-family "best research row" selection** — all family rows carry the badge; choosing
  a single representative research row is out of scope.
