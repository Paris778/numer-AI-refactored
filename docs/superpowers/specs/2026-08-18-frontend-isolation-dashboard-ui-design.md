# Front-End Isolation — `dashboard_ui/` Package — Design

**Date:** 2026-08-18
**Status:** Approved for implementation
**Branch:** `feature/frontend-isolation` (based on `main` @ `d4b14f8`)
**Audience:** Portfolio Manager / Director of Investing & autonomous LLM coding agents

---

## 1. Problem Statement & Goal

All presentation code currently lives in three top-level modules — `dashboard_charts.py`
(Plotly figure builders, 277 lines), `generate_dashboard.py` (HTML report compiler, 470
lines), `dashboard_app.py` (Streamlit app, 555 lines) — alongside the thin control-plane
scripts and the `nmr/` engine. This scatters front-end concerns across the repo root and
keeps the report's CSS and JS controller embedded as string literals inside Python.

This spec consolidates **all front-end code into one dedicated top-level package**
(`dashboard_ui/`), splits the embedded CSS and the multimetric JS controller out of the
Python string templates into real static asset files, and re-points the SSOT docs. The
rendering tech (Plotly + Streamlit) and the `nmr/dashboard.py` engine are **unchanged** —
this is a structural refactor with behavior-preserving output.

**Out of scope (deliberately):** dropping Plotly in favor of a from-scratch HTML/CSS/SVG
renderer (separate workstream — see §10), any `nmr/` engine change, Streamlit redesign.

---

## 2. Locked Decisions

| # | Decision | Choice |
|---|---|---|
| D1 | Scope | **Consolidate + split assets** — move all presentation code into one package AND extract the embedded CSS/JS into `static/` files. No rendering-tech change |
| D2 | Location | **Top-level `dashboard_ui/` package** (sibling of `nmr/`) — Plotly/Streamlit stay out of `nmr/`; the AGENTS.md invariant is re-pointed, not weakened |
| D3 | Approach | **Big-bang move with full import rewrite** — no transitional shims (the presentation layer is ~100% test-covered; behavior is pinned by existing assertions) |
| D4 | API surface | Public function names preserved; only import paths change (`dashboard_charts` → `dashboard_ui.charts`, `generate_dashboard` → `dashboard_ui.report`, `dashboard_app` → `dashboard_ui.app`) |
| D5 | Entry scripts | `generate_dashboard.py` / `dashboard_app.py` remain as **thin wrappers** (CLI + `streamlit run` surface unchanged); `dashboard_charts.py` is deleted |
| D6 | Asset inlining | `style.css` and `app.js` are **inlined at compile time** — the single-file `artifacts/dashboard.html` invariant is preserved (no external files, no CDN) |
| D7 | JS data contract | The multimetric controller reads its payload from a `<script type="application/json" id="dashboard-multimetric-data">` node (JSON.parse) instead of a JS literal |
| D8 | Test org | Tests stay in `tests/test_dashboard.py` (import paths only) + `tests/test_scripts.py` adjustments; 1–2 new tests pin the asset contract; no test-file reshuffle |
| D9 | `__init__.py` | Docstring only, **no re-exports** — all access via the three submodules (avoids the `main` name collision between `report` and `app`) |

---

## 3. Target Layout

```
dashboard_ui/                      # NEW top-level package (sibling of nmr/)
├── __init__.py                    # module docstring only (D9)
├── charts.py                      # ← dashboard_charts.py — Plotly figure builders + multimetric JS controller block
├── report.py                      # ← generate_dashboard.py — HTML compiler, KPI/table/group logic, CSS inlining
├── app.py                         # ← dashboard_app.py — Streamlit app + pure frame helpers
└── static/
    ├── style.css                  # extracted from the _build_html <style> block (single braces)
    └── app.js                     # extracted multimetric JS controller (METRIC_CONFIG + controls + payload read)
```

`nmr/dashboard.py` stays where it is — the pure engine remains the tested boundary; the
package is a **consumer** of `nmr.dashboard`, never the reverse.

---

## 4. Module Map & Responsibilities

| Module | Source | Contents (function names preserved) |
|---|---|---|
| `dashboard_ui/charts.py` | `dashboard_charts.py` | `build_leaderboard_bar_chart`, `build_similarity_matrix_chart`, `build_drawdown_chart`, `multimetric_chart_html` (now assembling div + data node + inlined `app.js`) |
| `dashboard_ui/report.py` | `generate_dashboard.py` | `generate_dashboard`, `main`, `_build_html` (now inlining `style.css`), `_kpi_cards`, `_table_rows`, `_row_html`, `_bar_input`, `_status_badge`, `_td_gate`, `_fmt`, `_bar_label`, `_diversification_stats`, `_ensemble_sharpe`, `_badge_html`, `_ensemble_card_html`, `_technical_entries` |
| `dashboard_ui/app.py` | `dashboard_app.py` | `main`, `load_registry_frame`, `load_benchmarks`, `merge_leaderboard`, `_shaped_leaderboard_pdf`, `robustness_matrix`, `champion_run_id`, `_load_registry_entries`, `_read_run_payload`, `_read_full_manifest`, all `render_*` functions, schema constants |

**Move rule:** each function moves byte-identical (except the two asset-extraction sites in
§5). Imports inside each module change to the new package-relative form
(`from dashboard_ui import charts` in `report.py`; `from nmr.dashboard import ...` unchanged).

---

## 5. Static Asset Extraction Contract

### 5.1 `static/style.css`

- Extract the `<style>` block from `_build_html` verbatim, **converting the f-string
  doubled braces `{{ }}` to single braces `{ }`** (the content is static — no dynamic
  values; verified: all rules are fixed colors/sizes).
- `report.py` inlines it at compile time: read `static/style.css` once (module-level cached
  read), emit one `<style>` tag. Rendered CSS must be byte-identical to today's output.
- Determinism: static content, no wall-clock, no paths.

### 5.2 `static/app.js`

- Extract the multimetric controller (the IIFE from `multimetric_chart_html`) verbatim,
  with ONE behavioral change (D7): replace the embedded `var payload = {payload_json};`
  with a DOM read:
  ```javascript
  var dataNode = document.getElementById("dashboard-multimetric-data");
  var payload = dataNode ? JSON.parse(dataNode.textContent) : null;
  ```
  and guard the rest of the IIFE with `if (payload) { ... }` (when the data node is absent,
  the controller does nothing — the degraded-empty case never emits the script).
- `charts.multimetric_chart_html(payload)` becomes: serialize the payload
  (`json.dumps(payload, sort_keys=True).replace("</", "<\\/")` — unchanged), then return
  the div + one data script node + the inlined `app.js` (cached read). The `</` escaping is
  kept — `JSON.parse` restores `</` correctly.
- The data node id is fixed: `dashboard-multimetric-data`. Exactly one data node per page.
- The empty-payload path (`payload.get("eras")` falsy) returns the existing
  "Timeseries data unavailable without local v5.3 assets" div and **no** script — identical
  to today.

### 5.3 Inlining mechanics

Both files are read via `(Path(__file__).parent / "static" / ...).read_text(encoding="utf-8")`
with a module-level cache (a small `@lru_cache`-style helper or a module global). `app.js` is
static — no per-call formatting — so the cache is safe. The plotly engine embed
(`plotly.offline.get_plotlyjs()`, exactly once in `<head>`) is unchanged and stays in
`report._build_html`.

---

## 6. Entry Scripts & Import Rewrites

### 6.1 Thin wrappers

`generate_dashboard.py` becomes:

```python
"""Thin entry point — all logic lives in dashboard_ui.report."""
from dashboard_ui.report import main

if __name__ == "__main__":
    raise SystemExit(main())
```

`dashboard_app.py` becomes:

```python
"""Thin entry point — all logic lives in dashboard_ui.app."""
from dashboard_ui.app import main

if __name__ == "__main__":
    main()
```

`dashboard_charts.py` is **deleted**.

### 6.2 Import rewrites (complete inventory, verified by grep)

| Site | Today | Becomes |
|---|---|---|
| `generate_dashboard.py:20` | `import dashboard_charts as charts` | `from dashboard_ui import charts` |
| `tests/test_dashboard.py:12` | `import dashboard_charts as charts` | `from dashboard_ui import charts` |
| `tests/test_dashboard.py:13` | `import generate_dashboard` | `from dashboard_ui import report as generate_dashboard` |
| `tests/test_dashboard.py:1225,1243,1258` | `import dashboard_app as app` | `from dashboard_ui import app` |
| `tests/test_scripts.py:8` | `import generate_dashboard` | unchanged — imports the thin wrapper (entry surface) |
| `tests/test_scripts.py:27,198` | `import dashboard_app` | unchanged — imports the thin wrapper |

Historical docs (`docs/superpowers/plans/2026-08-16-*.md`, `2026-08-17-*.md`) are left
untouched — they describe what was built when.

---

## 7. Testing Plan

- **`tests/test_dashboard.py`**: only the import lines change; all existing assertions
  (chart builders, `_build_html` structure, `window.Plotly` count == 1, no CDN, KPI/table/
  ensemble/diversification logic, Streamlit frame helpers) run unchanged against the moved
  code. This is the behavior-preservation proof.
- **New tests (2, appended to `tests/test_dashboard.py`):**
  1. `test_report_inlines_style_css_once` — `generate_dashboard._build_html(...)` output
     contains the style.css content exactly once and inside a `<style>` tag; assert a known
     marker rule (e.g. `.badge.full`) appears exactly once.
  2. `test_multimetric_chart_embeds_data_node_and_app_js` — `charts.multimetric_chart_html`
     output contains exactly one `id="dashboard-multimetric-data"` node, the serialized
     payload parses back via `json.loads` with `</` unescaped, and the app.js controller
     content (a marker such as `METRIC_CONFIG`) is inlined exactly once.
- **`tests/test_scripts.py`**: **no changes** — it imports the top-level wrappers
  (`import generate_dashboard`, `import dashboard_app`), which preserve the `main` entry
  surface; a quick run confirms they still pass against the wrappers.
- Test-count bump: 816 → 818 (+2 new tests) — AGENTS.md + CONTRIBUTING.md claims updated
  in the same commit as the test additions (docs-hygiene T3).

---

## 8. SSOT Docs Updates (same commit as code)

| File | Change |
|---|---|
| `AGENTS.md` | Line 82 invariant re-pointed: "Streamlit + Plotly … imported only in the `dashboard_ui/` package (`charts.py`, `report.py`, `app.py`); never in `nmr/`". Toolkit rows referencing the three old modules → new paths (`dashboard_ui/report.py`, `dashboard_ui/app.py`). Test-count bump to 818 |
| `ARCHITECTURE.md` | Module map / dependency graph: add `dashboard_ui/` (leaf consumer of `nmr.dashboard`; `report.py → charts.py`); update the dashboard section's file references |
| `README.md` | Annotated tree: add `dashboard_ui/`; update the entry-script descriptions (`generate_dashboard.py` / `dashboard_app.py` now thin wrappers) |
| `CONTRIBUTING.md` | Test-count claim only (if present) |

Historical specs (v2 `2026-08-16`, marker `2026-08-17`) stay as-is.

---

## 9. Verification Gates

```bash
# 1. Lint
./.venv/Scripts/python -m ruff check .

# 2. Functional gate (full suite; count claim 818 must match --collect-only)
./.venv/Scripts/python -m pytest -q

# 3. Targeted while iterating
./.venv/Scripts/python -m pytest -q tests/test_dashboard.py tests/test_scripts.py

# 4. Real-data compile + artifact validation
./.venv/Scripts/python generate_dashboard.py
./.venv/Scripts/python -c "
from pathlib import Path
text = Path('artifacts/dashboard.html').read_text(encoding='utf-8')
assert text.count('<!-- plotly-engine-embed -->') == 1   # engine embedded exactly once
assert '<script src' not in text                          # no external script tags (no CDN)
assert text.count('id=\"dashboard-multimetric-data\"') == 1   # app.js data node (with local data)
assert '.badge.full {' in text          # style.css inlined
assert 'var METRIC_CONFIG = {' in text  # app.js inlined
print('front-end isolation HTML verified')
"

# 5. No stray imports of the old module names in live code (aliases are fine)
grep -rn "import dashboard_charts\|from dashboard_charts\|import generate_dashboard\|from generate_dashboard\|import dashboard_app\|from dashboard_app" nmr/ dashboard_ui/ tests/ generate_dashboard.py dashboard_app.py
```

**End state:** all presentation code lives under `dashboard_ui/`; the two top-level scripts
are thin wrappers; `dashboard_charts.py` is gone; `nmr/` untouched by plotly/streamlit;
`artifacts/dashboard.html` byte-compatible rendering (style.css + app.js inlined, plotly
embedded once, no CDN).

---

## 10. Out of Scope / Follow-Ups

- **Drop Plotly → from-scratch HTML/CSS/SVG renderer** (report target < 100 KB): separate
  workstream; this refactor is the prerequisite that makes it tractable (assets already
  split; `dashboard_ui/` is the home).
- Any change to `nmr/dashboard.py` engine behavior or schemas.
- Streamlit visual redesign.
- Deleting the historical v1/v2 specs and plans (they remain as records).
