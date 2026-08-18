# Front-End Isolation (`dashboard_ui/`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate all presentation code into one dedicated top-level `dashboard_ui/` package (`charts.py`, `report.py`, `app.py` + `static/{style.css, app.js}`), turn the two entry scripts into thin wrappers, delete `dashboard_charts.py`, and re-point the SSOT docs — with byte-identical rendered output.

**Architecture:** Pure structural move with behavior preserved. `dashboard_ui/charts.py` holds the Plotly figure builders plus a rewritten `multimetric_chart_html` that reads its payload from a `<script type="application/json" id="dashboard-multimetric-data">` node (data contract) and inlines a static `static/app.js` controller. `dashboard_ui/report.py` is the HTML compiler, inlining `static/style.css`. `dashboard_ui/app.py` is the Streamlit app. `nmr/dashboard.py` is untouched and stays the tested engine; the package consumes it. Spec: `docs/superpowers/specs/2026-08-18-frontend-isolation-dashboard-ui-design.md`.

**Tech Stack:** Python 3.11+, Plotly, Streamlit, Polars, pytest, ruff (E/F/I/UP @120). No new dependencies.

## Global Constraints

- **Tested boundary:** `nmr/dashboard.py` unchanged — no engine edits in this plan. The `dashboard_ui/` package is a *consumer* of `nmr.dashboard`.
- **`nmr/` UI-free:** Plotly/Streamlit must never be imported in `nmr/`; they live only in `dashboard_ui/` (and transitively via the thin wrappers).
- **Single-file HTML invariant:** `artifacts/dashboard.html` stays a standalone file — `style.css` and `app.js` are inlined at compile time, the plotly engine is embedded exactly once (`window.Plotly` count == 1), no CDN (`cdn.plot.ly` absent).
- **Determinism:** no wall-clock, no absolute paths in the compiled HTML; asset reads are static and cached.
- **Public names preserved:** every function keeps its name; only import paths change. `dashboard_charts` → `dashboard_ui.charts`, `generate_dashboard` → `dashboard_ui.report`, `dashboard_app` → `dashboard_ui.app`.
- **Entry scripts stay** as thin wrappers; `dashboard_charts.py` is deleted.
- **Lint:** `./.venv/Scripts/python -m ruff check .` must pass (E/F/I/UP, line-length 120).
- **AGENTS.md test-count claim:** `tests/test_docs_hygiene.py::test_docs_test_count_matches_suite` runs `pytest --collect-only` on every full-suite run and fails if the "N tests" claims in AGENTS.md (two places) / CONTRIBUTING.md don't equal the collected count. Current count: **816**. Any task that adds a test node MUST bump all three claims in the same commit (READ the actual count via collect-only — do not guess).
- **Windows venv:** always `./.venv/Scripts/python -m ...`.
- **Git:** commit per task; do NOT push. The "LF will be replaced by CRLF" warning is benign.
- **Verification per task:** targeted tests → ruff → count bump (if tests added) → FULL suite (foreground with the Bash tool `timeout=300`; the suite takes ~3–5 min and if auto-backgrounded you MUST wait for it via TaskOutput `block=true` within your turn — background tasks die when your session ends) → commit.
- **Historical docs** (`docs/superpowers/plans/2026-08-16-*.md`, `2026-08-17-*.md`, the v1/v2 specs) are NOT touched.

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `dashboard_ui/__init__.py` | Module docstring only, no re-exports | Create (Task 1) |
| `dashboard_ui/charts.py` | Plotly figure builders + multimetric controller block (data node + inlined app.js) | Create from `dashboard_charts.py` (Task 1) |
| `dashboard_ui/static/app.js` | Extracted multimetric JS controller (reads `dashboard-multimetric-data`) | Create (Task 1) |
| `dashboard_ui/report.py` | HTML report compiler (inlines style.css) | Create from `generate_dashboard.py` (Task 2) |
| `dashboard_ui/static/style.css` | Extracted `<style>` block (single braces) | Create (Task 2) |
| `dashboard_ui/app.py` | Streamlit app + pure frame helpers | Create from `dashboard_app.py` (Task 3) |
| `generate_dashboard.py` | Thin wrapper re-exporting `generate_dashboard` + `main` | Replace (Task 2) |
| `dashboard_app.py` | Thin wrapper re-exporting `main` | Replace (Task 3) |
| `dashboard_charts.py` | — | Delete (Task 1) |
| `tests/test_dashboard.py` | Import rewrites + 1 updated test + 2 new tests | Modify (Tasks 1–3) |
| `tests/test_scripts.py` | `dashboard_app` imports → `dashboard_ui.app` | Modify (Task 3) |
| `AGENTS.md`, `ARCHITECTURE.md`, `README.md`, `CONTRIBUTING.md` | SSOT sync | Modify (Tasks 1–4 counts; Task 4 content) |

---

### Task 1: `dashboard_ui` scaffold + `charts.py` + `static/app.js`

**Files:**
- Create: `dashboard_ui/__init__.py`, `dashboard_ui/charts.py`, `dashboard_ui/static/app.js`
- Delete: `dashboard_charts.py`
- Modify: `generate_dashboard.py` (import line 20), `tests/test_dashboard.py` (import line 12; update the multimetric JS test ~line 920; add 1 new test)

**Interfaces:**
- Consumes: nothing new (Task 1 is the first move).
- Produces: `dashboard_ui.charts` with the SAME public names as `dashboard_charts` (`build_leaderboard_bar_chart(df, *, hurdle_sharpe)`, `build_similarity_matrix_chart(labels, matrix)`, `build_drawdown_chart(payload)`, `multimetric_chart_html(payload) -> str`); module global `_APP_JS` (cached inlined app.js). Task 2 imports `from dashboard_ui import charts`.

- [ ] **Step 1: Create the package skeleton**

Create `dashboard_ui/__init__.py`:

```python
"""Presentation layer for the executive dashboard.

All front-end code (Plotly charts, the HTML report compiler, the Streamlit
app, and static assets) lives here. Pure engine logic stays in ``nmr/``;
this package only consumes ``nmr.dashboard``.
"""
```

Create `dashboard_ui/static/app.js` — **extract the IIFE from the current `dashboard_charts.py::multimetric_chart_html` (lines 171–277) verbatim** with EXACTLY these two changes:

(a) Replace the payload literal line

```javascript
  var payload = {payload_json};
```

with a DOM read + guard (the whole body after this line must be wrapped so it only runs when the node exists):

```javascript
  var dataNode = document.getElementById("dashboard-multimetric-data");
  var payload = dataNode ? JSON.parse(dataNode.textContent) : null;
```

(b) Wrap the remainder of the IIFE body (everything from `var METRIC_CONFIG = {` through the final `applyState();`) inside:

```javascript
  if (payload) {
    ... (METRIC_CONFIG, currentMetric/currentView, mounted, esc, stressShapes,
        applyState, controls wiring, applyState() call) ...
  }
```

The `METRIC_CONFIG`, controls, `stressShapes()`, `applyState()`, and the load/switch event wiring are otherwise byte-identical to the current inline JS. The script must contain exactly one `var dataNode` and exactly one `var payload` declaration.

- [ ] **Step 2: Create `dashboard_ui/charts.py`**

Copy `dashboard_charts.py` (all 277 lines) verbatim, then apply EXACTLY these edits:

(a) Module docstring first line → `"""Plotly figure builders and the multimetric JS-controller block for the executive dashboard."""`

(b) Add to the imports:

```python
from pathlib import Path
```

(c) After the imports, add the asset reader (cached, deterministic):

```python
_STATIC_DIR = Path(__file__).parent / "static"


def _read_asset(name: str) -> str:
    """Read a static asset once (cached at import). Content is static."""
    return (_STATIC_DIR / name).read_text(encoding="utf-8")


_APP_JS = _read_asset("app.js")
```

(d) Replace the ENTIRE `multimetric_chart_html` function body (the current lines 157–277) with:

```python
def multimetric_chart_html(payload: dict[str, Any]) -> str:
    """Interactive 7-metric trajectory chart: data-node payload + static JS controller.

    No plotly ``updatemenus`` (state collision); two JS state variables drive
    ``Plotly.react``. The payload is serialized into a JSON script node that
    ``static/app.js`` reads via ``JSON.parse``; ``</`` is escaped so a hostile
    label can never close the script tag. Deterministic: fixed template +
    sorted-key JSON + static asset.
    """
    if not payload.get("eras"):
        return (
            '<div id="multimetric-chart" class="chart-box">'
            "<p>Timeseries data unavailable without local v5.3 assets</p></div>"
        )
    payload_json = json.dumps(payload, sort_keys=True).replace("</", "<\\/")
    return (
        '<div id="multimetric-chart" class="chart-box"></div>\n'
        '<script type="application/json" id="dashboard-multimetric-data">'
        f"{payload_json}</script>\n"
        f"<script>{_APP_JS}</script>"
    )
```

(e) Leave `build_leaderboard_bar_chart`, `build_similarity_matrix_chart`, and `build_drawdown_chart` byte-identical.

- [ ] **Step 3: Update the import sites**

`generate_dashboard.py` line 20: `import dashboard_charts as charts` → `from dashboard_ui import charts`

`tests/test_dashboard.py` line 12: `import dashboard_charts as charts` → `from dashboard_ui import charts`

Delete `dashboard_charts.py`.

- [ ] **Step 4: Update the multimetric JS test to the new data contract**

In `tests/test_dashboard.py`, update `test_multimetric_chart_html_embeds_payload_and_controls` (currently ~line 920). Replace the payload-round-trip assertion (currently `embedded = json.loads(block.split("var payload = ")[1].split(";")[0])`) with a data-node read:

```python
    block = charts.multimetric_chart_html(payload)
    assert 'id="multimetric-chart"' in block
    data_start = block.index('id="dashboard-multimetric-data"') + len('id="dashboard-multimetric-data">')
    data_end = block.index("</script>", data_start)
    embedded = json.loads(block[data_start:data_end])
    assert embedded == payload  # exact sorted-key serialization round-trips
    assert "var payload = {" not in block  # payload no longer a JS object literal
    assert block.count("<option") == 7
    assert "Cumulative View" in block and "Standard View" in block
    assert "METRIC_CONFIG" in block  # app.js inlined
    assert "Cumulative Wealth (1.0 Stake)" in block and "Per-Era Net Return" in block
    assert "hoverformat" in block
    assert 'hovertemplate: "%{y:"' in block
    assert "updatemenus" not in block
    assert "<script src" not in block
```

`test_multimetric_chart_html_empty_payload_annotation` stays unchanged (empty payload → div only, no data node, no script).

- [ ] **Step 5: Add the new asset-contract test**

Append to `tests/test_dashboard.py`:

```python
def test_multimetric_chart_embeds_data_node_and_app_js_once() -> None:
    payload = {
        "eras": ["0001", "0002"],
        "meta_downside_mask": [False, False],
        "metrics": {"payout": {"a": {"standard": [0.01, 0.02],
                                     "cumulative": [1.01, 1.0302], "label": "r"}}},
        "drawdowns": {"a": [0.0, 0.0]},
    }
    block = charts.multimetric_chart_html(payload)
    assert block.count('id="dashboard-multimetric-data"') == 1
    assert block.count("<script") == 2  # data node + app.js
    assert block.count("var METRIC_CONFIG = {") == 1  # app.js inlined exactly once
    # a marker from the controller body is present (dataNode read)
    assert 'getElementById("dashboard-multimetric-data")' in block
```

- [ ] **Step 6: Run the targeted tests**

Run: `./.venv/Scripts/python -m pytest -q tests/test_dashboard.py -k "multimetric or chart"`
Run: `./.venv/Scripts/python -m pytest -q tests/test_scripts.py`   (import surface — the generate_dashboard import changed)
Expected: PASS.

- [ ] **Step 7: Lint**

Run: `./.venv/Scripts/python -m ruff check dashboard_ui/ generate_dashboard.py tests/test_dashboard.py`
Expected: clean (watch isort ordering of the new `from dashboard_ui import charts` line).

- [ ] **Step 8: Bump the test-count claim (816 → 817)**

Run: `./.venv/Scripts/python -m pytest --collect-only -q 2>&1 | tail -2`
Read "N tests collected" (expect 817). Update EVERY "N tests" claim in AGENTS.md (two places) and CONTRIBUTING.md to N. Verify README.md has none.

- [ ] **Step 9: Run the full suite**

Run: `./.venv/Scripts/python -m pytest -q` (foreground, `timeout=300`; wait via TaskOutput `block=true` if auto-backgrounded). Expected: PASS, count matches the claims.

- [ ] **Step 10: Commit**

```bash
git add dashboard_ui/ dashboard_charts.py generate_dashboard.py tests/test_dashboard.py AGENTS.md CONTRIBUTING.md
git commit -m "refactor(dashboard): move charts into dashboard_ui package with static app.js data contract"
```

---

### Task 2: `dashboard_ui/report.py` + `static/style.css` + thin wrapper

**Files:**
- Create: `dashboard_ui/report.py`, `dashboard_ui/static/style.css`
- Replace: `generate_dashboard.py` (thin wrapper)
- Modify: `tests/test_dashboard.py` (import line 13; add 1 new test)

**Interfaces:**
- Consumes: `dashboard_ui.charts` (Task 1) and the unchanged `nmr.dashboard` imports.
- Produces: `dashboard_ui.report` with the SAME public names as `generate_dashboard` (`generate_dashboard(*, registry_dir=None, benchmark_path=None, output_path=None, open_browser=True) -> Path`, `main() -> int`, plus all `_`-prefixed helpers); module global `_STYLE_CSS`. Task 3 + the wrapper consume `main`.

- [ ] **Step 1: Create `dashboard_ui/static/style.css`**

Extract the `<style>` block from `generate_dashboard.py::_build_html` (currently the lines from `body {{ background: #0d1117; ...` through `h1, h2 {{ color: #e6edf3; }}`) verbatim, converting every `{{` to `{` and every `}}` to `}` (the f-string doubling is removed; the content is static). The file must contain the full set of rules including `.badge.full`, `.group-header td`, `.badge.champion`, `.badge.ready`, `.badge.research`, `.badge.hurdle`, `.badge.benchmark`, `.kpis`, `.kpi`, `.gate-fail`, `table`, `th, td`, `.num`, `details`, `summary`, `pre`, `h1, h2`.

- [ ] **Step 2: Create `dashboard_ui/report.py`**

Copy `generate_dashboard.py` (all 470 lines) verbatim, then apply EXACTLY these edits:

(a) Module docstring → update the reference to the figures module:

```python
"""Compile the executive HTML performance report from the shared engine.

Thin control plane only: data comes from ``nmr.dashboard``, figures from
``dashboard_ui.charts``, CSS from ``dashboard_ui.static``. No metric math here.
"""
```

(b) Confirm the charts import reads `from dashboard_ui import charts` (already changed in Task 1 — do not revert it)

(c) After `logger = logging.getLogger(__name__)`, add (the file already imports `Path` — do not add a second import):

```python
_STATIC_DIR = Path(__file__).parent / "static"


def _read_asset(name: str) -> str:
    """Read a static asset once (cached at import). Content is static."""
    return (_STATIC_DIR / name).read_text(encoding="utf-8")


_STYLE_CSS = _read_asset("style.css")
```

(d) In `_build_html`, replace the ENTIRE in-line `<style>` block content (the `body {{ ... }}` … `h1, h2 {{ ... }}` rules with doubled braces) with a single interpolation of the global:

```
  {_STYLE_CSS}
```

(an f-string interpolation of a module global named `_STYLE_CSS`; braces inside the interpolated VALUE are safe — f-strings only parse braces in the literal). The `<style>` open/close tags, the `<!-- plotly-engine-embed -->` comment, and everything else in `_build_html` stay byte-identical. The rendered CSS must be byte-identical to the pre-move output.

(e) Everything else in the file stays byte-identical.

- [ ] **Step 3: Replace `generate_dashboard.py` with the thin wrapper**

```python
"""Thin entry point — all logic lives in dashboard_ui.report."""

from dashboard_ui.report import generate_dashboard, main

__all__ = ["generate_dashboard", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
```

(Re-exporting `generate_dashboard` keeps `tests/test_scripts.py::test_generate_dashboard_import_surface` green unchanged.)

- [ ] **Step 4: Update the test import**

`tests/test_dashboard.py` line 13: `import generate_dashboard` → `from dashboard_ui import report as generate_dashboard`

(Every existing `generate_dashboard._*` / `generate_dashboard.generate_dashboard(...)` reference now resolves through the alias — no other test edits.)

- [ ] **Step 5: Add the new style-inline test**

Append to `tests/test_dashboard.py` (this mirrors `test_generate_dashboard_end_to_end_synthetic` — same public path, extra style markers; `_write_registry` and `_registry_entry` already exist in the file):

```python
def test_report_inlines_style_css_once(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64)])
    out = generate_dashboard.generate_dashboard(
        registry_dir=tmp_path, benchmark_path=False,
        output_path=tmp_path / "dashboard.html", open_browser=False,
    )
    text = out.read_text(encoding="utf-8")
    assert text.count(".badge.full {") == 1        # style.css inlined once — CSS rule, not the HTML class attr
    assert text.count(".group-header td {") == 1
    assert text.count("window.Plotly") == 1        # plotly engine embedded exactly once
    assert text.count("dashboard-multimetric-data") == 1  # app.js data node present
```

- [ ] **Step 6: Run the targeted tests**

Run: `./.venv/Scripts/python -m pytest -q tests/test_dashboard.py tests/test_scripts.py`
Expected: PASS (including the pre-existing `_build_html` determinism/structure tests — proof the CSS extraction is byte-identical).

- [ ] **Step 7: Lint**

Run: `./.venv/Scripts/python -m ruff check dashboard_ui/ generate_dashboard.py tests/test_dashboard.py`
Expected: clean.

- [ ] **Step 8: Bump the test-count claim (817 → 818)**

Run: `./.venv/Scripts/python -m pytest --collect-only -q 2>&1 | tail -2`
Update EVERY "N tests" claim in AGENTS.md (two places) and CONTRIBUTING.md to the collected count (expect 818).

- [ ] **Step 9: Run the full suite**

Run: `./.venv/Scripts/python -m pytest -q` (foreground, `timeout=300`; wait via TaskOutput `block=true` if auto-backgrounded). Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add dashboard_ui/ generate_dashboard.py tests/test_dashboard.py AGENTS.md CONTRIBUTING.md
git commit -m "refactor(dashboard): move report compiler into dashboard_ui with static style.css + thin wrapper"
```

---

### Task 3: `dashboard_ui/app.py` + thin wrapper

**Files:**
- Create: `dashboard_ui/app.py`
- Replace: `dashboard_app.py` (thin wrapper)
- Modify: `tests/test_dashboard.py` (3 import sites), `tests/test_scripts.py` (2 import sites)

**Interfaces:**
- Consumes: `dashboard_ui` sibling modules (not required — `app.py` uses `nmr.dashboard` only) and the unchanged `nmr.meta` import.
- Produces: `dashboard_ui.app` with the SAME public names as `dashboard_app` (`main`, `load_registry_frame`, `load_benchmarks`, `merge_leaderboard`, `_shaped_leaderboard_pdf`, `robustness_matrix`, `champion_run_id`, `_load_registry_entries`, `_read_run_payload`, `_read_full_manifest`, `render_leaderboard`, `render_run_detail`, `render_fleet`, `render_campaigns`, `render_robustness_matrix`, `load_campaigns`, `_CAMPAIGN_SCHEMA`, `_LEADERBOARD_SCHEMA`).

- [ ] **Step 1: Create `dashboard_ui/app.py`**

Copy `dashboard_app.py` (all 555 lines) verbatim, then apply EXACTLY these edits:

(a) Module docstring first paragraph → update any reference to itself ("dashboard_app") to "dashboard_ui.app"; keep the "headless; no server is launched" note.

(b) No import changes (the module imports `nmr.config`, `nmr.dashboard`, `nmr.meta` — all unchanged).

(c) Everything else byte-identical.

- [ ] **Step 2: Replace `dashboard_app.py` with the thin wrapper**

```python
"""Thin entry point — all logic lives in dashboard_ui.app."""

from dashboard_ui.app import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Update `tests/test_dashboard.py` import sites**

Lines 1225, 1243, 1258 (inside three test functions): `import dashboard_app as app` → `from dashboard_ui import app`

(The local name `app` binds to the `dashboard_ui.app` module — every `app.load_registry_frame(...)`, `app._shaped_leaderboard_pdf(...)`, `app.robustness_matrix(...)` reference resolves unchanged.)

- [ ] **Step 4: Update `tests/test_scripts.py` import sites**

Line 27: `import dashboard_app  # noqa: E402  (lazy: streamlit is heavy at module load)` → `from dashboard_ui import app as dashboard_app  # noqa: E402  (lazy: streamlit is heavy at module load)`

Line 198 (inside `test_dashboard_app_imports_without_launching`): `import dashboard_app  # noqa: F401` → `from dashboard_ui import app as dashboard_app`

(`test_dashboard_app_imports_without_launching` asserts `callable(dashboard_app.main)` plus the five `render_*` views, and the campaigns tests at ~line 188 use `dashboard_app.load_campaigns` / `dashboard_app._CAMPAIGN_SCHEMA` — all exist in `dashboard_ui.app`, so the alias keeps the assertions valid.)

- [ ] **Step 5: Run the targeted tests**

Run: `./.venv/Scripts/python -m pytest -q tests/test_dashboard.py tests/test_scripts.py`
Expected: PASS.

- [ ] **Step 6: Lint**

Run: `./.venv/Scripts/python -m ruff check dashboard_ui/ dashboard_app.py tests/test_dashboard.py tests/test_scripts.py`
Expected: clean.

- [ ] **Step 7: Run the full suite (no count change — no new test nodes)**

Run: `./.venv/Scripts/python -m pytest -q` (foreground, `timeout=300`; wait via TaskOutput `block=true` if auto-backgrounded). Expected: PASS at 818 (do NOT bump claims — no tests added).

- [ ] **Step 8: Commit**

```bash
git add dashboard_ui/app.py dashboard_app.py tests/test_dashboard.py tests/test_scripts.py
git commit -m "refactor(dashboard): move streamlit app into dashboard_ui with thin wrapper"
```

---

### Task 4: SSOT docs sync + final verification gate

**Files:**
- Modify: `AGENTS.md`, `ARCHITECTURE.md`, `README.md`, `CONTRIBUTING.md` (counts already current from Tasks 1–3)

**Interfaces:**
- Consumes: the moved surface from Tasks 1–3 (document it exactly as built).

- [ ] **Step 1: `AGENTS.md` — re-point the invariant and toolkit**

(a) Line 82 (the dependency carve-out): replace

```text
Streamlit + Plotly (interactive dashboard — imported only in the top-level control planes `dashboard_app.py`, `dashboard_charts.py`, `generate_dashboard.py`; never in `nmr/`)
```

with

```text
Streamlit + Plotly (interactive dashboard — imported only in the `dashboard_ui/` package (`charts.py`, `report.py`, `app.py`) and its thin entry wrappers `dashboard_app.py` / `generate_dashboard.py`; never in `nmr/`)
```

(b) Toolkit table rows referencing the old modules: `dashboard_charts.py` → `dashboard_ui/charts.py`; `generate_dashboard.py` → `dashboard_ui/report.py` (and its thin wrapper `generate_dashboard.py`); `dashboard_app.py` → `dashboard_ui/app.py` (and its thin wrapper). Update the dashboard engine row if it cites `dashboard_charts.py` (`dashboard_charts.py` + `generate_dashboard.py` + `dashboard_app.py` → the `dashboard_ui/` package).

(c) Verify the test-count claims already say 818 (bumped in Tasks 1–2); if a claim is stale, fix it.

(d) Budget: AGENTS.md must stay ≤ 32 KiB (docs-hygiene T4 enforces; current ~29 KB — the re-pointing is small).

- [ ] **Step 2: `ARCHITECTURE.md` — module map and dependency graph**

- In the module dependency graph / module map, replace the three dashboard leaf rows (`dashboard_charts.py`, `generate_dashboard.py`, `dashboard_app.py`) with:
  - `dashboard_ui/charts.py` — Plotly figure builders + multimetric JS controller (consumes `nmr.dashboard`)
  - `dashboard_ui/report.py` — HTML report compiler (consumes `nmr.dashboard` + `dashboard_ui.charts`)
  - `dashboard_ui/app.py` — Streamlit app (consumes `nmr.dashboard`)
  - `generate_dashboard.py` / `dashboard_app.py` — thin entry wrappers into `dashboard_ui.report` / `dashboard_ui.app`
- Update the dashboard section's file references and any artifact-layout mention of `dashboard_charts.py`.

- [ ] **Step 3: `README.md` — annotated tree and entry descriptions**

- Add `dashboard_ui/` to the annotated tree (e.g. `├── dashboard_ui/  # front-end: charts, report compiler, streamlit app, static assets`).
- Update the entry-script descriptions: `generate_dashboard.py` — "thin wrapper — builds the executive HTML dashboard (logic in `dashboard_ui.report`)"; `dashboard_app.py` — "thin wrapper — interactive Streamlit dashboard (logic in `dashboard_ui.app`)". Remove the `dashboard_charts.py` line.

- [ ] **Step 4: Docs-hygiene + targeted verification**

Run: `./.venv/Scripts/python -m pytest -q tests/test_docs_hygiene.py tests/test_dashboard.py tests/test_scripts.py`
Expected: PASS.

- [ ] **Step 5: Full lint + functional gate**

Run: `./.venv/Scripts/python -m ruff check .`
Run: `./.venv/Scripts/python -m pytest -q` (foreground, `timeout=300`; wait via TaskOutput `block=true` if auto-backgrounded)
Expected: both PASS; final collected count 818.

- [ ] **Step 6: Real-data compile + artifact validation**

Run: `./.venv/Scripts/python generate_dashboard.py`
Run:

```bash
./.venv/Scripts/python -c "
from pathlib import Path
text = Path('artifacts/dashboard.html').read_text(encoding='utf-8')
assert text.count('window.Plotly') == 1
assert 'cdn.plot.ly' not in text
assert text.count('dashboard-multimetric-data') == 1
assert '.badge.full {' in text          # style.css inlined
assert 'METRIC_CONFIG' in text          # app.js inlined
print('front-end isolation HTML verified')
"
```

Leave `artifacts/dashboard.html` in the working tree (machine-generated; do NOT commit).

- [ ] **Step 7: No stray imports of the old module names**

Run:

```bash
grep -rn "import dashboard_charts\|from dashboard_charts\|from generate_dashboard\|from dashboard_app" nmr/ dashboard_ui/ tests/ generate_dashboard.py dashboard_app.py || echo "CLEAN"
```

Expected: CLEAN (aliases like `import generate_dashboard` in tests/test_scripts.py and `import dashboard_app` are the thin WRAPPERS — the pattern above excludes them; verify the wrapper imports are `from dashboard_ui...`).

- [ ] **Step 8: Commit**

```bash
git add AGENTS.md ARCHITECTURE.md README.md
git commit -m "docs: re-point SSOT docs at dashboard_ui package; thin wrappers"
```

- [ ] **Step 9: Report**

Summarize: the final `dashboard_ui/` tree, the two wrappers, `dashboard_charts.py` deleted, final test count 818, lint + full-suite + HTML-validation results, and note the pending human browser check of the JS-controller chart (unchanged behavior, but worth re-verifying on the regenerated `artifacts/dashboard.html`).
