# Pure Vanilla HTML/CSS/SVG Executive Dashboard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Historical record:** This plan documents the original implementation sequence and estimates. It is superseded by the approved design specification and the current repository contracts in `AGENTS.md`, `ARCHITECTURE.md`, and `CONTRIBUTING.md`; its original size and test-count estimates are not active release requirements.

**Goal:** Eliminate Plotly from the repository and replace the executive dashboard's presentation layer with an isolated, zero-runtime-dependency Vanilla HTML/CSS/SVG front-end that compiles `artifacts/dashboard.html` under the current documented artifact budget.

**Architecture:** `nmr/dashboard.py` remains the renderer-neutral analytical engine (plotly-free already). `dashboard_ui/charts.py` gains pure, pytest-tested SVG geometry reference functions (`data_to_svg_path`, `svg_area_path`, `cumulative_series`, `drawdown_series`, `global_y_range`) and a payload builder emitting a standard-only, metric-first JSON contract. `dashboard_ui/static/` holds the raw assets (`layout.html`, `style.css`, `app.js`); `app.js` mirrors the geometry client-side and renders the five chart sections. `dashboard_ui/report.py` is the compiler (template + inlined assets + `#dashboard-data` node). `dashboard_ui/app.py` keeps the Streamlit app but drops `plotly.express` for native `st.*` charts.

**Tech Stack:** Python 3.11+, Polars, NumPy, pytest, ruff; vanilla HTML/CSS/JS (no Plotly, no node, no CDN).

**Spec:** `docs/superpowers/specs/2026-08-18-vanilla-dashboard-design.md`

## Global Constraints

- The current release contract is `nmr/dashboard.py` renderer-neutral alignment, a 112 KiB report budget, and 1,099 collected tests; see the active repository documents above rather than these historical task estimates.
- No `plotly` runtime dependency or visual renderer may remain in the active dashboard surface. Build-time Terser/CleanCSS commands are documented in `CONTRIBUTING.md`.
- `artifacts/dashboard.html` is gitignored (`artifacts/**`) — regenerating it never creates a diff; do not `git add` it.
- Windows venv: always `./.venv/Scripts/python -m ...` (never the `Scripts/pip` shim). Run the full suite in the foreground; if it auto-backgrounds, wait on it via `TaskOutput block=true`.
- ruff: E/F/I/UP, line length 120 (`ruff.toml`).
- Determinism: no wall-clock, no absolute paths in generated HTML; `json.dumps(..., sort_keys=True, allow_nan=False)` and `</` → `<\/` in the data node.
- Branch: `feature/vanilla-dashboard` off `main`. Ledger: `.superpowers/sdd/2026-08-18-vanilla-dashboard/progress.md`.

---

## File Structure

| File | Responsibility |
|---|---|
| `dashboard_ui/charts.py` (modify) | Pure geometry reference + payload builder; **Plotly figure builders deleted in Task 3** |
| `dashboard_ui/__init__.py` (modify) | Re-export the new `charts` public API |
| `dashboard_ui/static/layout.html` (create) | Semantic HTML scaffold with `{{ PLACEHOLDER }}` slots |
| `dashboard_ui/static/style.css` (rewrite) | Full design system (tokens, badges, grid, tooltip, svg rules) |
| `dashboard_ui/static/app.js` (rewrite) | Vanilla SVG renderer mirroring `charts.py` geometry |
| `dashboard_ui/report.py` (rewrite in Task 3) | Compiler: template + inlined assets + data node; keeps `_kpi_cards`, `_table_rows`, `_row_html`, `_technical_entries`, `_diversification_stats`, `_ensemble_sharpe`, `_bar_label`, `_bar_input` |
| `dashboard_ui/app.py` (modify in Task 4) | Native Streamlit charts; pure shaping helpers untouched |
| `tests/test_dashboard_ui.py` (create) | Presentation tests: geometry, payload, compiler, artifact contract |
| `tests/test_dashboard.py` (modify) | Engine-only; Plotly-era presentation tests pruned in Task 3 |
| `tests/test_scripts.py` (modify in Task 4) | Add no-plotly guard for `dashboard_ui/app.py` |
| `requirements.txt` (modify in Task 5) | Remove `plotly==6.6.0` |
| `AGENTS.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `README.md` (modify in Task 5) | SSOT sync |

---

### Task 1: charts.py — geometry reference + payload builder (additive)

**Files:**
- Modify: `dashboard_ui/charts.py` (add functions; **do not delete the Plotly builders yet** — their tests still live until Task 3)
- Modify: `dashboard_ui/__init__.py`
- Create: `tests/test_dashboard_ui.py`

**Interfaces:**
- Consumes: nothing new (stdlib + numpy + polars).
- Produces (later tasks rely on these exact signatures):
  - `charts.data_to_svg_path(values: Sequence[float], *, width: float, height: float, y_min: float | None = None, y_max: float | None = None, pad: tuple[float, float, float, float] = (20.0, 20.0, 20.0, 40.0)) -> str` — pad order `(top, right, bottom, left)`; `""` for empty input; SVG y inverted; zero-span guard.
  - `charts.svg_area_path(values: Sequence[float], *, width: float, height: float, y_min: float | None = None, y_max: float | None = None, y_baseline: float = 0.0, pad: tuple[float, float, float, float] = (20.0, 20.0, 20.0, 40.0)) -> str` — closed polygon ending `L xN,yBase L x0,yBase Z`.
  - `charts.cumulative_series(standard: Sequence[float], *, payout: bool) -> list[float]` — `cumprod(1+r)` when payout, else `cumsum`.
  - `charts.drawdown_series(cumulative: Sequence[float]) -> list[float]` — `wealth/peak - 1`.
  - `charts.global_y_range(*series: Sequence[float]) -> tuple[float, float]` — `(0.0, 1.0)` for empty input.
  - `charts.build_dashboard_payload(*, eras: Sequence[str], meta_downside_mask: Sequence[bool], metrics: Mapping[str, Mapping[str, Mapping[str, Any]]], leaderboard_bars: pl.DataFrame, similarity_labels: Sequence[str], similarity_matrix: Sequence[Sequence[float]], hurdle_sharpe: float, ensemble_sharpe: float | None) -> dict[str, Any]` — metric-first, standard-only.

- [ ] **Step 1: Write the failing tests** — create `tests/test_dashboard_ui.py`:

```python
"""Presentation-layer tests for the vanilla dashboard (geometry, payload)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import nmr.dashboard as nmr_dashboard
from dashboard_ui import charts


def test_data_to_svg_path_basic_polyline() -> None:
    path = charts.data_to_svg_path(
        [0.0, 1.0], width=100.0, height=100.0,
        y_min=0.0, y_max=1.0, pad=(0.0, 0.0, 0.0, 0.0),
    )
    assert path == "M 0.0,100.0 L 100.0,0.0"


def test_data_to_svg_path_inverts_y() -> None:
    # larger value -> smaller SVG y (returns rise on screen)
    path = charts.data_to_svg_path(
        [1.0, 2.0], width=100.0, height=100.0,
        y_min=0.0, y_max=2.0, pad=(0.0, 0.0, 0.0, 0.0),
    )
    assert path == "M 0.0,50.0 L 100.0,0.0"


def test_data_to_svg_path_zero_span_guard() -> None:
    path = charts.data_to_svg_path(
        [0.5, 0.5], width=100.0, height=100.0, pad=(10.0, 10.0, 10.0, 10.0),
    )
    assert "NaN" not in path and "Inf" not in path
    ys = [pt.split(",")[1] for pt in path.split(" L ")]
    assert ys[0] == ys[1]


def test_data_to_svg_path_single_point() -> None:
    path = charts.data_to_svg_path(
        [0.5], width=100.0, height=100.0,
        y_min=0.0, y_max=1.0, pad=(0.0, 0.0, 0.0, 0.0),
    )
    assert path == "M 0.0,50.0"


def test_data_to_svg_path_empty_input() -> None:
    assert charts.data_to_svg_path([], width=100.0, height=100.0) == ""


def test_svg_area_path_closes_to_baseline() -> None:
    path = charts.svg_area_path(
        [0.0, -0.1], width=100.0, height=100.0,
        y_min=-0.1, y_max=0.0, y_baseline=0.0, pad=(0.0, 0.0, 0.0, 0.0),
    )
    assert path.endswith(" Z")
    assert " L 100.0,0.0 L 0.0,0.0 Z" in path


def test_svg_area_path_empty_input() -> None:
    assert charts.svg_area_path([], width=100.0, height=100.0) == ""


def test_cumulative_series_parity_with_engine() -> None:
    standard = [0.01, -0.02, 0.03]
    assert charts.cumulative_series(standard, payout=True) == \
        nmr_dashboard._cumulative_from_standard(standard, payout=True)
    assert charts.cumulative_series(standard, payout=False) == \
        nmr_dashboard._cumulative_from_standard(standard, payout=False)


def test_drawdown_series_peak_trough() -> None:
    cum = [1.0, 0.9, 1.05]
    dd = charts.drawdown_series(cum)
    assert dd == pytest.approx([0.0, -0.1, 0.0])


def test_global_y_range_across_series() -> None:
    assert charts.global_y_range([1.0, 2.0], [3.0, 0.5]) == (0.5, 3.0)
    assert charts.global_y_range() == (0.0, 1.0)
    assert charts.global_y_range([], []) == (0.0, 1.0)


def test_build_dashboard_payload_metric_first_standard_only() -> None:
    engine_metrics = {
        "payout": {"a" * 64: {"standard": [0.01, 0.02],
                              "cumulative": [1.01, 1.0302], "label": "r1"}},
        "corr20": {"a" * 64: {"standard": [0.1, 0.2],
                              "cumulative": [0.1, 0.3], "label": "r1"}},
    }
    bars = pl.DataFrame([
        {"label": "r1 · abc", "corr_sharpe_ac": 0.8, "corr_sharpe_ac_ci_low": 0.6,
         "corr_sharpe_ac_ci_high": 1.0, "cagr_1y": 0.5, "max_drawdown": 0.1,
         "deflated_sharpe": 0.97, "champion": True},
    ])
    payload = charts.build_dashboard_payload(
        eras=["0001", "0002"], meta_downside_mask=[False, True],
        metrics=engine_metrics, leaderboard_bars=bars,
        similarity_labels=["a"], similarity_matrix=[[1.0]],
        hurdle_sharpe=0.78, ensemble_sharpe=1.2,
    )
    assert set(payload["metrics"]) == {"payout", "corr20"}   # metric-first keys
    entry = payload["metrics"]["payout"]["a" * 64]
    assert entry["standard"] == [0.01, 0.02]
    assert "cumulative" not in entry                         # standard-only
    assert entry["label"] == "r1"
    assert payload["eras"] == ["0001", "0002"]
    assert payload["meta_downside_mask"] == [False, True]
    assert payload["leaderboard"][0]["sharpe"] == 0.8
    assert payload["leaderboard"][0]["champion"] is True
    assert payload["similarity"] == {"labels": ["a"], "matrix": [[1.0]]}
    assert payload["hurdle_sharpe"] == 0.78
    assert payload["ensemble_sharpe"] == 1.2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_dashboard_ui.py -q`
Expected: FAIL with `AttributeError: module 'dashboard_ui.charts' has no attribute 'data_to_svg_path'` (functions don't exist yet).

- [ ] **Step 3: Implement `charts.py` additions** — append to `dashboard_ui/charts.py` (keep the existing Plotly builders and imports in place; add `import numpy as np` and, if not already present, `from collections.abc import Mapping, Sequence` and `from typing import Any`):

```python
_METRIC_NAMES = ("payout", "corr20", "mmc20", "corr60", "mmc60", "bmc", "cwmm")
_ZERO_SPAN_EPS = 1e-12
_PAYLOAD_ROUND = 6


def _round6(value: Any) -> Any:
    """Round payload floats to 6 decimals (display precision is 4) — keeps the
    data node honest while fitting the 112 KiB artifact budget (amendment)."""
    if isinstance(value, (int, float, np.floating)):
        return round(float(value), _PAYLOAD_ROUND)
    return value


def global_y_range(*series: Sequence[float]) -> tuple[float, float]:
    """Global min/max across all series (shared axis); (0.0, 1.0) when empty."""
    values = [v for s in series for v in s]
    if not values:
        return (0.0, 1.0)
    return (float(min(values)), float(max(values)))


def _resolve_range(
    values: Sequence[float], y_min: float | None, y_max: float | None
) -> tuple[float, float]:
    """Resolve the y range, expanding a degenerate flat span so scaling never divides by zero."""
    lo, hi = global_y_range(values) if y_min is None or y_max is None else (y_min, y_max)
    if abs(hi - lo) < _ZERO_SPAN_EPS:
        lo -= 1.0
        hi += 1.0
    return lo, hi


def data_to_svg_path(
    values: Sequence[float],
    *,
    width: float,
    height: float,
    y_min: float | None = None,
    y_max: float | None = None,
    pad: tuple[float, float, float, float] = (20.0, 20.0, 20.0, 40.0),
) -> str:
    """Map a series to an SVG polyline path (y axis inverted, top-left origin).

    ``pad`` order is (top, right, bottom, left). Empty input returns ``""``.
    """
    if not values:
        return ""
    lo, hi = _resolve_range(values, y_min, y_max)
    span = hi - lo
    pad_top, pad_right, pad_bottom, pad_left = pad
    inner_w = width - pad_left - pad_right
    inner_h = height - pad_top - pad_bottom
    denom = max(1, len(values) - 1)
    points = []
    for i, v in enumerate(values):
        x = pad_left + (i / denom) * inner_w
        y = pad_top + (1.0 - (v - lo) / span) * inner_h
        points.append(f"{x:.1f},{y:.1f}")
    return "M " + " L ".join(points)


def svg_area_path(
    values: Sequence[float],
    *,
    width: float,
    height: float,
    y_min: float | None = None,
    y_max: float | None = None,
    y_baseline: float = 0.0,
    pad: tuple[float, float, float, float] = (20.0, 20.0, 20.0, 40.0),
) -> str:
    """Closed SVG polygon: line path + baseline anchors (``L xN,yBase L x0,yBase Z``)."""
    if not values:
        return ""
    lo, hi = _resolve_range(values, y_min, y_max)
    span = hi - lo
    line = data_to_svg_path(values, width=width, height=height, y_min=lo, y_max=hi, pad=pad)
    pad_top, pad_right, pad_bottom, pad_left = pad
    inner_h = height - pad_top - pad_bottom
    inner_w = width - pad_left - pad_right
    y_base = pad_top + (1.0 - (y_baseline - lo) / span) * inner_h
    denom = max(1, len(values) - 1)
    x0 = pad_left
    x_n = pad_left + ((len(values) - 1) / denom) * inner_w
    return f"{line} L {x_n:.1f},{y_base:.1f} L {x0:.1f},{y_base:.1f} Z"


def cumulative_series(standard: Sequence[float], *, payout: bool) -> list[float]:
    """cumprod(1+r) for payout, cumsum(rho) for correlations (spec decision #9)."""
    values = np.asarray(standard, dtype=float)
    if payout:
        return [float(v) for v in np.cumprod(1.0 + values)]
    return [float(v) for v in np.cumsum(values)]


def drawdown_series(cumulative: Sequence[float]) -> list[float]:
    """wealth/peak - 1 (peak = running maximum)."""
    wealth = np.asarray(cumulative, dtype=float)
    peak = np.maximum.accumulate(wealth)
    return [float(v) for v in wealth / peak - 1.0]


def build_dashboard_payload(
    *,
    eras: Sequence[str],
    meta_downside_mask: Sequence[bool],
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    leaderboard_bars: pl.DataFrame,
    similarity_labels: Sequence[str],
    similarity_matrix: Sequence[Sequence[float]],
    hurdle_sharpe: float,
    ensemble_sharpe: float | None,
) -> dict[str, Any]:
    """Shape engine output into the standard-only, metric-first vanilla contract.

    ``metrics`` mirrors ``nmr.dashboard.extract_multimetric_timeseries``
    (metric-first); cumulative/drawdown are derived client-side, so only the
    ``standard`` arrays and labels are carried. ``leaderboard_bars`` must be a
    frame with columns ``label, corr_sharpe_ac, corr_sharpe_ac_ci_low,
    corr_sharpe_ac_ci_high, cagr_1y, max_drawdown, deflated_sharpe, champion``.
    """
    shaped_metrics: dict[str, Any] = {}
    for metric, models in metrics.items():
        shaped_metrics[metric] = {
            model_id: {"standard": [_round6(v) for v in series["standard"]],
                       "label": series["label"]}
            for model_id, series in models.items()
        }
    rows = [
        {
            "label": row["label"],
            "sharpe": _round6(row["corr_sharpe_ac"]),
            "ci_low": _round6(row["corr_sharpe_ac_ci_low"]),
            "ci_high": _round6(row["corr_sharpe_ac_ci_high"]),
            "cagr_1y": _round6(row.get("cagr_1y")),
            "max_drawdown": _round6(row.get("max_drawdown")),
            "deflated_sharpe": _round6(row.get("deflated_sharpe")),
            "champion": row["champion"],
        }
        for row in leaderboard_bars.to_dicts()
    ]
    return {
        "eras": list(eras),
        "meta_downside_mask": [bool(m) for m in meta_downside_mask],
        "metrics": shaped_metrics,
        "leaderboard": rows,
        "similarity": {
            "labels": list(similarity_labels),
            "matrix": [list(r) for r in similarity_matrix],
        },
        "hurdle_sharpe": float(hurdle_sharpe),
        "ensemble_sharpe": ensemble_sharpe,
    }
```

Also append these names to the module's `__all__` if one exists, or add one:

```python
__all__ = [
    "build_dashboard_payload",
    "cumulative_series",
    "data_to_svg_path",
    "drawdown_series",
    "global_y_range",
    "svg_area_path",
]
```

- [ ] **Step 4: Update `dashboard_ui/__init__.py`** to re-export the new public API:

```python
"""Presentation layer for the executive dashboard.

All front-end code (vanilla SVG/HTML/JS report compiler, the Streamlit app,
and static assets) lives here. Pure engine logic stays in ``nmr/``; this
package only consumes ``nmr.dashboard``.
"""

from dashboard_ui.charts import (
    build_dashboard_payload,
    cumulative_series,
    data_to_svg_path,
    drawdown_series,
    global_y_range,
    svg_area_path,
)

__all__ = [
    "build_dashboard_payload",
    "cumulative_series",
    "data_to_svg_path",
    "drawdown_series",
    "global_y_range",
    "svg_area_path",
]
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_dashboard_ui.py -q`
Expected: PASS (11 tests).

- [ ] **Step 6: Full gate + test-count sync + commit**

Run: `.\.venv\Scripts\python -m ruff check .` then `.\.venv\Scripts\python -m pytest -q`
Expected: all green (the old Plotly builders and their tests are untouched, so the previous 818 still pass; new total = 829).
Then update the three count claims (`AGENTS.md:33`, `AGENTS.md:197`, `CONTRIBUTING.md:30`) to the collected count from `pytest --collect-only -q | tail -1`, and commit:

```bash
git add dashboard_ui/charts.py dashboard_ui/__init__.py tests/test_dashboard_ui.py AGENTS.md CONTRIBUTING.md
git commit -m "feat(dashboard_ui): SVG geometry reference + metric-first payload builder"
```

---

### Task 2: static assets — style.css design system + app.js renderer

**Files:**
- Create: `dashboard_ui/static/layout.html`
- Rewrite: `dashboard_ui/static/style.css` (full replacement)
- Rewrite: `dashboard_ui/static/app.js` (full replacement)
- Test: `tests/test_dashboard_ui.py` (append two structural tests)

**Interfaces:**
- Consumes: the `#dashboard-data` JSON node shape produced by `charts.build_dashboard_payload` (Task 1) and the DOM ids below.
- Produces (Task 3 wires these into the compiler): `layout.html` with placeholders `{{ INLINE_STYLE }}`, `{{ N_ERAS }}`, `{{ DATA_VERSION }}`, `{{ KPI_CARDS }}`, `{{ METRIC_CONTROLS }}`, `{{ TIMESERIES_SVG }}`, `{{ LEADERBOARD_SVG }}`, `{{ DIVERSIFICATION_SECTION }}`, `{{ DECISION_TABLE }}`, `{{ DRAWDOWN_SVG }}`, `{{ AUDIT_ACCORDION }}`, `{{ INLINE_DATA_SCRIPT }}`; DOM ids `#dashboard-root`, `#metric-select`, `#view-standard`, `#view-cumulative`, `#axis-label`, `#timeseries-svg`, `#timeseries-tooltip`, `#leaderboard-svg`, `#similarity-host`, `#drawdown-svg`.

- [ ] **Step 1: Write the failing structural tests** — append to `tests/test_dashboard_ui.py`:

```python
def test_style_css_design_tokens() -> None:
    from dashboard_ui import report
    css = (Path(report._STATIC_DIR) / "style.css").read_text(encoding="utf-8")
    for token in ("--bg: #0d1117", "--surface: #161b22", "--border: #30363d",
                  "--text: #c9d1d9", "--accent: #58a6ff", "--danger: #f85149",
                  "--success: #3fb950", "--gold: #d29922"):
        assert token in css
    for selector in (".badge.champion", ".badge.ready", ".badge.research",
                     ".badge.hurdle", ".badge.benchmark", ".badge.full",
                     ".gate-fail", ".grid-line", ".crosshair", ".tooltip"):
        assert selector in css


def test_app_js_contains_renderer_functions() -> None:
    from dashboard_ui import report
    js = (Path(report._STATIC_DIR) / "app.js").read_text(encoding="utf-8")
    for fn in ("dataToSvgPath", "svgAreaPath", "cumulativeSeries",
               "drawdownSeries", "globalYRange", "renderTimeseries",
               "renderLeaderboard", "renderSimilarity", "renderDrawdown"):
        assert fn in js
    assert "</script" not in js   # inlined into a <script> node — must never close it


def test_layout_html_has_compiler_placeholders() -> None:
    from dashboard_ui import report
    layout = (Path(report._STATIC_DIR) / "layout.html").read_text(encoding="utf-8")
    for ph in ("{{ INLINE_STYLE }}", "{{ N_ERAS }}", "{{ DATA_VERSION }}",
               "{{ KPI_CARDS }}", "{{ METRIC_CONTROLS }}", "{{ TIMESERIES_SVG }}",
               "{{ LEADERBOARD_SVG }}", "{{ DIVERSIFICATION_SECTION }}",
               "{{ DECISION_TABLE }}", "{{ DRAWDOWN_SVG }}",
               "{{ AUDIT_ACCORDION }}", "{{ INLINE_DATA_SCRIPT }}"):
        assert ph in layout
```

Add the needed import at the top of the test file: `from pathlib import Path`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_dashboard_ui.py -q`
Expected: FAIL with `FileNotFoundError` (layout.html missing) and assertion failures (style.css too small / app.js lacks the functions).

- [ ] **Step 3: Create `dashboard_ui/static/layout.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NumerAI Executive Performance Report</title>
<style>
{{ INLINE_STYLE }}
</style>
</head>
<body>
<div id="dashboard-root">
<h1>🏆 NumerAI Executive Performance Report</h1>
<p>Evaluation window: {{ N_ERAS }} overlap eras · data version {{ DATA_VERSION }}</p>
<div class="kpis">{{ KPI_CARDS }}</div>

<h2>1. ALPHA GENERATION &amp; MULTI-METRIC PERFORMANCE TRAJECTORY</h2>
{{ METRIC_CONTROLS }}
{{ TIMESERIES_SVG }}

<h2>2. RISK-ADJUSTED RETURN LEADERBOARD</h2>
{{ LEADERBOARD_SVG }}

<h2>3. SIGNAL DIVERSIFICATION &amp; PAIRWISE SIMILARITY MATRIX</h2>
{{ DIVERSIFICATION_SECTION }}

<h2>4. EXECUTIVE ALLOCATION &amp; RISK DECISION TABLE</h2>
{{ DECISION_TABLE }}

<h2>5. CAPITAL DRAWDOWN (UNDERWATER TRAJECTORY)</h2>
{{ DRAWDOWN_SVG }}

<h2>Technical &amp; Audit Metadata</h2>
{{ AUDIT_ACCORDION }}
</div>
{{ INLINE_DATA_SCRIPT }}
</body>
</html>
```

- [ ] **Step 4: Rewrite `dashboard_ui/static/style.css`** (full replacement):

```css
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --border: #30363d;
  --text: #c9d1d9;
  --muted: #8b949e;
  --accent: #58a6ff;
  --danger: #f85149;
  --success: #3fb950;
  --gold: #d29922;
  --heading: #e6edf3;
}

* { box-sizing: border-box; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, "Segoe UI", sans-serif;
  margin: 0;
  padding: 1.5rem;
}

#dashboard-root { max-width: 1100px; margin: 0 auto; }

h1 { color: var(--heading); }
h2 {
  color: var(--heading); font-size: 1.1rem; margin-top: 2.5rem;
  border-bottom: 1px solid var(--border); padding-bottom: 0.4rem;
}

.kpis {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem; margin: 1rem 0 2rem;
}
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; }
.kpi .label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
.kpi .value { font-size: 1.4rem; font-weight: 600; color: var(--heading); }

.chart-box {
  position: relative; background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 0.75rem; margin: 0.75rem 0;
}
.chart-box svg { width: 100%; height: auto; display: block; }

.controls { display: flex; gap: 1rem; align-items: center; margin: 1rem 0 0.25rem; flex-wrap: wrap; }
.controls select, .controls button {
  background: var(--surface); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.4rem 0.8rem; font-size: 0.85rem; cursor: pointer;
}
.controls button.active { border-color: var(--accent); color: var(--accent); }
.axis-label { color: var(--muted); font-size: 0.85rem; }

.series-line { fill: none; stroke-width: 2; }
.grid-line { stroke: var(--border); stroke-dasharray: 4 4; }
.axis-text { fill: var(--muted); font-size: 11px; }
.hurdle-line { stroke: var(--danger); stroke-dasharray: 6 4; }
.hurdle-text { fill: var(--danger); font-size: 11px; }
.bar { fill: var(--accent); opacity: 0.85; }
.bar.champion-bar { fill: var(--gold); }
.bar-label { fill: var(--text); font-size: 12px; }
.bar-value { fill: var(--muted); font-size: 11px; }
.ci-whisker { stroke: var(--heading); stroke-width: 2; }
.drawdown-area { stroke-width: 1.5; }
.crosshair { stroke: var(--muted); stroke-dasharray: 3 3; }
svg .empty-note { fill: var(--muted); font-size: 13px; }
p.empty-note { color: var(--muted); text-align: center; padding: 2rem 0; }

.tooltip {
  position: absolute; left: 0; top: 0; z-index: 10;
  background: rgba(13, 17, 23, 0.95); border: 1px solid var(--border); border-radius: 6px;
  padding: 0.5rem 0.75rem; font-size: 0.8rem; pointer-events: none; max-width: 260px;
}

table {
  width: 100%; border-collapse: collapse; background: var(--surface);
  border: 1px solid var(--border);
}
th, td {
  padding: 0.6rem 0.8rem; border-bottom: 1px solid var(--border);
  text-align: left; font-size: 0.85rem;
}
th { background: #21262d; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--heading); }
.num { font-variant-numeric: tabular-nums; text-align: right; }
.gate-fail { color: var(--danger); font-weight: 500; }

.badge {
  display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px;
  font-size: 0.72rem; font-weight: 600; text-transform: uppercase; margin-right: 0.25rem;
}
.badge.champion { background: rgba(137, 87, 229, 0.2); color: #a371f7; border: 1px solid #8957e5; }
.badge.ready { background: rgba(46, 160, 67, 0.2); color: var(--success); border: 1px solid #2ea043; }
.badge.research { background: rgba(110, 118, 129, 0.2); color: var(--muted); border: 1px solid var(--border); }
.badge.hurdle { background: rgba(248, 81, 73, 0.2); color: var(--danger); border: 1px solid #da3633; }
.badge.benchmark { background: rgba(137, 87, 229, 0.12); color: #a371f7; border: 1px solid var(--border); }
.badge.full { background: rgba(210, 153, 34, 0.18); color: var(--gold); border: 1px solid #9e6a03; }

.group-header td {
  background: #21262d; color: var(--heading); font-weight: 600; text-transform: uppercase;
  font-size: 0.75rem; letter-spacing: 0.05em;
}

table.similarity th, table.similarity td { text-align: center; font-size: 0.75rem; }
table.similarity td.highlight { font-weight: 700; }

.badge-line { font-size: 0.9rem; }

details {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 0.5rem 1rem; margin: 0.5rem 0;
}
summary { cursor: pointer; color: var(--heading); }
pre { white-space: pre-wrap; font-size: 0.75rem; color: var(--text); }
```

- [ ] **Step 5: Rewrite `dashboard_ui/static/app.js`** (full replacement) — see the next step's full listing; write the complete file:

```javascript
/* dashboard_ui/static/app.js
   Vanilla SVG renderer for the executive report. Reads the #dashboard-data
   JSON node and renders the five chart sections into #dashboard-root.
   All geometry mirrors the tested Python reference in dashboard_ui/charts.py
   (data_to_svg_path / svg_area_path / cumulative_series / drawdown_series). */
(function () {
  "use strict";

  var root = document.getElementById("dashboard-root");
  var dataNode = document.getElementById("dashboard-data");
  if (!root || !dataNode) return;
  var payload;
  try {
    payload = JSON.parse(dataNode.textContent);
  } catch (err) {
    return; /* corrupt payload: keep the static report readable */
  }

  var METRIC_CONFIG = {
    payout: {standard: {label: "Per-Era Net Return", percent: true},
             cumulative: {label: "Cumulative Wealth (1.0 Stake)", percent: false}},
    corr20: {standard: {label: "Per-Era CORR (20D)", percent: false},
             cumulative: {label: "Cumulative CORR (20D)", percent: false}},
    mmc20:  {standard: {label: "Per-Era MMC (20D)", percent: false},
             cumulative: {label: "Cumulative MMC (20D)", percent: false}},
    corr60: {standard: {label: "Per-Era CORR (60D)", percent: false},
             cumulative: {label: "Cumulative CORR (60D)", percent: false}},
    mmc60:  {standard: {label: "Per-Era MMC (60D)", percent: false},
             cumulative: {label: "Cumulative MMC (60D)", percent: false}},
    bmc:    {standard: {label: "Per-Era BMC", percent: false},
             cumulative: {label: "Cumulative BMC", percent: false}},
    cwmm:   {standard: {label: "Per-Era CWMM", percent: false},
             cumulative: {label: "Cumulative CWMM", percent: false}}
  };
  var COLORS = ["#58a6ff", "#3fb950", "#d29922", "#a371f7", "#f85149", "#79c0ff", "#f0883e"];
  var TS = {width: 800, height: 320, pad: {top: 24, right: 24, bottom: 40, left: 56}};
  var LB = {width: 800, height: 420, pad: {top: 16, right: 40, bottom: 24, left: 190}};
  var DD = {width: 800, height: 240, pad: {top: 24, right: 24, bottom: 40, left: 56}};
  var currentMetric = "payout";
  var currentView = "standard";
  var eras = payload.eras || [];
  var stressMask = payload.meta_downside_mask || [];
  var metrics = payload.metrics || {};
  var crosshair = null;

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function fmt(v, percent) {
    if (v === null || v === undefined || !isFinite(v)) return "—";
    return percent ? (v * 100).toFixed(2) + "%" : v.toFixed(4);
  }

  function cumulativeSeries(values, payout) {
    var out = [];
    var acc = payout ? 1.0 : 0.0;
    for (var i = 0; i < values.length; i++) {
      acc = payout ? acc * (1.0 + values[i]) : acc + values[i];
      out.push(acc);
    }
    return out;
  }

  function drawdownSeries(cumulative) {
    var out = [];
    var peak = -Infinity;
    for (var i = 0; i < cumulative.length; i++) {
      if (cumulative[i] > peak) peak = cumulative[i];
      out.push(peak > 0 ? cumulative[i] / peak - 1.0 : 0.0);
    }
    return out;
  }

  function globalYRange(seriesList) {
    var lo = Infinity, hi = -Infinity;
    for (var s = 0; s < seriesList.length; s++) {
      var arr = seriesList[s];
      for (var i = 0; i < arr.length; i++) {
        var v = arr[i];
        if (!isFinite(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    if (!isFinite(lo)) return {min: 0.0, max: 1.0};
    return {min: lo, max: hi};
  }

  function dataToSvgPath(values, yMin, yMax, width, height, pad) {
    if (!values || !values.length) return "";
    var span = yMax - yMin;
    if (Math.abs(span) < 1e-12) { yMin -= 1.0; yMax += 1.0; span = 2.0; }
    var innerW = width - pad.left - pad.right;
    var innerH = height - pad.top - pad.bottom;
    var pts = [];
    var n = values.length;
    var denom = Math.max(1, n - 1);
    for (var i = 0; i < n; i++) {
      var x = pad.left + (i / denom) * innerW;
      var y = pad.top + (1.0 - (values[i] - yMin) / span) * innerH;
      pts.push(x.toFixed(1) + "," + y.toFixed(1));
    }
    return "M " + pts.join(" L ");
  }

  function svgAreaPath(values, yMin, yMax, yBase, width, height, pad) {
    var line = dataToSvgPath(values, yMin, yMax, width, height, pad);
    if (!line) return "";
    var span = yMax - yMin;
    if (Math.abs(span) < 1e-12) { yMin -= 1.0; yMax += 1.0; span = 2.0; }
    var innerH = height - pad.top - pad.bottom;
    var yBaseSvg = pad.top + (1.0 - (yBase - yMin) / span) * innerH;
    var innerW = width - pad.left - pad.right;
    var denom = Math.max(1, values.length - 1);
    var x0 = pad.left;
    var xN = pad.left + ((values.length - 1) / denom) * innerW;
    return line + " L " + xN.toFixed(1) + "," + yBaseSvg.toFixed(1) +
           " L " + x0.toFixed(1) + "," + yBaseSvg.toFixed(1) + " Z";
  }

  function activeSeries() {
    var metric = metrics[currentMetric] || {};
    var ids = Object.keys(metric).sort();
    var series = [];
    for (var i = 0; i < ids.length; i++) {
      var entry = metric[ids[i]];
      var standard = entry.standard || [];
      var values = currentView === "cumulative"
        ? cumulativeSeries(standard, currentMetric === "payout")
        : standard;
      series.push({id: ids[i], label: entry.label, values: values,
                   color: COLORS[i % COLORS.length]});
    }
    return series;
  }

  function stressShapes(svg) {
    var innerW = TS.width - TS.pad.left - TS.pad.right;
    var denom = Math.max(1, eras.length - 1);
    for (var i = 0; i < stressMask.length; i++) {
      if (!stressMask[i]) continue;
      var start = i;
      while (i + 1 < stressMask.length && stressMask[i + 1]) i++;
      var x0 = TS.pad.left + (start / denom) * innerW;
      var x1 = TS.pad.left + (i / denom) * innerW;
      var rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", x0.toFixed(1));
      rect.setAttribute("width", (x1 - x0).toFixed(1));
      rect.setAttribute("y", "0");
      rect.setAttribute("height", String(TS.height));
      rect.setAttribute("fill", "rgba(248, 81, 73, 0.10)");
      svg.appendChild(rect);
    }
  }

  function gridLines(svg, yMin, yMax, pad, fmtVal) {
    var span = yMax - yMin;
    var ticks = (yMin <= 0.0 && yMax >= 0.0)
      ? [yMax, 0.0, yMin]
      : [yMax, (yMin + yMax) / 2.0, yMin];
    var innerH = TS.height - pad.top - pad.bottom;
    for (var k = 0; k < ticks.length; k++) {
      var val = ticks[k];
      var y = pad.top + (1.0 - (val - yMin) / span) * innerH;
      var line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("class", "grid-line");
      line.setAttribute("x1", String(pad.left));
      line.setAttribute("x2", String(TS.width - pad.right));
      line.setAttribute("y1", y.toFixed(1));
      line.setAttribute("y2", y.toFixed(1));
      svg.appendChild(line);
      var label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("class", "axis-text");
      label.setAttribute("x", "4");
      label.setAttribute("y", (y - 4).toFixed(1));
      label.textContent = fmtVal(val);
      svg.appendChild(label);
    }
  }

  function renderTimeseries() {
    var svg = document.getElementById("timeseries-svg");
    if (!svg) return;
    var series = activeSeries();
    var cfg = METRIC_CONFIG[currentMetric][currentView];
    var labelSpan = document.getElementById("axis-label");
    if (labelSpan) labelSpan.textContent = cfg.label;
    var range = globalYRange(series.map(function (s) { return s.values; }));
    if (Math.abs(range.max - range.min) < 1e-12) { range.min -= 1.0; range.max += 1.0; }
    svg.textContent = "";
    stressShapes(svg);
    gridLines(svg, range.min, range.max, TS.pad, function (v) { return fmt(v, cfg.percent); });
    for (var i = 0; i < series.length; i++) {
      var path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", dataToSvgPath(series[i].values, range.min, range.max,
                                           TS.width, TS.height, TS.pad));
      path.setAttribute("stroke", series[i].color);
      path.setAttribute("class", "series-line");
      path.setAttribute("data-model", series[i].id);
      svg.appendChild(path);
    }
    crosshair = document.createElementNS("http://www.w3.org/2000/svg", "line");
    crosshair.setAttribute("class", "crosshair");
    crosshair.setAttribute("visibility", "hidden");
    crosshair.setAttribute("y1", String(TS.pad.top));
    crosshair.setAttribute("y2", String(TS.height - TS.pad.bottom));
    svg.appendChild(crosshair);
  }

  function renderLeaderboard() {
    var svg = document.getElementById("leaderboard-svg");
    if (!svg) return;
    var rows = payload.leaderboard || [];
    svg.textContent = "";
    if (!rows.length) {
      svg.appendChild(textNode(svg, "No models recorded yet", LB.width / 2, LB.height / 2));
      return;
    }
    var sorted = rows.slice().sort(function (a, b) {
      return (a.sharpe === null ? -Infinity : a.sharpe) - (b.sharpe === null ? -Infinity : b.sharpe);
    });
    var innerW = LB.width - LB.pad.left - LB.pad.right;
    var barH = 24, gap = 8;
    var totalH = Math.max(LB.height, LB.pad.top + LB.pad.bottom + sorted.length * (barH + gap));
    svg.setAttribute("viewBox", "0 0 " + LB.width + " " + totalH);
    var maxX = Math.max.apply(null, sorted.map(function (r) { return r.sharpe === null ? 0 : r.sharpe; }));
    if (!(maxX > 0)) maxX = 1.0;
    for (var i = 0; i < sorted.length; i++) {
      var row = sorted[i];
      var y = LB.pad.top + i * (barH + gap);
      var w = row.sharpe === null ? 0 : Math.max(0, row.sharpe / maxX) * innerW;
      var bar = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      bar.setAttribute("x", String(LB.pad.left));
      bar.setAttribute("y", String(y));
      bar.setAttribute("width", w.toFixed(1));
      bar.setAttribute("height", String(barH));
      bar.setAttribute("rx", "3");
      bar.setAttribute("class", row.champion ? "bar champion-bar" : "bar");
      svg.appendChild(bar);
      var label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("class", "bar-label");
      label.setAttribute("x", String(LB.pad.left - 8));
      label.setAttribute("y", (y + barH / 2 + 4).toFixed(1));
      label.setAttribute("text-anchor", "end");
      label.textContent = row.label;
      svg.appendChild(label);
      if (row.sharpe !== null) {
        var val = document.createElementNS("http://www.w3.org/2000/svg", "text");
        val.setAttribute("class", "bar-value");
        val.setAttribute("x", (LB.pad.left + w + 6).toFixed(1));
        val.setAttribute("y", (y + barH / 2 + 4).toFixed(1));
        val.textContent = row.sharpe.toFixed(3);
        svg.appendChild(val);
        if (row.ci_low !== null && row.ci_high !== null) {
          var xL = LB.pad.left + Math.max(0, row.ci_low / maxX) * innerW;
          var xH = LB.pad.left + Math.max(0, row.ci_high / maxX) * innerW;
          var whisker = document.createElementNS("http://www.w3.org/2000/svg", "line");
          whisker.setAttribute("x1", xL.toFixed(1));
          whisker.setAttribute("x2", xH.toFixed(1));
          whisker.setAttribute("y1", (y + barH / 2).toFixed(1));
          whisker.setAttribute("y2", (y + barH / 2).toFixed(1));
          whisker.setAttribute("class", "ci-whisker");
          svg.appendChild(whisker);
        }
      }
    }
    var hurdle = payload.hurdle_sharpe;
    if (hurdle !== null && hurdle !== undefined) {
      var hx = LB.pad.left + (hurdle / maxX) * innerW;
      var hline = document.createElementNS("http://www.w3.org/2000/svg", "line");
      hline.setAttribute("class", "hurdle-line");
      hline.setAttribute("x1", hx.toFixed(1));
      hline.setAttribute("x2", hx.toFixed(1));
      hline.setAttribute("y1", String(LB.pad.top));
      hline.setAttribute("y2", String(LB.height - LB.pad.bottom));
      svg.appendChild(hline);
      var htext = document.createElementNS("http://www.w3.org/2000/svg", "text");
      htext.setAttribute("class", "hurdle-text");
      htext.setAttribute("x", (hx + 4).toFixed(1));
      htext.setAttribute("y", String(LB.pad.top + 12));
      htext.textContent = "tier-4 hurdle " + hurdle.toFixed(2);
      svg.appendChild(htext);
    }
  }

  function renderSimilarity() {
    var host = document.getElementById("similarity-host");
    if (!host) return;
    var sim = payload.similarity || {labels: [], matrix: []};
    host.textContent = "";
    if (!sim.matrix.length) {
      host.appendChild(emptyNote("Similarity matrix unavailable without local v5.3 assets"));
      return;
    }
    var table = document.createElement("table");
    table.setAttribute("class", "similarity");
    var head = document.createElement("thead");
    var hr = document.createElement("tr");
    hr.appendChild(document.createElement("th"));
    for (var j = 0; j < sim.labels.length; j++) {
      var th = document.createElement("th");
      th.textContent = sim.labels[j];
      hr.appendChild(th);
    }
    head.appendChild(hr);
    table.appendChild(head);
    var body = document.createElement("tbody");
    for (var i = 0; i < sim.matrix.length; i++) {
      var tr = document.createElement("tr");
      var rowLabel = document.createElement("th");
      rowLabel.textContent = sim.labels[i];
      tr.appendChild(rowLabel);
      for (var k = 0; k < sim.matrix[i].length; k++) {
        var td = document.createElement("td");
        var v = sim.matrix[i][k];
        var isNum = v !== null && v !== undefined && isFinite(v);
        td.textContent = isNum ? v.toFixed(3) : "—";
        var alpha = isNum ? (0.05 + 0.85 * Math.abs(v)) : 0.0;
        var color = isNum ? (v < 0 ? "248, 81, 73" : "88, 166, 255") : "110, 118, 129";
        td.style.backgroundColor = "rgba(" + color + ", " + alpha.toFixed(2) + ")";
        if (i === 0 || k === 0) td.setAttribute("class", "highlight");
        tr.appendChild(td);
      }
      body.appendChild(tr);
    }
    table.appendChild(body);
    host.appendChild(table);
  }

  function renderDrawdown() {
    var svg = document.getElementById("drawdown-svg");
    if (!svg) return;
    svg.textContent = "";
    var payout = metrics.payout || {};
    var ids = Object.keys(payout).sort();
    if (!eras.length || !ids.length) {
      svg.appendChild(textNode(svg, "Timeseries data unavailable without local v5.3 assets",
                               DD.width / 2, DD.height / 2));
      return;
    }
    var paths = [];
    for (var i = 0; i < ids.length; i++) {
      var standard = payout[ids[i]].standard || [];
      paths.push(drawdownSeries(cumulativeSeries(standard, true)));
    }
    var range = globalYRange(paths);
    if (Math.abs(range.max - range.min) < 1e-12) { range.min -= 1.0; range.max += 1.0; }
    for (var k = 0; k < paths.length; k++) {
      var path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", svgAreaPath(paths[k], range.min, range.max, 0.0,
                                         DD.width, DD.height, DD.pad));
      path.setAttribute("class", "drawdown-area");
      path.setAttribute("fill", "rgba(248, 81, 73, 0.15)");
      path.setAttribute("stroke", COLORS[k % COLORS.length]);
      svg.appendChild(path);
    }
  }

  function textNode(svg, text, x, y) {
    var t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", String(x));
    t.setAttribute("y", String(y));
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("class", "empty-note");
    t.textContent = text;
    return t;
  }

  function emptyNote(text) {
    var p = document.createElement("p");
    p.setAttribute("class", "empty-note");
    p.textContent = text;
    return p;
  }

  function eraIndexFromX(x) {
    if (!eras.length) return -1;
    var innerW = TS.width - TS.pad.left - TS.pad.right;
    var t = Math.round((x - TS.pad.left) / (innerW / Math.max(1, eras.length - 1)));
    return Math.min(Math.max(t, 0), eras.length - 1);
  }

  function attachTooltip() {
    var svg = document.getElementById("timeseries-svg");
    var tip = document.getElementById("timeseries-tooltip");
    if (!svg || !tip || !eras.length) return;
    svg.addEventListener("mousemove", function (ev) {
      var rect = svg.getBoundingClientRect();
      var x = (ev.clientX - rect.left) * (TS.width / rect.width);
      var t = eraIndexFromX(x);
      if (t < 0 || !crosshair) { tip.hidden = true; return; }
      crosshair.setAttribute("visibility", "visible");
      var innerW = TS.width - TS.pad.left - TS.pad.right;
      var cx = TS.pad.left + (t / Math.max(1, eras.length - 1)) * innerW;
      crosshair.setAttribute("x1", cx.toFixed(1));
      crosshair.setAttribute("x2", cx.toFixed(1));
      var series = activeSeries();
      var cfg = METRIC_CONFIG[currentMetric][currentView];
      var lines = ["<b>Era " + esc(eras[t]) + "</b>"];
      for (var i = 0; i < series.length; i++) {
        lines.push('<span style="color:' + series[i].color + '">\u25CF</span> ' +
                   esc(series[i].label) + ": " + fmt(series[i].values[t], cfg.percent));
      }
      tip.innerHTML = lines.join("<br>");
      tip.hidden = false;
      tip.style.left = Math.min(x + 14, TS.width - 240) + "px";
      tip.style.top = "8px";
    });
    svg.addEventListener("mouseleave", function () {
      if (crosshair) crosshair.setAttribute("visibility", "hidden");
      tip.hidden = true;
    });
  }

  var select = document.getElementById("metric-select");
  var stdBtn = document.getElementById("view-standard");
  var cumBtn = document.getElementById("view-cumulative");
  if (select) select.addEventListener("change", function () {
    currentMetric = select.value;
    renderTimeseries();
  });
  if (stdBtn) stdBtn.addEventListener("click", function () { currentView = "standard"; renderTimeseries(); });
  if (cumBtn) cumBtn.addEventListener("click", function () { currentView = "cumulative"; renderTimeseries(); });

  attachTooltip();
  renderTimeseries();
  renderLeaderboard();
  renderSimilarity();
  renderDrawdown();
})();
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_dashboard_ui.py -q`
Expected: PASS (14 tests).

- [ ] **Step 7: Full gate + test-count sync + commit**

Run: `.\.venv\Scripts\python -m ruff check .` then `.\.venv\Scripts\python -m pytest -q`

**Plan amendment (controller, 2026-08-18):** rewriting `app.js` (inlined by the current `charts.multimetric_chart_html` / `report._build_html`) breaks 5 Plotly-era tests in `tests/test_dashboard.py` at this task — they assert old-Plotly content markers. All 5 are already on Task 3's prune list, so **prune them in this commit**: `test_html_escapes_user_strings_and_single_plotly_engine`, `test_generate_dashboard_end_to_end_synthetic`, `test_multimetric_chart_html_embeds_payload_and_controls`, `test_build_html_v2_sections_and_four_render_calls`, `test_multimetric_chart_embeds_data_node_and_app_js_once`. Also remove the now-unused `_charts_for_test`/`_kpis_for_test` fixtures and the `from plotly.colors import diverging` / `from dashboard_ui import charts` import lines **only if** no remaining test uses them (the remaining `_kpis_for_test` consumer `test_kpi_cards_stale_champion_pointer_degrades` builds its own frame — verify). Net count: 829 + 3 new − 5 pruned = **827**.

Expected: all green (total = 827).
Sync the three count claims to the collected number, then:

```bash
git add dashboard_ui/static/layout.html dashboard_ui/static/style.css dashboard_ui/static/app.js tests/test_dashboard_ui.py AGENTS.md CONTRIBUTING.md
git commit -m "feat(dashboard_ui): vanilla CSS design system, SVG renderer, HTML layout scaffold"
```

---

### Task 3: compiler rewrite — layout.html wiring, Plotly removal, test split (atomic swap)

**Files:**
- Rewrite: `dashboard_ui/charts.py` — delete the Plotly builders (`build_leaderboard_bar_chart`, `build_drawdown_chart`, `build_similarity_matrix_chart`, `multimetric_chart_html`) and the `import plotly...` / `_STATIC_DIR` / `_read_asset` / `_APP_JS` / color constants; keep only the Task 1 geometry + payload functions.
- Rewrite: `dashboard_ui/report.py` — compiler over `layout.html`; drop `plotly.io` / `get_plotlyjs`; keep all pure helpers.
- Modify: `tests/test_dashboard.py` — prune the Plotly-era presentation tests/fixtures/imports (exact list below).
- Modify: `tests/test_dashboard_ui.py` — add compiler + artifact-contract tests.

**Interfaces:**
- Consumes: `charts.build_dashboard_payload` + the Task 1 geometry functions; `layout.html` / `style.css` / `app.js` from Task 2; `nmr.dashboard` unchanged.
- Produces: `report.generate_dashboard(*, registry_dir=None, benchmark_path=None, output_path=None, open_browser=True) -> Path` (same signature as before — `generate_dashboard.py` wrapper and `test_scripts.py` stay valid); `report._build_html(*, kpis, table_html, diversification_html, accordion_html, payload) -> str`.

- [ ] **Step 1: Write the failing compiler tests** — append to `tests/test_dashboard_ui.py`:

```python
from dashboard_ui import report


def _kpis_for_test() -> dict:
    return {
        "champion_label": "champ · abc12345", "champion_detail": "Active",
        "top_contender_label": "top · def67890", "top_contender_sharpe": 0.9,
        "hurdle_sharpe": 0.78, "gap": 0.12, "fleet_best_cagr": 0.15,
        "worst_drawdown": -0.2, "capital_ready_count": 1, "fleet_count": 3,
        "data_version": "v5.3", "n_eras": 86,
    }


def _payload_for_test() -> dict:
    return {
        "eras": ["0001", "0002"],
        "meta_downside_mask": [False, True],
        "metrics": {
            "payout": {"a" * 64: {"standard": [0.01, -0.02], "label": "r · abc12345"}},
        },
        "leaderboard": [{"label": "r · abc12345", "sharpe": 0.8, "ci_low": 0.6,
                         "ci_high": 1.0, "cagr_1y": 0.5, "max_drawdown": 0.1,
                         "deflated_sharpe": 0.97, "champion": True}],
        "similarity": {"labels": ["r"], "matrix": [[1.0]]},
        "hurdle_sharpe": 0.78,
        "ensemble_sharpe": 1.2,
    }


def _build_html_kwargs() -> dict:
    return dict(
        kpis=_kpis_for_test(),
        table_html="<table><tbody><tr><td>x</td></tr></tbody></table>",
        diversification_html='<p>BADGE MODERATE OVERLAP</p><div id="similarity-host"></div><p>ENSEMBLE CARD 1.200</p>',
        accordion_html="<details><summary>s</summary><pre>j</pre></details>",
        payload=_payload_for_test(),
    )


def test_build_html_sections_and_data_node() -> None:
    html_text = report._build_html(**_build_html_kwargs())
    for section in ("ALPHA GENERATION", "SIGNAL DIVERSIFICATION",
                    "CAPITAL DRAWDOWN", "BADGE MODERATE OVERLAP",
                    "ENSEMBLE CARD"):
        assert section in html_text
    assert html_text.count('id="dashboard-data"') == 1
    assert "<script src" not in html_text
    assert 'id="metric-select"' in html_text
    assert 'id="timeseries-svg"' in html_text
    assert 'id="leaderboard-svg"' in html_text
    assert 'id="drawdown-svg"' in html_text
    assert 'id="similarity-host"' in html_text
    assert html_text.count("--bg: #0d1117") == 1          # style inlined once


def test_build_html_escapes_hostile_strings() -> None:
    payload = _payload_for_test()
    payload["metrics"]["payout"]["x" * 64] = {
        "standard": [0.0], "label": "<script>alert(1)</script>",
    }
    html_text = report._build_html(**{
        **_build_html_kwargs(),
        "kpis": {**_kpis_for_test(), "champion_label": '"><img src=x onerror=alert(2)>'},
        "payload": payload,
    })
    assert '"><img src=x' not in html_text
    assert "&lt;img src=x onerror=alert(2)&gt;" in html_text
    # the hostile label's closing tag must be neutralized inside the data node
    start = html_text.index('id="dashboard-data"')
    end = html_text.index("</script>", start)
    node = html_text[start:end]
    assert "</script" not in node
    assert "<\\/script>" in node


def test_build_html_deterministic_across_calls() -> None:
    a = report._build_html(**_build_html_kwargs())
    b = report._build_html(**_build_html_kwargs())
    assert a == b


def test_build_html_empty_payload_placeholder_message() -> None:
    html_text = report._build_html(**{
        **_build_html_kwargs(),
        "payload": {**_payload_for_test(), "eras": [], "metrics": {}},
    })
    assert "Timeseries data unavailable without local v5.3 assets" in html_text


def test_technical_entries_summary_only(tmp_path: Path) -> None:
    # < 112 KiB budget: the audit accordion must carry config summaries, not
    # full run.json dumps (~25 KB per run; 29 runs = ~715 KB measured)
    _write_registry(tmp_path, [_registry_entry("c" * 64)])
    entries = report._technical_entries(tmp_path)
    assert len(entries) == 1
    assert "backend" in entries[0]["json_text"]
    assert '"scorecard"' not in entries[0]["json_text"]
    assert '"metrics"' not in entries[0]["json_text"]
    assert len(entries[0]["json_text"]) < 2048


def test_build_html_rejects_non_finite_payload() -> None:
    # fail loud: NaN/Inf must never serialize into the data node (browser
    # JSON.parse would throw on them at runtime)
    kwargs = _build_html_kwargs()
    payload = dict(kwargs["payload"])
    payload["metrics"]["payout"]["a" * 64]["standard"] = [0.01, float("nan")]
    with pytest.raises(ValueError):
        report._build_html(kpis=kwargs["kpis"], table_html=kwargs["table_html"],
                           diversification_html=kwargs["diversification_html"],
                           accordion_html=kwargs["accordion_html"], payload=payload)


def test_asset_resolution_independent_of_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert "--bg: #0d1117" in report._read_asset("style.css")
    assert "dataToSvgPath" in report._read_asset("app.js")
    assert "{{ INLINE_DATA_SCRIPT }}" in report._read_asset("layout.html")


def test_table_html_gate_fail_tint() -> None:
    rows = pl.DataFrame(
        [{"model_id": "a" * 64, "source": "trained", "run_name": "sample-run",
          "corr_sharpe_ac": 0.5, "corr_sharpe_ac_ci_low": None,
          "corr_sharpe_ac_ci_high": None, "cagr_1y": 0.1, "max_drawdown": 0.2,
          "gain_to_pain_ratio": 1.0, "mmc_down": 0.02, "deflated_sharpe": 0.5,
          "gate_corr_sharpe_ac": False, "gate_cagr_1y": True,
          "gate_gain_to_pain_ratio": True, "gate_deflated_sharpe": True,
          "status": "RESEARCH"}]
    )
    html_text = report._table_html(rows, champion=None)
    assert 'class="num gate-fail"' in html_text
    assert "badge research" in html_text
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.\.venv\Scripts\python -m pytest tests/test_dashboard_ui.py -q`
Expected: FAIL — `report._build_html` / `report._table_html` do not exist with the new signature (the old `_build_html` needs `figures`/`multimetric_block` positional args, so the kwarg call raises `TypeError`).

- [ ] **Step 3: Rewrite `dashboard_ui/report.py`** (full replacement):

```python
"""Compile the executive HTML performance report from the shared engine.

Thin control plane only: data comes from ``nmr.dashboard``, payload/geometry
from ``dashboard_ui.charts``, raw assets from ``dashboard_ui.static``. No
metric math here. The output is a single self-contained HTML file (vanilla
CSS + JS, no Plotly, no CDN, < 112 KiB) that runs offline from ``file://``.
"""

from __future__ import annotations

import html
import json
import logging
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from dashboard_ui import charts
from nmr.benchmark import load_benchmark_file
from nmr.config import REPO_ROOT
from nmr.dashboard import (
    DEFAULT_DATA_DIR,
    DEFAULT_GATE_PATH,
    DEFAULT_REGISTRY_DIR,
    EVALUABLE_ROWS,
    evaluate_gate_status,
    extract_multimetric_timeseries,
    extract_pairwise_similarity_matrix,
    load_unified_leaderboard,
    read_champion_pointer,
    reconcile_capital_metrics,
)

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _read_asset(name: str) -> str:
    """Read a static asset once (cached at import). Content is static."""
    return (_STATIC_DIR / name).read_text(encoding="utf-8")


_STYLE_CSS = _read_asset("style.css")
_APP_JS = _read_asset("app.js")
_LAYOUT_HTML = _read_asset("layout.html")

_METRIC_CONTROLS_HTML = (
    '<div class="controls">'
    '<select id="metric-select">'
    '<option value="payout">Net Payout Return</option>'
    '<option value="corr20">CORR (20D)</option>'
    '<option value="mmc20">MMC (20D)</option>'
    '<option value="corr60">CORR (60D)</option>'
    '<option value="mmc60">MMC (60D)</option>'
    '<option value="bmc">BMC</option>'
    '<option value="cwmm">CWMM</option>'
    "</select>"
    '<button id="view-standard" class="active">Standard View</button>'
    '<button id="view-cumulative">Cumulative View</button>'
    '<span id="axis-label" class="axis-label"></span>'
    "</div>"
)
_TS_CHART_HTML = (
    '<div class="chart-box"><svg id="timeseries-svg" viewBox="0 0 800 320"></svg>'
    '<div id="timeseries-tooltip" class="tooltip" hidden></div></div>'
)
_LB_CHART_HTML = '<div class="chart-box"><svg id="leaderboard-svg" viewBox="0 0 800 420"></svg></div>'
_DD_CHART_HTML = '<div class="chart-box"><svg id="drawdown-svg" viewBox="0 0 800 240"></svg></div>'
_EMPTY_TS_HTML = (
    '<div class="chart-box"><p>Timeseries data unavailable without local v5.3 assets</p></div>'
)


def _fmt(value, *, pct: bool = False) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:  # NaN
        return "—"
    if number == float("inf"):
        return "∞"
    if pct:
        return f"{number:.2%}"
    return f"{number:.4f}"


def _bar_label(row: dict) -> str:
    model_id = row["model_id"] or "?"
    if row["source"] == "benchmark":
        return f"{row['run_name']} · {model_id}"
    return f"{row['run_name']} · {model_id[:8]}"


def _bar_input(leaderboard: pl.DataFrame, champion: str | None) -> pl.DataFrame:
    evaluable = leaderboard.filter(EVALUABLE_ROWS)
    top = evaluable.sort("corr_sharpe_ac", descending=True, nulls_last=True).head(10)
    return pl.DataFrame(
        [
            {
                "label": _bar_label(row),
                "corr_sharpe_ac": row["corr_sharpe_ac"],
                "corr_sharpe_ac_ci_low": row["corr_sharpe_ac_ci_low"],
                "corr_sharpe_ac_ci_high": row["corr_sharpe_ac_ci_high"],
                "champion": row["model_id"] == champion,
                "cagr_1y": row.get("cagr_1y"),
                "max_drawdown": row.get("max_drawdown"),
                "deflated_sharpe": row.get("deflated_sharpe"),
            }
            for row in top.to_dicts()
        ]
    )


def _kpi_cards(leaderboard: pl.DataFrame, champion: str | None,
               hurdle_sharpe: float) -> dict:
    fleet = leaderboard.filter(pl.col("source").is_in(["trained", "trained_legacy"]))
    top = fleet.sort("corr_sharpe_ac", descending=True, nulls_last=True).head(1)
    top_row = top.row(0, named=True) if top.height else None
    cagr_values = [
        row["cagr_1y"] for row in fleet.to_dicts()
        if row["cagr_1y"] is not None
    ]
    champion_row = None
    if champion is not None:
        champ_frame = leaderboard.filter(pl.col("model_id") == champion)
        if champ_frame.height:
            champion_row = champ_frame.row(0, named=True)
        else:
            logger.warning(
                "dashboard_ui.report: champion %s not found in leaderboard; "
                "treating as none designated", champion,
            )
    return {
        "champion_label": "None Designated" if champion_row is None
                          else _bar_label(champion_row),
        "champion_detail": "(Unallocated)" if champion_row is None else "Active",
        "top_contender_label": _bar_label(top_row) if top_row else "—",
        "top_contender_sharpe": top_row["corr_sharpe_ac"] if top_row else None,
        "hurdle_sharpe": hurdle_sharpe,
        "gap": (top_row["corr_sharpe_ac"] - hurdle_sharpe)
               if top_row and top_row["corr_sharpe_ac"] is not None else None,
        "fleet_best_cagr": max(cagr_values) if cagr_values else None,
        "worst_drawdown": min(
            [row["max_drawdown"] for row in fleet.to_dicts()
             if row["max_drawdown"] is not None],
            default=None,
        ),
        "capital_ready_count": fleet.join(
            leaderboard.select(["model_id", "status"]), on="model_id", how="left"
        ).filter(pl.col("status") == "CAPITAL READY").height,
        "fleet_count": fleet.height,
        "data_version": "v5.3",
        "n_eras": leaderboard.get_column("n_eras").drop_nulls().max()
                  if leaderboard.height else None,
    }


def _table_rows(leaderboard: pl.DataFrame, champion: str | None) -> list[dict]:
    rows = leaderboard.to_dicts()
    champion_rows = [r for r in rows if champion is not None and r["model_id"] == champion]
    full_rows = sorted(
        [r for r in rows if r["source"] == "full"],
        key=lambda r: (str(r["run_name"] or ""), str(r["model_id"])),
    )
    fleet_rows = sorted(
        [r for r in rows
         if r["source"] in ("trained", "trained_legacy") and r["model_id"] != champion],
        key=lambda r: (-(r["corr_sharpe_ac"] if r["corr_sharpe_ac"] is not None
                        else float("-inf")), r["model_id"]),
    )
    bench_rows = sorted(
        [r for r in rows if r["source"] == "benchmark"],
        key=lambda r: ((r["tier"] if r["tier"] is not None else 99), r["model_id"]),
    )
    if full_rows:
        return champion_rows + [{"_group_header": "Promoted Full Versions"}] + full_rows + fleet_rows + bench_rows
    return champion_rows + fleet_rows + bench_rows


_STATUS_BADGE = {
    "CHAMPION": "champion",
    "CAPITAL READY": "ready",
    "RESEARCH": "research",
    "GATE HURDLE": "hurdle",
    "BENCHMARK": "benchmark",
    "FULL": "full",
}


def _status_badge(status: str) -> str:
    cls = _STATUS_BADGE.get(status, "research")
    return f'<span class="badge {cls}">{html.escape(status)}</span>'


def _td_gate(value_str: str, gate_pass: bool | None) -> str:
    if gate_pass is False:
        return f'<td class="num gate-fail">{value_str}</td>'
    return f'<td class="num">{value_str}</td>'


def _row_html(row: dict) -> str:
    if row.get("_group_header"):
        return (
            '<tr class="group-header"><td colspan="9">'
            f"{html.escape(row['_group_header'])}</td></tr>"
        )
    status = _status_badge(row.get("status", "RESEARCH"))
    sharpe = _fmt(row.get("corr_sharpe_ac"))
    ci = "—"
    if row.get("corr_sharpe_ac_ci_low") is not None and row.get("corr_sharpe_ac_ci_high") is not None:
        ci = f"[{_fmt(row['corr_sharpe_ac_ci_low'])}–{_fmt(row['corr_sharpe_ac_ci_high'])}]"
    model_label = html.escape(_bar_label(row))
    if row.get("has_full_version"):
        model_label += ' <span class="badge full">FULL</span>'
    return (
        "<tr>"
        f"<td>{status}</td>"
        f"<td>{model_label}</td>"
        f"{_td_gate(_fmt(row.get('cagr_1y'), pct=True), row.get('gate_cagr_1y'))}"
        f"{_td_gate(sharpe, row.get('gate_corr_sharpe_ac'))}"
        f"<td class=\"num\">{ci}</td>"
        f"<td class=\"num\">{_fmt(row.get('max_drawdown'), pct=True)}</td>"
        f"{_td_gate(_fmt(row.get('gain_to_pain_ratio')), row.get('gate_gain_to_pain_ratio'))}"
        f"<td class=\"num\">{_fmt(row.get('mmc_down'))}</td>"
        f"{_td_gate(_fmt(row.get('deflated_sharpe')), row.get('gate_deflated_sharpe'))}"
        "</tr>"
    )


def _technical_entries(registry_dir: Path) -> list[dict]:
    """Per-run config summaries for the audit accordion (bounded size).

    Full ``run.json`` dumps (~25 KB per run) exceed the 112 KiB artifact budget
    (measured: 29 runs = ~715 KB), so the accordion carries the curated config
    summary only; the immutable full payload lives in the registry.
    """
    entries = []
    for run_file in sorted(registry_dir.glob("*/run.json")):
        try:
            payload = json.loads(run_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        manifest = payload.get("manifest") or {}
        cfg = manifest.get("config") or {}
        run_cfg = cfg.get("run") or {}
        summary = {
            "backend": (cfg.get("model") or {}).get("backend"),
            "preset": (cfg.get("model") or {}).get("preset"),
            "feature_set": (cfg.get("data") or {}).get("feature_set"),
            "feature_subset": (cfg.get("data") or {}).get("feature_subset"),
            "neutralization_proportion": (cfg.get("risk") or {}).get(
                "neutralization_proportion"
            ),
            "seed": run_cfg.get("seed"),
            "device": manifest.get("oof_device"),
            "targets": (cfg.get("data") or {}).get("targets"),
        }
        entries.append(
            {
                "label": f"{run_cfg.get('name', 'unknown')} · "
                         f"{str(payload.get('run_id') or run_file.parent.name)[:8]}",
                "summary": summary,
                "json_text": json.dumps(summary, indent=2, sort_keys=True),
            }
        )
    return entries


def _diversification_stats(matrix: list[list[float]]) -> dict:
    """Max/mean off-diagonal overlap + badge tier (decision #16)."""
    n = len(matrix)
    off = [matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    mean = float(np.mean(off)) if off else None
    maximum = float(max(off)) if off else None
    if mean is None:
        badge = "—"
    elif mean < 0.65:
        badge = "EXCELLENT DIVERSIFICATION"
    elif mean <= 0.85:
        badge = "MODERATE OVERLAP"
    else:
        badge = "HIGH REDUNDANCY"
    return {"mean_overlap": mean, "max_overlap": maximum, "badge": badge}


def _ensemble_sharpe(payout_metric: dict) -> float | None:
    """Equal-weighted blended Sharpe from per-era payout series (decision #17).

    SR_blended = mean(mu) / sqrt(w^T Sigma w), w uniform; None when fewer
    than 3 usable series (decision #27) or zero variance.
    """
    series = [
        np.asarray(v["standard"], dtype=float)
        for v in payout_metric.values()
        if v.get("standard")
    ]
    if len(series) < 3:
        return None
    stacked = np.vstack(series)
    mu = np.mean(stacked, axis=1)
    weights = np.full(len(mu), 1.0 / len(mu))
    variance = float(weights @ np.cov(stacked) @ weights)
    if variance <= 0.0 or not np.isfinite(variance):
        return None
    return float(np.mean(mu) / np.sqrt(variance))


def _badge_html(stats: dict, stress: dict) -> str:
    delta = stress.get("mean_delta")
    delta_text = "—" if delta is None else f"{delta:+.3f}"
    mean_text = "—" if stats["mean_overlap"] is None else f"{stats['mean_overlap']:.3f}"
    max_text = "—" if stats["max_overlap"] is None else f"{stats['max_overlap']:.3f}"
    return (
        f'<p class="badge-line"><b>{html.escape(stats["badge"])}</b> · '
        f"Mean Overlap {mean_text} · Max Overlap {max_text} · "
        f"Stress-Regime Δρ {delta_text}</p>"
    )


def _ensemble_card_html(value: float | None) -> str:
    text = "—" if value is None else f"{value:.3f}"
    return (
        '<div class="kpi"><div class="label">Equal-Weight Ensemble Sharpe '
        f"(top-3, heuristic)</div><div class=\"value\">{text}</div></div>"
    )


def _kpi_cards_html(kpis: dict) -> str:
    return (
        '<div class="kpi"><div class="label">Active Champion</div>'
        f'<div class="value">{html.escape(kpis["champion_label"])}</div>'
        f'<div>{html.escape(kpis["champion_detail"])}</div></div>'
        '<div class="kpi"><div class="label">Top Research Contender</div>'
        f'<div class="value">{html.escape(kpis["top_contender_label"])}</div>'
        f'<div>Sharpe {_fmt(kpis["top_contender_sharpe"])} vs hurdle {_fmt(kpis["hurdle_sharpe"])}</div></div>'
        '<div class="kpi"><div class="label">Fleet Best Return (CAGR)</div>'
        f'<div class="value">{_fmt(kpis["fleet_best_cagr"], pct=True)}</div></div>'
        '<div class="kpi"><div class="label">Worst Fleet Drawdown</div>'
        f'<div class="value">{_fmt(kpis["worst_drawdown"], pct=True)}</div></div>'
        '<div class="kpi"><div class="label">Capital Readiness</div>'
        f'<div class="value">{kpis["capital_ready_count"]} / {kpis["fleet_count"]}</div></div>'
    )


def _table_html(leaderboard: pl.DataFrame, champion: str | None) -> str:
    rows_html = "".join(_row_html(row) for row in _table_rows(leaderboard, champion))
    return (
        "<table><thead><tr><th>Status</th><th>Model</th><th>Ann. Return</th>"
        "<th>Sharpe (AC)</th><th>Sharpe CI</th><th>Max DD</th><th>Gain-to-Pain</th>"
        "<th>Downside</th><th>Confidence (DSR)</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


def _accordion_html(technical_entries: list[dict]) -> str:
    accordion = ""
    for entry in technical_entries:
        accordion += (
            "<details><summary>"
            f"{html.escape(entry['label'])} — technical &amp; audit</summary>"
            f"<pre>{html.escape(entry['json_text'])}</pre></details>"
        )
    return accordion


def _diversification_html(badge_html: str, ensemble_card_html: str) -> str:
    return (
        badge_html
        + '<div id="similarity-host" class="chart-box"></div>'
        + ensemble_card_html
    )


def _build_html(
    *,
    kpis: dict,
    table_html: str,
    diversification_html: str,
    accordion_html: str,
    payload: dict[str, Any],
) -> str:
    """Assemble the full HTML document from the layout template + payload.

    Deterministic: fixed template + sorted-key JSON + static assets. The
    data-node substitution runs LAST so payload text can never be re-processed
    by a later placeholder replacement.
    """
    payload_json = json.dumps(
        payload, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).replace("</", "<\\/")
    ts_html = _TS_CHART_HTML if payload.get("eras") else _EMPTY_TS_HTML
    replacements = [
        ("{{ INLINE_STYLE }}", _STYLE_CSS),
        ("{{ N_ERAS }}", str(kpis["n_eras"]) if kpis["n_eras"] is not None else "—"),
        ("{{ DATA_VERSION }}", html.escape(kpis["data_version"])),
        ("{{ KPI_CARDS }}", _kpi_cards_html(kpis)),
        ("{{ METRIC_CONTROLS }}", _METRIC_CONTROLS_HTML),
        ("{{ TIMESERIES_SVG }}", ts_html),
        ("{{ LEADERBOARD_SVG }}", _LB_CHART_HTML),
        ("{{ DIVERSIFICATION_SECTION }}", diversification_html),
        ("{{ DECISION_TABLE }}", table_html),
        ("{{ DRAWDOWN_SVG }}", _DD_CHART_HTML),
        ("{{ AUDIT_ACCORDION }}", accordion_html),
        ("{{ INLINE_DATA_SCRIPT }}",
         '<script type="application/json" id="dashboard-data">'
         f"{payload_json}</script>\n<script>\n{_APP_JS}</script>"),
    ]
    html_text = _LAYOUT_HTML
    for key, value in replacements:
        html_text = html_text.replace(key, value)
    return html_text


def generate_dashboard(
    *,
    registry_dir: Path | None = None,
    benchmark_path: Path | None | bool = None,
    output_path: Path | None = None,
    open_browser: bool = True,
) -> Path:
    """Build the executive HTML report and write it to disk."""
    registry_dir = Path(registry_dir) if registry_dir is not None else DEFAULT_REGISTRY_DIR
    output_path = Path(output_path) if output_path is not None else REPO_ROOT / "artifacts" / "dashboard.html"

    leaderboard = load_unified_leaderboard(registry_dir, benchmark_path=benchmark_path)
    leaderboard = reconcile_capital_metrics(leaderboard, DEFAULT_DATA_DIR)
    statuses = evaluate_gate_status(leaderboard, DEFAULT_GATE_PATH, registry_dir / "champion.json")
    leaderboard = leaderboard.join(statuses, on="model_id", how="left")

    gate_cfg = load_benchmark_file(DEFAULT_GATE_PATH)
    assert gate_cfg.reference_column is not None
    tier4_column = str(gate_cfg.reference_column)
    hurdle_sharpe = float(gate_cfg.gate.corr_sharpe_ac_min)

    champion = read_champion_pointer(registry_dir / "champion.json")
    fleet = leaderboard.filter(pl.col("source").is_in(["trained", "trained_legacy"]))
    top3_ids = fleet.sort("corr_sharpe_ac", descending=True, nulls_last=True) \
        .head(3).get_column("model_id").to_list()
    engine_payload = extract_multimetric_timeseries(
        registry_dir, DEFAULT_DATA_DIR, run_ids=top3_ids,
        include_tier4_ref=True, tier4_column=tier4_column,
    )
    top5_ids = fleet.sort("corr_sharpe_ac", descending=True, nulls_last=True) \
        .head(5).get_column("model_id").to_list()
    labels, _sim_ids, matrix, stress = extract_pairwise_similarity_matrix(
        registry_dir, DEFAULT_DATA_DIR, run_ids=top5_ids,
        include_tier4_ref=True, tier4_column=tier4_column,
    )
    stats = _diversification_stats(matrix)
    payout_metric = (engine_payload.get("metrics") or {}).get("payout") or {}
    top3_payout = {mid: payout_metric[mid] for mid in top3_ids if mid in payout_metric}
    ensemble_value = _ensemble_sharpe(top3_payout)

    payload = charts.build_dashboard_payload(
        eras=engine_payload.get("eras") or [],
        meta_downside_mask=engine_payload.get("meta_downside_mask") or [],
        metrics=engine_payload.get("metrics") or {},
        leaderboard_bars=_bar_input(leaderboard, champion),
        similarity_labels=labels,
        similarity_matrix=matrix,
        hurdle_sharpe=hurdle_sharpe,
        ensemble_sharpe=ensemble_value,
    )

    html_text = _build_html(
        kpis=_kpi_cards(leaderboard, champion, hurdle_sharpe),
        table_html=_table_html(leaderboard, champion),
        diversification_html=_diversification_html(
            _badge_html(stats, stress), _ensemble_card_html(ensemble_value)
        ),
        accordion_html=_accordion_html(_technical_entries(registry_dir)),
        payload=payload,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    if open_browser:
        webbrowser.open(output_path.as_uri())
    return output_path


def main() -> int:
    output = generate_dashboard()
    print(f"Dashboard written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Rewrite `dashboard_ui/charts.py`** (full replacement — deletes the Plotly builders and keeps the Task 1 geometry/payload functions):

```python
"""Pure geometry + payload builders for the vanilla executive dashboard.

Presentation math only: SVG coordinate scaling, series transforms, and the
JSON data contract for ``static/app.js``. No metric formulas, no file I/O, no
registry access; ``nmr.dashboard`` stays the analytical engine. All geometry
here is mirrored client-side by ``static/app.js`` and covered by
``tests/test_dashboard_ui.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl

__all__ = [
    "build_dashboard_payload",
    "cumulative_series",
    "data_to_svg_path",
    "drawdown_series",
    "global_y_range",
    "svg_area_path",
]

_ZERO_SPAN_EPS = 1e-12
_PAYLOAD_ROUND = 6


def _round6(value: Any) -> Any:
    """Round payload floats to 6 decimals (display precision is 4) — keeps the
    data node honest while fitting the 112 KiB artifact budget (amendment)."""
    if isinstance(value, (int, float, np.floating)):
        return round(float(value), _PAYLOAD_ROUND)
    return value


def global_y_range(*series: Sequence[float]) -> tuple[float, float]:
    """Global min/max across all series (shared axis); (0.0, 1.0) when empty."""
    values = [v for s in series for v in s]
    if not values:
        return (0.0, 1.0)
    return (float(min(values)), float(max(values)))


def _resolve_range(
    values: Sequence[float], y_min: float | None, y_max: float | None
) -> tuple[float, float]:
    """Resolve the y range, expanding a degenerate flat span so scaling never divides by zero."""
    lo, hi = global_y_range(values) if y_min is None or y_max is None else (y_min, y_max)
    if abs(hi - lo) < _ZERO_SPAN_EPS:
        lo -= 1.0
        hi += 1.0
    return lo, hi


def data_to_svg_path(
    values: Sequence[float],
    *,
    width: float,
    height: float,
    y_min: float | None = None,
    y_max: float | None = None,
    pad: tuple[float, float, float, float] = (20.0, 20.0, 20.0, 40.0),
) -> str:
    """Map a series to an SVG polyline path (y axis inverted, top-left origin).

    ``pad`` order is (top, right, bottom, left). Empty input returns ``""``.
    """
    if not values:
        return ""
    lo, hi = _resolve_range(values, y_min, y_max)
    span = hi - lo
    pad_top, pad_right, pad_bottom, pad_left = pad
    inner_w = width - pad_left - pad_right
    inner_h = height - pad_top - pad_bottom
    denom = max(1, len(values) - 1)
    points = []
    for i, v in enumerate(values):
        x = pad_left + (i / denom) * inner_w
        y = pad_top + (1.0 - (v - lo) / span) * inner_h
        points.append(f"{x:.1f},{y:.1f}")
    return "M " + " L ".join(points)


def svg_area_path(
    values: Sequence[float],
    *,
    width: float,
    height: float,
    y_min: float | None = None,
    y_max: float | None = None,
    y_baseline: float = 0.0,
    pad: tuple[float, float, float, float] = (20.0, 20.0, 20.0, 40.0),
) -> str:
    """Closed SVG polygon: line path + baseline anchors (``L xN,yBase L x0,yBase Z``)."""
    if not values:
        return ""
    lo, hi = _resolve_range(values, y_min, y_max)
    span = hi - lo
    line = data_to_svg_path(values, width=width, height=height, y_min=lo, y_max=hi, pad=pad)
    pad_top, pad_right, pad_bottom, pad_left = pad
    inner_h = height - pad_top - pad_bottom
    inner_w = width - pad_left - pad_right
    y_base = pad_top + (1.0 - (y_baseline - lo) / span) * inner_h
    denom = max(1, len(values) - 1)
    x0 = pad_left
    x_n = pad_left + ((len(values) - 1) / denom) * inner_w
    return f"{line} L {x_n:.1f},{y_base:.1f} L {x0:.1f},{y_base:.1f} Z"


def cumulative_series(standard: Sequence[float], *, payout: bool) -> list[float]:
    """cumprod(1+r) for payout, cumsum(rho) for correlations (spec decision #9)."""
    values = np.asarray(standard, dtype=float)
    if payout:
        return [float(v) for v in np.cumprod(1.0 + values)]
    return [float(v) for v in np.cumsum(values)]


def drawdown_series(cumulative: Sequence[float]) -> list[float]:
    """wealth/peak - 1 (peak = running maximum)."""
    wealth = np.asarray(cumulative, dtype=float)
    peak = np.maximum.accumulate(wealth)
    return [float(v) for v in wealth / peak - 1.0]


def build_dashboard_payload(
    *,
    eras: Sequence[str],
    meta_downside_mask: Sequence[bool],
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    leaderboard_bars: pl.DataFrame,
    similarity_labels: Sequence[str],
    similarity_matrix: Sequence[Sequence[float]],
    hurdle_sharpe: float,
    ensemble_sharpe: float | None,
) -> dict[str, Any]:
    """Shape engine output into the standard-only, metric-first vanilla contract.

    ``metrics`` mirrors ``nmr.dashboard.extract_multimetric_timeseries``
    (metric-first); cumulative/drawdown are derived client-side, so only the
    ``standard`` arrays and labels are carried. ``leaderboard_bars`` must be a
    frame with columns ``label, corr_sharpe_ac, corr_sharpe_ac_ci_low,
    corr_sharpe_ac_ci_high, cagr_1y, max_drawdown, deflated_sharpe, champion``.
    """
    shaped_metrics: dict[str, Any] = {}
    for metric, models in metrics.items():
        shaped_metrics[metric] = {
            model_id: {"standard": [_round6(v) for v in series["standard"]],
                       "label": series["label"]}
            for model_id, series in models.items()
        }
    rows = [
        {
            "label": row["label"],
            "sharpe": _round6(row["corr_sharpe_ac"]),
            "ci_low": _round6(row["corr_sharpe_ac_ci_low"]),
            "ci_high": _round6(row["corr_sharpe_ac_ci_high"]),
            "cagr_1y": _round6(row.get("cagr_1y")),
            "max_drawdown": _round6(row.get("max_drawdown")),
            "deflated_sharpe": _round6(row.get("deflated_sharpe")),
            "champion": row["champion"],
        }
        for row in leaderboard_bars.to_dicts()
    ]
    return {
        "eras": list(eras),
        "meta_downside_mask": [bool(m) for m in meta_downside_mask],
        "metrics": shaped_metrics,
        "leaderboard": rows,
        "similarity": {
            "labels": list(similarity_labels),
            "matrix": [list(r) for r in similarity_matrix],
        },
        "hurdle_sharpe": float(hurdle_sharpe),
        "ensemble_sharpe": ensemble_sharpe,
    }
```

- [ ] **Step 5: Prune the Plotly-era tests from `tests/test_dashboard.py`** — delete exactly:

1. Import line `from plotly.colors import diverging` (line 10).
2. Import line `from dashboard_ui import charts` (line 15) — no remaining engine test uses it.
3. The fixture block `_bar_input` (601–612), `_ts_payload` (614–621), `_multimetric_payload` (623–634).
4. Tests: `test_leaderboard_chart_traces_and_hurdle_line` (637), `test_leaderboard_chart_hover_fields` (647), `test_chart_hovertemplate_escapes_labels` (655), `test_drawdown_chart_underwater_fill` (663), `test_timeseries_charts_empty_payload_render_annotation` (670), `test_leaderboard_chart_empty_frame_render_annotation` (678).
5. The fixture block `_charts_for_test` (690–696), `_kpis_for_test` (698–707).
6. `test_html_escapes_user_strings_and_single_plotly_engine` (710–740) — **already deleted in Task 2** (plan amendment); skip if absent.
7. `test_generate_dashboard_end_to_end_synthetic` (758–779) — **already deleted in Task 2** (plan amendment); skip if absent. Replaced by the new e2e test in Task 3 Step 6.
8. `test_build_html_deterministic_across_calls` (782–802).
9. `test_multimetric_chart_html_embeds_payload_and_controls` (920) — **already deleted in Task 2**; skip if absent. `test_multimetric_chart_html_empty_payload_annotation` (947), `test_similarity_chart_heatmap_and_highlight` (955), `test_similarity_chart_empty_matrix_annotation` (971), `test_drawdown_chart_v2_payload` (977), `test_build_html_v2_sections_and_four_render_calls` (1020) — **already deleted in Task 2**; skip if absent.
10. `test_multimetric_chart_embeds_data_node_and_app_js_once` (1277) — **already deleted in Task 2**; skip if absent. `test_report_inlines_style_css_once` (1293).

**Keep** `test_kpi_cards_stale_champion_pointer_degrades` (743) — it builds its own local frame and tests `report._kpi_cards`, which stays.

After pruning, run `.\.venv\Scripts\python -m pytest tests/test_dashboard.py -q`. If a `NameError` surfaces from a surviving test using a pruned fixture, re-instate that one fixture locally (do not re-add whole blocks).

- [ ] **Step 6: Add the end-to-end + artifact-contract tests** — append to `tests/test_dashboard_ui.py` (reuse the registry fixtures from `tests/test_dashboard.py` by importing them, or duplicate the two small helpers `_registry_entry` / `_write_registry` from that file — importing test modules across files is brittle under collection, so **duplicate** them here):

```python
def _registry_entry(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "metrics": {"mean": 0.1, "std": 0.2, "sharpe": 0.5, "max_drawdown": 0.05},
        "manifest": {
            "oof_device": "cpu",
            "config": {
                "data": {"feature_set": "small", "feature_subset": None,
                         "targets": ["target"]},
                "model": {"backend": "lightgbm", "preset": "fast"},
                "risk": {"neutralization_proportion": 1.0},
                "run": {"name": "sample-run"},
            },
        },
        "scorecard": {
            "corr": 0.12, "corr_ci_low": 0.05, "corr_ci_high": 0.19, "corr_n_eras": 30,
            "corr_sharpe_ac": 0.8, "corr_sharpe_ac_ci_low": 0.6,
            "corr_sharpe_ac_ci_high": 1.0, "max_drawdown": 0.1, "std_corr": 0.2,
            "deflated_sharpe": 0.97, "max_feature_exposure": 0.3, "bmc": 0.02,
            "fnc": 0.05, "n_eras": 30, "cagr_1y": 1.5, "gain_to_pain_ratio": 2.0,
            "kelly_fraction": 0.4, "mmc_down": 0.01, "mmc_down_reason": None,
        },
    }


def _write_registry(tmp_path: Path, entries: list[dict]) -> None:
    for entry in entries:
        run_dir = tmp_path / entry["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps(entry), encoding="utf-8")


def test_generate_dashboard_end_to_end_synthetic(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64)])
    out = report.generate_dashboard(
        registry_dir=tmp_path, benchmark_path=False,
        output_path=tmp_path / "dashboard.html", open_browser=False,
    )
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "sample-run" in text
    # the synthetic fixture genuinely clears the real tier-4 gate -> CAPITAL READY badge
    assert "CAPITAL READY" in text
    for section in ("ALPHA GENERATION", "SIGNAL DIVERSIFICATION", "CAPITAL DRAWDOWN"):
        assert section in text
    assert 'id="dashboard-data"' in text
    assert "plotly" not in text.lower()


def test_generate_dashboard_artifact_contract(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("b" * 64)])
    out = report.generate_dashboard(
        registry_dir=tmp_path, benchmark_path=False,
        output_path=tmp_path / "dashboard.html", open_browser=False,
    )
    size_kb = out.stat().st_size / 1024
    assert size_kb < 100, f"bundle too large: {size_kb:.2f} KB"
    text = out.read_text(encoding="utf-8")
    assert "plotly" not in text.lower()
    assert "<script src=" not in text
    assert 'id="dashboard-data"' in text


def test_generate_dashboard_empty_registry_compiles(tmp_path: Path) -> None:
    out = report.generate_dashboard(
        registry_dir=tmp_path, benchmark_path=False,
        output_path=tmp_path / "dashboard.html", open_browser=False,
    )
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "ALPHA GENERATION" in text
    assert 'id="dashboard-data"' in text
```

Add `import json` at the top of `tests/test_dashboard_ui.py` if not already present.

- [ ] **Step 7: Run the new compiler tests to verify they pass**

Run: `.\.venv\Scripts\python -m pytest tests/test_dashboard_ui.py tests/test_dashboard.py -q`
Expected: PASS — the new compiler tests green; the pruned engine file green.

- [ ] **Step 8: Full gate + count sync + commit**

Run: `.\.venv\Scripts\python -m ruff check .` then `.\.venv\Scripts\python -m pytest -q`
Expected: all green (the atomic swap keeps the suite green at the commit boundary; net count changes as tests were pruned/added — sync the three doc claims to the collected number):

```bash
git add dashboard_ui/charts.py dashboard_ui/report.py tests/test_dashboard.py tests/test_dashboard_ui.py AGENTS.md CONTRIBUTING.md
git commit -m "refactor(dashboard_ui): vanilla HTML/CSS/SVG compiler, drop Plotly figure layer"
```

---

### Task 4: Streamlit rewrite — native charts, no Plotly

**Files:**
- Modify: `dashboard_ui/app.py`
- Modify: `tests/test_scripts.py` (add the no-plotly guard)

**Interfaces:**
- Consumes: existing pure helpers untouched (`load_registry_frame`, `load_benchmarks`, `merge_leaderboard`, `load_campaigns`, `robustness_matrix`, `champion_run_id`, `_shaped_leaderboard_pdf`, `_read_run_payload`, `_read_full_manifest`, `_load_registry_entries`, `_bar_label`).
- Produces: `render_leaderboard`, `render_fleet`, `render_robustness_matrix` with the same names/signatures, now Plotly-free.

- [ ] **Step 1: Write the failing guard test** — append to `tests/test_scripts.py`:

```python
def test_dashboard_app_has_no_plotly_reference() -> None:
    import inspect
    from dashboard_ui import app as dashboard_app
    assert "plotly" not in inspect.getsource(dashboard_app).lower()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.\.venv\Scripts\python -m pytest tests/test_scripts.py::test_dashboard_app_has_no_plotly_reference -q`
Expected: FAIL — `dashboard_ui/app.py` still imports `plotly.express`.

- [ ] **Step 3: Rewrite the three render functions in `dashboard_ui/app.py`**

Remove the import line `import plotly.express as px` (line 30). Replace the bodies of the three render functions:

```python
def render_leaderboard(leaderboard: pl.DataFrame, champion: str | None) -> None:
    """Bar chart of ``corr_sharpe_ac`` + sortable dataframe (native st.* charts).

    The chart shows evaluable rows only (EVALUABLE_ROWS — full-version rows
    carry no validation metrics); CI bounds render as columns in the table
    (native charts have no error bars). The dataframe keeps all rows; full
    rows are pinned first by ``_shaped_leaderboard_pdf``.
    """
    if leaderboard.height == 0:
        st.info("No runs to display — train one with `train_first_model.py`.")
        return
    evaluable = leaderboard.filter(EVALUABLE_ROWS)
    if evaluable.height == 0:
        st.info("No evaluable runs to display (all rows are full versions).")
        return
    pdf = _shaped_leaderboard_pdf(evaluable, champion)
    st.bar_chart(pdf.set_index("label")[[_BAR_METRIC]], horizontal=True, height=400)
    st.caption("Bars: CORR Sharpe (auto-correlated). CI bounds in the table below.")
    table_pdf = _shaped_leaderboard_pdf(leaderboard, champion)
    st.dataframe(
        table_pdf.drop(columns=["champion", "ci_plus", "ci_minus", "label"]),
        column_config={
            "has_full_version": st.column_config.CheckboxColumn(
                label="Full",
                help="Has a promoted full (train+validation) version",
            ),
        },
    )
```

```python
def render_fleet(
    registry_entries: Sequence[dict],
    *,
    n_trials: int,
    dsr_confidence: float,
) -> None:
    """Fleet table via ``nmr.meta.fleet_summary`` + native scatter."""
    if not registry_entries:
        st.info("No registry entries to analyze.")
        return
    summary = fleet_summary(
        registry_entries, n_trials=n_trials, dsr_confidence=dsr_confidence
    )
    st.dataframe(summary)
    chart_df = summary.to_pandas()
    st.scatter_chart(
        chart_df,
        x="neutralization_proportion",
        y="metric",
        color="preset",
    )
```

```python
def render_robustness_matrix(registry: pl.DataFrame) -> None:
    """Heatmap-styled dataframe over ``robustness_matrix`` (booleans as 0/1)."""
    matrix = robustness_matrix(registry)
    if matrix.height == 0:
        st.info("No evaluable runs in the registry.")
        return
    numeric = matrix.with_columns(pl.col(flag).cast(pl.Int8) for flag in _ROBUSTNESS_CELLS)
    pdf = numeric.to_pandas().set_index("model_id").astype(float)
    styled = pdf.style.background_gradient(cmap="RdYlGn", axis=None).format(na_rep="—")
    st.dataframe(styled, use_container_width=True)
    st.caption(
        "Boolean cells (has_*) shown as 0/1; numeric cells "
        "(max_feature_exposure, std_corr, max_drawdown) shown raw; "
        "missing values render as —."
    )
    st.dataframe(matrix)
```

Update the module docstring line that says "streamlit/plotly import headless" to "streamlit imports headless" (no Plotly). Leave every other function untouched.

- [ ] **Step 4: Run the guard test to verify it passes**

Run: `.\.venv\Scripts\python -m pytest tests/test_scripts.py -q`
Expected: PASS — the new guard plus the existing pure-helper tests.

- [ ] **Step 5: Full gate + count sync + commit**

Run: `.\.venv\Scripts\python -m ruff check .` then `.\.venv\Scripts\python -m pytest -q`
Expected: all green (count +1 from the guard test; sync the three doc claims):

```bash
git add dashboard_ui/app.py tests/test_scripts.py AGENTS.md CONTRIBUTING.md
git commit -m "refactor(dashboard_ui): native Streamlit charts, drop plotly.express"
```

---

### Task 5: docs, requirements, audit, artifact regeneration

**Files:**
- Modify: `requirements.txt` (remove `plotly==6.6.0`)
- Modify: `AGENTS.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `README.md`

**Interfaces:** none (documentation + hygiene only).

- [ ] **Step 1: Remove the Plotly pin from `requirements.txt`** — delete the line `plotly==6.6.0` (line 20). No other line changes.

- [ ] **Step 2: Update `AGENTS.md`**

1. Dependency-exception line (the "Streamlit + Plotly (interactive dashboard — imported only in the `dashboard_ui/` package ...)" clause): replace with:

```
Streamlit (interactive dashboard — imported only in `dashboard_ui/app.py` and its thin entry wrapper `dashboard_app.py`; never in `nmr/`; the static executive report is a zero-dependency vanilla HTML/CSS/SVG compiler in `dashboard_ui/` (`report.py` + `static/`), no charting library)
```

2. Executive-dashboard toolkit row (line 143): replace the spec pointer `docs/superpowers/specs/2026-08-16-executive-dashboard-design.md` with `docs/superpowers/specs/2026-08-18-vanilla-dashboard-design.md`.
3. The two test-count claims in the active governance documents are synchronized to the collected count (currently 1,099); this historical plan no longer owns those values.

- [ ] **Step 3: Update `ARCHITECTURE.md`**

1. Line 42 (pipeline map): change `(executive report — offline single-file HTML)` to `(executive report — offline single-file vanilla HTML/CSS/SVG, < 112 KiB)`.
2. Module table rows 287–291: `generate_dashboard.py` note gains `vanilla HTML/CSS/SVG, < 112 KiB`; `dashboard_app.py` gains `native-Streamlit (no Plotly)`; `dashboard_ui/charts.py` row becomes `pure SVG geometry reference + metric-first payload builder (data_to_svg_path, svg_area_path, cumulative_series, drawdown_series, build_dashboard_payload); tested in tests/test_dashboard_ui.py; static/app.js mirrors the geometry client-side`; `dashboard_ui/report.py` row becomes `HTML report compiler — generate_dashboard / main; inlines static/{style.css, app.js, layout.html}; consumes nmr.dashboard + dashboard_ui.charts`.
3. §W (lines 395–397): replace the tail sentence "`dashboard_ui/charts.py` adds the JS-controller multimetric chart (no `updatemenus`) and the similarity heatmap; `dashboard_ui/report.py` compiles the HTML (inlining `static/style.css`) and `dashboard_ui/app.py` renders the Streamlit views." with: "`dashboard_ui/charts.py` provides the tested SVG geometry reference + metric-first payload builder; `static/app.js` mirrors the geometry client-side and renders all charts (timeseries + stress spans, Sharpe leaderboard + CI whiskers + hurdle, similarity matrix, underwater drawdown); `dashboard_ui/report.py` compiles the single-file HTML (< 112 KiB, inlining `static/{style.css, app.js, layout.html}`); `dashboard_ui/app.py` renders the Streamlit views natively (no Plotly). Presentation tests live in `tests/test_dashboard_ui.py`."

- [ ] **Step 4: Update `CONTRIBUTING.md` and `README.md`**

1. `CONTRIBUTING.md:30`: `green 818-test` → the collected count (same number as AGENTS.md).
2. `README.md:5` stack line: replace `Streamlit / Plotly` with `Streamlit (interactive dashboard; static report is vanilla HTML/CSS/SVG)`.

- [ ] **Step 5: Plotly removal audit**

Run:

```bash
grep -rn "import plotly\|from plotly\|plotly\." dashboard_ui/ tests/ configs/ requirements.txt *.py
```

Expected: **no output** — zero Plotly imports, attribute usage, or pin references. (Amendment: a bare `grep -ri plotly` cannot be silent — the plan's own Task 3 docstring in `report.py` says "no Plotly, no CDN", and negative-assertion guards in `tests/test_scripts.py` / `tests/test_dashboard_ui.py` intentionally contain the word. The audit therefore targets *usage*; the word "Plotly" is allowed only in: historical `docs/superpowers/specs/2026-08-16-*.md`, docstrings describing the removal, and negative-assertion test guards.)

- [ ] **Step 6: Regenerate the artifact and verify the contract**

Run:

```powershell
.\.venv\Scripts\python generate_dashboard.py
.\.venv\Scripts\python -c "
from pathlib import Path
p = Path('artifacts/dashboard.html')
assert p.exists()
size_kb = p.stat().st_size / 1024
assert size_kb < 100, f'Bundle size too large: {size_kb:.2f} KB'
text = p.read_text(encoding='utf-8')
assert 'plotly' not in text.lower()
assert '<script src=' not in text
assert 'id=\"dashboard-data\"' in text
print(f'SUCCESS: {size_kb:.2f} KB, pure HTML/CSS/JS')
"
```

Expected: `SUCCESS: < ~112 KiB, pure HTML/CSS/JS`. The artifact is gitignored — do not commit it. Open it in a browser and sanity-check the five sections render (charts + table + accordion).

- [ ] **Step 7: Full gate + commit**

Run: `.\.venv\Scripts\python -m ruff check .` then `.\.venv\Scripts\python -m pytest -q`
Expected: all green with the final count (doc claims already synced in Task 4; re-verify `tests/test_docs_hygiene.py` passes within the full run):

```bash
git add requirements.txt AGENTS.md ARCHITECTURE.md CONTRIBUTING.md README.md
git commit -m "chore(dashboard): drop plotly pin, sync SSOT docs to vanilla dashboard"
```

---

## Final Verification (end of Task 5)

```powershell
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python generate_dashboard.py
.\.venv\Scripts\python -c "
from pathlib import Path
p = Path('artifacts/dashboard.html')
size_kb = p.stat().st_size / 1024
text = p.read_text(encoding='utf-8')
assert size_kb < 100 and 'plotly' not in text.lower() and '<script src=' not in text
assert 'id=\"dashboard-data\"' in text
print(f'OK: {size_kb:.2f} KB')
"
```

Then run the whole-branch review (fresh reviewer, `superpowers:requesting-code-review` template), one final fix wave if needed, and `superpowers:finishing-a-development-branch` to merge.
