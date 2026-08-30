# Dashboard Architecture History, August 2026

> **Status:** Historical provenance. For the current implementation, use
> [`ARCHITECTURE.md`](../../ARCHITECTURE.md), section W.

The dashboard moved through three short-lived implementations before settling
on the current offline Model Tournament:

1. A Streamlit/Plotly executive dashboard presented portfolio and scenario
   views.
2. A shared component experiment attempted parallel Streamlit and HTML
   renderers. It was retired because it duplicated presentation behavior.
3. The current design retained `nmr/dashboard.py` as the deterministic analytics
   boundary and made `dashboard_ui/report.py` the single vanilla HTML/CSS/SVG
   renderer. `dashboard_ui/app.py` embeds that same document in Streamlit.

Durable decisions from the delivery are part of the active architecture:

- the dashboard is read-only and reports offline evaluation evidence;
- ranking direction, cohort semantics, missing-value handling, and champion
  status are computed by the tested Python engine;
- the wire payload excludes absolute paths, wall-clock generation values, and
  timing fields;
- generated minified assets derive from readable sources in
  `dashboard_ui/static/`;
- registry, experiment, deployment, and cache state are never mutated by the
  dashboard.

The former root delivery report, refactor plan, session report, and delivery
summary were removed after these durable decisions were integrated into the
architecture and tests. Obsolete localhost claims, copied APIs, test counts,
file inventories, and transient verification failures were intentionally not
retained.