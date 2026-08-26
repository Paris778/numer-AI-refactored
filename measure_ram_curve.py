"""Thin control plane: three-point full-history COMMIT curve (measured, not estimated).

Gates on **commit charge** (``PeakPagefileUsage``) — the quantity that
produced the documented full-universe thrash — never working set. Each point
runs the spawned full-history worker fit in a FRESH subprocess so the
parent-side peaks are per-point (not lifetime-across-points), and both child
and parent working-set AND commit are recorded.

Model form ``peak = a + b*rows`` is structural, not fitted by convention:

| Component                          | Scales with rows? |
|------------------------------------|-------------------|
| float32 input matrix               | linear            |
| binned uint8 dataset               | linear            |
| histogram buffers (num_leaves x feats x bins) | constant   |
| process overhead (polars, lightgbm, numpy)     | constant   |

The intercept ``a`` captures histograms + overhead; forcing through zero
distorts the slope, which is why the previous single-point extrapolation
through the origin was rejected. Results (with both anchors and an honest
uncertainty label) land in ``artifacts/reports/ram_curve.json``.

Usage:
    python measure_ram_curve.py --run-id <64-hex> --family <family>
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

from nmr.promote import DEFAULT_MODELS_DIR, _build_truncated_data, _load_run_record

logger = logging.getLogger("measure_ram_curve")

# (label, target rows, train eras, validation eras) — era counts are rough;
# the script reports the ACTUAL row counts measured.
_POINTS = [
    ("p1", 68_000, 6, 6),
    ("p2", 500_000, 45, 46),
    ("p3", 1_500_000, 137, 137),
]
_FULL_ROWS = 6_853_308  # measured train+validation rows (2026-08-18)
# Anchor A (commit, historical): the recorded full-universe fit peaked at
# ~71 GiB COMMIT on 3,555 features x 2.12M train rows. Scaling by cell count
# AND row count to the full version (780 features x 6.85M rows):
_ANCHOR_HISTORICAL_COMMIT_GIB = 71.0 * (780 / 3555) * (_FULL_ROWS / 2_121_870)
# Anchor B (binning math, ~working set): float32 input 19.9 GiB + binned
# uint8 5.0 GiB + modest overhead.
_ANCHOR_BINNING_GIB = 27.0

_INPUT_JSON = "ram_curve_input.json"

_HELPER = (
    "import json,os,sys; "
    "from nmr.promote import measure_full_history_peak; "
    "cfg=json.load(open(sys.argv[1], encoding='utf-8')); "
    "os.environ['NMR_FULL_HISTORY_SPAWN_MIN_BYTES']='1'; "
    "r=measure_full_history_peak(cfg['config'], cfg['feature_cols'], "
    "cfg['target_cols'], cfg['weights'], data_dir=sys.argv[2]); "
    "print(json.dumps({'child_ws':r[0],'child_commit':r[1],"
    "'parent_ws':r[2],'parent_commit':r[3],'rows':r[4]}))"
)


def _fit_commit(points: list[dict]) -> dict:
    xs = np.asarray([p["rows"] for p in points], dtype=float)
    ys = np.asarray([p["child_commit"] for p in points], dtype=float) / 2**30
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = intercept + slope * xs
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    extrapolated_child = intercept + slope * _FULL_ROWS
    parent_fixed = float(np.median([p["parent_commit"] for p in points])) / 2**30
    combined = extrapolated_child + parent_fixed
    largest_rows = max(p["rows"] for p in points)
    return {
        "intercept_gib": round(float(intercept), 3),
        "slope_gib_per_row": round(float(slope), 12),
        "r2": round(r2, 6),
        "extrapolated_child_commit_gib_at_full": round(float(extrapolated_child), 2),
        "parent_fixed_commit_gib": round(float(parent_fixed), 2),
        "combined_commit_gib_at_full": round(float(combined), 2),
        "extrapolation_factor": round(_FULL_ROWS / largest_rows, 1),
        "uncertainty_note": (
            "extrapolated from the largest measured point "
            f"({largest_rows:,} rows) by {_FULL_ROWS / largest_rows:.1f}x; "
            "an estimate with uncertainty, not a measurement — a confirmation "
            "point near 3M rows is recommended before committing to Stage 2"
        ),
        "anchor_historical_commit_gib": round(_ANCHOR_HISTORICAL_COMMIT_GIB, 1),
        "anchor_binning_gib": _ANCHOR_BINNING_GIB,
        "model_form": (
            "peak = a + b*rows; a = histograms + process overhead (constant), "
            "b*rows = float32 input + binned uint8 dataset (row-linear)"
        ),
    }


def _fit_ws(points: list[dict]) -> dict:
    xs = np.asarray([p["rows"] for p in points], dtype=float)
    ys = np.asarray([p["child_ws"] for p in points], dtype=float) / 2**30
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = intercept + slope * xs
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "intercept_gib": round(float(intercept), 3),
        "slope_gib_per_row": round(float(slope), 12),
        "r2": round(r2, 6),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--family", required=True, help="family slug of the run (run.name)"
    )
    parser.add_argument("--rehearsal-data-root", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    payload = _load_run_record(args.family, args.run_id)
    stored_config = (payload.get("manifest") or {}).get("config")
    if not isinstance(stored_config, dict):
        raise ValueError("run manifest has no config dict")
    feature_cols = list((payload.get("manifest") or {}).get("feature_cols") or [])
    weights = list((payload.get("manifest") or {}).get("weights") or [1.0])
    target_cols = list(stored_config.get("data", {}).get("targets") or ["target"])
    if not feature_cols:
        raise ValueError("run manifest has no feature_cols")

    rehearsal_root = (
        Path(args.rehearsal_data_root)
        if args.rehearsal_data_root is not None
        else DEFAULT_MODELS_DIR.parent / "cache" / "rehearsal_data"
    )
    cache_dir = DEFAULT_MODELS_DIR.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    input_path = cache_dir / _INPUT_JSON
    input_path.write_text(
        json.dumps(
            {"config": stored_config, "feature_cols": feature_cols,
             "target_cols": target_cols, "weights": weights},
            default=str,
        ),
        encoding="utf-8",
    )

    points: list[dict] = []
    for label, _, train_eras, val_eras in _POINTS:
        logger.info(
            "[ram_curve] %s: truncating %d train + %d validation eras",
            label, train_eras, val_eras,
        )
        _build_truncated_data(
            stored_config, rehearsal_root,
            train_eras=train_eras, validation_eras=val_eras,
        )
        proc = subprocess.run(
            [sys.executable, "-c", _HELPER, str(input_path), str(rehearsal_root)],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"[ram_curve] {label} fit failed: {proc.stderr[-2000:]}"
            )
        point = json.loads(proc.stdout.strip().splitlines()[-1])
        points.append(point)
        logger.info(
            "[ram_curve] %s: rows=%d child ws=%.0f MiB commit=%.0f MiB | "
            "parent ws=%.0f MiB commit=%.0f MiB",
            label, point["rows"],
            (point["child_ws"] or 0) / 2**20, (point["child_commit"] or 0) / 2**20,
            (point["parent_ws"] or 0) / 2**20, (point["parent_commit"] or 0) / 2**20,
        )

    fit = _fit_commit(points)
    fit_ws = _fit_ws(points)
    report = {
        "points": [
            {
                "rows": p["rows"],
                "child_ws_gib": round((p["child_ws"] or 0) / 2**30, 4),
                "child_commit_gib": round((p["child_commit"] or 0) / 2**30, 4),
                "parent_ws_gib": round((p["parent_ws"] or 0) / 2**30, 4),
                "parent_commit_gib": round((p["parent_commit"] or 0) / 2**30, 4),
            }
            for p in points
        ],
        "fit": fit,
        "fit_ws": fit_ws,
    }
    out = DEFAULT_MODELS_DIR.parent / "reports" / "ram_curve.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\ncurve written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
