"""Freeze the sklearn-Ridge reference for the medium benchmark cell (D1a).

Run BEFORE any change to the ridge benchmark path. Captures the current
implementation's standardized feature statistics, prediction digest,
per-era CORR (validation `target`), zero-variance counts, finite/null
filtering counts, and measured peak memory for `linear_ridge_medium` on
the real v5.3 data.

Writes ``artifacts/reports/ridge_reference_<git-sha>.json``. The D1a
tolerance tests compare the memory-safe implementation against this
frozen artifact; the SHA-256 prediction digest is forensic evidence
(bit-exact only if no last-ulp change occurs).

Business logic stays in ``nmr.*``; this script only wires and records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from nmr.benchmark import (
    DEFAULT_BENCHMARK_PURGE_ERAS,
    _polars_feature_statistics,
    generate_ridge_predictions,
    load_benchmark_suite_config,
    resolve_benchmark_feature_cols,
    train_validation_purged_split,
)
from nmr.evaluation import EvaluationEngine
from nmr.models import _peak_memory_counters


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _atomic_write_json(path: Path, payload: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/v5.3")
    parser.add_argument("--configs", default="configs/benchmarks")
    parser.add_argument("--cell-id", default="linear_ridge_medium")
    parser.add_argument("--report-dir", default="artifacts/reports")
    parser.add_argument(
        "--label",
        default="pre",
        help="phase label embedded in the filename and payload (pre/post)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    suite = load_benchmark_suite_config(args.configs)
    try:
        cell = next(c for c in suite.cells if c.benchmark_id == args.cell_id)
    except StopIteration as exc:
        raise SystemExit(f"cell {args.cell_id!r} not found in {args.configs}") from exc

    schema_cols = list(pl.read_parquet_schema(data_dir / "train.parquet").keys())
    feature_cols = resolve_benchmark_feature_cols(
        data_dir / "features.json", cell.input_space, schema_cols
    )
    print(f"[freeze] cell={cell.benchmark_id} features={len(feature_cols)}")

    train = pl.read_parquet(
        data_dir / "train.parquet",
        columns=["era", "id", *feature_cols, *cell.targets],
    )
    val = pl.read_parquet(
        data_dir / "validation.parquet",
        columns=["era", "id", *feature_cols],
    )

    trimmed_train_eras, _ = train_validation_purged_split(
        train.get_column("era").unique().to_list(),
        val.get_column("era").unique().to_list(),
        purge_eras=DEFAULT_BENCHMARK_PURGE_ERAS,
    )

    # Bounded statistics via the same helper the current implementation uses
    # (no full-width float64 temporaries; the pre-fix np.std deviation matrix
    # was ~15.7 GiB at this geometry).
    mu32, scale32 = _polars_feature_statistics(
        train, feature_cols, trimmed_train_eras, "era"
    )
    mu = mu32.astype(np.float64)
    inv_sigma = scale32.astype(np.float64)
    sigma = np.where(inv_sigma > 0.0, 1.0 / inv_sigma, 0.0)
    train_rows = train.filter(pl.col("era").is_in(trimmed_train_eras))
    finite_feature_columns = sum(
        1
        for ok in train_rows.select(
            [pl.col(c).is_finite().all() for c in feature_cols]
        ).row(0)
        if ok
    )
    finite_train_rows = train_rows.filter(
        pl.all_horizontal([pl.col(c).is_finite() for c in feature_cols])
    ).height

    y_finite_counts: dict[str, int] = {}
    for target in cell.targets:
        y = train_rows.get_column(target).cast(pl.Float64).to_numpy()
        y_finite_counts[target] = int(np.isfinite(y).sum())

    params = dict(cell.params)
    preds = generate_ridge_predictions(
        train,
        val,
        targets=list(cell.targets),
        feature_cols=feature_cols,
        alpha=float(params["alpha"]),
        seed=cell.seed,
    )
    pred_values = preds.get_column("prediction").to_numpy()
    pred_digest = hashlib.sha256(
        np.ascontiguousarray(pred_values).tobytes()
    ).hexdigest()

    val_targets = pl.read_parquet(
        data_dir / "validation.parquet", columns=["era", "id", "target"]
    )
    joined = preds.join(val_targets, on=["era", "id"], how="inner")
    engine = EvaluationEngine("custom")
    per_era = engine.per_era_corr(
        joined, pred_col="prediction", target_col="target", era_col="era"
    )
    summary = engine.summarize(per_era)

    peak_ws, peak_commit = _peak_memory_counters()

    payload = {
        "kind": "ridge_reference",
        "phase": args.label,
        "cell_id": cell.benchmark_id,
        "git_sha": _git_sha(),
        "feature_count": len(feature_cols),
        "purge_eras": DEFAULT_BENCHMARK_PURGE_ERAS,
        "train_rows_purged": train_rows.height,
        "val_rows": int(val.height),
        "feature_statistics": {
            "mu": [float(v) for v in mu],
            "sigma": [float(v) for v in sigma],
            "zero_variance_features": int((sigma == 0.0).sum()),
            "finite_feature_columns": finite_feature_columns,
            "finite_train_rows": finite_train_rows,
        },
        "finite_y_counts": y_finite_counts,
        "predictions": {
            "sha256": pred_digest,
            "n": int(pred_values.size),
            "finite": bool(np.isfinite(pred_values).all()),
        },
        "per_era_corr": {
            "n_eras": len(per_era),
            "mean": summary.mean,
            "std": summary.std,
            "sharpe": summary.sharpe,
            "by_era": {
                str(k): float(v)
                for k, v in sorted(per_era.items(), key=lambda kv: int(kv[0]))
            },
        },
        "peak_memory": {
            "ws_gib": round(float(peak_ws) / (2**30), 2),
            "commit_gib": round(float(peak_commit) / (2**30), 2),
        },
    }

    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"ridge_reference_{payload['git_sha'][:12]}_{args.label}.json"
    _atomic_write_json(out, payload)
    print(
        f"[freeze] wrote {out} corr_mean={summary.mean:.6f} "
        f"peak_ws={payload['peak_memory']['ws_gib']}GiB "
        f"peak_commit={payload['peak_memory']['commit_gib']}GiB"
    )


if __name__ == "__main__":
    main()
