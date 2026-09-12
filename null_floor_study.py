"""Tier-0 null-floor calibration study (director diagnostic, 2026-09-09).

Reproduces the exact hierarchy null scoring on the current meta-overlap
window, binds a cheap per-era-CORR + AC-Sharpe path against the full
``evaluate_model`` scorecard, then sweeps a fixed seed set through the
three structural null kinds. Records per-kind and family-wise AC-Sharpe
behavior. No threshold changes, no model training — evidence only.

With ``--emit-calibration PATH`` it also writes the committed, digest-bound
Tier-0 null-floor calibration reference (pre-registered family-wise p99
threshold, window identity, seed-42 expected values), which the benchmark
gate loads and refuses to use against a different data window.

Writes ``artifacts/reports/null_floor_study.json`` by default. Business logic
stays in ``nmr.*``; this script only wires and records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from nmr.benchmark import (
    NULL_FLOOR_KINDS,
    generate_null_predictions,
    load_benchmark_data,
    load_benchmark_suite_config,
    resolve_benchmark_feature_cols,
)
from nmr.evaluation import EvaluationEngine
from nmr.inference import ac_adjusted_sharpe, resolve_bandwidth
from nmr.payout import PAYOUT_FACTOR_FILENAME, era_payout_factors, resolve_payout_policy
from nmr.predictions import validation_key_fingerprint
from nmr.scorecard import evaluate_model

_CALIBRATION_SCHEMA_VERSION = 1


def _canonical_digest(payload: dict) -> str:
    """SHA-256 over canonical JSON (sorted keys, compact separators)."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _study_code_fingerprint() -> str:
    """SHA-256 of this study script (CRLF-normalized), the study-code identity."""
    raw = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


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


def _cheap_ac_sharpe(
    preds: pl.DataFrame,
    targets: pl.DataFrame,
    window_eras: list[str],
    *,
    scoring_target: str,
    horizon: str,
) -> dict[str, float]:
    """Per-era CORR over the exact window + the scorecard AC-Sharpe formula."""
    engine = EvaluationEngine("custom")
    joined = preds.join(targets.select(["era", "id", scoring_target]), on=["era", "id"])
    joined = joined.filter(pl.col("era").is_in(window_eras))
    per_era = engine.per_era_corr(
        joined, pred_col="prediction", target_col=scoring_target, era_col="era"
    )
    values = [per_era[era] for era in sorted(per_era, key=int)]
    corr_mean = float(np.mean(values)) if values else float("nan")
    ac = ac_adjusted_sharpe(values, horizon=horizon)
    sr = float(np.mean(values) / np.std(values, ddof=0)) if np.std(values) else 0.0
    return {
        "corr": corr_mean,
        "corr_sharpe_ac": float(ac),
        "corr_sharpe": float(sr),
        "std_corr": float(np.std(values, ddof=0)),
        "n_eras": len(values),
    }


def _synthetic_ac_check(horizon: str) -> dict:
    """Verify the AC-Sharpe formula against a hand-computed Newey-West value."""
    rng = np.random.default_rng(0)
    n = 100
    rho = 0.3
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal()
    got = ac_adjusted_sharpe(x, horizon=horizon)
    k_max = resolve_bandwidth(n, horizon)
    mean = float(np.mean(x))
    centered = x - mean
    denom = float(np.dot(centered, centered))
    weights = 1.0 - (np.arange(1, k_max + 1, dtype=float) / (k_max + 1.0))
    rhos = [
        float(np.dot(centered[:-k], centered[k:])) / denom for k in range(1, k_max + 1)
    ]
    d_term = max(1.0 + 2.0 * float(np.sum(weights * np.asarray(rhos))), 1e-12)
    expected = float((mean / np.std(x, ddof=0)) / np.sqrt(d_term))
    iid = rng.normal(size=200)
    iid_got = ac_adjusted_sharpe(iid, horizon=horizon)
    return {
        "n": n,
        "rho": rho,
        "bandwidth": k_max,
        "computed": got,
        "hand_computed": expected,
        "abs_diff": abs(got - expected),
        "iid_computed": float(iid_got),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/v5.3")
    parser.add_argument("--configs", default="configs/benchmarks")
    parser.add_argument("--report-dir", default="artifacts/reports")
    parser.add_argument("--sweep-seeds", type=int, default=200)
    parser.add_argument(
        "--binding-seeds",
        default="42",
        help="comma-separated seeds to bind against the full evaluate_model path",
    )
    parser.add_argument(
        "--emit-calibration",
        default=None,
        help="write the digest-bound Tier-0 null-floor calibration reference here",
    )
    parser.add_argument(
        "--calibration-quantile",
        type=float,
        default=0.99,
        help="pre-registered family-wise false-positive quantile",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="print a flushed progress line + ETA every N seeds (0 = off)",
    )
    args = parser.parse_args()
    binding_seeds = tuple(
        int(part) for part in args.binding_seeds.split(",") if part.strip()
    )

    data_dir = Path(args.data_dir)
    data = load_benchmark_data(data_dir)
    spec = load_benchmark_suite_config(args.configs)
    if spec.gate is None:
        raise SystemExit("no tier-4 gate config found")
    gate_policy = resolve_payout_policy(spec.gate.payout_policy_id)
    scoring_target = gate_policy.target or "target"
    scoring_horizon = gate_policy.scoring_horizon or "20D"

    pf_map = era_payout_factors(
        Path(data.validation_path).parent / PAYOUT_FACTOR_FILENAME
    )
    scoring_pf = pf_map if gate_policy.fixed_payout_factor is None else None

    window_eras = sorted(data.meta_model.get_column("era").unique().to_list(), key=int)
    window_key_fingerprint = validation_key_fingerprint(
        data.meta_model.select(["era", "id"])
    )
    print(
        f"[study] window eras={len(window_eras)} target={scoring_target} "
        f"horizon={scoring_horizon} policy={gate_policy.policy_id} "
        f"key_fp={window_key_fingerprint[:12]}"
    )

    schema_cols = list(pl.read_parquet_schema(data.validation_path).keys())
    small_cols = resolve_benchmark_feature_cols(
        data.features_json, "small", schema_cols
    )
    val_id = pl.read_parquet(data.validation_path, columns=["era", "id"])
    val_features = pl.read_parquet(
        data.validation_path, columns=["era", "id", *small_cols]
    )
    targets = pl.read_parquet(
        data.validation_path, columns=["era", "id", "target", scoring_target]
    )

    # --- Bind the cheap path to the full scorecard path (binding seeds) ---
    bindings: dict[str, dict] = {}
    for kind in NULL_FLOOR_KINDS:
        for seed in binding_seeds:
            preds = generate_null_predictions(val_id, kind=kind, seed=seed)
            scorecard = evaluate_model(
                preds,
                meta_model=data.meta_model,
                benchmarks=data.benchmarks,
                features=val_features,
                targets=targets,
                n_trials=1,
                seed=seed,
                payout_policy=gate_policy,
                horizon=scoring_horizon,
                main_target=scoring_target,
                benchmark_col=spec.reference_column,
                pf=scoring_pf,
                n_boot=1,
                min_overlap_eras=20,
                model_id=f"{kind}__seed{seed}",
            )
            cheap = _cheap_ac_sharpe(
                preds,
                targets,
                window_eras,
                scoring_target=scoring_target,
                horizon=scoring_horizon,
            )
            bindings[f"{kind}__seed{seed}"] = {
                "full_corr": float(scorecard.corr.value),
                "full_corr_sharpe_ac": float(scorecard.corr_sharpe_ac.value),
                "full_n_eras": int(scorecard.corr.n_eras),
                "cheap_corr": cheap["corr"],
                "cheap_corr_sharpe_ac": cheap["corr_sharpe_ac"],
                "cheap_n_eras": cheap["n_eras"],
                "corr_abs_diff": abs(float(scorecard.corr.value) - cheap["corr"]),
                "ac_abs_diff": abs(
                    float(scorecard.corr_sharpe_ac.value) - cheap["corr_sharpe_ac"]
                ),
            }
            print(
                f"[study] bind {kind} seed={seed} full_ac="
                f"{scorecard.corr_sharpe_ac.value:.6f} cheap_ac="
                f"{cheap['corr_sharpe_ac']:.6f}"
            )

    max_ac_diff = max(v["ac_abs_diff"] for v in bindings.values())
    max_corr_diff = max(v["corr_abs_diff"] for v in bindings.values())
    print(
        f"[study] binding max |ac diff|={max_ac_diff:.3e} "
        f"max |corr diff|={max_corr_diff:.3e}"
    )

    # --- Seed sweep with the cheap path ---
    sweep_seeds = tuple(range(min(args.sweep_seeds, 2000)))
    sweep: dict[str, list[dict]] = {kind: [] for kind in NULL_FLOOR_KINDS}
    family_max: dict[int, float] = {}
    progress_path = Path(args.report_dir) / "null_floor_progress.json"
    if args.progress_every:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_start = time.perf_counter()
    print(
        f"[study] sweep start: {len(sweep_seeds)} seeds x "
        f"{len(NULL_FLOOR_KINDS)} null kinds",
        flush=True,
    )
    for completed, seed in enumerate(sweep_seeds, start=1):
        per_kind: dict[str, dict] = {}
        for kind in NULL_FLOOR_KINDS:
            preds = generate_null_predictions(val_id, kind=kind, seed=seed)
            cheap = _cheap_ac_sharpe(
                preds,
                targets,
                window_eras,
                scoring_target=scoring_target,
                horizon=scoring_horizon,
            )
            per_kind[kind] = cheap
            sweep[kind].append({"seed": seed, **cheap})
        family_max[seed] = float(
            max(abs(per_kind[k]["corr_sharpe_ac"]) for k in per_kind)
        )
        if args.progress_every and completed % args.progress_every == 0:
            elapsed = time.perf_counter() - sweep_start
            rate = elapsed / completed
            remaining = len(sweep_seeds) - completed
            print(
                f"[study] progress {completed}/{len(sweep_seeds)} "
                f"elapsed={elapsed:.0f}s eta={rate * remaining:.0f}s "
                f"({rate:.2f}s/seed)",
                flush=True,
            )
            _atomic_write_json(
                progress_path,
                {
                    "kind": "null_floor_study_progress",
                    "phase": "sweep",
                    "completed_seeds": completed,
                    "total_seeds": len(sweep_seeds),
                    "units_per_seed": len(NULL_FLOOR_KINDS),
                    "elapsed_s": round(elapsed, 3),
                    "sec_per_seed": round(rate, 4),
                    "eta_s": round(rate * remaining, 3),
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )

    print(
        f"[study] sweep done: {len(sweep_seeds)} seeds in "
        f"{time.perf_counter() - sweep_start:.0f}s",
        flush=True,
    )

    def _quantiles(values: list[float]) -> dict:
        arr = np.asarray(values, dtype=float)
        return {
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "p95": float(np.quantile(arr, 0.95)),
            "p99": float(np.quantile(arr, 0.99)),
            "max": float(np.max(arr)),
        }

    report = {
        "kind": "null_floor_study",
        "generated_at": datetime.now(UTC).isoformat(),
        "window": {
            "n_eras": len(window_eras),
            "first_era": window_eras[0],
            "last_era": window_eras[-1],
            "val_rows": int(val_id.height),
            "payout_policy_id": gate_policy.policy_id,
            "scoring_target": scoring_target,
            "scoring_horizon": scoring_horizon,
            "scoring_backend": "custom",
            "n_boot": 1,
            "min_overlap_eras": 20,
        },
        "binding": {
            "validation_seeds": list(binding_seeds),
            "max_ac_abs_diff": max_ac_diff,
            "max_corr_abs_diff": max_corr_diff,
            "rows": bindings,
        },
        "sweep": {
            "n_seeds": len(sweep_seeds),
            "seeds": list(sweep_seeds),
            "per_kind": {
                kind: {
                    "corr_mean_of_means": float(np.mean([r["corr"] for r in rows])),
                    "corr_std": float(np.std([r["corr"] for r in rows])),
                    "ac_sharpe_quantiles_of_abs": _quantiles(
                        [abs(r["corr_sharpe_ac"]) for r in rows]
                    ),
                    "ac_sharpe_min": float(min(r["corr_sharpe_ac"] for r in rows)),
                    "ac_sharpe_max": float(max(r["corr_sharpe_ac"] for r in rows)),
                    "seed42": next(r for r in rows if r["seed"] == 42),
                }
                for kind, rows in sweep.items()
            },
            "family_wise_max_abs_ac_sharpe": _quantiles(
                [family_max[s] for s in sweep_seeds]
            ),
            "seed42_family_max": family_max[42],
            "rows": sweep,
        },
        "synthetic_ac_check": _synthetic_ac_check(scoring_horizon),
    }

    out = Path(args.report_dir) / "null_floor_study.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out, report)
    print(f"[study] wrote {out}")
    print("[study] per-kind AC-Sharpe |abs| quantiles:")
    for kind, rows in sweep.items():
        q = _quantiles([abs(r["corr_sharpe_ac"]) for r in rows])
        print(
            f"  {kind}: p50={q['p50']:.4f} p90={q['p90']:.4f} "
            f"p95={q['p95']:.4f} p99={q['p99']:.4f} max={q['max']:.4f}"
        )
    fq = _quantiles([family_max[s] for s in sweep_seeds])
    family_values = [family_max[s] for s in sweep_seeds]
    threshold = float(np.quantile(family_values, args.calibration_quantile))
    print(
        f"  family-wise max: p50={fq['p50']:.4f} p90={fq['p90']:.4f} "
        f"p95={fq['p95']:.4f} p99={fq['p99']:.4f} max={fq['max']:.4f}"
    )
    print(
        f"  seed42 family max={family_max[42]:.4f} "
        f"calibrated p{args.calibration_quantile * 100:.0f}={threshold:.4f}"
    )

    if args.emit_calibration:
        calibration: dict = {
            "schema_version": _CALIBRATION_SCHEMA_VERSION,
            "kind": "tier0_null_floor_calibration",
            "method": "family_wise_quantile",
            "data_version": Path(args.data_dir).name,
            "window_key_fingerprint": window_key_fingerprint,
            "validation_eras": window_eras,
            "validation_rows": int(data.meta_model.height),
            "scoring_identity": {
                "payout_policy_id": gate_policy.policy_id,
                "scoring_target": scoring_target,
                "scoring_horizon": scoring_horizon,
                "scoring_backend": "custom",
                "n_boot": 1,
                "min_overlap_eras": 20,
            },
            "null_kinds": list(NULL_FLOOR_KINDS),
            "seed_set": {"start": 0, "count": len(sweep_seeds)},
            "selected_quantile": args.calibration_quantile,
            "selected_threshold": threshold,
            "per_kind_distributions": {
                kind: {
                    "corr_mean_of_means": float(np.mean([r["corr"] for r in rows])),
                    "corr_std": float(np.std([r["corr"] for r in rows])),
                    "abs_ac_sharpe_quantiles": _quantiles(
                        [abs(r["corr_sharpe_ac"]) for r in rows]
                    ),
                    "seed42": next(r for r in rows if r["seed"] == 42),
                }
                for kind, rows in sweep.items()
            },
            "family_wise_max_abs_ac_sharpe_quantiles": fq,
            "seed42_expected": {
                kind: next(r for r in rows if r["seed"] == 42)
                for kind, rows in sweep.items()
            },
            "seed42_family_max": family_max[42],
            "binding_evidence": {
                "validation_seeds": list(binding_seeds),
                "max_ac_abs_diff": max_ac_diff,
                "max_corr_abs_diff": max_corr_diff,
            },
            "study_code_fingerprint": _study_code_fingerprint(),
            "generated_at": datetime.now(UTC).isoformat(),
        }
        calibration["digest"] = _canonical_digest(calibration)
        cal_path = Path(args.emit_calibration)
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(cal_path, calibration)
        print(
            f"[study] wrote calibration {cal_path} "
            f"threshold={threshold:.4f} digest={calibration['digest'][:12]}"
        )


if __name__ == "__main__":
    main()
