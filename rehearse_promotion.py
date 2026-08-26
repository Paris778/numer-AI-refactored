"""Thin control plane for the D7 truncated-window rehearsal.

Proves the promotion writer end-to-end on real data in minutes: truncated
train+validation subset, forced fresh-process fit, measured peak RAM
extrapolation, and the Phase D acceptance criterion (raw artifact output
validated by the official numerai_tools validator on the real local
live.parquet). A rehearsal is NOT a capital deployment.

Usage:
    python rehearse_promotion.py --run-id <64-hex> --family <family>
"""

# ruff: noqa: E402 — apply_thread_limits() must run before the imports below:
# polars/OpenMP/BLAS read their pool sizes at first use, not at import.
from __future__ import annotations

from nmr.hardware import apply_thread_limits

apply_thread_limits()

import argparse
import logging
from pathlib import Path

from nmr.promote import rehearse_promotion

logger = logging.getLogger("rehearse_promotion")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="64-hex run id to rehearse")
    parser.add_argument(
        "--family", required=True, help="model family name (run.name convention)"
    )
    parser.add_argument("--rehearsal-data-root", type=Path, default=None)
    parser.add_argument("--train-eras", type=int, default=6)
    parser.add_argument("--validation-eras", type=int, default=6)
    parser.add_argument("--live-features", type=Path, default=None)
    parser.add_argument("--live-benchmark", type=Path, default=None)
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = rehearse_promotion(
        args.run_id,
        args.family,
        rehearsal_data_root=args.rehearsal_data_root,
        train_eras=args.train_eras,
        validation_eras=args.validation_eras,
        live_features_path=args.live_features,
        live_benchmark_path=args.live_benchmark,
    )
    peak_mib = (result.measured_peak_bytes or 0) / 2**20
    print(f"\nrehearsal artifact: {result.artifact_path}")
    print(f"acceptance passed: {result.acceptance_passed}")
    print(
        f"measured peak: {peak_mib:.1f} MiB on {result.train_validation_rows} rows"
    )
    print(f"RAM estimate: {result.ram_estimate_path}")
    print("\nThis is a REHEARSAL — no capital was deployed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
