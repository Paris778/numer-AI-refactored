"""Thin control plane for the promotion writer (nmr/promote.py).

Trains the full version (train+validation) for an experiments-layout run and
publishes it under ``experiments/<family>/exports/<scope>/<run_id>/`` with
the atomic ``current.json`` pointer. The promotion run is a REHEARSAL unless
the user explicitly uploads the artifact — see the printed instructions.

Usage:
    python promote_model.py --run-id <64-hex> --family <family> [--override-gate] [--force]
    python promote_model.py --champion --family <family> [--override-gate]
"""

# ruff: noqa: E402 — apply_thread_limits() must run before the imports below:
# polars/OpenMP/BLAS read their pool sizes at first use, not at import.
from __future__ import annotations

from nmr.hardware import apply_thread_limits

apply_thread_limits()

import argparse
import logging

from nmr import paths
from nmr.promote import promote_full_version
from nmr.registry import RunRegistry

logger = logging.getLogger("promote_model")


def _resolve_champion_run_id() -> str:
    """Run id of the current champion — the atomic experiments-root pointer."""
    champion = RunRegistry(paths.EXPERIMENTS_ROOT).resolve_champion()
    if champion is None:
        raise FileNotFoundError(
            f"no champion: {paths.champion_path()} missing — promote a run first"
        )
    return champion[0]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="64-hex run id to promote")
    parser.add_argument(
        "--champion", action="store_true", help="promote the current champion run"
    )
    parser.add_argument(
        "--family", required=True, help="model family name (run.name convention)"
    )
    parser.add_argument(
        "--override-gate",
        action="store_true",
        help="promote/rehearse despite a tier-4 gate failure "
        "(recorded as tier4_gate_passed: false in the manifest; "
        "never covers contract validity)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing slot / repoint current.json",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if bool(args.run_id) == bool(args.champion):
        _build_parser().error("provide exactly one of --run-id or --champion")

    run_id = args.run_id
    if args.champion:
        run_id = _resolve_champion_run_id()

    result = promote_full_version(
        run_id,
        args.family,
        override_gate=args.override_gate,
        force=args.force,
    )
    print(f"\npublished full version: {result.artifact_path}")
    print(f"manifest: {result.manifest_path}")
    print(
        f"tier4_gate_passed: {result.tier4_gate_passed}  "
        f"override_used: {result.override_used}"
    )
    print(
        "\nNumerai Model Uploads: upload predict.pkl on the Submissions page "
        "(Upload Model, Python 3.12). Uploading disables manual API "
        "submissions for that slot. This promotion is a REHEARSAL unless you "
        "explicitly upload it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
