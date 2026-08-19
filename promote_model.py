"""Thin control plane for the promotion writer (nmr/promote.py).

Trains the full version (train+validation) for a registry run and publishes
it under ``artifacts/models/<family>/full/<run_id>/`` with the atomic
``current.json`` pointer. The promotion run is a REHEARSAL unless the user
explicitly uploads the artifact — see the printed instructions.

Usage:
    python promote_model.py --run-id <64-hex> --family <family> [--override-gate] [--force]
    python promote_model.py --champion --family <family> [--override-gate]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nmr.promote import promote_full_version, resolve_champion_run_id

logger = logging.getLogger("promote_model")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="64-hex run id to promote")
    parser.add_argument(
        "--champion", action="store_true", help="promote the current champion run"
    )
    parser.add_argument(
        "--family", required=True, help="model family name (run.name convention)"
    )
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--registry-dir", type=Path, default=None)
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
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if bool(args.run_id) == bool(args.champion):
        parser.error("provide exactly one of --run-id or --champion")

    registry_dir = args.registry_dir or (Path("artifacts") / "registry")
    run_id = args.run_id
    if args.champion:
        run_id = resolve_champion_run_id(registry_dir)

    result = promote_full_version(
        run_id,
        args.family,
        models_dir=args.models_dir,
        registry_dir=registry_dir,
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
