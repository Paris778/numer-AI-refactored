"""Run a named batch of experiment configs and record trial lineage.

Thin control plane: argument parsing, wiring, and printing only. All logic
lives in ``nmr.campaign`` / ``nmr.runner`` / ``nmr.registry``.

Usage:
    python run_campaign.py --config configs/a.yaml --config configs/b.yaml \
        --name my-campaign [--registry artifacts/registry] \
        [--campaigns-dir artifacts/campaigns] [--deploy] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nmr import ExperimentRunner, RunRegistry, load_config
from nmr.campaign import CampaignRun, build_campaign_log, write_campaign_log

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_campaign")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True,
                        help="path to an experiment config YAML (repeatable)")
    parser.add_argument("--name", required=True, help="campaign name")
    parser.add_argument("--registry", default="artifacts/registry",
                        help="registry root directory")
    parser.add_argument("--campaigns-dir", default="artifacts/campaigns",
                        help="campaign log output directory")
    parser.add_argument("--deploy", action="store_true",
                        help="pass deploy=True to ExperimentRunner.run")
    parser.add_argument("--dry-run", action="store_true",
                        help="print run ids without training or writing")
    args = parser.parse_args(argv)

    config_paths = [Path(p) for p in args.config]
    # RunRegistry.__init__ creates its root directory, so defer construction
    # out of --dry-run: dry-run must not write anything (no registry dir).
    registry: RunRegistry | None = None
    existing: set[str] = set()
    if not args.dry_run:
        registry = RunRegistry(args.registry)
        existing = {entry["run_id"] for entry in registry.list()}

    runs: list[CampaignRun] = []
    failed = 0
    for path in config_paths:
        try:
            cfg = load_config(path)
            run_id = ExperimentRunner.compute_run_id(cfg)
        except Exception as exc:  # validation/config failures are campaign-level
            logger.error("[campaign] config %s invalid: %s", path, exc)
            runs.append(CampaignRun(str(path), run_id=None, status="error", error=str(exc)))
            failed += 1
            continue

        if args.dry_run:
            logger.info("[campaign] dry-run: %s -> %s", path, run_id)
            runs.append(CampaignRun(str(path), run_id=run_id, status="skipped"))
            continue

        if run_id in existing:
            logger.info("[campaign] %s already recorded; skipping", run_id)
            existing.add(run_id)
            runs.append(CampaignRun(str(path), run_id=run_id, status="skipped"))
            continue

        try:
            if registry is None:
                # constructed whenever not args.dry_run — never rely on assert
                # for control flow (stripped under python -O)
                raise RuntimeError(
                    "internal error: registry is None outside dry-run mode"
                )
            result = ExperimentRunner(cfg).run(deploy=args.deploy)
            registry.record(result)
            existing.add(result.run_id)
            runs.append(CampaignRun(str(path), run_id=result.run_id, status="recorded"))
            logger.info("[campaign] recorded %s -> %s", path, result.run_id)
        except Exception as exc:
            logger.exception("[campaign] run failed for %s", path)
            runs.append(CampaignRun(str(path), run_id=None, status="error", error=str(exc)))
            failed += 1

    if args.dry_run:
        for run in runs:
            print(f"dry-run\t{run.config_path}\t{run.run_id}")
        return 0

    log = build_campaign_log(args.name, config_paths, runs)
    log_path = write_campaign_log(log, args.campaigns_dir)
    logger.info("[campaign] log written to %s", log_path)
    for run in runs:
        print(f"{run.status}\t{run.config_path}\t{run.run_id}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
