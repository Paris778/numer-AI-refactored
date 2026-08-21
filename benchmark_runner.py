"""Control plane for the 5-tier benchmark hierarchy (the line in the sand).

Thin wrapper only: argument parsing, data wiring, output writing, exit codes.
All benchmark logic lives in ``nmr.benchmark``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from nmr.benchmark import (
    BenchmarkHierarchy,
    gate_report_frame,
    hierarchy_frame,
    load_benchmark_data,
    load_benchmark_suite_config,
    tier_max_corrs,
)
from nmr.benchmark_fleet import (
    BenchmarkFleet,
    load_fleet_suite_config,
    load_tier_rungs_from_csv,
    select_fleet_cells,
    write_fleet_csv,
)


def _min_one_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("n-boot must be >= 1")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic 5-tier benchmark hierarchy runner."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "v5.3")
    parser.add_argument(
        "--configs", type=Path, default=Path("configs") / "benchmarks"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts")
        / "reports"
        / "benchmark_hierarchy_scorecard.csv",
    )
    parser.add_argument(
        "--gate-report",
        type=Path,
        default=Path("artifacts") / "reports" / "benchmark_gate_report.csv",
    )
    parser.add_argument(
        "--fleet-configs",
        type=Path,
        default=Path("configs") / "benchmarks" / "fleet",
    )
    parser.add_argument(
        "--fleet-output",
        type=Path,
        default=Path("artifacts") / "reports" / "benchmark_fleet_scorecard.csv",
    )
    parser.add_argument("--no-fleet", action="store_true")
    parser.add_argument(
        "--only-fleet", action="store_true",
        help="skip the tiered hierarchy entirely; fleet placement rungs are "
             "loaded from the last hierarchy scorecard CSV (--rungs-csv). "
             "Hard gates are skipped in this mode.",
    )
    parser.add_argument(
        "--rungs-csv", type=Path, default=None,
        help="hierarchy scorecard CSV to source placement rungs from "
             "(default: --output). Used by --only-fleet.",
    )
    parser.add_argument(
        "--fleet-ids", default="",
        help="comma-separated fleet benchmark_ids to run (default: all).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=_min_one_int, default=1000)
    parser.add_argument("--min-overlap-eras", type=int, default=20)
    parser.add_argument("--horizon", choices=("20D", "60D"), default="20D")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument("--fast-mode", action="store_true")
    return parser


def _parse_args() -> argparse.Namespace:
    return _build_parser().parse_args()


def _parse_args_with(argv: list[str]) -> argparse.Namespace:
    """Test hook: parse an explicit argument vector."""
    return _build_parser().parse_args(argv)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("benchmark_runner")

    if args.no_fleet and args.only_fleet:
        log.error("--no-fleet and --only-fleet are mutually exclusive")
        return 1
    fleet_ids = tuple(
        part.strip() for part in args.fleet_ids.split(",") if part.strip()
    )

    if args.only_fleet:
        try:
            spec = load_benchmark_suite_config(args.configs)  # gate config only
            data = load_benchmark_data(args.data_dir)
            fleet_cells = load_fleet_suite_config(args.fleet_configs)
            fleet_cells = select_fleet_cells(fleet_cells, fleet_ids)
            rungs_path = args.rungs_csv or args.output
            rungs = load_tier_rungs_from_csv(rungs_path)
            fleet = BenchmarkFleet(
                spec=fleet_cells,
                data=data,
                seed=args.seed,
                horizon=args.horizon,
                n_boot=1 if args.fast_mode else args.n_boot,
                min_overlap_eras=args.min_overlap_eras,
                fast_mode=args.fast_mode,
            )
        except (ValueError, FileNotFoundError) as exc:
            log.error("FLEET FAILURE: %s", exc)
            return 1
        t0 = time.perf_counter()
        log.info(
            "Running %d fleet cells (only-fleet; rungs from %s)%s",
            len(fleet_cells), rungs_path,
            " (fast mode)" if args.fast_mode else "",
        )
        fleet_result = fleet.run(tier_rungs=rungs, gate=spec.gate)
        log.info("Fleet scored in %.1fs", time.perf_counter() - t0)
        write_fleet_csv(fleet_result, args.fleet_output)
        log.info("Fleet scorecard written to %s", args.fleet_output)
        for mid in fleet_result.scorecards:
            log.info(
                "fleet %s: placement=%s selection_bias=%s",
                mid, fleet_result.placements[mid],
                fleet_result.selection_bias[mid],
            )
        log.info("Hard gates skipped in --only-fleet mode (no live hierarchy).")
        return 0

    log.info("Loading benchmark suite config from %s", args.configs)
    spec = load_benchmark_suite_config(args.configs)
    log.info("Loading benchmark data from %s", args.data_dir)
    data = load_benchmark_data(args.data_dir)

    hierarchy = BenchmarkHierarchy(
        spec=spec,
        data=data,
        seed=args.seed,
        horizon=args.horizon,
        n_boot=1 if args.fast_mode else args.n_boot,
        min_overlap_eras=args.min_overlap_eras,
        fast_mode=args.fast_mode,
    )

    t0 = time.perf_counter()
    log.info(
        "Running %d benchmark cells%s",
        len(spec.cells),
        " (fast mode)" if args.fast_mode else "",
    )
    result = hierarchy.run()
    log.info("Hierarchy scored in %.1fs", time.perf_counter() - t0)

    for path in (args.output, args.gate_report):
        path.parent.mkdir(parents=True, exist_ok=True)

    hierarchy_frame(result).write_csv(args.output)
    gate_frame = gate_report_frame(result)
    gate_frame.write_csv(args.gate_report)
    log.info("Scorecard frame written to %s", args.output)
    log.info("Gate report written to %s", args.gate_report)

    for row in gate_frame.iter_rows(named=True):
        log.info(
            "tier4 gate %s: measured=%s threshold=%s pass=%s",
            row["field"], row["measured"], row["threshold"], row["pass"],
        )

    if not args.no_fleet:
        try:
            fleet_cells = select_fleet_cells(
                load_fleet_suite_config(args.fleet_configs), fleet_ids
            )
        except ValueError as exc:
            log.error("FLEET CONFIG FAILURE: %s", exc)
            return 1
        rungs = tier_max_corrs(result.scorecards, result.tier_of)
        fleet = BenchmarkFleet(
            spec=fleet_cells,
            data=data,
            seed=args.seed,
            horizon=args.horizon,
            n_boot=1 if args.fast_mode else args.n_boot,
            min_overlap_eras=args.min_overlap_eras,
            fast_mode=args.fast_mode,
        )
        t1 = time.perf_counter()
        log.info(
            "Running %d fleet cells%s",
            len(fleet_cells), " (fast mode)" if args.fast_mode else "",
        )
        fleet_result = fleet.run(tier_rungs=rungs, gate=spec.gate)
        log.info("Fleet scored in %.1fs", time.perf_counter() - t1)
        write_fleet_csv(fleet_result, args.fleet_output)
        log.info("Fleet scorecard written to %s", args.fleet_output)
        for mid in fleet_result.scorecards:
            log.info(
                "fleet %s: placement=%s selection_bias=%s",
                mid, fleet_result.placements[mid],
                fleet_result.selection_bias[mid],
            )

    hard_failures: list[str] = []
    if not result.null_floor_ok:
        hard_failures.extend(result.null_floor_errors)
    hard_failures.extend(result.tier4_violations)
    if not result.monotone_ok:
        if args.fast_mode:
            log.warning(
                "Monotonicity not enforced in fast mode (degraded tier params): %s",
                result.monotone_error,
            )
        else:
            hard_failures.append(result.monotone_error or "monotone failure")

    if hard_failures:
        for message in hard_failures:
            log.error("GATE FAILURE: %s", message)
        return 1

    log.info("All hard gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
