"""Round-aware Numerai dataset refresh — thin control plane.

Downloads/updates data/v5.3 assets via the public Numerai API and maintains
data/numerai_era_data.csv (the refresh ledger). All decision logic lives in
``nmr/refresh.py``; this script only wires numerapi calls, file I/O, and
argument parsing.

Exit codes: 0 = ok (or advisory-only warning), 1 = hard failure,
3 = gate tripped (--check-only/--strict).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import polars as pl

from nmr._atomicio import atomic_write_text
from nmr.features import resolve_feature_sets
from nmr.refresh import (
    CURRENT_DATA_VERSION,
    build_era_manifest,
    classify_refresh_plan,
    detect_newer_version,
    needs_live_refresh,
)

import numerapi

_VERSION_ALERT = (
    "[WARNING] New Numerai data version detected: {newer} is available. "
    "This repo's pipeline targets {current}. Consider migrating before the "
    "next campaign; continuing with {current}."
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh Numerai datasets and the era ledger."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--version", default=CURRENT_DATA_VERSION)
    parser.add_argument(
        "--era-csv",
        type=Path,
        default=Path("data") / "numerai_era_data.csv",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live-only", action="store_true")
    return parser.parse_args(argv)


def _read_last_live_round(era_csv: Path) -> int | None:
    if not era_csv.exists():
        return None
    df = pl.read_csv(era_csv, try_parse_dates=False)
    live = df.filter(pl.col("dataset") == "live")
    if live.is_empty() or "round_id" not in live.columns:
        return None
    rounds = live.get_column("round_id").drop_nulls().cast(pl.Int64)
    return None if rounds.is_empty() else int(rounds.max())


def _era_range(path: Path) -> tuple[str | None, str | None]:
    """Read (min_era, max_era) from a parquet file; None when unreadable."""
    if not path.exists():
        return None, None
    try:
        agg = (
            pl.scan_parquet(path)
            .select(
                pl.col("era").min().alias("min_era"),
                pl.col("era").max().alias("max_era"),
            )
            .collect()
        )
        return agg.row(0)
    except Exception:
        return None, None


def _validate_and_swap(name: str, tmp: Path, target: Path) -> None:
    """Integrity-check a downloaded temp file, then atomically swap it in."""
    if name == "features.json":
        sets = resolve_feature_sets(tmp)  # raises on malformed/empty feature_sets
        if not sets:
            raise ValueError(f"{name}: feature_sets is empty after validation")
        raw = json.loads(tmp.read_text(encoding="utf-8"))
        if not raw.get("targets"):
            raise ValueError(f"{name}: 'targets' list is missing or empty")
    elif name.endswith(".parquet"):
        pl.scan_parquet(tmp).collect_schema()  # raises on truncated/corrupt parquet
    else:  # example-pred CSV
        if tmp.stat().st_size == 0:
            raise ValueError(f"{name}: downloaded file is empty")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp.replace(target)


def _manifest_to_csv(records: Sequence[dict[str, str | int | None]]) -> str:
    """Serialize manifest rows to the legacy ledger format (CRLF)."""
    out = ["date,dataset,start_era,end_era,round_id"]
    for rec in records:
        start = "X" if rec["start_era"] is None else rec["start_era"]
        end = "X" if rec["end_era"] is None else rec["end_era"]
        round_id = rec["round_id"]
        rid = "" if round_id is None else str(float(round_id))
        out.append(f"{rec['date']},{rec['dataset']},{start},{end},{rid}")
    return "\r\n".join(out) + "\r\n"


def _load_existing_records(
    era_csv: Path, today: str
) -> list[dict[str, str | int | None]]:
    """Existing ledger rows minus today's (they are rebuilt fresh)."""
    if not era_csv.exists():
        return []
    df = pl.read_csv(era_csv, try_parse_dates=False)
    df = df.filter(pl.col("date") != today)
    return [
        {
            "date": str(row["date"]),
            "dataset": str(row["dataset"]),
            "start_era": None if row["start_era"] == "X" else str(row["start_era"]),
            "end_era": None if row["end_era"] == "X" else str(row["end_era"]),
            "round_id": (
                int(float(row["round_id"]))
                if row["round_id"] not in (None, "")
                else None
            ),
        }
        for row in df.iter_rows(named=True)
    ]


def _refresh(
    napi: object,
    args: argparse.Namespace,
    version: str,
    round_num: int,
    plan: dict[str, str],
) -> None:
    version_dir = args.data_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    # clean stale .part files from crashed runs
    for stale in version_dir.glob("*.part"):
        stale.unlink()

    for name, decision in plan.items():
        target = version_dir / name
        if decision == "ensure" and target.exists():
            continue
        if decision == "skip":
            continue
        print(f"downloading {version}/{name} ...")
        fd, tmp_name = tempfile.mkstemp(
            dir=version_dir, prefix=f"{name}.tmp.", suffix=".part"
        )
        os.close(fd)  # Windows: release the handle so os.replace/unlink can work
        tmp = Path(tmp_name)
        try:
            napi.download_dataset(f"{version}/{name}", dest_path=str(tmp))  # type: ignore[attr-defined]
            _validate_and_swap(name, tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink()

    # Ledger: read era ranges for every parquet present on disk (downloaded
    # this run or already present). Write the ledger only when all three
    # exist — a partial checkout (e.g. --live-only) simply skips the write.
    era_ranges: dict[str, tuple[str | None, str | None]] = {}
    for dataset in ("train", "validation", "live"):
        target = version_dir / f"{dataset}.parquet"
        if target.exists():
            era_ranges[dataset] = _era_range(target)
    if set(era_ranges) == {"train", "validation", "live"}:
        records = build_era_manifest(era_ranges, round_num, str(date.today()))
        existing_records = _load_existing_records(args.era_csv, str(date.today()))
        atomic_write_text(args.era_csv, _manifest_to_csv(existing_records + records))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    version = args.version
    napi = numerapi.NumerAPI()

    if args.dry_run:
        print("dry-run: no downloads or writes will be performed")

    # 1. current round (None -> abort)
    round_num = napi.get_current_round()
    if round_num is None:
        print("ERROR: could not determine the current tournament round", file=sys.stderr)
        return 1

    # 2. version alert
    available = napi.list_datasets()
    prefixes = sorted({f.split("/", 1)[0] for f in available})
    newer = detect_newer_version(prefixes, version)
    if newer is not None:
        print(_VERSION_ALERT.format(newer=newer, current=version))
        if args.strict:
            return 3

    # 3. plan
    last_round = _read_last_live_round(args.era_csv)
    live_exists = (args.data_dir / version / "live.parquet").exists()
    round_advanced = needs_live_refresh(round_num, last_round, live_exists)
    version_dir = args.data_dir / version
    if version_dir.exists():
        files = {p.name for p in version_dir.glob("*") if p.is_file()}
        existing = files - {p.name for p in version_dir.glob("*.part")}
    else:
        existing = set()
    plan = classify_refresh_plan(round_advanced, existing, live_only=args.live_only)

    if args.check_only:
        if newer is not None or any(v == "refresh" for v in plan.values()):
            print("check-only: refresh needed (newer version or stale files)")
            return 3
        print("everything current")
        return 0

    if args.dry_run:
        for name, decision in sorted(plan.items()):
            print(f"  {decision:>8}  {name}")
        print("dry-run complete (exit 0)")
        return 0

    # 4. execute
    _refresh(napi, args, version, round_num, plan)
    print(f"refresh complete for round {round_num}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
