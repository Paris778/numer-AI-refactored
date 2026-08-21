"""Live progress monitor for a running experiment campaign (display only).

# TODO : This is a one-off script and should be removed later potentially if it is not needed

Thin control plane: parses the campaign's log/status files and the config's
n_estimators, renders a live per-fold progress view. No business logic — all
model semantics live in nmr/. Stdlib only, read-only, safe to run alongside
the campaign (near-zero CPU).

Usage:
    ./.venv/Scripts/python campaign_status.py                 # live, refresh every 3s
    ./.venv/Scripts/python campaign_status.py --once          # one snapshot
    ./.venv/Scripts/python campaign_status.py --name mt-std-v1 --interval 5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

_FOLD_MARKER = re.compile(
    r"(\d{2}:\d{2}:\d{2}) \| nmr\.models \| INFO \| "
    r"\[train_cross_validation\] (\S+): fold (\d+)/(\d+)"
)
_COMPLETE_MARKER = re.compile(
    r"\[_fit_predict_fold\] (\S+) fold (\d+): fit complete in ([\d.]+)s"
)
_ITER_MARKER = re.compile(r"\[fit\] lightgbm iteration (\d+)")
_N_ESTIMATORS = re.compile(r"n_estimators:\s*(\d+)")

_STANDARD_TREES = 20000  # fallback when the config does not pin n_estimators


def _hhmm_to_seconds(stamp: str) -> int:
    hours, minutes, seconds = (int(part) for part in stamp.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _now_seconds() -> int:
    now = datetime.now()
    return now.hour * 3600 + now.minute * 60 + now.second


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def _bar(frac: float, width: int = 24) -> str:
    filled = max(0, min(width, int(round(frac * width))))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _load(args: argparse.Namespace) -> dict[str, object]:
    """Read the campaign log/status with retries.

    The campaign periodically rewrites its .log and .status files in a
    multi-second pass; a read can catch an empty window. Retry until a
    non-trivial snapshot is obtained (the live loop also keeps the last
    good state, so the display never regresses).
    """
    for _ in range(60):
        state = _load_once(args)
        if state["completed"] or state["last_fold"] or state["iterations"]:
            return state
        time.sleep(0.5)
    return _load_once(args)


def _load_once(args: argparse.Namespace) -> dict[str, object]:
    base = Path(args.artifacts) / "campaigns"
    log_path = base / f"{args.name}.log"
    status_path = base / f"{args.name}.status"

    log = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.exists()
        else ""
    )

    completed: dict[tuple[str, int], float] = {}
    for source in (status_path, None):
        if source is None:
            text = log
        elif source.exists():
            text = source.read_text(encoding="utf-8", errors="replace")
        else:
            continue
        for match in _COMPLETE_MARKER.finditer(text):
            key = (match.group(1), int(match.group(2)))
            completed.setdefault(key, float(match.group(3)))
    completed_list = [
        (target, fold, secs) for (target, fold), secs in sorted(completed.items())
    ]

    total_trees = _STANDARD_TREES
    config_path = Path(args.config)
    if config_path.exists():
        cfg = config_path.read_text(encoding="utf-8", errors="replace")
        found = _N_ESTIMATORS.search(cfg)
        if found:
            total_trees = int(found.group(1))

    last_fold: tuple[str, int, int, int] | None = (
        None  # (target, fold_1based, total, start_secs)
    )
    for match in _FOLD_MARKER.finditer(log):
        start_secs = _hhmm_to_seconds(match.group(1))
        last_fold = (
            match.group(2),
            int(match.group(3)),
            int(match.group(4)),
            start_secs,
        )

    iterations = [int(v) for v in _ITER_MARKER.findall(log)]
    return {
        "completed": completed_list,
        "last_fold": last_fold,
        "iterations": iterations,
        "total_trees": total_trees,
        "log_mtime": log_path.stat().st_mtime if log_path.exists() else None,
        "_base": str(base),
        "_name": args.name,
    }


def _render(state: dict[str, object]) -> str:
    completed: list[tuple[str, int, float]] = state["completed"]
    iterations: list[int] = state["iterations"]
    done_by_target: dict[str, set[int]] = {}
    times_by_target: dict[str, list[float]] = {}
    for target, fold, secs in completed:
        done_by_target.setdefault(target, set()).add(fold)
        times_by_target.setdefault(target, []).append(secs)

    targets = [
        "target_cyrusd_20",
        "target_ender_20",
        "target_jasper_20",
        "target_teager2b_20",
    ]
    last_fold = state["last_fold"]
    current_target = last_fold[0] if last_fold else None
    current_fold = (last_fold[1] - 1) if last_fold else None  # 0-based

    lines = ["=== campaign live status ===", ""]
    if not completed and not last_fold and not iterations:
        lines.append("  no readable snapshot right now.")
        lines.append(
            "  watching: "
            + str(Path(state["_base"]) / f"{state['_name']}.log")
        )
        lines.append(
            "  check that this terminal's working directory is the repo root"
        )
        lines.append("  (C:/dev/numer-AI-refactored) and restart the monitor.")
        return "\n".join(lines)
    for target in targets:
        done = done_by_target.get(target, set())
        if target == current_target and current_fold is not None:
            state_str = "FITTING"
        elif done:
            state_str = "done" if len(done) >= 4 else "partial"
        else:
            state_str = "queued"
        done_cells = " ".join("X" if fold in done else "." for fold in range(4))
        timing = times_by_target.get(target)
        if timing and len(timing) >= 1:
            avg = sum(timing) / len(timing)
            timing_str = f"avg {_fmt_duration(avg)}/fold"
        else:
            timing_str = ""
        lines.append(f"  {target:<20} [{done_cells}] {state_str:<8} {timing_str}")

    lines.append("")
    iterations = state["iterations"]
    if last_fold and iterations:
        total_trees = int(state["total_trees"])
        iters = iterations[-1]
        frac = min(1.0, iters / total_trees)
        elapsed = max(0, _now_seconds() - last_fold[3])
        eta = (elapsed / iters * (total_trees - iters)) if iters else 0.0
        lines.append(
            f"  fitting: {last_fold[0]} fold {last_fold[1]}/{last_fold[2]}  "
            f"{_bar(frac)} {frac * 100:5.1f}%  "
            f"iter {iters}/{total_trees}  elapsed {_fmt_duration(elapsed)}  "
            f"eta {_fmt_duration(eta)}"
        )
    elif last_fold:
        lines.append(
            f"  fitting: {last_fold[0]} fold {last_fold[1]}/{last_fold[2]} (no iteration ticks yet)"
        )
    else:
        lines.append("  no folds started yet")

    total_done = len(completed)
    lines.append("")
    lines.append(f"  folds complete: {total_done}/16")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live campaign progress monitor (read-only)."
    )
    parser.add_argument("--name", default="mt-std-v1")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--config", default="configs/mt-std-v1.yaml")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument(
        "--once", action="store_true", help="print one snapshot and exit"
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    last_mtime = None
    last_state: dict[str, object] | None = None
    while True:
        state = _load(args)
        if (
            not state["completed"]
            and not state["last_fold"]
            and not state["iterations"]
            and last_state is not None
        ):
            # transient empty window in the campaign's log rewrite — keep the
            # last good snapshot instead of regressing the display
            state = last_state
        else:
            last_state = state
        if os.name == "nt":
            os.system("cls")
        else:
            sys.stdout.write("\033[2J\033[H")
        print(_render(state))
        if args.once:
            return 0
        if state["log_mtime"] == last_mtime:
            time.sleep(args.interval)
        else:
            last_mtime = state["log_mtime"]
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
