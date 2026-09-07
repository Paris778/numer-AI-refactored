"""Thin CLI wrapper for HGB Bayesian optimization."""

# ruff: noqa: E402 — apply thread limits before importing numerical modules.
from nmr.hardware import apply_thread_limits

apply_thread_limits()

from nmr.opt import hgb_hpo_main  # noqa: I001 — intentionally after thread limits.

if __name__ == "__main__":
    raise SystemExit(hgb_hpo_main())
