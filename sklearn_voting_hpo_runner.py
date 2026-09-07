"""Thin CLI wrapper for VotingRegressor Bayesian optimization."""

# ruff: noqa: E402 — apply thread limits before importing numerical modules.
from nmr.hardware import apply_thread_limits

apply_thread_limits()

from nmr.opt import voting_hpo_main  # noqa: I001 — intentionally after thread limits.

if __name__ == "__main__":
    raise SystemExit(voting_hpo_main())
