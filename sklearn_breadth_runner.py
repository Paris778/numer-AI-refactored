"""Thin CLI wrapper for the sklearn breadth benchmark."""

# ruff: noqa: E402 — apply thread limits before importing numerical modules.
from nmr.hardware import apply_thread_limits

apply_thread_limits()

from nmr.sklearn_breadth import main  # noqa: I001 — intentionally after thread limits.

if __name__ == "__main__":
    raise SystemExit(main())
