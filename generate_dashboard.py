"""Thin entry point — all logic lives in dashboard_ui.report."""

from dashboard_ui.report import generate_dashboard, main

__all__ = ["generate_dashboard", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
