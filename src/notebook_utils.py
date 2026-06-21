from __future__ import annotations

from IPython.display import Markdown, display

from src.runner import PromotionRunResult


def render_promotion_report(result: PromotionRunResult) -> None:
    """Render a concise promotion tear sheet for notebook workflows."""
    status_label = "PASS" if result.smoke_test_passed else "FAIL"
    smoke_label = "PASS (Exact Match)" if result.smoke_test_passed else "FAIL"
    mean_fnc = _format_optional(result.evaluation_summary.mean_fnc, precision=5)
    max_exposure = _format_optional(
        result.evaluation_summary.max_feature_exposure,
        precision=5,
    )

    report = [
        f"### Promotion Run: {status_label}",
        "---",
        "#### 1. Out-of-Fold Evaluation (Custom Backend)",
        f"- **Mean CORR**: `{result.evaluation_summary.mean_corr:.5f}`",
        f"- **Sharpe**: `{result.evaluation_summary.sharpe_corr:.3f}`",
        f"- **Max Drawdown**: `{result.evaluation_summary.max_drawdown_corr:.4f}`",
        f"- **Mean FNC**: `{mean_fnc}`",
        f"- **Max Feature Exposure**: `{max_exposure}`",
        "#### 2. Institutional Gates",
        f"- **Fast-Fail Gate**: `{'PASS' if result.gate_result.passed else 'FAIL'}`",
        f"- **Oracle Parity** (Eras: {', '.join(result.parity_eras)}): `PASS`",
        f"- **Stress Test Degradation**: `{result.stress_result.degradation_pct:.2f}%`",
        "#### 3. Deployment Artifact",
        f"- **Path**: `{result.payload_path}`",
        f"- **Smoke Test**: `{smoke_label}`",
        "---",
        "<details><summary><b>Execution Logs (Expand)</b></summary>",
        "",
        "```text",
    ]
    report.extend(result.log_lines)
    report.extend(["```", "", "</details>"])
    display(Markdown("\n".join(report)))


def _format_optional(value: float | None, *, precision: int) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"
