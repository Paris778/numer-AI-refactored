"""Unified, cached data layer for all dashboard formats (Streamlit, HTML, REST API).

Single source of truth for:
- Leaderboard (trained runs + benchmarks)
- Campaigns
- Registry entries (for fleet summary)
- Gates + status badges
- Timeseries metrics
- Similarity matrices
- Full-version manifests

All methods return strongly-typed Pydantic models. Caching via @st.cache_data
(Streamlit) or manual (non-Streamlit). Mtime-based invalidation.

NOTE: This module is presentation-only; all business logic lives in nmr.*.
The service is a thin data aggregation + formatting layer.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, Field

from nmr.config import REPO_ROOT
from nmr.dashboard import load_benchmark_frame, load_unified_leaderboard

logger = logging.getLogger(__name__)


def _profile_label(model_id: str) -> str | None:
    """Best-effort human-readable label from the explainer catalog.

    Lazily imports ``nmr.explainers`` to avoid a circular import
    (``nmr.explainers`` imports this service for dynamic profiles).
    Returns None when no profile exists (caller falls back to the raw ID).
    """
    try:
        from nmr.explainers import get_model_profile

        profile = get_model_profile(model_id)
        return profile.summary if profile else None
    except Exception:
        return None


# ============================================================================
# PYDANTIC MODELS — Type-safe return contracts
# ============================================================================


class LeaderboardRowModel(BaseModel):
    """Single row in the leaderboard (trained run or benchmark)."""

    model_id: str
    source: str  # "trained" | "trained_legacy" | "full" | "benchmark"
    family: str | None = None
    training_scope: str | None = None
    has_full_version: bool = False
    run_name: str
    run_dir: str
    backend: str | None = None
    preset: str | None = None
    feature_set: str | None = None
    feature_subset: str | None = None
    n_targets: int | None = None
    targets: str | None = None
    neutralization_proportion: float | None = None
    oof_device: str | None = None

    # Core metrics
    corr: float | None = None
    corr_ci_low: float | None = None
    corr_ci_high: float | None = None
    corr_sharpe_ac: float | None = None
    corr_sharpe_ac_ci_low: float | None = None
    corr_sharpe_ac_ci_high: float | None = None
    max_drawdown: float | None = None
    std_corr: float | None = None
    deflated_sharpe: float | None = None
    max_feature_exposure: float | None = None

    # Robustness flags
    has_bmc: bool | None = None
    has_horizon: bool | None = None
    has_perturb: bool | None = None
    has_regime: bool | None = None

    # Capital readiness
    status: str | None = None  # "CHAMPION" | "CAPITAL READY" | "RESEARCH" | etc.


class LeaderboardFrame(BaseModel):
    """Complete leaderboard: all trained runs + benchmarks."""

    rows: list[LeaderboardRowModel]
    total_rows: int
    evaluable_rows: int
    n_overlap_eras: int | None = None
    data_version: str = "v5.3"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def __len__(self) -> int:
        return len(self.rows)

    def filter_by_source(self, sources: list[str]) -> LeaderboardFrame:
        """Filter rows by source."""
        filtered = [r for r in self.rows if r.source in sources]
        return LeaderboardFrame(
            rows=filtered,
            total_rows=len(filtered),
            evaluable_rows=sum(
                1 for r in filtered if r.source in ["trained", "trained_legacy"]
            ),
            n_overlap_eras=self.n_overlap_eras,
            data_version=self.data_version,
        )

    def filter_by_backend(self, backends: list[str]) -> LeaderboardFrame:
        """Filter rows by backend."""
        filtered = [r for r in self.rows if r.backend in backends]
        return LeaderboardFrame(
            rows=filtered,
            total_rows=len(filtered),
            evaluable_rows=sum(
                1 for r in filtered if r.source in ["trained", "trained_legacy"]
            ),
            n_overlap_eras=self.n_overlap_eras,
            data_version=self.data_version,
        )

    def filter_by_preset(self, presets: list[str]) -> LeaderboardFrame:
        """Filter rows by preset."""
        filtered = [r for r in self.rows if r.preset in presets]
        return LeaderboardFrame(
            rows=filtered,
            total_rows=len(filtered),
            evaluable_rows=sum(
                1 for r in filtered if r.source in ["trained", "trained_legacy"]
            ),
            n_overlap_eras=self.n_overlap_eras,
            data_version=self.data_version,
        )

    def sort_by_metric(
        self, metric: str = "corr_sharpe_ac", descending: bool = True
    ) -> LeaderboardFrame:
        """Sort rows by metric."""
        sorted_rows = sorted(
            self.rows,
            key=lambda r: getattr(r, metric)
            or (float("-inf") if descending else float("inf")),
            reverse=descending,
        )
        return LeaderboardFrame(
            rows=sorted_rows,
            total_rows=len(sorted_rows),
            evaluable_rows=self.evaluable_rows,
            n_overlap_eras=self.n_overlap_eras,
            data_version=self.data_version,
        )


class KPISnapshot(BaseModel):
    """Key performance indicators for the dashboard."""

    champion_label: str
    champion_detail: str
    top_contender_label: str
    top_contender_sharpe: float | None = None
    hurdle_sharpe: float
    gap: float | None = None
    fleet_best_cagr: float | None = None
    worst_drawdown: float | None = None
    capital_ready_count: int
    fleet_count: int
    data_version: str = "v5.3"
    n_eras: int | None = None


class TopPerformerRowModel(BaseModel):
    """A single ranked model in the top-performers view.

    Carries the metrics a capital allocator needs to manually decide how much
    to invest in each model: risk-adjusted return (Sharpe), raw CORR with CI,
    deflated Sharpe (probability of genuine edge), era-to-era stability,
    maximum drawdown, and robustness flags.
    """

    rank: int
    model_id: str
    run_name: str
    label: str  # human-readable (backend · preset or run name)
    backend: str | None = None
    preset: str | None = None
    feature_set: str | None = None
    corr: float | None = None
    corr_ci_low: float | None = None
    corr_ci_high: float | None = None
    corr_sharpe_ac: float | None = None
    corr_sharpe_ac_ci_low: float | None = None
    corr_sharpe_ac_ci_high: float | None = None
    std_corr: float | None = None  # lower = more era-stable
    deflated_sharpe: float | None = None  # prob. genuine edge (higher=better)
    max_drawdown: float | None = None
    has_bmc: bool | None = None
    has_horizon: bool | None = None
    has_perturb: bool | None = None
    has_regime: bool | None = None
    robustness_score: int = 0  # count of robustness checks available


class TopPerformersResult(BaseModel):
    """Ranked list of top performers for manual capital decisions."""

    rows: list[TopPerformerRowModel]
    sort_metric: str
    total_considered: int
    champion: TopPerformerRowModel | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def __len__(self) -> int:
        return len(self.rows)


class CampaignRun(BaseModel):
    """Single run within a campaign."""

    campaign_id: str
    name: str | None = None
    config_path: str | None = None
    run_id: str | None = None
    status: str | None = None
    error: str | None = None


class CampaignLog(BaseModel):
    """Collection of campaign runs."""

    runs: list[CampaignRun]
    total_campaigns: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RobustnessMatrixRow(BaseModel):
    """Single row in robustness matrix."""

    model_id: str
    has_bmc: bool | None = None
    has_horizon: bool | None = None
    has_perturb: bool | None = None
    has_regime: bool | None = None
    max_feature_exposure: float | None = None
    std_corr: float | None = None
    max_drawdown: float | None = None


class RobustnessMatrix(BaseModel):
    """Robustness metrics for heatmap."""

    rows: list[RobustnessMatrixRow]
    boolean_cells: list[str] = Field(
        default=["has_bmc", "has_horizon", "has_perturb", "has_regime"]
    )
    numeric_cells: list[str] = Field(
        default=["max_feature_exposure", "std_corr", "max_drawdown"]
    )


# ============================================================================
# DASHBOARD DATA SERVICE — Unified data aggregation
# ============================================================================


class DashboardDataService:
    """Unified, cached data layer for all dashboard formats.

    All methods return strongly-typed Pydantic models. The service is thin:
    it loads data from nmr.* and formats it; no business logic here.

    Caching:
    - For Streamlit: use @st.cache_data decorator
    - For non-Streamlit: manual LRU cache + mtime sentinel
    - Invalidate on file mtime changes in registry_dir, benchmark_path, data_dir
    """

    def __init__(
        self,
        registry_dir: Path | None = None,
        benchmark_path: Path | None = None,
        data_dir: Path | None = None,
    ):
        """Initialize data service.

        Args:
            registry_dir: Path to artifacts/registry (default: REPO_ROOT / artifacts / registry)
            benchmark_path: Path to benchmark CSV (default: auto-resolve)
            data_dir: Path to data/ directory (default: REPO_ROOT / data)
        """
        self.registry_dir = (
            Path(registry_dir) if registry_dir else REPO_ROOT / "artifacts" / "registry"
        )
        self.benchmark_path = (
            Path(benchmark_path) if benchmark_path else self._resolve_benchmark_path()
        )
        # Timeseries extraction expects the versioned data folder (validation.parquet
        # + meta_model.parquet live under data/v5.3), not the bare data/ root.
        self.data_dir = Path(data_dir) if data_dir else REPO_ROOT / "data" / "v5.3"

        self._mtime_registry: float | None = None
        self._mtime_benchmark: float | None = None
        self._leaderboard_cache: LeaderboardFrame | None = None
        self._timeseries_cache: dict[tuple[str, ...], dict[str, Any]] = {}
        self._full_history_cache: dict[tuple[str, ...], dict[str, Any]] = {}

    def _resolve_benchmark_path(self) -> Path | None:
        """Resolve benchmark CSV path; default to hierarchy scorecard."""
        default_path = (
            REPO_ROOT / "artifacts" / "reports" / "benchmark_hierarchy_scorecard.csv"
        )
        if default_path.exists():
            return default_path
        return None

    def _check_cache_valid(self) -> bool:
        """Check if cached data is still valid (mtime unchanged)."""
        if self._leaderboard_cache is None:
            return False
        if not self.registry_dir.exists():
            return False

        current_mtime = self.registry_dir.stat().st_mtime
        if self._mtime_registry is None:
            return False

        # Simple heuristic: if registry mtime changed, invalidate
        if current_mtime != self._mtime_registry:
            return False

        return True

    def load_leaderboard(self) -> LeaderboardFrame:
        """Load and merge registry + benchmark runs into unified leaderboard.

        Returns:
            LeaderboardFrame: Merged, typed leaderboard.

        Cache:
            Invalidate on mtime change in registry_dir or benchmark_path.
        """
        # Return cached if valid
        if self._check_cache_valid() and self._leaderboard_cache:
            logger.debug("Using cached leaderboard")
            return self._leaderboard_cache

        # Update mtime sentinel
        if self.registry_dir.exists():
            self._mtime_registry = self.registry_dir.stat().st_mtime

        # Load from nmr.dashboard (engine)
        try:
            # Load registry runs
            registry_frame = load_unified_leaderboard(
                self.registry_dir, benchmark_path=False, models_dir=None
            )
            registry_frame = registry_frame.filter(
                pl.col("source").is_in(["trained", "trained_legacy", "full"])
            )

            # Load benchmarks if available
            benchmark_frame = (
                pl.DataFrame()
                if not self.benchmark_path
                else load_benchmark_frame(self.benchmark_path)
            )

            # Merge
            if registry_frame.height == 0 and benchmark_frame.height == 0:
                rows = []
            elif registry_frame.height == 0:
                rows = self._polars_to_pydantic_rows(benchmark_frame)
            elif benchmark_frame.height == 0:
                rows = self._polars_to_pydantic_rows(registry_frame)
            else:
                merged = pl.concat([registry_frame, benchmark_frame], how="vertical")
                rows = self._polars_to_pydantic_rows(merged)

            # Count evaluable rows (trained only, not full-version)
            evaluable_count = sum(
                1 for r in rows if r.source in ["trained", "trained_legacy"]
            )

            # Get n_overlap_eras from any row
            n_eras = None
            for row in rows:
                if row.n_targets is not None:
                    n_eras = row.n_targets
                    break

            # Build result
            leaderboard = LeaderboardFrame(
                rows=rows,
                total_rows=len(rows),
                evaluable_rows=evaluable_count,
                n_overlap_eras=n_eras,
                data_version="v5.3",
            )

            # Cache
            self._leaderboard_cache = leaderboard
            logger.info(
                f"Loaded leaderboard: {len(rows)} rows, {evaluable_count} evaluable"
            )
            return leaderboard

        except Exception as e:
            logger.error(f"Failed to load leaderboard: {e}")
            return LeaderboardFrame(rows=[], total_rows=0, evaluable_rows=0)

    def _polars_to_pydantic_rows(
        self, frame: pl.DataFrame
    ) -> list[LeaderboardRowModel]:
        """Convert Polars frame to Pydantic model rows."""
        rows = []
        for row_dict in frame.to_dicts():
            try:
                rows.append(LeaderboardRowModel(**row_dict))
            except Exception as e:
                logger.warning(f"Failed to convert row: {e}")
                continue
        return rows

    def load_campaigns(self) -> CampaignLog:
        """Load campaign logs from artifacts/campaigns/*.json.

        Returns:
            CampaignLog: Flattened campaign runs.
        """
        campaigns_dir = REPO_ROOT / "artifacts" / "campaigns"
        runs = []

        if not campaigns_dir.is_dir():
            return CampaignLog(runs=[], total_campaigns=0)

        campaign_ids = set()
        for log_file in sorted(campaigns_dir.glob("*.json")):
            try:
                payload = json.loads(log_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Corrupt campaign log {log_file}: {e}")
                continue

            campaign_id = payload.get("campaign_id") or log_file.stem
            campaign_ids.add(campaign_id)
            name = payload.get("name")

            for run in payload.get("runs") or []:
                runs.append(
                    CampaignRun(
                        campaign_id=campaign_id,
                        name=name,
                        config_path=run.get("config_path"),
                        run_id=run.get("run_id"),
                        status=run.get("status"),
                        error=run.get("error"),
                    )
                )

        return CampaignLog(runs=runs, total_campaigns=len(campaign_ids))

    def load_registry_entries(self) -> list[dict]:
        """Load raw registry run.json payloads for fleet_summary() analysis.

        Returns:
            List of run manifests (passed to nmr.meta.fleet_summary).
        """
        entries = []
        for run_file in sorted(self.registry_dir.glob("*/run.json")):
            try:
                payload = json.loads(run_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Corrupt run.json {run_file}: {e}")
                continue

            if isinstance(payload, dict) and payload.get("run_id"):
                entries.append(payload)

        return entries

    def compute_robustness_matrix(self) -> RobustnessMatrix:
        """Extract robustness cells for heatmap.

        Returns:
            RobustnessMatrix: Boolean + numeric robustness metrics.
        """
        leaderboard = self.load_leaderboard()

        # Filter to evaluable rows only
        evaluable_rows = [
            r for r in leaderboard.rows if r.source in ["trained", "trained_legacy"]
        ]

        # Map to RobustnessMatrixRow
        rows = [
            RobustnessMatrixRow(
                model_id=row.model_id,
                has_bmc=row.has_bmc,
                has_horizon=row.has_horizon,
                has_perturb=row.has_perturb,
                has_regime=row.has_regime,
                max_feature_exposure=row.max_feature_exposure,
                std_corr=row.std_corr,
                max_drawdown=row.max_drawdown,
            )
            for row in evaluable_rows
        ]

        return RobustnessMatrix(rows=rows)

    def format_model_label(
        self, source: str, run_name: str, model_id: str | None, short_id_len: int = 8
    ) -> str:
        """Format a human-readable model label.

        Args:
            source: "trained" | "benchmark" | "full"
            run_name: Config name or strategy name
            model_id: Model UUID or strategy ID
            short_id_len: Truncate model_id to this length for trained runs

        Returns:
            Formatted label (e.g., "config-name · model_id")
        """
        model_id = model_id or "?"
        if source == "benchmark":
            return f"{run_name} · {model_id}"
        return f"{run_name} · {model_id[:short_id_len]}"

    def prepare_leaderboard_for_display(
        self,
        metric: str = "corr_sharpe_ac",
        sort_desc: bool = True,
        champion_id: str | None = None,
    ) -> list[dict]:
        """Prepare leaderboard rows for display (bar chart, table).

        Adds computed fields: champion flag, CI deltas, sort order.

        Args:
            metric: Metric to sort by
            sort_desc: Sort descending
            champion_id: Highlight this model as champion

        Returns:
            List of dicts ready for rendering (with added fields).
        """
        leaderboard = self.load_leaderboard()

        # Filter to evaluable rows
        evaluable_rows = [
            r for r in leaderboard.rows if r.source in ["trained", "trained_legacy"]
        ]

        # Sort
        sorted_rows = sorted(
            evaluable_rows,
            key=lambda r: getattr(r, metric)
            or (float("-inf") if sort_desc else float("inf")),
            reverse=sort_desc,
        )

        # Build display rows
        display_rows = []
        for row in sorted_rows:
            ci_low = getattr(row, f"{metric}_ci_low", None)
            ci_high = getattr(row, f"{metric}_ci_high", None)
            value = getattr(row, metric, None)

            ci_plus = (ci_high - value) if (ci_high and value) else None
            ci_minus = (value - ci_low) if (ci_low and value) else None

            display_rows.append(
                {
                    "model_id": row.model_id,
                    "label": self.format_model_label(
                        row.source, row.run_name, row.model_id
                    ),
                    "source": row.source,
                    "run_name": row.run_name,
                    "backend": row.backend,
                    "preset": row.preset,
                    "metric_value": value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_plus": ci_plus,
                    "ci_minus": ci_minus,
                    "champion": row.model_id == champion_id,
                    "status": getattr(row, "status", None),
                    # Copy all original fields for table rendering
                    **row.model_dump(),
                }
            )

        return display_rows

    def compute_top_performers(
        self,
        sort_metric: str = "corr_sharpe_ac",
        top_n: int = 10,
        min_sharpe: float | None = None,
        sources: list[str] | None = None,
    ) -> TopPerformersResult:
        """Rank the best models for manual capital allocation decisions.

        Ranks evaluable (validation-scored) models by ``sort_metric``, then
        decorates each row with the full decision surface: raw CORR + CI,
        risk-adjusted Sharpe + CI, deflated Sharpe, era-stability (std_corr),
        max drawdown, and a robustness score (count of available checks).

        Args:
            sort_metric: Metric to rank by (default corr_sharpe_ac).
            top_n: How many top models to return.
            min_sharpe: Optional floor on corr_sharpe_ac to exclude weak models.
            sources: Optional source allowlist (default trained/trained_legacy).

        Returns:
            TopPerformersResult with ranked, strongly-typed rows.
        """
        leaderboard = self.load_leaderboard()
        allow = sources or ["trained", "trained_legacy"]
        evaluable = [r for r in leaderboard.rows if r.source in allow]

        # Optional Sharpe floor
        if min_sharpe is not None:
            evaluable = [
                r
                for r in evaluable
                if r.corr_sharpe_ac is not None and r.corr_sharpe_ac >= min_sharpe
            ]

        # Sort by chosen metric (None sorts last)
        ranked = sorted(
            evaluable,
            key=lambda r: getattr(r, sort_metric)
            or (float("-inf") if sort_metric else float("inf")),
            reverse=True,
        )[:top_n]

        rows: list[TopPerformerRowModel] = []
        for i, row in enumerate(ranked, start=1):
            robustness_score = sum(
                1
                for flag in ("has_bmc", "has_horizon", "has_perturb", "has_regime")
                if getattr(row, flag) is True
            )
            rows.append(
                TopPerformerRowModel(
                    rank=i,
                    model_id=row.model_id,
                    run_name=row.run_name,
                    label=self.format_model_label(
                        row.source, row.run_name, row.model_id
                    ),
                    backend=row.backend,
                    preset=row.preset,
                    feature_set=row.feature_set,
                    corr=row.corr,
                    corr_ci_low=row.corr_ci_low,
                    corr_ci_high=row.corr_ci_high,
                    corr_sharpe_ac=row.corr_sharpe_ac,
                    corr_sharpe_ac_ci_low=row.corr_sharpe_ac_ci_low,
                    corr_sharpe_ac_ci_high=row.corr_sharpe_ac_ci_high,
                    std_corr=row.std_corr,
                    deflated_sharpe=row.deflated_sharpe,
                    max_drawdown=row.max_drawdown,
                    has_bmc=row.has_bmc,
                    has_horizon=row.has_horizon,
                    has_perturb=row.has_perturb,
                    has_regime=row.has_regime,
                    robustness_score=robustness_score,
                )
            )

        champion = rows[0] if rows else None
        return TopPerformersResult(
            rows=rows,
            sort_metric=sort_metric,
            total_considered=len(evaluable),
            champion=champion,
        )

    def load_timeseries(self, run_ids: list[str]) -> dict[str, Any]:
        """Load real per-era timeseries for the given run IDs.

        Wraps ``nmr.dashboard.extract_multimetric_timeseries`` (the analytical
        engine) with memoization keyed on the exact run-id tuple. Returns the
        vanilla payload: ``{eras, meta_downside_mask, metrics, drawdowns}``.

        Args:
            run_ids: Model/run IDs to extract timeseries for.

        Returns:
            The timeseries payload dict (see nmr.dashboard for schema).
        """
        key = tuple(sorted(run_ids))
        if key in self._timeseries_cache:
            return self._timeseries_cache[key]

        from nmr.dashboard import extract_multimetric_timeseries

        payload = extract_multimetric_timeseries(
            registry_dir=self.registry_dir,
            data_dir=self.data_dir,
            run_ids=list(key),
            include_tier4_ref=True,
        )
        self._timeseries_cache[key] = payload
        return payload

    def load_full_history(self, run_ids: list[str]) -> dict[str, Any]:
        """Per-era CORR over each model's FULL validation history.

        The meta-anchored view (``load_timeseries``) is capped at the ~86-era
        meta-model window because MMC / downside classification need the meta
        model. But each trained model's ``validation_preds.parquet`` spans the
        full validation window (~600+ eras), so its own per-era CORR — and the
        cumulative / drawdown / streak statistics that follow — can be computed
        over the ENTIRE history. This method does exactly that, using the
        oracle-parity ``nmr.evaluation.EvaluationEngine.per_era_corr``.

        Memoized on the run-id tuple. Returns::

            {
              "series":   {model_id: {"eras": [...], "standard": [...],
                                      "cumulative": [...], "label": ...}},
              "drawdowns":{model_id: [...]},
              "stats":    {model_id: {"n": int, "mean_corr": float,
                                      "std_corr": float, "corr_sharpe": float,
                                      "max_drawdown": float,
                                      "pct_positive": float, "win_streak": int}},
            }

        Args:
            run_ids: Model/run IDs to extract full-history CORR for.

        Returns:
            Full-history timeseries payload (never raises on missing preds —
            missing models are simply absent from the result).
        """
        key = tuple(sorted(run_ids))
        if key in self._full_history_cache:
            return self._full_history_cache[key]

        import numpy as np

        from nmr.evaluation import EvaluationEngine

        engine = EvaluationEngine()
        targets = pl.read_parquet(
            self.data_dir / "validation.parquet", columns=["era", "id", "target"]
        )

        series: dict[str, Any] = {}
        drawdowns: dict[str, list[float]] = {}
        stats: dict[str, dict[str, Any]] = {}

        for model_id in key:
            preds_path = self.registry_dir / model_id / "validation_preds.parquet"
            if not preds_path.exists():
                logger.warning("dashboard_ui.service: missing preds %s", preds_path)
                continue

            preds = pl.read_parquet(preds_path, columns=["era", "id", "prediction"])
            joined = preds.join(targets, on=["era", "id"], how="inner")
            corr = engine.per_era_corr(
                joined, pred_col="prediction", target_col="target"
            )
            eras = sorted(corr.keys())
            vals = np.array([corr[e] for e in eras], dtype=float)
            if vals.size == 0:
                continue

            cum = np.cumsum(vals)
            peak = np.maximum.accumulate(cum)
            dd = cum - peak  # drawdown of cumulative CORR (<= 0)

            pos = vals > 0.0
            streak = 0
            best = 0
            for flag in pos:
                streak = streak + 1 if flag else 0
                best = max(best, streak)

            label = _profile_label(model_id) or f"{model_id[:16]}"

            series[model_id] = {
                "eras": [str(e) for e in eras],
                "standard": [float(v) for v in vals],
                "cumulative": [float(v) for v in cum],
                "label": label,
            }
            drawdowns[model_id] = [float(v) for v in dd]
            stats[model_id] = {
                "n": int(vals.size),
                "mean_corr": float(vals.mean()),
                "std_corr": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "corr_sharpe": (
                    float(vals.mean() / vals.std(ddof=1)) if vals.size > 1 else 0.0
                ),
                "max_drawdown": float(np.min(dd)),
                "pct_positive": float((vals > 0.0).mean()),
                "win_streak": int(best),
            }

        payload = {"series": series, "drawdowns": drawdowns, "stats": stats}
        self._full_history_cache[key] = payload
        return payload


# ============================================================================
# STREAMLIT INTEGRATION (Optional)
# ============================================================================

# If Streamlit is available, provide cached versions
try:
    import streamlit as st

    @st.cache_data
    def get_dashboard_service(
        registry_dir: Path | str | None = None,
        benchmark_path: Path | str | None = None,
    ) -> DashboardDataService:
        """Get or create a cached DashboardDataService (Streamlit singleton)."""
        return DashboardDataService(
            registry_dir=registry_dir,
            benchmark_path=benchmark_path,
        )

except ImportError:
    # Streamlit not available; users should cache manually
    def get_dashboard_service(
        registry_dir: Path | str | None = None,
        benchmark_path: Path | str | None = None,
    ) -> DashboardDataService:
        """Non-Streamlit version (no automatic caching)."""
        return DashboardDataService(
            registry_dir=registry_dir,
            benchmark_path=benchmark_path,
        )


__all__ = [
    "DashboardDataService",
    "get_dashboard_service",
    "LeaderboardFrame",
    "LeaderboardRowModel",
    "KPISnapshot",
    "CampaignLog",
    "CampaignRun",
    "RobustnessMatrix",
    "RobustnessMatrixRow",
]
