"""Scenario engine for observed-return capital allocation research.

Answers: "What if I remove model X?" or "What if I weight model Y at 2x?"
Portfolio metrics are computed only from aligned observed per-era payout series.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from nmr.inference import era_series_stats
from nmr.payout import max_drawdown


@dataclass(frozen=True)
class AllocationConstraints:
    """Capital allocation constraints."""

    min_sharpe_threshold: float = 0.5  # Min model Sharpe to include
    max_single_weight: float = 0.25  # No single model > 25%
    max_combined_weight: float = 0.50  # Selected group max 50% total
    min_models: int = 3  # Minimum portfolio size
    max_models: int = 10  # Maximum portfolio size

    # Regime constraints (optional)
    gc_robustness_min: float | None = None  # Min GMC score
    drawdown_max: float | None = 0.20  # Max acceptable drawdown magnitude
    volatility_max: float | None = None  # Max acceptable volatility


@dataclass(frozen=True)
class ModelAllocation:
    """Single model in the allocation."""

    model_id: str
    weight: float  # 0.0 to 1.0
    corr_sharpe_ac: float  # Model's individual Sharpe
    conviction: str  # "high", "medium", "low"
    reason: str  # Why this model was selected


@dataclass(frozen=True)
class AllocationScenario:
    """A complete capital allocation scenario."""

    scenario_id: str
    name: str
    description: str = ""
    allocations: tuple[ModelAllocation, ...] = field(default_factory=tuple)
    constraints: AllocationConstraints = field(default_factory=AllocationConstraints)

    # Computed portfolio metrics
    portfolio_sharpe: float | None = None
    portfolio_volatility: float | None = None
    portfolio_max_drawdown: float | None = None
    portfolio_ic: float | None = None
    avg_correlation: float | None = None
    metrics_reason: str | None = None

    # Comparison to baseline
    sharpe_vs_baseline: float | None = None  # Delta
    sharpe_delta_pct: float | None = None  # % change

    @property
    def total_weight(self) -> float:
        """Sum of all weights (should be ~1.0)."""
        return sum(a.weight for a in self.allocations)

    @property
    def num_models(self) -> int:
        """Number of models in allocation."""
        return len(self.allocations)

    @property
    def is_valid(self) -> bool:
        """Check if allocation meets all constraints."""
        if self.num_models < self.constraints.min_models:
            return False
        if self.num_models > self.constraints.max_models:
            return False
        weights = np.asarray([allocation.weight for allocation in self.allocations])
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            return False
        if not np.isclose(float(np.sum(weights)), 1.0, atol=1e-9):
            return False
        if np.any(weights > self.constraints.max_single_weight):
            return False
        if len(weights) >= 2 and float(np.sum(np.sort(weights)[-2:])) > (
            self.constraints.max_combined_weight + 1e-12
        ):
            return False
        if any(
            not np.isfinite(allocation.corr_sharpe_ac)
            or allocation.corr_sharpe_ac < self.constraints.min_sharpe_threshold
            for allocation in self.allocations
        ):
            return False
        if self.constraints.drawdown_max is not None and (
            self.portfolio_max_drawdown is None
            or not np.isfinite(self.portfolio_max_drawdown)
            or self.portfolio_max_drawdown > self.constraints.drawdown_max
        ):
            return False
        if self.constraints.volatility_max is not None and (
            self.portfolio_volatility is None
            or not np.isfinite(self.portfolio_volatility)
            or self.portfolio_volatility > self.constraints.volatility_max
        ):
            return False
        if self.constraints.gc_robustness_min is not None:
            return False
        return True


class ScenarioEngine:
    """Interactive allocation scenario builder and evaluator."""

    def __init__(self, baseline_models: list[dict]):
        """
        Initialize scenario engine.

        Args:
            baseline_models: List of {model_id, corr_sharpe_ac, avg_correlation, ...}
        """
        self.baseline_models = {m["model_id"]: m for m in baseline_models}
        self.scenarios: dict[str, AllocationScenario] = {}

    def scenario_remove_model(
        self,
        base_scenario: AllocationScenario,
        model_id_to_remove: str,
        rebalance: bool = True,
    ) -> AllocationScenario:
        """
        Create a scenario with one model removed.

        Args:
            base_scenario: Starting scenario
            model_id_to_remove: Model to remove
            rebalance: If True, redistribute removed weight to others

        Returns:
            New scenario with model removed
        """
        new_allocations = tuple(
            replace(a)
            for a in base_scenario.allocations
            if a.model_id != model_id_to_remove
        )

        if rebalance and new_allocations:
            removed_weight = sum(
                a.weight
                for a in base_scenario.allocations
                if a.model_id == model_id_to_remove
            )
            # Proportional rebalance
            total_remaining = sum(a.weight for a in new_allocations)
            new_allocations = tuple(
                replace(
                    allocation,
                    weight=allocation.weight * (1.0 + removed_weight / total_remaining),
                )
                for allocation in new_allocations
            )

        scenario = AllocationScenario(
            scenario_id=f"{base_scenario.scenario_id}__remove_{model_id_to_remove}",
            name=f"{base_scenario.name} (without {model_id_to_remove[:8]})",
            description=f"Removes {model_id_to_remove}; {len(new_allocations)} models remain",
            allocations=new_allocations,
            constraints=base_scenario.constraints,
        )

        baseline = self._compute_portfolio_metrics(base_scenario)
        return self._compute_portfolio_metrics(scenario, baseline)

    def scenario_reweight_model(
        self,
        base_scenario: AllocationScenario,
        model_id: str,
        new_weight: float,
    ) -> AllocationScenario:
        """
        Create a scenario with one model reweighted.

        Args:
            base_scenario: Starting scenario
            model_id: Model to reweight
            new_weight: New weight (0.0 to 1.0)

        Returns:
            New scenario with reweighted model
        """
        copied = tuple(replace(allocation) for allocation in base_scenario.allocations)
        old_weight = 0.0
        for allocation in copied:
            if allocation.model_id == model_id:
                old_weight = allocation.weight
                break
        new_allocations = tuple(
            (
                replace(allocation, weight=new_weight)
                if allocation.model_id == model_id
                else allocation
            )
            for allocation in copied
        )

        # Rebalance others if weight changed
        if old_weight > 0 and new_weight != old_weight:
            delta = new_weight - old_weight
            others = [a for a in new_allocations if a.model_id != model_id]
            if others:
                total_other = sum(a.weight for a in others)
                if total_other > 0:
                    factor = 1.0 - delta / total_other
                    new_allocations = tuple(
                        (
                            allocation
                            if allocation.model_id == model_id
                            else replace(allocation, weight=allocation.weight * factor)
                        )
                        for allocation in new_allocations
                    )

        scenario = AllocationScenario(
            scenario_id=f"{base_scenario.scenario_id}__reweight_{model_id}",
            name=(
                f"{base_scenario.name} (2x {model_id[:8]})"
                if new_weight > old_weight
                else base_scenario.name
            ),
            description=f"Reweights {model_id} to {new_weight:.1%}",
            allocations=new_allocations,
            constraints=base_scenario.constraints,
        )

        baseline = self._compute_portfolio_metrics(base_scenario)
        return self._compute_portfolio_metrics(scenario, baseline)

    def scenario_apply_constraints(
        self,
        base_models: list[dict],
        constraints: AllocationConstraints,
    ) -> AllocationScenario:
        """
        Build an allocation scenario by applying constraints to base models.

        Filters models by Sharpe threshold, limits single weights, etc.

        Args:
            base_models: [{model_id, corr_sharpe_ac, ...}]
            constraints: AllocationConstraints to apply

        Returns:
            New scenario respecting all constraints
        """
        # Filter by Sharpe threshold
        eligible = [
            m
            for m in base_models
            if m.get("corr_sharpe_ac", 0) >= constraints.min_sharpe_threshold
        ]

        if not eligible:
            return AllocationScenario(
                scenario_id="empty",
                name="No eligible models",
                description="No models meet Sharpe threshold",
            )

        # Sort by Sharpe (descending)
        eligible.sort(key=lambda m: m.get("corr_sharpe_ac", 0), reverse=True)

        min_count = max(
            constraints.min_models,
            int(np.ceil(1.0 / constraints.max_single_weight)),
            int(np.ceil(2.0 / constraints.max_combined_weight)),
        )
        selected = eligible[: constraints.max_models]
        if len(selected) < min_count:
            return AllocationScenario(
                scenario_id="infeasible",
                name="Infeasible constraints",
                description="Eligible models cannot satisfy allocation caps",
                constraints=constraints,
            )
        weight = 1.0 / len(selected)
        allocations = [
            ModelAllocation(
                model_id=model["model_id"],
                weight=weight,
                corr_sharpe_ac=float(model.get("corr_sharpe_ac", 0.0)),
                conviction="high" if i < 2 else ("medium" if i < 4 else "low"),
                reason=(
                    f"Ranked #{i + 1} by Sharpe; "
                    f"{float(model.get('corr_sharpe_ac', 0.0)):.3f} Sharpe"
                ),
            )
            for i, model in enumerate(selected)
        ]

        scenario = AllocationScenario(
            scenario_id="constrained",
            name=f"{len(allocations)}-Model Portfolio",
            description=(
                f"Top {len(allocations)} models by Sharpe; respects "
                f"{constraints.max_single_weight:.0%} single limit"
            ),
            allocations=tuple(allocations),
            constraints=constraints,
        )

        return self._compute_portfolio_metrics(scenario)

    def scenario_regime_robust(
        self,
        base_models: list[dict],
        downside_threshold: float = 0.10,
    ) -> AllocationScenario:
        """
        Build regime-robust scenario: prefer models with low drawdown.

        Args:
            base_models: [{model_id, max_drawdown, ...}]
            downside_threshold: Maximum nonnegative drawdown magnitude

        Returns:
            Scenario biased toward drawdown-resistant models
        """
        eligible = [
            model
            for model in base_models
            if 0.0
            <= float(model.get("max_drawdown", float("inf")))
            <= downside_threshold
        ]

        eligible.sort(key=lambda model: float(model["max_drawdown"]))

        allocations = []
        selected = eligible[:5]
        for i, model in enumerate(selected):
            allocations.append(
                ModelAllocation(
                    model_id=model["model_id"],
                    weight=1.0 / len(selected),
                    corr_sharpe_ac=model.get("corr_sharpe_ac", 0),
                    conviction="high",
                    reason=f"Drawdown resilience: {model.get('max_drawdown', 0):.1%} max DD",
                )
            )

        scenario = AllocationScenario(
            scenario_id="regime_robust",
            name="Regime-Robust Portfolio",
            description="Weighted toward models with lowest historical drawdowns",
            allocations=tuple(allocations),
        )

        return self._compute_portfolio_metrics(scenario)

    def _compute_portfolio_metrics(
        self,
        scenario: AllocationScenario,
        baseline: AllocationScenario | None = None,
    ) -> AllocationScenario:
        """Attach metrics computed from aligned observed per-era payouts."""
        if not scenario.allocations:
            return scenario
        series_by_model: dict[str, dict[str, float]] = {}
        for allocation in scenario.allocations:
            model = self.baseline_models.get(allocation.model_id, {})
            series = model.get("payout_by_era")
            if not isinstance(series, dict) or not series:
                return replace(
                    scenario,
                    metrics_reason="observed_payout_series_unavailable",
                )
            series_by_model[allocation.model_id] = {
                str(era): float(value) for era, value in series.items()
            }
        overlap = set.intersection(
            *(set(series) for series in series_by_model.values())
        )
        if len(overlap) < 2:
            return replace(scenario, metrics_reason="insufficient_overlap_eras")
        eras = sorted(overlap, key=int)
        weights = np.asarray(
            [allocation.weight for allocation in scenario.allocations], dtype=float
        )
        matrix = np.column_stack(
            [
                [series_by_model[allocation.model_id][era] for era in eras]
                for allocation in scenario.allocations
            ]
        )
        if not np.isfinite(matrix).all() or not np.isfinite(weights).all():
            raise ValueError("scenario payout series and weights must be finite")
        portfolio = matrix @ weights
        stats = era_series_stats(portfolio)
        average_correlation = None
        if matrix.shape[1] > 1:
            standard_deviations = np.std(matrix, axis=0, ddof=0)
            if np.all(standard_deviations > 0.0):
                correlations = np.corrcoef(matrix, rowvar=False)
                upper = correlations[np.triu_indices_from(correlations, k=1)]
                candidate = float(np.mean(upper))
                if np.isfinite(candidate):
                    average_correlation = candidate
        sharpe_vs_baseline = None
        sharpe_delta_pct = None
        if baseline is not None and baseline.portfolio_sharpe is not None:
            sharpe_vs_baseline = stats.sharpe - baseline.portfolio_sharpe
            sharpe_delta_pct = (
                sharpe_vs_baseline / baseline.portfolio_sharpe * 100
                if baseline.portfolio_sharpe != 0.0
                else None
            )
        return replace(
            scenario,
            portfolio_sharpe=stats.sharpe,
            portfolio_volatility=stats.std,
            portfolio_max_drawdown=max_drawdown(portfolio),
            avg_correlation=average_correlation,
            metrics_reason=None,
            sharpe_vs_baseline=sharpe_vs_baseline,
            sharpe_delta_pct=sharpe_delta_pct,
        )


__all__ = [
    "AllocationConstraints",
    "ModelAllocation",
    "AllocationScenario",
    "ScenarioEngine",
]
