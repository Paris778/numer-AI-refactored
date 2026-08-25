"""Scenario engine for interactive capital allocation modeling.

Answers: "What if I remove model X?" or "What if I weight model Y at 2x?"
Computes portfolio-level metrics (Sharpe, volatility, drawdown) from model allocations.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AllocationConstraints:
    """Capital allocation constraints."""

    min_sharpe_threshold: float = 0.5  # Min model Sharpe to include
    max_single_weight: float = 0.25  # No single model > 25%
    max_combined_weight: float = 0.50  # Selected group max 50% total
    min_models: int = 3  # Minimum portfolio size
    max_models: int = 10  # Maximum portfolio size

    # Regime constraints (optional)
    gc_robustness_min: float | None = None  # Min GMC score
    drawdown_max: float | None = -0.20  # Max acceptable drawdown
    volatility_max: float | None = None  # Max acceptable volatility


@dataclass
class ModelAllocation:
    """Single model in the allocation."""

    model_id: str
    weight: float  # 0.0 to 1.0
    corr_sharpe_ac: float  # Model's individual Sharpe
    conviction: str  # "high", "medium", "low"
    reason: str  # Why this model was selected


@dataclass
class AllocationScenario:
    """A complete capital allocation scenario."""

    scenario_id: str
    name: str
    description: str = ""
    allocations: list[ModelAllocation] = field(default_factory=list)
    constraints: AllocationConstraints = field(default_factory=AllocationConstraints)

    # Computed portfolio metrics
    portfolio_sharpe: float | None = None
    portfolio_volatility: float | None = None
    portfolio_max_drawdown: float | None = None
    portfolio_ic: float | None = None
    avg_correlation: float | None = None

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
        if any(a.weight > self.constraints.max_single_weight for a in self.allocations):
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
        new_allocations = [
            a for a in base_scenario.allocations if a.model_id != model_id_to_remove
        ]

        if rebalance and new_allocations:
            removed_weight = sum(
                a.weight
                for a in base_scenario.allocations
                if a.model_id == model_id_to_remove
            )
            # Proportional rebalance
            total_remaining = sum(a.weight for a in new_allocations)
            for a in new_allocations:
                a.weight = a.weight * (1.0 + removed_weight / total_remaining)

        scenario = AllocationScenario(
            scenario_id=f"{base_scenario.scenario_id}__remove_{model_id_to_remove}",
            name=f"{base_scenario.name} (without {model_id_to_remove[:8]})",
            description=f"Removes {model_id_to_remove}; {len(new_allocations)} models remain",
            allocations=new_allocations,
            constraints=base_scenario.constraints,
        )

        # Compute portfolio metrics (ensure baseline metrics exist for comparison)
        self._compute_portfolio_metrics(base_scenario)
        self._compute_portfolio_metrics(scenario, base_scenario)

        return scenario

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
        new_allocations = []
        old_weight = 0.0

        for a in base_scenario.allocations:
            if a.model_id == model_id:
                old_weight = a.weight
                a.weight = new_weight
            new_allocations.append(a)

        # Rebalance others if weight changed
        if old_weight > 0 and new_weight != old_weight:
            delta = new_weight - old_weight
            others = [a for a in new_allocations if a.model_id != model_id]
            if others:
                total_other = sum(a.weight for a in others)
                if total_other > 0:
                    for a in others:
                        a.weight = a.weight * (1.0 - delta / total_other)

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

        self._compute_portfolio_metrics(scenario, base_scenario)

        return scenario

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

        # Build allocation respecting limits
        allocations = []
        total_weight = 0.0

        for i, model in enumerate(eligible):
            if len(allocations) >= constraints.max_models:
                break

            # Max single weight
            max_w = min(
                constraints.max_single_weight,
                (1.0 - total_weight) * 0.5,  # Leave room for diversity
            )

            # Equal weight within limit (can adjust to Sharpe-weighted)
            weight = max_w if total_weight + max_w <= 1.0 else (1.0 - total_weight)

            if weight > 0:
                allocations.append(
                    ModelAllocation(
                        model_id=model["model_id"],
                        weight=weight,
                        corr_sharpe_ac=model.get("corr_sharpe_ac", 0),
                        conviction="high" if i < 2 else ("medium" if i < 4 else "low"),
                        reason=f"Ranked #{i+1} by Sharpe; {model.get('corr_sharpe_ac', 0):.3f} Sharpe",
                    )
                )
                total_weight += weight

        # Normalize weights to sum to 1.0
        if allocations and total_weight > 0:
            for a in allocations:
                a.weight = a.weight / total_weight

        scenario = AllocationScenario(
            scenario_id="constrained",
            name=f"{len(allocations)}-Model Portfolio",
            description=(
                f"Top {len(allocations)} models by Sharpe; respects "
                f"{constraints.max_single_weight:.0%} single limit"
            ),
            allocations=allocations,
            constraints=constraints,
        )

        self._compute_portfolio_metrics(scenario)

        return scenario

    def scenario_regime_robust(
        self,
        base_models: list[dict],
        downside_threshold: float = -0.10,
    ) -> AllocationScenario:
        """
        Build regime-robust scenario: prefer models with low drawdown.

        Args:
            base_models: [{model_id, max_drawdown, ...}]
            downside_threshold: Only include models with drawdown > threshold

        Returns:
            Scenario biased toward drawdown-resistant models
        """
        # Filter by drawdown
        eligible = [
            m for m in base_models if m.get("max_drawdown", 0) > downside_threshold
        ]

        # Sort by max_drawdown (least negative = best)
        eligible.sort(key=lambda m: m.get("max_drawdown", 0), reverse=True)

        allocations = []
        for i, model in enumerate(eligible[:5]):  # Top 5 by drawdown resilience
            allocations.append(
                ModelAllocation(
                    model_id=model["model_id"],
                    weight=0.20,  # Equal weight
                    corr_sharpe_ac=model.get("corr_sharpe_ac", 0),
                    conviction="high",
                    reason=f"Drawdown resilience: {model.get('max_drawdown', 0):.1%} max DD",
                )
            )

        scenario = AllocationScenario(
            scenario_id="regime_robust",
            name="Regime-Robust Portfolio",
            description="Weighted toward models with lowest historical drawdowns",
            allocations=allocations,
        )

        self._compute_portfolio_metrics(scenario)

        return scenario

    def _compute_portfolio_metrics(
        self,
        scenario: AllocationScenario,
        baseline: AllocationScenario | None = None,
    ) -> None:
        """Compute portfolio-level metrics."""
        if not scenario.allocations:
            return

        # Simplified metrics (would use real payout timeseries in production)
        weights = np.array([a.weight for a in scenario.allocations])
        sharpes = np.array([a.corr_sharpe_ac for a in scenario.allocations])

        # Weighted average Sharpe
        scenario.portfolio_sharpe = float(np.dot(weights, sharpes))

        # Assume portfolio volatility decreases with diversification
        scenario.portfolio_volatility = 0.15 * (1.0 - 0.1 * len(scenario.allocations))

        # Assume drawdown decreases with more models
        scenario.portfolio_max_drawdown = -0.25 * (
            1.0 - 0.05 * len(scenario.allocations)
        )

        # Average correlation (simplified)
        if len(scenario.allocations) > 1:
            scenario.avg_correlation = 0.5 - 0.05 * len(scenario.allocations)

        if baseline and baseline.portfolio_sharpe:
            scenario.sharpe_vs_baseline = (
                scenario.portfolio_sharpe - baseline.portfolio_sharpe
            )
            scenario.sharpe_delta_pct = (
                scenario.sharpe_vs_baseline / baseline.portfolio_sharpe * 100
                if baseline.portfolio_sharpe > 0
                else 0
            )


__all__ = [
    "AllocationConstraints",
    "ModelAllocation",
    "AllocationScenario",
    "ScenarioEngine",
]
