"""Base protocol for optimizer wrappers.

All optimizers target a FitnessAdapter and produce an OptimizationResult
with convergence history for benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from neurojax.bench.fitness import FitnessAdapter, FitnessResult


@dataclass
class OptimizationResult:
    """Result of an optimization run."""

    best_params: dict[str, float]
    best_fitness: FitnessResult
    history: list[dict]  # per-generation stats: {gen, best_fc, mean_fc, n_evals, ...}
    total_evaluations: int
    wall_time: float
    optimizer_name: str
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class Optimizer(Protocol):
    """Protocol for optimizer wrappers."""

    name: str

    def optimize(
        self,
        adapter: FitnessAdapter,
        budget: int,
        seed: int = 0,
    ) -> OptimizationResult:
        """Run optimization for a given evaluation budget.

        Args:
            adapter: FitnessAdapter providing evaluate() and parameter_space.
            budget: Maximum number of fitness evaluations.
            seed: Random seed for reproducibility.

        Returns:
            OptimizationResult with best params, fitness, and convergence history.
        """
        ...
