"""BenchmarkRunner — orchestrates multi-optimizer comparison.

Runs multiple optimizers on the same FitnessAdapter with the same
budget and seeds, collects OptimizationResults, and produces
comparison statistics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from neurojax.bench.fitness import FitnessAdapter
from neurojax.bench.optimizers.base import Optimizer, OptimizationResult


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    budgets: list[int] = field(default_factory=lambda: [100])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    name: str = "benchmark"


@dataclass
class ComparisonResult:
    """Result of comparing multiple optimizers."""

    results: dict[str, list[OptimizationResult]]  # optimizer_name -> [per-seed results]
    config: BenchmarkConfig
    wall_time: float

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-optimizer summary statistics."""
        summary = {}
        for opt_name, runs in self.results.items():
            fcs = [r.best_fitness.fc_correlation for r in runs]
            times = [r.wall_time for r in runs]
            evals = [r.total_evaluations for r in runs]
            summary[opt_name] = {
                "fc_mean": float(np.mean(fcs)),
                "fc_std": float(np.std(fcs)),
                "fc_max": float(np.max(fcs)),
                "time_mean": float(np.mean(times)),
                "evals_mean": float(np.mean(evals)),
                "n_runs": len(runs),
            }
        return summary


class BenchmarkRunner:
    """Runs multiple optimizers on the same problem for comparison."""

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or BenchmarkConfig()

    def run(
        self,
        adapter: FitnessAdapter,
        optimizers: list[Optimizer],
        budget: int | None = None,
    ) -> ComparisonResult:
        """Run all optimizers across all seeds.

        Args:
            adapter: FitnessAdapter defining the optimization problem.
            optimizers: List of optimizer wrappers to compare.
            budget: Override budget (uses config.budgets[0] if None).

        Returns:
            ComparisonResult with all runs.
        """
        t0 = time.perf_counter()
        eval_budget = budget or self.config.budgets[0]
        results: dict[str, list[OptimizationResult]] = {}

        for opt in optimizers:
            opt_results = []
            for seed in self.config.seeds:
                result = opt.optimize(adapter, eval_budget, seed=seed)
                opt_results.append(result)
            results[opt.name] = opt_results

        return ComparisonResult(
            results=results,
            config=self.config,
            wall_time=time.perf_counter() - t0,
        )
