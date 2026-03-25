"""CMA-ES optimizer wrapper using pycma.

Covariance Matrix Adaptation Evolution Strategy — the standard
derivative-free optimizer for continuous parameter spaces. Strong
baseline for whole-brain model fitting (Wischnewski et al. 2022).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from neurojax.bench.fitness import FitnessAdapter, FitnessResult
from neurojax.bench.optimizers.base import OptimizationResult


class CMAESWrapper:
    """CMA-ES optimizer via pycma."""

    name: str = "CMA-ES"

    def __init__(
        self,
        sigma0: float = 0.3,
        popsize: Optional[int] = None,
    ):
        """
        Args:
            sigma0: Initial step size (fraction of parameter range).
            popsize: Population size. None = pycma default (4 + 3*ln(N)).
        """
        self.sigma0 = sigma0
        self.popsize = popsize

    def optimize(
        self,
        adapter: FitnessAdapter,
        budget: int,
        seed: int = 0,
    ) -> OptimizationResult:
        import cma

        t0 = time.perf_counter()
        ps = adapter.parameter_space
        param_names = list(ps.keys())
        n_params = len(param_names)

        # Initial point: center of bounds
        x0 = [(lo + hi) / 2 for lo, hi in ps.values()]
        bounds_lo = [lo for lo, _ in ps.values()]
        bounds_hi = [hi for _, hi in ps.values()]

        opts = {
            "bounds": [bounds_lo, bounds_hi],
            "maxfevals": budget,
            "seed": seed,
            "verbose": -9,  # suppress output
        }
        if self.popsize is not None:
            opts["popsize"] = self.popsize

        es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)

        history = []
        total_evals = 0
        best_result: Optional[FitnessResult] = None
        best_params: dict[str, float] = {}
        best_fc = -float("inf")
        gen = 0

        while not es.stop() and total_evals < budget:
            solutions = es.ask()
            n_sols = len(solutions)

            # Evaluate each candidate
            fitnesses = []
            for x in solutions:
                params = {name: float(x[i]) for i, name in enumerate(param_names)}
                result = adapter.evaluate(params)
                # CMA-ES minimizes, so negate FC correlation
                fitnesses.append(-result.fc_correlation)

                if result.fc_correlation > best_fc:
                    best_fc = result.fc_correlation
                    best_result = result
                    best_params = params

            es.tell(solutions, fitnesses)
            total_evals += n_sols
            gen += 1

            history.append({
                "gen": gen,
                "best_fc": best_fc,
                "mean_fc": -float(np.mean(fitnesses)),
                "n_evals": total_evals,
                "sigma": es.sigma,
            })

        wall_time = time.perf_counter() - t0

        if best_result is None:
            best_result = FitnessResult(fc_correlation=0.0, fcd_ks_distance=1.0)

        return OptimizationResult(
            best_params=best_params,
            best_fitness=best_result,
            history=history,
            total_evaluations=total_evals,
            wall_time=wall_time,
            optimizer_name=self.name,
        )
