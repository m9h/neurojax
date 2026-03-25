"""Gradient-based optimizer using optax through vbjax autodiff.

Leverages JAX automatic differentiation to compute gradients of the
FC loss w.r.t. model parameters. This is only possible because vbjax
provides a differentiable simulation pipeline — a unique advantage
over neurolib's numpy/numba stack.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import optax

from neurojax.bench.fitness import FitnessAdapter, FitnessResult
from neurojax.bench.optimizers.base import OptimizationResult


class GradientOptimizer:
    """Gradient-based optimizer using optax Adam through vbjax autodiff.

    Requires a VbjaxFitnessAdapter (which provides fc_loss).
    """

    name: str = "Adam-JAX"

    def __init__(
        self,
        learning_rate: float = 1e-3,
        algorithm: str = "adam",
    ):
        self.learning_rate = learning_rate
        self.algorithm = algorithm

    def optimize(
        self,
        adapter: FitnessAdapter,
        budget: int,
        seed: int = 0,
    ) -> OptimizationResult:
        from neurojax.bench.adapters.vbjax_adapter import VbjaxFitnessAdapter

        if not isinstance(adapter, VbjaxFitnessAdapter):
            raise TypeError(
                "GradientOptimizer requires a VbjaxFitnessAdapter "
                "(needs fc_loss for autodiff)"
            )

        t0 = time.perf_counter()
        ps = adapter.parameter_space
        param_names = list(ps.keys())

        # Initialize at center of bounds
        x = jnp.array([(lo + hi) / 2 for lo, hi in ps.values()])
        bounds_lo = jnp.array([lo for lo, _ in ps.values()])
        bounds_hi = jnp.array([hi for _, hi in ps.values()])

        # Setup optax optimizer
        if self.algorithm == "adam":
            opt = optax.adam(self.learning_rate)
        elif self.algorithm == "sgd":
            opt = optax.sgd(self.learning_rate)
        else:
            opt = optax.adam(self.learning_rate)

        opt_state = opt.init(x)

        # Gradient function
        grad_fn = jax.grad(adapter.fc_loss)

        history = []
        best_fc = -float("inf")
        best_params = {}
        best_result = None

        for step in range(budget):
            # Compute gradient
            g = grad_fn(x)

            # Check for NaN gradients
            if not jnp.all(jnp.isfinite(g)):
                break

            # Update
            updates, opt_state = opt.update(g, opt_state)
            x = optax.apply_updates(x, updates)

            # Clip to bounds
            x = jnp.clip(x, bounds_lo, bounds_hi)

            # Evaluate (fc_loss returns negative correlation)
            loss = float(adapter.fc_loss(x))
            fc_corr = -loss

            if fc_corr > best_fc:
                best_fc = fc_corr
                best_params = {
                    name: float(x[i]) for i, name in enumerate(param_names)
                }

            history.append({
                "gen": step + 1,
                "best_fc": best_fc,
                "mean_fc": fc_corr,
                "n_evals": step + 1,
                "loss": loss,
                "grad_norm": float(jnp.linalg.norm(g)),
            })

        # Final evaluation for full FitnessResult
        if best_params:
            best_result = adapter.evaluate(best_params)
        else:
            best_result = FitnessResult(fc_correlation=0.0, fcd_ks_distance=1.0)

        return OptimizationResult(
            best_params=best_params,
            best_fitness=best_result,
            history=history,
            total_evaluations=len(history),
            wall_time=time.perf_counter() - t0,
            optimizer_name=self.name,
        )
