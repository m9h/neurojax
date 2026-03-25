"""Gradient-based optimizer using optax through vbjax autodiff.

Leverages JAX automatic differentiation to compute gradients of the
multi-modal loss w.r.t. model parameters. This is only possible because
vbjax provides a differentiable simulation pipeline — a unique advantage
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

    Requires a VbjaxFitnessAdapter (which provides loss/fc_loss).

    Parameters
    ----------
    learning_rate : float
        Step size for the optimizer.
    algorithm : str
        Optax algorithm name: "adam", "sgd", "adamw".
    use_multimodal_loss : bool
        If True, use adapter.loss() (multi-modal: FC + FCD + sensor).
        If False, use adapter.fc_loss() (FC only, backward compatible).
    """

    name: str = "Adam-JAX"

    def __init__(
        self,
        learning_rate: float = 1e-3,
        algorithm: str = "adam",
        use_multimodal_loss: bool = False,
    ):
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.use_multimodal_loss = use_multimodal_loss

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
                "(needs loss/fc_loss for autodiff)"
            )

        t0 = time.perf_counter()

        # Initialize at center of bounds
        x = adapter.default_param_array()
        bounds_lo, bounds_hi = adapter.bounds_arrays()

        # Select loss function
        loss_fn = adapter.loss if self.use_multimodal_loss else adapter.fc_loss

        # Setup optax optimizer
        if self.algorithm == "adam":
            opt = optax.adam(self.learning_rate)
        elif self.algorithm == "sgd":
            opt = optax.sgd(self.learning_rate)
        elif self.algorithm == "adamw":
            opt = optax.adamw(self.learning_rate)
        else:
            opt = optax.adam(self.learning_rate)

        opt_state = opt.init(x)
        grad_fn = jax.grad(loss_fn)

        history = []
        best_loss = float("inf")
        best_x = x
        param_names = list(adapter.parameter_space.keys())

        for step in range(budget):
            g = grad_fn(x)

            if not jnp.all(jnp.isfinite(g)):
                break

            updates, opt_state = opt.update(g, opt_state)
            x = optax.apply_updates(x, updates)
            x = jnp.clip(x, bounds_lo, bounds_hi)

            current_loss = float(loss_fn(x))

            if current_loss < best_loss:
                best_loss = current_loss
                best_x = x

            history.append({
                "gen": step + 1,
                "best_fc": -best_loss if not self.use_multimodal_loss else None,
                "loss": current_loss,
                "best_loss": best_loss,
                "n_evals": step + 1,
                "grad_norm": float(jnp.linalg.norm(g)),
            })

        # Final evaluation for full FitnessResult
        best_params = {
            name: float(best_x[i]) for i, name in enumerate(param_names)
        }
        best_result = adapter.evaluate(best_params)

        return OptimizationResult(
            best_params=best_params,
            best_fitness=best_result,
            history=history,
            total_evaluations=len(history),
            wall_time=time.perf_counter() - t0,
            optimizer_name=self.name,
        )
