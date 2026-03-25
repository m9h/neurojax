"""VbjaxFitnessAdapter — wraps vbjax JR simulation for whole-brain benchmarking.

Runs Jansen-Rit neural mass model on a connectome, generates BOLD signal,
computes FC and FCD, and returns FitnessResult. Supports JAX autodiff
through the entire pipeline via evaluate_differentiable().
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import vbjax

from neurojax.bench.fitness import FitnessAdapter, FitnessResult, ObjectiveSpec
from neurojax.bench.monitors.fc import fc, matrix_correlation
from neurojax.bench.monitors.fcd import fcd_ks_distance


# JR parameters we expose for optimization (subset of JRTheta fields)
_JR_PARAM_BOUNDS = {
    "A": (2.0, 5.0),
    "B": (10.0, 40.0),
    "a": (0.05, 0.2),
    "b": (0.02, 0.1),
    "mu": (0.0, 0.5),
    "I": (0.0, 1.0),
    "K_gl": (0.0, 0.03),  # global coupling strength (stable range)
}


@dataclass
class VbjaxSimConfig:
    """Configuration for a vbjax JR network simulation."""

    dt: float = 0.1  # ms
    duration: float = 60_000.0  # ms (60s)
    bold_dt: float = 2000.0  # ms (TR = 2s, 0.5 Hz)
    noise_sigma: float = 0.1
    warmup: float = 5_000.0  # ms to discard
    seed: int = 42


class VbjaxFitnessAdapter:
    """FitnessAdapter wrapping vbjax Jansen-Rit network simulation.

    Given a connectome (weights, delays) and an empirical FC target,
    evaluates how well a parameter set reproduces the target FC.
    """

    def __init__(
        self,
        weights: jnp.ndarray,
        empirical_fc: jnp.ndarray,
        empirical_bold: Optional[jnp.ndarray] = None,
        config: Optional[VbjaxSimConfig] = None,
        param_bounds: Optional[dict[str, tuple[float, float]]] = None,
    ):
        self.weights = jnp.asarray(weights)
        self.n_regions = weights.shape[0]
        self.empirical_fc = jnp.asarray(empirical_fc)
        self.empirical_bold = (
            jnp.asarray(empirical_bold) if empirical_bold is not None else None
        )
        self.config = config or VbjaxSimConfig()
        self._param_bounds = param_bounds or _JR_PARAM_BOUNDS

        # Precompute coupling function
        self._cfun = vbjax.make_linear_cfun(self.weights)

        # Pre-build SDE integrator for gradient path
        # The drift/diffusion closures capture self._cfun but NOT parameters —
        # parameters are passed through the `p` argument as (theta, k_gl) tuple.
        def _grad_drift(y, p):
            theta, k_gl = p
            psp = y[1] - y[2]
            c = self._cfun(psp) * k_gl
            return vbjax.jr_dfun(y, c, theta)

        def _grad_diffusion(y, p):
            return self.config.noise_sigma

        self._grad_step, self._grad_loop = vbjax.make_sde(
            self.config.dt, _grad_drift, _grad_diffusion
        )

    @property
    def parameter_space(self) -> dict[str, tuple[float, float]]:
        return dict(self._param_bounds)

    @property
    def objectives(self) -> list[ObjectiveSpec]:
        return [
            ObjectiveSpec("fc_correlation", "maximize"),
            ObjectiveSpec("fcd_ks_distance", "minimize"),
        ]

    def _make_theta(self, params: dict[str, float]) -> vbjax.JRTheta:
        """Create JRTheta from param dict, using defaults for unspecified."""
        defaults = vbjax.jr_default_theta._asdict()
        # Filter out non-JRTheta params (like K_gl)
        jr_params = {k: v for k, v in params.items() if k in defaults}
        defaults.update(jr_params)
        return vbjax.JRTheta(**defaults)

    def _simulate(
        self, params: dict[str, float], key: jax.Array
    ) -> jnp.ndarray:
        """Run JR network simulation, return BOLD timeseries.

        Returns:
            (n_regions, n_timepoints) BOLD array.
        """
        theta = self._make_theta(params)
        n_steps = int(self.config.duration / self.config.dt)
        warmup_steps = int(self.config.warmup / self.config.dt)

        # vbjax JR expects state shape (6, n_nodes)
        y0 = jnp.zeros((6, self.n_regions))

        # Noise shape must match state: (n_steps, 6, n_regions)
        noise = (
            jax.random.normal(key, (n_steps, 6, self.n_regions))
            * self.config.noise_sigma
        )

        # Global coupling strength
        k_gl = params.get("K_gl", 0.01)

        # Drift: JR with linear coupling
        # jr_dfun(ys, c, p) where ys is (6, n_nodes), c is (n_nodes,)
        def drift(y, p):
            # Coupling based on pyramidal output (y[1] - y[2])
            psp = y[1] - y[2]
            c = self._cfun(psp) * k_gl
            return vbjax.jr_dfun(y, c, p)

        def diffusion(y, p):
            return self.config.noise_sigma

        _, loop = vbjax.make_sde(self.config.dt, drift, diffusion)

        # Run simulation
        states = loop(y0, noise, theta)  # (n_steps, 6, n_regions)

        # Extract pyramidal PSP (y[1] - y[2]) as neural activity
        neural_activity = states[:, 1, :] - states[:, 2, :]  # (n_steps, n_regions)

        # Discard warmup
        neural_activity = neural_activity[warmup_steps:]

        # Subsample to BOLD temporal resolution
        bold_subsample = int(self.config.bold_dt / self.config.dt)
        bold_signal = neural_activity[::bold_subsample]  # (n_bold, n_regions)

        return bold_signal.T  # (n_regions, n_bold)

    def evaluate(self, params: dict[str, float]) -> FitnessResult:
        """Run simulation and compute fitness against empirical FC."""
        t0 = time.perf_counter()

        key = jax.random.PRNGKey(self.config.seed)
        bold = self._simulate(params, key)

        sim_fc = fc(bold)
        r_fc = float(matrix_correlation(sim_fc, self.empirical_fc))

        # FCD KS distance (only if empirical BOLD available)
        d_fcd = 0.0
        if self.empirical_bold is not None:
            d_fcd = float(
                fcd_ks_distance(bold, self.empirical_bold, window_size=30, step_size=5)
            )

        wall_time = time.perf_counter() - t0

        return FitnessResult(
            fc_correlation=r_fc,
            fcd_ks_distance=d_fcd,
            simulated_fc=np.asarray(sim_fc),
            simulated_bold=np.asarray(bold),
            wall_time=wall_time,
        )

    def evaluate_batch(
        self, params_batch: list[dict[str, float]]
    ) -> list[FitnessResult]:
        """Sequential batch evaluation. Override with vmap for GPU."""
        return [self.evaluate(p) for p in params_batch]

    def fc_loss(self, param_array: jnp.ndarray) -> jnp.ndarray:
        """Differentiable scalar loss for gradient-based optimization.

        Args:
            param_array: 1D array of parameter values in parameter_space order.

        Returns:
            Negative FC correlation (minimize this). Fully JAX-traceable.
        """
        param_names = list(self._param_bounds.keys())
        # Build params dict without calling float() — keep as JAX tracers
        params = {name: param_array[i] for i, name in enumerate(param_names)}

        key = jax.random.PRNGKey(self.config.seed)
        bold = self._simulate_jax(params, key)
        sim_fc = fc(bold)
        return -matrix_correlation(sim_fc, self.empirical_fc)

    def _simulate_jax(
        self, params: dict[str, jnp.ndarray], key: jax.Array
    ) -> jnp.ndarray:
        """JAX-traceable simulation using pre-built integrator.

        Parameters are passed as (theta, k_gl) pytree through the `p`
        argument of the scan loop — no closure over traced values.
        """
        theta = self._make_theta(
            {k: v for k, v in params.items() if k in vbjax.jr_default_theta._asdict()}
        )
        k_gl = params.get("K_gl", jnp.array(0.01))
        combined_params = (theta, k_gl)

        n_steps = int(self.config.duration / self.config.dt)
        warmup_steps = int(self.config.warmup / self.config.dt)

        y0 = jnp.zeros((6, self.n_regions))
        noise = (
            jax.random.normal(key, (n_steps, 6, self.n_regions))
            * self.config.noise_sigma
        )

        states = self._grad_loop(y0, noise, combined_params)

        neural_activity = states[:, 1, :] - states[:, 2, :]
        neural_activity = neural_activity[warmup_steps:]

        bold_subsample = int(self.config.bold_dt / self.config.dt)
        bold_signal = neural_activity[::bold_subsample]
        return bold_signal.T
