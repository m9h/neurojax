"""Local Linearization (LL) filter for SDE integration.

A higher-order SDE integrator that linearizes the drift function
at each step and solves the resulting linear SDE exactly via the
matrix exponential. More accurate than Euler/Heun for stiff systems
(like neural mass models with fast GABA and slow NMDA time constants).

Core idea:
    At each step, approximate f(x) ≈ f(x₀) + J(x₀)(x - x₀)
    where J = ∂f/∂x is the Jacobian (computed automatically by JAX).
    The linearized SDE has an exact solution involving exp(J·dt).

Advantages over Euler/Heun:
    - Stable for stiff systems at larger timesteps
    - Naturally captures fast decay modes via matrix exponential
    - Fully differentiable: jax.grad flows through expm + jacfwd
    - Automatic Jacobian — no manual derivation needed

Interface compatible with vbjax.make_sde():
    step, loop = make_ll_sde(dt, drift_fn, noise_sigma)
    states = loop(x0, noise_array, params)

References
----------
Riera JJ et al. (2004). Fusing EEG and fMRI with a multimodal
bilinear model for source imaging. NeuroImage.

Ozaki T (1992). A bridge between nonlinear time series models and
nonlinear stochastic dynamical systems: A local linearization
approach. Statistica Sinica 2:113-135.

Jimenez JC, Biscay R, Ozaki T (2006). Inference methods for
discretely observed continuous-time stochastic volatility models:
A commented overview. Asia-Pacific Financial Markets 12:109-141.
"""

from __future__ import annotations
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg


def ll_step(
    x: jnp.ndarray,
    drift_fn: Callable,
    dt: float,
    params: Optional[jnp.ndarray],
    noise: jnp.ndarray,
) -> jnp.ndarray:
    """Single Local Linearization step.

    Given current state x, computes x(t+dt) by:
    1. Evaluating f(x) and J(x) = df/dx at current state
    2. Computing the matrix exponential exp(J·dt)
    3. Applying the exact solution of the linearized SDE

    Parameters
    ----------
    x : (n,) current state.
    drift_fn : callable(x, params) → dx/dt.
    dt : float — timestep.
    params : any — parameters passed to drift_fn.
    noise : (n,) noise increment (pre-scaled by sqrt(dt) * sigma).

    Returns
    -------
    x_next : (n,) state at t+dt.
    """
    n = x.shape[0]

    # Evaluate drift and Jacobian at current state
    f = drift_fn(x, params)
    J = jax.jacfwd(drift_fn)(x, params)  # (n, n) Jacobian

    # Matrix exponential: Φ = exp(J·dt)
    Jdt = J * dt
    Phi = jax.scipy.linalg.expm(Jdt)  # (n, n)

    # Exact solution of linearized ODE:
    # x(t+dt) = exp(J·dt) · x + J^{-1} · (exp(J·dt) - I) · f(x₀)
    #         = Phi · x + J^{-1} · (Phi - I) · (f - J·x)
    # where (f - J·x) is the constant term in the affine drift
    #
    # For numerical stability, compute J^{-1}(Phi - I) directly
    # using the identity: J^{-1}(exp(J·dt) - I) = dt · phi_1(J·dt)
    # where phi_1(z) = (exp(z) - I) / z
    #
    # Practical: use the series expansion for small ||J·dt||
    # or direct computation for moderate sizes

    # Direct approach: solve J · η = (Phi - I) · c  where c = f - J @ x
    c = f - J @ x  # constant term (affine part of linearized drift)
    Phi_minus_I = Phi - jnp.eye(n)

    # η = J^{-1} (Phi - I) c
    # Solve: J η = (Phi - I) c  ⟹  η = solve(J, (Phi-I) @ c)
    # Regularize J for numerical stability (near-singular for slow modes)
    J_reg = J + 1e-8 * jnp.eye(n)
    rhs = Phi_minus_I @ c
    eta = jnp.linalg.solve(J_reg, rhs)

    # Deterministic update
    x_det = Phi @ x + eta

    # Add noise (additive noise model)
    x_next = x_det + noise

    return x_next


def ll_loop(
    x0: jnp.ndarray,
    drift_fn: Callable,
    dt: float,
    params: Optional[jnp.ndarray],
    noise: jnp.ndarray,
) -> jnp.ndarray:
    """Scan-based LL integration loop.

    Parameters
    ----------
    x0 : (n,) initial state.
    drift_fn : callable(x, params) → dx/dt.
    dt : float — timestep.
    params : any — model parameters.
    noise : (n_steps, n) — noise array (pre-scaled).

    Returns
    -------
    states : (n_steps, n) — state trajectory.
    """
    def scan_fn(x, noise_t):
        x_next = ll_step(x, drift_fn, dt, params, noise_t)
        return x_next, x_next

    _, states = jax.lax.scan(scan_fn, x0, noise)
    return states


def make_ll_sde(
    dt: float,
    drift_fn: Callable,
    noise_sigma: float,
) -> tuple[Callable, Callable]:
    """Create LL integrator with vbjax-compatible interface.

    Drop-in replacement for vbjax.make_sde():
        step, loop = make_ll_sde(dt, drift_fn, noise_sigma)
        states = loop(x0, noise_samples, params)

    Parameters
    ----------
    dt : float — timestep.
    drift_fn : callable(x, params) → dx/dt.
    noise_sigma : float — noise standard deviation.

    Returns
    -------
    step : callable(x, noise_t, params) → x_next
    loop : callable(x0, noise_array, params) → states
    """
    sqrt_dt = jnp.sqrt(dt)

    def step(x, noise_t, params):
        scaled_noise = sqrt_dt * noise_sigma * noise_t
        return ll_step(x, drift_fn, dt, params, scaled_noise)

    @jax.jit
    def loop(x0, noise_array, params):
        def scan_fn(x, noise_t):
            x_next = step(x, noise_t, params)
            return x_next, x_next
        _, states = jax.lax.scan(scan_fn, x0, noise_array)
        return states

    return step, loop
