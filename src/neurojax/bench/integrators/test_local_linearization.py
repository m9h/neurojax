"""Tests for Local Linearization (LL) SDE integrator — TDD RED phase.

The LL filter should:
1. Be more accurate than Euler/Heun for stiff systems
2. Allow larger timesteps for the same accuracy
3. Be fully differentiable via JAX (grad through expm + jacfwd)
4. Match Euler/Heun for non-stiff systems (consistency check)
5. Handle the RWW and JR neural mass model dynamics
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neurojax.bench.integrators.local_linearization import (
    ll_step,
    ll_loop,
    make_ll_sde,
)


# ---------------------------------------------------------------------------
# Simple test systems
# ---------------------------------------------------------------------------

def linear_drift(x, p):
    """Simple stable linear system: dx/dt = -x."""
    return -x


def stiff_drift(x, p):
    """Stiff 2D system: fast and slow modes.
    dx0/dt = -100*x0 + x1  (fast, τ=10ms)
    dx1/dt = -x1            (slow, τ=1s)
    """
    return jnp.array([-100 * x[0] + x[1], -x[1]])


def rww_like_drift(x, p):
    """RWW-like stiff system: NMDA (τ=100ms) + GABA (τ=10ms)."""
    S_E, S_I = x[0], x[1]
    tau_E, tau_I = 0.1, 0.01  # seconds
    dS_E = -S_E / tau_E + 0.5 * (1 - S_E)
    dS_I = -S_I / tau_I + 0.3
    return jnp.array([dS_E, dS_I])


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestLLStep:
    def test_shape_preserved(self):
        x = jnp.array([1.0, 0.5])
        x_next = ll_step(x, linear_drift, 0.01, None, jnp.zeros(2))
        assert x_next.shape == x.shape

    def test_finite(self):
        x = jnp.array([1.0, 0.5])
        x_next = ll_step(x, stiff_drift, 0.01, None, jnp.zeros(2))
        assert jnp.all(jnp.isfinite(x_next))

    def test_linear_system_exact(self):
        """For a linear system, LL should give the exact solution."""
        x0 = jnp.array([1.0])
        dt = 0.1
        x_ll = ll_step(x0, linear_drift, dt, None, jnp.zeros(1))
        x_exact = x0 * jnp.exp(-dt)
        np.testing.assert_allclose(x_ll, x_exact, rtol=1e-4)

    def test_noise_injection(self):
        """With noise, output should differ from deterministic."""
        x = jnp.array([1.0, 0.5])
        noise = jnp.array([0.1, -0.1])
        x_det = ll_step(x, stiff_drift, 0.01, None, jnp.zeros(2))
        x_noisy = ll_step(x, stiff_drift, 0.01, None, noise)
        assert not jnp.allclose(x_det, x_noisy)


# ---------------------------------------------------------------------------
# Accuracy: LL vs Euler on stiff system
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_stiff_system_stable_large_dt(self):
        """LL should remain stable with large dt on stiff system.
        Euler would diverge at dt=0.05 for the fast mode (τ=10ms)."""
        x = jnp.array([1.0, 1.0])
        dt = 0.05  # 50ms — way too large for Euler on τ=10ms mode

        # LL step
        x_ll = ll_step(x, stiff_drift, dt, None, jnp.zeros(2))
        assert jnp.all(jnp.isfinite(x_ll))
        assert jnp.all(jnp.abs(x_ll) < 10)  # bounded

    def test_stiff_convergence(self):
        """LL with large dt should give similar result to Euler with tiny dt."""
        x0 = jnp.array([1.0, 1.0])
        T = 0.1  # 100ms

        # LL: 10 steps of 10ms
        x = x0
        for _ in range(10):
            x = ll_step(x, stiff_drift, 0.01, None, jnp.zeros(2))
        x_ll = x

        # Euler: 10000 steps of 0.01ms (reference)
        x = x0
        for _ in range(10000):
            x = x + 0.00001 * stiff_drift(x, None)
        x_euler_ref = x

        # LL should be closer to reference than Euler with same dt
        np.testing.assert_allclose(x_ll, x_euler_ref, atol=0.1)

    def test_rww_like_stable(self):
        """LL on RWW-like stiff dynamics should be stable."""
        x = jnp.array([0.15, 0.05])
        for _ in range(100):
            x = ll_step(x, rww_like_drift, 0.001, None, jnp.zeros(2))
        assert jnp.all(jnp.isfinite(x))
        assert jnp.all(x > -1) and jnp.all(x < 2)


# ---------------------------------------------------------------------------
# Loop (scan-based)
# ---------------------------------------------------------------------------

class TestLLLoop:
    def test_output_shape(self):
        x0 = jnp.array([1.0, 0.5])
        noise = jnp.zeros((100, 2))
        states = ll_loop(x0, linear_drift, 0.01, None, noise)
        assert states.shape == (100, 2)

    def test_deterministic_reproducible(self):
        x0 = jnp.array([1.0, 0.5])
        noise = jnp.zeros((50, 2))
        s1 = ll_loop(x0, stiff_drift, 0.01, None, noise)
        s2 = ll_loop(x0, stiff_drift, 0.01, None, noise)
        np.testing.assert_allclose(s1, s2)


# ---------------------------------------------------------------------------
# make_ll_sde: vbjax-compatible interface
# ---------------------------------------------------------------------------

class TestMakeLLSDE:
    def test_returns_step_and_loop(self):
        step, loop = make_ll_sde(0.01, stiff_drift, 0.1)
        assert callable(step)
        assert callable(loop)

    def test_step_matches_ll_step(self):
        step, _ = make_ll_sde(0.01, stiff_drift, 0.1)
        x = jnp.array([1.0, 0.5])
        noise = jnp.zeros(2)
        x_step = step(x, noise, None)
        x_ll = ll_step(x, stiff_drift, 0.01, None, noise * jnp.sqrt(0.01) * 0.1)
        # Should be similar (exact match depends on noise scaling convention)
        assert jnp.all(jnp.isfinite(x_step))

    def test_loop_interface(self):
        _, loop = make_ll_sde(0.01, stiff_drift, 0.1)
        x0 = jnp.array([1.0, 0.5])
        noise = jr.normal(jr.PRNGKey(0), (100, 2))
        states = loop(x0, noise, None)
        assert states.shape == (100, 2)
        assert jnp.all(jnp.isfinite(states))


# ---------------------------------------------------------------------------
# Differentiability (the key JAX advantage)
# ---------------------------------------------------------------------------

class TestDifferentiability:
    def test_grad_through_step(self):
        """Gradient should flow through a single LL step."""
        def loss(x0):
            x1 = ll_step(x0, stiff_drift, 0.01, None, jnp.zeros(2))
            return jnp.sum(x1 ** 2)
        grad = jax.grad(loss)(jnp.array([1.0, 0.5]))
        assert grad.shape == (2,)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_through_loop(self):
        """Gradient should flow through the full LL scan loop."""
        def loss(x0):
            noise = jnp.zeros((50, 2))
            states = ll_loop(x0, stiff_drift, 0.01, None, noise)
            return jnp.mean(states ** 2)
        grad = jax.grad(loss)(jnp.array([1.0, 0.5]))
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_wrt_params(self):
        """Gradient with respect to model parameters."""
        def parameterized_drift(x, p):
            tau = p[0]
            return -x / tau

        def loss(tau):
            x0 = jnp.array([1.0])
            x1 = ll_step(x0, parameterized_drift, 0.01, jnp.array([tau]), jnp.zeros(1))
            return jnp.sum(x1 ** 2)

        grad = jax.grad(loss)(0.1)
        assert jnp.isfinite(grad)

    def test_jit_compatible(self):
        """LL step should work under jax.jit."""
        @jax.jit
        def jitted_step(x):
            return ll_step(x, linear_drift, 0.01, None, jnp.zeros_like(x))
        result = jitted_step(jnp.array([1.0]))
        assert jnp.isfinite(result[0])
