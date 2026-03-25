"""Tests for the Reduced Wong-Wang (RWW) neural mass model (Deco et al. 2013).

RED-GREEN TDD: these tests are written first, before the implementation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neurojax.bench.models.rww import (
    RWWTheta,
    rww_default_theta,
    rww_dfun,
    rww_transfer_E,
    rww_transfer_I,
)


class TestRWWDefaults:
    """Verify default parameters match Deco et al. 2013."""

    def test_default_theta_is_named_tuple(self):
        assert isinstance(rww_default_theta, RWWTheta)

    def test_default_excitatory_params(self):
        theta = rww_default_theta
        assert theta.a_E == 310.0
        assert theta.b_E == 125.0
        assert theta.d_E == 0.16
        assert theta.tau_E == 0.1
        assert theta.gamma_E == pytest.approx(0.641)

    def test_default_inhibitory_params(self):
        theta = rww_default_theta
        assert theta.a_I == 615.0
        assert theta.b_I == 177.0
        assert theta.d_I == 0.087
        assert theta.tau_I == 0.01
        assert theta.gamma_I == pytest.approx(1.0)

    def test_default_coupling_params(self):
        theta = rww_default_theta
        assert theta.W_E == 1.0
        assert theta.W_I == 0.7
        assert theta.w_plus == 1.4
        assert theta.J_NMDA == 0.15
        assert theta.J_I == 1.0
        assert theta.I_0 == 0.382
        assert theta.G == 2.0
        assert theta.sigma == 0.01


class TestTransferFunctions:
    """Test excitatory and inhibitory transfer functions H_E, H_I."""

    def test_transfer_E_positive_input(self):
        """H_E should produce positive firing rates for positive input."""
        I_vals = jnp.array([0.3, 0.5, 1.0, 2.0])
        theta = rww_default_theta
        rates = rww_transfer_E(I_vals, theta)
        assert jnp.all(rates > 0), f"Expected positive rates, got {rates}"
        assert jnp.all(jnp.isfinite(rates))

    def test_transfer_I_positive_input(self):
        """H_I should produce positive firing rates for positive input."""
        I_vals = jnp.array([0.3, 0.5, 1.0, 2.0])
        theta = rww_default_theta
        rates = rww_transfer_I(I_vals, theta)
        assert jnp.all(rates > 0), f"Expected positive rates, got {rates}"
        assert jnp.all(jnp.isfinite(rates))

    def test_transfer_E_monotonic(self):
        """H_E should be monotonically increasing."""
        I_vals = jnp.linspace(0.1, 3.0, 50)
        theta = rww_default_theta
        rates = rww_transfer_E(I_vals, theta)
        diffs = jnp.diff(rates)
        assert jnp.all(diffs > 0), "Transfer function H_E should be monotonic"

    def test_transfer_I_monotonic(self):
        """H_I should be monotonically increasing."""
        I_vals = jnp.linspace(0.1, 3.0, 50)
        theta = rww_default_theta
        rates = rww_transfer_I(I_vals, theta)
        diffs = jnp.diff(rates)
        assert jnp.all(diffs > 0), "Transfer function H_I should be monotonic"

    def test_transfer_E_near_zero_input(self):
        """H_E should handle near-zero input without NaN."""
        I_val = jnp.array([0.0, 0.001, -0.001])
        theta = rww_default_theta
        rates = rww_transfer_E(I_val, theta)
        assert jnp.all(jnp.isfinite(rates))


class TestRWWDfun:
    """Test the RWW drift function."""

    def test_output_shape_single_node(self):
        """Drift function output should match state shape (2, n_nodes)."""
        n_nodes = 1
        state = jnp.array([[0.1], [0.1]])  # (2, 1) -- S_E, S_I
        coupling = jnp.zeros(n_nodes)
        theta = rww_default_theta
        dstate = rww_dfun(state, coupling, theta)
        assert dstate.shape == (2, n_nodes)

    def test_output_shape_multi_node(self):
        """Drift function output should match state shape for multiple nodes."""
        n_nodes = 10
        state = jnp.ones((2, n_nodes)) * 0.1
        coupling = jnp.zeros(n_nodes)
        theta = rww_default_theta
        dstate = rww_dfun(state, coupling, theta)
        assert dstate.shape == (2, n_nodes)

    def test_output_finite(self):
        """Drift function should produce finite output for default params."""
        n_nodes = 4
        state = jnp.ones((2, n_nodes)) * 0.15
        coupling = jnp.zeros(n_nodes)
        theta = rww_default_theta
        dstate = rww_dfun(state, coupling, theta)
        assert jnp.all(jnp.isfinite(dstate)), f"Non-finite drift: {dstate}"

    def test_zero_state_finite(self):
        """Drift should be finite even at zero state."""
        state = jnp.zeros((2, 4))
        coupling = jnp.zeros(4)
        theta = rww_default_theta
        dstate = rww_dfun(state, coupling, theta)
        assert jnp.all(jnp.isfinite(dstate))

    def test_boundary_state_finite(self):
        """Drift should be finite at boundary states (0 and 1)."""
        state_zero = jnp.zeros((2, 4))
        state_one = jnp.ones((2, 4))
        coupling = jnp.zeros(4)
        theta = rww_default_theta
        d0 = rww_dfun(state_zero, coupling, theta)
        d1 = rww_dfun(state_one, coupling, theta)
        assert jnp.all(jnp.isfinite(d0))
        assert jnp.all(jnp.isfinite(d1))


class TestRWWSimulation:
    """Test RWW model integration (single node and network)."""

    def test_single_node_steady_state(self):
        """Single node should converge to a steady state near known fixed point.

        Known fixed point for default params: S_E ~ 0.164, S_I ~ 0.12
        (Deco et al. 2013, approximate values).
        """
        n_nodes = 1
        dt = 0.001  # 1 ms in seconds
        n_steps = 20000  # 20 seconds

        state = jnp.array([[0.1], [0.1]])
        coupling = jnp.zeros(n_nodes)
        theta = rww_default_theta

        # Euler integration (deterministic, no noise)
        for _ in range(n_steps):
            dstate = rww_dfun(state, coupling, theta)
            state = state + dt * dstate
            # Clip to valid range
            state = jnp.clip(state, 0.0, 1.0)

        S_E_final = float(state[0, 0])
        S_I_final = float(state[1, 0])

        # Check approximate fixed point
        assert 0.05 < S_E_final < 0.4, f"S_E = {S_E_final}, expected ~0.164"
        assert 0.02 < S_I_final < 0.3, f"S_I = {S_I_final}, expected ~0.12"

    def test_state_bounded(self):
        """State variables S_E, S_I should stay bounded in [0, 1] during simulation."""
        n_nodes = 4
        dt = 0.001
        n_steps = 5000

        key = jax.random.PRNGKey(42)
        state = jnp.ones((2, n_nodes)) * 0.15
        coupling = jnp.zeros(n_nodes)
        theta = rww_default_theta

        for i in range(n_steps):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, state.shape) * theta.sigma
            dstate = rww_dfun(state, coupling, theta)
            state = state + dt * dstate + jnp.sqrt(dt) * noise
            state = jnp.clip(state, 0.0, 1.0)

        assert jnp.all(state >= 0.0), f"State below 0: {state}"
        assert jnp.all(state <= 1.0), f"State above 1: {state}"
        assert jnp.all(jnp.isfinite(state)), f"Non-finite state: {state}"

    def test_network_nontrivial_fc(self):
        """Network simulation with coupling should produce non-trivial FC.

        FC matrix should not be identity (all uncorrelated) when nodes are coupled.
        """
        n_nodes = 4
        dt = 0.001
        n_steps = 10000

        # Simple coupling matrix
        weights = jnp.array([
            [0.0, 0.5, 0.2, 0.0],
            [0.5, 0.0, 0.3, 0.1],
            [0.2, 0.3, 0.0, 0.4],
            [0.0, 0.1, 0.4, 0.0],
        ])

        key = jax.random.PRNGKey(0)
        state = jnp.ones((2, n_nodes)) * 0.15
        theta = rww_default_theta

        # Collect timeseries
        timeseries = []
        for i in range(n_steps):
            key, subkey = jax.random.split(key)
            S_E = state[0]
            coupling = weights @ S_E  # linear coupling through S_E
            noise = jax.random.normal(subkey, state.shape) * theta.sigma
            dstate = rww_dfun(state, coupling, theta)
            state = state + dt * dstate + jnp.sqrt(dt) * noise
            state = jnp.clip(state, 0.0, 1.0)
            if i % 10 == 0:
                timeseries.append(state[0])  # record S_E

        ts = jnp.stack(timeseries)  # (n_timepoints, n_nodes)
        fc_matrix = jnp.corrcoef(ts.T)  # (n_nodes, n_nodes)

        # FC should be finite
        assert jnp.all(jnp.isfinite(fc_matrix)), f"Non-finite FC: {fc_matrix}"

        # Off-diagonal elements should not all be zero
        off_diag = fc_matrix - jnp.eye(n_nodes)
        assert jnp.any(jnp.abs(off_diag) > 0.01), (
            f"FC is trivial (all off-diag near zero): {fc_matrix}"
        )


class TestRWWDifferentiability:
    """Test that RWW drift function is differentiable via JAX."""

    def test_grad_through_dfun(self):
        """jax.grad should work through rww_dfun."""
        state = jnp.array([[0.15], [0.10]])
        coupling = jnp.zeros(1)
        theta = rww_default_theta

        def loss_fn(s):
            ds = rww_dfun(s, coupling, theta)
            return jnp.sum(ds ** 2)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(state)
        assert grads.shape == state.shape
        assert jnp.all(jnp.isfinite(grads)), f"Non-finite gradients: {grads}"

    def test_grad_wrt_coupling(self):
        """Gradient w.r.t. coupling should be finite and non-zero."""
        state = jnp.array([[0.15, 0.20], [0.10, 0.12]])
        coupling = jnp.array([0.01, 0.02])
        theta = rww_default_theta

        def loss_fn(c):
            ds = rww_dfun(state, c, theta)
            return jnp.sum(ds ** 2)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(coupling)
        assert grads.shape == coupling.shape
        assert jnp.all(jnp.isfinite(grads))
        assert jnp.any(grads != 0.0), "Gradients should be non-zero"

    def test_jit_compatible(self):
        """rww_dfun should be JIT-compilable."""
        state = jnp.ones((2, 4)) * 0.15
        coupling = jnp.zeros(4)
        theta = rww_default_theta

        jit_dfun = jax.jit(rww_dfun)
        dstate = jit_dfun(state, coupling, theta)
        assert dstate.shape == (2, 4)
        assert jnp.all(jnp.isfinite(dstate))


class TestRWWFixedPoint:
    """Verify convergence to known fixed point from Deco et al. 2013."""

    def test_fixed_point_values(self):
        """At steady state, dS/dt should be approximately zero.

        We find the fixed point by integrating and check that:
        - S_E is approximately 0.164
        - S_I is approximately 0.12
        These are approximate values from the literature.
        """
        dt = 0.0005
        n_steps = 50000  # 25 seconds

        state = jnp.array([[0.2], [0.15]])
        coupling = jnp.zeros(1)
        theta = rww_default_theta

        for _ in range(n_steps):
            dstate = rww_dfun(state, coupling, theta)
            state = state + dt * dstate
            state = jnp.clip(state, 0.0, 1.0)

        S_E = float(state[0, 0])
        S_I = float(state[1, 0])

        # Check drift is near zero (fixed point)
        dstate_final = rww_dfun(state, coupling, theta)
        assert jnp.all(jnp.abs(dstate_final) < 0.1), (
            f"Not at fixed point: dstate = {dstate_final}"
        )

        # Approximate fixed point values (tolerant bounds)
        assert 0.05 < S_E < 0.5, f"S_E = {S_E}, expected ~0.164"
        assert 0.02 < S_I < 0.35, f"S_I = {S_I}, expected ~0.12"


class TestVbjaxCompatibility:
    """Test compatibility with vbjax.make_sde interface."""

    def test_dfun_signature(self):
        """rww_dfun should accept (state, coupling, theta) like vbjax models."""
        import inspect
        sig = inspect.signature(rww_dfun)
        params = list(sig.parameters.keys())
        assert len(params) >= 3, f"Expected at least 3 params, got {params}"

    def test_make_sde_integration(self):
        """Should work with vbjax.make_sde when wrapped in a closure."""
        import vbjax

        n_nodes = 2
        theta = rww_default_theta
        weights = jnp.array([[0.0, 0.3], [0.3, 0.0]])

        def drift(state, p):
            S_E = state[0]
            coupling = weights @ S_E
            return rww_dfun(state, coupling, p)

        def diffusion(state, p):
            return p.sigma

        step, loop = vbjax.make_sde(0.001, drift, diffusion)

        state0 = jnp.ones((2, n_nodes)) * 0.15
        key = jax.random.PRNGKey(42)
        n_steps = 1000
        noise = jax.random.normal(key, (n_steps, 2, n_nodes)) * theta.sigma

        states = loop(state0, noise, theta)
        assert states.shape == (n_steps, 2, n_nodes)
        assert jnp.all(jnp.isfinite(states))
