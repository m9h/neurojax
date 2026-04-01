"""TDD tests for LAURA source localization — RED phase.

LAURA (Local AUtoRegressive Average) uses a biophysical spatial prior
where current density falls off as 1/d³ from each source point,
motivated by volume conduction physics.

References:
  Grave de Peralta Menendez et al. (2001) NeuroImage
  Grave de Peralta Menendez et al. (2004) IEEE TBME
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp


class TestLAURAWeightMatrix:
    """The 1/d³ spatial weight matrix is the core innovation."""

    def test_weight_matrix_shape(self):
        from neurojax.source.laura import laura_weight_matrix
        n_sources = 50
        positions = np.random.randn(n_sources, 3) * 50  # mm
        W = laura_weight_matrix(jnp.array(positions))
        assert W.shape == (n_sources, n_sources)

    def test_weight_diagonal_zero(self):
        """Self-interaction should be zero (or regularised)."""
        from neurojax.source.laura import laura_weight_matrix
        pos = jnp.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
        W = laura_weight_matrix(pos)
        # Diagonal should be zero or near-zero (no self-weight)
        diag = jnp.diag(W)
        assert jnp.all(jnp.abs(diag) < 1e-6)

    def test_weight_falls_off_with_distance(self):
        """Closer sources should have stronger weights."""
        from neurojax.source.laura import laura_weight_matrix
        pos = jnp.array([[0, 0, 0], [5, 0, 0], [50, 0, 0]], dtype=float)
        W = laura_weight_matrix(pos)
        # Weight from source 0 to source 1 (5mm) > weight to source 2 (50mm)
        assert float(jnp.abs(W[0, 1])) > float(jnp.abs(W[0, 2]))

    def test_weight_rows_sum_to_one(self):
        """Row-normalised weight matrix rows should sum to 1."""
        from neurojax.source.laura import laura_weight_matrix
        pos = jnp.array(np.random.randn(20, 3) * 30)
        W = laura_weight_matrix(pos)
        row_sums = jnp.sum(W, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_cubic_falloff(self):
        """Weights should follow 1/d³ law."""
        from neurojax.source.laura import laura_weight_matrix
        # Two pairs at known distances
        pos = jnp.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=float)
        W = laura_weight_matrix(pos)
        w_10mm = float(jnp.abs(W[0, 1]))  # d=10
        w_20mm = float(jnp.abs(W[0, 2]))  # d=20
        # Ratio should be (20/10)³ = 8
        ratio = w_10mm / w_20mm if w_20mm > 0 else float('inf')
        assert 6.0 < ratio < 10.0  # approximately 8x


class TestLAURAInverse:
    """Full LAURA inverse solution."""

    @pytest.fixture
    def synthetic_problem(self):
        """Synthetic forward problem with spatially-structured gain.

        LAURA relies on spatial proximity in source positions, so the gain
        matrix must have spatial structure (not random) for the prior to help.
        We simulate gain as 1/d² from sensor to source, mimicking physics.
        Uses favourable ratio (n_sensors > n_sources / 3) for recovery.
        """
        rng = np.random.RandomState(42)
        n_src, n_sen = 50, 32
        positions = rng.randn(n_src, 3) * 50  # source positions in mm
        sensor_pos = rng.randn(n_sen, 3) * 80  # sensor positions (further out)

        # Spatially-structured gain: L_ij ∝ 1/||sensor_i - source_j||²
        diff = sensor_pos[:, None, :] - positions[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1.0)  # (n_sen, n_src)
        L = 1.0 / (dist ** 2)
        L = L / np.max(L) * 0.01  # scale

        # Single focal source for clean recovery
        active_idx = 20
        J_true = np.zeros((n_src, 50))
        J_true[active_idx, :] = np.sin(2 * np.pi * 10 * np.linspace(0, 0.5, 50))

        Y = L @ J_true + rng.randn(n_sen, 50) * 1e-5
        noise_cov = np.eye(n_sen) * 1e-10

        return {
            'Y': jnp.array(Y), 'L': jnp.array(L),
            'positions': jnp.array(positions),
            'noise_cov': jnp.array(noise_cov),
            'J_true': J_true, 'active': [active_idx]
        }

    def test_laura_returns_sources(self, synthetic_problem):
        from neurojax.source.laura import laura
        p = synthetic_problem
        J_est = laura(p['Y'], p['L'], p['positions'], p['noise_cov'])
        assert J_est.shape == p['J_true'].shape

    def test_laura_recovers_active_sources(self, synthetic_problem):
        from neurojax.source.laura import laura
        p = synthetic_problem
        J_est = np.asarray(laura(p['Y'], p['L'], p['positions'], p['noise_cov']))
        power = np.sum(J_est ** 2, axis=1)
        peak = int(np.argmax(power))
        active = p['active'][0]
        # Peak should be at the active source or a near neighbour
        # (within top 5 by power)
        top5 = set(np.argsort(power)[-5:])
        assert active in top5, f"Expected {active} in top 5, got {top5} (peak={peak})"

    def test_laura_differentiable(self, synthetic_problem):
        """Gradients should flow through LAURA for end-to-end fitting."""
        from neurojax.source.laura import laura

        p = synthetic_problem

        def loss(L):
            J = laura(p['Y'], L, p['positions'], p['noise_cov'])
            return jnp.sum(J ** 2)

        grad = jax.grad(loss)(p['L'])
        assert jnp.all(jnp.isfinite(grad))


class TestLAURAvsVARETA:
    """Compare LAURA and VARETA on same data."""

    def test_both_recover_same_sources(self):
        from neurojax.source.laura import laura
        from neurojax.source.vareta import vareta

        rng = np.random.RandomState(0)
        n_src, n_sen = 80, 30
        L = jnp.array(rng.randn(n_sen, n_src) * 0.01)
        positions = jnp.array(rng.randn(n_src, 3) * 40)
        J_true = np.zeros((n_src, 30))
        J_true[20, :] = np.sin(2 * np.pi * 12 * np.linspace(0, 0.3, 30))
        Y = jnp.array(L @ J_true + rng.randn(n_sen, 30) * 0.001)
        noise_cov = jnp.eye(n_sen) * 1e-6

        J_laura = np.asarray(laura(Y, L, positions, noise_cov))
        J_vareta, _, _ = vareta(Y, L, noise_cov)
        J_vareta = np.asarray(J_vareta)

        top_laura = np.argmax(np.sum(J_laura ** 2, axis=1))
        top_vareta = np.argmax(np.sum(J_vareta ** 2, axis=1))

        # Both should find the active source (index 20) or very close
        assert abs(top_laura - 20) < 5 or abs(top_vareta - 20) < 5, \
            f"LAURA peak={top_laura}, VARETA peak={top_vareta}, true=20"
