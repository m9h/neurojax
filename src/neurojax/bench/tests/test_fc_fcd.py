"""Layer 0 tests for FC and FCD computation.

Tests correctness properties of the JAX FC/FCD implementations
without any external dependencies (neurolib, vbjax, etc.).
"""

import jax
import jax.numpy as jnp
import pytest

from neurojax.bench.monitors.fc import fc, fc_triu, matrix_correlation
from neurojax.bench.monitors.fcd import fcd, fcd_triu, fcd_ks_distance, _ks_statistic


# --- FC tests ---


class TestFC:
    def test_identity_fc(self):
        """FC of identical timeseries rows produces all-ones matrix."""
        ts = jnp.tile(jnp.sin(jnp.linspace(0, 10, 100)), (4, 1))
        # All rows identical → all correlations = 1.0
        result = fc(ts)
        assert jnp.allclose(result, jnp.ones((4, 4)), atol=1e-5)

    def test_symmetric(self, synthetic_bold_4node):
        """FC matrix is symmetric."""
        result = fc(synthetic_bold_4node)
        assert jnp.allclose(result, result.T, atol=1e-6)

    def test_diagonal_ones(self, synthetic_bold_4node):
        """Diagonal of FC is 1.0 (self-correlation)."""
        result = fc(synthetic_bold_4node)
        assert jnp.allclose(jnp.diag(result), jnp.ones(4), atol=1e-6)

    def test_bounded(self, synthetic_bold_4node):
        """FC values are in [-1, 1]."""
        result = fc(synthetic_bold_4node)
        assert jnp.all(result >= -1.0 - 1e-6)
        assert jnp.all(result <= 1.0 + 1e-6)

    def test_constant_timeseries_no_nan(self, constant_timeseries):
        """Constant timeseries produces finite FC (no NaN)."""
        result = fc(constant_timeseries)
        assert jnp.all(jnp.isfinite(result))

    def test_shape(self, synthetic_bold_4node):
        """FC has shape (n_regions, n_regions)."""
        result = fc(synthetic_bold_4node)
        assert result.shape == (4, 4)

    def test_anticorrelated(self):
        """Perfectly anticorrelated signals give FC = -1."""
        t = jnp.linspace(0, 10, 100)
        ts = jnp.stack([jnp.sin(t), -jnp.sin(t)])
        result = fc(ts)
        assert jnp.isclose(result[0, 1], -1.0, atol=1e-5)


class TestFCTriu:
    def test_length(self, synthetic_bold_4node):
        """Upper-triangular has n*(n-1)/2 elements."""
        result = fc_triu(synthetic_bold_4node)
        assert result.shape == (6,)  # 4*3/2

    def test_excludes_diagonal(self):
        """Triu does not include diagonal (self-correlation) values."""
        ts = jax.random.normal(jax.random.PRNGKey(0), (3, 50))
        triu = fc_triu(ts)
        assert triu.shape == (3,)  # 3*2/2


class TestMatrixCorrelation:
    def test_identical_matrices(self, synthetic_bold_4node):
        """Correlation of identical FC matrices is 1.0."""
        fc_mat = fc(synthetic_bold_4node)
        r = matrix_correlation(fc_mat, fc_mat)
        assert jnp.isclose(r, 1.0, atol=1e-5)

    def test_symmetric_metric(self, synthetic_bold_4node):
        """matrix_correlation(A, B) == matrix_correlation(B, A)."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(7))
        ts1 = jax.random.normal(key1, (4, 100))
        ts2 = jax.random.normal(key2, (4, 100))
        fc1, fc2 = fc(ts1), fc(ts2)
        assert jnp.isclose(
            matrix_correlation(fc1, fc2),
            matrix_correlation(fc2, fc1),
            atol=1e-6,
        )

    def test_bounded(self, synthetic_bold_4node):
        """Correlation is in [-1, 1]."""
        key = jax.random.PRNGKey(99)
        ts2 = jax.random.normal(key, (4, 200))
        r = matrix_correlation(fc(synthetic_bold_4node), fc(ts2))
        assert -1.0 - 1e-6 <= float(r) <= 1.0 + 1e-6


# --- FCD tests ---


class TestFCD:
    def test_symmetric(self, synthetic_bold_4node):
        """FCD matrix is symmetric."""
        result = fcd(synthetic_bold_4node, window_size=20, step_size=5)
        assert jnp.allclose(result, result.T, atol=1e-6)

    def test_diagonal_ones(self, synthetic_bold_4node):
        """FCD diagonal is 1.0 (self-correlation of FC windows)."""
        result = fcd(synthetic_bold_4node, window_size=20, step_size=5)
        assert jnp.allclose(jnp.diag(result), jnp.ones(result.shape[0]), atol=1e-5)

    def test_bounded(self, synthetic_bold_4node):
        """FCD values are in [-1, 1]."""
        result = fcd(synthetic_bold_4node, window_size=20, step_size=5)
        assert jnp.all(result >= -1.0 - 1e-6)
        assert jnp.all(result <= 1.0 + 1e-6)

    def test_shape(self, synthetic_bold_4node):
        """FCD has shape (n_windows, n_windows)."""
        result = fcd(synthetic_bold_4node, window_size=20, step_size=5)
        # n_time=200, windows: 0,5,10,...,180 → (200-20)//5 + 1 = 37 windows
        expected_n_windows = (200 - 20) // 5 + 1
        assert result.shape == (expected_n_windows, expected_n_windows)


class TestKSStatistic:
    def test_identical_distributions_zero(self):
        """KS distance between identical samples is 0."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert jnp.isclose(_ks_statistic(x, x), 0.0, atol=1e-6)

    def test_bounded(self):
        """KS distance is in [0, 1]."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(0))
        x = jax.random.normal(key1, (100,))
        y = jax.random.normal(key2, (100,)) + 5.0  # shifted
        ks = _ks_statistic(x, y)
        assert 0.0 <= float(ks) <= 1.0

    def test_completely_separated(self):
        """Completely separated distributions give KS = 1.0."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([10.0, 11.0, 12.0])
        assert jnp.isclose(_ks_statistic(x, y), 1.0, atol=1e-6)

    def test_symmetric(self):
        """KS(x, y) == KS(y, x)."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        x = jax.random.normal(key1, (50,))
        y = jax.random.normal(key2, (50,))
        assert jnp.isclose(
            _ks_statistic(x, y), _ks_statistic(y, x), atol=1e-6
        )


class TestFCDKSDistance:
    def test_identical_timeseries_zero(self, identical_timeseries):
        """FCD KS distance between identical timeseries is 0."""
        ts1, ts2 = identical_timeseries
        ks = fcd_ks_distance(ts1, ts2, window_size=20, step_size=5)
        assert jnp.isclose(ks, 0.0, atol=1e-5)

    def test_nonnegative(self, synthetic_bold_4node):
        """FCD KS distance is >= 0."""
        key = jax.random.PRNGKey(77)
        ts2 = jax.random.normal(key, (4, 200))
        ks = fcd_ks_distance(synthetic_bold_4node, ts2, window_size=20, step_size=5)
        assert float(ks) >= 0.0


# --- Differentiability tests ---


class TestDifferentiability:
    def test_fc_grad(self, synthetic_bold_4node):
        """FC computation is differentiable via JAX."""
        def loss(ts):
            target = jnp.eye(4)
            return jnp.sum((fc(ts) - target) ** 2)

        grad_fn = jax.grad(loss)
        g = grad_fn(synthetic_bold_4node)
        assert g.shape == synthetic_bold_4node.shape
        assert jnp.all(jnp.isfinite(g))

    def test_matrix_correlation_grad(self):
        """matrix_correlation is differentiable through FC pipeline."""
        # Target must have non-constant upper-triangle for meaningful gradient
        key = jax.random.PRNGKey(1)
        target_ts = jax.random.normal(key, (4, 100))
        target_fc = fc(target_ts)

        def loss(ts):
            sim_fc = fc(ts)
            return -matrix_correlation(sim_fc, target_fc)

        ts = jax.random.normal(jax.random.PRNGKey(0), (4, 100))
        g = jax.grad(loss)(ts)
        assert g.shape == ts.shape
        assert jnp.all(jnp.isfinite(g))
