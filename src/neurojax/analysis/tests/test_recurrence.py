"""Tests for recurrence analysis."""
import jax.numpy as jnp
import numpy as np
import pytest
from neurojax.analysis.recurrence import (
    average_diagonal_length, determinism, diagonal_entropy, distance_matrix,
    laminarity, max_diagonal_length, max_vertical_length, recurrence_matrix,
    recurrence_rate_measure, rqa_summary, trapping_time,
)

class TestDistanceMatrix:
    def test_shape(self):
        x = jnp.ones((50, 3))
        D = distance_matrix(x)
        assert D.shape == (50, 50)
    def test_zero_diagonal(self):
        x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        D = distance_matrix(x)
        np.testing.assert_allclose(jnp.diag(D), 0.0, atol=1e-7)
    def test_symmetric(self):
        x = jnp.array([[0.0], [1.0], [3.0]])
        D = distance_matrix(x)
        np.testing.assert_allclose(D, D.T, atol=1e-7)
    def test_manhattan(self):
        x = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        D = distance_matrix(x, metric="manhattan")
        assert float(D[0, 1]) == pytest.approx(2.0)
    def test_supremum(self):
        x = jnp.array([[0.0, 0.0], [3.0, 1.0]])
        D = distance_matrix(x, metric="supremum")
        assert float(D[0, 1]) == pytest.approx(3.0)

class TestRecurrenceMatrix:
    def test_binary(self):
        x = jnp.array([[0.0], [0.1], [5.0], [5.1]])
        R = recurrence_matrix(x, threshold=0.5)
        assert jnp.all((R == 0) | (R == 1))
    def test_symmetric(self):
        x = jnp.array([[0.0], [1.0], [2.0], [0.5]])
        R = recurrence_matrix(x, threshold=1.0)
        np.testing.assert_allclose(R, R.T)
    def test_recurrence_rate_target(self):
        x = jnp.linspace(0, 1, 50)[:, None]
        R = recurrence_matrix(x, recurrence_rate=0.1)
        rr = recurrence_rate_measure(R)
        assert abs(rr - 0.1) < 0.05

class TestRQAMeasures:
    @pytest.fixture
    def periodic_R(self):
        t = jnp.linspace(0, 4 * jnp.pi, 100)
        x = jnp.sin(t)[:, None]
        return recurrence_matrix(x, threshold=0.3)
    @pytest.fixture
    def identity_R(self):
        return jnp.eye(20)

    def test_det_periodic_high(self, periodic_R):
        det = determinism(periodic_R)
        assert det > 0.3
    def test_det_identity(self, identity_R):
        # Identity has no off-diagonal recurrences
        det = determinism(identity_R)
        assert det == 0.0
    def test_laminarity_bounded(self, periodic_R):
        lam = laminarity(periodic_R)
        assert 0.0 <= lam <= 1.0
    def test_avg_diag_positive(self, periodic_R):
        adl = average_diagonal_length(periodic_R)
        assert adl >= 2.0
    def test_trapping_time_positive(self, periodic_R):
        tt = trapping_time(periodic_R)
        assert tt >= 0.0
    def test_max_diag_positive(self, periodic_R):
        assert max_diagonal_length(periodic_R) >= 1
    def test_max_vert_nonneg(self, periodic_R):
        assert max_vertical_length(periodic_R) >= 0
    def test_entropy_nonneg(self, periodic_R):
        assert diagonal_entropy(periodic_R) >= 0.0
    def test_rqa_summary_keys(self, periodic_R):
        s = rqa_summary(periodic_R)
        assert "determinism" in s
        assert "laminarity" in s
        assert "recurrence_rate" in s
        assert "diagonal_entropy" in s
    def test_empty_matrix(self):
        R = jnp.zeros((10, 10))
        assert determinism(R) == 0.0
        assert laminarity(R) == 0.0
