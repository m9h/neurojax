"""Tests for functional network construction."""
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from neurojax.analysis.funcnet import (
    betweenness_centrality, characteristic_path_length, clustering_coefficient,
    degree, global_clustering, lagged_cross_correlation, mutual_information,
    mutual_information_matrix, optimal_lag, small_world_index, threshold_matrix,
)

class TestMutualInformation:
    def test_identical_signals_high_mi(self):
        x = jr.normal(jr.PRNGKey(0), (500,))
        mi = mutual_information(x, x)
        assert mi > 1.0  # high MI for identical
    def test_independent_signals_low_mi(self):
        x = jr.normal(jr.PRNGKey(0), (1000,))
        y = jr.normal(jr.PRNGKey(1), (1000,))
        mi = mutual_information(x, y)
        assert mi < 0.5  # binning MI has residual bias
    def test_nonnegative(self):
        x = jr.normal(jr.PRNGKey(0), (200,))
        y = jr.normal(jr.PRNGKey(1), (200,))
        assert mutual_information(x, y) >= 0
    def test_matrix_symmetric(self):
        data = jr.normal(jr.PRNGKey(0), (300, 4))
        mi = mutual_information_matrix(data)
        np.testing.assert_allclose(mi, mi.T, atol=1e-6)
    def test_matrix_diagonal_positive(self):
        data = jr.normal(jr.PRNGKey(0), (300, 3))
        mi = mutual_information_matrix(data)
        assert jnp.all(jnp.diag(mi) > 0)

class TestLaggedCrossCorrelation:
    def test_shape(self):
        x = jr.normal(jr.PRNGKey(0), (200,))
        y = jr.normal(jr.PRNGKey(1), (200,))
        cc = lagged_cross_correlation(x, y, max_lag=20)
        assert cc.shape == (41,)  # 2*20 + 1
    def test_self_peaks_at_zero(self):
        x = jr.normal(jr.PRNGKey(0), (200,))
        cc = lagged_cross_correlation(x, x, max_lag=20)
        assert int(jnp.argmax(cc)) == 20  # lag=0 is at index max_lag
    def test_shifted_signal_correct_lag(self):
        x = jnp.zeros(200)
        x = x.at[50:60].set(1.0)
        y = jnp.zeros(200)
        y = y.at[55:65].set(1.0)  # shifted by 5
        lag, corr = optimal_lag(x, y, max_lag=20)
        assert lag == 5 or lag == -5  # direction depends on convention

class TestNetworkMeasures:
    @pytest.fixture
    def complete_graph(self):
        N = 5
        return jnp.ones((N, N)) - jnp.eye(N)
    @pytest.fixture
    def ring_graph(self):
        N = 6
        A = jnp.zeros((N, N))
        for i in range(N):
            A = A.at[i, (i + 1) % N].set(1)
            A = A.at[(i + 1) % N, i].set(1)
        return A

    def test_degree_complete(self, complete_graph):
        deg = degree(complete_graph)
        np.testing.assert_allclose(deg, 4.0)
    def test_clustering_complete(self, complete_graph):
        assert global_clustering(complete_graph) == pytest.approx(1.0)
    def test_clustering_ring(self, ring_graph):
        assert global_clustering(ring_graph) == pytest.approx(0.0)
    def test_path_length_complete(self, complete_graph):
        assert characteristic_path_length(complete_graph) == pytest.approx(1.0)
    def test_betweenness_shape(self, ring_graph):
        bc = betweenness_centrality(ring_graph)
        assert bc.shape == (6,)
    def test_threshold_density(self):
        W = jr.normal(jr.PRNGKey(0), (10, 10))
        W = (W + W.T) / 2
        A = threshold_matrix(jnp.abs(W), density=0.2)
        actual = float(jnp.sum(A) / (10 * 9))  # exclude diagonal
        assert abs(actual - 0.2) < 0.1
