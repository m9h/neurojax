"""Tests for visibility graphs."""
import jax.numpy as jnp
import numpy as np
import pytest
from neurojax.analysis.visibility import (
    horizontal_visibility_graph, natural_visibility_graph,
    vg_clustering, vg_degree, vg_degree_distribution, vg_mean_degree,
)

class TestNaturalVisibilityGraph:
    def test_shape(self):
        ts = jnp.array([1.0, 3.0, 2.0, 4.0, 1.0])
        adj = natural_visibility_graph(ts)
        assert adj.shape == (5, 5)
    def test_symmetric(self):
        ts = jnp.array([1.0, 3.0, 2.0, 4.0, 1.0])
        adj = natural_visibility_graph(ts)
        np.testing.assert_allclose(adj, adj.T)
    def test_adjacent_always_connected(self):
        ts = jnp.array([1.0, 2.0, 3.0, 4.0])
        adj = natural_visibility_graph(ts)
        for i in range(3):
            assert float(adj[i, i + 1]) == 1.0
    def test_monotone_adjacent_connected(self):
        """Monotonically increasing → at minimum all adjacent pairs connected.
        Non-adjacent pairs may be blocked if intermediate points lie exactly
        on the line of sight (collinear case)."""
        ts = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        adj = natural_visibility_graph(ts)
        for i in range(4):
            assert float(adj[i, i + 1]) == 1.0

    def test_convex_fully_connected(self):
        """Strictly convex series → all pairs visible (no collinearity)."""
        ts = jnp.array([1.0, 4.0, 9.0, 16.0, 25.0])  # x^2
        adj = natural_visibility_graph(ts)
        expected = jnp.ones((5, 5)) - jnp.eye(5)
        np.testing.assert_allclose(adj, expected)
    def test_known_small_example(self):
        # [3, 1, 2]: node 0-1 connected, 1-2 connected, 0-2: check if 1<3+(2-3)*(1/2)=2.5 → 1<2.5 yes
        ts = jnp.array([3.0, 1.0, 2.0])
        adj = natural_visibility_graph(ts)
        assert float(adj[0, 2]) == 1.0

class TestHorizontalVisibilityGraph:
    def test_shape(self):
        ts = jnp.array([1.0, 3.0, 2.0, 4.0])
        adj = horizontal_visibility_graph(ts)
        assert adj.shape == (4, 4)
    def test_symmetric(self):
        ts = jnp.array([3.0, 1.0, 2.0, 4.0])
        adj = horizontal_visibility_graph(ts)
        np.testing.assert_allclose(adj, adj.T)
    def test_adjacent_connected(self):
        ts = jnp.array([1.0, 2.0, 3.0])
        adj = horizontal_visibility_graph(ts)
        assert float(adj[0, 1]) == 1.0
        assert float(adj[1, 2]) == 1.0

class TestVGMeasures:
    def test_degree_shape(self):
        ts = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        adj = natural_visibility_graph(ts)
        deg = vg_degree(adj)
        assert deg.shape == (5,)
    def test_mean_degree_positive(self):
        ts = jnp.array([1.0, 3.0, 2.0, 4.0, 1.0])
        adj = natural_visibility_graph(ts)
        assert vg_mean_degree(adj) > 0
    def test_clustering_bounded(self):
        ts = jnp.array([1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0])
        adj = natural_visibility_graph(ts)
        cc = vg_clustering(adj)
        assert jnp.all(cc >= 0)
        assert jnp.all(cc <= 1.0 + 1e-6)
    def test_degree_distribution(self):
        ts = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        adj = natural_visibility_graph(ts)
        degs, counts = vg_degree_distribution(adj)
        assert jnp.sum(counts) == 5
