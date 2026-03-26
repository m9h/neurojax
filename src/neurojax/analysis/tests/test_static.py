"""Tests for static analysis baseline (time-averaged, non-dynamic)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neurojax.analysis.static import (
    static_power,
    static_connectivity,
    static_summary,
)


class TestStaticPower:
    def test_shape(self):
        data = [jr.normal(jr.PRNGKey(0), (1000, 4))]
        psd, freqs = static_power(data, fs=100.0)
        assert psd.shape[1] == 4  # n_channels
        assert freqs.shape[0] == psd.shape[0]

    def test_nonnegative(self):
        data = [jr.normal(jr.PRNGKey(0), (1000, 4))]
        psd, _ = static_power(data, fs=100.0)
        assert jnp.all(psd >= 0)

    def test_multi_session_averages(self):
        """Static power from multiple sessions should be the average."""
        data = [
            jr.normal(jr.PRNGKey(0), (500, 3)),
            jr.normal(jr.PRNGKey(1), (500, 3)),
        ]
        psd, _ = static_power(data, fs=100.0)
        assert psd.shape[1] == 3


class TestStaticConnectivity:
    def test_shape(self):
        data = [jr.normal(jr.PRNGKey(0), (1000, 5))]
        conn = static_connectivity(data)
        assert conn.shape == (5, 5)

    def test_symmetric(self):
        data = [jr.normal(jr.PRNGKey(0), (1000, 4))]
        conn = static_connectivity(data)
        np.testing.assert_allclose(conn, conn.T, atol=1e-6)

    def test_diagonal_is_one(self):
        data = [jr.normal(jr.PRNGKey(0), (1000, 4))]
        conn = static_connectivity(data)
        np.testing.assert_allclose(jnp.diag(conn), 1.0, atol=1e-4)

    def test_bounded(self):
        data = [jr.normal(jr.PRNGKey(0), (1000, 4))]
        conn = static_connectivity(data)
        assert jnp.all(conn >= -1.0 - 1e-6)
        assert jnp.all(conn <= 1.0 + 1e-6)

    def test_identical_channels_perfect_correlation(self):
        s = jr.normal(jr.PRNGKey(0), (500, 1))
        data = [jnp.concatenate([s, s, s], axis=1)]
        conn = static_connectivity(data)
        np.testing.assert_allclose(conn, jnp.ones((3, 3)), atol=1e-4)


class TestStaticSummary:
    def test_returns_all_keys(self):
        data = [jr.normal(jr.PRNGKey(0), (1000, 4))]
        result = static_summary(data, fs=100.0)
        assert "psd" in result
        assert "frequencies" in result
        assert "connectivity" in result
        assert "mean" in result
        assert "std" in result

    def test_mean_shape(self):
        data = [jr.normal(jr.PRNGKey(0), (1000, 5))]
        result = static_summary(data, fs=100.0)
        assert result["mean"].shape == (5,)
        assert result["std"].shape == (5,)
