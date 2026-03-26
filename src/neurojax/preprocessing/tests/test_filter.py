"""Tests for the JAX lfilter implementation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import signal as sp_signal

from neurojax.preprocessing.filter import filter_data, lfilter


class TestLfilter1D:
    """Test 1-D filtering against scipy reference."""

    def test_fir_filter(self):
        """FIR filter (a=[1]) should match scipy."""
        b = jnp.array([0.2, 0.3, 0.5])
        a = jnp.array([1.0])
        x = jax.random.normal(jax.random.PRNGKey(0), (100,))
        y_jax = np.array(lfilter(b, a, x))
        y_scipy = sp_signal.lfilter(np.array(b), np.array(a), np.array(x))
        np.testing.assert_allclose(y_jax, y_scipy, atol=1e-5)

    def test_iir_filter(self):
        """IIR filter should match scipy."""
        # Simple first-order IIR: y[n] = x[n] + 0.5*y[n-1]
        b = jnp.array([1.0])
        a = jnp.array([1.0, -0.5])
        x = jax.random.normal(jax.random.PRNGKey(1), (200,))
        y_jax = np.array(lfilter(b, a, x))
        y_scipy = sp_signal.lfilter(np.array(b), np.array(a), np.array(x))
        np.testing.assert_allclose(y_jax, y_scipy, atol=1e-5)

    def test_butterworth_lowpass(self):
        """2nd order Butterworth lowpass should match scipy."""
        b, a = sp_signal.butter(2, 0.1)
        b_jax, a_jax = jnp.array(b), jnp.array(a)
        x = jax.random.normal(jax.random.PRNGKey(2), (500,))
        y_jax = np.array(lfilter(b_jax, a_jax, x))
        y_scipy = sp_signal.lfilter(b, a, np.array(x))
        np.testing.assert_allclose(y_jax, y_scipy, atol=1e-4)

    def test_identity_filter(self):
        """b=[1], a=[1] should pass signal unchanged."""
        x = jnp.arange(10, dtype=float)
        y = lfilter(jnp.array([1.0]), jnp.array([1.0]), x)
        np.testing.assert_allclose(y, x, atol=1e-7)

    def test_delay_filter(self):
        """b=[0,1], a=[1] should delay by one sample."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = lfilter(jnp.array([0.0, 1.0]), jnp.array([1.0]), x)
        expected = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(y, expected, atol=1e-7)


class TestLfilterBatched:
    """Test multi-dimensional input."""

    def test_2d_input(self):
        """(channels, time) input should filter each channel."""
        b, a = sp_signal.butter(2, 0.2)
        x = jax.random.normal(jax.random.PRNGKey(3), (4, 200))
        y = lfilter(jnp.array(b), jnp.array(a), x)
        assert y.shape == (4, 200)
        # Check each channel matches scipy
        for ch in range(4):
            y_ref = sp_signal.lfilter(b, a, np.array(x[ch]))
            np.testing.assert_allclose(np.array(y[ch]), y_ref, atol=1e-4)

    def test_3d_input(self):
        """(batch, channels, time) input."""
        b = jnp.array([0.5, 0.5])
        a = jnp.array([1.0])
        x = jax.random.normal(jax.random.PRNGKey(4), (2, 3, 100))
        y = lfilter(b, a, x)
        assert y.shape == (2, 3, 100)


class TestFilterDifferentiability:
    """Filter should be differentiable for gradient-based optimization."""

    def test_grad_through_filter(self):
        b = jnp.array([0.5, 0.3, 0.2])
        a = jnp.array([1.0, -0.4])
        x = jax.random.normal(jax.random.PRNGKey(5), (50,))

        def loss(b_):
            y = lfilter(b_, a, x)
            return jnp.sum(y ** 2)

        grad = jax.grad(loss)(b)
        assert grad.shape == b.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_compatible(self):
        b = jnp.array([1.0, -0.5])
        a = jnp.array([1.0, 0.3])
        x = jax.random.normal(jax.random.PRNGKey(6), (100,))
        y = jax.jit(lfilter)(b, a, x)
        assert y.shape == (100,)
        assert jnp.all(jnp.isfinite(y))


class TestFilterDataAPI:
    """Test the filter_data convenience function."""

    def test_filter_data_matches_lfilter(self):
        b = jnp.array([0.5, 0.5])
        a = jnp.array([1.0])
        x = jax.random.normal(jax.random.PRNGKey(7), (3, 100))
        y1 = filter_data(x, b, a)
        y2 = lfilter(b, a, x)
        np.testing.assert_allclose(y1, y2, atol=1e-7)
