"""Tests for surrogate data generation and significance testing."""
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from neurojax.analysis.surrogates import (
    aaft_surrogate, block_shuffle_surrogate, phase_randomized_surrogate,
    shuffle_surrogate, surrogate_test,
)

class TestPhaseRandomized:
    def test_shape_1d(self):
        x = jr.normal(jr.PRNGKey(0), (200,))
        s = phase_randomized_surrogate(x, jr.PRNGKey(1))
        assert s.shape == (200,)
    def test_shape_2d(self):
        x = jr.normal(jr.PRNGKey(0), (200, 3))
        s = phase_randomized_surrogate(x, jr.PRNGKey(1))
        assert s.shape == (200, 3)
    def test_preserves_power_spectrum(self):
        x = jr.normal(jr.PRNGKey(0), (500,))
        s = phase_randomized_surrogate(x, jr.PRNGKey(1))
        psd_orig = jnp.abs(jnp.fft.rfft(x)) ** 2
        psd_surr = jnp.abs(jnp.fft.rfft(s)) ** 2
        np.testing.assert_allclose(psd_orig, psd_surr, rtol=1e-4)
    def test_changes_temporal_structure(self):
        x = jr.normal(jr.PRNGKey(0), (500,))
        s = phase_randomized_surrogate(x, jr.PRNGKey(1))
        assert not jnp.allclose(x, s)
    def test_reproducible_with_same_key(self):
        x = jr.normal(jr.PRNGKey(0), (100,))
        s1 = phase_randomized_surrogate(x, jr.PRNGKey(42))
        s2 = phase_randomized_surrogate(x, jr.PRNGKey(42))
        np.testing.assert_allclose(s1, s2)

class TestAAFT:
    def test_shape(self):
        x = jr.normal(jr.PRNGKey(0), (200,))
        s = aaft_surrogate(x, jr.PRNGKey(1))
        assert s.shape == (200,)
    def test_preserves_amplitude_distribution(self):
        x = jr.normal(jr.PRNGKey(0), (500,))
        s = aaft_surrogate(x, jr.PRNGKey(1))
        # Sorted values should match (AAFT preserves amplitude dist exactly)
        np.testing.assert_allclose(jnp.sort(x), jnp.sort(s), atol=1e-5)
    def test_multichannel(self):
        x = jr.normal(jr.PRNGKey(0), (200, 3))
        s = aaft_surrogate(x, jr.PRNGKey(1))
        assert s.shape == (200, 3)

class TestShuffle:
    def test_shape(self):
        x = jr.normal(jr.PRNGKey(0), (100,))
        s = shuffle_surrogate(x, jr.PRNGKey(1))
        assert s.shape == (100,)
    def test_same_values(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = shuffle_surrogate(x, jr.PRNGKey(0))
        np.testing.assert_allclose(jnp.sort(s), jnp.sort(x))
    def test_destroys_order(self):
        x = jnp.arange(100, dtype=float)
        s = shuffle_surrogate(x, jr.PRNGKey(0))
        assert not jnp.allclose(s, x)

class TestBlockShuffle:
    def test_shape(self):
        x = jr.normal(jr.PRNGKey(0), (200,))
        s = block_shuffle_surrogate(x, jr.PRNGKey(1), block_size=50)
        assert s.shape == (200,)
    def test_preserves_local_structure(self):
        x = jr.normal(jr.PRNGKey(0), (200,))
        s = block_shuffle_surrogate(x, jr.PRNGKey(1), block_size=50)
        # Each block of 50 should appear somewhere intact in the surrogate
        orig_blocks = [np.asarray(x[i*50:(i+1)*50]) for i in range(4)]
        surr_blocks = [np.asarray(s[i*50:(i+1)*50]) for i in range(4)]
        matches = 0
        for sb in surr_blocks:
            for ob in orig_blocks:
                if np.allclose(sb, ob):
                    matches += 1
                    break
        assert matches == 4  # all blocks found

class TestSurrogateTest:
    def test_correlated_significant(self):
        key = jr.PRNGKey(0)
        x = jr.normal(key, (200,))
        # Statistic: lag-1 autocorrelation (should be near 0 for white noise)
        # Use a signal with structure
        t = jnp.arange(200) / 10.0
        x = jnp.sin(t)  # periodic → high autocorrelation
        def acf1(data):
            return float(jnp.corrcoef(data[:-1], data[1:])[0, 1])
        result = surrogate_test(x, acf1, shuffle_surrogate, n_surrogates=50)
        assert "p_value" in result
        assert "observed" in result
        assert result["observed"] > 0.5  # periodic signal has high ACF
    def test_independent_not_significant(self):
        x = jr.normal(jr.PRNGKey(42), (200,))
        def acf1(data):
            return float(jnp.abs(jnp.corrcoef(data[:-1], data[1:])[0, 1]))
        result = surrogate_test(x, acf1, shuffle_surrogate, n_surrogates=50)
        # White noise ACF should not be significant
        assert result["p_value"] > 0.01
