"""Tests for regression-based spectral analysis (DyNeMo GLM spectra)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neurojax.analysis.regression_spectra import (
    compute_regression_spectra,
    compute_spectrogram,
)


class TestComputeSpectrogram:
    def test_shape(self):
        x = jr.normal(jr.PRNGKey(0), (1000, 4))  # (T, C)
        spec, freqs, times = compute_spectrogram(
            x, fs=100.0, window_length=128, step_size=64
        )
        assert spec.ndim == 3  # (n_windows, n_freqs, n_channels)
        assert spec.shape[2] == 4
        assert freqs.shape[0] == spec.shape[1]

    def test_nonnegative(self):
        x = jr.normal(jr.PRNGKey(0), (500, 2))
        spec, _, _ = compute_spectrogram(x, fs=100.0)
        assert jnp.all(spec >= 0)

    def test_times_increasing(self):
        x = jr.normal(jr.PRNGKey(0), (1000, 3))
        _, _, times = compute_spectrogram(x, fs=100.0, window_length=128, step_size=64)
        assert jnp.all(jnp.diff(times) > 0)


class TestRegressionSpectra:
    @pytest.fixture
    def synthetic_alpha_and_data(self):
        """DyNeMo alpha + raw data for regression spectra."""
        key = jr.PRNGKey(42)
        T, C, K = 2000, 4, 3
        k1, k2 = jr.split(key)
        data = jr.normal(k1, (T, C))
        alpha = jax.nn.softmax(jr.normal(k2, (T, K)), axis=1)
        return data, alpha

    def test_output_shape(self, synthetic_alpha_and_data):
        data, alpha = synthetic_alpha_and_data
        result = compute_regression_spectra(
            data, alpha, fs=100.0, window_length=128, step_size=64
        )
        assert "psd" in result
        assert "frequencies" in result
        K = alpha.shape[1]
        C = data.shape[1]
        n_freqs = result["frequencies"].shape[0]
        assert result["psd"].shape == (K, n_freqs, C)

    def test_psd_finite(self, synthetic_alpha_and_data):
        data, alpha = synthetic_alpha_and_data
        result = compute_regression_spectra(data, alpha, fs=100.0)
        assert jnp.all(jnp.isfinite(result["psd"]))

    def test_different_modes_different_spectra(self, synthetic_alpha_and_data):
        data, alpha = synthetic_alpha_and_data
        result = compute_regression_spectra(data, alpha, fs=100.0)
        # Different modes should generally produce different spectra
        psd = result["psd"]
        assert not jnp.allclose(psd[0], psd[1], atol=1e-3)

    def test_frequencies_correct(self, synthetic_alpha_and_data):
        data, alpha = synthetic_alpha_and_data
        result = compute_regression_spectra(data, alpha, fs=100.0)
        freqs = result["frequencies"]
        assert float(jnp.min(freqs)) >= 0
        assert float(jnp.max(freqs)) <= 50.0 + 1.0  # Nyquist
