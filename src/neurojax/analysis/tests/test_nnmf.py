"""Tests for NNMF spectral component separation — TDD RED phase."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neurojax.analysis.nnmf import fit_nnmf, separate_spectral_components


class TestFitNNMF:
    def test_output_shapes(self):
        """W (n_freq, n_components), H (n_components, n_parcels*n_states)."""
        key = jr.PRNGKey(0)
        V = jnp.abs(jr.normal(key, (50, 12))) + 0.01  # (n_freq, n_features)
        W, H = fit_nnmf(V, n_components=3)
        assert W.shape == (50, 3)
        assert H.shape == (3, 12)

    def test_nonnegative(self):
        V = jnp.abs(jr.normal(jr.PRNGKey(0), (30, 10))) + 0.01
        W, H = fit_nnmf(V, n_components=2)
        assert jnp.all(W >= 0)
        assert jnp.all(H >= 0)

    def test_reconstruction_improves(self):
        """W @ H should approximate V."""
        V = jnp.abs(jr.normal(jr.PRNGKey(0), (40, 8))) + 0.01
        W, H = fit_nnmf(V, n_components=3, n_iter=100)
        recon = W @ H
        error = float(jnp.mean((V - recon) ** 2))
        trivial_error = float(jnp.mean(V ** 2))
        assert error < trivial_error

    def test_more_components_better_fit(self):
        V = jnp.abs(jr.normal(jr.PRNGKey(0), (40, 8))) + 0.01
        W2, H2 = fit_nnmf(V, n_components=2, n_iter=50)
        W5, H5 = fit_nnmf(V, n_components=5, n_iter=50)
        err2 = float(jnp.mean((V - W2 @ H2) ** 2))
        err5 = float(jnp.mean((V - W5 @ H5) ** 2))
        assert err5 <= err2 + 0.01

    def test_single_component(self):
        V = jnp.abs(jr.normal(jr.PRNGKey(0), (20, 5))) + 0.01
        W, H = fit_nnmf(V, n_components=1)
        assert W.shape == (20, 1)
        assert H.shape == (1, 5)


class TestSeparateSpectralComponents:
    """Full pipeline: state PSDs → NNMF → per-component power maps."""

    @pytest.fixture
    def synthetic_psds(self):
        """3 states, 50 freq bins, 6 parcels.
        State 0: alpha peak, State 1: beta peak, State 2: broadband.
        """
        n_states, n_freqs, n_parcels = 3, 50, 6
        freqs = jnp.linspace(1, 45, n_freqs)

        psd = jnp.zeros((n_states, n_freqs, n_parcels))
        # State 0: alpha peak at 10 Hz
        alpha_peak = jnp.exp(-0.5 * ((freqs - 10) / 2) ** 2)
        psd = psd.at[0].set(alpha_peak[:, None] * jnp.ones(n_parcels))
        # State 1: beta peak at 20 Hz
        beta_peak = jnp.exp(-0.5 * ((freqs - 20) / 3) ** 2)
        psd = psd.at[1].set(beta_peak[:, None] * jnp.ones(n_parcels))
        # State 2: broadband
        psd = psd.at[2].set(jnp.ones((n_freqs, n_parcels)) * 0.3)

        return psd, freqs

    def test_output_keys(self, synthetic_psds):
        psd, freqs = synthetic_psds
        result = separate_spectral_components(psd, freqs, n_components=2)
        assert "spectral_components" in result
        assert "activation_maps" in result
        assert "component_psds" in result

    def test_spectral_components_shape(self, synthetic_psds):
        psd, freqs = synthetic_psds
        result = separate_spectral_components(psd, freqs, n_components=2)
        # W: (n_freqs, n_components) — spectral profile of each component
        assert result["spectral_components"].shape == (50, 2)

    def test_activation_maps_shape(self, synthetic_psds):
        psd, freqs = synthetic_psds
        result = separate_spectral_components(psd, freqs, n_components=2)
        # (n_states, n_components, n_parcels) — spatial activation per component per state
        assert result["activation_maps"].shape == (3, 2, 6)

    def test_component_psds_shape(self, synthetic_psds):
        psd, freqs = synthetic_psds
        result = separate_spectral_components(psd, freqs, n_components=2)
        # (n_states, n_components, n_freqs) — PSD per component per state
        assert result["component_psds"].shape == (3, 2, 50)

    def test_components_nonnegative(self, synthetic_psds):
        psd, freqs = synthetic_psds
        result = separate_spectral_components(psd, freqs, n_components=2)
        assert jnp.all(result["spectral_components"] >= 0)
        assert jnp.all(result["activation_maps"] >= 0)

    def test_two_components_separate_alpha_beta(self, synthetic_psds):
        """With 2 components, NNMF should roughly separate the alpha and beta peaks."""
        psd, freqs = synthetic_psds
        result = separate_spectral_components(psd, freqs, n_components=2)
        W = result["spectral_components"]  # (50, 2)
        # Each component should peak at a different frequency
        peak_freqs = [float(freqs[jnp.argmax(W[:, c])]) for c in range(2)]
        peak_freqs.sort()
        # One should be near 10 Hz (alpha), other near 20 Hz (beta)
        assert peak_freqs[0] < 15, f"Lower peak at {peak_freqs[0]} Hz"
        assert peak_freqs[1] > 15, f"Upper peak at {peak_freqs[1]} Hz"
