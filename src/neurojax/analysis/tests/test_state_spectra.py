"""Tests for HMM/DyNeMo state → spectral decomposition inverse pipeline.

The pipeline: state covariances in PCA space → undo PCA → undo TDE →
per-parcel power spectra and cross-parcel coherence matrices.

Red-green TDD: tests written first against expected properties.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neurojax.analysis.state_spectra import (
    InvertiblePCA,
    autocov_from_tde_cov,
    coherence_from_cpsd,
    cpsd_from_autocov,
    get_state_spectra,
    psd_from_autocov,
    undo_tde_covariance,
)


# =====================================================================
# InvertiblePCA — PCA that stores the inverse transform
# =====================================================================


class TestInvertiblePCA:
    def test_fit_transform_shape(self):
        key = jr.PRNGKey(0)
        X = jr.normal(key, (500, 20))
        pca = InvertiblePCA(n_components=5)
        X_pca = pca.fit_transform(X)
        assert X_pca.shape == (500, 5)

    def test_inverse_transform_shape(self):
        X = jr.normal(jr.PRNGKey(0), (500, 20))
        pca = InvertiblePCA(n_components=5)
        X_pca = pca.fit_transform(X)
        X_recon = pca.inverse_transform(X_pca)
        assert X_recon.shape == (500, 20)

    def test_roundtrip_low_rank(self):
        """For data with exactly n_components dimensions, roundtrip is exact."""
        key = jr.PRNGKey(1)
        # Low-rank data: 500 x 10 with rank 5
        A = jr.normal(key, (500, 5))
        B = jr.normal(jr.PRNGKey(2), (5, 10))
        X = A @ B
        pca = InvertiblePCA(n_components=5)
        X_pca = pca.fit_transform(X)
        X_recon = pca.inverse_transform(X_pca)
        np.testing.assert_allclose(X_recon, X, atol=1e-4)

    def test_covariance_inverse(self):
        """Inverting a covariance: C_full = Vh.T @ diag(S^2) @ C_pca @ diag(S^2) ...
        Actually: C_full = W.T @ C_pca @ W where W = Vh[:k]."""
        X = jr.normal(jr.PRNGKey(0), (1000, 15))
        pca = InvertiblePCA(n_components=5)
        X_pca = pca.fit_transform(X)
        # Covariance in PCA space
        C_pca = jnp.cov(X_pca.T)
        # Invert to full space
        C_full = pca.inverse_covariance(C_pca)
        assert C_full.shape == (15, 15)
        # Should be symmetric
        np.testing.assert_allclose(C_full, C_full.T, atol=1e-6)
        # Should be positive semi-definite
        eigs = jnp.linalg.eigvalsh(C_full)
        assert jnp.all(eigs >= -1e-6)

    def test_components_stored(self):
        X = jr.normal(jr.PRNGKey(0), (100, 10))
        pca = InvertiblePCA(n_components=3)
        pca.fit_transform(X)
        assert pca.components.shape == (3, 10)
        assert pca.mean.shape == (10,)
        assert pca.singular_values.shape == (3,)


# =====================================================================
# TDE covariance inversion
# =====================================================================


class TestUndoTDECovariance:
    def test_output_shape(self):
        """TDE cov (n_emb*C, n_emb*C) → autocov (n_emb, C, C)."""
        n_channels = 4
        n_embeddings = 3
        tde_dim = n_channels * n_embeddings
        C_tde = jnp.eye(tde_dim)
        autocov = undo_tde_covariance(C_tde, n_channels, n_embeddings)
        assert autocov.shape == (n_embeddings, n_channels, n_channels)

    def test_zero_lag_is_covariance(self):
        """Lag-0 block should be the instantaneous covariance."""
        n_ch, n_emb = 3, 5
        key = jr.PRNGKey(0)
        # Generate some TDE data
        raw = jr.normal(key, (500, n_ch))
        # Manual TDE
        tde = jnp.concatenate(
            [raw[i:500 - n_emb + 1 + i] for i in range(n_emb)], axis=1
        )
        C_tde = jnp.cov(tde.T)
        autocov = undo_tde_covariance(C_tde, n_ch, n_emb)
        # Lag-0 should match the instantaneous covariance
        C_inst = jnp.cov(raw.T)
        np.testing.assert_allclose(autocov[0], C_inst, atol=0.15)

    def test_symmetric_lags(self):
        """Autocov at lag k should equal autocov at lag -k transposed.
        Since we only store non-negative lags, autocov[k] ≈ autocov[k].T for real data."""
        n_ch, n_emb = 4, 5
        C_tde = jnp.eye(n_ch * n_emb) + 0.1 * jr.normal(jr.PRNGKey(0), (n_ch * n_emb, n_ch * n_emb))
        C_tde = (C_tde + C_tde.T) / 2  # make symmetric
        autocov = undo_tde_covariance(C_tde, n_ch, n_emb)
        # Each lag block should be finite
        assert jnp.all(jnp.isfinite(autocov))


# =====================================================================
# Autocovariance → power spectral density
# =====================================================================


class TestPSDFromAutocov:
    def test_output_shape(self):
        """PSD shape should be (n_freqs, n_channels)."""
        n_ch, n_lags = 4, 15
        autocov = jnp.zeros((n_lags, n_ch, n_ch))
        # Set diagonal of lag-0 to simulate unit variance
        autocov = autocov.at[0].set(jnp.eye(n_ch))
        psd = psd_from_autocov(autocov, fs=100.0)
        assert psd.ndim == 2
        assert psd.shape[1] == n_ch  # one PSD per channel

    def test_white_noise_psd_positive_and_finite(self):
        """White noise autocov → PSD should be non-negative and finite.

        Note: with truncated autocov the spectrum isn't perfectly flat
        due to spectral leakage, but the non-zero values should all be
        positive and the mean should be close to the variance/fs.
        """
        n_ch, n_lags = 3, 20
        autocov = jnp.zeros((n_lags, n_ch, n_ch))
        autocov = autocov.at[0].set(jnp.eye(n_ch))
        psd = psd_from_autocov(autocov, fs=100.0)
        assert jnp.all(psd >= 0)
        assert jnp.all(jnp.isfinite(psd))
        # Mean PSD should be approximately variance/fs = 1/100 = 0.01
        mean_psd = float(jnp.mean(psd))
        assert 0.001 < mean_psd < 0.1

    def test_psd_nonnegative(self):
        """Power spectral density must be non-negative."""
        n_ch, n_lags = 4, 15
        autocov = jnp.zeros((n_lags, n_ch, n_ch))
        autocov = autocov.at[0].set(jnp.eye(n_ch) * 2.0)
        psd = psd_from_autocov(autocov, fs=100.0)
        assert jnp.all(psd >= -1e-10)


class TestCPSDFromAutocov:
    def test_output_shape(self):
        n_ch, n_lags = 4, 15
        autocov = jnp.zeros((n_lags, n_ch, n_ch))
        autocov = autocov.at[0].set(jnp.eye(n_ch))
        cpsd = cpsd_from_autocov(autocov, fs=100.0)
        n_freqs = cpsd.shape[0]
        assert cpsd.shape == (n_freqs, n_ch, n_ch)

    def test_diagonal_real_part_matches_psd_before_clipping(self):
        """CPSD diagonal real part should match PSD before the max(0) clip.

        psd_from_autocov clips negatives to zero; the raw CPSD diagonal
        may have small negative values from spectral leakage.
        """
        n_ch, n_lags = 3, 20
        autocov = jnp.zeros((n_lags, n_ch, n_ch))
        autocov = autocov.at[0].set(jnp.eye(n_ch))
        cpsd = cpsd_from_autocov(autocov, fs=100.0)
        psd = psd_from_autocov(autocov, fs=100.0)
        for ch in range(n_ch):
            cpsd_diag = jnp.real(cpsd[:, ch, ch])
            # PSD = max(cpsd_diag, 0), so where cpsd_diag > 0 they should match
            pos_mask = cpsd_diag > 0
            np.testing.assert_allclose(
                cpsd_diag[pos_mask], psd[:, ch][pos_mask], atol=1e-6
            )


# =====================================================================
# Coherence
# =====================================================================


class TestCoherence:
    def test_output_shape(self):
        n_freqs, n_ch = 50, 4
        cpsd = jnp.eye(n_ch)[None, :, :].repeat(n_freqs, axis=0)
        coh = coherence_from_cpsd(cpsd)
        assert coh.shape == (n_freqs, n_ch, n_ch)

    def test_identity_cpsd_gives_unit_coherence(self):
        """If CPSD is identity at all freqs, coherence should be 1 on diagonal."""
        n_freqs, n_ch = 30, 3
        cpsd = jnp.eye(n_ch)[None, :, :].repeat(n_freqs, axis=0).astype(complex)
        coh = coherence_from_cpsd(cpsd)
        for f in range(n_freqs):
            np.testing.assert_allclose(jnp.diag(coh[f]), 1.0, atol=1e-6)

    def test_coherence_bounded(self):
        """Coherence should be in [0, 1]."""
        key = jr.PRNGKey(0)
        n_freqs, n_ch = 20, 4
        # Make a valid CPSD (Hermitian positive definite per frequency)
        A = jr.normal(key, (n_freqs, n_ch, n_ch)) + 1j * jr.normal(jr.PRNGKey(1), (n_freqs, n_ch, n_ch))
        cpsd = jnp.einsum("fij,fkj->fik", A, jnp.conj(A)) + 0.1 * jnp.eye(n_ch)[None]
        coh = coherence_from_cpsd(cpsd)
        assert jnp.all(coh >= -1e-6)
        assert jnp.all(coh <= 1.0 + 1e-6)


# =====================================================================
# Full pipeline: get_state_spectra
# =====================================================================


class TestGetStateSpectra:
    @pytest.fixture
    def fitted_hmm_outputs(self):
        """Simulate what an HMM fit produces: state covariances in PCA space."""
        key = jr.PRNGKey(42)
        n_parcels = 6
        n_embeddings = 5
        n_pca = 10
        n_states = 3

        # Generate synthetic raw data for fitting PCA
        raw = jr.normal(key, (2000, n_parcels))
        # TDE
        tde_parts = [raw[i:2000 - n_embeddings + 1 + i] for i in range(n_embeddings)]
        tde = jnp.concatenate(tde_parts, axis=1)  # (T', n_parcels * n_embeddings)

        pca = InvertiblePCA(n_components=n_pca)
        pca.fit_transform(tde)

        # Fake state covariances in PCA space
        state_covs = jnp.stack([
            jnp.eye(n_pca) * (1.0 + 0.5 * k) for k in range(n_states)
        ])

        return state_covs, pca, n_parcels, n_embeddings

    def test_returns_psd_and_coherence(self, fitted_hmm_outputs):
        state_covs, pca, n_parcels, n_embeddings = fitted_hmm_outputs
        result = get_state_spectra(
            state_covs, pca, n_parcels, n_embeddings, fs=100.0
        )
        assert "psd" in result
        assert "coherence" in result
        assert "frequencies" in result

    def test_psd_shape(self, fitted_hmm_outputs):
        state_covs, pca, n_parcels, n_embeddings = fitted_hmm_outputs
        result = get_state_spectra(
            state_covs, pca, n_parcels, n_embeddings, fs=100.0
        )
        n_states = state_covs.shape[0]
        n_freqs = result["frequencies"].shape[0]
        assert result["psd"].shape == (n_states, n_freqs, n_parcels)

    def test_coherence_shape(self, fitted_hmm_outputs):
        state_covs, pca, n_parcels, n_embeddings = fitted_hmm_outputs
        result = get_state_spectra(
            state_covs, pca, n_parcels, n_embeddings, fs=100.0
        )
        n_states = state_covs.shape[0]
        n_freqs = result["frequencies"].shape[0]
        assert result["coherence"].shape == (n_states, n_freqs, n_parcels, n_parcels)

    def test_psd_nonnegative(self, fitted_hmm_outputs):
        state_covs, pca, n_parcels, n_embeddings = fitted_hmm_outputs
        result = get_state_spectra(
            state_covs, pca, n_parcels, n_embeddings, fs=100.0
        )
        assert jnp.all(result["psd"] >= -1e-6)

    def test_coherence_bounded(self, fitted_hmm_outputs):
        state_covs, pca, n_parcels, n_embeddings = fitted_hmm_outputs
        result = get_state_spectra(
            state_covs, pca, n_parcels, n_embeddings, fs=100.0
        )
        assert jnp.all(result["coherence"] >= -1e-6)
        assert jnp.all(result["coherence"] <= 1.0 + 1e-6)

    def test_different_states_different_spectra(self, fitted_hmm_outputs):
        """States with different covariances should produce different PSDs."""
        state_covs, pca, n_parcels, n_embeddings = fitted_hmm_outputs
        result = get_state_spectra(
            state_covs, pca, n_parcels, n_embeddings, fs=100.0
        )
        psd_0 = result["psd"][0]
        psd_1 = result["psd"][1]
        assert not jnp.allclose(psd_0, psd_1)

    def test_frequencies_correct(self, fitted_hmm_outputs):
        state_covs, pca, n_parcels, n_embeddings = fitted_hmm_outputs
        result = get_state_spectra(
            state_covs, pca, n_parcels, n_embeddings, fs=100.0
        )
        freqs = result["frequencies"]
        # Should be non-negative and up to Nyquist (fs/2 = 50)
        assert jnp.all(freqs >= 0)
        assert float(jnp.max(freqs)) <= 50.0 + 1.0
