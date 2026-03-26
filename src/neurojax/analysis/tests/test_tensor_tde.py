"""Tests for tensor-based TDE preparation (Tucker/TT alternatives to TDE+PCA).

Compares TuckerTDE and TensorTrainTDE against the standard TDE+PCA pipeline
on the same synthetic data, verifying shapes, invertibility, and that HMM
state covariances can be decomposed through both pathways.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neurojax.analysis.tensor_tde import (
    TuckerTDE,
    build_tde_tensor,
)
from neurojax.analysis.state_spectra import InvertiblePCA
from neurojax.data.loading import prepare_tde


# ---------------------------------------------------------------------------
# build_tde_tensor: raw data → 3-way tensor (time × channels × lags)
# ---------------------------------------------------------------------------


class TestBuildTDETensor:
    def test_shape(self):
        data = jr.normal(jr.PRNGKey(0), (500, 6))  # (T, C)
        tensor = build_tde_tensor(data, n_embeddings=10)
        assert tensor.shape == (491, 10, 6)  # (T-L+1, L, C)

    def test_matches_flat_tde(self):
        """Reshaping the tensor (T', L, C) to (T', L*C) should match flat TDE."""
        data = jr.normal(jr.PRNGKey(0), (200, 4))
        n_emb = 7
        tensor = build_tde_tensor(data, n_embeddings=n_emb)
        flat_tde = prepare_tde(data, n_embeddings=n_emb)
        # Tensor is (T', L, C), flat TDE is (T', L*C) in lags-major order
        tensor_flat = tensor.reshape(tensor.shape[0], -1)
        np.testing.assert_allclose(tensor_flat, flat_tde, atol=1e-5)

    def test_lag_structure(self):
        """tensor[t, l, c] should equal data[t+l, c]."""
        data = jnp.arange(30).reshape(10, 3).astype(float)
        tensor = build_tde_tensor(data, n_embeddings=4)
        # Check specific values: tensor[t, l, c] = data[t+l, c]
        assert float(tensor[0, 0, 0]) == float(data[0, 0])
        assert float(tensor[0, 3, 0]) == float(data[3, 0])
        assert float(tensor[2, 1, 1]) == float(data[3, 1])

    def test_single_lag(self):
        data = jr.normal(jr.PRNGKey(0), (100, 5))
        tensor = build_tde_tensor(data, n_embeddings=1)
        assert tensor.shape == (100, 1, 5)
        np.testing.assert_allclose(tensor[:, 0, :], data, atol=1e-6)


# ---------------------------------------------------------------------------
# TuckerTDE: Tucker decomposition on the 3-way TDE tensor
# ---------------------------------------------------------------------------


class TestTuckerTDE:
    def test_fit_transform_shape(self):
        data = jr.normal(jr.PRNGKey(0), (500, 6))
        tucker = TuckerTDE(n_embeddings=10, rank_channels=4, rank_lags=5)
        X_reduced = tucker.fit_transform(data)
        # Output: (T', rank_channels * rank_lags)
        assert X_reduced.shape == (491, 4 * 5)

    def test_inverse_transform_shape(self):
        data = jr.normal(jr.PRNGKey(0), (500, 6))
        tucker = TuckerTDE(n_embeddings=10, rank_channels=4, rank_lags=5)
        X_reduced = tucker.fit_transform(data)
        X_recon = tucker.inverse_transform(X_reduced)
        # Should recover the TDE tensor shape flattened: (T', C*L)
        assert X_recon.shape == (491, 6 * 10)

    def test_roundtrip_reconstruction(self):
        """With full rank, Tucker roundtrip should be near-exact."""
        data = jr.normal(jr.PRNGKey(0), (200, 4))
        n_emb = 5
        tucker = TuckerTDE(n_embeddings=n_emb, rank_channels=4, rank_lags=5)
        X_reduced = tucker.fit_transform(data)
        X_recon = tucker.inverse_transform(X_reduced)
        X_orig = prepare_tde(data, n_embeddings=n_emb)
        np.testing.assert_allclose(X_recon, X_orig, atol=1e-3)

    def test_covariance_inverse(self):
        """Covariance in Tucker space → invert to TDE space."""
        data = jr.normal(jr.PRNGKey(0), (500, 5))
        tucker = TuckerTDE(n_embeddings=8, rank_channels=3, rank_lags=4)
        X_reduced = tucker.fit_transform(data)
        C_tucker = jnp.cov(X_reduced.T)
        C_tde = tucker.inverse_covariance(C_tucker)
        # Should be (C*L, C*L) = (40, 40)
        assert C_tde.shape == (40, 40)
        # Symmetric
        np.testing.assert_allclose(C_tde, C_tde.T, atol=1e-5)

    def test_core_and_factors_stored(self):
        data = jr.normal(jr.PRNGKey(0), (300, 4))
        tucker = TuckerTDE(n_embeddings=6, rank_channels=3, rank_lags=4)
        tucker.fit_transform(data)
        assert tucker.U_channels is not None
        assert tucker.U_lags is not None
        assert tucker.U_channels.shape == (4, 3)
        assert tucker.U_lags.shape == (6, 4)

    def test_variance_retained(self):
        """Tucker should retain most variance when keeping most ranks."""
        data = jr.normal(jr.PRNGKey(0), (500, 6))
        # Keep all 6 channels, 8 of 10 lags → should retain most variance
        tucker = TuckerTDE(n_embeddings=10, rank_channels=6, rank_lags=8)
        X_reduced = tucker.fit_transform(data)
        X_recon = tucker.inverse_transform(X_reduced)
        X_orig = prepare_tde(data, n_embeddings=10)
        error = float(jnp.mean((X_orig - X_recon) ** 2))
        signal = float(jnp.mean(X_orig ** 2))
        assert error / signal < 0.3  # retain >70% variance


# ---------------------------------------------------------------------------
# Comparison: TuckerTDE vs PCA on the same data
# ---------------------------------------------------------------------------


class TestTuckerVsPCA:
    @pytest.fixture
    def synthetic_data(self):
        return jr.normal(jr.PRNGKey(42), (1000, 8))

    def test_same_reduced_dimensionality(self, synthetic_data):
        """Both methods should produce the same output dimensionality."""
        n_emb = 10
        n_pca = 20
        # PCA path
        flat_tde = prepare_tde(synthetic_data, n_embeddings=n_emb)
        pca = InvertiblePCA(n_components=n_pca)
        X_pca = pca.fit_transform(flat_tde)

        # Tucker path: same total reduced dim
        # rank_channels * rank_lags = n_pca
        tucker = TuckerTDE(n_embeddings=n_emb, rank_channels=5, rank_lags=4)
        X_tucker = tucker.fit_transform(synthetic_data)

        assert X_pca.shape[1] == X_tucker.shape[1] == 20

    def test_both_covariances_invertible(self, synthetic_data):
        """Both pathways should produce valid covariance inverses."""
        n_emb = 10
        flat_tde = prepare_tde(synthetic_data, n_embeddings=n_emb)

        # PCA path
        pca = InvertiblePCA(n_components=20)
        X_pca = pca.fit_transform(flat_tde)
        C_pca_inv = pca.inverse_covariance(jnp.cov(X_pca.T))

        # Tucker path
        tucker = TuckerTDE(n_embeddings=n_emb, rank_channels=5, rank_lags=4)
        X_tucker = tucker.fit_transform(synthetic_data)
        C_tucker_inv = tucker.inverse_covariance(jnp.cov(X_tucker.T))

        # Both should be symmetric positive semi-definite
        assert C_pca_inv.shape == C_tucker_inv.shape
        np.testing.assert_allclose(C_pca_inv, C_pca_inv.T, atol=1e-5)
        np.testing.assert_allclose(C_tucker_inv, C_tucker_inv.T, atol=1e-5)

    def test_tucker_factors_are_interpretable(self, synthetic_data):
        """Tucker factor matrices have clear interpretation:
        U_channels shows which channels contribute to each component,
        U_lags shows which temporal lags contribute."""
        tucker = TuckerTDE(n_embeddings=10, rank_channels=4, rank_lags=5)
        tucker.fit_transform(synthetic_data)
        # Channel factors: (n_channels, rank_channels) — spatial patterns
        assert tucker.U_channels.shape == (8, 4)
        # Lag factors: (n_embeddings, rank_lags) — temporal patterns
        assert tucker.U_lags.shape == (10, 5)
        # Both should have orthonormal columns
        UtU_ch = tucker.U_channels.T @ tucker.U_channels
        np.testing.assert_allclose(UtU_ch, jnp.eye(4), atol=0.05)
        UtU_lag = tucker.U_lags.T @ tucker.U_lags
        np.testing.assert_allclose(UtU_lag, jnp.eye(5), atol=0.05)
