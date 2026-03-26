"""State spectral decomposition: HMM/DyNeMo state → power maps + coherence.

Inverts the TDE-PCA preparation pipeline to convert state covariances
(in PCA space) back to per-parcel power spectral densities and
cross-parcel coherence matrices.

Pipeline::

    state_covs (K, n_pca, n_pca)          # from HMM/DyNeMo
    → inverse PCA → C_tde (K, D, D)       # D = n_parcels * n_embeddings
    → undo TDE → autocov (K, n_emb, C, C) # lagged covariances per state
    → FFT → CPSD (K, n_freqs, C, C)       # cross-power spectral density
    → diag → PSD (K, n_freqs, C)          # power maps
    → normalize → coherence (K, n_freqs, C, C)

This follows the osl-dynamics ``analysis.spectral`` module approach.

All operations are JAX-native.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# InvertiblePCA — stores the transform for round-tripping
# ---------------------------------------------------------------------------


class InvertiblePCA:
    """PCA that stores the projection matrix for inverse transforms.

    Unlike ``prepare_pca`` in ``data.loading``, this class retains the
    principal component matrix ``Vh[:k]`` so that covariances can be
    projected back to the original feature space.

    Parameters
    ----------
    n_components : int
        Number of principal components to retain.
    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components: Optional[jnp.ndarray] = None  # (k, D)
        self.mean: Optional[jnp.ndarray] = None  # (D,)
        self.singular_values: Optional[jnp.ndarray] = None  # (k,)

    def fit_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """Fit PCA and return transformed data.

        Parameters
        ----------
        X : (n_samples, n_features)

        Returns
        -------
        X_pca : (n_samples, n_components)
        """
        k = self.n_components
        self.mean = jnp.mean(X, axis=0)
        X_c = X - self.mean

        U, S, Vh = jnp.linalg.svd(X_c, full_matrices=False)
        self.components = Vh[:k]  # (k, D)
        self.singular_values = S[:k]

        return U[:, :k] * S[:k]

    def inverse_transform(self, X_pca: jnp.ndarray) -> jnp.ndarray:
        """Project PCA data back to original space.

        Parameters
        ----------
        X_pca : (n_samples, n_components)

        Returns
        -------
        X_recon : (n_samples, n_features)
        """
        return X_pca @ self.components + self.mean

    def inverse_covariance(self, C_pca: jnp.ndarray) -> jnp.ndarray:
        """Project a covariance matrix from PCA space to original space.

        C_full = W^T @ C_pca @ W  where W = components (k, D)

        Parameters
        ----------
        C_pca : (n_components, n_components)

        Returns
        -------
        C_full : (n_features, n_features)
        """
        W = self.components  # (k, D)
        return W.T @ C_pca @ W


# ---------------------------------------------------------------------------
# TDE covariance inversion
# ---------------------------------------------------------------------------


def undo_tde_covariance(
    C_tde: jnp.ndarray,
    n_channels: int,
    n_embeddings: int,
) -> jnp.ndarray:
    """Extract lagged autocovariance blocks from a TDE covariance matrix.

    The TDE covariance ``C_tde`` has shape ``(D, D)`` where
    ``D = n_channels * n_embeddings``. It is a block matrix where the
    ``(i, j)`` block of size ``(n_channels, n_channels)`` represents
    the covariance between embedding lag ``i`` and lag ``j``.

    The autocovariance at lag ``k`` is the average of blocks where
    ``|i - j| == k``.

    Parameters
    ----------
    C_tde : (D, D) TDE-space covariance matrix.
    n_channels : int — number of parcels/channels.
    n_embeddings : int — number of TDE lags.

    Returns
    -------
    autocov : (n_embeddings, n_channels, n_channels)
        Autocovariance at lags 0, 1, ..., n_embeddings-1.
    """
    C = n_channels
    E = n_embeddings

    # Reshape to block matrix: (E, C, E, C) → blocks[i, :, j, :] = C_tde block (i, j)
    blocks = C_tde.reshape(E, C, E, C)

    autocov = jnp.zeros((E, C, C))
    for lag in range(E):
        # Average all blocks with |i - j| == lag
        count = 0
        block_sum = jnp.zeros((C, C))
        for i in range(E):
            j = i + lag
            if j < E:
                block_sum = block_sum + blocks[i, :, j, :]
                count += 1
        autocov = autocov.at[lag].set(block_sum / count)

    return autocov


def autocov_from_tde_cov(
    C_tde: jnp.ndarray,
    n_channels: int,
    n_embeddings: int,
) -> jnp.ndarray:
    """Alias for ``undo_tde_covariance``."""
    return undo_tde_covariance(C_tde, n_channels, n_embeddings)


# ---------------------------------------------------------------------------
# Autocovariance → spectral quantities
# ---------------------------------------------------------------------------


def cpsd_from_autocov(
    autocov: jnp.ndarray,
    fs: float = 1.0,
) -> jnp.ndarray:
    """Cross-power spectral density from autocovariance sequence.

    Applies the Wiener-Khinchin theorem: CPSD = FFT of autocovariance.
    Uses a one-sided spectrum (positive frequencies only).

    Parameters
    ----------
    autocov : (n_lags, n_channels, n_channels)
    fs : float — sampling frequency (Hz).

    Returns
    -------
    cpsd : (n_freqs, n_channels, n_channels) complex array.
    """
    n_lags, C, _ = autocov.shape

    # Build the full (symmetric) autocovariance for FFT:
    # R(-k) = R(k)^T
    # Full sequence: R(-(n-1)), ..., R(-1), R(0), R(1), ..., R(n-1)
    autocov_neg = jnp.flip(autocov[1:], axis=0)  # R(1)..R(n-1) reversed
    autocov_neg = jnp.transpose(autocov_neg, (0, 2, 1))  # transpose for R(-k)
    full_autocov = jnp.concatenate([autocov_neg, autocov], axis=0)
    # shape: (2*n_lags - 1, C, C)

    # FFT along the lag axis
    cpsd_full = jnp.fft.fft(full_autocov, axis=0) / fs

    # Take one-sided spectrum (positive frequencies)
    n_fft = full_autocov.shape[0]
    n_freqs = n_fft // 2 + 1
    cpsd = cpsd_full[:n_freqs]

    return cpsd


def psd_from_autocov(
    autocov: jnp.ndarray,
    fs: float = 1.0,
) -> jnp.ndarray:
    """Power spectral density (diagonal of CPSD).

    Parameters
    ----------
    autocov : (n_lags, n_channels, n_channels)
    fs : float — sampling frequency.

    Returns
    -------
    psd : (n_freqs, n_channels) real non-negative array.
    """
    cpsd = cpsd_from_autocov(autocov, fs)
    # PSD is the real part of the diagonal
    psd = jnp.real(jnp.diagonal(cpsd, axis1=-2, axis2=-1))
    # Ensure non-negative (numerical noise can make tiny negatives)
    return jnp.maximum(psd, 0.0)


def coherence_from_cpsd(cpsd: jnp.ndarray) -> jnp.ndarray:
    """Magnitude-squared coherence from cross-power spectral density.

    Coh_{ij}(f) = |CPSD_{ij}(f)|^2 / (PSD_i(f) * PSD_j(f))

    Parameters
    ----------
    cpsd : (n_freqs, n_channels, n_channels) complex.

    Returns
    -------
    coherence : (n_freqs, n_channels, n_channels) real in [0, 1].
    """
    psd = jnp.real(jnp.diagonal(cpsd, axis1=-2, axis2=-1))  # (n_freqs, C)
    # Outer product of PSDs for normalization
    psd_outer = psd[:, :, None] * psd[:, None, :]  # (n_freqs, C, C)
    # Avoid division by zero
    psd_outer = jnp.maximum(psd_outer, 1e-20)
    coh = jnp.abs(cpsd) ** 2 / psd_outer
    return jnp.clip(coh, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def get_state_spectra(
    state_covs: jnp.ndarray,
    pca: InvertiblePCA,
    n_parcels: int,
    n_embeddings: int,
    fs: float = 1.0,
) -> dict:
    """Convert HMM/DyNeMo state covariances to power maps and coherence.

    Parameters
    ----------
    state_covs : (K, n_pca, n_pca) — covariances in PCA space.
    pca : InvertiblePCA — fitted PCA transform.
    n_parcels : int — number of brain parcels/channels.
    n_embeddings : int — number of TDE lags used in preparation.
    fs : float — sampling frequency (Hz).

    Returns
    -------
    dict with keys:
        ``"psd"`` : (K, n_freqs, n_parcels) — power spectral density per state.
        ``"coherence"`` : (K, n_freqs, n_parcels, n_parcels) — coherence per state.
        ``"frequencies"`` : (n_freqs,) — frequency axis in Hz.
        ``"cpsd"`` : (K, n_freqs, n_parcels, n_parcels) — complex CPSD.
    """
    K = state_covs.shape[0]

    all_psd = []
    all_coh = []
    all_cpsd = []
    freqs = None

    for k in range(K):
        # 1. Invert PCA
        C_tde = pca.inverse_covariance(state_covs[k])

        # 2. Undo TDE → autocovariance
        autocov = undo_tde_covariance(C_tde, n_parcels, n_embeddings)

        # 3. Autocovariance → CPSD
        cpsd = cpsd_from_autocov(autocov, fs)

        # 4. PSD (diagonal)
        psd = jnp.real(jnp.diagonal(cpsd, axis1=-2, axis2=-1))
        psd = jnp.maximum(psd, 0.0)

        # 5. Coherence
        coh = coherence_from_cpsd(cpsd)

        all_psd.append(psd)
        all_coh.append(coh)
        all_cpsd.append(cpsd)

        if freqs is None:
            n_freqs = cpsd.shape[0]
            n_fft = 2 * (n_freqs - 1) + 1  # reconstruct n_fft from one-sided
            freqs = jnp.fft.rfftfreq(n_fft, d=1.0 / fs)[:n_freqs]

    return {
        "psd": jnp.stack(all_psd),
        "coherence": jnp.stack(all_coh),
        "cpsd": jnp.stack(all_cpsd),
        "frequencies": freqs,
    }
