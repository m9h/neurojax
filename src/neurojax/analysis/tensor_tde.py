"""Tensor-based Time-Delay Embedding: Tucker decomposition alternative to TDE+PCA.

Standard osl-dynamics pipeline: TDE → flatten → PCA → HMM/DyNeMo.
This module replaces the flatten+PCA step with Tucker decomposition
on the natural 3-way tensor (time × channels × lags), preserving the
multilinear structure.

Advantages over TDE+PCA:
- Separately controls spatial (channel) and temporal (lag) resolution
- Factor matrices are directly interpretable as spatial and temporal patterns
- Covariance inversion doesn't mix channel and lag dimensions
- More data-efficient: fewer parameters for the same effective rank

Usage::

    tucker = TuckerTDE(n_embeddings=15, rank_channels=10, rank_lags=8)
    X_reduced = tucker.fit_transform(data)           # (T', 80)
    model.fit([X_reduced])                            # HMM/DyNeMo
    C_tde = tucker.inverse_covariance(state_cov)      # back to TDE space
    autocov = undo_tde_covariance(C_tde, n_ch, n_emb) # then to spectra

All operations are JAX-native and use jaxctrl's HOSVD.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np
from jaxctrl import hosvd, tucker_to_tensor


def build_tde_tensor(
    data: jnp.ndarray,
    n_embeddings: int,
) -> jnp.ndarray:
    """Build a 3-way TDE tensor from raw time series.

    Instead of flattening TDE to (T', C*L), keep the structure as
    a tensor (T', C, L) where T' = T - L + 1.

    Parameters
    ----------
    data : (T, C) — raw time series.
    n_embeddings : int — number of lags (L).

    Returns
    -------
    tensor : (T', C, L) — 3-way TDE tensor where tensor[t, c, l] = data[t+l, c].
    """
    T, C = data.shape
    L = n_embeddings
    T_out = T - L + 1

    # Build index array for lag dimension
    # tensor[t, c, l] = data[t + l, c]
    t_indices = jnp.arange(T_out)[:, None] + jnp.arange(L)[None, :]  # (T', L)
    # Gather: data[t_indices, :] gives (T', L, C), then transpose to (T', C, L)
    tensor = data[t_indices]  # (T', L, C)

    return tensor


class TuckerTDE:
    """Tucker decomposition on the TDE tensor.

    Replaces TDE+PCA: instead of flattening the (time × channel × lag)
    tensor and applying matrix PCA, we apply Tucker decomposition
    (HOSVD) to reduce channels and lags independently.

    The reduced representation has dimensionality
    ``rank_channels * rank_lags`` (analogous to n_pca_components).

    Parameters
    ----------
    n_embeddings : int
        Number of TDE lags.
    rank_channels : int
        Number of channel components to retain.
    rank_lags : int
        Number of lag components to retain.
    """

    def __init__(
        self,
        n_embeddings: int,
        rank_channels: int,
        rank_lags: int,
    ):
        self.n_embeddings = n_embeddings
        self.rank_channels = rank_channels
        self.rank_lags = rank_lags

        # Stored after fit
        self.U_channels: Optional[jnp.ndarray] = None  # (C, rank_channels)
        self.U_lags: Optional[jnp.ndarray] = None  # (L, rank_lags)
        self.n_channels: Optional[int] = None
        self._mean: Optional[jnp.ndarray] = None  # (C, L) mean of tensor

    def fit_transform(self, data: jnp.ndarray) -> jnp.ndarray:
        """Build TDE tensor, fit Tucker, return reduced representation.

        Parameters
        ----------
        data : (T, C) — raw time series.

        Returns
        -------
        X_reduced : (T', rank_channels * rank_lags) — reduced data
            suitable for HMM/DyNeMo.
        """
        tensor = build_tde_tensor(data, self.n_embeddings)
        T_out, L, C = tensor.shape  # (T', L, C)
        self.n_channels = C

        # Centre the tensor (subtract mean across time)
        self._mean = jnp.mean(tensor, axis=0)  # (L, C)
        tensor_c = tensor - self._mean[None, :, :]

        # HOSVD on the centred tensor
        # Mode-1 (lags): unfold to (L, T'*C)
        mode1 = tensor_c.transpose(1, 0, 2).reshape(L, -1)  # (L, T'*C)
        U1, S1, _ = jnp.linalg.svd(mode1, full_matrices=False)
        self.U_lags = U1[:, :self.rank_lags]  # (L, r_l)

        # Mode-2 (channels): unfold to (C, T'*L)
        mode2 = tensor_c.transpose(2, 0, 1).reshape(C, -1)  # (C, T'*L)
        U2, S2, _ = jnp.linalg.svd(mode2, full_matrices=False)
        self.U_channels = U2[:, :self.rank_channels]  # (C, r_c)

        # Project: core[t, i, j] = sum_{l,c} tensor[t,l,c] * U_lags[l,i] * U_ch[c,j]
        projected = jnp.einsum("tlc,li,cj->tij", tensor_c,
                                self.U_lags, self.U_channels)
        # projected: (T', r_l, r_c)

        # Flatten to (T', r_c * r_l) for HMM/DyNeMo
        # Use (r_c, r_l) order to match parameter convention
        projected = projected.transpose(0, 2, 1)  # (T', r_c, r_l)
        X_reduced = projected.reshape(T_out, -1)

        return X_reduced

    def inverse_transform(self, X_reduced: jnp.ndarray) -> jnp.ndarray:
        """Project reduced data back to TDE space (flattened).

        Parameters
        ----------
        X_reduced : (T', rank_channels * rank_lags)

        Returns
        -------
        X_tde : (T', n_channels * n_embeddings) — reconstructed flat TDE.
        """
        T_out = X_reduced.shape[0]
        r_c = self.rank_channels
        r_l = self.rank_lags

        # Unflatten to (T', r_c, r_l)
        core = X_reduced.reshape(T_out, r_c, r_l)

        # Reconstruct: tensor[t,l,c] = sum_{i,j} core[t,j,i] * U_lag[l,i] * U_ch[c,j]
        # core is (T', r_c, r_l) → transpose to (T', r_l, r_c) for einsum
        core_t = core.transpose(0, 2, 1)  # (T', r_l, r_c)
        recon = jnp.einsum("tij,li,cj->tlc", core_t,
                            self.U_lags, self.U_channels)
        # Add mean back
        recon = recon + self._mean[None, :, :]

        # Flatten to (T', L*C) matching prepare_tde layout
        return recon.reshape(T_out, -1)

    def inverse_covariance(self, C_reduced: jnp.ndarray) -> jnp.ndarray:
        """Project a covariance from Tucker space to TDE space.

        The covariance in Tucker space has shape (r_c*r_l, r_c*r_l).
        We project it back using the Kronecker structure:
        C_tde = (U_lag ⊗ U_ch) @ C_reduced @ (U_lag ⊗ U_ch)^T

        Parameters
        ----------
        C_reduced : (r_c * r_l, r_c * r_l) — covariance in Tucker space.

        Returns
        -------
        C_tde : (C*L, C*L) — covariance in TDE space.
        """
        r_c = self.rank_channels
        r_l = self.rank_lags
        C = self.n_channels
        L = self.n_embeddings

        # Reduced layout is (r_c, r_l) flattened. Reshape to 4-index.
        C_4d = C_reduced.reshape(r_c, r_l, r_c, r_l)

        # Project back to TDE space via Kronecker structure.
        # Build the projection matrix P = U_lags ⊗ U_channels, shape (L*C, r_l*r_c)
        # Then C_tde = P @ C_reduced_reordered @ P^T
        # Reorder reduced cov from (r_c, r_l, r_c, r_l) to (r_l, r_c, r_l, r_c)
        # since flat layout is (r_c * r_l) but TDE flat is (L * C)
        C_reordered = C_4d.transpose(1, 0, 3, 2).reshape(r_l * r_c, r_l * r_c)

        # Kronecker: P = U_lags ⊗ U_channels
        P = jnp.kron(self.U_lags, self.U_channels)  # (L*C, r_l*r_c)

        C_tde = P @ C_reordered @ P.T

        return C_tde
