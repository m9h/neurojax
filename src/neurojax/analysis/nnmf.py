"""Non-Negative Matrix Factorization for spectral component separation.

Decomposes state power spectra into a small number of spectral components
(e.g., alpha, beta, gamma bands) using multiplicative update NNMF.

Given state PSDs of shape (K, n_freqs, n_parcels), NNMF finds:
- Spectral components W (n_freqs, n_components): frequency profiles
- Activation maps H (n_components, K*n_parcels): spatial + state weights

This is how osl-dynamics separates oscillatory bands per brain state
for visualization in publications.

All operations are JAX-native.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr


def fit_nnmf(
    V: jnp.ndarray,
    n_components: int,
    n_iter: int = 200,
    key: jax.Array | None = None,
    eps: float = 1e-10,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fit NNMF using multiplicative update rules (Lee & Seung 2001).

    Factorizes V ≈ W @ H where V, W, H are all non-negative.

    Parameters
    ----------
    V : (n, m) non-negative matrix.
    n_components : int — number of components (rank of factorization).
    n_iter : int — number of update iterations.
    key : PRNG key for initialization.
    eps : float — small constant to prevent division by zero.

    Returns
    -------
    W : (n, n_components)
    H : (n_components, m)
    """
    if key is None:
        key = jr.PRNGKey(0)

    n, m = V.shape
    k1, k2 = jr.split(key)

    # Random positive initialization
    W = jnp.abs(jr.normal(k1, (n, n_components))) + eps
    H = jnp.abs(jr.normal(k2, (n_components, m))) + eps

    for _ in range(n_iter):
        # Update H
        numerator_H = W.T @ V
        denominator_H = W.T @ W @ H + eps
        H = H * (numerator_H / denominator_H)

        # Update W
        numerator_W = V @ H.T
        denominator_W = W @ H @ H.T + eps
        W = W * (numerator_W / denominator_W)

    return W, H


def separate_spectral_components(
    psd: jnp.ndarray,
    frequencies: jnp.ndarray,
    n_components: int = 2,
    n_iter: int = 200,
    key: jax.Array | None = None,
) -> dict:
    """Separate state PSDs into spectral components using NNMF.

    Parameters
    ----------
    psd : (n_states, n_freqs, n_parcels) — power spectra per state.
    frequencies : (n_freqs,) — frequency axis in Hz.
    n_components : int — number of spectral components to extract.
    n_iter : int — NNMF iterations.
    key : PRNG key.

    Returns
    -------
    dict with keys:
        ``"spectral_components"`` : (n_freqs, n_components)
            Frequency profile of each component.
        ``"activation_maps"`` : (n_states, n_components, n_parcels)
            Spatial activation weight per component per state.
        ``"component_psds"`` : (n_states, n_components, n_freqs)
            PSD contribution of each component per state
            (outer product of spectral component and activation weight).
    """
    K, n_freqs, n_parcels = psd.shape

    # Reshape PSD to (n_freqs, K * n_parcels) for NNMF
    V = psd.transpose(1, 0, 2).reshape(n_freqs, K * n_parcels)  # (n_freqs, K*P)
    V = jnp.maximum(V, 0.0)  # ensure non-negative

    W, H = fit_nnmf(V, n_components=n_components, n_iter=n_iter, key=key)

    # Reshape H back to (n_components, K, n_parcels)
    H_reshaped = H.reshape(n_components, K, n_parcels)

    # Activation maps: (n_states, n_components, n_parcels)
    activation_maps = H_reshaped.transpose(1, 0, 2)

    # Component PSDs per state: W[:, c] * H[c, k, :] → (n_freqs,) per component per state
    # Shape: (n_states, n_components, n_freqs)
    component_psds = jnp.einsum("fc,ckp->kcf", W, H_reshaped)
    # Average over parcels to get per-state per-component spectral profile
    # Actually, the outer product gives the full contribution
    # For a summary: component_psds[k, c, f] = W[f, c] * mean(H[c, k, :])
    component_psds = W[None, :, :].transpose(0, 2, 1) * jnp.mean(activation_maps, axis=-1)[:, :, None]
    # shape: (K, n_components, n_freqs)

    return {
        "spectral_components": W,
        "activation_maps": activation_maps,
        "component_psds": component_psds,
    }
