"""Static (time-averaged) analysis baseline.

Provides the same spectral and connectivity metrics as the dynamic
analysis but computed on the full time series without state segmentation.
This serves as the "is dynamics even needed?" baseline comparison.

If the dynamic model doesn't improve over static metrics, the added
complexity isn't justified.
"""

from __future__ import annotations

from typing import List

import jax.numpy as jnp

from neurojax.analysis.multitaper import multitaper_psd


def static_power(
    data: List[jnp.ndarray],
    fs: float = 1.0,
    bandwidth: float = 4.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time-averaged power spectral density across sessions.

    Parameters
    ----------
    data : list of (T_i, C) arrays.
    fs : float — sampling frequency.
    bandwidth : float — multitaper bandwidth.

    Returns
    -------
    psd : (n_freqs, C) — average PSD.
    freqs : (n_freqs,)
    """
    all_psd = []
    freqs = None
    for x in data:
        psd, freqs = multitaper_psd(x, fs=fs, bandwidth=bandwidth)
        all_psd.append(psd)

    return jnp.mean(jnp.stack(all_psd), axis=0), freqs


def static_connectivity(
    data: List[jnp.ndarray],
) -> jnp.ndarray:
    """Time-averaged Pearson correlation (functional connectivity).

    Parameters
    ----------
    data : list of (T_i, C) arrays.

    Returns
    -------
    conn : (C, C) — correlation matrix.
    """
    # Concatenate all sessions
    X = jnp.concatenate(data, axis=0)  # (N, C)
    # Pearson correlation: corr = (X^T X / N) normalized
    centered = X - jnp.mean(X, axis=0, keepdims=True)
    norms = jnp.sqrt(jnp.sum(centered ** 2, axis=0, keepdims=True))
    norms = jnp.maximum(norms, 1e-12)
    normalized = centered / norms
    return normalized.T @ normalized


def static_summary(
    data: List[jnp.ndarray],
    fs: float = 1.0,
    bandwidth: float = 4.0,
) -> dict:
    """Complete static analysis: PSD, connectivity, basic stats.

    Parameters
    ----------
    data : list of (T_i, C) arrays.
    fs : float — sampling frequency.
    bandwidth : float — multitaper bandwidth.

    Returns
    -------
    dict with keys: ``"psd"``, ``"frequencies"``, ``"connectivity"``,
    ``"mean"``, ``"std"``.
    """
    psd, freqs = static_power(data, fs=fs, bandwidth=bandwidth)
    conn = static_connectivity(data)

    X = jnp.concatenate(data, axis=0)
    mean = jnp.mean(X, axis=0)
    std = jnp.std(X, axis=0)

    return {
        "psd": psd,
        "frequencies": freqs,
        "connectivity": conn,
        "mean": mean,
        "std": std,
    }
