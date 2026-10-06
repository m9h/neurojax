# © NeuroJAX developers
#
# License: BSD (3-clause)
"""Multivariate autoregressive (MVAR) models and directed connectivity.

A parametric complement to the nonparametric cross-spectra in
:mod:`neurojax.analysis.connectivity_spectral`. A least-squares MVAR(p) fit gives
the AR coefficient matrices ``A_k``; from the spectral AR matrix
``Ā(f) = I − Σ_k A_k e^{-i2πf k/fs}`` and the transfer function ``H(f) = Ā(f)^{-1}``
come the two standard frequency-domain directed measures:

* **PDC** — Partial Directed Coherence (Baccalá & Sameshima 2001):
  ``π_{ij}(f) = |Ā_{ij}(f)| / sqrt(Σ_k |Ā_{kj}(f)|²)`` — directed influence
  ``j → i``, normalised over outflow from the source ``j``.
* **DTF** — Directed Transfer Function (Kamiński & Blinowska 1991):
  ``γ_{ij}(f) = |H_{ij}(f)| / sqrt(Σ_k |H_{ik}(f)|²)`` — normalised over the
  inflow to the sink ``i``.

This also provides the MAR-spectra building block OSL/osl-dynamics exposes
(parametric CSD ``S(f) = H(f) Σ H(f)^H``; see :func:`mvar_spectrum`).
"""

import jax.numpy as jnp

__all__ = [
    "fit_mvar",
    "mvar_transfer",
    "mvar_spectrum",
    "pdc",
    "dtf",
]


def fit_mvar(data: jnp.ndarray, order: int):
    """Least-squares MVAR(``order``) fit of a multichannel time series.

    Model ``x(t) = Σ_{k=1}^p A_k x(t−k) + e(t)``.

    Parameters
    ----------
    data : (T, C) real time series.
    order : int, model order ``p``.

    Returns
    -------
    A : (p, C, C) — AR matrices; ``A[k-1][i, j]`` is the effect of ``x_j(t−k)``
        on ``x_i(t)``.
    Sigma : (C, C) — residual (innovation) covariance.
    """
    T, C = data.shape
    p = order
    Y = data[p:]  # (T-p, C)
    # design: columns are [x(t-1) | x(t-2) | ... | x(t-p)], each a C-block.
    lags = [data[p - k:T - k] for k in range(1, p + 1)]
    X = jnp.concatenate(lags, axis=1)  # (T-p, p*C)
    B, _, _, _ = jnp.linalg.lstsq(X, Y, rcond=None)  # (p*C, C)
    resid = Y - X @ B
    Sigma = resid.T @ resid / resid.shape[0]
    # B block k (rows k*C:(k+1)*C) maps x(t-(k+1)) -> Y, so A_{k+1}[i,j] = B[k*C+j, i].
    A = jnp.stack([B[k * C:(k + 1) * C, :].T for k in range(p)])  # (p, C, C)
    return A, Sigma


def mvar_transfer(A: jnp.ndarray, freqs: jnp.ndarray, fs: float):
    """Spectral AR matrix ``Ā(f)`` and transfer function ``H(f) = Ā(f)^{-1}``.

    Returns ``(Abar, H)`` each of shape ``(n_freqs, C, C)`` complex.
    """
    p, C, _ = A.shape
    k = jnp.arange(1, p + 1)
    phase = jnp.exp(-2j * jnp.pi * jnp.outer(freqs, k) / fs)  # (F, p)
    eye = jnp.eye(C, dtype=jnp.complex64)[None]
    Abar = eye - jnp.einsum("fp,pij->fij", phase, A.astype(jnp.complex64))
    H = jnp.linalg.inv(Abar)
    return Abar, H


def mvar_spectrum(A: jnp.ndarray, Sigma: jnp.ndarray, freqs: jnp.ndarray,
                  fs: float) -> jnp.ndarray:
    """Parametric cross-spectral density ``S(f) = H(f) Σ H(f)^H`` (MAR spectra)."""
    _, H = mvar_transfer(A, freqs, fs)
    Hc = jnp.conjugate(jnp.swapaxes(H, -1, -2))
    return H @ Sigma.astype(jnp.complex64) @ Hc


def pdc(A: jnp.ndarray, freqs: jnp.ndarray, fs: float) -> jnp.ndarray:
    """Partial Directed Coherence ``π_{ij}(f)`` — directed influence ``j → i``."""
    Abar, _ = mvar_transfer(A, freqs, fs)
    num = jnp.abs(Abar)
    den = jnp.sqrt(jnp.sum(jnp.abs(Abar) ** 2, axis=1, keepdims=True))  # over rows
    return num / jnp.maximum(den, 1e-20)


def dtf(A: jnp.ndarray, freqs: jnp.ndarray, fs: float) -> jnp.ndarray:
    """Directed Transfer Function ``γ_{ij}(f)`` — directed influence ``j → i``."""
    _, H = mvar_transfer(A, freqs, fs)
    num = jnp.abs(H)
    den = jnp.sqrt(jnp.sum(jnp.abs(H) ** 2, axis=2, keepdims=True))  # over cols
    return num / jnp.maximum(den, 1e-20)
