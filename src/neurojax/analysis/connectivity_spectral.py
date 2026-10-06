# © NeuroJAX developers
#
# License: BSD (3-clause)
"""Partial coherence and imaginary coherency from a cross-spectral density.

These complement the magnitude-squared coherence already provided by
:func:`neurojax.analysis.state_spectra.coherence_from_cpsd` and
:func:`neurojax.analysis.multitaper.multitaper_coherence`. Both functions here
take a cross-spectral density ``cpsd`` of shape ``(n_freqs, C, C)`` (as returned
by :func:`neurojax.analysis.multitaper.multitaper_cpsd`), so they drop straight
into the existing spectral pipeline.

* **Partial coherence** conditions out all other channels via the inverse CSD
  (the spectral precision matrix), isolating the direct association between a
  pair — the spectral analogue of partial correlation.
* **Imaginary coherency** (Nolte et al. 2004, *Clin. Neurophysiol.* 115:2292) is
  the imaginary part of the complex coherency. Zero-lag/instantaneous mixing
  (volume conduction, a common reference) contributes only to the real part, so
  the imaginary coherency is robust to those spurious zero-lag couplings.
"""

import jax.numpy as jnp

__all__ = [
    "partial_coherence_from_cpsd",
    "imaginary_coherency_from_cpsd",
]


def partial_coherence_from_cpsd(cpsd: jnp.ndarray, eps: float = 1e-10) -> jnp.ndarray:
    """Partial coherence from a cross-power spectral density.

    With ``P(f) = CPSD(f)^{-1}`` the spectral precision matrix,

        PartialCoh_{ij}(f) = |P_{ij}(f)|^2 / (P_{ii}(f) P_{jj}(f))

    which removes the linear contribution of all other channels. The diagonal is
    1 by construction.

    Parameters
    ----------
    cpsd : (n_freqs, C, C) complex
        Cross-power spectral density (Hermitian per frequency).
    eps : float
        Tikhonov term added to the diagonal before inversion, for stability.

    Returns
    -------
    partial_coherence : (n_freqs, C, C) real in [0, 1].
    """
    C = cpsd.shape[-1]
    eye = jnp.eye(C, dtype=cpsd.dtype)[None]
    prec = jnp.linalg.inv(cpsd + eps * eye)  # (n_freqs, C, C)
    diag = jnp.real(jnp.diagonal(prec, axis1=-2, axis2=-1))  # (n_freqs, C)
    denom = diag[:, :, None] * diag[:, None, :]
    denom = jnp.maximum(denom, 1e-20)
    pcoh = jnp.abs(prec) ** 2 / denom
    return jnp.clip(pcoh, 0.0, 1.0)


def imaginary_coherency_from_cpsd(cpsd: jnp.ndarray) -> jnp.ndarray:
    """Imaginary coherency from a cross-power spectral density (Nolte 2004).

        Coherency_{ij}(f) = CPSD_{ij}(f) / sqrt(PSD_i(f) PSD_j(f))
        ImCoh_{ij}(f)     = Im( Coherency_{ij}(f) )

    Real (zero-lag) cross-spectra give zero imaginary coherency, so the measure
    is insensitive to instantaneous volume-conduction mixing.

    Parameters
    ----------
    cpsd : (n_freqs, C, C) complex.

    Returns
    -------
    imaginary_coherency : (n_freqs, C, C) real in [-1, 1].
    """
    psd = jnp.real(jnp.diagonal(cpsd, axis1=-2, axis2=-1))  # (n_freqs, C)
    denom = jnp.sqrt(jnp.maximum(psd[:, :, None] * psd[:, None, :], 1e-20))
    coherency = cpsd / denom
    return jnp.imag(coherency)
