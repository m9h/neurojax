"""Multitaper spectral estimation in pure JAX.

Implements DPSS (Slepian) tapers and multitaper PSD/CPSD/coherence.
More robust than periodogram-based approaches for MEG/EEG data due
to reduced spectral leakage.

References
----------
Thomson DJ (1982). Spectrum estimation and harmonic analysis.
Proceedings of the IEEE 70(9):1055-1096.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np
import scipy.signal


@functools.lru_cache(maxsize=64)
def _dpss_np(n_samples: int, bandwidth: float, n_tapers: int) -> np.ndarray:
    tapers = scipy.signal.windows.dpss(n_samples, bandwidth, Kmax=n_tapers, norm=2)  # unit energy
    tapers = np.array(tapers, dtype=np.float64)
    for k in range(n_tapers):  # sign convention: positive at the centre
        if tapers[k, n_samples // 2] < 0:
            tapers[k] *= -1
    return tapers


def dpss_tapers(
    n_samples: int,
    bandwidth: float = 4.0,
    n_tapers: int | None = None,
) -> jnp.ndarray:
    """Compute DPSS (Slepian) tapers.

    ``scipy.signal.windows.dpss`` (tridiagonal eigenproblem, O(N) per taper)
    with the result cached per ``(n_samples, bandwidth, n_tapers)``: the
    former dense N×N eigendecomposition cost ~10 s per 2500-sample epoch.

    Parameters
    ----------
    n_samples : int — length of each taper.
    bandwidth : float — time-half-bandwidth product (NW). Default: 4.
    n_tapers : int — number of tapers. Default: 2*bandwidth - 1.

    Returns
    -------
    tapers : (n_tapers, n_samples) — orthonormal (unit-energy) DPSS tapers.
    """
    if n_tapers is None:
        n_tapers = int(2 * bandwidth - 1)
    return jnp.array(_dpss_np(int(n_samples), float(bandwidth), int(n_tapers)))


def multitaper_psd(
    data: jnp.ndarray,
    fs: float = 1.0,
    bandwidth: float = 4.0,
    n_tapers: int | None = None,
    nfft: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Multitaper power spectral density estimate.

    Parameters
    ----------
    data : (T, C) time series.
    fs : float — sampling frequency.
    bandwidth : float — time-half-bandwidth product.
    n_tapers : int — number of DPSS tapers.
    nfft : int — FFT length (zero-pads if > T).

    Returns
    -------
    psd : (n_freqs, C) — power spectral density.
    freqs : (n_freqs,) — frequency axis in Hz.
    """
    T, C = data.shape
    if nfft is None:
        nfft = T

    tapers = dpss_tapers(T, bandwidth, n_tapers)
    # Apply tapers and FFT
    # tapered_data: (K, T, C) = tapers[:, :, None] * data[None, :, :]
    tapered = tapers[:, :, None] * data[None, :, :]  # (K, T, C)

    # FFT along time axis, one-sided
    Xf = jnp.fft.rfft(tapered, n=nfft, axis=1)  # (K, n_freqs, C)

    # PSD = mean over tapers of |Xf|^2 / fs: the tapers have unit energy, so
    # this is a density (units^2 / Hz; a white process of variance v gives 2 v / fs)
    psd = jnp.mean(jnp.abs(Xf) ** 2, axis=0) / fs

    # Double the non-DC, non-Nyquist bins for one-sided
    psd = psd.at[1:-1].multiply(2.0)

    freqs = jnp.fft.rfftfreq(nfft, d=1.0 / fs)

    return psd, freqs


def multitaper_cpsd(
    data: jnp.ndarray,
    fs: float = 1.0,
    bandwidth: float = 4.0,
    n_tapers: int | None = None,
    nfft: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Multitaper cross-power spectral density.

    Parameters
    ----------
    data : (T, C) time series.
    fs, bandwidth, n_tapers, nfft : as in multitaper_psd.

    Returns
    -------
    cpsd : (n_freqs, C, C) complex — cross-spectral density matrix.
    freqs : (n_freqs,) — frequency axis.
    """
    T, C = data.shape
    if nfft is None:
        nfft = T

    tapers = dpss_tapers(T, bandwidth, n_tapers)

    tapered = tapers[:, :, None] * data[None, :, :]
    Xf = jnp.fft.rfft(tapered, n=nfft, axis=1)  # (K, n_freqs, C)

    # CPSD: mean over tapers of Xf * conj(Xf)^T
    # cpsd[f, i, j] = mean_k( Xf[k, f, i] * conj(Xf[k, f, j]) ) / fs  (unit-energy tapers: a density)
    cpsd = jnp.mean(
        Xf[:, :, :, None] * jnp.conj(Xf[:, :, None, :]),
        axis=0,
    ) / fs

    cpsd = cpsd.at[1:-1].multiply(2.0)

    freqs = jnp.fft.rfftfreq(nfft, d=1.0 / fs)

    return cpsd, freqs


def multitaper_coherence(
    data: jnp.ndarray,
    fs: float = 1.0,
    bandwidth: float = 4.0,
    n_tapers: int | None = None,
    nfft: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Multitaper magnitude-squared coherence.

    Parameters
    ----------
    data : (T, C) time series.
    fs, bandwidth, n_tapers, nfft : as in multitaper_psd.

    Returns
    -------
    coherence : (n_freqs, C, C) — in [0, 1].
    freqs : (n_freqs,)
    """
    cpsd, freqs = multitaper_cpsd(data, fs, bandwidth, n_tapers, nfft)

    psd = jnp.real(jnp.diagonal(cpsd, axis1=-2, axis2=-1))  # (n_freqs, C)
    psd_outer = psd[:, :, None] * psd[:, None, :]
    psd_outer = jnp.maximum(psd_outer, 1e-20)

    coh = jnp.abs(cpsd) ** 2 / psd_outer
    coh = jnp.clip(coh, 0.0, 1.0)

    return coh, freqs
