"""Regression-based spectral analysis for DyNeMo.

An alternative to the covariance-inversion approach: regress DyNeMo's
soft mixing coefficients (alpha) onto time-frequency spectrograms to
obtain per-mode power spectra.

This is the approach used in osl-dynamics ``analysis.regression`` and
described in the DyNeMo paper (Gohil et al., 2024).

Pipeline::

    raw data (T, C)  → spectrogram (n_windows, n_freqs, C)
    alpha (T, K)     → downsample to (n_windows, K)
    regression: spectrogram ~ alpha @ beta
    → beta gives per-mode power: (K, n_freqs, C)
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp


def compute_spectrogram(
    data: jnp.ndarray,
    fs: float = 1.0,
    window_length: int = 256,
    step_size: int | None = None,
    nfft: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute short-time Fourier transform power spectrogram.

    Parameters
    ----------
    data : (T, C) time series.
    fs : float — sampling frequency.
    window_length : int — STFT window length in samples.
    step_size : int — hop size. Default: window_length // 2.
    nfft : int — FFT size. Default: window_length.

    Returns
    -------
    spectrogram : (n_windows, n_freqs, C) — power spectrogram.
    frequencies : (n_freqs,)
    times : (n_windows,) — centre time of each window in seconds.
    """
    T, C = data.shape
    if step_size is None:
        step_size = window_length // 2
    if nfft is None:
        nfft = window_length

    n_windows = (T - window_length) // step_size + 1
    starts = jnp.arange(n_windows) * step_size

    # Hann window
    window = 0.5 * (1 - jnp.cos(2 * jnp.pi * jnp.arange(window_length) / window_length))

    def _window_psd(start):
        segment = jax.lax.dynamic_slice(
            data, (start, 0), (window_length, C)
        )
        windowed = segment * window[:, None]
        Xf = jnp.fft.rfft(windowed, n=nfft, axis=0)
        power = jnp.abs(Xf) ** 2 / (fs * nfft)
        return power

    spectrogram = jax.vmap(_window_psd)(starts)  # (n_windows, n_freqs, C)

    frequencies = jnp.fft.rfftfreq(nfft, d=1.0 / fs)
    times = (starts + window_length / 2) / fs

    return spectrogram, frequencies, times


def compute_regression_spectra(
    data: jnp.ndarray,
    alpha: jnp.ndarray,
    fs: float = 1.0,
    window_length: int = 256,
    step_size: int | None = None,
) -> dict:
    """Compute mode-specific power spectra via regression.

    Regresses the DyNeMo mixing coefficients onto the spectrogram
    using ordinary least squares.

    Parameters
    ----------
    data : (T, C) — raw time series.
    alpha : (T, K) — DyNeMo mixing coefficients.
    fs : float — sampling frequency.
    window_length : int — STFT window length.
    step_size : int — STFT hop size.

    Returns
    -------
    dict with:
        ``"psd"`` : (K, n_freqs, C) — per-mode power spectra.
        ``"frequencies"`` : (n_freqs,)
        ``"spectrogram"`` : (n_windows, n_freqs, C)
        ``"alpha_downsampled"`` : (n_windows, K)
    """
    T, C = data.shape
    K = alpha.shape[1]

    if step_size is None:
        step_size = window_length // 2

    # Compute spectrogram
    spec, freqs, times = compute_spectrogram(
        data, fs=fs, window_length=window_length, step_size=step_size
    )
    n_windows = spec.shape[0]
    n_freqs = spec.shape[1]

    # Downsample alpha to match spectrogram windows
    # Take alpha at window centre indices
    centre_indices = (jnp.arange(n_windows) * step_size + window_length // 2).astype(int)
    centre_indices = jnp.clip(centre_indices, 0, T - 1)
    alpha_ds = alpha[centre_indices]  # (n_windows, K)

    # Regression: spec ~ alpha_ds @ beta
    # For each (freq, channel) pair, solve: spec[:, f, c] = alpha_ds @ beta[:, f, c]
    # Reshape spec to (n_windows, n_freqs * C)
    spec_flat = spec.reshape(n_windows, -1)  # (n_windows, n_freqs * C)

    # OLS: beta = (A^T A)^{-1} A^T Y
    AtA = alpha_ds.T @ alpha_ds  # (K, K)
    AtA_reg = AtA + 1e-6 * jnp.eye(K)  # regularize
    AtY = alpha_ds.T @ spec_flat  # (K, n_freqs * C)
    beta = jnp.linalg.solve(AtA_reg, AtY)  # (K, n_freqs * C)

    # Reshape beta to (K, n_freqs, C)
    psd = beta.reshape(K, n_freqs, C)

    return {
        "psd": psd,
        "frequencies": freqs,
        "spectrogram": spec,
        "alpha_downsampled": alpha_ds,
    }
