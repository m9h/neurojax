"""Simulated spectra with realistic estimation noise, and Welch estimates from time series."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def simulate_log_psd(model, theta, freqs, n_segments, key):
    """``ln S_hat`` where ``S_hat = S(theta) * Gamma(n, 1/n)``: a Welch estimate of *n_segments* periodograms."""
    log_s = model.log_psd(freqs, theta)
    g = jax.random.gamma(key, float(n_segments), shape=log_s.shape) / float(n_segments)
    return log_s + jnp.log(g)


def psd_from_timeseries(x, fs, segment_s=2.0, f_lo=1.0, f_hi=45.0):
    """Welch PSD (Hann, 50% overlap) restricted to ``[f_lo, f_hi]``.

    Returns ``(freqs, log_psd, n_segments)``; ``n_segments`` is the effective
    number of independent periodograms for the Whittle likelihood, taken as
    the number of non-overlapping segments (a conservative count).
    """
    from scipy.signal import welch

    x = np.asarray(x, dtype=float)
    nper = int(round(segment_s * fs))
    f, p = welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, window="hann")
    m = (f >= f_lo) & (f <= f_hi)
    n_segments = max(1, x.shape[-1] // nper)
    return f[m], np.log(p[..., m]), n_segments
