# © NeuroJAX developers
#
# License: BSD (3-clause)
"""Standard spectral baselines for the consciousness benchmark.

What the widely used tools report on the same epochs: specparam/FOOOF
(Donoghue et al. 2020) aperiodic offset and exponent and the alpha peak, plus
Welch relative band powers, log total power, spectral edge frequency and the
raw alpha peak frequency. These sit in :data:`SPECTRAL_MEASURES` with the
same ``Measure`` interface as :data:`neurojax.analysis.nltsa.MEASURES` so the
harness scores them side by side; :func:`neurojax.bench.consciousness_bench.
spectral_explained` then asks how much of each NLTSA measure is linear in
these parameters.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import welch

from neurojax.analysis.nltsa import BANDS, Measure

__all__ = ["SPECTRAL_MEASURES", "specparam_features", "band_powers", "spectral_edge", "alpha_peak_frequency"]

SPECPARAM_RANGE = (1.0, 40.0)
ALPHA = (7.0, 13.0)


def _psd(x, fs, nperseg_s=2.0):
    x = np.asarray(x, float)
    return welch(x, fs=fs, nperseg=min(len(x), int(nperseg_s * fs)))


@lru_cache(maxsize=256)
def _specparam_cached(x_bytes: bytes, n: int, fs: float, f_lo: float, f_hi: float, max_n_peaks: int) -> dict:
    from specparam import SpectralModel

    x = np.frombuffer(x_bytes, dtype=np.float64, count=n)
    f, p = _psd(x, fs)
    sm = SpectralModel(peak_width_limits=[1, 12], max_n_peaks=max_n_peaks, min_peak_height=0.05,
                       aperiodic_mode="fixed", verbose=False)
    sm.fit(f, p, (f_lo, f_hi))
    ap = sm.results.params.aperiodic.params
    pk = np.atleast_2d(sm.results.params.periodic.params) if sm.results.params.periodic.params.size else np.zeros((0, 3))
    alpha = pk[(pk[:, 0] >= ALPHA[0]) & (pk[:, 0] <= ALPHA[1])] if len(pk) else pk
    best = alpha[np.argmax(alpha[:, 1])] if len(alpha) else None
    return dict(offset=float(ap[0]), exponent=float(ap[1]),
                r_squared=float(sm.results.metrics.get_metrics("gof_rsquared")), n_peaks=int(len(pk)),
                alpha_cf=float(best[0]) if best is not None else np.nan,
                alpha_power=float(best[1]) if best is not None else 0.0,
                alpha_bw=float(best[2]) if best is not None else np.nan)


def specparam_features(x, fs: float, f_range=SPECPARAM_RANGE, max_n_peaks: int = 6) -> dict:
    """specparam 2.0 fixed-mode fit of the Welch PSD (2 s segments) over
    ``f_range``: aperiodic ``offset``/``exponent``, ``r_squared``, ``n_peaks``,
    and the strongest alpha-range peak (``alpha_cf``, ``alpha_power`` above
    the aperiodic fit in log10 units, 0 if no peak, ``alpha_bw``)."""
    x = np.ascontiguousarray(np.asarray(x, np.float64))
    return _specparam_cached(x.tobytes(), len(x), float(fs), *f_range, max_n_peaks)


def band_powers(x, fs: float, bands=BANDS, total=(1.0, 45.0)) -> dict:
    """Relative Welch band powers (fraction of 1-45 Hz power) and log10 total power."""
    f, p = _psd(x, fs)
    tot = p[(f >= total[0]) & (f <= total[1])].sum()
    out = {f"relpow_{k}": float(p[(f >= lo) & (f < hi)].sum() / tot) for k, (lo, hi) in bands.items()}
    out["log_total_power"] = float(np.log10(tot))
    return out


def spectral_edge(x, fs: float, q: float = 0.95, f_max: float = 45.0) -> float:
    """Spectral edge frequency: frequency below which ``q`` of 0-``f_max`` Hz power lies."""
    f, p = _psd(x, fs)
    sel = f <= f_max
    c = np.cumsum(p[sel]) / p[sel].sum()
    return float(f[sel][np.searchsorted(c, q)])


def alpha_peak_frequency(x, fs: float, band=ALPHA) -> float:
    """Frequency of the raw PSD maximum within the alpha band (no aperiodic model)."""
    f, p = _psd(x, fs)
    sel = (f >= band[0]) & (f <= band[1])
    return float(f[sel][np.argmax(p[sel])])


def _sp(key):
    return lambda x, fs: specparam_features(x, fs)[key]


def _bp(key):
    return lambda x, fs: band_powers(x, fs)[key]


SPECTRAL_MEASURES: dict[str, Measure] = {m.name: m for m in (
    Measure("specparam_exponent", _sp("exponent"), "spectral", "Donoghue 2020 specparam, fixed, 1-40 Hz"),
    Measure("specparam_offset", _sp("offset"), "spectral", "Donoghue 2020 specparam, fixed, 1-40 Hz"),
    Measure("specparam_alpha_cf", _sp("alpha_cf"), "spectral", "Donoghue 2020; strongest 7-13 Hz peak"),
    Measure("specparam_alpha_power", _sp("alpha_power"), "spectral", "Donoghue 2020; peak power over aperiodic"),
    Measure("specparam_r2", _sp("r_squared"), "spectral", "Donoghue 2020 goodness of fit"),
    *(Measure(f"relpow_{b}", _bp(f"relpow_{b}"), "spectral", f"Welch relative {b} power {lo}-{hi} Hz")
      for b, (lo, hi) in BANDS.items()),
    Measure("log_total_power", _bp("log_total_power"), "spectral", "Welch 1-45 Hz"),
    Measure("spectral_edge_95", spectral_edge, "spectral", "SEF95, Rampil 1998"),
    Measure("alpha_peak_frequency", alpha_peak_frequency, "spectral", "raw PSD argmax 7-13 Hz"),
)}
