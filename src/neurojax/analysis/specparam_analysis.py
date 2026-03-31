"""specparam (FOOOF) analysis for HMM/DyNeMo state power spectra.

Extracts aperiodic (1/f slope) and periodic (oscillatory peaks) components
from per-state, per-parcel PSDs. These features connect MEG dynamics to:
  - MRS GABA/glutamate (aperiodic exponent ∝ E/I balance)
  - AxCaliber conduction velocity (IAF ∝ thalamocortical speed)
  - Cortical thickness (steeper 1/f → more layers)
  - QMT myelin (more myelin → different peak structure)

Usage:
    state_psd = np.load("hmm/state_psd.npy")  # (8, n_freqs, 68)
    freqs = np.load("hmm/frequencies.npy")
    features = extract_fooof_features(state_psd, freqs)
"""
from __future__ import annotations

import logging

import numpy as np
from specparam import SpectralModel

logger = logging.getLogger(__name__)


def fit_single_psd(
    psd: np.ndarray,
    freqs: np.ndarray,
    freq_range: tuple[float, float] = (1, 45),
    max_n_peaks: int = 6,
) -> dict:
    """Fit specparam to a single PSD.

    Parameters
    ----------
    psd : (n_freqs,)
    freqs : (n_freqs,)
    freq_range : frequency range for fitting
    max_n_peaks : max oscillatory peaks to fit

    Returns
    -------
    dict with aperiodic_offset, aperiodic_exponent, peak_freq, peak_power,
    peak_bw, n_peaks, r_squared
    """
    sm = SpectralModel(
        peak_width_limits=[1, 12],
        max_n_peaks=max_n_peaks,
        min_peak_height=0.05,
        aperiodic_mode="fixed",
        verbose=False,
    )
    sm.fit(freqs, psd, freq_range)

    # specparam 2.0 API
    ap_params = sm.results.params.aperiodic.params  # [offset, exponent]
    peak_params = sm.results.params.periodic.params  # (n_peaks, 3) or empty

    result = {
        "aperiodic_offset": float(ap_params[0]) if len(ap_params) >= 2 else 0.0,
        "aperiodic_exponent": float(ap_params[1]) if len(ap_params) >= 2 else 0.0,
        "r_squared": float(sm.results.metrics.get_metrics("gof_rsquared")),
        "n_peaks": int(peak_params.shape[0]) if peak_params.ndim == 2 else 0,
    }

    # Dominant peak (highest power)
    if peak_params.ndim == 2 and peak_params.shape[0] > 0:
        best = np.argmax(peak_params[:, 1])
        result["peak_freq"] = float(peak_params[best, 0])
        result["peak_power"] = float(peak_params[best, 1])
        result["peak_bw"] = float(peak_params[best, 2])
    else:
        result["peak_freq"] = 0.0
        result["peak_power"] = 0.0
        result["peak_bw"] = 0.0

    return result


def fit_state_spectra(
    state_psd: np.ndarray,
    freqs: np.ndarray,
    freq_range: tuple[float, float] = (1, 45),
) -> dict[str, np.ndarray]:
    """Fit specparam to all states × parcels.

    Parameters
    ----------
    state_psd : (n_states, n_freqs, n_parcels)
    freqs : (n_freqs,)

    Returns
    -------
    dict with arrays of shape (n_states, n_parcels):
        aperiodic_exponent, aperiodic_offset, peak_freq, peak_power, r_squared
    """
    n_states, n_freqs, n_parcels = state_psd.shape

    out = {
        "aperiodic_exponent": np.zeros((n_states, n_parcels)),
        "aperiodic_offset": np.zeros((n_states, n_parcels)),
        "peak_freq": np.zeros((n_states, n_parcels)),
        "peak_power": np.zeros((n_states, n_parcels)),
        "r_squared": np.zeros((n_states, n_parcels)),
    }

    for s in range(n_states):
        for p in range(n_parcels):
            psd = state_psd[s, :, p]
            if np.all(psd <= 0) or not np.all(np.isfinite(psd)):
                continue
            try:
                result = fit_single_psd(psd, freqs, freq_range)
                out["aperiodic_exponent"][s, p] = result["aperiodic_exponent"]
                out["aperiodic_offset"][s, p] = result["aperiodic_offset"]
                out["peak_freq"][s, p] = result["peak_freq"]
                out["peak_power"][s, p] = result["peak_power"]
                out["r_squared"][s, p] = result["r_squared"]
            except Exception:
                pass

    logger.info("Fit %d states × %d parcels. Mean R²=%.3f, mean exponent=%.2f",
                n_states, n_parcels,
                np.mean(out["r_squared"][out["r_squared"] > 0]),
                np.mean(out["aperiodic_exponent"][out["aperiodic_exponent"] > 0]))
    return out


def extract_fooof_features(
    state_psd: np.ndarray,
    freqs: np.ndarray,
    freq_range: tuple[float, float] = (1, 45),
) -> np.ndarray:
    """Extract a per-state feature vector from specparam fits.

    For each state, computes parcel-averaged features:
      [mean_exponent, std_exponent, mean_IAF, std_IAF,
       mean_peak_power, mean_r_squared]

    Parameters
    ----------
    state_psd : (n_states, n_freqs, n_parcels)
    freqs : (n_freqs,)

    Returns
    -------
    features : (n_states, 6) feature matrix
    """
    fits = fit_state_spectra(state_psd, freqs, freq_range)
    n_states = state_psd.shape[0]
    features = np.zeros((n_states, 6))

    for s in range(n_states):
        exp = fits["aperiodic_exponent"][s]
        pf = fits["peak_freq"][s]
        pp = fits["peak_power"][s]
        r2 = fits["r_squared"][s]

        valid_exp = exp[exp > 0]
        valid_pf = pf[pf > 0]

        features[s, 0] = np.mean(valid_exp) if len(valid_exp) > 0 else 0
        features[s, 1] = np.std(valid_exp) if len(valid_exp) > 0 else 0
        features[s, 2] = np.mean(valid_pf) if len(valid_pf) > 0 else 0
        features[s, 3] = np.std(valid_pf) if len(valid_pf) > 0 else 0
        features[s, 4] = np.mean(pp)
        features[s, 5] = np.mean(r2)

    return features
