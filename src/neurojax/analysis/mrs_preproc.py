"""MRS preprocessing functions: apodization, eddy current correction,
frequency referencing.

Implements common preprocessing steps for MRS FID data:
- Exponential apodization (matched-filter line broadening)
- Gaussian apodization (resolution enhancement)
- Eddy current correction (Klose method)
- Frequency referencing to known metabolite peaks

References:
    de Graaf (2019) In Vivo NMR Spectroscopy, 3rd ed. Wiley.

    Klose (1990) In vivo proton spectroscopy in the presence of eddy
    currents. MRM 14:26-30.

    Near et al. (2021) Preprocessing, analysis and quantification in
    single-voxel magnetic resonance spectroscopy: experts' consensus
    recommendations. NMR Biomed 34:e4257.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Apodization
# ---------------------------------------------------------------------------

def exponential_apodization(
    fid: np.ndarray,
    dwell_time: float,
    broadening_hz: float,
) -> np.ndarray:
    """Apply exponential (Lorentzian) apodization to an FID.

    Multiplies the FID by exp(-pi * broadening_hz * t), which adds
    *broadening_hz* Hz to the Lorentzian linewidth of every peak.

    Parameters
    ----------
    fid : ndarray
        Time-domain FID. The first axis is the spectral (time) dimension.
        Additional dimensions (coils, dynamics, ...) are broadcast.
    dwell_time : float
        Dwell time in seconds.
    broadening_hz : float
        Additional line broadening in Hz.

    Returns
    -------
    ndarray
        Apodized FID, same shape as input.
    """
    if broadening_hz == 0.0:
        return fid.copy()

    n = fid.shape[0]
    t = np.arange(n) * dwell_time

    # Reshape t for broadcasting with multi-dim FIDs
    shape = [n] + [1] * (fid.ndim - 1)
    t = t.reshape(shape)

    window = np.exp(-np.pi * broadening_hz * t)
    return fid * window


def gaussian_apodization(
    fid: np.ndarray,
    dwell_time: float,
    broadening_hz: float,
) -> np.ndarray:
    """Apply Gaussian apodization to an FID.

    Multiplies the FID by exp(-pi * (broadening_hz * t)^2 / (4 * ln2)),
    which produces a Gaussian lineshape contribution with FWHM =
    broadening_hz in the frequency domain.

    Parameters
    ----------
    fid : ndarray
        Time-domain FID. First axis is spectral (time) dimension.
    dwell_time : float
        Dwell time in seconds.
    broadening_hz : float
        Gaussian broadening FWHM in Hz.

    Returns
    -------
    ndarray
        Apodized FID, same shape as input.
    """
    if broadening_hz == 0.0:
        return fid.copy()

    n = fid.shape[0]
    t = np.arange(n) * dwell_time

    shape = [n] + [1] * (fid.ndim - 1)
    t = t.reshape(shape)

    # Gaussian window: FWHM in frequency domain = broadening_hz
    # sigma_t = sqrt(2 * ln2) / (pi * FWHM)
    # window = exp(-t^2 / (2 * sigma_t^2))
    #        = exp(-pi^2 * FWHM^2 * t^2 / (4 * ln2))
    window = np.exp(-np.pi**2 * broadening_hz**2 * t**2 / (4.0 * np.log(2)))
    return fid * window


# ---------------------------------------------------------------------------
# Eddy current correction
# ---------------------------------------------------------------------------

def eddy_current_correction(
    fid: np.ndarray,
    water_ref: np.ndarray,
) -> np.ndarray:
    """Eddy current correction using a water reference (Klose method).

    Removes time-dependent phase distortions caused by eddy currents
    by subtracting the instantaneous phase of the water reference from
    the metabolite FID, point by point.

    Parameters
    ----------
    fid : ndarray
        Metabolite FID to correct. First axis is spectral (time).
    water_ref : ndarray
        Water reference FID acquired with the same sequence/timings.
        Must have the same number of spectral points along axis 0.

    Returns
    -------
    ndarray
        Phase-corrected FID, same shape as input.

    References
    ----------
    Klose (1990) MRM 14:26-30.
    """
    # Extract time-dependent phase from water reference
    water_phase = np.angle(water_ref)

    # For multi-dim water_ref, broadcast to match fid shape
    # If water_ref has fewer dims, the phase is applied along axis 0
    correction = np.exp(-1j * water_phase)

    # Handle shape broadcasting: if fid and water_ref have different
    # trailing dimensions, reshape correction
    if fid.ndim > water_ref.ndim:
        extra = fid.ndim - water_ref.ndim
        correction = correction.reshape(correction.shape + (1,) * extra)

    return fid * correction


# ---------------------------------------------------------------------------
# Frequency referencing
# ---------------------------------------------------------------------------

def frequency_reference(
    fid: np.ndarray,
    dwell_time: float,
    centre_freq: float,
    target_ppm: float,
    target_peak_ppm: float,
    search_window_ppm: float = 0.3,
) -> np.ndarray:
    """Shift FID in frequency so a known peak lands at its canonical ppm.

    Finds the tallest peak near *target_peak_ppm* and shifts the entire
    spectrum so that peak sits exactly at *target_ppm*.

    Parameters
    ----------
    fid : ndarray
        Time-domain FID (1-D complex array).
    dwell_time : float
        Dwell time in seconds.
    centre_freq : float
        Spectrometer centre frequency in Hz.
    target_ppm : float
        Desired ppm location for the peak after correction.
    target_peak_ppm : float
        Approximate current ppm of the peak to lock onto.
    search_window_ppm : float
        Half-width of the ppm search window around *target_peak_ppm*.

    Returns
    -------
    ndarray
        Frequency-shifted FID.
    """
    n = fid.shape[0]

    # Compute spectrum and ppm axis
    spec = np.fft.fftshift(np.fft.fft(fid))
    freq = np.fft.fftshift(np.fft.fftfreq(n, dwell_time))
    ppm = freq / (centre_freq / 1e6) + 4.65

    # Search for peak near target
    mask = (
        (ppm > target_peak_ppm - search_window_ppm)
        & (ppm < target_peak_ppm + search_window_ppm)
    )
    peak_idx_local = np.argmax(np.abs(spec[mask]))
    peak_ppm_found = ppm[mask][peak_idx_local]

    # Compute required frequency shift
    delta_ppm = target_ppm - peak_ppm_found
    delta_hz = delta_ppm * (centre_freq / 1e6)  # ppm -> Hz

    # Apply shift in time domain
    t = np.arange(n) * dwell_time
    return fid * np.exp(2j * np.pi * delta_hz * t)
