"""MRS (Magnetic Resonance Spectroscopy) preprocessing for in vivo data.

Spectroscopy preprocessing routines targeting WAND sLASER (7 T) and
MEGA-PRESS (3 T) acquisitions.  All functions are pure (no file I/O) and
operate on NumPy arrays of complex FID / spectrum data.

Implemented pipeline steps:
    - Zero- and first-order automatic phase correction
    - Frequency alignment of individual averages
    - HLSVD-based water suppression
    - Coil combination (sensitivity-weighted and Tucker)
    - Utility helpers (ppm axis, FID -> spectrum)

References:
    Clarke WT, Stagg CJ, Jbabdi S.  FSL-MRS: An end-to-end spectroscopy
    analysis package.  *Magnetic Resonance in Medicine* 85(6):2950-2964,
    2021.  doi:10.1002/mrm.28630

    de Graaf RA.  *In Vivo NMR Spectroscopy: Principles and Techniques*,
    3rd edition. Wiley, 2019.  Chapters 3 (signal processing) and
    10 (spectral editing).

    Barkhuijsen H, de Beer R, van Ormondt D.  Improved algorithm for
    noniterative time-domain model fitting to exponentially damped
    magnetic resonance signals.  *Journal of Magnetic Resonance*
    73(3):553-557, 1987.  (HLSVD original formulation)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import svd as scipy_svd


# =====================================================================
# Utility helpers
# =====================================================================

def ppm_axis(n_points: int, dwell: float, cf_mhz: float = 297.2) -> np.ndarray:
    """Compute the chemical-shift axis in ppm, referenced to water at 4.7 ppm.

    The axis is constructed so that the centre of the FFT corresponds to
    the transmitter frequency, which is assumed to be on-resonance with
    water (4.7 ppm).  The axis runs from high ppm to low ppm (standard
    MRS convention) after ``fftshift``.

    Parameters
    ----------
    n_points : int
        Number of spectral points (= FID length).
    dwell : float
        Dwell time in seconds (1 / spectral bandwidth).
    cf_mhz : float, optional
        Centre (Larmor) frequency in MHz.  Default 297.2 MHz
        (proton at 7 T).

    Returns
    -------
    ppm : np.ndarray, shape ``(n_points,)``
        Chemical-shift axis in ppm with water at 4.7 ppm.

    References
    ----------
    de Graaf, *In Vivo NMR Spectroscopy*, ch. 3.
    """
    freq_hz = np.fft.fftshift(np.fft.fftfreq(n_points, d=dwell))
    ppm = 4.7 - freq_hz / cf_mhz               # water-referenced
    return ppm


def fid_to_spectrum(fid: np.ndarray, dwell: float = None,
                    lb_hz: float = 3.0) -> np.ndarray:
    """Convert a complex FID to a frequency-domain spectrum.

    Applies optional exponential apodisation (line broadening), then
    FFT + ``fftshift``.

    Parameters
    ----------
    fid : np.ndarray
        Complex free-induction decay; may be 1-D ``(n,)`` or 2-D
        ``(n, ...)``.
    dwell : float or None
        Dwell time in seconds.  Required when *lb_hz* > 0.
    lb_hz : float, optional
        Exponential line broadening in Hz (default 3 Hz).  Set to 0 to
        disable apodisation.

    Returns
    -------
    spectrum : np.ndarray
        Complex spectrum after ``fftshift``, same shape as *fid*.

    References
    ----------
    de Graaf, *In Vivo NMR Spectroscopy*, section 3.4.
    Clarke et al. 2021, FSL-MRS.
    """
    fid = np.asarray(fid, dtype=np.complex128)

    if lb_hz > 0:
        if dwell is None:
            raise ValueError(
                "`dwell` must be provided when lb_hz > 0."
            )
        n = fid.shape[0]
        t = np.arange(n) * dwell
        # Reshape for broadcasting when fid has trailing dimensions
        shape = (n,) + (1,) * (fid.ndim - 1)
        apod = np.exp(-np.pi * lb_hz * t).reshape(shape)
        fid = fid * apod

    spectrum = np.fft.fftshift(np.fft.fft(fid, axis=0), axes=0)
    return spectrum


# =====================================================================
# Automatic phase correction
# =====================================================================

def auto_phase_correct_0th(spectrum: np.ndarray) -> tuple[np.ndarray, float]:
    """Zero-order automatic phase correction.

    The optimal zero-order phase is the angle that maximises the
    integral of the real part of the spectrum.  A brute-force search
    over 0-360 degrees in 1-degree steps is used.

    Parameters
    ----------
    spectrum : np.ndarray
        Complex spectrum (1-D).

    Returns
    -------
    corrected : np.ndarray
        Phase-corrected complex spectrum.
    phase_deg : float
        Applied phase correction in degrees.

    References
    ----------
    de Graaf, *In Vivo NMR Spectroscopy*, section 3.5 (phase correction).
    Clarke et al. 2021, FSL-MRS — ``phase_correct`` routine.
    """
    spectrum = np.asarray(spectrum, dtype=np.complex128)
    angles = np.arange(360)                         # degrees
    radians = np.deg2rad(angles)                    # (360,)

    # Vectorised: apply every candidate phase at once
    # rotated shape: (360, n_spectral)
    rotated = spectrum[np.newaxis, :] * np.exp(1j * radians[:, np.newaxis])
    integrals = rotated.real.sum(axis=1)            # (360,)

    best_idx = int(np.argmax(integrals))
    phase_deg = float(angles[best_idx])
    corrected = spectrum * np.exp(1j * np.deg2rad(phase_deg))
    return corrected, phase_deg


def auto_phase_correct_1st(
    spectrum: np.ndarray,
    ppm: np.ndarray,
    pivot_ppm: float = 2.02,
) -> tuple[np.ndarray, float, float]:
    """Combined zero- and first-order automatic phase correction.

    Zero-order correction is applied first (maximise real integral),
    then a first-order term is searched that minimises baseline
    dispersion (standard deviation of the real spectrum in baseline
    regions).

    The first-order phase ramp is:

    .. math::
        \\phi_1(\\nu) = \\exp\\bigl(-i\\,\\alpha\\,(\\text{ppm} - \\text{pivot})\\bigr)

    where *alpha* is varied over a grid of candidate values.

    Parameters
    ----------
    spectrum : np.ndarray
        Complex spectrum (1-D).
    ppm : np.ndarray
        Chemical-shift axis in ppm (same length as *spectrum*).
    pivot_ppm : float, optional
        Pivot point for the first-order correction in ppm.  Default is
        2.02 (NAA singlet).

    Returns
    -------
    corrected : np.ndarray
        Phase-corrected complex spectrum.
    phase0_deg : float
        Applied zero-order phase in degrees.
    phase1_deg : float
        Applied first-order phase expressed as the slope
        ``alpha`` in degrees (i.e., the total phase accrued per ppm).

    References
    ----------
    de Graaf, *In Vivo NMR Spectroscopy*, section 3.5.
    Clarke et al. 2021, FSL-MRS — ``phase_correct`` routine.
    """
    spectrum = np.asarray(spectrum, dtype=np.complex128)
    ppm = np.asarray(ppm, dtype=np.float64)

    # --- 0th-order correction ---
    spec_0, phase0_deg = auto_phase_correct_0th(spectrum)

    # --- 1st-order correction ---
    # Identify baseline regions (far from metabolite peaks).
    # Typical baseline: ppm < 0.5 or ppm > 8.0
    baseline_mask = (ppm < 0.5) | (ppm > 8.0)
    if baseline_mask.sum() < 5:
        # Fallback: use outer 10 % of the spectrum
        n = len(ppm)
        idx = np.arange(n)
        baseline_mask = (idx < int(0.05 * n)) | (idx > int(0.95 * n))

    dppm = ppm - pivot_ppm  # frequency offset from pivot

    # Search alpha from -180 to 180 degrees/ppm in 0.5-degree steps
    alphas = np.arange(-180, 180.5, 0.5)           # degrees/ppm
    alpha_rad = np.deg2rad(alphas)                  # rad/ppm

    # Vectorised: (n_alpha, n_spectral)
    ramp = np.exp(-1j * alpha_rad[:, np.newaxis] * dppm[np.newaxis, :])
    candidates = spec_0[np.newaxis, :] * ramp       # apply ramp

    # Dispersion metric: std of the real part in baseline regions
    baseline_real = candidates[:, baseline_mask].real
    dispersion = baseline_real.std(axis=1)

    best_idx = int(np.argmin(dispersion))
    phase1_deg = float(alphas[best_idx])

    corrected = spec_0 * np.exp(-1j * np.deg2rad(phase1_deg) * dppm)
    return corrected, phase0_deg, phase1_deg


# =====================================================================
# Frequency alignment
# =====================================================================

def frequency_align(
    fids: np.ndarray,
    dwell: float,
    ref_ppm: float = 2.02,
    cf_mhz: float = 297.2,
) -> np.ndarray:
    """Frequency-align multiple FID averages to a reference peak.

    Each average is individually FFT'd and the peak closest to
    *ref_ppm* is identified.  The frequency offset relative to the
    first average is corrected by multiplying a phase ramp in the
    time domain:

    .. math::
        \\text{FID}_{\\text{aligned}}(t) =
            \\text{FID}(t)\\,\\exp\\bigl(-2\\pi i\\,\\Delta f\\, t\\bigr)

    Parameters
    ----------
    fids : np.ndarray, shape ``(n_spectral, n_averages)``
        Complex FID matrix.
    dwell : float
        Dwell time in seconds.
    ref_ppm : float, optional
        Reference peak position in ppm (default 2.02, NAA singlet).
    cf_mhz : float, optional
        Centre frequency in MHz (default 297.2, proton 7 T).

    Returns
    -------
    aligned : np.ndarray, shape ``(n_spectral, n_averages)``
        Frequency-aligned FIDs.

    References
    ----------
    Clarke et al. 2021, FSL-MRS — frequency alignment module.
    de Graaf, *In Vivo NMR Spectroscopy*, section 3.7.
    """
    fids = np.asarray(fids, dtype=np.complex128)
    n_spectral, n_avg = fids.shape

    ppm = ppm_axis(n_spectral, dwell, cf_mhz)
    freq_hz = np.fft.fftshift(np.fft.fftfreq(n_spectral, d=dwell))

    # Narrow search window: +/- 0.5 ppm around ref_ppm
    search_mask = (ppm > ref_ppm - 0.5) & (ppm < ref_ppm + 0.5)

    spectra = np.fft.fftshift(np.fft.fft(fids, axis=0), axes=0)

    # Reference peak position (from first average)
    ref_spec = np.abs(spectra[:, 0])
    ref_spec_masked = np.where(search_mask, ref_spec, 0.0)
    ref_peak_idx = int(np.argmax(ref_spec_masked))
    ref_freq = freq_hz[ref_peak_idx]

    t = np.arange(n_spectral) * dwell  # time vector

    aligned = np.empty_like(fids)
    for k in range(n_avg):
        spec_k = np.abs(spectra[:, k])
        spec_k_masked = np.where(search_mask, spec_k, 0.0)
        peak_idx = int(np.argmax(spec_k_masked))
        delta_f = freq_hz[peak_idx] - ref_freq      # Hz shift

        # Correct in time domain
        aligned[:, k] = fids[:, k] * np.exp(-2j * np.pi * delta_f * t)

    return aligned


# =====================================================================
# HLSVD water removal
# =====================================================================

def hlsvd_water_removal(
    fid: np.ndarray,
    dwell: float,
    n_components: int = 25,
    water_range_ppm: tuple = (4.5, 5.0),
    cf_mhz: float = 297.2,
) -> np.ndarray:
    """Remove the water resonance from a FID using a simplified HLSVD method.

    A Hankel matrix is formed from the FID and decomposed with a
    truncated SVD.  Components whose frequencies fall inside
    *water_range_ppm* are identified and subtracted from the original
    FID.

    Parameters
    ----------
    fid : np.ndarray
        Complex FID (1-D).
    dwell : float
        Dwell time in seconds.
    n_components : int, optional
        Number of singular components to retain (default 25).
    water_range_ppm : tuple of float, optional
        PPM range for water (default (4.5, 5.0)).
    cf_mhz : float, optional
        Centre frequency in MHz (default 297.2, proton 7 T).

    Returns
    -------
    fid_clean : np.ndarray
        Water-removed FID (same length as input).

    Notes
    -----
    This is a simplified, pedagogical implementation.  For clinical
    pipelines consider FSL-MRS or HLSVDPRO which use optimised
    Lanczos-based solvers.

    References
    ----------
    Barkhuijsen H, de Beer R, van Ormondt D.  *J Magn Reson*
    73(3):553-557, 1987.
    Clarke et al. 2021, FSL-MRS.
    de Graaf, *In Vivo NMR Spectroscopy*, section 3.8.
    """
    fid = np.asarray(fid, dtype=np.complex128).ravel()
    n = len(fid)

    # --- build Hankel matrix ---
    L = n // 2                                  # matrix pencil parameter
    M = n - L
    H = np.empty((M, L), dtype=np.complex128)
    for row in range(M):
        H[row, :] = fid[row: row + L]

    # --- truncated SVD ---
    K = min(n_components, min(M, L) - 1)
    U, s, Vh = scipy_svd(H, full_matrices=False)
    U = U[:, :K]
    s = s[:K]
    Vh = Vh[:K, :]

    # --- extract frequencies from state-space approach ---
    # Form the "signal pole" matrix via the shift-invariance property
    # of the truncated Hankel matrix.
    U_top = U[:-1, :]          # rows 0 .. M-2
    U_bot = U[1:, :]           # rows 1 .. M-1
    # Least-squares: Z = pinv(U_top) @ U_bot
    Z = np.linalg.lstsq(U_top, U_bot, rcond=None)[0]
    poles = np.linalg.eigvals(Z)                    # complex poles z_k

    # Convert poles to frequencies in Hz
    freq_hz = np.angle(poles) / (2.0 * np.pi * dwell)
    # Convert to ppm
    freq_ppm = 4.7 - freq_hz / cf_mhz

    # --- identify water components ---
    lo, hi = min(water_range_ppm), max(water_range_ppm)
    water_mask = (freq_ppm >= lo) & (freq_ppm <= hi)

    if not np.any(water_mask):
        # No water components found; return the original FID unchanged
        return fid.copy()

    # --- reconstruct water signal ---
    # Fit amplitudes for water components: fid = sum_k a_k * z_k^n
    water_poles = poles[water_mask]
    n_water = len(water_poles)
    t_idx = np.arange(n)                            # 0 .. N-1

    # Basis matrix: (N, n_water)
    basis = np.empty((n, n_water), dtype=np.complex128)
    for k, zk in enumerate(water_poles):
        basis[:, k] = zk ** t_idx

    # Least-squares amplitude fit
    amplitudes = np.linalg.lstsq(basis, fid, rcond=None)[0]

    water_signal = basis @ amplitudes
    fid_clean = fid - water_signal
    return fid_clean


# =====================================================================
# Coil combination
# =====================================================================

def sensitivity_weighted_combine(data: np.ndarray) -> np.ndarray:
    """Sensitivity-weighted coil combination for MRS data.

    Each coil is weighted by its first FID point amplitude, and
    phase-aligned to the first coil.  This is the standard combination
    method for single-voxel MRS recommended in FSL-MRS.

    Parameters
    ----------
    data : np.ndarray, shape ``(n_spectral, n_channels, n_averages)``
        Complex multi-coil FID data.

    Returns
    -------
    combined : np.ndarray, shape ``(n_spectral, n_averages)``
        Combined complex FID.

    References
    ----------
    Clarke et al. 2021, FSL-MRS — ``coil_combine`` module.
    de Graaf, *In Vivo NMR Spectroscopy*, section 10.3.
    """
    data = np.asarray(data, dtype=np.complex128)
    # data shape: (n_spectral, n_channels, n_averages)
    n_spectral, n_channels, n_averages = data.shape

    # First-point amplitudes and phases, averaged across averages
    first_pts = data[0, :, :]                       # (n_channels, n_averages)
    mean_first = first_pts.mean(axis=1)             # (n_channels,)

    weights = np.abs(mean_first)                    # amplitude weights
    phases = np.angle(mean_first)                   # phase offsets

    # Phase of first coil is the reference
    phase_ref = phases[0]
    phase_corr = np.exp(-1j * (phases - phase_ref)) # align to coil 0

    # Normalise weights to sum to 1
    weights = weights / weights.sum()

    # Apply weights and phase correction
    # Broadcast: weights (n_channels,) -> (1, n_channels, 1)
    w = (weights * phase_corr).reshape(1, n_channels, 1)
    combined = (data * w).sum(axis=1)               # (n_spectral, n_averages)
    return combined


def tucker_coil_combine(data: np.ndarray, rank_coil: int = 1) -> np.ndarray:
    """Coil combination via Tucker decomposition on the channel mode.

    The coil dimension is "unfolded" (matricised) and a truncated SVD
    identifies the dominant mode.  This is equivalent to the rank-1
    Tucker decomposition along the coil axis and generalises the
    sensitivity-weighted approach to data-driven weighting.

    Parameters
    ----------
    data : np.ndarray, shape ``(n_spectral, n_channels, n_averages)``
        Complex multi-coil FID data.
    rank_coil : int, optional
        Number of coil modes to retain (default 1).

    Returns
    -------
    combined : np.ndarray, shape ``(n_spectral, n_averages)``
        Combined complex FID (for ``rank_coil=1``, a single combined
        signal; for higher ranks the first mode is returned).

    Notes
    -----
    For ``rank_coil = 1`` this reduces to using the first left singular
    vector of the coil-mode unfolding as the combination weight vector.

    References
    ----------
    Clarke et al. 2021, FSL-MRS.
    de Graaf, *In Vivo NMR Spectroscopy*, section 10.3.
    Kolda TG, Bader BW.  Tensor decompositions and applications.
    *SIAM Review* 51(3):455-500, 2009.
    """
    data = np.asarray(data, dtype=np.complex128)
    n_spectral, n_channels, n_averages = data.shape

    # Mode-2 unfolding (coil mode): reshape to (n_channels, n_spectral * n_averages)
    unfolded = data.transpose(1, 0, 2).reshape(n_channels, -1)

    # SVD on the unfolded matrix
    U, s, Vh = scipy_svd(unfolded, full_matrices=False)

    # Dominant coil weight vector (first left singular vector)
    w = U[:, :rank_coil]                            # (n_channels, rank_coil)

    # Combine: project data onto the dominant coil mode(s)
    # For rank_coil = 1 this yields (n_spectral, n_averages)
    # w^H @ data[spectral, :, avg] for each spectral, avg
    combined = np.einsum("cr,scn->srn", w.conj(), data)

    if rank_coil == 1:
        combined = combined[:, 0, :]                # squeeze rank dim
    else:
        # Return only the first mode for consistency
        combined = combined[:, 0, :]

    return combined
