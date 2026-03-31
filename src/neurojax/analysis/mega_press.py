"""MEGA-PRESS spectral editing pipeline for GABA quantification.

Implements the standard MEGA-PRESS processing chain:
1. Coil combination (SVD or weighted)
2. Frequency/phase alignment of individual transients (spectral registration)
3. Edit-ON/OFF separation and averaging
4. Difference spectrum computation (GABA at 3.0 ppm)
5. Optional: edit-OFF spectrum for standard metabolites

References:
    Near et al. (2015) Frequency and phase drift correction of magnetic
    resonance spectroscopy data by spectral registration in the time domain.
    MRM 73:44-50.

    Edden et al. (2014) Gannet: A batch-processing tool for the quantitative
    analysis of GABA-edited MRS. JMRI 40:1445-1452.
"""

import numpy as np
from typing import NamedTuple


class MegaPressResult(NamedTuple):
    """Result of MEGA-PRESS processing."""
    diff: np.ndarray          # Difference spectrum (edit-ON - edit-OFF)
    edit_on: np.ndarray       # Averaged edit-ON spectrum
    edit_off: np.ndarray      # Averaged edit-OFF spectrum (standard metabs)
    sum_spec: np.ndarray      # Sum spectrum (edit-ON + edit-OFF)
    freq_shifts: np.ndarray   # Per-transient frequency corrections (Hz)
    phase_shifts: np.ndarray  # Per-transient phase corrections (rad)
    rejected: np.ndarray      # Boolean mask of rejected transients
    n_averages: int           # Number of averages used per condition
    dwell_time: float         # Dwell time in seconds
    bandwidth: float          # Spectral bandwidth in Hz


def coil_combine_svd(data: np.ndarray) -> np.ndarray:
    """Combine multi-coil data using SVD (first singular vector).

    Parameters
    ----------
    data : ndarray, shape (n_spec, n_coils, ...)
        Multi-coil spectral data. First two dims must be spectral × coils.

    Returns
    -------
    combined : ndarray, shape (n_spec, ...)
        Coil-combined data.
    """
    orig_shape = data.shape
    n_spec, n_coils = orig_shape[:2]
    extra_dims = orig_shape[2:]

    # Reshape to (n_spec, n_coils, n_extra)
    data_2d = data.reshape(n_spec, n_coils, -1)
    n_extra = data_2d.shape[2]

    # Use first transient for weight estimation
    ref = data_2d[:, :, 0]  # (n_spec, n_coils)

    # SVD on coil dimension
    U, S, Vh = np.linalg.svd(ref.T, full_matrices=False)
    weights = U[:, 0].conj()  # First right singular vector

    # Phase-align weights to first coil
    weights *= np.exp(-1j * np.angle(weights[0]))

    # Apply weights
    combined = np.einsum('c,sc...->s...', weights, data_2d)
    return combined.reshape((n_spec,) + extra_dims)


def spectral_registration(
    fid: np.ndarray,
    reference: np.ndarray,
    dwell_time: float,
    freq_range: tuple[float, float] = (1.8, 4.2),
    centre_freq: float = 123.0e6,
) -> tuple[float, float]:
    """Estimate frequency and phase shift of a single FID relative to reference.

    Uses spectral registration in the time domain (Near et al. 2015):
    minimize ||ref - fid * exp(i*(freq*t + phase))||^2 over a frequency range.

    Parameters
    ----------
    fid : ndarray, shape (n_spec,)
        Single FID to align.
    reference : ndarray, shape (n_spec,)
        Reference FID.
    dwell_time : float
        Dwell time in seconds.
    freq_range : tuple
        PPM range to use for alignment.
    centre_freq : float
        Spectrometer frequency in Hz.

    Returns
    -------
    freq_shift : float
        Frequency correction in Hz.
    phase_shift : float
        Phase correction in radians.
    """
    n = len(fid)
    t = np.arange(n) * dwell_time

    # Convert ppm range to frequency indices
    bw = 1.0 / dwell_time
    freq_axis = np.fft.fftshift(np.fft.fftfreq(n, dwell_time))
    ppm_axis = freq_axis / (centre_freq / 1e6) + 4.65  # relative to water

    mask = (ppm_axis >= freq_range[0]) & (ppm_axis <= freq_range[1])

    # Grid search over frequency shifts
    freq_grid = np.linspace(-20, 20, 201)  # ±20 Hz
    best_cost = np.inf
    best_freq = 0.0
    best_phase = 0.0

    ref_spec = np.fft.fftshift(np.fft.fft(reference))

    for df in freq_grid:
        shifted = fid * np.exp(2j * np.pi * df * t)
        shifted_spec = np.fft.fftshift(np.fft.fft(shifted))

        # Optimal phase for this frequency shift
        cross = np.sum(ref_spec[mask].conj() * shifted_spec[mask])
        phi = -np.angle(cross)

        # Cost: residual after phase correction
        corrected = shifted_spec * np.exp(1j * phi)
        cost = np.sum(np.abs(ref_spec[mask] - corrected[mask]) ** 2).real

        if cost < best_cost:
            best_cost = cost
            best_freq = df
            best_phase = phi

    return best_freq, best_phase


def apply_correction(
    fid: np.ndarray,
    freq_shift: float,
    phase_shift: float,
    dwell_time: float,
) -> np.ndarray:
    """Apply frequency and phase correction to a single FID."""
    t = np.arange(len(fid)) * dwell_time
    return fid * np.exp(2j * np.pi * freq_shift * t + 1j * phase_shift)


def align_edit_pairs(
    edit_on: np.ndarray,
    edit_off: np.ndarray,
    dwell_time: float,
    centre_freq: float = 123.0e6,
    freq_range: tuple[float, float] = (1.8, 4.2),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Frequency-and-phase correction per edit pair.

    Estimates the correction from each OFF transient (more stable peaks)
    and applies the SAME correction to both ON[i] and OFF[i], preserving
    their relative phase relationship for clean subtraction.

    Parameters
    ----------
    edit_on : ndarray, shape (n_spec, n_dyn)
        Edit-ON FIDs.
    edit_off : ndarray, shape (n_spec, n_dyn)
        Edit-OFF FIDs.
    dwell_time : float
        Dwell time in seconds.
    centre_freq : float
        Spectrometer frequency in Hz.
    freq_range : tuple
        PPM range for alignment.

    Returns
    -------
    edit_on_aligned : ndarray, shape (n_spec, n_dyn)
    edit_off_aligned : ndarray, shape (n_spec, n_dyn)
    freq_shifts : ndarray, shape (n_dyn,)
        Per-pair frequency corrections in Hz.
    phase_shifts : ndarray, shape (n_dyn,)
        Per-pair phase corrections in radians.
    """
    n_spec, n_dyn = edit_off.shape

    # Reference: mean of all edit-OFF transients
    ref = edit_off.mean(axis=1)

    freq_shifts = np.zeros(n_dyn)
    phase_shifts = np.zeros(n_dyn)

    edit_on_out = edit_on.copy()
    edit_off_out = edit_off.copy()

    for i in range(n_dyn):
        # Estimate correction from OFF (more stable peaks)
        df, dp = spectral_registration(
            edit_off[:, i], ref, dwell_time,
            freq_range=freq_range, centre_freq=centre_freq,
        )
        freq_shifts[i] = df
        phase_shifts[i] = dp

        # Apply SAME correction to both ON and OFF
        edit_off_out[:, i] = apply_correction(edit_off[:, i], df, dp, dwell_time)
        edit_on_out[:, i] = apply_correction(edit_on[:, i], df, dp, dwell_time)

    return edit_on_out, edit_off_out, freq_shifts, phase_shifts


def reject_outliers(
    fids: np.ndarray,
    dwell_time: float,
    threshold: float = 3.0,
) -> np.ndarray:
    """Reject transients with outlier residuals.

    Parameters
    ----------
    fids : ndarray, shape (n_spec, n_transients)
        Aligned FIDs.
    dwell_time : float
        Dwell time in seconds.
    threshold : float
        Number of standard deviations for rejection.

    Returns
    -------
    rejected : ndarray of bool, shape (n_transients,)
        True for rejected transients.
    """
    mean_fid = fids.mean(axis=1)
    residuals = np.sqrt(np.mean(np.abs(fids - mean_fid[:, None]) ** 2, axis=0))
    median_res = np.median(residuals)
    mad = np.median(np.abs(residuals - median_res))
    z_scores = 0.6745 * (residuals - median_res) / (mad + 1e-10)
    return np.abs(z_scores) > threshold


def process_mega_press(
    data: np.ndarray,
    dwell_time: float,
    centre_freq: float = 123.0e6,
    align: bool = True,
    reject: bool = True,
    reject_threshold: float = 3.0,
    paired_alignment: bool = False,
) -> MegaPressResult:
    """Process MEGA-PRESS data from raw multi-coil to difference spectrum.

    Parameters
    ----------
    data : ndarray, shape (n_spec, n_coils, n_edit, n_dyn) or (n_spec, n_edit, n_dyn)
        Raw MEGA-PRESS data. n_edit=2 (edit-ON, edit-OFF).
    dwell_time : float
        Dwell time in seconds.
    centre_freq : float
        Spectrometer frequency in Hz.
    align : bool
        Whether to perform frequency/phase alignment.
    reject : bool
        Whether to reject outlier transients.
    reject_threshold : float
        Z-score threshold for outlier rejection.
    paired_alignment : bool
        If True, use paired frequency-and-phase correction (FPC):
        estimate correction from OFF transients and apply the same
        correction to both ON and OFF for each dynamic. This preserves
        the relative phase relationship for clean subtraction.

    Returns
    -------
    MegaPressResult
    """
    bw = 1.0 / dwell_time

    # Step 1: Coil combination
    if data.ndim == 4:
        # (n_spec, n_coils, n_edit, n_dyn)
        combined = coil_combine_svd(data)  # (n_spec, n_edit, n_dyn)
    elif data.ndim == 3:
        combined = data  # Already single-coil
    else:
        raise ValueError(f"Expected 3D or 4D data, got {data.ndim}D")

    n_spec, n_edit, n_dyn = combined.shape
    assert n_edit == 2, f"Expected 2 edit conditions, got {n_edit}"

    edit_on = combined[:, 0, :]   # (n_spec, n_dyn)
    edit_off = combined[:, 1, :]  # (n_spec, n_dyn)

    # Step 2: Frequency/phase alignment
    freq_shifts = np.zeros(2 * n_dyn)
    phase_shifts = np.zeros(2 * n_dyn)

    if align and paired_alignment:
        # Paired FPC: estimate from OFF, apply same to both ON and OFF
        edit_on, edit_off, pair_freqs, pair_phases = align_edit_pairs(
            edit_on, edit_off, dwell_time, centre_freq=centre_freq,
        )
        # Store: OFF shifts in [:n_dyn], ON shifts in [n_dyn:] (same values)
        freq_shifts[:n_dyn] = pair_freqs
        freq_shifts[n_dyn:] = pair_freqs
        phase_shifts[:n_dyn] = pair_phases
        phase_shifts[n_dyn:] = pair_phases

    elif align:
        # Independent alignment (original behaviour)
        ref = edit_off.mean(axis=1)

        for i in range(n_dyn):
            # Align edit-OFF
            df, dp = spectral_registration(edit_off[:, i], ref, dwell_time, centre_freq=centre_freq)
            freq_shifts[i] = df
            phase_shifts[i] = dp
            edit_off[:, i] = apply_correction(edit_off[:, i], df, dp, dwell_time)

            # Align edit-ON to the same reference
            df, dp = spectral_registration(edit_on[:, i], ref, dwell_time, centre_freq=centre_freq)
            freq_shifts[n_dyn + i] = df
            phase_shifts[n_dyn + i] = dp
            edit_on[:, i] = apply_correction(edit_on[:, i], df, dp, dwell_time)

    # Step 3: Outlier rejection
    rejected = np.zeros(2 * n_dyn, dtype=bool)
    if reject:
        rej_off = reject_outliers(edit_off, dwell_time, reject_threshold)
        rej_on = reject_outliers(edit_on, dwell_time, reject_threshold)
        rejected[:n_dyn] = rej_off
        rejected[n_dyn:] = rej_on

        edit_off = edit_off[:, ~rej_off]
        edit_on = edit_on[:, ~rej_on]

    # Step 4: Average and compute difference
    avg_on = edit_on.mean(axis=1)
    avg_off = edit_off.mean(axis=1)
    diff = avg_on - avg_off
    sum_spec = avg_on + avg_off

    n_used = min(edit_on.shape[1], edit_off.shape[1])

    return MegaPressResult(
        diff=diff,
        edit_on=avg_on,
        edit_off=avg_off,
        sum_spec=sum_spec,
        freq_shifts=freq_shifts,
        phase_shifts=phase_shifts,
        rejected=rejected,
        n_averages=n_used,
        dwell_time=dwell_time,
        bandwidth=bw,
    )
