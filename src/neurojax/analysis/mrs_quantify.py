"""End-to-end MEGA-PRESS quantification pipeline.

Chains the existing processing functions into a single high-level call
that goes from raw TWIX-shaped data to GABA concentration in mM:

    1. process_mega_press()          — coil combine, align, average, difference
    2. zero_order_phase_correction() — phase the difference spectrum
    3. fit_gaba_gaussian()           — Gaussian fit to GABA+ peak at ~3.0 ppm
    4. water_referenced_quantification() — absolute concentration (if water ref)

Returns a comprehensive dict of quantification metrics.

References:
    Edden et al. (2014) Gannet: A batch-processing tool for the quantitative
    analysis of GABA-edited MRS. JMRI 40:1445-1452.

    Gasparovic et al. (2006) Use of tissue water as a concentration reference
    for proton spectroscopic imaging. MRM 55:1219-1226.
"""

import numpy as np

from neurojax.analysis.mega_press import process_mega_press
from neurojax.analysis.mrs_phase import (
    zero_order_phase_correction,
    fit_gaba_gaussian,
    water_referenced_quantification,
)


def _compute_ppm_axis(n: int, dwell_time: float, centre_freq: float) -> np.ndarray:
    """Compute ppm axis from spectral parameters.

    Parameters
    ----------
    n : int
        Number of spectral points.
    dwell_time : float
        Dwell time in seconds.
    centre_freq : float
        Centre frequency in Hz.

    Returns
    -------
    ppm : ndarray, shape (n,)
        Chemical shift axis in ppm (water referenced at 4.65 ppm).
    """
    freq = np.fft.fftshift(np.fft.fftfreq(n, dwell_time))
    cf_mhz = centre_freq / 1e6
    return freq / cf_mhz + 4.65


def _estimate_snr(diff_spec_real: np.ndarray, ppm: np.ndarray,
                  signal_range: tuple[float, float] = (2.8, 3.2),
                  noise_range: tuple[float, float] = (8.0, 10.0)) -> float:
    """Estimate SNR of the GABA peak from the real difference spectrum.

    SNR = max(signal in signal_range) / std(noise in noise_range).

    Parameters
    ----------
    diff_spec_real : ndarray
        Real part of the phased difference spectrum.
    ppm : ndarray
        PPM axis.
    signal_range : tuple
        PPM range for the GABA signal.
    noise_range : tuple
        PPM range for noise estimation.

    Returns
    -------
    snr : float
        Signal-to-noise ratio.
    """
    signal_mask = (ppm >= signal_range[0]) & (ppm <= signal_range[1])
    noise_mask = (ppm >= noise_range[0]) & (ppm <= noise_range[1])

    if not np.any(signal_mask) or not np.any(noise_mask):
        # Fallback: use first/last quarter of spectrum
        n = len(ppm)
        signal_mask = np.zeros(n, dtype=bool)
        signal_mask[n // 4: 3 * n // 4] = True
        noise_mask = np.zeros(n, dtype=bool)
        noise_mask[:n // 8] = True

    signal_max = np.max(np.abs(diff_spec_real[signal_mask]))
    noise_std = np.std(diff_spec_real[noise_mask])

    if noise_std == 0:
        return float('inf')
    return float(signal_max / noise_std)


def _fit_naa_in_edit_off(edit_off_fid: np.ndarray, dwell_time: float,
                          centre_freq: float) -> dict:
    """Fit a Gaussian to the NAA peak at ~2.01 ppm in the edit-OFF spectrum.

    Parameters
    ----------
    edit_off_fid : ndarray
        Edit-OFF averaged FID.
    dwell_time : float
        Dwell time in seconds.
    centre_freq : float
        Centre frequency in Hz.

    Returns
    -------
    dict with keys: centre_ppm, amplitude, area, etc.
    """
    n = len(edit_off_fid)
    ppm = _compute_ppm_axis(n, dwell_time, centre_freq)

    # Phase correct the edit-OFF spectrum
    corrected = zero_order_phase_correction(edit_off_fid)
    spec = np.fft.fftshift(np.fft.fft(corrected))
    spec_real = np.real(spec)

    # Fit NAA Gaussian in the 1.8-2.2 ppm range
    from scipy.optimize import curve_fit

    mask = (ppm >= 1.8) & (ppm <= 2.25)
    x = ppm[mask]
    y = spec_real[mask]

    def gaussian(x, amp, centre, sigma, baseline):
        return amp * np.exp(-0.5 * ((x - centre) / sigma) ** 2) + baseline

    peak_idx = np.argmax(np.abs(y))
    amp0 = y[peak_idx]
    centre0 = x[peak_idx]

    try:
        popt, pcov = curve_fit(
            gaussian, x, y,
            p0=[amp0, centre0, 0.03, np.median(y)],
            bounds=(
                [-np.inf, 1.8, 0.005, -np.inf],
                [np.inf, 2.25, 0.2, np.inf],
            ),
            maxfev=5000,
        )
        amp, centre, sigma, baseline = popt
        area = abs(amp) * abs(sigma) * np.sqrt(2 * np.pi)
        return {
            'centre_ppm': float(centre),
            'amplitude': float(amp),
            'area': float(area),
        }
    except (RuntimeError, ValueError):
        # Fallback: use peak amplitude * approximate width
        return {
            'centre_ppm': float(centre0),
            'amplitude': float(amp0),
            'area': float(abs(amp0) * 0.03 * np.sqrt(2 * np.pi)),
        }


def quantify_mega_press(
    data: np.ndarray,
    dwell_time: float,
    centre_freq: float = 123.25e6,
    water_ref: np.ndarray | None = None,
    tissue_fracs: dict[str, float] | None = None,
    te: float | None = None,
    tr: float | None = None,
    align: bool = True,
    reject: bool = True,
    reject_threshold: float = 3.0,
    paired_alignment: bool = False,
    gaba_fit_range: tuple[float, float] = (2.7, 3.3),
    metab_t1: float = 1.3,
    metab_t2: float = 0.16,
    field_strength: float = 3.0,
) -> dict:
    """End-to-end MEGA-PRESS quantification from raw data to GABA concentration.

    Parameters
    ----------
    data : ndarray, shape (n_spec, n_coils, 2, n_dyn) or (n_spec, 2, n_dyn)
        Raw MEGA-PRESS data. Second-to-last dim must be 2 (ON/OFF).
    dwell_time : float
        Dwell time in seconds.
    centre_freq : float
        Spectrometer centre frequency in Hz.
    water_ref : ndarray, optional
        Unsuppressed water reference FID, shape (n_spec,).
        If provided, absolute quantification is performed.
    tissue_fracs : dict, optional
        Tissue fractions: {'gm': float, 'wm': float, 'csf': float}.
        Required for water-referenced quantification.
    te : float, optional
        Echo time in seconds. Required for water-referenced quantification.
    tr : float, optional
        Repetition time in seconds. Required for water-referenced quantification.
    align : bool
        Perform spectral registration alignment.
    reject : bool
        Reject outlier transients.
    reject_threshold : float
        Z-score threshold for outlier rejection.
    paired_alignment : bool
        Use paired frequency-phase correction.
    gaba_fit_range : tuple
        PPM range for GABA Gaussian fitting.
    metab_t1 : float
        GABA T1 relaxation time in seconds (for water-ref quantification).
    metab_t2 : float
        GABA T2 relaxation time in seconds (for water-ref quantification).
    field_strength : float
        Field strength in Tesla.

    Returns
    -------
    results : dict
        Quantification results with keys:
        - gaba_conc_mM: float or None — absolute GABA concentration (mM)
        - gaba_naa_ratio: float — GABA/NAA area ratio
        - gaba_area: float — fitted GABA peak area
        - naa_area: float — fitted NAA peak area (from edit-OFF)
        - snr: float — GABA peak SNR
        - crlb_percent: float — Cramer-Rao lower bound (%)
        - gaba_centre_ppm: float — fitted GABA peak centre
        - gaba_fwhm_ppm: float — fitted GABA peak FWHM
        - diff_fid: ndarray — difference FID
        - edit_on_fid: ndarray — averaged edit-ON FID
        - edit_off_fid: ndarray — averaged edit-OFF FID
        - sum_spec_fid: ndarray — sum spectrum FID
        - freq_shifts: ndarray — per-transient frequency corrections
        - phase_shifts: ndarray — per-transient phase corrections
        - rejected: ndarray — boolean mask of rejected transients
        - n_averages: int — number of averages used
    """
    # --- Step 1: MEGA-PRESS preprocessing ---
    mega_result = process_mega_press(
        data,
        dwell_time=dwell_time,
        centre_freq=centre_freq,
        align=align,
        reject=reject,
        reject_threshold=reject_threshold,
        paired_alignment=paired_alignment,
    )

    diff_fid = mega_result.diff          # (n_spec,) complex
    edit_on_fid = mega_result.edit_on    # (n_spec,) complex
    edit_off_fid = mega_result.edit_off  # (n_spec,) complex

    n_spec = len(diff_fid)
    ppm = _compute_ppm_axis(n_spec, dwell_time, centre_freq)

    # --- Step 2: Zero-order phase correction of the difference spectrum ---
    diff_phased, phi0 = zero_order_phase_correction(diff_fid, return_phase=True)

    # Compute the phased difference spectrum (frequency domain)
    diff_spec = np.fft.fftshift(np.fft.fft(diff_phased))
    diff_spec_real = np.real(diff_spec)

    # --- Step 3: Fit GABA Gaussian to the phased difference spectrum ---
    gaba_fit = fit_gaba_gaussian(diff_spec_real, ppm, fit_range=gaba_fit_range)
    gaba_area = gaba_fit['area']
    gaba_crlb = gaba_fit['crlb_percent']

    # --- Step 3b: Fit NAA in the edit-OFF spectrum ---
    naa_fit = _fit_naa_in_edit_off(edit_off_fid, dwell_time, centre_freq)
    naa_area = naa_fit['area']

    # GABA/NAA ratio
    if naa_area > 0:
        gaba_naa_ratio = gaba_area / naa_area
    else:
        gaba_naa_ratio = float('inf')

    # SNR
    snr = _estimate_snr(diff_spec_real, ppm)

    # --- Step 4: Water-referenced quantification (if water ref provided) ---
    gaba_conc_mM = None

    if water_ref is not None:
        # Compute water area from the water reference FID
        water_spec = np.fft.fftshift(np.fft.fft(water_ref))
        water_area = float(np.max(np.abs(water_spec)))

        # Default tissue fractions and timing if not provided
        if tissue_fracs is None:
            tissue_fracs = {'gm': 0.6, 'wm': 0.4, 'csf': 0.0}
        if te is None:
            te = 0.068  # 68 ms (typical MEGA-PRESS)
        if tr is None:
            tr = 2.0  # 2 s

        gaba_conc_mM = water_referenced_quantification(
            metab_area=gaba_area,
            water_area=water_area,
            tissue_fracs=tissue_fracs,
            te=te,
            tr=tr,
            metab_t1=metab_t1,
            metab_t2=metab_t2,
            field_strength=field_strength,
        )

    # --- Assemble results ---
    results = {
        # Primary quantification
        'gaba_conc_mM': gaba_conc_mM,
        'gaba_naa_ratio': float(gaba_naa_ratio),
        'gaba_area': float(gaba_area),
        'naa_area': float(naa_area),
        'snr': float(snr),
        'crlb_percent': float(gaba_crlb),

        # GABA fit details
        'gaba_centre_ppm': float(gaba_fit['centre_ppm']),
        'gaba_fwhm_ppm': float(gaba_fit.get('fwhm_ppm', 0.0)),
        'gaba_amplitude': float(gaba_fit['amplitude']),
        'gaba_baseline': float(gaba_fit.get('baseline', 0.0)),
        'gaba_residual': float(gaba_fit.get('residual', 0.0)),

        # NAA fit details
        'naa_centre_ppm': float(naa_fit['centre_ppm']),
        'naa_amplitude': float(naa_fit['amplitude']),

        # Phase correction
        'phase_correction_rad': float(phi0),

        # Preprocessing outputs (FIDs for QC report generation)
        'diff_fid': diff_fid,
        'edit_on_fid': edit_on_fid,
        'edit_off_fid': edit_off_fid,
        'sum_spec_fid': mega_result.sum_spec,
        'freq_shifts': np.asarray(mega_result.freq_shifts),
        'phase_shifts': np.asarray(mega_result.phase_shifts),
        'rejected': np.asarray(mega_result.rejected),
        'n_averages': mega_result.n_averages,

        # Acquisition parameters
        'dwell_time': dwell_time,
        'centre_freq': centre_freq,
        'bandwidth': mega_result.bandwidth,
    }

    return results
