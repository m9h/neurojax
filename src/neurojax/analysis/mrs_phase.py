"""MRS phase correction and quantification.

Implements:
- Zero-order phase correction (constant phase)
- First-order phase correction (linear phase / group delay)
- GABA Gaussian fitting for MEGA-PRESS difference spectra
- Water-referenced absolute quantification (mM)

References:
    Ernst T (1992) Phase correction in NMR spectroscopy.
    Near J et al. (2021) Preprocessing, analysis and quantification in
    single-voxel magnetic resonance spectroscopy. NMR Biomed 34:e4257.
"""

import numpy as np
from scipy.optimize import minimize_scalar, curve_fit


def zero_order_phase_correction(
    fid: np.ndarray,
    return_phase: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Estimate and correct zero-order phase using the largest peak.

    Finds the phase that maximizes the real part of the tallest spectral peak,
    ensuring it appears as pure absorption.

    Parameters
    ----------
    fid : ndarray, shape (n_spec,)
        Complex FID.
    return_phase : bool
        If True, return (corrected_fid, phase_rad).

    Returns
    -------
    corrected : ndarray
        Phase-corrected FID.
    phi0 : float (optional)
        Applied zero-order phase in radians.
    """
    spec = np.fft.fftshift(np.fft.fft(fid))
    n = len(fid)

    # Maximize the sum of the real part of the spectrum over a grid of phases
    # This is more robust than using a single peak (which might be water)
    best_phi = 0.0
    best_score = -np.inf
    for phi in np.linspace(-np.pi, np.pi, 361):
        score = np.sum(np.real(spec * np.exp(1j * phi)))
        if score > best_score:
            best_score = score
            best_phi = phi

    phi0 = best_phi

    corrected = fid * np.exp(1j * phi0)

    if return_phase:
        return corrected, phi0
    return corrected


def first_order_phase_correction(
    fid: np.ndarray,
    dwell_time: float,
    ppm_range: tuple[float, float] = (0.5, 4.2),
    cf: float = 123.25e6,
) -> np.ndarray:
    """Estimate and correct zero + first-order phase.

    Minimizes dispersive signal (imaginary part) across the spectrum by
    jointly optimizing phi0 (constant) and phi1 (linear with frequency).

    Parameters
    ----------
    fid : ndarray, shape (n_spec,)
        Complex FID.
    dwell_time : float
        Dwell time in seconds.
    ppm_range : tuple
        PPM range to optimize over.
    cf : float
        Centre frequency in Hz.

    Returns
    -------
    corrected : ndarray
        Phase-corrected FID.
    """
    n = len(fid)
    spec = np.fft.fftshift(np.fft.fft(fid))
    freq = np.fft.fftshift(np.fft.fftfreq(n, dwell_time))
    ppm = freq / (cf / 1e6) + 4.65

    mask = (ppm >= ppm_range[0]) & (ppm <= ppm_range[1])

    def cost(params):
        phi0, phi1 = params
        phase = phi0 + phi1 * (freq - freq[n // 2])
        corrected_spec = spec * np.exp(1j * phase)
        # Minimize: sum of absolute imaginary part (want pure absorption)
        # Maximize: sum of real part (want positive peaks)
        return -np.sum(np.real(corrected_spec[mask]))

    from scipy.optimize import minimize
    result = minimize(cost, [0.0, 0.0], method='Nelder-Mead',
                      options={'xatol': 1e-6, 'fatol': 1e-6})
    phi0, phi1 = result.x

    # Apply correction in frequency domain, convert back
    phase = phi0 + phi1 * (freq - freq[n // 2])
    corrected_spec = spec * np.exp(1j * phase)
    corrected_fid = np.fft.ifft(np.fft.ifftshift(corrected_spec))

    return corrected_fid


def fit_gaba_gaussian(
    spectrum_real: np.ndarray,
    ppm: np.ndarray,
    fit_range: tuple[float, float] = (2.7, 3.3),
) -> dict:
    """Fit a Gaussian to the GABA+ peak at ~3.0 ppm in a difference spectrum.

    Parameters
    ----------
    spectrum_real : ndarray
        Real part of the phased difference spectrum.
    ppm : ndarray
        PPM axis (same length as spectrum_real).
    fit_range : tuple
        PPM range for fitting.

    Returns
    -------
    dict with keys:
        centre_ppm, amplitude, fwhm_ppm, area, crlb_percent, residual
    """
    mask = (ppm >= fit_range[0]) & (ppm <= fit_range[1])
    x = ppm[mask]
    y = spectrum_real[mask]

    def gaussian(x, amp, centre, sigma, baseline):
        return amp * np.exp(-0.5 * ((x - centre) / sigma) ** 2) + baseline

    # Initial guesses
    peak_idx = np.argmax(np.abs(y))
    amp0 = y[peak_idx]
    centre0 = x[peak_idx]
    sigma0 = 0.05  # ~0.12 ppm FWHM
    baseline0 = np.median(y)

    try:
        popt, pcov = curve_fit(
            gaussian, x, y,
            p0=[amp0, centre0, sigma0, baseline0],
            bounds=(
                [-np.inf, fit_range[0], 0.01, -np.inf],
                [np.inf, fit_range[1], 0.3, np.inf]
            ),
            maxfev=5000,
        )

        amp, centre, sigma, baseline = popt
        fwhm = 2.355 * abs(sigma)  # FWHM = 2*sqrt(2*ln2) * sigma
        area = abs(amp) * abs(sigma) * np.sqrt(2 * np.pi)

        # CRLB from covariance
        perr = np.sqrt(np.diag(pcov))
        amp_err = perr[0]
        crlb = 100 * abs(amp_err / amp) if abs(amp) > 0 else 999.0

        # Residual
        fitted = gaussian(x, *popt)
        residual = np.sqrt(np.mean((y - fitted) ** 2))

        return {
            'centre_ppm': float(centre),
            'amplitude': float(amp),
            'sigma_ppm': float(sigma),
            'fwhm_ppm': float(fwhm),
            'area': float(area),
            'baseline': float(baseline),
            'crlb_percent': float(crlb),
            'residual': float(residual),
        }

    except (RuntimeError, ValueError) as e:
        return {
            'centre_ppm': float(centre0),
            'amplitude': float(amp0),
            'sigma_ppm': 0.0,
            'fwhm_ppm': 0.0,
            'area': 0.0,
            'baseline': 0.0,
            'crlb_percent': 999.0,
            'residual': float('inf'),
            'error': str(e),
        }


def water_referenced_quantification(
    metab_area: float,
    water_area: float,
    tissue_fracs: dict[str, float],
    te: float,
    tr: float,
    metab_t1: float = 1.3,
    metab_t2: float = 0.16,
    field_strength: float = 3.0,
) -> float:
    """Convert metabolite area to absolute concentration (mM) using water reference.

    Uses the standard formula from Gasparovic et al. (2006):
        [M] = (S_M / S_W) * f_W * R * [W]

    where R accounts for relaxation differences and f_W for tissue water content.

    Parameters
    ----------
    metab_area : float
        Metabolite signal area (fitted).
    water_area : float
        Water reference signal area.
    tissue_fracs : dict
        Tissue fractions: {'gm': float, 'wm': float, 'csf': float}.
    te : float
        Echo time in seconds.
    tr : float
        Repetition time in seconds.
    metab_t1, metab_t2 : float
        Metabolite T1 and T2 relaxation times in seconds.
    field_strength : float
        Static field strength in Tesla (for water relaxation lookup).

    Returns
    -------
    concentration : float
        Metabolite concentration in mM.

    Raises
    ------
    ValueError
        If water_area is zero.
    """
    if water_area == 0:
        raise ValueError("Water area cannot be zero")

    f_gm = tissue_fracs.get('gm', 0.6)
    f_wm = tissue_fracs.get('wm', 0.4)
    f_csf = tissue_fracs.get('csf', 0.0)

    # Water content per tissue type (g/ml)
    alpha_gm = 0.78
    alpha_wm = 0.65
    alpha_csf = 0.97

    # Water relaxation times at 3T (Wansapura 1999, Stanisz 2005)
    if field_strength < 5:
        t1_gm_w, t2_gm_w = 1.33, 0.110
        t1_wm_w, t2_wm_w = 0.83, 0.080
        t1_csf_w, t2_csf_w = 4.16, 1.650
    else:
        # 7T values
        t1_gm_w, t2_gm_w = 2.13, 0.045
        t1_wm_w, t2_wm_w = 1.22, 0.027
        t1_csf_w, t2_csf_w = 4.43, 0.800

    # Water signal attenuation (relaxation-weighted water content)
    def water_atten(f, alpha, t1, t2):
        return f * alpha * (1 - np.exp(-tr / t1)) * np.exp(-te / t2)

    f_w = (water_atten(f_gm, alpha_gm, t1_gm_w, t2_gm_w)
           + water_atten(f_wm, alpha_wm, t1_wm_w, t2_wm_w)
           + water_atten(f_csf, alpha_csf, t1_csf_w, t2_csf_w))

    # Metabolite relaxation correction
    r_metab = (1 - np.exp(-tr / metab_t1)) * np.exp(-te / metab_t2)

    # Pure water concentration
    water_conc = 55556.0  # mM (pure water = 1000g/L / 18.015 g/mol * 1000)

    # Tissue fraction for metabolite (exclude CSF)
    f_tissue = f_gm + f_wm
    if f_tissue == 0:
        raise ValueError("No tissue fraction (all CSF)")

    # Concentration
    concentration = (metab_area / water_area) * (f_w / r_metab) * water_conc / f_tissue

    return concentration
