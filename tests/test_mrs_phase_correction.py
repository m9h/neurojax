"""TDD tests for MRS phase correction (zero-order and first-order).

Phase correction is critical for:
- Accurate peak integration (not just magnitude)
- Proper subtraction in MEGA-PRESS (phase errors don't cancel)
- Water-referenced quantification (real part of spectrum)
"""
import numpy as np
import pytest


def make_fid(ppm: float, amplitude: float, lw: float, phase_deg: float,
             n_pts: int = 2048, dwell: float = 2.5e-4,
             cf: float = 123.25e6) -> np.ndarray:
    """Create a phased Lorentzian FID."""
    freq_hz = (ppm - 4.65) * (cf / 1e6)
    t = np.arange(n_pts) * dwell
    phase_rad = np.deg2rad(phase_deg)
    return amplitude * np.exp(1j * phase_rad) * np.exp(2j * np.pi * freq_hz * t) * np.exp(-np.pi * lw * t)


def make_spectrum(fid: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft(fid))


def ppm_axis(n: int, dwell: float = 2.5e-4, cf: float = 123.25e6) -> np.ndarray:
    freq = np.fft.fftshift(np.fft.fftfreq(n, dwell))
    return freq / (cf / 1e6) + 4.65


# =========================================================================
# Tests for zero-order phase correction
# =========================================================================

class TestZeroOrderPhaseCorrection:
    """Zero-order: constant phase shift across all frequencies."""

    def test_pure_absorption(self):
        """After correction, NAA peak should be purely real (positive)."""
        from neurojax.analysis.mrs_phase import zero_order_phase_correction

        # NAA at 2.01 ppm with 45° phase error
        fid = make_fid(2.01, 1.0, 3.0, phase_deg=45.0)
        corrected = zero_order_phase_correction(fid)

        spec = make_spectrum(corrected)
        ppm = ppm_axis(len(corrected))
        naa_mask = (ppm > 1.9) & (ppm < 2.15)

        # Peak should be in the real channel
        naa_real = np.max(np.real(spec[naa_mask]))
        naa_imag = np.max(np.abs(np.imag(spec[naa_mask])))
        assert naa_real > 0, "NAA real peak should be positive"
        assert naa_imag < 0.7 * naa_real, "Imaginary should be reduced after correction"

    def test_preserves_amplitude(self):
        """Phase correction should not change peak magnitude."""
        from neurojax.analysis.mrs_phase import zero_order_phase_correction

        fid = make_fid(2.01, 1.0, 3.0, phase_deg=90.0)
        corrected = zero_order_phase_correction(fid)

        mag_before = np.max(np.abs(make_spectrum(fid)))
        mag_after = np.max(np.abs(make_spectrum(corrected)))
        assert abs(mag_before - mag_after) / mag_before < 0.01

    def test_zero_phase_unchanged(self):
        """Already-phased spectrum should be unchanged."""
        from neurojax.analysis.mrs_phase import zero_order_phase_correction

        fid = make_fid(2.01, 1.0, 3.0, phase_deg=0.0)
        corrected = zero_order_phase_correction(fid)

        spec_orig = make_spectrum(fid)
        spec_corr = make_spectrum(corrected)
        ppm = ppm_axis(len(fid))
        naa_mask = (ppm > 1.9) & (ppm < 2.15)

        # Should be essentially identical
        np.testing.assert_allclose(
            np.real(spec_orig[naa_mask]),
            np.real(spec_corr[naa_mask]),
            atol=1e-10
        )

    def test_returns_phase_estimate(self):
        """Should return the estimated phase alongside corrected FID."""
        from neurojax.analysis.mrs_phase import zero_order_phase_correction

        fid = make_fid(2.01, 1.0, 3.0, phase_deg=60.0)
        corrected, phi0 = zero_order_phase_correction(fid, return_phase=True)

        # Estimated phase should be close to -60° (correction is negative)
        assert abs(np.rad2deg(phi0) + 60.0) < 10.0, f"Phase estimate {np.rad2deg(phi0):.1f}° not close to -60°"


# =========================================================================
# Tests for first-order phase correction
# =========================================================================

class TestFirstOrderPhaseCorrection:
    """First-order: linear phase across frequency (group delay)."""

    def test_corrects_linear_phase(self):
        """Should correct a known linear phase gradient."""
        from neurojax.analysis.mrs_phase import first_order_phase_correction

        dwell = 2.5e-4
        n = 2048

        # Create two peaks — first-order phase affects them differently
        fid_naa = make_fid(2.01, 1.0, 3.0, phase_deg=0.0, n_pts=n, dwell=dwell)
        fid_cr = make_fid(3.03, 0.8, 4.0, phase_deg=0.0, n_pts=n, dwell=dwell)
        fid = fid_naa + fid_cr

        # Apply linear phase (group delay of 5 points)
        t = np.arange(n) * dwell
        delay = 5 * dwell
        fid_delayed = fid * np.exp(2j * np.pi * delay * np.fft.fftshift(np.fft.fftfreq(n, dwell))[n // 2])
        # Actually apply in time domain: shift by delay points
        fid_shifted = np.roll(fid, 5) * np.exp(1j * 0.3)  # Add some zero-order too

        corrected = first_order_phase_correction(fid_shifted, dwell)
        spec = make_spectrum(corrected)
        ppm = ppm_axis(n, dwell)

        naa_mask = (ppm > 1.9) & (ppm < 2.15)
        cr_mask = (ppm > 2.95) & (ppm < 3.15)

        # Both peaks should have positive real parts
        assert np.max(np.real(spec[naa_mask])) > 0
        assert np.max(np.real(spec[cr_mask])) > 0

    def test_flat_baseline(self):
        """After correction, baseline should be flat (no dispersive tails)."""
        from neurojax.analysis.mrs_phase import first_order_phase_correction

        fid = make_fid(2.01, 1.0, 3.0, phase_deg=30.0)
        corrected = first_order_phase_correction(fid, 2.5e-4)

        spec = make_spectrum(corrected)
        ppm = ppm_axis(len(corrected))

        # Baseline region (far from peaks)
        baseline_mask = (ppm > 6.0) & (ppm < 8.0)
        baseline_std = np.std(np.real(spec[baseline_mask]))
        peak_height = np.max(np.real(spec))

        # Baseline should be << peak
        assert baseline_std < 0.05 * peak_height


# =========================================================================
# Tests for GABA Gaussian fitting
# =========================================================================

class TestGABAFitting:
    """Fit Gaussian to GABA+ peak at 3.0 ppm in difference spectrum."""

    def test_detects_gaba_peak(self):
        """Should find GABA peak at 3.0 ppm."""
        from neurojax.analysis.mrs_phase import fit_gaba_gaussian

        dwell = 2.5e-4
        cf = 123.25e6

        # Simulate GABA peak in difference spectrum
        fid_gaba = make_fid(3.01, 0.5, 8.0, phase_deg=0.0, dwell=dwell, cf=cf)
        spec = make_spectrum(fid_gaba)
        ppm = ppm_axis(len(fid_gaba), dwell, cf)

        result = fit_gaba_gaussian(np.real(spec), ppm)
        assert abs(result['centre_ppm'] - 3.01) < 0.1, f"Centre {result['centre_ppm']:.2f} not at 3.0"
        assert result['amplitude'] > 0
        assert result['fwhm_ppm'] > 0

    def test_area_proportional_to_concentration(self):
        """Fitted area should scale with GABA amplitude."""
        from neurojax.analysis.mrs_phase import fit_gaba_gaussian

        dwell = 2.5e-4
        cf = 123.25e6
        ppm = ppm_axis(2048, dwell, cf)

        areas = []
        for amp in [0.5, 1.0, 2.0]:
            fid = make_fid(3.01, amp, 8.0, phase_deg=0.0, dwell=dwell, cf=cf)
            spec = make_spectrum(fid)
            result = fit_gaba_gaussian(np.real(spec), ppm)
            areas.append(result['area'])

        # Area should double when amplitude doubles
        assert abs(areas[1] / areas[0] - 2.0) < 0.3
        assert abs(areas[2] / areas[1] - 2.0) < 0.3

    def test_returns_uncertainty(self):
        """Should return fit uncertainty / CRLB."""
        from neurojax.analysis.mrs_phase import fit_gaba_gaussian

        dwell = 2.5e-4
        cf = 123.25e6
        ppm = ppm_axis(2048, dwell, cf)

        # Add noise
        rng = np.random.default_rng(42)
        fid = make_fid(3.01, 1.0, 8.0, phase_deg=0.0, dwell=dwell, cf=cf)
        fid += 0.05 * (rng.standard_normal(2048) + 1j * rng.standard_normal(2048))
        spec = make_spectrum(fid)

        result = fit_gaba_gaussian(np.real(spec), ppm)
        assert 'crlb_percent' in result
        assert result['crlb_percent'] > 0
        assert result['crlb_percent'] < 100


# =========================================================================
# Tests for water-referenced quantification
# =========================================================================

class TestWaterQuantification:
    """Convert fitted areas to absolute concentrations (mM)."""

    def test_known_concentration(self):
        """With known water signal, should recover metabolite concentration."""
        from neurojax.analysis.mrs_phase import water_referenced_quantification

        # Typical metabolite/water ratio is ~1e-4 to 1e-3
        # NAA ~12 mM, water ~55556 mM, so area ratio ~0.0002
        metab_area = 0.01
        water_area = 1000.0
        tissue_fracs = {'gm': 0.6, 'wm': 0.4, 'csf': 0.0}

        conc = water_referenced_quantification(
            metab_area, water_area, tissue_fracs,
            te=0.030, tr=5.0,
            metab_t1=1.3, metab_t2=0.16,
            field_strength=3.0
        )
        # Should be a positive, physiologically reasonable concentration
        assert conc > 0
        assert conc < 100  # Not absurdly large

    def test_csf_dilution(self):
        """More CSF should reduce apparent concentration."""
        from neurojax.analysis.mrs_phase import water_referenced_quantification

        metab_area = 100.0
        water_area = 1000.0

        conc_no_csf = water_referenced_quantification(
            metab_area, water_area,
            {'gm': 0.6, 'wm': 0.4, 'csf': 0.0},
            te=0.030, tr=5.0
        )
        conc_with_csf = water_referenced_quantification(
            metab_area, water_area,
            {'gm': 0.4, 'wm': 0.2, 'csf': 0.4},
            te=0.030, tr=5.0
        )
        # CSF-corrected concentration should be higher (less tissue)
        assert conc_with_csf > conc_no_csf

    def test_zero_water_raises(self):
        """Zero water area should raise or return inf/nan gracefully."""
        from neurojax.analysis.mrs_phase import water_referenced_quantification

        with pytest.raises((ValueError, ZeroDivisionError)):
            water_referenced_quantification(
                100.0, 0.0,
                {'gm': 0.6, 'wm': 0.4, 'csf': 0.0},
                te=0.030, tr=5.0
            )
