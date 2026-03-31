"""TDD tests for MRS preprocessing functions.

Tests for apodization (exponential & Gaussian), eddy current correction,
and frequency referencing to metabolite peaks (NAA, Cr).

All tests use synthetic FID data -- no external data dependencies.

References:
    de Graaf (2019) In Vivo NMR Spectroscopy, 3rd ed. Wiley.
    Klose (1990) In vivo proton spectroscopy in the presence of eddy
    currents. MRM 14:26-30.
"""
import numpy as np
import pytest

from neurojax.analysis.mrs_preproc import (
    exponential_apodization,
    gaussian_apodization,
    eddy_current_correction,
    frequency_reference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_singlet(ppm: float, amplitude: float, lw: float, n_pts: int,
                 dwell: float, cf: float) -> np.ndarray:
    """Create a single Lorentzian FID at a given ppm."""
    freq_hz = (ppm - 4.65) * (cf / 1e6)
    t = np.arange(n_pts) * dwell
    return amplitude * np.exp(2j * np.pi * freq_hz * t) * np.exp(-np.pi * lw * t)


def ppm_axis(n: int, dwell: float, cf: float) -> np.ndarray:
    """Compute chemical shift axis in ppm."""
    freq = np.fft.fftshift(np.fft.fftfreq(n, dwell))
    return freq / (cf / 1e6) + 4.65


# Standard parameters for synthetic data
N_PTS = 2048
DWELL = 2.5e-4      # 4 kHz bandwidth
CF = 123.25e6        # ~3T
LW_NAA = 3.0         # Hz
LW_CR = 4.0          # Hz


# ---------------------------------------------------------------------------
# Apodization tests
# ---------------------------------------------------------------------------

class TestExponentialApodization:
    """Exponential apodization applies exp(-pi * broadening * t) window."""

    def test_exponential_apodization(self):
        """Broadens linewidth by the expected amount."""
        fid = make_singlet(2.01, 10.0, LW_NAA, N_PTS, DWELL, CF)
        broadening = 5.0  # Hz

        fid_apod = exponential_apodization(fid, DWELL, broadening)

        # In frequency domain, the peak should be broader.
        # Original linewidth = LW_NAA, after apodization = LW_NAA + broadening.
        # Check that the FID decays faster (compare magnitude at midpoint).
        mid = N_PTS // 2
        # Apodized FID should have smaller magnitude at midpoint
        assert np.abs(fid_apod[mid]) < np.abs(fid[mid]), (
            "Apodized FID should decay faster than original"
        )

        # Quantitative check: ratio at midpoint should be ~exp(-pi*broadening*t_mid)
        t_mid = mid * DWELL
        expected_ratio = np.exp(-np.pi * broadening * t_mid)
        actual_ratio = np.abs(fid_apod[mid]) / np.abs(fid[mid])
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-6)


class TestGaussianApodization:
    """Gaussian apodization applies Gaussian window."""

    def test_gaussian_apodization(self):
        """Applies a Gaussian envelope that attenuates the tail."""
        fid = make_singlet(2.01, 10.0, LW_NAA, N_PTS, DWELL, CF)
        broadening = 5.0  # Hz

        fid_apod = gaussian_apodization(fid, DWELL, broadening)

        # Gaussian decays faster than exponential at late times
        late = int(0.8 * N_PTS)
        assert np.abs(fid_apod[late]) < np.abs(fid[late]), (
            "Gaussian apodization should suppress FID tail"
        )

        # At t=0 the value should be unchanged (Gaussian window = 1 at t=0)
        np.testing.assert_allclose(fid_apod[0], fid[0], rtol=1e-10)


class TestApodizationPreservesShape:
    """Apodization output has the same shape as input."""

    def test_apodization_preserves_shape(self):
        fid = make_singlet(2.01, 10.0, LW_NAA, N_PTS, DWELL, CF)
        assert exponential_apodization(fid, DWELL, 5.0).shape == fid.shape
        assert gaussian_apodization(fid, DWELL, 5.0).shape == fid.shape

        # Also for multi-dimensional input (n_spec, n_coils)
        fid_2d = np.stack([fid, fid * 0.5], axis=1)  # (N_PTS, 2)
        assert exponential_apodization(fid_2d, DWELL, 5.0).shape == fid_2d.shape
        assert gaussian_apodization(fid_2d, DWELL, 5.0).shape == fid_2d.shape


class TestApodizationZeroBroadening:
    """Zero broadening leaves the FID unchanged."""

    def test_apodization_zero_broadening_unchanged(self):
        fid = make_singlet(2.01, 10.0, LW_NAA, N_PTS, DWELL, CF)

        fid_exp = exponential_apodization(fid, DWELL, 0.0)
        np.testing.assert_array_equal(fid_exp, fid)

        fid_gauss = gaussian_apodization(fid, DWELL, 0.0)
        np.testing.assert_array_equal(fid_gauss, fid)


# ---------------------------------------------------------------------------
# Eddy current correction tests
# ---------------------------------------------------------------------------

class TestECCRemovesPhaseDistortion:
    """ECC using water reference removes time-dependent phase from FID."""

    def test_ecc_removes_phase_distortion(self):
        """Add time-dependent phase to both FID and water, then correct."""
        rng = np.random.default_rng(42)
        fid_clean = make_singlet(2.01, 10.0, LW_NAA, N_PTS, DWELL, CF)

        # Simulate eddy-current-induced phase: slowly varying in time
        t = np.arange(N_PTS) * DWELL
        ec_phase = 0.5 * np.exp(-t / 0.1) + 0.2 * np.exp(-t / 0.5)

        # Apply distortion
        distortion = np.exp(1j * ec_phase)
        fid_distorted = fid_clean * distortion

        # Water reference with same eddy-current phase
        water = np.exp(-np.pi * 2.0 * t) * distortion

        # Correct
        fid_corrected = eddy_current_correction(fid_distorted, water)

        # After correction, the phase of the corrected FID should match
        # the phase of the clean FID (up to a global constant)
        phase_corrected = np.angle(fid_corrected[:100])
        phase_clean = np.angle(fid_clean[:100])

        # Remove global phase offset
        offset = phase_corrected[0] - phase_clean[0]
        phase_diff = phase_corrected - phase_clean - offset

        # Phase difference should be near zero
        np.testing.assert_allclose(phase_diff, 0.0, atol=0.05)


class TestECCPreservesAmplitude:
    """ECC should not change the signal magnitude."""

    def test_ecc_preserves_amplitude(self):
        t = np.arange(N_PTS) * DWELL
        fid = make_singlet(2.01, 10.0, LW_NAA, N_PTS, DWELL, CF)

        # Add some eddy current phase
        ec_phase = 0.3 * np.exp(-t / 0.2)
        distortion = np.exp(1j * ec_phase)
        fid_distorted = fid * distortion
        water = np.exp(-np.pi * 2.0 * t) * distortion

        fid_corrected = eddy_current_correction(fid_distorted, water)

        # Magnitude should be preserved
        np.testing.assert_allclose(
            np.abs(fid_corrected), np.abs(fid_distorted), rtol=1e-10
        )


# ---------------------------------------------------------------------------
# Frequency referencing tests
# ---------------------------------------------------------------------------

class TestFrequencyReferenceToNAA:
    """Shifts spectrum so NAA peak is at 2.01 ppm."""

    def test_frequency_reference_to_naa(self):
        # Create FID with NAA slightly off-resonance (shifted by 5 Hz)
        shift_hz = 5.0
        naa_actual_ppm = 2.01 + shift_hz / (CF / 1e6)
        fid = make_singlet(naa_actual_ppm, 10.0, LW_NAA, N_PTS, DWELL, CF)

        fid_ref = frequency_reference(
            fid, DWELL, CF, target_ppm=2.01, target_peak_ppm=2.01
        )

        # After referencing, the peak in the spectrum should be at 2.01 ppm
        spec = np.fft.fftshift(np.fft.fft(fid_ref))
        ppm = ppm_axis(N_PTS, DWELL, CF)

        # Find peak location in the NAA region
        naa_mask = (ppm > 1.8) & (ppm < 2.2)
        peak_idx = np.argmax(np.abs(spec[naa_mask]))
        peak_ppm = ppm[naa_mask][peak_idx]

        # Peak should be within 0.02 ppm of target
        assert abs(peak_ppm - 2.01) < 0.02, (
            f"NAA peak at {peak_ppm:.3f} ppm, expected ~2.01 ppm"
        )


class TestFrequencyReferenceToCr:
    """Shifts spectrum so Cr peak is at 3.03 ppm."""

    def test_frequency_reference_to_cr(self):
        # Create FID with Cr slightly off-resonance (shifted by -3 Hz)
        shift_hz = -3.0
        cr_actual_ppm = 3.03 + shift_hz / (CF / 1e6)
        fid = make_singlet(cr_actual_ppm, 8.0, LW_CR, N_PTS, DWELL, CF)

        fid_ref = frequency_reference(
            fid, DWELL, CF, target_ppm=3.03, target_peak_ppm=3.03
        )

        # After referencing, the peak should be at 3.03 ppm
        spec = np.fft.fftshift(np.fft.fft(fid_ref))
        ppm = ppm_axis(N_PTS, DWELL, CF)

        cr_mask = (ppm > 2.8) & (ppm < 3.2)
        peak_idx = np.argmax(np.abs(spec[cr_mask]))
        peak_ppm = ppm[cr_mask][peak_idx]

        assert abs(peak_ppm - 3.03) < 0.02, (
            f"Cr peak at {peak_ppm:.3f} ppm, expected ~3.03 ppm"
        )
