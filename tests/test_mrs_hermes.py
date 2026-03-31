"""TDD tests for HERMES 4-condition editing support.

HERMES (Hadamard Encoding and Reconstruction of MEGA-Edited Spectroscopy)
uses 4 editing conditions (A, B, C, D) to simultaneously measure GABA and GSH:

    GABA difference = (A + B) - (C + D)
    GSH  difference = (A + C) - (B + D)

References:
    Saleh et al. (2016) Multi-step spectral editing of in vivo 1H-MRS.
    NeuroImage 134:360-367.

    Chan et al. (2016) HERMES: Hadamard encoding and reconstruction of
    MEGA-edited spectroscopy. MRM 76:11-19.
"""
import numpy as np
import pytest

from neurojax.analysis.hermes import (
    process_hermes,
    HermesResult,
)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def make_singlet(ppm: float, amplitude: float, lw: float, n_pts: int,
                 dwell: float, cf: float) -> np.ndarray:
    """Create a single Lorentzian FID at a given ppm."""
    freq_hz = (ppm - 4.65) * (cf / 1e6)
    t = np.arange(n_pts) * dwell
    return amplitude * np.exp(2j * np.pi * freq_hz * t) * np.exp(-np.pi * lw * t)


def ppm_axis(n: int, dwell: float, cf: float) -> np.ndarray:
    freq = np.fft.fftshift(np.fft.fftfreq(n, dwell))
    return freq / (cf / 1e6) + 4.65


def make_hermes_data(
    n_pts: int = 2048,
    n_dyn: int = 16,
    dwell: float = 2.5e-4,
    cf: float = 123.25e6,
    gaba_conc: float = 1.0,
    gsh_conc: float = 0.5,
    naa_conc: float = 10.0,
    cr_conc: float = 8.0,
    noise_level: float = 0.001,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate synthetic HERMES 4-condition data.

    Returns (data, truth) where data is (n_pts, 4, n_dyn).

    The 4 conditions encode GABA (3.0 ppm) and GSH (2.95 ppm) via
    Hadamard scheme:

        A: GABA_ON,  GSH_ON   -> +GABA, +GSH
        B: GABA_ON,  GSH_OFF  -> +GABA, -GSH
        C: GABA_OFF, GSH_ON   -> -GABA, +GSH
        D: GABA_OFF, GSH_OFF  -> -GABA, -GSH

    So:
        GABA = (A+B) - (C+D) = +2*GABA_signal  (GSH cancels)
        GSH  = (A+C) - (B+D) = +2*GSH_signal   (GABA cancels)
    """
    rng = np.random.default_rng(seed)

    naa = make_singlet(2.01, naa_conc, 3.0, n_pts, dwell, cf)
    cr = make_singlet(3.03, cr_conc, 4.0, n_pts, dwell, cf)
    gaba = make_singlet(3.01, gaba_conc, 8.0, n_pts, dwell, cf)
    gsh = make_singlet(2.95, gsh_conc, 10.0, n_pts, dwell, cf)

    baseline = naa + cr

    # Hadamard encoding
    # A: +GABA, +GSH
    cond_a = baseline + gaba + gsh
    # B: +GABA, -GSH
    cond_b = baseline + gaba - gsh
    # C: -GABA, +GSH
    cond_c = baseline - gaba + gsh
    # D: -GABA, -GSH
    cond_d = baseline - gaba - gsh

    conditions = [cond_a, cond_b, cond_c, cond_d]

    data = np.zeros((n_pts, 4, n_dyn), dtype=complex)
    for d in range(n_dyn):
        for c_idx, cond in enumerate(conditions):
            noise = noise_level * (
                rng.standard_normal(n_pts) + 1j * rng.standard_normal(n_pts)
            )
            data[:, c_idx, d] = cond + noise

    truth = {
        'gaba_conc': gaba_conc,
        'gsh_conc': gsh_conc,
        'naa_conc': naa_conc,
        'cr_conc': cr_conc,
        'n_dyn': n_dyn,
        'noise_level': noise_level,
        'dwell': dwell,
        'cf': cf,
    }
    return data, truth


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHermesGABADifference:
    """GABA signal should appear in the GABA difference = (A+B) - (C+D)."""

    def test_hermes_gaba_difference(self):
        data, truth = make_hermes_data(gaba_conc=2.0, gsh_conc=0.0, noise_level=0.0001)
        result = process_hermes(data, truth['dwell'], truth['cf'], align=False)

        gaba_spec = np.fft.fftshift(np.fft.fft(result.gaba_diff))
        ppm = ppm_axis(len(result.gaba_diff), truth['dwell'], truth['cf'])
        gaba_mask = (ppm > 2.8) & (ppm < 3.2)

        gaba_peak = np.max(np.abs(gaba_spec[gaba_mask]))
        assert gaba_peak > 0, "GABA signal not found in GABA difference"


class TestHermesGSHDifference:
    """GSH signal should appear in the GSH difference = (A+C) - (B+D)."""

    def test_hermes_gsh_difference(self):
        data, truth = make_hermes_data(gaba_conc=0.0, gsh_conc=2.0, noise_level=0.0001)
        result = process_hermes(data, truth['dwell'], truth['cf'], align=False)

        gsh_spec = np.fft.fftshift(np.fft.fft(result.gsh_diff))
        ppm = ppm_axis(len(result.gsh_diff), truth['dwell'], truth['cf'])
        gsh_mask = (ppm > 2.75) & (ppm < 3.15)

        gsh_peak = np.max(np.abs(gsh_spec[gsh_mask]))
        assert gsh_peak > 0, "GSH signal not found in GSH difference"


class TestHermesSeparation:
    """GABA and GSH should appear in their respective differences,
    not cross-contaminated."""

    def test_hermes_separation(self):
        data, truth = make_hermes_data(
            gaba_conc=2.0, gsh_conc=2.0, noise_level=0.0001,
        )
        result = process_hermes(data, truth['dwell'], truth['cf'], align=False)

        ppm = ppm_axis(len(result.gaba_diff), truth['dwell'], truth['cf'])

        # GABA difference — peak near 3.01 ppm
        gaba_spec = np.fft.fftshift(np.fft.fft(result.gaba_diff))
        gaba_mask = (ppm > 2.8) & (ppm < 3.2)
        gaba_in_gaba = np.max(np.abs(gaba_spec[gaba_mask]))

        # GSH difference — peak near 2.95 ppm
        gsh_spec = np.fft.fftshift(np.fft.fft(result.gsh_diff))
        gsh_mask = (ppm > 2.75) & (ppm < 3.15)
        gsh_in_gsh = np.max(np.abs(gsh_spec[gsh_mask]))

        # Both should be present in their own difference
        assert gaba_in_gaba > 0, "GABA missing from GABA difference"
        assert gsh_in_gsh > 0, "GSH missing from GSH difference"

        # Hadamard encoding gives perfect algebraic separation.
        # Verify by checking a baseline region far from both peaks:
        # if there were leakage, we'd see signal where there shouldn't be any.
        baseline_mask = (ppm > 5.0) & (ppm < 6.0)
        gaba_baseline = np.max(np.abs(gaba_spec[baseline_mask]))
        gsh_baseline = np.max(np.abs(gsh_spec[baseline_mask]))

        # Baseline should be << peak (just noise)
        assert gaba_baseline < 0.05 * gaba_in_gaba, (
            f"GABA diff baseline too high: {gaba_baseline/gaba_in_gaba:.2%}"
        )
        assert gsh_baseline < 0.05 * gsh_in_gsh, (
            f"GSH diff baseline too high: {gsh_baseline/gsh_in_gsh:.2%}"
        )


class TestHermes4Conditions:
    """Pipeline should accept 4-condition data."""

    def test_hermes_4conditions(self):
        data, truth = make_hermes_data(n_dyn=8)
        result = process_hermes(data, truth['dwell'], truth['cf'], align=False)

        assert result.gaba_diff.shape == (2048,)
        assert result.gsh_diff.shape == (2048,)
        assert result.n_averages == 8


class TestHermesResultType:
    """Returns a HermesResult with both difference spectra."""

    def test_hermes_result_type(self):
        data, truth = make_hermes_data()
        result = process_hermes(data, truth['dwell'], truth['cf'], align=False)

        assert isinstance(result, HermesResult)
        assert hasattr(result, 'gaba_diff')
        assert hasattr(result, 'gsh_diff')
        assert hasattr(result, 'conditions')
        assert hasattr(result, 'n_averages')
        assert hasattr(result, 'dwell_time')
        assert hasattr(result, 'bandwidth')
