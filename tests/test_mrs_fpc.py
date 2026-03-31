"""TDD tests for Frequency-and-Phase Correction per edit pair (FPC).

Problem: Independent alignment of edit-ON and edit-OFF transients can
introduce differential frequency/phase errors between ON[i] and OFF[i],
creating subtraction artifacts in the difference spectrum.

Solution: Estimate correction from the OFF transient (more stable peaks)
and apply the SAME correction to both ON[i] and OFF[i] for each dynamic.

References:
    Mikkelsen et al. (2018) Robust frequency and phase correction for
    MEGA-PRESS spectral editing. MRM 80:48-56.
"""
import numpy as np
import pytest

from neurojax.analysis.mega_press import (
    process_mega_press,
    align_edit_pairs,
    apply_correction,
    MegaPressResult,
)


# ---------------------------------------------------------------------------
# Synthetic data generators (same pattern as test_mega_press.py)
# ---------------------------------------------------------------------------

def make_singlet(ppm: float, amplitude: float, lw: float, n_pts: int,
                 dwell: float, cf: float) -> np.ndarray:
    """Create a single Lorentzian FID at a given ppm."""
    freq_hz = (ppm - 4.65) * (cf / 1e6)
    t = np.arange(n_pts) * dwell
    return amplitude * np.exp(2j * np.pi * freq_hz * t) * np.exp(-np.pi * lw * t)


def make_mega_pair_data(
    n_pts: int = 2048,
    n_dyn: int = 32,
    dwell: float = 2.5e-4,
    cf: float = 123.25e6,
    gaba_conc: float = 1.0,
    naa_conc: float = 10.0,
    cr_conc: float = 8.0,
    noise_level: float = 0.001,
    freq_drift_hz: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate synthetic single-coil MEGA-PRESS data.

    Returns (data, truth) where data is (n_pts, 2, n_dyn).
    """
    rng = np.random.default_rng(seed)

    naa = make_singlet(2.01, naa_conc, 3.0, n_pts, dwell, cf)
    cr = make_singlet(3.03, cr_conc, 4.0, n_pts, dwell, cf)
    gaba = make_singlet(3.01, gaba_conc, 8.0, n_pts, dwell, cf)

    edit_on_signal = naa + cr + gaba
    edit_off_signal = naa + cr - gaba

    t = np.arange(n_pts) * dwell
    data = np.zeros((n_pts, 2, n_dyn), dtype=complex)

    for d in range(n_dyn):
        drift = freq_drift_hz * d / max(n_dyn - 1, 1)
        phase_mod = np.exp(2j * np.pi * drift * t)

        noise_on = noise_level * (rng.standard_normal(n_pts) + 1j * rng.standard_normal(n_pts))
        noise_off = noise_level * (rng.standard_normal(n_pts) + 1j * rng.standard_normal(n_pts))

        data[:, 0, d] = edit_on_signal * phase_mod + noise_on
        data[:, 1, d] = edit_off_signal * phase_mod + noise_off

    truth = {
        'gaba_conc': gaba_conc,
        'naa_conc': naa_conc,
        'cr_conc': cr_conc,
        'n_dyn': n_dyn,
        'freq_drift_hz': freq_drift_hz,
        'noise_level': noise_level,
        'dwell': dwell,
        'cf': cf,
    }
    return data, truth


def ppm_axis(n: int, dwell: float, cf: float) -> np.ndarray:
    freq = np.fft.fftshift(np.fft.fftfreq(n, dwell))
    return freq / (cf / 1e6) + 4.65


def gaba_peak_height(result: MegaPressResult) -> float:
    """Extract GABA peak height from difference spectrum."""
    diff_spec = np.fft.fftshift(np.fft.fft(result.diff))
    n = len(result.diff)
    ppm = ppm_axis(n, result.dwell_time, 123.25e6)
    gaba_mask = (ppm > 2.8) & (ppm < 3.2)
    return float(np.max(np.abs(diff_spec[gaba_mask])))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFPCPreservesSubtraction:
    """Applying the same correction to both ON/OFF should not introduce
    artifacts in the difference spectrum."""

    def test_fpc_preserves_subtraction(self):
        """With paired alignment, Cr should still cancel in the difference."""
        data, truth = make_mega_pair_data(
            gaba_conc=0.0, cr_conc=10.0, noise_level=0.0001,
            freq_drift_hz=3.0,
        )
        result = process_mega_press(
            data, truth['dwell'], truth['cf'],
            align=True, reject=False, paired_alignment=True,
        )

        diff_spec = np.fft.fftshift(np.fft.fft(result.diff))
        off_spec = np.fft.fftshift(np.fft.fft(result.edit_off))
        ppm = ppm_axis(len(result.diff), truth['dwell'], truth['cf'])
        cr_mask = (ppm > 2.9) & (ppm < 3.15)

        cr_diff = np.max(np.abs(diff_spec[cr_mask]))
        cr_off = np.max(np.abs(off_spec[cr_mask]))

        # Cr should cancel well in the difference — subtraction artifact < 5%
        assert cr_diff < 0.05 * cr_off, (
            f"Cr subtraction artifact {cr_diff/cr_off:.2%} exceeds 5%"
        )


class TestFPCVsIndependent:
    """FPC should give equal or better GABA peak height than independent."""

    def test_fpc_vs_independent(self):
        data, truth = make_mega_pair_data(
            gaba_conc=2.0, noise_level=0.001, freq_drift_hz=5.0,
        )

        result_ind = process_mega_press(
            data.copy(), truth['dwell'], truth['cf'],
            align=True, reject=False, paired_alignment=False,
        )
        result_fpc = process_mega_press(
            data.copy(), truth['dwell'], truth['cf'],
            align=True, reject=False, paired_alignment=True,
        )

        gaba_ind = gaba_peak_height(result_ind)
        gaba_fpc = gaba_peak_height(result_fpc)

        # FPC should preserve or improve GABA peak (within 10% tolerance)
        assert gaba_fpc >= gaba_ind * 0.90, (
            f"FPC GABA {gaba_fpc:.4f} worse than independent {gaba_ind:.4f}"
        )


class TestFPCWithDrift:
    """With known frequency drift, FPC should track it while preserving
    subtraction quality."""

    def test_fpc_with_drift(self):
        data, truth = make_mega_pair_data(
            gaba_conc=1.0, noise_level=0.001, freq_drift_hz=8.0,
        )

        result = process_mega_press(
            data, truth['dwell'], truth['cf'],
            align=True, reject=False, paired_alignment=True,
        )

        # Frequency shifts should be non-zero (drift was applied)
        n_dyn = truth['n_dyn']
        off_shifts = result.freq_shifts[:n_dyn]
        assert np.std(off_shifts) > 0.5, (
            f"Freq shifts std {np.std(off_shifts):.2f} too small for 8 Hz drift"
        )

        # GABA should still be detectable
        height = gaba_peak_height(result)
        assert height > 0, "GABA not detected after FPC with drift"


class TestFPCOutputShape:
    """Output shapes must match input dimensions."""

    def test_fpc_output_shape(self):
        n_pts = 2048
        n_dyn = 16
        data, truth = make_mega_pair_data(n_pts=n_pts, n_dyn=n_dyn)

        result = process_mega_press(
            data, truth['dwell'], truth['cf'],
            align=True, reject=False, paired_alignment=True,
        )

        assert result.diff.shape == (n_pts,)
        assert result.edit_on.shape == (n_pts,)
        assert result.edit_off.shape == (n_pts,)
        assert result.sum_spec.shape == (n_pts,)
        assert result.freq_shifts.shape == (2 * n_dyn,)
        assert result.phase_shifts.shape == (2 * n_dyn,)

    def test_align_edit_pairs_shape(self):
        """align_edit_pairs returns correctly shaped arrays."""
        n_pts = 1024
        n_dyn = 8
        dwell = 2.5e-4
        cf = 123.25e6

        edit_on = np.random.randn(n_pts, n_dyn) + 1j * np.random.randn(n_pts, n_dyn)
        edit_off = np.random.randn(n_pts, n_dyn) + 1j * np.random.randn(n_pts, n_dyn)

        on_out, off_out, freqs, phases = align_edit_pairs(
            edit_on, edit_off, dwell, centre_freq=cf,
        )

        assert on_out.shape == (n_pts, n_dyn)
        assert off_out.shape == (n_pts, n_dyn)
        assert freqs.shape == (n_dyn,)
        assert phases.shape == (n_dyn,)
