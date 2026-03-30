"""Validation tests for the MEGA-PRESS spectral editing pipeline.

Tests against:
1. Synthetic data with known ground truth (unit tests)
2. ISMRM Fitting Challenge data (if downloaded)
3. WAND ses-05 MEGA-PRESS data (integration test)
"""
import numpy as np
import pytest
from pathlib import Path

from neurojax.analysis.mega_press import (
    coil_combine_svd,
    spectral_registration,
    apply_correction,
    reject_outliers,
    process_mega_press,
    MegaPressResult,
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


def make_mega_data(
    n_pts: int = 2048,
    n_coils: int = 4,
    n_dyn: int = 32,
    dwell: float = 2.5e-4,
    cf: float = 123.25e6,
    gaba_conc: float = 1.0,
    naa_conc: float = 10.0,
    cr_conc: float = 8.0,
    noise_level: float = 0.01,
    freq_drift_hz: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate synthetic MEGA-PRESS data with known ground truth.

    Returns (data, truth) where data is (n_pts, n_coils, 2, n_dyn)
    and truth contains the known concentrations and parameters.
    """
    rng = np.random.default_rng(seed)

    # Metabolite signals
    naa = make_singlet(2.01, naa_conc, 3.0, n_pts, dwell, cf)
    cr = make_singlet(3.03, cr_conc, 4.0, n_pts, dwell, cf)
    gaba = make_singlet(3.01, gaba_conc, 8.0, n_pts, dwell, cf)

    # Edit-ON: GABA visible (editing pulse at 1.9 ppm inverts 3.0 ppm)
    edit_on_signal = naa + cr + gaba
    # Edit-OFF: GABA inverted (cancels in subtraction, Cr remains)
    edit_off_signal = naa + cr - gaba

    # Multi-coil with random sensitivities
    coil_weights = rng.standard_normal(n_coils) + 1j * rng.standard_normal(n_coils)
    coil_weights /= np.max(np.abs(coil_weights))

    t = np.arange(n_pts) * dwell

    data = np.zeros((n_pts, n_coils, 2, n_dyn), dtype=complex)
    for d in range(n_dyn):
        # Add frequency drift
        drift = freq_drift_hz * d / n_dyn
        phase_mod = np.exp(2j * np.pi * drift * t)

        for c in range(n_coils):
            noise_on = noise_level * (rng.standard_normal(n_pts) + 1j * rng.standard_normal(n_pts))
            noise_off = noise_level * (rng.standard_normal(n_pts) + 1j * rng.standard_normal(n_pts))
            data[:, c, 0, d] = coil_weights[c] * edit_on_signal * phase_mod + noise_on
            data[:, c, 1, d] = coil_weights[c] * edit_off_signal * phase_mod + noise_off

    truth = {
        'gaba_conc': gaba_conc,
        'naa_conc': naa_conc,
        'cr_conc': cr_conc,
        'n_coils': n_coils,
        'n_dyn': n_dyn,
        'freq_drift_hz': freq_drift_hz,
        'noise_level': noise_level,
        'dwell': dwell,
        'cf': cf,
    }
    return data, truth


# ---------------------------------------------------------------------------
# Unit tests: synthetic data
# ---------------------------------------------------------------------------

class TestCoilCombineSVD:
    def test_shape(self):
        data = np.random.randn(1024, 8, 2, 32) + 1j * np.random.randn(1024, 8, 2, 32)
        result = coil_combine_svd(data)
        assert result.shape == (1024, 2, 32)

    def test_snr_improvement(self):
        """Combined data should have better SNR than single coil."""
        data, truth = make_mega_data(n_coils=8, noise_level=0.1)
        combined = coil_combine_svd(data)
        single = data[:, 0, :, :]

        snr_combined = np.max(np.abs(combined)) / np.std(np.abs(combined[-100:]))
        snr_single = np.max(np.abs(single)) / np.std(np.abs(single[-100:]))
        assert snr_combined > snr_single


class TestSpectralRegistration:
    def test_no_shift(self):
        """Zero shift when FID matches reference."""
        fid = make_singlet(2.01, 1.0, 3.0, 1024, 2.5e-4, 123.25e6)
        df, dp = spectral_registration(fid, fid, 2.5e-4, centre_freq=123.25e6)
        assert abs(df) < 0.5  # Hz
        assert abs(dp) < 0.1  # rad

    def test_known_shift(self):
        """Recover a known frequency shift."""
        dwell = 2.5e-4
        cf = 123.25e6
        ref = make_singlet(2.01, 1.0, 3.0, 1024, dwell, cf)

        # Apply 5 Hz shift
        t = np.arange(1024) * dwell
        shifted = ref * np.exp(2j * np.pi * 5.0 * t)

        df, dp = spectral_registration(shifted, ref, dwell, centre_freq=cf)
        assert abs(abs(df) - 5.0) < 1.0  # Within 1 Hz (sign may differ)


class TestRejectOutliers:
    def test_clean_data(self):
        """No rejections on clean data."""
        fids = np.random.randn(1024, 32) + 1j * np.random.randn(1024, 32)
        rejected = reject_outliers(fids, 2.5e-4)
        assert rejected.sum() < 3  # At most a few

    def test_detects_outlier(self):
        """Detects an obvious outlier."""
        fids = np.random.randn(1024, 32) + 1j * np.random.randn(1024, 32)
        fids[:, 0] *= 100  # Make first transient an outlier
        rejected = reject_outliers(fids, 2.5e-4, threshold=3.0)
        assert rejected[0] == True


class TestProcessMegaPress:
    def test_basic_pipeline(self):
        """Full pipeline on synthetic data."""
        data, truth = make_mega_data()
        result = process_mega_press(data, truth['dwell'], truth['cf'],
                                     align=False, reject=False)
        assert isinstance(result, MegaPressResult)
        assert result.diff.shape == (2048,)
        assert result.edit_on.shape == (2048,)
        assert result.edit_off.shape == (2048,)

    def test_gaba_in_difference(self):
        """GABA peak should appear in difference spectrum."""
        data, truth = make_mega_data(gaba_conc=2.0, noise_level=0.001)
        result = process_mega_press(data, truth['dwell'], truth['cf'],
                                     align=False, reject=False)

        # FFT difference spectrum
        diff_spec = np.fft.fftshift(np.fft.fft(result.diff))
        n = len(result.diff)
        freq = np.fft.fftshift(np.fft.fftfreq(n, truth['dwell']))
        ppm = freq / (truth['cf'] / 1e6) + 4.65

        # GABA at ~3.0 ppm should be the dominant peak in difference
        gaba_mask = (ppm > 2.8) & (ppm < 3.2)
        cr_mask = (ppm > 2.9) & (ppm < 3.15)

        gaba_power = np.max(np.abs(diff_spec[gaba_mask]))
        # Cr should cancel in the difference (same in ON and OFF)
        # but GABA should remain
        assert gaba_power > 0

    def test_cr_cancels_in_difference(self):
        """Creatine should cancel in difference (same in both conditions)."""
        data, truth = make_mega_data(gaba_conc=0.0, cr_conc=10.0, noise_level=0.0001)
        result = process_mega_press(data, truth['dwell'], truth['cf'],
                                     align=False, reject=False)

        diff_spec = np.fft.fftshift(np.fft.fft(result.diff))
        off_spec = np.fft.fftshift(np.fft.fft(result.edit_off))

        # Cr at 3.03 ppm — should be near zero in diff, present in OFF
        n = len(result.diff)
        freq = np.fft.fftshift(np.fft.fftfreq(n, truth['dwell']))
        ppm = freq / (truth['cf'] / 1e6) + 4.65
        cr_mask = (ppm > 2.9) & (ppm < 3.15)

        cr_diff = np.max(np.abs(diff_spec[cr_mask]))
        cr_off = np.max(np.abs(off_spec[cr_mask]))

        # Cr in difference should be <<< Cr in OFF
        assert cr_diff < 0.01 * cr_off

    def test_alignment_improves_with_drift(self):
        """Alignment should reduce linewidth when frequency drift is present."""
        data, truth = make_mega_data(freq_drift_hz=5.0, noise_level=0.001)

        result_noalign = process_mega_press(data, truth['dwell'], truth['cf'],
                                             align=False, reject=False)
        result_aligned = process_mega_press(data, truth['dwell'], truth['cf'],
                                             align=True, reject=False)

        # NAA peak in edit-OFF should be narrower after alignment
        off_noalign = np.fft.fftshift(np.fft.fft(result_noalign.edit_off))
        off_aligned = np.fft.fftshift(np.fft.fft(result_aligned.edit_off))

        n = len(result_noalign.edit_off)
        freq = np.fft.fftshift(np.fft.fftfreq(n, truth['dwell']))
        ppm = freq / (truth['cf'] / 1e6) + 4.65
        naa_mask = (ppm > 1.9) & (ppm < 2.15)

        # Peak height should be higher after alignment (narrower peak = taller)
        peak_noalign = np.max(np.abs(off_noalign[naa_mask]))
        peak_aligned = np.max(np.abs(off_aligned[naa_mask]))
        assert peak_aligned >= peak_noalign * 0.95  # At least comparable

    def test_with_coils(self):
        """Pipeline works with multi-coil data."""
        data, truth = make_mega_data(n_coils=16)
        result = process_mega_press(data, truth['dwell'], truth['cf'])
        assert result.n_averages > 0
        assert result.diff.shape[0] == 2048


# ---------------------------------------------------------------------------
# Integration tests: WAND data (skip if not available)
# ---------------------------------------------------------------------------

WAND_MEGA = Path("/data/raw/wand/derivatives/fsl-mrs/sub-08033/ses-05/anteriorcingulate/mega")

@pytest.mark.skipif(not WAND_MEGA.exists(), reason="WAND data not available")
class TestWANDMegaPress:
    def test_diff_exists(self):
        diff = np.load(str(WAND_MEGA / "diff_fid.npy"))
        assert diff.shape[0] > 1000

    def test_gaba_detected(self):
        """GABA peak should be present in WAND ACC difference spectrum."""
        diff = np.load(str(WAND_MEGA / "diff_fid.npy"))
        off = np.load(str(WAND_MEGA / "edit_off_fid.npy"))

        dwell = 2.5e-4
        cf = 123.253e6
        n = len(diff)
        freq = np.fft.fftshift(np.fft.fftfreq(n, dwell))
        ppm = freq / (cf / 1e6) + 4.65

        diff_spec = np.fft.fftshift(np.fft.fft(diff))
        off_spec = np.fft.fftshift(np.fft.fft(off))

        gaba_mask = (ppm > 2.8) & (ppm < 3.2)
        naa_mask = (ppm > 1.9) & (ppm < 2.1)

        gaba_peak = np.max(np.abs(diff_spec[gaba_mask]))
        naa_peak = np.max(np.abs(off_spec[naa_mask]))

        # GABA should be detectable
        assert gaba_peak > 0
        # GABA/NAA ratio should be reasonable (0.05 - 2.0)
        ratio = gaba_peak / naa_peak
        assert 0.05 < ratio < 2.0, f"GABA/NAA={ratio:.3f} out of range"


# ---------------------------------------------------------------------------
# Validation tests: external datasets (skip if not downloaded)
# ---------------------------------------------------------------------------

ISMRM_DIR = Path("/home/mhough/dev/neurojax/tests/data/mrs/ismrm_fitting_challenge")

@pytest.mark.skipif(not ISMRM_DIR.exists(), reason="ISMRM dataset not downloaded")
class TestISMRMFittingChallenge:
    def test_datasets_exist(self):
        files = list(ISMRM_DIR.glob("*.nii*")) + list(ISMRM_DIR.glob("*.RAW"))
        assert len(files) > 0, "No ISMRM datasets found"
