"""TDD tests for end-to-end MEGA-PRESS quantification pipeline.

Tests the quantify_mega_press() function that chains:
1. process_mega_press() — preprocessing
2. zero_order_phase_correction() — phase the difference spectrum
3. fit_gaba_gaussian() — Gaussian fit to GABA peak
4. water_referenced_quantification() — absolute concentration (if water ref)

All tests should be RED until mrs_quantify.py is implemented.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: synthetic MEGA-PRESS data generators
# ---------------------------------------------------------------------------

def _ppm_axis(n: int, dwell: float = 2.5e-4, cf: float = 123.25e6) -> np.ndarray:
    """Frequency axis in ppm (water at 4.65 ppm)."""
    freq = np.fft.fftshift(np.fft.fftfreq(n, dwell))
    return freq / (cf / 1e6) + 4.65


def _make_fid(ppm: float, amplitude: float, lw: float, phase_deg: float = 0.0,
              n_pts: int = 2048, dwell: float = 2.5e-4,
              cf: float = 123.25e6) -> np.ndarray:
    """Create a phased Lorentzian FID at a given ppm position."""
    freq_hz = (ppm - 4.65) * (cf / 1e6)
    t = np.arange(n_pts) * dwell
    phase_rad = np.deg2rad(phase_deg)
    return amplitude * np.exp(1j * phase_rad) * np.exp(
        2j * np.pi * freq_hz * t) * np.exp(-np.pi * lw * t)


def _make_mega_data(gaba_amp: float = 0.5, naa_amp: float = 5.0,
                    n_dyn: int = 16, n_pts: int = 2048,
                    dwell: float = 2.5e-4, cf: float = 123.25e6,
                    phase_deg: float = 0.0,
                    noise_level: float = 0.01,
                    seed: int = 42) -> np.ndarray:
    """Synthesise MEGA-PRESS data: shape (n_pts, 2, n_dyn).

    Edit-ON (idx 0): GABA at 3.01 ppm + NAA at 2.01 ppm + Cr at 3.03 ppm
    Edit-OFF (idx 1): NAA at 2.01 ppm + Cr at 3.03 ppm  (no GABA editing)

    The difference (ON - OFF) isolates the GABA peak at 3.01 ppm.
    """
    rng = np.random.default_rng(seed)

    # Metabolite FIDs
    gaba_fid = _make_fid(3.01, gaba_amp, 8.0, phase_deg=phase_deg,
                         n_pts=n_pts, dwell=dwell, cf=cf)
    naa_fid = _make_fid(2.01, naa_amp, 3.0, phase_deg=phase_deg,
                        n_pts=n_pts, dwell=dwell, cf=cf)
    cr_fid = _make_fid(3.03, naa_amp * 0.5, 4.0, phase_deg=phase_deg,
                       n_pts=n_pts, dwell=dwell, cf=cf)

    data = np.zeros((n_pts, 2, n_dyn), dtype=np.complex128)
    for d in range(n_dyn):
        noise_on = noise_level * (rng.standard_normal(n_pts) +
                                  1j * rng.standard_normal(n_pts))
        noise_off = noise_level * (rng.standard_normal(n_pts) +
                                   1j * rng.standard_normal(n_pts))
        data[:, 0, d] = gaba_fid + naa_fid + cr_fid + noise_on   # edit-ON
        data[:, 1, d] = naa_fid + cr_fid + noise_off              # edit-OFF

    return data


def _make_water_fid(amplitude: float = 1000.0, n_pts: int = 2048,
                    dwell: float = 2.5e-4, cf: float = 123.25e6) -> np.ndarray:
    """Unsuppressed water FID at 4.65 ppm."""
    return _make_fid(4.65, amplitude, 2.0, n_pts=n_pts, dwell=dwell, cf=cf)


# ---------------------------------------------------------------------------
# Test 1: Synthetic GABA quantification — linearity
# ---------------------------------------------------------------------------

class TestQuantifySyntheticGABA:
    """Recovered GABA concentration should scale linearly with input amplitude."""

    def test_quantify_synthetic_gaba(self):
        """GABA area in the quantification result should scale linearly
        with the input GABA amplitude."""
        from neurojax.analysis.mrs_quantify import quantify_mega_press

        dwell = 2.5e-4
        cf = 123.25e6
        areas = []

        for amp in [0.25, 0.5, 1.0, 2.0]:
            data = _make_mega_data(gaba_amp=amp, dwell=dwell, cf=cf,
                                   noise_level=0.001, n_dyn=32)
            result = quantify_mega_press(data, dwell_time=dwell,
                                         centre_freq=cf)
            areas.append(result['gaba_area'])

        # Check approximate linearity: ratios should be close to 2x
        for i in range(len(areas) - 1):
            ratio = areas[i + 1] / areas[i]
            assert 1.5 < ratio < 2.8, (
                f"Area ratio between amp levels should be ~2x, "
                f"got {ratio:.2f} for levels {i} -> {i+1}"
            )


# ---------------------------------------------------------------------------
# Test 2: quantify returns all expected metrics
# ---------------------------------------------------------------------------

class TestQuantifyReturnsAllMetrics:
    """quantify_mega_press should return a dict with all key metrics."""

    def test_quantify_returns_all_metrics(self):
        from neurojax.analysis.mrs_quantify import quantify_mega_press

        dwell = 2.5e-4
        cf = 123.25e6
        data = _make_mega_data(dwell=dwell, cf=cf)
        water = _make_water_fid(dwell=dwell, cf=cf)

        result = quantify_mega_press(
            data, dwell_time=dwell, centre_freq=cf,
            water_ref=water,
            tissue_fracs={'gm': 0.6, 'wm': 0.4, 'csf': 0.0},
            te=0.068, tr=2.0,
        )

        required_keys = [
            'gaba_conc_mM', 'gaba_naa_ratio', 'gaba_area',
            'naa_area', 'snr', 'crlb_percent',
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
            assert np.isfinite(result[key]), f"Key {key} is not finite: {result[key]}"


# ---------------------------------------------------------------------------
# Test 3: with water reference => absolute concentration
# ---------------------------------------------------------------------------

class TestQuantifyWithWaterRef:
    """When water reference is provided, return absolute concentration."""

    def test_quantify_with_water_ref(self):
        from neurojax.analysis.mrs_quantify import quantify_mega_press

        dwell = 2.5e-4
        cf = 123.25e6
        data = _make_mega_data(dwell=dwell, cf=cf)
        water = _make_water_fid(dwell=dwell, cf=cf)

        result = quantify_mega_press(
            data, dwell_time=dwell, centre_freq=cf,
            water_ref=water,
            tissue_fracs={'gm': 0.6, 'wm': 0.4, 'csf': 0.0},
            te=0.068, tr=2.0,
        )

        # Absolute concentration should be present and positive
        assert 'gaba_conc_mM' in result
        assert result['gaba_conc_mM'] > 0, (
            f"GABA concentration should be positive, got {result['gaba_conc_mM']}"
        )


# ---------------------------------------------------------------------------
# Test 4: without water reference => ratio-based only
# ---------------------------------------------------------------------------

class TestQuantifyWithoutWaterRef:
    """Without water ref, return ratio-based metrics only."""

    def test_quantify_without_water_ref(self):
        from neurojax.analysis.mrs_quantify import quantify_mega_press

        dwell = 2.5e-4
        cf = 123.25e6
        data = _make_mega_data(dwell=dwell, cf=cf)

        result = quantify_mega_press(data, dwell_time=dwell, centre_freq=cf)

        # Should have ratio-based metrics
        assert 'gaba_naa_ratio' in result
        assert result['gaba_naa_ratio'] > 0

        # gaba_conc_mM should be None when no water ref
        assert result.get('gaba_conc_mM') is None, (
            "gaba_conc_mM should be None without water reference"
        )


# ---------------------------------------------------------------------------
# Test 5: tissue fraction correction changes concentration
# ---------------------------------------------------------------------------

class TestQuantifyWithTissueFracs:
    """Tissue fraction correction should change the reported concentration."""

    def test_quantify_with_tissue_fracs(self):
        from neurojax.analysis.mrs_quantify import quantify_mega_press

        dwell = 2.5e-4
        cf = 123.25e6
        data = _make_mega_data(dwell=dwell, cf=cf, noise_level=0.001)
        water = _make_water_fid(dwell=dwell, cf=cf)

        # Pure tissue (no CSF)
        result_no_csf = quantify_mega_press(
            data, dwell_time=dwell, centre_freq=cf,
            water_ref=water,
            tissue_fracs={'gm': 0.6, 'wm': 0.4, 'csf': 0.0},
            te=0.068, tr=2.0,
        )

        # With CSF (partial volume)
        result_csf = quantify_mega_press(
            data, dwell_time=dwell, centre_freq=cf,
            water_ref=water,
            tissue_fracs={'gm': 0.4, 'wm': 0.2, 'csf': 0.4},
            te=0.068, tr=2.0,
        )

        # CSF correction should increase apparent concentration
        # (metabolite is concentrated in tissue, not CSF)
        assert result_csf['gaba_conc_mM'] != result_no_csf['gaba_conc_mM'], (
            "Tissue fraction correction should change concentration"
        )
        assert result_csf['gaba_conc_mM'] > result_no_csf['gaba_conc_mM'], (
            "More CSF should increase tissue-corrected concentration"
        )


# ---------------------------------------------------------------------------
# Test 6: phase correction is applied before fitting
# ---------------------------------------------------------------------------

class TestQuantifyPhaseCorrected:
    """Phase correction should be applied before GABA fitting."""

    def test_quantify_phase_corrected(self):
        from neurojax.analysis.mrs_quantify import quantify_mega_press

        dwell = 2.5e-4
        cf = 123.25e6

        # Create data with a significant phase error
        data_phased = _make_mega_data(dwell=dwell, cf=cf, phase_deg=45.0,
                                      noise_level=0.001, n_dyn=32)

        result = quantify_mega_press(data_phased, dwell_time=dwell,
                                      centre_freq=cf)

        # Phase-corrected fit should still find the GABA peak near 3.0 ppm
        assert abs(result['gaba_centre_ppm'] - 3.01) < 0.15, (
            f"GABA peak should be near 3.0 ppm even with phase error, "
            f"got {result['gaba_centre_ppm']:.2f}"
        )
        # Area should be positive (not corrupted by phase)
        assert result['gaba_area'] > 0, (
            "GABA area should be positive after phase correction"
        )
