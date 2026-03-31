"""Tests for specparam/FOOOF per-state spectral decomposition.

Red-green TDD: validates aperiodic + periodic extraction from HMM/DyNeMo state PSDs.
"""
import numpy as np
import pytest


def _make_synthetic_psd(freqs, aperiodic_offset=1.0, aperiodic_exponent=1.5,
                        peak_freq=10.0, peak_power=0.5, peak_bw=2.0):
    """Generate a synthetic PSD with known aperiodic + periodic components."""
    psd = 10 ** (aperiodic_offset - aperiodic_exponent * np.log10(freqs + 0.1))
    psd += peak_power * np.exp(-0.5 * ((freqs - peak_freq) / peak_bw) ** 2)
    return psd


@pytest.fixture
def synthetic_state_psd():
    """8 states × 60 freqs × 68 parcels — mimics HMM state spectra."""
    freqs = np.linspace(1, 45, 60)
    n_states, n_parcels = 8, 68
    rng = np.random.default_rng(42)
    psd = np.zeros((n_states, len(freqs), n_parcels))
    for s in range(n_states):
        for p in range(n_parcels):
            exp = 1.0 + rng.uniform(0, 1.0)
            peak_f = 8 + rng.uniform(0, 4)
            peak_pow = 0.2 + rng.uniform(0, 0.5)
            psd[s, :, p] = _make_synthetic_psd(
                freqs, aperiodic_exponent=exp,
                peak_freq=peak_f, peak_power=peak_pow)
    return psd, freqs


@pytest.fixture
def simple_psd():
    """Single PSD with known 10 Hz peak."""
    freqs = np.linspace(1, 45, 100)
    psd = _make_synthetic_psd(freqs, aperiodic_exponent=1.5, peak_freq=10.0, peak_power=1.0)
    return psd, freqs


class TestFitSinglePSD:
    def test_detects_peak_frequency(self, simple_psd):
        from neurojax.analysis.specparam_analysis import fit_single_psd
        psd, freqs = simple_psd
        result = fit_single_psd(psd, freqs)
        assert abs(result["peak_freq"] - 10.0) < 2.0, \
            f"Expected peak ~10 Hz, got {result['peak_freq']}"

    def test_aperiodic_exponent_positive(self, simple_psd):
        from neurojax.analysis.specparam_analysis import fit_single_psd
        psd, freqs = simple_psd
        result = fit_single_psd(psd, freqs)
        assert result["aperiodic_exponent"] > 0

    def test_returns_expected_keys(self, simple_psd):
        from neurojax.analysis.specparam_analysis import fit_single_psd
        psd, freqs = simple_psd
        result = fit_single_psd(psd, freqs)
        for key in ["aperiodic_offset", "aperiodic_exponent",
                     "peak_freq", "peak_power", "peak_bw", "n_peaks", "r_squared"]:
            assert key in result, f"Missing key: {key}"

    def test_r_squared_reasonable(self, simple_psd):
        from neurojax.analysis.specparam_analysis import fit_single_psd
        psd, freqs = simple_psd
        result = fit_single_psd(psd, freqs)
        assert result["r_squared"] > 0.8, f"Bad fit: R²={result['r_squared']}"


class TestFitStateSpectra:
    def test_output_shape(self, synthetic_state_psd):
        from neurojax.analysis.specparam_analysis import fit_state_spectra
        psd, freqs = synthetic_state_psd
        result = fit_state_spectra(psd, freqs)
        assert result["aperiodic_exponent"].shape == (8, 68)
        assert result["peak_freq"].shape == (8, 68)

    def test_all_exponents_positive(self, synthetic_state_psd):
        from neurojax.analysis.specparam_analysis import fit_state_spectra
        psd, freqs = synthetic_state_psd
        result = fit_state_spectra(psd, freqs)
        assert np.all(result["aperiodic_exponent"] > 0)

    def test_peak_freqs_in_alpha_range(self, synthetic_state_psd):
        from neurojax.analysis.specparam_analysis import fit_state_spectra
        psd, freqs = synthetic_state_psd
        result = fit_state_spectra(psd, freqs)
        valid = result["peak_freq"][result["peak_freq"] > 0]
        assert np.all(valid > 4) and np.all(valid < 20), \
            f"Peaks outside expected range: {valid.min()}-{valid.max()}"


class TestExtractFeatures:
    def test_feature_matrix_shape(self, synthetic_state_psd):
        from neurojax.analysis.specparam_analysis import extract_fooof_features
        psd, freqs = synthetic_state_psd
        features = extract_fooof_features(psd, freqs)
        # Should have per-state features: 8 states × n_features
        assert features.shape[0] == 8
        assert features.shape[1] > 0

    def test_features_finite(self, synthetic_state_psd):
        from neurojax.analysis.specparam_analysis import extract_fooof_features
        psd, freqs = synthetic_state_psd
        features = extract_fooof_features(psd, freqs)
        assert np.all(np.isfinite(features))
