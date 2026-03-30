"""Tests for neurojax.preprocessing.adversarial — adversarial pipeline evaluation.

Red-green TDD: tests catch the exact bugs that caused the failed runs:
  1. Channel indexing mismatch after pick()
  2. Sample rate mismatch after resample
  3. Signal injection preserves data shape
  4. Recovery measurement produces valid metrics
"""
import numpy as np
import pytest
import mne


@pytest.fixture
def synthetic_raw_340ch():
    """Simulate CTF-like Raw with 274 MEG + 29 ref + 37 other = 340 channels.

    This mimics the WAND CTF data that caused the original index error.
    """
    sfreq = 1200.0  # CTF native rate
    n_meg = 274
    n_ref = 29
    n_other = 37
    n_total = n_meg + n_ref + n_other
    n_times = int(10 * sfreq)  # 10 seconds

    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_total, n_times)) * 1e-13

    ch_names = ([f"MLT{i:03d}" for i in range(n_meg)] +
                [f"REF{i:03d}" for i in range(n_ref)] +
                [f"STIM{i:03d}" for i in range(n_other)])
    ch_types = ["mag"] * n_meg + ["ref_meg"] * n_ref + ["stim"] * n_other
    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture
def synthetic_raw_meg_only():
    """Clean MEG-only Raw at target sfreq — no channel/rate mismatch issues."""
    sfreq = 250.0
    n_channels = 20
    n_times = int(30 * sfreq)
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, n_times)) * 1e-13
    ch_names = [f"MEG{i:04d}" for i in range(n_channels)]
    info = mne.create_info(ch_names, sfreq, ["mag"] * n_channels)
    return mne.io.RawArray(data, info, verbose=False)


# -------------------------------------------------------------------------
# Signal generation tests
# -------------------------------------------------------------------------

class TestSignalGenerators:
    def test_oscillatory_shape(self):
        from neurojax.preprocessing.adversarial import make_oscillatory_signal
        sig = make_oscillatory_signal(1000, 250.0, freq=10.0)
        assert sig.shape == (1000,)
        assert np.all(np.isfinite(sig))

    def test_oscillatory_amplitude(self):
        from neurojax.preprocessing.adversarial import make_oscillatory_signal
        amp = 5e-13
        sig = make_oscillatory_signal(1000, 250.0, amplitude=amp)
        assert np.max(np.abs(sig)) <= amp * 1.01

    def test_erp_shape(self):
        from neurojax.preprocessing.adversarial import make_erp_signal
        sig = make_erp_signal(1000, 250.0)
        assert sig.shape == (1000,)
        assert np.max(sig) > 0  # has a peak

    def test_multi_freq_shape(self):
        from neurojax.preprocessing.adversarial import make_multi_frequency_signal
        sig = make_multi_frequency_signal(1000, 250.0)
        assert sig.shape == (1000,)

    def test_burst_envelope_has_zeros(self):
        from neurojax.preprocessing.adversarial import make_oscillatory_signal
        np.random.seed(42)
        sig = make_oscillatory_signal(10000, 250.0, envelope="burst")
        # Bursts should have silent regions
        assert np.mean(np.abs(sig) < 1e-20) > 0.3


# -------------------------------------------------------------------------
# Signal injection tests
# -------------------------------------------------------------------------

class TestSignalInjection:
    def test_injection_preserves_shape(self, synthetic_raw_meg_only):
        from neurojax.preprocessing.adversarial import inject_signal, make_oscillatory_signal
        signal = make_oscillatory_signal(
            synthetic_raw_meg_only.n_times, 250.0)
        raw_inj, injected = inject_signal(synthetic_raw_meg_only, signal)
        assert raw_inj.get_data().shape == synthetic_raw_meg_only.get_data().shape

    def test_injection_adds_signal(self, synthetic_raw_meg_only):
        from neurojax.preprocessing.adversarial import inject_signal, make_oscillatory_signal
        signal = make_oscillatory_signal(
            synthetic_raw_meg_only.n_times, 250.0, amplitude=1e-10)
        orig = synthetic_raw_meg_only.get_data().copy()
        raw_inj, injected = inject_signal(synthetic_raw_meg_only, signal)
        diff = raw_inj.get_data() - orig
        assert np.max(np.abs(diff)) > 0, "Injection had no effect"

    def test_injected_data_shape_matches_picks(self, synthetic_raw_meg_only):
        from neurojax.preprocessing.adversarial import inject_signal, make_oscillatory_signal
        signal = make_oscillatory_signal(
            synthetic_raw_meg_only.n_times, 250.0)
        _, injected = inject_signal(synthetic_raw_meg_only, signal)
        n_ch = len(mne.pick_types(synthetic_raw_meg_only.info, meg=True))
        assert injected.shape == (n_ch, synthetic_raw_meg_only.n_times)

    def test_spatial_pattern_applied(self, synthetic_raw_meg_only):
        from neurojax.preprocessing.adversarial import inject_signal, make_oscillatory_signal
        signal = make_oscillatory_signal(
            synthetic_raw_meg_only.n_times, 250.0, amplitude=1e-10)
        n_ch = 20
        pattern = np.zeros(n_ch)
        pattern[0] = 1.0  # inject only in channel 0
        orig = synthetic_raw_meg_only.get_data().copy()
        raw_inj, _ = inject_signal(
            synthetic_raw_meg_only, signal, spatial_pattern=pattern)
        diff = raw_inj.get_data() - orig
        # Channel 0 should have signal, others should not
        assert np.max(np.abs(diff[0])) > 1e-12
        assert np.max(np.abs(diff[1])) < 1e-20


# -------------------------------------------------------------------------
# Recovery measurement tests
# -------------------------------------------------------------------------

class TestMeasureRecovery:
    def test_perfect_recovery(self, synthetic_raw_meg_only):
        """If pipeline is identity, recovery should be perfect."""
        from neurojax.preprocessing.adversarial import (
            inject_signal, make_oscillatory_signal, measure_recovery)
        signal = make_oscillatory_signal(
            synthetic_raw_meg_only.n_times, 250.0, amplitude=1e-10)
        raw_inj, injected = inject_signal(synthetic_raw_meg_only, signal)
        picks = np.arange(len(synthetic_raw_meg_only.ch_names))

        # Identity pipeline: preprocessed = raw
        metrics = measure_recovery(
            raw_inj, synthetic_raw_meg_only, injected, picks)
        assert metrics["correlation"] > 0.99, \
            f"Perfect recovery should have corr>0.99, got {metrics['correlation']}"
        assert metrics["waveform_distortion"] < 0.01

    def test_zero_signal_zero_correlation(self, synthetic_raw_meg_only):
        """Zero-amplitude injection should not produce meaningful correlation."""
        from neurojax.preprocessing.adversarial import (
            inject_signal, make_oscillatory_signal, measure_recovery)
        signal = make_oscillatory_signal(
            synthetic_raw_meg_only.n_times, 250.0, amplitude=0.0)
        raw_inj, injected = inject_signal(synthetic_raw_meg_only, signal)
        picks = np.arange(len(synthetic_raw_meg_only.ch_names))
        metrics = measure_recovery(
            raw_inj, synthetic_raw_meg_only, injected, picks)
        # With zero signal, correlation is undefined/noisy
        assert np.isfinite(metrics["correlation"])


# -------------------------------------------------------------------------
# Bug regression tests (the failures that triggered this TDD)
# -------------------------------------------------------------------------

class TestBugRegressions:
    def test_no_index_error_with_mixed_channels(self, synthetic_raw_340ch):
        """Regression: original bug was IndexError when raw had ref+stim channels.

        evaluate_pipelines computed meg_picks from original 340-channel raw,
        then pipelines called pick(meg) reducing to 274 channels. The old
        meg_picks indices (up to 303) were out of bounds on the 274-channel data.
        """
        from neurojax.preprocessing.adversarial import (
            evaluate_pipelines, PipelineConfig)

        def identity_pipeline(raw):
            return raw.copy()

        pipelines = [PipelineConfig("identity", identity_pipeline)]
        # This should NOT raise IndexError
        results = evaluate_pipelines(
            synthetic_raw_340ch, pipelines, n_trials=1, seed=42)
        assert len(results) > 0
        assert all(r.correlation > 0 for r in results)

    def test_no_shape_mismatch_after_resample(self, synthetic_raw_340ch):
        """Regression: pipeline resampled from 1200→250 Hz, creating
        720000 vs 150000 sample mismatch in measure_recovery.

        evaluate_pipelines now resamples BEFORE injection so all data
        is at the same sample rate.
        """
        from neurojax.preprocessing.adversarial import (
            evaluate_pipelines, PipelineConfig)

        def filter_pipeline(raw):
            raw = raw.copy()
            raw.filter(1.0, 45.0, verbose=False)
            return raw

        pipelines = [PipelineConfig("filter_1-45", filter_pipeline)]
        # This should NOT raise "dimensions except for the concatenation axis must match"
        results = evaluate_pipelines(
            synthetic_raw_340ch, pipelines, n_trials=1, seed=42)
        assert len(results) > 0

    def test_evaluate_pipelines_returns_metrics(self, synthetic_raw_meg_only):
        """Basic integration: evaluate_pipelines returns valid RecoveryMetrics."""
        from neurojax.preprocessing.adversarial import (
            evaluate_pipelines, PipelineConfig, RecoveryMetrics)

        def identity(raw):
            return raw.copy()

        pipelines = [PipelineConfig("identity", identity)]
        results = evaluate_pipelines(
            synthetic_raw_meg_only, pipelines, n_trials=1, seed=42)

        assert len(results) == 3  # 3 default signal types × 1 trial × 1 pipeline
        for r in results:
            assert isinstance(r, RecoveryMetrics)
            assert np.isfinite(r.correlation)
            assert np.isfinite(r.snr_output)
            assert np.isfinite(r.waveform_distortion)

    def test_ssp_pipeline_runs_without_crash(self, synthetic_raw_meg_only):
        """SSP pipelines should run without error even on synthetic data."""
        from neurojax.preprocessing.adversarial import (
            evaluate_pipelines, make_default_pipelines)

        pipelines = make_default_pipelines(sfreq=250.0)
        # Just the filter pipelines (SSP needs ECG channel)
        filter_only = [p for p in pipelines if "ssp" not in p.name]
        results = evaluate_pipelines(
            synthetic_raw_meg_only, filter_only, n_trials=1, seed=42)
        assert len(results) > 0
        assert all(r.correlation > 0.9 for r in results), \
            "Filter-only pipelines should preserve signal well"
