"""Tests for adversarial evaluation at realistic SNR levels.

Red-green TDD: these tests define what the SNR-sweep evaluation must produce.
At low SNR, artifact removal pipelines should outperform filter-only.
At high SNR, all pipelines should preserve the signal.
"""
import numpy as np
import pytest
import mne


@pytest.fixture
def noisy_raw_with_ecg():
    """Synthetic Raw with realistic ECG artifact for pipeline differentiation.

    The ECG artifact is strong enough that SSP removal should improve
    signal recovery at low injection SNR.
    """
    sfreq = 250.0
    n_channels = 30
    n_times = int(60 * sfreq)  # 60 seconds

    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, n_times)) * 1e-13

    # Strong periodic ECG artifact (1.2 Hz = 72 bpm) on all channels
    t = np.arange(n_times) / sfreq
    for i in range(n_channels):
        # QRS-like pulse train: sharp peaks at heartbeat intervals
        heartbeat_times = np.arange(0, t[-1], 1 / 1.2)
        for hb in heartbeat_times:
            peak_idx = int(hb * sfreq)
            if peak_idx + 25 < n_times:
                qrs = np.zeros(25)
                qrs[10:15] = 1.0  # sharp peak
                data[i, peak_idx:peak_idx + 25] += 3e-13 * qrs * (1 - 0.02 * i)

    ch_names = [f"MEG{i:04d}" for i in range(n_channels)]
    info = mne.create_info(ch_names, sfreq, ["mag"] * n_channels)
    return mne.io.RawArray(data, info, verbose=False)


class TestSNRSweep:
    def test_sweep_returns_results_per_snr_level(self, noisy_raw_with_ecg):
        from neurojax.preprocessing.adversarial import (
            evaluate_pipelines_snr_sweep, PipelineConfig)

        def identity(raw):
            return raw.copy()

        pipelines = [PipelineConfig("identity", identity)]
        results = evaluate_pipelines_snr_sweep(
            noisy_raw_with_ecg, pipelines,
            snr_levels_db=[-10, 0, 10], n_trials=1, seed=42)

        assert len(results) > 0
        snrs_seen = set(r.snr_input for r in results)
        # Should have results at each SNR level
        assert len(snrs_seen) >= 3

    def test_higher_snr_better_recovery(self, noisy_raw_with_ecg):
        """Signal recovery should improve with higher injection SNR."""
        from neurojax.preprocessing.adversarial import (
            evaluate_pipelines_snr_sweep, PipelineConfig)

        def bandpass(raw):
            raw = raw.copy()
            raw.filter(1.0, 45.0, verbose=False)
            return raw

        pipelines = [PipelineConfig("bandpass", bandpass)]
        results = evaluate_pipelines_snr_sweep(
            noisy_raw_with_ecg, pipelines,
            snr_levels_db=[-20, -10, 0, 10, 20], n_trials=2, seed=42)

        # Group by SNR, compute mean correlation
        from collections import defaultdict
        by_snr = defaultdict(list)
        for r in results:
            by_snr[r.snr_input].append(r.correlation)
        mean_corrs = {snr: np.mean(corrs) for snr, corrs in by_snr.items()}

        snrs_sorted = sorted(mean_corrs.keys())
        # Correlation at highest SNR should be >= correlation at lowest
        assert mean_corrs[snrs_sorted[-1]] >= mean_corrs[snrs_sorted[0]], \
            f"Expected higher SNR → better recovery: {mean_corrs}"

    def test_sweep_output_has_amplitude_field(self, noisy_raw_with_ecg):
        from neurojax.preprocessing.adversarial import (
            evaluate_pipelines_snr_sweep, PipelineConfig)

        def identity(raw):
            return raw.copy()

        pipelines = [PipelineConfig("identity", identity)]
        results = evaluate_pipelines_snr_sweep(
            noisy_raw_with_ecg, pipelines,
            snr_levels_db=[0], n_trials=1, seed=42)

        for r in results:
            assert hasattr(r, "snr_input")
            assert hasattr(r, "correlation")
            assert np.isfinite(r.correlation)


class TestPipelineDifferentiation:
    def test_at_low_snr_pipelines_differ(self, noisy_raw_with_ecg):
        """Pipelines with different passbands produce different recovery for
        signals near the filter edge.

        A 2 Hz signal passes the 1-45 Hz filter but is attenuated by a
        4-30 Hz filter → the wide filter should recover it better.
        """
        from neurojax.preprocessing.adversarial import (
            evaluate_pipelines_snr_sweep, PipelineConfig)

        def bandpass_wide(raw):
            raw = raw.copy()
            raw.filter(1.0, 45.0, verbose=False)
            return raw

        def bandpass_narrow(raw):
            raw = raw.copy()
            raw.filter(4.0, 30.0, verbose=False)
            return raw

        pipelines = [
            PipelineConfig("wide_1-45", bandpass_wide),
            PipelineConfig("narrow_4-30", bandpass_narrow),
        ]
        # Use 2 Hz signal — inside wide passband, outside narrow passband
        results = evaluate_pipelines_snr_sweep(
            noisy_raw_with_ecg, pipelines,
            snr_levels_db=[0], signal_freqs=(2.0,), n_trials=3, seed=42)

        from collections import defaultdict
        by_pipeline = defaultdict(list)
        for r in results:
            by_pipeline[r.pipeline].append(r.correlation)
        mean_corrs = {p: np.mean(c) for p, c in by_pipeline.items()}

        # Wide should recover 2 Hz signal better than narrow
        assert mean_corrs["wide_1-45"] > mean_corrs["narrow_4-30"], \
            f"Wide filter should recover 2Hz better: {mean_corrs}"
