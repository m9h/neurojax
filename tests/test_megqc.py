"""Tests for neurojax.preprocessing.megqc — MEG quality control metrics.

Red-green TDD: tests written first.
"""
import numpy as np
import pytest
import mne


@pytest.fixture
def synthetic_raw():
    """Create a synthetic MEG-like Raw object for testing."""
    sfreq = 250.0
    n_channels = 10
    n_times = int(60 * sfreq)  # 60 seconds

    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, n_times)) * 1e-13

    # Make channel 0 noisy (100x amplitude — clear outlier even with 10 channels)
    data[0] *= 100
    # Make channel 9 flat (near-zero)
    data[9] *= 1e-6

    ch_names = [f"MEG{i:04d}" for i in range(n_channels)]
    ch_types = ["mag"] * n_channels
    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture
def synthetic_raw_with_ecg():
    """Synthetic Raw with injected periodic ECG-like artifact."""
    sfreq = 250.0
    n_channels = 20
    n_times = int(120 * sfreq)  # 2 minutes

    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, n_times)) * 1e-13

    # Add ECG-like periodic signal (1.2 Hz = 72 bpm)
    t = np.arange(n_times) / sfreq
    ecg_signal = 5e-13 * np.sin(2 * np.pi * 1.2 * t)
    # Add to first 5 channels with varying amplitude
    for i in range(5):
        data[i] += ecg_signal * (1.0 - i * 0.15)

    ch_names = [f"MEG{i:04d}" for i in range(n_channels)]
    ch_types = ["mag"] * n_channels
    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


class TestMEGQCResult:
    def test_result_fields(self, synthetic_raw):
        from neurojax.preprocessing.megqc import run_megqc
        result = run_megqc(synthetic_raw, subject="test-sub", task="rest")
        assert result.subject == "test-sub"
        assert result.task == "rest"
        assert result.n_channels == 10
        assert result.sfreq == 250.0
        assert result.duration_s > 0

    def test_gqi_between_0_and_1(self, synthetic_raw):
        from neurojax.preprocessing.megqc import run_megqc
        result = run_megqc(synthetic_raw, subject="test-sub")
        assert 0 <= result.gqi <= 1

    def test_to_json_roundtrip(self, synthetic_raw, tmp_path):
        from neurojax.preprocessing.megqc import run_megqc
        import json
        result = run_megqc(synthetic_raw, subject="test-sub")
        path = tmp_path / "qc.json"
        result.to_json(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["subject"] == "test-sub"
        assert loaded["n_channels"] == 10
        assert isinstance(loaded["gqi"], float)

    def test_to_tsv_row(self, synthetic_raw):
        from neurojax.preprocessing.megqc import run_megqc
        result = run_megqc(synthetic_raw, subject="test-sub")
        row = result.to_tsv_row()
        assert "noisy_channels" not in row  # should be excluded
        assert "subject" in row


class TestSTDMetrics:
    def test_detects_noisy_channel(self, synthetic_raw):
        from neurojax.preprocessing.megqc import compute_std_metrics
        # Use lower multiplier for small channel count (10 channels)
        info, noisy, flat = compute_std_metrics(synthetic_raw, std_multiplier=2.0)
        assert "MEG0000" in noisy, f"Expected MEG0000 in noisy, got {noisy}"

    def test_detects_flat_channel(self, synthetic_raw):
        from neurojax.preprocessing.megqc import compute_std_metrics
        info, noisy, flat = compute_std_metrics(synthetic_raw, std_multiplier=2.0)
        assert "MEG0009" in flat, f"Expected MEG0009 in flat, got {flat}"

    def test_thresholds_positive(self, synthetic_raw):
        from neurojax.preprocessing.megqc import compute_std_metrics
        info, _, _ = compute_std_metrics(synthetic_raw)
        assert info["std_mean"] > 0
        assert info["std_threshold_noisy"] > info["std_mean"]

    def test_normal_data_no_outliers(self):
        """All-equal-variance channels should have zero noisy/flat."""
        from neurojax.preprocessing.megqc import compute_std_metrics
        rng = np.random.default_rng(0)
        data = rng.standard_normal((20, 5000)) * 1e-13
        info = mne.create_info([f"MEG{i:04d}" for i in range(20)], 250.0, ["mag"] * 20)
        raw = mne.io.RawArray(data, info, verbose=False)
        _, noisy, flat = compute_std_metrics(raw)
        assert len(noisy) == 0
        assert len(flat) == 0


class TestPSDMetrics:
    def test_detects_line_noise(self):
        """Inject strong 50 Hz component, should be detected."""
        from neurojax.preprocessing.megqc import compute_psd_metrics
        sfreq = 1000.0
        n_times = int(30 * sfreq)
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5, n_times)) * 1e-13
        t = np.arange(n_times) / sfreq
        data += 1e-12 * np.sin(2 * np.pi * 50 * t)[None, :]  # strong 50 Hz
        info = mne.create_info([f"MEG{i:04d}" for i in range(5)], sfreq, ["mag"] * 5)
        raw = mne.io.RawArray(data, info, verbose=False)
        result = compute_psd_metrics(raw)
        assert result["has_line_noise"]
        assert result["line_noise_freq"] == 50.0

    def test_clean_data_no_line_noise(self, synthetic_raw):
        from neurojax.preprocessing.megqc import compute_psd_metrics
        result = compute_psd_metrics(synthetic_raw)
        assert not result["has_line_noise"]


class TestMuscleMetrics:
    def test_returns_fraction(self, synthetic_raw):
        from neurojax.preprocessing.megqc import compute_muscle_metrics
        result = compute_muscle_metrics(synthetic_raw)
        assert 0 <= result["muscle_fraction"] <= 1


class TestGQI:
    def test_perfect_data_high_gqi(self):
        """Clean data with no artifacts should get high GQI."""
        from neurojax.preprocessing.megqc import MEGQCResult, compute_gqi
        result = MEGQCResult(
            subject="test", task="rest",
            n_noisy=0, n_flat=0,
            has_line_noise=False,
            ecg_channel_found=True, heartbeat_rate_bpm=72,
            muscle_fraction=0.0,
        )
        gqi = compute_gqi(result)
        assert gqi > 0.9

    def test_bad_data_low_gqi(self):
        """Data with many artifacts should get low GQI."""
        from neurojax.preprocessing.megqc import MEGQCResult, compute_gqi
        result = MEGQCResult(
            subject="test", task="rest",
            n_noisy=25, n_flat=5,
            has_line_noise=True,
            ecg_channel_found=True, heartbeat_rate_bpm=150,
            muscle_fraction=0.5,
        )
        gqi = compute_gqi(result)
        assert gqi < 0.5
