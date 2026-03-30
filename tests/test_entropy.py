"""Tests for neurojax.analysis.entropy — JAX-accelerated entropy measures.

Red-green TDD: tests written first, implementation must pass all.
Cross-validates against mne-features reference implementations.
"""
import numpy as np
import pytest

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def gaussian_data(rng):
    """303 channels × 2500 samples (10s at 250 Hz) — realistic MEG shape."""
    return rng.standard_normal((303, 2500)).astype(np.float32)


@pytest.fixture
def small_data(rng):
    """5 channels × 500 samples — fast for CI."""
    return rng.standard_normal((5, 500)).astype(np.float32)


@pytest.fixture
def deterministic_data():
    """Known signal for exact value checks."""
    return np.array([[1, -1, 1, -1, 0, 1, -1, 1]], dtype=np.float32)


@pytest.fixture
def constant_data():
    """Constant signal — edge case."""
    return np.ones((2, 100), dtype=np.float32)


@pytest.fixture
def sine_data():
    """Pure sine — low entropy expected."""
    t = np.linspace(0, 2 * np.pi * 10, 1000)
    return np.array([np.sin(t), np.sin(2 * t)], dtype=np.float32)


# -------------------------------------------------------------------------
# Output shape tests
# -------------------------------------------------------------------------

class TestOutputShapes:
    def test_sample_entropy_shape(self, small_data):
        from neurojax.analysis.entropy import sample_entropy
        result = sample_entropy(small_data)
        assert result.shape == (5,)

    def test_approx_entropy_shape(self, small_data):
        from neurojax.analysis.entropy import approx_entropy
        result = approx_entropy(small_data)
        assert result.shape == (5,)

    def test_svd_entropy_shape(self, small_data):
        from neurojax.analysis.entropy import svd_entropy
        result = svd_entropy(small_data)
        assert result.shape == (5,)

    def test_spectral_entropy_shape(self, small_data):
        from neurojax.analysis.entropy import spectral_entropy
        result = spectral_entropy(small_data)
        assert result.shape == (5,)

    def test_compute_all_shape(self, small_data):
        from neurojax.analysis.entropy import compute_all_entropies
        result = compute_all_entropies(small_data)
        assert set(result.keys()) == {
            "sample_entropy", "approx_entropy", "svd_entropy", "spectral_entropy"
        }
        for v in result.values():
            assert v.shape == (5,)


# -------------------------------------------------------------------------
# Finite value tests (no NaN/Inf)
# -------------------------------------------------------------------------

class TestFiniteValues:
    def test_sample_entropy_finite(self, small_data):
        from neurojax.analysis.entropy import sample_entropy
        result = sample_entropy(small_data)
        assert np.all(np.isfinite(result))

    def test_approx_entropy_finite(self, small_data):
        from neurojax.analysis.entropy import approx_entropy
        result = approx_entropy(small_data)
        assert np.all(np.isfinite(result))

    def test_svd_entropy_finite(self, small_data):
        from neurojax.analysis.entropy import svd_entropy
        result = svd_entropy(small_data)
        assert np.all(np.isfinite(result))

    def test_spectral_entropy_finite(self, small_data):
        from neurojax.analysis.entropy import spectral_entropy
        result = spectral_entropy(small_data)
        assert np.all(np.isfinite(result))


# -------------------------------------------------------------------------
# Value range tests
# -------------------------------------------------------------------------

class TestValueRanges:
    def test_sample_entropy_positive(self, small_data):
        """SampEn should be non-negative for random data."""
        from neurojax.analysis.entropy import sample_entropy
        result = sample_entropy(small_data)
        assert np.all(result >= 0)

    def test_approx_entropy_positive(self, small_data):
        """AppEn should be non-negative for random data."""
        from neurojax.analysis.entropy import approx_entropy
        result = approx_entropy(small_data)
        assert np.all(result >= 0)

    def test_svd_entropy_positive(self, small_data):
        """SVD entropy should be positive."""
        from neurojax.analysis.entropy import svd_entropy
        result = svd_entropy(small_data)
        assert np.all(result > 0)

    def test_spectral_entropy_positive(self, small_data):
        """Spectral entropy should be positive."""
        from neurojax.analysis.entropy import spectral_entropy
        result = spectral_entropy(small_data)
        assert np.all(result > 0)

    def test_sine_lower_entropy_than_noise(self, sine_data, small_data):
        """Sine wave should have lower spectral entropy than random noise."""
        from neurojax.analysis.entropy import spectral_entropy
        sine_ent = spectral_entropy(sine_data)
        noise_ent = spectral_entropy(small_data[:2])
        assert np.mean(sine_ent) < np.mean(noise_ent)

    def test_gaussian_sampen_around_2(self, gaussian_data):
        """SampEn of Gaussian noise with emb=2 should be ~2.0-2.3."""
        from neurojax.analysis.entropy import sample_entropy
        result = sample_entropy(gaussian_data[:10])  # 10 channels for speed
        mean_se = float(np.mean(result))
        assert 1.5 < mean_se < 3.0, f"Mean SampEn = {mean_se}, expected ~2.1"


# -------------------------------------------------------------------------
# Cross-validation against mne-features reference
# -------------------------------------------------------------------------

class TestCrossValidation:
    """Compare JAX results against mne-features KDTree implementation."""

    def test_samp_entropy_matches_mne_features(self, small_data):
        """JAX sample entropy should match mne-features within tolerance."""
        from neurojax.analysis.entropy import sample_entropy
        from mne_features.univariate import compute_samp_entropy

        jax_result = np.array(sample_entropy(small_data))
        mne_result = compute_samp_entropy(small_data.astype(np.float64))

        np.testing.assert_allclose(jax_result, mne_result, rtol=0.15,
                                   err_msg="SampEn JAX vs mne-features mismatch")

    def test_app_entropy_matches_mne_features(self, small_data):
        """JAX approx entropy should match mne-features within tolerance."""
        from neurojax.analysis.entropy import approx_entropy
        from mne_features.univariate import compute_app_entropy

        jax_result = np.array(approx_entropy(small_data))
        mne_result = compute_app_entropy(small_data.astype(np.float64))

        np.testing.assert_allclose(jax_result, mne_result, rtol=0.15,
                                   err_msg="AppEn JAX vs mne-features mismatch")

    def test_svd_entropy_matches_mne_features(self, small_data):
        """JAX SVD entropy should match mne-features within tolerance."""
        from neurojax.analysis.entropy import svd_entropy
        from mne_features.univariate import compute_svd_entropy

        jax_result = np.array(svd_entropy(small_data))
        mne_result = compute_svd_entropy(small_data.astype(np.float64))

        np.testing.assert_allclose(jax_result, mne_result, rtol=0.05,
                                   err_msg="SVD entropy JAX vs mne-features mismatch")

    def test_spect_entropy_matches_mne_features(self, small_data):
        """JAX spectral entropy should match mne-features within tolerance."""
        from neurojax.analysis.entropy import spectral_entropy
        from mne_features.univariate import compute_spect_entropy

        jax_result = np.array(spectral_entropy(small_data, fs=250.0))
        mne_result = compute_spect_entropy(250.0, small_data.astype(np.float64),
                                            psd_method='fft')

        # Spectral entropy depends on PSD method; FFT should be close
        np.testing.assert_allclose(jax_result, mne_result, rtol=0.20,
                                   err_msg="SpectEn JAX vs mne-features mismatch")


# -------------------------------------------------------------------------
# Deterministic known-value tests
# -------------------------------------------------------------------------

class TestKnownValues:
    def test_samp_entropy_known(self, deterministic_data):
        """Match the mne-features test: SampEn([1,-1,1,-1,0,1,-1,1]) = log(3)."""
        from neurojax.analysis.entropy import sample_entropy
        result = sample_entropy(deterministic_data)
        expected = np.log(3)
        np.testing.assert_allclose(float(result[0]), expected, rtol=0.1,
                                   err_msg=f"Expected ~{expected:.4f}, got {float(result[0]):.4f}")


# -------------------------------------------------------------------------
# High channel count (realistic MEG) — performance test
# -------------------------------------------------------------------------

class TestHighChannelCount:
    def test_303_channels_completes(self, gaussian_data):
        """303 channels should complete without OOM or timeout."""
        from neurojax.analysis.entropy import compute_all_entropies
        result = compute_all_entropies(gaussian_data)
        for v in result.values():
            assert v.shape == (303,)
            assert np.all(np.isfinite(v))
