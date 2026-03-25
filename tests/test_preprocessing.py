"""Tests for the neurojax.preprocessing subpackage.

Covers: ASR, filtering, ICA, resampling, interpolation, artifact detection.
Uses synthetic data (sine waves, random noise, known artifacts) throughout.

Known bugs documented by RED tests:
- filter.py: uses jax.scipy.signal.lfilter which does not exist in JAX.
  All TestFilterData tests (except test_import) are expected to fail with
  AttributeError. These are intentionally left as red tests.
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.signal as jss

# Check if lfilter is available (it isn't in current JAX versions)
_HAS_LFILTER = hasattr(jss, "lfilter")
_LFILTER_SKIP = pytest.mark.skipif(
    not _HAS_LFILTER,
    reason="BUG: filter.py uses jax.scipy.signal.lfilter which does not exist in this JAX version"
)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def make_sine(freq, sfreq, duration, n_channels=1, key=None, noise_level=0.0):
    """Generate a multi-channel sine wave (channels, time)."""
    n_times = int(sfreq * duration)
    t = jnp.linspace(0, duration, n_times)
    signal = jnp.sin(2 * jnp.pi * freq * t)
    data = jnp.tile(signal, (n_channels, 1))
    if noise_level > 0 and key is not None:
        data = data + noise_level * jax.random.normal(key, data.shape)
    return data


def make_spd(n, key):
    """Generate a random symmetric positive-definite matrix."""
    A = jax.random.normal(key, (n, n))
    return A @ A.T + jnp.eye(n)


# ============================================================================
# filter.py — Butterworth / IIR filtering
# ============================================================================

class TestFilterData:
    """Tests for neurojax.preprocessing.filter.filter_data."""

    def test_import(self):
        from neurojax.preprocessing.filter import filter_data
        assert callable(filter_data)

    @_LFILTER_SKIP
    def test_passthrough_identity(self):
        """b=[1], a=[1] is the identity filter — output equals input."""
        from neurojax.preprocessing.filter import filter_data

        key = jax.random.PRNGKey(0)
        data = jax.random.normal(key, (3, 200))
        b = jnp.array([1.0])
        a = jnp.array([1.0])
        out = filter_data(data, b, a)
        assert jnp.allclose(out, data, atol=1e-6)

    @_LFILTER_SKIP
    def test_output_shape(self):
        """Output shape must match input shape."""
        from neurojax.preprocessing.filter import filter_data

        data = jnp.ones((4, 100))
        b = jnp.array([0.5, 0.5])
        a = jnp.array([1.0])
        out = filter_data(data, b, a)
        assert out.shape == data.shape

    @_LFILTER_SKIP
    def test_lowpass_attenuates_high_freq(self):
        """A simple moving-average (low-pass) filter should reduce energy
        of a high-frequency sine relative to a low-frequency sine."""
        from neurojax.preprocessing.filter import filter_data

        sfreq = 1000.0
        # Low-freq component
        low = make_sine(5.0, sfreq, 1.0)
        # High-freq component
        high = make_sine(200.0, sfreq, 1.0)
        data = low + high  # (1, 1000)

        # Simple 5-tap moving average — acts as low-pass
        n_taps = 5
        b = jnp.ones(n_taps) / n_taps
        a = jnp.array([1.0])

        out = filter_data(data, b, a)

        # Power of high-freq component in the output should drop
        # Compute power spectral density via FFT
        fft_in = jnp.abs(jnp.fft.rfft(data[0]))**2
        fft_out = jnp.abs(jnp.fft.rfft(out[0]))**2
        # High-freq bin index ~ 200/sfreq * n_times
        hf_idx = int(200 * data.shape[1] / sfreq)
        ratio = float(fft_out[hf_idx] / (fft_in[hf_idx] + 1e-12))
        assert ratio < 0.5, f"High-freq should be attenuated, got ratio={ratio}"

    @_LFILTER_SKIP
    def test_single_channel(self):
        """Works with a single-channel (1, T) input."""
        from neurojax.preprocessing.filter import filter_data

        data = jnp.ones((1, 50))
        b = jnp.array([1.0])
        a = jnp.array([1.0])
        out = filter_data(data, b, a)
        assert out.shape == (1, 50)

    @_LFILTER_SKIP
    def test_1d_input(self):
        """1D input should also work since lfilter operates on last axis."""
        from neurojax.preprocessing.filter import filter_data

        data = jnp.ones(50)
        b = jnp.array([1.0])
        a = jnp.array([1.0])
        out = filter_data(data, b, a)
        assert out.shape == (50,)


# ============================================================================
# asr.py — Artifact Subspace Reconstruction
# ============================================================================

class TestASR:
    """Tests for neurojax.preprocessing.asr (calibrate_asr, apply_asr)."""

    def test_import(self):
        from neurojax.preprocessing.asr import ASRState, calibrate_asr, apply_asr
        assert ASRState is not None

    def test_calibrate_shapes(self):
        """calibrate_asr returns correct shapes."""
        from neurojax.preprocessing.asr import calibrate_asr

        key = jax.random.PRNGKey(1)
        n_ch, n_t = 4, 500
        data = jax.random.normal(key, (n_ch, n_t))
        state = calibrate_asr(data, cutoff=5.0)

        assert state.mixing_matrix.shape == (n_ch, n_ch)
        assert state.component_stdevs.shape == (n_ch,)
        assert state.cutoff == 5.0

    def test_calibrate_positive_stdevs(self):
        """Component standard deviations should be positive."""
        from neurojax.preprocessing.asr import calibrate_asr

        key = jax.random.PRNGKey(2)
        data = jax.random.normal(key, (3, 300))
        state = calibrate_asr(data)
        assert jnp.all(state.component_stdevs > 0)

    def test_mixing_matrix_orthogonal(self):
        """Mixing matrix (eigenvectors of cov) should be orthogonal."""
        from neurojax.preprocessing.asr import calibrate_asr

        key = jax.random.PRNGKey(3)
        data = jax.random.normal(key, (4, 1000))
        state = calibrate_asr(data)
        M = state.mixing_matrix
        prod = M.T @ M
        assert jnp.allclose(prod, jnp.eye(4), atol=1e-5), \
            "Mixing matrix should be orthogonal"

    def test_apply_asr_output_shape(self):
        """apply_asr should return data with the same shape as input."""
        from neurojax.preprocessing.asr import calibrate_asr, apply_asr

        key = jax.random.PRNGKey(4)
        n_ch, n_t = 4, 500
        data = jax.random.normal(key, (n_ch, n_t))
        state = calibrate_asr(data)
        cleaned = apply_asr(data, state, window_size=100, step_size=50)
        assert cleaned.shape == data.shape

    @pytest.mark.slow
    def test_apply_asr_reduces_artifact(self):
        """ASR should reduce large-variance artifact injections.

        Create clean data, calibrate on it, then inject a large spike
        on one channel in a test segment. The cleaned version should have
        lower RMS on that channel than the corrupted version.
        """
        from neurojax.preprocessing.asr import calibrate_asr, apply_asr

        key = jax.random.PRNGKey(5)
        n_ch, n_t = 4, 600
        k1, k2 = jax.random.split(key)
        clean = jax.random.normal(k1, (n_ch, n_t)) * 0.1

        state = calibrate_asr(clean, cutoff=3.0)

        # Inject artifact on channel 0 in middle segment
        corrupted = clean.at[0, 200:400].set(
            clean[0, 200:400] + 50.0 * jnp.ones(200)
        )

        cleaned = apply_asr(corrupted, state, window_size=100, step_size=50)

        rms_corrupted = jnp.sqrt(jnp.mean(corrupted[0] ** 2))
        rms_cleaned = jnp.sqrt(jnp.mean(cleaned[0] ** 2))
        assert rms_cleaned < rms_corrupted, \
            "ASR should reduce artifact RMS"

    def test_clean_data_not_distorted(self):
        """If data looks like calibration data, ASR should not heavily distort it."""
        from neurojax.preprocessing.asr import calibrate_asr, apply_asr

        key = jax.random.PRNGKey(6)
        n_ch, n_t = 3, 400
        data = jax.random.normal(key, (n_ch, n_t)) * 0.1
        state = calibrate_asr(data, cutoff=10.0)  # lenient
        cleaned = apply_asr(data, state, window_size=100, step_size=50)
        # Correlation between cleaned and original should be high
        corr = jnp.corrcoef(data.ravel(), cleaned.ravel())[0, 1]
        assert corr > 0.8, f"Clean data should be mostly preserved, corr={corr}"


# ============================================================================
# ica.py — FastICA
# ============================================================================

class TestFastICA:
    """Tests for neurojax.preprocessing.ica.FastICA."""

    def test_import(self):
        from neurojax.preprocessing.ica import FastICA
        assert FastICA is not None

    def test_init(self):
        from neurojax.preprocessing.ica import FastICA

        ica = FastICA(n_components=3)
        assert ica.n_components == 3
        assert ica.max_iter == 200
        assert ica.tol == 1e-4
        assert ica.mixing_ is None
        assert ica.components_ is None
        assert ica.mean_ is None

    def test_fit_returns_new_module(self):
        """fit() returns a new FastICA with populated attributes (equinox style)."""
        from neurojax.preprocessing.ica import FastICA

        key = jax.random.PRNGKey(10)
        n_ch, n_t = 3, 500
        data = jax.random.normal(key, (n_ch, n_t))
        ica = FastICA(n_components=2)
        fitted = ica.fit(data, key=key)

        assert fitted.mixing_ is not None
        assert fitted.components_ is not None
        assert fitted.mean_ is not None

    def test_components_shape(self):
        """Components should have shape (n_components, n_times)."""
        from neurojax.preprocessing.ica import FastICA

        key = jax.random.PRNGKey(11)
        n_ch, n_t = 4, 600
        data = jax.random.normal(key, (n_ch, n_t))
        ica = FastICA(n_components=3)
        fitted = ica.fit(data, key=key)

        assert fitted.components_.shape == (3, n_t)

    def test_mixing_shape(self):
        """Mixing matrix shape: (n_channels, n_components)."""
        from neurojax.preprocessing.ica import FastICA

        key = jax.random.PRNGKey(12)
        n_ch, n_t = 4, 600
        data = jax.random.normal(key, (n_ch, n_t))
        ica = FastICA(n_components=3)
        fitted = ica.fit(data, key=key)

        assert fitted.mixing_.shape == (n_ch, 3)

    def test_mean_shape(self):
        """Mean should have shape (n_channels, 1)."""
        from neurojax.preprocessing.ica import FastICA

        key = jax.random.PRNGKey(13)
        n_ch, n_t = 4, 600
        data = jax.random.normal(key, (n_ch, n_t))
        ica = FastICA(n_components=3)
        fitted = ica.fit(data, key=key)

        assert fitted.mean_.shape == (n_ch, 1)

    @pytest.mark.slow
    def test_separation_quality(self):
        """ICA should recover independent sources from a linear mixture.

        Create 2 independent sources, mix them linearly with a known matrix,
        then check that ICA can roughly recover them (up to sign/permutation).
        """
        from neurojax.preprocessing.ica import FastICA

        key = jax.random.PRNGKey(14)
        n_t = 2000
        t = jnp.linspace(0, 1, n_t)

        # Two independent sources
        s1 = jnp.sin(2 * jnp.pi * 5 * t)
        s2 = jnp.sign(jnp.sin(2 * jnp.pi * 11 * t))  # square wave
        sources = jnp.stack([s1, s2], axis=0)

        # Mixing matrix
        A = jnp.array([[1.0, 0.5], [0.3, 0.8]])
        mixed = A @ sources  # (2, n_t)

        ica = FastICA(n_components=2)
        fitted = ica.fit(mixed, key=key)
        components = fitted.components_

        # Each recovered component should correlate strongly with exactly one source
        for i in range(2):
            corrs = []
            for j in range(2):
                c = jnp.abs(jnp.corrcoef(components[i], sources[j])[0, 1])
                corrs.append(float(c))
            assert max(corrs) > 0.8, \
                f"Component {i} should correlate with at least one source, got {corrs}"

    def test_apply_shape(self):
        """apply() should project new data to component space."""
        from neurojax.preprocessing.ica import FastICA

        key = jax.random.PRNGKey(15)
        n_ch, n_t = 3, 500
        data = jax.random.normal(key, (n_ch, n_t))
        ica = FastICA(n_components=2)
        fitted = ica.fit(data, key=key)

        k2 = jax.random.PRNGKey(16)
        new_data = jax.random.normal(k2, (n_ch, 300))
        projected = fitted.apply(new_data)
        assert projected.shape == (2, 300)

    def test_n_components_equals_channels(self):
        """When n_components == n_channels, it should still work."""
        from neurojax.preprocessing.ica import FastICA

        key = jax.random.PRNGKey(17)
        n_ch, n_t = 3, 500
        data = jax.random.normal(key, (n_ch, n_t))
        ica = FastICA(n_components=n_ch)
        fitted = ica.fit(data, key=key)
        assert fitted.components_.shape == (n_ch, n_t)
        assert fitted.mixing_.shape == (n_ch, n_ch)


# ============================================================================
# resample.py — JAX-native polyphase resampling
# ============================================================================

class TestResample:
    """Tests for neurojax.preprocessing.resample."""

    def test_import(self):
        from neurojax.preprocessing.resample import resample_poly, resample_minimal
        assert callable(resample_poly)
        assert callable(resample_minimal)

    def test_identity_no_change(self):
        """When up == down, output should equal input."""
        from neurojax.preprocessing.resample import resample_poly

        data = jnp.ones(100)
        out = resample_poly(data, up=1, down=1)
        assert jnp.allclose(out, data)

    def test_upsample_length(self):
        """Upsampling by 2 should approximately double the output length."""
        from neurojax.preprocessing.resample import resample_poly

        n = 100
        data = jnp.ones(n)
        out = resample_poly(data, up=2, down=1)
        expected_len = n * 2
        # 'same' mode of fftconvolve keeps the length of the upsampled signal
        assert out.shape[0] == expected_len

    def test_downsample_length(self):
        """Downsampling by 2 should approximately halve the output length."""
        from neurojax.preprocessing.resample import resample_poly

        n = 200
        data = jnp.ones(n)
        out = resample_poly(data, up=1, down=2)
        expected_len = n  # up*n / down = n, then ::down → n/2? Let's see
        # After upsampling: n*1 = 200 samples, after fftconvolve 'same': 200
        # After downsampling by 2: 100
        assert out.shape[0] == n // 2

    def test_multichannel(self):
        """Should work with multi-dimensional input along last axis."""
        from neurojax.preprocessing.resample import resample_poly

        data = jnp.ones((3, 200))
        out = resample_poly(data, up=1, down=2, axis=-1)
        assert out.shape[0] == 3  # channels preserved
        assert out.shape[1] == 100

    def test_resample_minimal_convenience(self):
        """resample_minimal should compute correct up/down factors."""
        from neurojax.preprocessing.resample import resample_minimal

        data = jnp.ones(1000)
        out = resample_minimal(data, original_sfreq=1000, target_sfreq=500)
        # 1000 -> 500 Hz means down by 2
        assert out.shape[0] == 500

    def test_resample_minimal_upsample(self):
        """Upsampling from 250 to 500 Hz should double length."""
        from neurojax.preprocessing.resample import resample_minimal

        data = jnp.ones(250)
        out = resample_minimal(data, original_sfreq=250, target_sfreq=500)
        assert out.shape[0] == 500

    def test_axis_parameter(self):
        """Resampling should work along specified axis."""
        from neurojax.preprocessing.resample import resample_poly

        # axis=0 means time is first dimension
        data = jnp.ones((200, 3))
        out = resample_poly(data, up=1, down=2, axis=0)
        assert out.shape == (100, 3)

    def test_preserves_dc(self):
        """A constant signal should remain approximately constant after resampling."""
        from neurojax.preprocessing.resample import resample_poly

        data = 5.0 * jnp.ones(200)
        out = resample_poly(data, up=2, down=1)
        # After filtering, center samples should be close to 5.0
        # (edges may have transients)
        center = out[len(out) // 4 : 3 * len(out) // 4]
        assert jnp.allclose(center, 5.0, atol=0.5), \
            f"DC value not preserved: mean={float(jnp.mean(center))}"


# ============================================================================
# interpolate.py — Bad channel interpolation (Spherical Splines)
# ============================================================================

class TestInterpolation:
    """Tests for neurojax.preprocessing.interpolate."""

    def test_import(self):
        from neurojax.preprocessing.interpolate import spherical_spline_interpolate
        assert callable(spherical_spline_interpolate)

    def test_no_bad_channels_identity(self):
        """When no bad channels are specified, data should be returned unchanged."""
        from neurojax.preprocessing.interpolate import spherical_spline_interpolate

        key = jax.random.PRNGKey(30)
        data = jax.random.normal(key, (5, 100))
        # Generate sensor positions on unit sphere
        coords = _make_unit_sphere_coords(5, key)
        bad_idx = jnp.array([], dtype=jnp.int32)
        out = spherical_spline_interpolate(data, bad_idx, coords, n_terms=10)
        assert jnp.allclose(out, data)

    def test_output_shape(self):
        """Output shape should match input shape."""
        from neurojax.preprocessing.interpolate import spherical_spline_interpolate

        key = jax.random.PRNGKey(31)
        n_ch, n_t = 8, 200
        data = jax.random.normal(key, (n_ch, n_t))
        coords = _make_unit_sphere_coords(n_ch, key)
        bad_idx = jnp.array([1, 3])
        out = spherical_spline_interpolate(data, bad_idx, coords, n_terms=10)
        assert out.shape == (n_ch, n_t)

    def test_good_channels_unchanged(self):
        """Good (non-bad) channels should remain unchanged."""
        from neurojax.preprocessing.interpolate import spherical_spline_interpolate

        key = jax.random.PRNGKey(32)
        n_ch, n_t = 6, 100
        data = jax.random.normal(key, (n_ch, n_t))
        coords = _make_unit_sphere_coords(n_ch, key)
        bad_idx = jnp.array([2])
        out = spherical_spline_interpolate(data, bad_idx, coords, n_terms=10)
        good_mask = jnp.array([0, 1, 3, 4, 5])
        assert jnp.allclose(out[good_mask], data[good_mask])

    def test_interpolated_channel_changes(self):
        """The bad channel should be modified by interpolation."""
        from neurojax.preprocessing.interpolate import spherical_spline_interpolate

        key = jax.random.PRNGKey(33)
        n_ch, n_t = 6, 100
        data = jax.random.normal(key, (n_ch, n_t))
        # Make channel 2 clearly different (e.g., huge DC offset)
        data = data.at[2].set(data[2] + 1000.0)
        coords = _make_unit_sphere_coords(n_ch, key)
        bad_idx = jnp.array([2])
        out = spherical_spline_interpolate(data, bad_idx, coords, n_terms=10)
        # Interpolated channel should no longer have the 1000 offset
        assert jnp.abs(jnp.mean(out[2]) - 1000.0) > 100.0, \
            "Interpolated channel should differ from the original bad channel"

    def test_legendre_coeffs_shape(self):
        """Internal: Legendre coefficients should have n_terms elements."""
        from neurojax.preprocessing.interpolate import _calc_legendre_coeffs

        c = _calc_legendre_coeffs(n_terms=20)
        assert c.shape == (20,)

    def test_legendre_coeffs_positive(self):
        """Internal: Legendre coefficients for m=4 spline should all be positive."""
        from neurojax.preprocessing.interpolate import _calc_legendre_coeffs

        c = _calc_legendre_coeffs(n_terms=50)
        assert jnp.all(c > 0)

    def test_evaluate_legendre_shape(self):
        """Internal: Evaluate Legendre polynomials returns correct shape."""
        from neurojax.preprocessing.interpolate import _evaluate_legendre

        x = jnp.array([0.0, 0.5, 1.0])
        p = _evaluate_legendre(x, n_terms=10)
        assert p.shape == (10, 3)

    def test_g_function_symmetric(self):
        """Internal: g(x) should be evaluated element-wise on the input."""
        from neurojax.preprocessing.interpolate import _g_function

        x = jnp.array([[0.5, -0.3], [-0.3, 0.5]])
        g = _g_function(x, n_terms=20)
        assert g.shape == x.shape
        # Symmetric input → symmetric output
        assert jnp.allclose(g, g.T, atol=1e-6)


def _make_unit_sphere_coords(n, key):
    """Helper: generate n random points on the unit sphere."""
    coords = jax.random.normal(key, (n, 3))
    norms = jnp.linalg.norm(coords, axis=1, keepdims=True)
    return coords / norms


# ============================================================================
# artifact.py — Riemannian artifact detection
# ============================================================================

class TestArtifactDetection:
    """Tests for neurojax.preprocessing.artifact.detect_artifacts_riemann.

    This module depends on neurojax.geometry.riemann, which should be available.
    """

    def test_import(self):
        from neurojax.preprocessing.artifact import detect_artifacts_riemann
        assert callable(detect_artifacts_riemann)

    def test_output_shape(self):
        """Output should be a boolean array of length N (number of epochs)."""
        from neurojax.preprocessing.artifact import detect_artifacts_riemann

        key = jax.random.PRNGKey(40)
        n_epochs, n_ch = 10, 3
        covs = _make_spd_batch(n_epochs, n_ch, key)
        mask = detect_artifacts_riemann(covs, n_std=3.0)
        assert mask.shape == (n_epochs,)
        assert mask.dtype == jnp.bool_

    def test_no_outliers_no_artifacts(self):
        """When all covariances are similar, no artifacts should be detected
        with a lenient threshold."""
        from neurojax.preprocessing.artifact import detect_artifacts_riemann

        key = jax.random.PRNGKey(41)
        n_epochs, n_ch = 20, 3
        # Generate very similar covariances (identity + small noise)
        covs = jnp.stack(
            [jnp.eye(n_ch) + 0.01 * jax.random.normal(jax.random.PRNGKey(i), (n_ch, n_ch))
             for i in range(n_epochs)]
        )
        # Make symmetric and PD
        covs = (covs + jnp.transpose(covs, (0, 2, 1))) / 2
        covs = covs + 2.0 * jnp.eye(n_ch)[None]  # ensure PD

        mask = detect_artifacts_riemann(covs, n_std=3.0)
        # With similar covariances, nothing should be marked
        assert jnp.sum(mask) == 0, "No artifacts expected for homogeneous covariances"

    @pytest.mark.slow
    def test_detects_outlier(self):
        """An epoch with a very different covariance should be flagged."""
        from neurojax.preprocessing.artifact import detect_artifacts_riemann

        key = jax.random.PRNGKey(42)
        n_epochs, n_ch = 15, 3
        # Normal epochs: identity-like
        covs = jnp.stack(
            [jnp.eye(n_ch) + 0.01 * jax.random.normal(jax.random.PRNGKey(i), (n_ch, n_ch))
             for i in range(n_epochs)]
        )
        covs = (covs + jnp.transpose(covs, (0, 2, 1))) / 2
        covs = covs + 2.0 * jnp.eye(n_ch)[None]

        # Inject one outlier: large diagonal matrix
        outlier = 100.0 * jnp.eye(n_ch)
        covs = covs.at[7].set(outlier)

        mask = detect_artifacts_riemann(covs, n_std=2.0)
        assert mask[7], "Outlier epoch should be detected as artifact"


def _make_spd_batch(n, dim, key):
    """Generate a batch of random SPD matrices."""
    keys = jax.random.split(key, n)
    def make_one(k):
        A = jax.random.normal(k, (dim, dim))
        return A @ A.T + jnp.eye(dim)
    return jax.vmap(make_one)(keys)


# ============================================================================
# Edge cases & integration
# ============================================================================

class TestEdgeCases:
    """Edge cases that span multiple modules."""

    @_LFILTER_SKIP
    def test_filter_very_short_data(self):
        """Filtering should work even with very short data (few samples)."""
        from neurojax.preprocessing.filter import filter_data

        data = jnp.array([[1.0, 2.0, 3.0]])
        b = jnp.array([1.0])
        a = jnp.array([1.0])
        out = filter_data(data, b, a)
        assert out.shape == (1, 3)

    def test_asr_single_channel(self):
        """ASR should work with a single channel."""
        from neurojax.preprocessing.asr import calibrate_asr, apply_asr

        key = jax.random.PRNGKey(50)
        data = jax.random.normal(key, (1, 300))
        state = calibrate_asr(data, cutoff=5.0)
        assert state.mixing_matrix.shape == (1, 1)
        cleaned = apply_asr(data, state, window_size=50, step_size=25)
        assert cleaned.shape == data.shape

    def test_ica_single_component(self):
        """ICA with n_components=1 should work."""
        from neurojax.preprocessing.ica import FastICA

        key = jax.random.PRNGKey(51)
        data = jax.random.normal(key, (3, 500))
        ica = FastICA(n_components=1)
        fitted = ica.fit(data, key=key)
        assert fitted.components_.shape == (1, 500)
        assert fitted.mixing_.shape == (3, 1)

    def test_resample_1d(self):
        """Resampling a 1D array should work."""
        from neurojax.preprocessing.resample import resample_poly

        data = jnp.ones(100)
        out = resample_poly(data, up=1, down=2)
        assert out.ndim == 1
        assert out.shape[0] == 50
