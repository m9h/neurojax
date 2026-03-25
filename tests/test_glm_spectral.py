"""Tests for neurojax.glm and neurojax.spectral modules."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------

class TestGLMImportability:
    def test_glm_importable(self):
        from neurojax.glm import GeneralLinearModel, run_permutation_test
        assert GeneralLinearModel is not None
        assert callable(run_permutation_test)


class TestSpectralImportability:
    def test_spectral_importable(self):
        from neurojax.spectral import PowerSpectrumModel, fit_spectrum
        assert PowerSpectrumModel is not None
        assert callable(fit_spectrum)


# ---------------------------------------------------------------------------
# GeneralLinearModel
# ---------------------------------------------------------------------------

class TestGLMConstruction:
    def test_create(self):
        from neurojax.glm import GeneralLinearModel

        X = jnp.ones((50, 2))
        Y = jnp.zeros((50, 3))
        glm = GeneralLinearModel(X, Y)
        assert glm.betas is None
        assert glm.residuals is None
        assert glm.design_matrix.shape == (50, 2)
        assert glm.data.shape == (50, 3)

    def test_is_equinox_module(self):
        import equinox as eqx
        from neurojax.glm import GeneralLinearModel

        X = jnp.ones((10, 1))
        Y = jnp.zeros((10, 1))
        glm = GeneralLinearModel(X, Y)
        assert isinstance(glm, eqx.Module)


class TestGLMFit:
    """Test GLM fitting with known parameters."""

    @pytest.fixture()
    def known_regression(self):
        """Create a design matrix and data with known betas.

        Y = X @ beta_true + noise
        For a well-conditioned system, fitted betas should be close to true.
        """
        rng = np.random.default_rng(42)
        n_time, n_regressors, n_sensors = 200, 3, 5

        # Well-conditioned design matrix
        X = rng.standard_normal((n_time, n_regressors)).astype(np.float32)
        beta_true = np.array([
            [1.0, 2.0, -1.0, 0.5, 3.0],
            [-0.5, 1.5, 0.0, 2.0, -1.0],
            [0.3, -0.7, 1.2, -0.3, 0.8],
        ], dtype=np.float32)
        noise = rng.standard_normal((n_time, n_sensors)).astype(np.float32) * 0.01
        Y = X @ beta_true + noise

        return jnp.array(X), jnp.array(Y), jnp.array(beta_true)

    def test_fit_returns_new_model(self, known_regression):
        from neurojax.glm import GeneralLinearModel

        X, Y, _ = known_regression
        glm = GeneralLinearModel(X, Y)
        fitted = glm.fit()
        # Original model should still have None betas
        assert glm.betas is None
        # Fitted model should have betas
        assert fitted.betas is not None

    def test_fit_beta_shape(self, known_regression):
        from neurojax.glm import GeneralLinearModel

        X, Y, _ = known_regression
        glm = GeneralLinearModel(X, Y)
        fitted = glm.fit()
        assert fitted.betas.shape == (3, 5)  # (n_regressors, n_sensors)

    def test_fit_recovers_betas(self, known_regression):
        from neurojax.glm import GeneralLinearModel

        X, Y, beta_true = known_regression
        glm = GeneralLinearModel(X, Y)
        fitted = glm.fit()
        np.testing.assert_allclose(
            np.array(fitted.betas), np.array(beta_true), atol=0.05,
        )

    def test_residuals_shape(self, known_regression):
        from neurojax.glm import GeneralLinearModel

        X, Y, _ = known_regression
        glm = GeneralLinearModel(X, Y)
        fitted = glm.fit()
        assert fitted.residuals.shape == Y.shape

    def test_residuals_small(self, known_regression):
        """With low noise, residuals should be small."""
        from neurojax.glm import GeneralLinearModel

        X, Y, _ = known_regression
        glm = GeneralLinearModel(X, Y)
        fitted = glm.fit()
        max_residual = jnp.max(jnp.abs(fitted.residuals))
        assert max_residual < 0.5  # generous bound

    def test_fit_noiseless_exact(self):
        """Noiseless regression should recover exact betas."""
        from neurojax.glm import GeneralLinearModel

        rng = np.random.default_rng(7)
        X = jnp.array(rng.standard_normal((100, 2)), dtype=jnp.float32)
        beta_true = jnp.array([[3.0, -1.0], [2.0, 4.0]], dtype=jnp.float32)
        Y = X @ beta_true

        glm = GeneralLinearModel(X, Y)
        fitted = glm.fit()
        np.testing.assert_allclose(
            np.array(fitted.betas), np.array(beta_true), atol=1e-4,
        )


class TestGLMTStats:
    """Test t-statistic computation."""

    @pytest.fixture()
    def fitted_model(self):
        from neurojax.glm import GeneralLinearModel

        rng = np.random.default_rng(42)
        n_time, n_reg, n_sens = 200, 2, 3
        X = jnp.array(rng.standard_normal((n_time, n_reg)), dtype=jnp.float32)
        beta_true = jnp.array([[5.0, 0.0, 3.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
        noise = jnp.array(
            rng.standard_normal((n_time, n_sens)).astype(np.float32) * 0.1,
        )
        Y = X @ beta_true + noise
        glm = GeneralLinearModel(X, Y)
        return glm.fit(), beta_true

    def test_t_stats_shape(self, fitted_model):
        fitted, _ = fitted_model
        contrast = jnp.array([1.0, 0.0])
        t = fitted.get_t_stats(contrast)
        assert t.shape == (3,)

    def test_t_stats_finite(self, fitted_model):
        fitted, _ = fitted_model
        contrast = jnp.array([1.0, 0.0])
        t = fitted.get_t_stats(contrast)
        assert jnp.isfinite(t).all()

    def test_large_t_for_true_effect(self, fitted_model):
        """Sensor 0 has a true beta=5 for regressor 0; t should be large."""
        fitted, _ = fitted_model
        contrast = jnp.array([1.0, 0.0])
        t = fitted.get_t_stats(contrast)
        # Sensor 0 has large effect, sensor 1 has ~zero effect
        assert jnp.abs(t[0]) > 10.0  # should be very large given low noise
        assert jnp.abs(t[1]) < 5.0   # should be near zero

    def test_unfitted_raises(self):
        from neurojax.glm import GeneralLinearModel

        X = jnp.ones((10, 2))
        Y = jnp.zeros((10, 3))
        glm = GeneralLinearModel(X, Y)
        with pytest.raises(ValueError, match="fitted"):
            glm.get_t_stats(jnp.array([1.0, 0.0]))


class TestGLMLogLikelihood:
    def test_log_likelihood_finite(self):
        from neurojax.glm import GeneralLinearModel

        rng = np.random.default_rng(42)
        X = jnp.array(rng.standard_normal((100, 2)), dtype=jnp.float32)
        Y = jnp.array(rng.standard_normal((100, 3)), dtype=jnp.float32)
        fitted = GeneralLinearModel(X, Y).fit()
        ll = fitted.log_likelihood()
        assert jnp.isfinite(ll)

    def test_log_likelihood_negative(self):
        from neurojax.glm import GeneralLinearModel

        rng = np.random.default_rng(42)
        X = jnp.array(rng.standard_normal((100, 2)), dtype=jnp.float32)
        Y = jnp.array(rng.standard_normal((100, 3)), dtype=jnp.float32)
        fitted = GeneralLinearModel(X, Y).fit()
        ll = fitted.log_likelihood()
        # Log-likelihood of normal data is typically negative
        assert ll < 0

    def test_unfitted_raises(self):
        from neurojax.glm import GeneralLinearModel

        X = jnp.ones((10, 2))
        Y = jnp.zeros((10, 3))
        glm = GeneralLinearModel(X, Y)
        with pytest.raises(ValueError, match="fitted"):
            glm.log_likelihood()


class TestPermutationTest:
    """Test the permutation testing function."""

    def test_permutation_test_output_shapes(self):
        from neurojax.glm import GeneralLinearModel, run_permutation_test

        rng = np.random.default_rng(42)
        n_time, n_reg, n_sens = 50, 2, 3
        X = jnp.array(rng.standard_normal((n_time, n_reg)), dtype=jnp.float32)
        Y = jnp.array(rng.standard_normal((n_time, n_sens)), dtype=jnp.float32)
        model = GeneralLinearModel(X, Y)
        contrast = jnp.array([1.0, 0.0])
        key = jax.random.PRNGKey(0)

        true_t, p_values = run_permutation_test(model, contrast, key, n_perms=50)
        assert true_t.shape == (n_sens,)
        assert p_values.shape == (n_sens,)

    def test_pvalues_in_range(self):
        from neurojax.glm import GeneralLinearModel, run_permutation_test

        rng = np.random.default_rng(42)
        n_time, n_reg, n_sens = 50, 2, 3
        X = jnp.array(rng.standard_normal((n_time, n_reg)), dtype=jnp.float32)
        Y = jnp.array(rng.standard_normal((n_time, n_sens)), dtype=jnp.float32)
        model = GeneralLinearModel(X, Y)
        contrast = jnp.array([1.0, 0.0])
        key = jax.random.PRNGKey(0)

        _, p_values = run_permutation_test(model, contrast, key, n_perms=50)
        assert jnp.all(p_values >= 0.0)
        assert jnp.all(p_values <= 1.0)

    def test_strong_signal_low_pvalue(self):
        """A very strong signal should produce a low p-value."""
        from neurojax.glm import GeneralLinearModel, run_permutation_test

        rng = np.random.default_rng(42)
        n_time = 100
        X = jnp.array(rng.standard_normal((n_time, 2)), dtype=jnp.float32)
        # Strong signal: beta=[50, 0], very low noise
        beta = jnp.array([[50.0], [0.0]])
        noise = jnp.array(rng.standard_normal((n_time, 1)).astype(np.float32) * 0.01)
        Y = X @ beta + noise

        model = GeneralLinearModel(X, Y)
        contrast = jnp.array([1.0, 0.0])
        key = jax.random.PRNGKey(123)

        _, p_values = run_permutation_test(model, contrast, key, n_perms=100)
        # p-value should be very small (likely 0 with 100 perms)
        assert p_values[0] < 0.05


# ---------------------------------------------------------------------------
# PowerSpectrumModel
# ---------------------------------------------------------------------------

class TestPowerSpectrumModel:
    """Test the parametric power spectrum model."""

    def test_call_shape(self):
        from neurojax.spectral import PowerSpectrumModel

        model = PowerSpectrumModel()
        freqs = jnp.linspace(1.0, 50.0, 100)
        # 2 aperiodic + 3*1 peak params = 5
        params = jnp.array([1.0, 1.0, 10.0, 0.5, 1.0])
        out = model(freqs, params)
        assert out.shape == (100,)

    def test_call_finite(self):
        from neurojax.spectral import PowerSpectrumModel

        model = PowerSpectrumModel()
        freqs = jnp.linspace(1.0, 50.0, 100)
        params = jnp.array([1.0, 1.0, 10.0, 0.5, 1.0])
        out = model(freqs, params)
        assert jnp.isfinite(out).all()

    def test_aperiodic_decreasing(self):
        """With only aperiodic (no peaks), power should decrease with frequency."""
        from neurojax.spectral import PowerSpectrumModel

        model = PowerSpectrumModel()
        freqs = jnp.linspace(1.0, 50.0, 100)
        # Set peak power very low (near 0 after softplus) by using large negative pw
        params = jnp.array([2.0, 1.5, 10.0, -20.0, 1.0])
        out = model(freqs, params)
        # Aperiodic part: offset - exponent * log10(f)
        # Should be monotonically decreasing
        diffs = jnp.diff(out)
        # Most diffs should be negative (allow small positive due to tiny peak)
        assert jnp.sum(diffs < 0) > 80

    def test_multiple_peaks(self):
        from neurojax.spectral import PowerSpectrumModel

        model = PowerSpectrumModel()
        freqs = jnp.linspace(1.0, 50.0, 200)
        # 2 peaks: one at 10 Hz, one at 30 Hz
        params = jnp.array([1.0, 1.0, 10.0, 2.0, 1.0, 30.0, 1.5, 1.0])
        out = model(freqs, params)
        assert out.shape == (200,)
        assert jnp.isfinite(out).all()

    def test_peak_at_correct_frequency(self):
        """A strong peak at 10 Hz should produce max power near 10 Hz
        (after subtracting aperiodic)."""
        from neurojax.spectral import PowerSpectrumModel

        model = PowerSpectrumModel()
        freqs = jnp.linspace(1.0, 50.0, 500)
        params = jnp.array([0.0, 0.0, 10.0, 5.0, 0.5])
        out = model(freqs, params)

        # Find peak frequency
        peak_idx = jnp.argmax(out)
        peak_freq = freqs[peak_idx]
        # Should be close to 10 Hz
        assert jnp.abs(peak_freq - 10.0) < 2.0

    def test_jit_compatible(self):
        from neurojax.spectral import PowerSpectrumModel

        model = PowerSpectrumModel()

        @jax.jit
        def evaluate(freqs, params):
            return model(freqs, params)

        freqs = jnp.linspace(1.0, 50.0, 100)
        params = jnp.array([1.0, 1.0, 10.0, 0.5, 1.0])
        out = evaluate(freqs, params)
        assert jnp.isfinite(out).all()


class TestFitSpectrum:
    """Test spectrum fitting on synthetic data."""

    def test_fit_returns_params(self):
        from neurojax.spectral import PowerSpectrumModel, fit_spectrum

        freqs = jnp.linspace(1.0, 50.0, 200)
        model = PowerSpectrumModel()

        # Generate synthetic spectrum with known params
        true_params = jnp.array([2.0, 1.0, 10.0, 1.0, 1.0])
        psd = model(freqs, true_params)

        fitted_params = fit_spectrum(freqs, psd, n_peaks=1)
        assert fitted_params.shape == true_params.shape
        assert jnp.isfinite(fitted_params).all()

    def test_fit_recovers_aperiodic(self):
        """Fit should approximately recover the aperiodic parameters."""
        from neurojax.spectral import PowerSpectrumModel, fit_spectrum

        freqs = jnp.linspace(1.0, 50.0, 200)
        model = PowerSpectrumModel()

        true_params = jnp.array([2.0, 1.5, 10.0, 1.0, 1.0])
        psd = model(freqs, true_params)

        fitted_params = fit_spectrum(freqs, psd, n_peaks=1, initial_params=true_params * 1.1)
        # Offset and exponent should be close to true
        np.testing.assert_allclose(float(fitted_params[0]), 2.0, atol=0.5)
        np.testing.assert_allclose(float(fitted_params[1]), 1.5, atol=0.5)

    def test_fit_with_custom_initial_params(self):
        from neurojax.spectral import fit_spectrum

        freqs = jnp.linspace(1.0, 50.0, 100)
        psd = jnp.ones(100)
        init = jnp.array([1.0, 1.0, 10.0, 0.5, 1.0])

        result = fit_spectrum(freqs, psd, n_peaks=1, initial_params=init)
        assert result.shape == (5,)
        assert jnp.isfinite(result).all()

    def test_fit_sinusoid_peak(self):
        """Generate a power spectrum from a known sinusoid and verify the
        fitted model places a peak near the correct frequency."""
        from neurojax.spectral import PowerSpectrumModel, fit_spectrum

        # Create a signal with a 10 Hz sinusoid
        sfreq = 256.0
        n_samples = 2048
        t = jnp.arange(n_samples) / sfreq
        signal = jnp.sin(2 * jnp.pi * 10.0 * t)

        # Compute power spectrum
        fft_vals = jnp.fft.rfft(np.array(signal))
        freqs = jnp.fft.rfftfreq(n_samples, d=1.0 / sfreq)
        psd = jnp.abs(fft_vals) ** 2 / n_samples

        # Use only positive frequencies above 1 Hz
        mask = freqs >= 1.0
        freqs_pos = freqs[mask]
        psd_pos = jnp.log10(psd[mask] + 1e-20)  # log space for model

        # Fit
        init = jnp.array([0.0, 0.0, 10.0, 5.0, 0.5])
        fitted = fit_spectrum(freqs_pos, psd_pos, n_peaks=1, initial_params=init)

        # The peak center_freq (index 2) should be near 10 Hz
        peak_cf = float(fitted[2])
        assert abs(peak_cf - 10.0) < 3.0, f"Peak at {peak_cf}, expected ~10 Hz"
