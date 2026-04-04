"""TDD tests for analytic signal processing module.

Tests cover: Hilbert transform, envelope, instantaneous phase/frequency,
PLV, imaginary PLV, envelope correlation, phase-amplitude coupling,
narrowband analytic, circular statistics.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp


class TestHilbert:
    """Core Hilbert transform and analytic signal."""

    def test_analytic_signal_is_complex(self):
        from neurojax.analysis.analytic import hilbert
        x = jnp.sin(2 * jnp.pi * 10 * jnp.linspace(0, 1, 1000))
        z = hilbert(x)
        assert jnp.iscomplexobj(z)

    def test_real_part_equals_input(self):
        from neurojax.analysis.analytic import hilbert
        x = jnp.sin(2 * jnp.pi * 10 * jnp.linspace(0, 1, 1000))
        z = hilbert(x)
        np.testing.assert_allclose(jnp.real(z), x, atol=1e-5)

    def test_envelope_of_sine_is_one(self):
        from neurojax.analysis.analytic import envelope
        x = jnp.sin(2 * jnp.pi * 10 * jnp.linspace(0, 1, 1000))
        env = envelope(x)
        # Envelope of a pure sine should be ~1.0 (except at edges)
        np.testing.assert_allclose(env[50:-50], 1.0, atol=0.05)

    def test_envelope_of_modulated_signal(self):
        from neurojax.analysis.analytic import envelope
        t = jnp.linspace(0, 1, 1000)
        mod = 0.5 + 0.5 * jnp.sin(2 * jnp.pi * 2 * t)  # 2 Hz modulation
        x = mod * jnp.sin(2 * jnp.pi * 40 * t)  # 40 Hz carrier
        env = envelope(x)
        # Envelope should track the modulation
        corr = float(jnp.corrcoef(env[100:-100], mod[100:-100])[0, 1])
        assert corr > 0.9

    def test_instantaneous_phase_range(self):
        from neurojax.analysis.analytic import instantaneous_phase
        x = jnp.sin(2 * jnp.pi * 10 * jnp.linspace(0, 1, 1000))
        phase = instantaneous_phase(x)
        assert float(phase.min()) >= -jnp.pi - 0.01
        assert float(phase.max()) <= jnp.pi + 0.01

    def test_instantaneous_frequency_of_sine(self):
        from neurojax.analysis.analytic import instantaneous_frequency
        x = jnp.sin(2 * jnp.pi * 10 * jnp.linspace(0, 1, 1000))
        freq = instantaneous_frequency(x, sfreq=1000)
        # Should be ~10 Hz everywhere (except edges)
        np.testing.assert_allclose(freq[50:-50], 10.0, atol=0.5)

    def test_batch_dimensions(self):
        from neurojax.analysis.analytic import hilbert, envelope
        x = jnp.sin(2 * jnp.pi * 10 * jnp.linspace(0, 1, 500))
        x_batch = jnp.stack([x, 2 * x, 0.5 * x])  # (3, 500)
        z = hilbert(x_batch)
        assert z.shape == (3, 500)
        env = envelope(x_batch)
        assert env.shape == (3, 500)

    def test_differentiable(self):
        from neurojax.analysis.analytic import envelope
        x = jnp.sin(2 * jnp.pi * 10 * jnp.linspace(0, 1, 200))
        grad = jax.grad(lambda x: jnp.sum(envelope(x)))(x)
        assert jnp.all(jnp.isfinite(grad))


class TestConnectivity:
    """Phase-based and envelope connectivity measures."""

    def test_plv_identical_signals(self):
        from neurojax.analysis.analytic import phase_locking_value
        x = jnp.sin(2 * jnp.pi * 10 * jnp.linspace(0, 1, 1000))
        plv = phase_locking_value(x, x)
        assert float(plv) > 0.99

    def test_plv_orthogonal_frequencies(self):
        from neurojax.analysis.analytic import phase_locking_value
        t = jnp.linspace(0, 1, 1000)
        x = jnp.sin(2 * jnp.pi * 10 * t)
        y = jnp.sin(2 * jnp.pi * 11 * t)  # different frequency
        plv = phase_locking_value(x, y)
        assert float(plv) < 0.3  # low PLV for different frequencies

    def test_plv_matrix_shape(self):
        from neurojax.analysis.analytic import plv_matrix
        data = jax.random.normal(jax.random.PRNGKey(0), (5, 500))
        plv = plv_matrix(data)
        assert plv.shape == (5, 5)
        # Diagonal should be 1
        np.testing.assert_allclose(jnp.diag(plv), 1.0, atol=0.01)

    def test_plv_matrix_symmetric(self):
        from neurojax.analysis.analytic import plv_matrix
        data = jax.random.normal(jax.random.PRNGKey(0), (5, 500))
        plv = plv_matrix(data)
        np.testing.assert_allclose(plv, plv.T, atol=1e-5)

    def test_imaginary_plv_zero_for_zero_lag(self):
        from neurojax.analysis.analytic import imaginary_plv
        t = jnp.linspace(0, 1, 1000)
        # Two copies of same signal (zero lag = volume conduction)
        data = jnp.stack([jnp.sin(2*jnp.pi*10*t), jnp.sin(2*jnp.pi*10*t)])
        iplv = imaginary_plv(data)
        # Imaginary part should be ~0 for zero-lag signals
        assert float(iplv[0, 1]) < 0.1

    def test_envelope_correlation_shape(self):
        from neurojax.analysis.analytic import envelope_correlation
        data = jax.random.normal(jax.random.PRNGKey(0), (4, 500))
        ec = envelope_correlation(data)
        assert ec.shape == (4, 4)
        # Diagonal should be ~1
        np.testing.assert_allclose(jnp.diag(ec), 1.0, atol=0.05)

    def test_envelope_correlation_symmetric(self):
        from neurojax.analysis.analytic import envelope_correlation
        data = jax.random.normal(jax.random.PRNGKey(0), (4, 500))
        ec = envelope_correlation(data)
        np.testing.assert_allclose(ec, ec.T, atol=1e-5)


class TestPAC:
    """Phase-amplitude coupling."""

    def test_pac_with_coupled_signals(self):
        from neurojax.analysis.analytic import phase_amplitude_coupling
        t = jnp.linspace(0, 2, 2000)
        theta = jnp.sin(2 * jnp.pi * 6 * t)
        # Gamma amplitude modulated by theta phase
        theta_phase = jnp.angle(jnp.sin(2*jnp.pi*6*t) + 1j*jnp.cos(2*jnp.pi*6*t))
        gamma = (1 + 0.5 * jnp.cos(theta_phase)) * jnp.sin(2 * jnp.pi * 40 * t)
        mi = phase_amplitude_coupling(theta, gamma)
        assert float(mi) > 0.1  # significant coupling

    def test_pac_uncoupled(self):
        from neurojax.analysis.analytic import phase_amplitude_coupling
        t = jnp.linspace(0, 2, 2000)
        x = jnp.sin(2 * jnp.pi * 6 * t)
        y = jnp.sin(2 * jnp.pi * 40 * t)  # uncoupled
        mi = phase_amplitude_coupling(x, y)
        assert float(mi) < 0.2  # low coupling


class TestNarrowband:
    """Narrowband analytic signal."""

    def test_narrowband_extracts_band(self):
        from neurojax.analysis.analytic import narrowband_analytic
        t = jnp.linspace(0, 1, 1000)
        # Mix of 10 Hz and 40 Hz
        x = jnp.sin(2*jnp.pi*10*t) + jnp.sin(2*jnp.pi*40*t)
        # Extract alpha (8-13 Hz)
        z_alpha = narrowband_analytic(x, sfreq=1000, fmin=8, fmax=13)
        # Should have ~10 Hz, not 40 Hz
        env = jnp.abs(z_alpha)
        # Envelope should be ~1 (just the 10 Hz component)
        np.testing.assert_allclose(env[100:-100], 1.0, atol=0.3)

    def test_narrowband_is_complex(self):
        from neurojax.analysis.analytic import narrowband_analytic
        x = jnp.sin(2*jnp.pi*10*jnp.linspace(0, 1, 500))
        z = narrowband_analytic(x, sfreq=500, fmin=8, fmax=13)
        assert jnp.iscomplexobj(z)


class TestCircularStats:
    """Circular statistics on phase."""

    def test_circular_mean_concentrated(self):
        from neurojax.analysis.analytic import circular_mean
        # All phases near 0
        phases = jnp.array([0.1, -0.1, 0.05, -0.05, 0.0])
        mu = circular_mean(phases)
        assert abs(float(mu)) < 0.1

    def test_circular_mean_near_pi(self):
        from neurojax.analysis.analytic import circular_mean
        phases = jnp.array([3.1, -3.1, 3.0, -3.0])
        mu = circular_mean(phases)
        assert abs(float(mu)) > 2.9  # near +/-pi

    def test_circular_variance_uniform(self):
        from neurojax.analysis.analytic import circular_variance
        # Uniform phases → high variance
        phases = jnp.linspace(-jnp.pi, jnp.pi, 100)
        cv = circular_variance(phases)
        assert float(cv) > 0.9

    def test_circular_variance_concentrated(self):
        from neurojax.analysis.analytic import circular_variance
        phases = jnp.zeros(100) + 0.01 * jax.random.normal(jax.random.PRNGKey(0), (100,))
        cv = circular_variance(phases)
        assert float(cv) < 0.1

    def test_rayleigh_z_significant(self):
        from neurojax.analysis.analytic import rayleigh_z
        # Concentrated phases → large Z
        phases = jnp.zeros(100) + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (100,))
        z = rayleigh_z(phases)
        assert float(z) > 10  # highly significant

    def test_rayleigh_z_uniform(self):
        from neurojax.analysis.analytic import rayleigh_z
        phases = jnp.linspace(-jnp.pi, jnp.pi, 100)
        z = rayleigh_z(phases)
        assert float(z) < 5  # not significant


class TestArtifactDetection:
    """Envelope-based artifact detection."""

    def test_detect_clean_data(self):
        from neurojax.analysis.analytic import detect_artifacts
        data = jax.random.normal(jax.random.PRNGKey(0), (5, 1000))
        bad = detect_artifacts(data, threshold=5.0)
        # Clean Gaussian data should have very few "artifacts"
        assert float(bad.mean()) < 0.05

    def test_detect_injected_artifact(self):
        from neurojax.analysis.analytic import detect_artifacts
        data = jax.random.normal(jax.random.PRNGKey(0), (5, 1000)) * 0.1
        # Inject large artifact at sample 500
        data = data.at[0, 498:502].set(10.0)
        bad = detect_artifacts(data, threshold=4.0)
        # Should detect the artifact region
        assert bool(bad[500])
