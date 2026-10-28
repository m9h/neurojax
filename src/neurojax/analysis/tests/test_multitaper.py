"""Tests for multitaper spectral estimation — TDD RED phase."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from neurojax.analysis.multitaper import (
    dpss_tapers,
    multitaper_coherence,
    multitaper_cpsd,
    multitaper_psd,
)


class TestDPSSTapers:
    def test_shape(self):
        tapers = dpss_tapers(n_samples=256, bandwidth=4.0, n_tapers=3)
        assert tapers.shape == (3, 256)

    def test_orthogonal(self):
        """DPSS tapers should be approximately orthonormal."""
        tapers = dpss_tapers(n_samples=256, bandwidth=4.0, n_tapers=4)
        G = tapers @ tapers.T  # should be ~identity
        np.testing.assert_allclose(G, jnp.eye(4), atol=0.05)

    def test_unit_energy(self):
        """Each taper should have unit energy (sum of squares = 1)."""
        tapers = dpss_tapers(n_samples=128, bandwidth=4.0, n_tapers=3)
        energies = jnp.sum(tapers ** 2, axis=1)
        np.testing.assert_allclose(energies, 1.0, atol=0.01)


class TestMultitaperPSD:
    def test_shape(self):
        x = jr.normal(jr.PRNGKey(0), (500, 4))  # (T, C)
        psd, freqs = multitaper_psd(x, fs=100.0, bandwidth=4.0)
        assert psd.shape[1] == 4  # n_channels
        assert freqs.shape[0] == psd.shape[0]

    def test_nonnegative(self):
        x = jr.normal(jr.PRNGKey(0), (500, 3))
        psd, _ = multitaper_psd(x, fs=100.0)
        assert jnp.all(psd >= 0)

    def test_nyquist_limit(self):
        x = jr.normal(jr.PRNGKey(0), (500, 2))
        _, freqs = multitaper_psd(x, fs=200.0)
        assert float(jnp.max(freqs)) <= 100.0 + 1.0

    def test_sine_wave_peak(self):
        """Pure sine at 10 Hz → PSD should peak near 10 Hz."""
        t = jnp.arange(1000) / 100.0  # 10 seconds at 100 Hz
        x = jnp.sin(2 * jnp.pi * 10.0 * t)[:, None]  # (1000, 1)
        psd, freqs = multitaper_psd(x, fs=100.0, bandwidth=2.0)
        peak_freq = float(freqs[jnp.argmax(psd[:, 0])])
        assert abs(peak_freq - 10.0) < 2.0

    def test_white_noise_flat(self):
        """White noise → PSD should be roughly flat."""
        x = jr.normal(jr.PRNGKey(0), (2000, 2))
        psd, _ = multitaper_psd(x, fs=100.0)
        for ch in range(2):
            cv = float(jnp.std(psd[:, ch]) / jnp.mean(psd[:, ch]))
            assert cv < 0.5, f"PSD not flat enough for white noise, CV={cv}"

    def test_density_normalisation(self):
        """White noise of variance v has one-sided density 2 v / fs (matches scipy.signal.welch)."""
        import scipy.signal

        x = np.random.default_rng(0).normal(size=(4000, 1)) * 3.0
        psd, freqs = multitaper_psd(jnp.asarray(x), fs=250.0, bandwidth=2.0)
        band = (np.asarray(freqs) > 5) & (np.asarray(freqs) < 120)
        np.testing.assert_allclose(float(np.mean(np.asarray(psd)[band, 0])), 2 * 9.0 / 250.0, rtol=0.1)
        fw, pw = scipy.signal.welch(x[:, 0], fs=250.0, nperseg=500)
        np.testing.assert_allclose(float(np.mean(pw[(fw > 5) & (fw < 120)])), 2 * 9.0 / 250.0, rtol=0.1)

    def test_finite(self):
        x = jr.normal(jr.PRNGKey(0), (500, 3))
        psd, freqs = multitaper_psd(x, fs=100.0)
        assert jnp.all(jnp.isfinite(psd))
        assert jnp.all(jnp.isfinite(freqs))


class TestMultitaperCPSD:
    def test_shape(self):
        x = jr.normal(jr.PRNGKey(0), (500, 3))
        cpsd, freqs = multitaper_cpsd(x, fs=100.0)
        n_freqs = freqs.shape[0]
        assert cpsd.shape == (n_freqs, 3, 3)

    def test_diagonal_matches_psd(self):
        x = jr.normal(jr.PRNGKey(0), (500, 3))
        cpsd, freqs_c = multitaper_cpsd(x, fs=100.0)
        psd, freqs_p = multitaper_psd(x, fs=100.0)
        for ch in range(3):
            np.testing.assert_allclose(
                jnp.real(cpsd[:, ch, ch]), psd[:, ch], atol=1e-5
            )


class TestMultitaperCoherence:
    def test_shape(self):
        x = jr.normal(jr.PRNGKey(0), (500, 4))
        coh, freqs = multitaper_coherence(x, fs=100.0)
        assert coh.shape[1] == 4
        assert coh.shape[2] == 4

    def test_bounded(self):
        x = jr.normal(jr.PRNGKey(0), (500, 3))
        coh, _ = multitaper_coherence(x, fs=100.0)
        assert jnp.all(coh >= -1e-6)
        assert jnp.all(coh <= 1.0 + 1e-6)

    def test_diagonal_is_one(self):
        x = jr.normal(jr.PRNGKey(0), (500, 3))
        coh, _ = multitaper_coherence(x, fs=100.0)
        for ch in range(3):
            np.testing.assert_allclose(coh[:, ch, ch], 1.0, atol=1e-4)

    def test_correlated_channels_high_coherence(self):
        """Two identical channels should have coherence = 1."""
        key = jr.PRNGKey(0)
        s = jr.normal(key, (1000, 1))
        x = jnp.concatenate([s, s], axis=1)  # two identical channels
        coh, _ = multitaper_coherence(x, fs=100.0)
        # Coherence between ch0 and ch1 should be ~1
        mean_coh = float(jnp.mean(coh[:, 0, 1]))
        assert mean_coh > 0.95
