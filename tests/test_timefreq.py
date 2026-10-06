"""Tests for the EEGLAB-style complex Morlet time-frequency transform."""

import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from neurojax.analysis.timefreq import morlet_cwt, eeglab_cycles

TWO_PI = 2 * math.pi


def tone(fs, T, f, amp=1.0):
    t = jnp.arange(T) / fs
    return amp * jnp.sin(TWO_PI * f * t)


def test_power_peaks_at_tone_frequency():
    fs = 250.0
    x = tone(fs, 2500, 10.0)
    freqs = jnp.arange(2.0, 41.0, 1.0)
    C = morlet_cwt(x, fs, freqs, n_cycles=7.0)
    power = np.asarray(jnp.mean(jnp.abs(C) ** 2, axis=-1))
    assert abs(float(freqs[power.argmax()]) - 10.0) < 1.0


def test_amplitude_recovery():
    fs = 250.0
    x = tone(fs, 2500, 10.0, amp=3.0)
    C = morlet_cwt(x, fs, jnp.array([10.0]), n_cycles=7.0)
    amp = float(jnp.mean(jnp.abs(C[0, 500:-500])))   # skip wavelet edge effects
    np.testing.assert_allclose(amp, 3.0, atol=0.5)


def test_phase_tracks_frequency():
    fs = 250.0
    x = tone(fs, 2500, 10.0)
    C = morlet_cwt(x, fs, jnp.array([10.0]), n_cycles=7.0)
    phase = np.unwrap(np.angle(np.asarray(C[0, 500:-500])))
    ifreq = np.diff(phase) * fs / TWO_PI
    np.testing.assert_allclose(np.median(ifreq), 10.0, atol=0.5)


def test_two_tones_two_peaks():
    fs = 250.0
    x = tone(fs, 3000, 8.0) + tone(fs, 3000, 22.0)
    freqs = jnp.arange(2.0, 41.0, 1.0)
    power = np.asarray(jnp.mean(jnp.abs(morlet_cwt(x, fs, freqs, 7.0)) ** 2, axis=-1))
    # local maxima near 8 and 22 Hz
    assert power[np.argmin(np.abs(np.asarray(freqs) - 8))] > power[np.argmin(np.abs(np.asarray(freqs) - 15))]
    assert power[np.argmin(np.abs(np.asarray(freqs) - 22))] > power[np.argmin(np.abs(np.asarray(freqs) - 15))]


def test_more_cycles_sharpens_frequency():
    fs = 250.0
    x = tone(fs, 4000, 10.0)
    freqs = jnp.arange(5.0, 16.0, 0.25)
    def bandwidth(nc):
        p = np.asarray(jnp.mean(jnp.abs(morlet_cwt(x, fs, freqs, nc)) ** 2, axis=-1))
        p = p / p.max()
        return np.sum(p > 0.5)            # width of the peak at half-max
    assert bandwidth(12.0) < bandwidth(3.0)


def test_multichannel_shape():
    fs = 250.0
    x = jr.normal(jr.PRNGKey(0), (5, 1000))
    freqs = jnp.arange(2.0, 12.0, 1.0)
    C = morlet_cwt(x, fs, freqs, 7.0)
    assert C.shape == (5, freqs.shape[0], 1000)


class TestEEGLABCycles:
    def test_constant_when_no_expansion(self):
        freqs = jnp.arange(2.0, 41.0, 1.0)
        nc = np.asarray(eeglab_cycles(freqs, c0=3.0, expansion=0.0))
        np.testing.assert_allclose(nc, 3.0, atol=1e-6)

    def test_increases_with_expansion(self):
        freqs = jnp.arange(2.0, 41.0, 1.0)
        nc = np.asarray(eeglab_cycles(freqs, c0=3.0, expansion=0.5))
        np.testing.assert_allclose(nc[0], 3.0, atol=1e-6)   # c0 at lowest freq
        assert nc[-1] > nc[0]                                # grows with frequency
        assert np.all(np.diff(nc) >= -1e-9)                  # monotone
