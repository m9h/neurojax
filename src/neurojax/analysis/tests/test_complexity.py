"""Tests for neurojax.analysis.complexity — Lempel-Ziv complexity with named
variants, and the Mediano/Rosas/Barrett/Bor spectral-vs-phasic decomposition
of a between-condition feature difference (arXiv:2009.10015)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neurojax.analysis.complexity import (
    lz76,
    lz_complexity,
    spectral_phase_decomposition,
)


def _ar2(key, n, f0, fs=250.0, r=0.95, n_ch=1):
    """Linear Gaussian AR(2) with a resonance at f0 Hz: no phase structure."""
    a1 = 2 * r * np.cos(2 * np.pi * f0 / fs)
    a2 = -(r ** 2)
    e = np.asarray(jax.random.normal(key, (n + 500, n_ch)))
    x = np.zeros_like(e)
    for t in range(2, len(e)):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + e[t]
    return jnp.asarray(x[500:])


class TestLZ76:
    def test_kaspar_schuster_example(self):
        # Kaspar & Schuster (1987): 0001101001000101 parses into 6 words
        s = np.array([int(c) for c in "0001101001000101"], dtype=np.uint8)
        assert lz76(s) == 6

    def test_constant_and_alternating_are_minimal(self):
        assert lz76(np.zeros(1000, dtype=np.uint8)) == 2
        assert lz76(np.tile([0, 1], 500).astype(np.uint8)) == 3

    def test_random_string_is_near_asymptotic_bound(self):
        rng = np.random.RandomState(0)
        s = (rng.rand(20000) > 0.5).astype(np.uint8)
        n = len(s)
        assert 0.9 < lz76(s) / (n / np.log2(n)) < 1.2


class TestLZComplexity:
    def test_variants_are_named_and_distinct(self):
        x = _ar2(jax.random.PRNGKey(0), 4000, 10.0)[:, 0]
        c_env = lz_complexity(x, binarise="envelope-median")
        c_raw = lz_complexity(x, binarise="median")
        assert c_env != c_raw
        with pytest.raises(ValueError):
            lz_complexity(x, binarise="something-else")
        with pytest.raises(ValueError):
            lz_complexity(x, normalise="magic")

    def test_analytic_normalisation_bounds(self):
        rng = np.random.RandomState(1)
        white = jnp.asarray(rng.randn(8000))
        c = lz_complexity(white, binarise="median", normalise="analytic")
        assert 0.9 < float(c) < 1.2
        slow = jnp.asarray(np.sin(2 * np.pi * np.arange(8000) / 400.0))
        assert float(lz_complexity(slow, binarise="median", normalise="analytic")) < 0.1

    def test_phase_normalisation_is_unity_for_linear_gaussian(self):
        x = _ar2(jax.random.PRNGKey(2), 8000, 10.0)[:, 0]
        c = lz_complexity(x, binarise="envelope-median", normalise="phase",
                          key=jax.random.PRNGKey(3), n_surrogates=20)
        assert 0.9 < float(c) < 1.1

    def test_multichannel_returns_per_channel(self):
        x = _ar2(jax.random.PRNGKey(4), 2000, 10.0, n_ch=3)
        c = lz_complexity(x, binarise="median", normalise="analytic")
        assert c.shape == (3,)

    def test_spectral_confound_is_mechanical(self):
        """Less alpha / more broadband raises raw LZ with no change in
        phase structure — the confound the decomposition is for."""
        alpha = _ar2(jax.random.PRNGKey(5), 8000, 10.0, r=0.98)[:, 0]
        broad = _ar2(jax.random.PRNGKey(6), 8000, 10.0, r=0.5)[:, 0]
        assert float(lz_complexity(broad, "envelope-median", "analytic")) > \
            float(lz_complexity(alpha, "envelope-median", "analytic"))


class TestSpectralPhaseDecomposition:
    def test_linear_spectral_shift_is_all_spectral(self):
        """Two linear Gaussian processes with different spectra: the whole LZ
        difference is spectral, the phasic part is zero within surrogate noise."""
        a = _ar2(jax.random.PRNGKey(10), 8000, 10.0, r=0.98)[:, 0]
        b = _ar2(jax.random.PRNGKey(11), 8000, 10.0, r=0.6)[:, 0]
        feat = lambda x: lz_complexity(x, "envelope-median", "analytic")
        d = spectral_phase_decomposition(feat, a, b, key=jax.random.PRNGKey(12),
                                         n_surrogates=30)
        for k in ("total", "spectral", "phasic", "phasic_std", "spectral_std"):
            assert k in d
        np.testing.assert_allclose(d["total"], d["spectral"] + d["phasic"], rtol=1e-6)
        assert d["total"] > 0
        assert abs(d["phasic"]) < 2.5 * d["phasic_std"]
        assert d["spectral"] / d["total"] > 0.8

    def test_nonlinear_phase_structure_shows_in_phasic_part(self):
        """Condition b = condition a with the same spectrum but phase-coupled
        bursts (amplitude gated by a slow phase): the phasic part is non-zero."""
        key = jax.random.PRNGKey(20)
        a = _ar2(key, 8000, 10.0, r=0.95)[:, 0]
        # same amplitude spectrum, destroyed phase structure, as condition a
        from neurojax.analysis.surrogates import phase_randomized_surrogate
        a_lin = phase_randomized_surrogate(a, jax.random.PRNGKey(21))
        t = np.arange(8000)
        gate = (1 + np.sign(np.sin(2 * np.pi * t / 500.0))) / 2  # square bursts
        b = jnp.asarray(np.asarray(a) * (0.2 + gate))
        feat = lambda x: lz_complexity(x, "envelope-median", "analytic")
        d = spectral_phase_decomposition(feat, a_lin, b, key=jax.random.PRNGKey(22),
                                         n_surrogates=30)
        assert abs(d["phasic"]) > 3 * d["phasic_std"]

    def test_surrogate_feature_values_returned_for_intervals(self):
        a = _ar2(jax.random.PRNGKey(30), 2000, 10.0)[:, 0]
        b = _ar2(jax.random.PRNGKey(31), 2000, 20.0)[:, 0]
        feat = lambda x: lz_complexity(x, "median", "analytic")
        d = spectral_phase_decomposition(feat, a, b, key=jax.random.PRNGKey(32),
                                         n_surrogates=5)
        assert d["surrogate_a"].shape == (5,) and d["surrogate_b"].shape == (5,)
