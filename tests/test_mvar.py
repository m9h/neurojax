"""Multivariate AR (MVAR) model + directed connectivity: PDC and DTF.

Fills the OSL/osl-dynamics "MAR spectra" + directed-connectivity gap. A least-
squares MVAR fit gives the AR coefficient matrices; from those come Partial
Directed Coherence (Baccalá & Sameshima 2001) and the Directed Transfer Function
(Kamiński & Blinowska 1991), the two standard frequency-domain Granger-style
directed measures. Validated on a 2-channel system with a known one-way coupling.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from neurojax.analysis.mvar import dtf, fit_mvar, pdc


def _simulate_var1(T=3000, seed=0):
    # x is autonomous; y is driven by past x  ->  directed coupling x -> y.
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.0], [0.8, 0.5]])  # A[i,j]: influence of x_j(t-1) on x_i(t)
    X = np.zeros((T, 2))
    e = 0.5 * rng.standard_normal((T, 2))
    for t in range(1, T):
        X[t] = A @ X[t - 1] + e[t]
    return X, A


def test_fit_mvar_recovers_coeffs():
    X, A_true = _simulate_var1()
    A, Sigma = fit_mvar(jnp.asarray(X), order=1)
    assert A.shape == (1, 2, 2)
    assert Sigma.shape == (2, 2)
    assert np.allclose(np.asarray(A[0]), A_true, atol=0.1)


def test_pdc_is_directional_and_bounded():
    X, _ = _simulate_var1()
    A, _ = fit_mvar(jnp.asarray(X), order=1)
    freqs = jnp.linspace(0.0, 50.0, 33)
    p = pdc(A, freqs, fs=100.0)
    assert p.shape == (33, 2, 2)
    # element [i=1, j=0] is influence x(0) -> y(1): must dominate the reverse.
    assert float(jnp.mean(p[:, 1, 0])) > float(jnp.mean(p[:, 0, 1])) + 0.2
    assert jnp.all(p >= -1e-6) and jnp.all(p <= 1.0 + 1e-6)


def test_dtf_is_directional():
    X, _ = _simulate_var1()
    A, _ = fit_mvar(jnp.asarray(X), order=1)
    freqs = jnp.linspace(0.0, 50.0, 33)
    g = dtf(A, freqs, fs=100.0)
    assert g.shape == (33, 2, 2)
    assert float(jnp.mean(g[:, 1, 0])) > float(jnp.mean(g[:, 0, 1]))
    assert jnp.all(g >= -1e-6) and jnp.all(g <= 1.0 + 1e-6)
