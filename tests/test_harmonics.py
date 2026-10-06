"""Tests for connectome harmonics (graph-Laplacian eigenbasis; Atasoy 2016)."""

import numpy as np
import jax.numpy as jnp

from neurojax.spatial.harmonics import (
    connectome_harmonics,
    project_harmonics,
    harmonic_power_spectrum,
)


def path_graph(n):
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0
    return W


def ring_graph(n):
    W = path_graph(n)
    W[0, -1] = W[-1, 0] = 1.0
    return W


def test_path_graph_eigenvalues():
    # combinatorial Laplacian of a path: lambda_k = 2(1 - cos(pi k / n)), k=0..n-1
    n = 16
    evals, Phi = connectome_harmonics(jnp.asarray(path_graph(n)), normalized=False)
    expected = np.sort(2 * (1 - np.cos(np.pi * np.arange(n) / n)))
    np.testing.assert_allclose(np.asarray(evals), expected, atol=1e-5)
    assert evals[0] < 1e-6                                  # constant mode at 0
    assert np.all(np.diff(np.asarray(evals)) >= -1e-9)      # ascending (spatial freq)


def test_harmonics_orthonormal_and_reconstruct():
    n = 20
    W = ring_graph(n)
    evals, Phi = connectome_harmonics(jnp.asarray(W), normalized=False)
    Phi = np.asarray(Phi)
    np.testing.assert_allclose(Phi.T @ Phi, np.eye(n), atol=1e-5)   # orthonormal
    # full-basis projection round-trips an arbitrary signal
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, n))
    coeffs = np.asarray(project_harmonics(jnp.asarray(X), jnp.asarray(Phi)))
    recon = coeffs @ Phi.T
    np.testing.assert_allclose(recon, X, atol=3e-3)        # float32 round-trip


def test_lowest_mode_is_smooth():
    # the first non-trivial harmonic of a ring is a single sinusoid (lowest spatial freq)
    n = 24
    evals, Phi = connectome_harmonics(jnp.asarray(ring_graph(n)), normalized=False)
    Phi = np.asarray(Phi)
    v1 = Phi[:, 1]
    # one dominant spatial frequency -> its DFT has a single off-DC peak
    spec = np.abs(np.fft.rfft(v1 - v1.mean()))
    assert spec.argmax() == 1


def test_normalized_laplacian_modes():
    n = 12
    evals, Phi = connectome_harmonics(jnp.asarray(ring_graph(n)), normalized=True)
    assert evals[0] < 1e-6 and evals[-1] <= 2.0 + 1e-6      # symmetric-normalized: [0,2]
    assert harmonic_power_spectrum(jnp.ones((5, n)), Phi).shape == (n,)
