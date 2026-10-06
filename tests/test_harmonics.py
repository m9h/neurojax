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


# ---------------------------------------------------------------------------
# Surface-scale basis (mesh + tract graph, sparse solver) and CHD metrics
# ---------------------------------------------------------------------------

import pytest
import scipy.sparse as sp

from neurojax.spatial.harmonics import (
    connectome_adjacency,
    harmonic_energy,
    log_binned_spectrum,
    mesh_edges,
    repertoire_entropy,
)


def grid_faces(n):
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            faces += [[a, b, c], [b, d, c]]
    return np.asarray(faces), n * n


def test_mesh_edges_unique_undirected():
    edges = mesh_edges(np.array([[0, 1, 2], [1, 3, 2]]))
    assert edges.shape == (5, 2) and np.all(edges[:, 0] < edges[:, 1])


def test_connectome_adjacency_binary_symmetric_with_tracts():
    faces, n = grid_faces(4)
    A = connectome_adjacency(n, mesh_edges(faces), tract_edges=[[0, 15], [15, 0], [3, 3]])
    assert sp.issparse(A)
    A = A.toarray()
    np.testing.assert_array_equal(A, A.T)
    assert A.max() == 1.0 and np.all(np.diag(A) == 0) and A[0, 15] == 1.0


def test_sparse_solver_matches_dense():
    faces, n = grid_faces(12)
    A = connectome_adjacency(n, mesh_edges(faces))
    ev_s, Phi_s = connectome_harmonics(A, n_modes=30, normalized=True)
    ev_d, Phi_d = connectome_harmonics(jnp.asarray(A.toarray()), n_modes=30, normalized=True)
    np.testing.assert_allclose(np.asarray(ev_s), np.asarray(ev_d), atol=1e-4)
    np.testing.assert_allclose(np.asarray(Phi_s.T @ Phi_s), np.eye(30), atol=1e-3)   # float32
    assert ev_s[0] < 1e-6


def test_sparse_requires_n_modes_and_rejects_isolated_vertices():
    A = connectome_adjacency(5, np.array([[0, 1], [1, 2]]))
    with pytest.raises(ValueError, match="n_modes"):
        connectome_harmonics(A, normalized=True)
    with pytest.raises(ValueError, match="isolated"):
        connectome_harmonics(A, n_modes=2, normalized=True)


def test_harmonic_energy_lambda_squared():
    # E_k = |a_k|^2 lambda_k^2 (Atasoy 2017); white noise therefore loads high k
    alpha = jnp.array([[1.0, -2.0, 3.0]])
    np.testing.assert_allclose(
        harmonic_energy(alpha, jnp.array([0.0, 0.5, 2.0])), [[0.0, 1.0, 36.0]])


def test_log_binned_spectrum_covers_each_harmonic_once():
    binned, edges = log_binned_spectrum(jnp.ones((1, 100)), n_bins=15, reduce="sum")
    assert binned.shape == (1, 15) and edges.shape == (16,)
    assert float(binned.sum()) == 100.0
    mean, _ = log_binned_spectrum(jnp.ones((4, 100)), n_bins=15)
    np.testing.assert_allclose(mean, 1.0, atol=1e-6)


def test_repertoire_entropy_bounds():
    np.testing.assert_allclose(repertoire_entropy(jnp.ones((1, 64))), [np.log(64)], rtol=1e-5)
    np.testing.assert_allclose(
        repertoire_entropy(jnp.zeros((1, 64)).at[0, 3].set(1.0)), [0.0], atol=1e-5)
