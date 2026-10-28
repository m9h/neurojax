"""HADES (Vohryzek 2024, NSR) functional-harmonic basis + spacetime metrics.

Mirror of hippy-feat/tests/test_hades.py; neurojax is the canonical copy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from neurojax.spatial.harmonics import (
    connectome_harmonics,
    cosine_projection,
    fractional_occupancy,
    functional_harmonics_adjacency,
    harmonic_contributions,
    latent_spread,
    lifetimes,
    transition_matrix,
    winner_takes_all,
)


def test_knn_adjacency_symmetric_binary():
    FC = np.corrcoef(np.random.default_rng(0).normal(size=(30, 50)))
    A = functional_harmonics_adjacency(FC, knn=5)
    assert sp.issparse(A)
    A = A.toarray()
    assert A.shape == (30, 30) and np.array_equal(A, A.T)
    assert set(np.unique(A)) <= {0.0, 1.0} and np.all(np.diag(A) == 0)
    assert np.all(A.sum(1) >= 5)


def test_fisher_z_unwound():
    FC = np.corrcoef(np.random.default_rng(1).normal(size=(20, 50)))
    A_r = functional_harmonics_adjacency(FC, knn=4)
    A_z = functional_harmonics_adjacency(np.arctanh(np.clip(FC, -0.999, 0.999)), knn=4, fisher_z=True)
    assert (A_r != A_z).nnz == 0


def test_below_row_median_dropped():
    FC = np.eye(6)
    FC[0, 1] = FC[1, 0] = 0.9
    FC[0, 2] = FC[2, 0] = 0.8
    A = functional_harmonics_adjacency(FC, knn=5).toarray()
    assert A[0, 1] == 1 and A[0, 2] == 1 and A[0, 3:].sum() == 0


def test_combinatorial_functional_harmonics():
    FC = np.corrcoef(np.random.default_rng(2).normal(size=(40, 100)))
    A = functional_harmonics_adjacency(FC, knn=6)
    vals, psi = connectome_harmonics(A, n_modes=10, normalized=False)
    L = np.diag(np.asarray(A.sum(1)).ravel()) - A.toarray()
    np.testing.assert_allclose(np.asarray(vals), np.sort(np.linalg.eigvalsh(L))[:10], atol=1e-6)
    assert vals[0] == pytest.approx(0.0, abs=1e-6)


def test_cosine_projection_matches_matlab():
    rng = np.random.default_rng(4)
    psi = np.linalg.qr(rng.normal(size=(20, 3)))[0]
    X = rng.normal(size=(5, 20))
    cos = (X @ psi) / (np.linalg.norm(X, axis=1)[:, None] * np.linalg.norm(psi, axis=0)[None])
    P = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    ref = (90 - np.minimum(P, 180 - P)) / 90
    np.testing.assert_allclose(np.asarray(cosine_projection(X, psi)), ref, atol=1e-5)
    assert cosine_projection(-5.0 * psi[:, 2][None], psi)[0, 2] == pytest.approx(1.0, abs=1e-6)


@pytest.fixture
def alpha():
    return jnp.asarray(np.random.default_rng(5).normal(size=(100, 6)) * np.array([5, 1, 2, 1, 1, 1]))


def test_contributions(alpha):
    c = harmonic_contributions(alpha)
    a = np.abs(np.asarray(alpha))
    np.testing.assert_allclose(c["mean"], a.mean(0), rtol=1e-5)
    np.testing.assert_allclose(c["std"], a.std(0, ddof=1), rtol=1e-5)
    assert c["rel_mean"].sum() == pytest.approx(1.0, abs=1e-5)


def test_winner_takes_all_signed():
    a = jnp.asarray([[1.0, -9.0, 2.0], [0.5, 0.1, 0.4], [-3.0, 2.0, 2.5]])
    np.testing.assert_array_equal(winner_takes_all(a), [2, 0, 2])
    np.testing.assert_array_equal(winner_takes_all(a, k=2), [0, 0, 1])


def test_fractional_occupancy():
    np.testing.assert_allclose(fractional_occupancy(jnp.asarray([0, 0, 1, 2, 2, 2, 1, 1]), 3),
                               [2 / 8, 3 / 8, 3 / 8])


def test_lifetimes_leida_rules():
    s = jnp.asarray([0, 0, 1, 2, 2, 2, 0, 0, 1, 1])
    np.testing.assert_allclose(lifetimes(s, 3, tr=2.0), [4.0, 2.0, 6.0])
    np.testing.assert_allclose(lifetimes(jnp.zeros(5, int), 2, tr=1.0), [0.0, 0.0])
    np.testing.assert_allclose(lifetimes(jnp.asarray([1, 0, 1]), 2, tr=1.0), [1.0, 0.0])


def test_transition_matrix_and_jit():
    s = jnp.asarray([0, 1, 1, 2, 0])
    P = transition_matrix(s, 3)
    assert P.sum() == pytest.approx(1.0) and P[0, 1] == pytest.approx(0.25)
    f = jax.jit(lambda s: (fractional_occupancy(s, 3), lifetimes(s, 3, 1.0), transition_matrix(s, 3)))
    po, lt, Pj = f(s)
    np.testing.assert_allclose(Pj, P)
    np.testing.assert_allclose(lt, lifetimes(s, 3, 1.0))


def test_latent_spread(alpha):
    assert latent_spread(alpha) == pytest.approx(np.asarray(alpha).std(0, ddof=1).mean(), rel=1e-5)
