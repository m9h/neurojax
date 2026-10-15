"""Canonical microcircuit CSD port: the model-level checks of SPM 26's DCM M/EEG suite.

Tests 1 (steady-state Jacobian stability), 2 (initialisation at the fixed
point) and 6 (finite, non-zero prior CSD for single- and four-node LFP
networks) are reproduced here on the JAX port.  A numerical comparison
against live SPM output is in ``test_dcm_cmc_oracle.py`` and needs the
MATLAB Runtime; the DCM shipped with SPM's CSD example is SPM8-era and is
not a valid reference for the current equations.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from neurojax.dcm import (  # noqa: E402
    cmc_csd, cmc_priors, delayed_jacobian, four_node_test_architecture, fx_cmc, gen_Q,
)
from neurojax.dcm.cmc_csd import delay_matrix, jacobians  # noqa: E402

HZ = np.arange(2, 65, dtype=float)


def _single():
    return cmc_priors([np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))], (), np.ones((1, 1)))


def test_02_zero_is_a_fixed_point_single_and_four():
    for pE, n in ((_single()[0], 1), (cmc_priors(*four_node_test_architecture()[:3])[0], 4)):
        f = fx_cmc(jnp.zeros((n, 8)), jnp.zeros(n), pE)
        assert float(jnp.abs(f).max()) == 0.0


@pytest.mark.parametrize("n", [1, 4])
def test_01_jacobian_is_stable_at_prior_mean(n):
    if n == 1:
        pE, _ = _single()
    else:
        A, B, C, _ = four_node_test_architecture(); pE, _ = cmc_priors(A, B, C)
    J, dfdu = jacobians(jnp.zeros((n, 8)), jnp.zeros(n), pE)
    assert J.shape == (8 * n, 8 * n)
    lam = np.linalg.eigvals(np.asarray(J))
    assert lam.real.max() < 1e-9, lam.real.max()
    D = delay_matrix(pE, n, J)
    Jd, Qop = delayed_jacobian(J, D)
    assert np.isfinite(np.asarray(Jd)).all()
    lam_d = np.linalg.eigvals(np.asarray(Jd))
    assert lam_d.real.max() < 1e-9, lam_d.real.max()
    # voltage-current couplings are not delayed (nodelay = 1)
    assert float(jnp.abs(D[J == 1.0]).max()) == 0.0 if bool((J == 1.0).any()) else True
    # the input enters the spiny-stellate conductance of every source
    rows = np.nonzero(np.asarray(dfdu).any(axis=1))[0]
    assert set(rows.tolist()) == set(range(n, 2 * n))


def test_06_prior_csd_is_finite_and_nonzero_single_and_four():
    pE, _ = _single()
    y = cmc_csd(pE, HZ, np.zeros((1, 0)))
    assert len(y) == 1 and y[0].shape == (HZ.size, 1, 1)
    assert np.isfinite(np.asarray(y[0])).all() and float(jnp.abs(y[0]).sum()) > 1e-12
    A, B, C, X = four_node_test_architecture()
    pE, _ = cmc_priors(A, B, C)
    y = cmc_csd(pE, HZ, X)
    assert len(y) == 2 and y[0].shape == (HZ.size, 4, 4)
    for G in y:
        G = np.asarray(G)
        assert np.isfinite(G).all() and np.abs(G).sum() > 1e-12
        # Hermitian cross-spectra with real, positive auto-spectra
        np.testing.assert_allclose(G, np.conj(np.transpose(G, (0, 2, 1))), atol=1e-12)
        assert (np.real(np.diagonal(G, axis1=1, axis2=2)) > 0).all()


def test_gen_Q_applies_B_to_every_A_and_to_intrinsic_gain():
    A, B, C, X = four_node_test_architecture()
    pE, _ = cmc_priors(A, B, C)
    P = dict(pE); P["B"] = [jnp.asarray(B[0]) * 0.5]
    Q = gen_Q(P, X[1])
    for j in range(4):
        np.testing.assert_allclose(np.asarray(Q["A"][j] - pE["A"][j]), 0.5 * np.asarray(B[0]))
    np.testing.assert_allclose(np.asarray(Q["G"][:, 0]), 0.5 * np.diag(B[0]))
    assert "B" not in Q


def test_csd_is_differentiable_in_the_parameters():
    pE, _ = _single()
    def power_at_alpha(g):
        P = dict(pE); P["G"] = pE["G"] + g
        G = cmc_csd(P, HZ, np.zeros((1, 0)))[0]
        return jnp.real(G[8, 0, 0])
    grad = jax.grad(power_at_alpha)(jnp.zeros((1, 4)))
    assert np.isfinite(np.asarray(grad)).all() and float(jnp.abs(grad).max()) > 0


def _spm_delay_reference(J, D, N=256):
    """Plain-loop transcription of spm_dcm_delay's Robbins-Monro scheme (data-dependent breaks)."""
    J = np.asarray(J, float); D = -np.asarray(D, float)
    m = J.shape[0]
    QJ = np.linalg.solve(np.eye(m) - D * J, J)
    a = 0.25
    tol = np.abs(QJ).sum(axis=1).max() * 1e-6
    dn = np.ones_like(D); Dn = []
    for n in range(N):
        dn = dn * D; Dn.append(dn * J)
    Q = J
    for _ in range(N):
        QJn = np.eye(m); Q = J.copy()
        for n in range(N):
            QJn = QJn @ QJ / (n + 1)
            dQ = Dn[n] @ QJn
            dQ[np.isnan(dQ)] = 0
            Q = Q + dQ
            if np.abs(dQ).sum(axis=1).max() < tol:
                break
        QJ = QJ * (1 - a) + Q * a
        if np.abs(QJ - Q).sum(axis=1).max() < tol:
            break
    return Q


@pytest.mark.parametrize("n", [1, 4])
def test_delay_operator_matches_spm_algorithm(n):
    if n == 1:
        pE, _ = _single()
    else:
        A, B, C, _ = four_node_test_architecture(); pE, _ = cmc_priors(A, B, C)
    P = dict(pE); P["D"] = pE["D"] + 0.3   # non-trivial delays
    J, _ = jacobians(jnp.zeros((n, 8)), jnp.zeros(n), P)
    D = delay_matrix(P, n, J)
    Jd, _ = delayed_jacobian(J, D)
    ref = _spm_delay_reference(J, D)
    np.testing.assert_allclose(np.asarray(Jd), ref, rtol=1e-6, atol=1e-6 * np.abs(ref).max())
