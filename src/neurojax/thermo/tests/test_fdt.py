"""Analytic oracle for the FDT / non-equilibrium machinery.

Linear OU processes are exactly solvable, so correctness can be asserted against theory rather
than against a reference implementation:

  * detailed balance (symmetric A with isotropic noise) => FDT violation, entropy production and
    time-reversal asymmetry are all exactly zero;
  * all three increase monotonically with the antisymmetric (irreversible) part of A;
  * the Lyapunov solution really satisfies A C + C A^T + Q = 0;
  * the measures are differentiable (this is the point of the JAX implementation).
"""
import jax
import jax.numpy as jnp
import pytest

from neurojax.thermo.fdt import (entropy_production, fdt_violation, lyapunov,
                                 time_reversal_asymmetry)

N = 6
TAUS = jnp.linspace(0.05, 1.0, 8)
Q = jnp.eye(N)


def _sym_stable(key):
    M = jax.random.normal(key, (N, N))
    S = (M + M.T) / 2
    return S - jnp.eye(N) * (jnp.max(jnp.linalg.eigvalsh(S)) + 1.0)


def _asym(key, mix):
    k1, k2 = jax.random.split(key)
    S = _sym_stable(k1)
    K = jax.random.normal(k2, (N, N))
    return S + mix * (K - K.T) / 2


def test_lyapunov_residual():
    A = _asym(jax.random.PRNGKey(1), 0.7)
    C = lyapunov(A, Q)
    res = jnp.linalg.norm(A @ C + C @ A.T + Q)
    assert float(res) < 1e-4, f"Lyapunov residual {res}"


def test_detailed_balance_gives_zero_violation():
    """The analytic oracle: symmetric A + isotropic noise = equilibrium."""
    A = _sym_stable(jax.random.PRNGKey(2))
    assert float(fdt_violation(A, Q, TAUS)) < 1e-6
    assert float(entropy_production(A, Q)) < 1e-6
    assert float(time_reversal_asymmetry(A, Q, 0.5)) < 1e-6


def test_monotone_in_irreversibility():
    key = jax.random.PRNGKey(3)
    vals = [float(fdt_violation(_asym(key, m), Q, TAUS)) for m in (0.0, 0.25, 0.5, 1.0)]
    assert all(a < b for a, b in zip(vals, vals[1:])), vals


def test_measures_agree_on_equilibrium_boundary():
    """FDT violation, entropy production and time-reversal asymmetry vanish together."""
    key = jax.random.PRNGKey(4)
    for m in (0.0, 0.6):
        A = _asym(key, m)
        v, e, t = (float(fdt_violation(A, Q, TAUS)), float(entropy_production(A, Q)),
                   float(time_reversal_asymmetry(A, Q, 0.5)))
        assert (v < 1e-6) == (e < 1e-6) == (t < 1e-6)


def test_differentiable():
    A = _asym(jax.random.PRNGKey(5), 0.5)
    g = jax.grad(lambda a: fdt_violation(a, Q, TAUS))(A)
    assert g.shape == (N, N) and bool(jnp.all(jnp.isfinite(g)))
