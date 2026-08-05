"""Non-equilibrium thermodynamics of linear (Ornstein-Uhlenbeck) brain models, in JAX.

Implements the machinery behind Berjaga-Buisan et al. (2026) *Cell Reports* 45:117782,
"Thermodynamics of consciousness: Non-equilibrium brain dynamics track conscious states":
a multivariate OU / Langevin model is fitted to spontaneous activity ("generative effective
connectivity", GEC), and departures from the fluctuation-dissipation theorem (FDT) are used as a
stimulation-free correlate of conscious state.

Two differences from the reference MATLAB implementation:
  * GEC is fitted by **true autodiff gradients** through the Lyapunov solve, rather than the
    pseudo-gradient heuristic of the Gilson/Deco MOU-EC line of work.
  * Everything is jit/vmap-able, so subjects x conditions x random restarts run as one batched
    computation (the axis the reference pipeline farms out to SLURM).

Model
-----
    dx = A x dt + sqrt(2 D) dW ,        D = Q / 2

    stationary covariance   A C0 + C0 A^T + Q = 0        (Lyapunov)
    lagged covariance       C(tau) = expm(A tau) C0
    true response           R(tau) = expm(A tau)
    FDT-predicted response  R_FDT(tau) = -dC(tau)/dtau @ inv(D)   [equilibrium relation]

In equilibrium (detailed balance) the two responses coincide; the brain is not in equilibrium, so
they do not, and the mismatch is the FDT violation. Detailed balance for this model holds exactly
when `A @ C0` is symmetric -- which for isotropic noise means symmetric `A`. That gives an
**analytic oracle**: symmetric A must yield ~zero violation, asymmetric A must not.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl


def lyapunov(A: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """Solve A C + C A^T + Q = 0 for the stationary covariance C (vectorized Kronecker form)."""
    n = A.shape[0]
    I = jnp.eye(n)
    M = jnp.kron(I, A) + jnp.kron(A, I)          # vec(AC + CA^T) = (I⊗A + A⊗I) vec(C)
    c = jnp.linalg.solve(M, -Q.reshape(-1, order="F"))
    return c.reshape((n, n), order="F")


def lagged_covariance(A: jnp.ndarray, C0: jnp.ndarray, tau: float) -> jnp.ndarray:
    """C(tau) = expm(A tau) @ C0."""
    return jsl.expm(A * tau) @ C0


def fdt_violation(A: jnp.ndarray, Q: jnp.ndarray, taus: jnp.ndarray) -> jnp.ndarray:
    """Relative mismatch between the true and FDT-predicted response, averaged over `taus`.

    Returns a scalar in [0, inf): 0 = detailed balance (equilibrium), larger = further from
    equilibrium. Differentiable w.r.t. A and Q.
    """
    C0 = lyapunov(A, Q)
    D = Q / 2.0
    Dinv = jnp.linalg.inv(D)

    def per_tau(tau):
        E = jsl.expm(A * tau)
        R_true = E                      # impulse response
        dC = A @ E @ C0                 # d/dtau [expm(A tau) C0]
        R_fdt = -dC @ Dinv              # equilibrium FDT relation
        num = jnp.linalg.norm(R_true - R_fdt)
        den = jnp.linalg.norm(R_true) + 1e-12
        return num / den

    return jnp.mean(jax.vmap(per_tau)(taus))


def entropy_production(A: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """Steady-state entropy production rate of the OU process (0 iff detailed balance).

    Sigma = Tr(Q^{-1} (A C0 + C0 A^T + ... )) forms vary; here we use the standard
    irreversible-drift form: with the irreversible part A_irr = A + D C0^{-1},
    Sigma = Tr(A_irr C0 A_irr^T D^{-1}).
    """
    C0 = lyapunov(A, Q)
    D = Q / 2.0
    A_irr = A + D @ jnp.linalg.inv(C0)
    return jnp.trace(A_irr @ C0 @ A_irr.T @ jnp.linalg.inv(D))


def time_reversal_asymmetry(A: jnp.ndarray, Q: jnp.ndarray, tau: float) -> jnp.ndarray:
    """||C(tau) - C(tau)^T||_F / ||C(tau)||_F. Zero iff the process is time-reversible."""
    C0 = lyapunov(A, Q)
    C = lagged_covariance(A, C0, tau)
    return jnp.linalg.norm(C - C.T) / (jnp.linalg.norm(C) + 1e-12)
