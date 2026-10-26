"""Parametric empirical Bayes (PEB) over first-level Laplace posteriors.

Second-level model of Friston et al. (2016, NeuroImage 128:413-431;
``spm_dcm_peb``): subject ``i``'s first-level parameters are
``theta_i = X_i beta + eps_i`` with ``eps_i ~ N(0, Sigma)``, where the
first-level evidence for ``theta_i`` is carried by its Laplace posterior
``N(Ep_i, Cp_i)`` under the first-level prior ``N(pE, pC)``.  Because the
likelihood of the first-level data as a function of ``theta_i`` is (up to a
constant) ``N(theta_i | Ep_i, Cp_i) / N(theta_i | pE, pC)`` (Bayesian model
reduction identity), it is itself Gaussian with precision
``Lambda_i = Cp_i⁻¹ − pC⁻¹`` and mean ``m_i``; integrating ``theta_i`` out
leaves a linear-Gaussian model ``m_i ~ N(X_i beta, Sigma + Lambda_i⁻¹)``
whose evidence is closed-form in ``beta`` and maximised over the
between-subject log-precisions ``gamma`` (``Sigma = diag(pC) exp(−gamma)``)
by Newton steps on the free energy.

``Lambda_i`` is positive definite only where the first-level posterior is
tighter than its prior; eigenvalues below ``lambda_floor`` times the
largest are floored (the number floored is reported) — parameters the
first level did not learn about carry no second-level information, as in
SPM.

Bayesian model reduction (``log_evidence_reduce``, Friston & Penny 2011)
gives the evidence of any reduced prior from the full posterior without
refitting; ``bmr_effects`` applies it to each covariate × parameter effect.

This is the hierarchical step of the EEG/MEG mega-analysis: with a design
matrix of study and drug indicators the ``beta`` rows are the per-study /
per-drug effects on synaptic parameters, the analogue of Girn et al.'s
hyperpriors on complexity measures.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


def _logdet(M):
    return jnp.linalg.slogdet(M)[1]


def log_evidence_reduce(
    mu0: jnp.ndarray, S0: jnp.ndarray, mu: jnp.ndarray, S: jnp.ndarray, mur: jnp.ndarray, Sr: jnp.ndarray
) -> jnp.ndarray:
    """log p(y | reduced prior) − log p(y | full prior) from the full posterior.

    ``(mu0, S0)`` full prior, ``(mu, S)`` full posterior, ``(mur, Sr)`` reduced
    prior.  Exact for Gaussian priors/posteriors (Friston & Penny 2011).
    """
    P0, Pp, Pr = jnp.linalg.inv(S0), jnp.linalg.inv(S), jnp.linalg.inv(Sr)
    Q = Pr + Pp - P0
    b = Pr @ mur + Pp @ mu - P0 @ mu0
    return (
        0.5 * _logdet(S0)
        - 0.5 * _logdet(Sr)
        - 0.5 * _logdet(S)
        - 0.5 * _logdet(Q)
        + 0.5 * b @ jnp.linalg.solve(Q, b)
        - 0.5 * (mur @ Pr @ mur + mu @ Pp @ mu - mu0 @ P0 @ mu0)
    )


class PEBResult(NamedTuple):
    Eb: jnp.ndarray  # (n_cov, n_par) second-level effects
    Cb: jnp.ndarray  # (n_cov*n_par, n_cov*n_par) posterior covariance of vec(Eb) (row-major)
    Sigma: jnp.ndarray  # (n_par, n_par) between-subject covariance
    gamma: jnp.ndarray  # (n_par,) log-precision scalings
    F: jnp.ndarray  # free energy (log evidence bound)
    bE: jnp.ndarray  # prior mean of vec(beta)
    bC: jnp.ndarray  # prior covariance of vec(beta)
    X: jnp.ndarray
    n_floored: int  # first-level precision eigenvalues floored
    m: jnp.ndarray  # (n_sub, n_par) first-level likelihood means
    Lambda: jnp.ndarray  # (n_sub, n_par, n_par) first-level likelihood precisions


def peb(
    Ep: jnp.ndarray,
    Cp: jnp.ndarray,
    pE: jnp.ndarray,
    pC: jnp.ndarray,
    X: jnp.ndarray,
    gamma_prior_var: float = 4.0,
    lambda_floor: float = 1e-6,
    n_iter: int = 32,
) -> PEBResult:
    """Fit the second-level model.  ``Ep`` (n_sub, n_par), ``Cp`` (n_sub, n_par, n_par), ``X`` (n_sub, n_cov).

    The prior on each row of ``beta`` is ``N(0, pC)`` (SPM uses the
    first-level prior for the intercept); ``gamma ~ N(0, gamma_prior_var)``.
    """
    Ep, Cp, pE, pC, X = (jnp.asarray(a, dtype=jnp.float64) for a in (Ep, Cp, pE, pC, X))
    n, p = Ep.shape
    c = X.shape[1]
    P0 = jnp.linalg.inv(pC)
    Pp = jnp.linalg.inv(Cp)  # (n, p, p)
    Lam = Pp - P0[None]
    # floor the eigenvalues of each Lambda_i
    w, V = jnp.linalg.eigh(Lam)
    floor = lambda_floor * jnp.max(w, axis=1, keepdims=True)
    n_floored = int(jnp.sum(w < floor))
    w = jnp.maximum(w, floor)
    Lam = jnp.einsum("nij,nj,nkj->nik", V, w, V)
    m = jnp.linalg.solve(Lam, (jnp.einsum("nij,nj->ni", Pp, Ep) - (P0 @ pE)[None])[..., None])[..., 0]
    Lam_inv = jnp.linalg.inv(Lam)
    bE = jnp.zeros(c * p)
    bC = jnp.kron(jnp.eye(c), pC)
    bP = jnp.linalg.inv(bC)
    # m_i ≈ A_i vec(beta) with vec row-major over (cov, par): row k of A_i picks beta[cov, k] weighted by X[i, cov]
    A = jnp.einsum("nc,kq->nkcq", X, jnp.eye(p)).reshape(n, p, c * p)
    sig_base = jnp.diag(pC)

    def free_energy(gamma):
        Sigma = jnp.diag(sig_base * jnp.exp(-gamma))
        V = Lam_inv + Sigma[None]
        W = jnp.linalg.inv(V)
        Pb = bP + jnp.einsum("nkq,nkl,nlr->qr", A, W, A)
        g = jnp.einsum("nkq,nkl,nl->q", A, W, m)
        Eb = jnp.linalg.solve(Pb, g)
        F = (
            -0.5 * jnp.sum(jax.vmap(_logdet)(V))
            - 0.5 * n * p * jnp.log(2 * jnp.pi)
            - 0.5 * _logdet(bC)
            - 0.5 * _logdet(Pb)
            - 0.5 * jnp.einsum("nk,nkl,nl->", m, W, m)
            + 0.5 * Eb @ Pb @ Eb
            - 0.5 * gamma @ gamma / gamma_prior_var
            - 0.5 * p * jnp.log(2 * jnp.pi * gamma_prior_var)
        )
        return F, (Eb, Pb, Sigma)

    f = jax.jit(lambda g: free_energy(g)[0])
    grad = jax.jit(jax.grad(f))
    hess = jax.jit(jax.hessian(f))
    gamma = jnp.zeros(p)
    F = f(gamma)
    for _ in range(n_iter):
        H = hess(gamma)
        H = H - 1e-6 * jnp.eye(p)
        # Newton step on a concave surrogate: use −|H| so the step ascends even where H is indefinite
        wH, VH = jnp.linalg.eigh(H)
        step = VH @ ((VH.T @ grad(gamma)) / jnp.maximum(jnp.abs(wH), 1e-8))
        v = 1.0
        for _ in range(10):
            g_new = gamma + v * step
            F_new = f(g_new)
            if bool(jnp.isfinite(F_new)) and bool(F_new >= F):
                break
            v *= 0.5
        else:
            break
        if float(F_new - F) < 1e-6:
            gamma, F = g_new, F_new
            break
        gamma, F = g_new, F_new
    F, (Eb, Pb, Sigma) = free_energy(gamma)
    return PEBResult(Eb.reshape(c, p), jnp.linalg.inv(Pb), Sigma, gamma, F, bE, bC, X, n_floored, m, Lam)


def bmr_effects(res: PEBResult, var_reduced: float = 1e-8) -> np.ndarray:
    """F(full) − F(effect pinned to zero) for every covariate × parameter effect (n_cov, n_par).

    Positive values favour keeping the effect; ~3 nats ≈ posterior probability 0.95.
    """
    c, p = res.Eb.shape
    mu = res.Eb.reshape(-1)
    out = np.zeros((c, p))
    for k in range(c * p):
        Sr = res.bC.at[k, k].set(var_reduced)
        out.flat[k] = -float(log_evidence_reduce(res.bE, res.bC, mu, res.Cb, res.bE, Sr))
    return out


__all__ = ["PEBResult", "bmr_effects", "log_evidence_reduce", "peb"]
