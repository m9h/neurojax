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
by Newton steps on the free energy.  (The ``m_i`` are never formed.)

``Lambda_i`` is positive semi-definite only where the first-level
posterior is tighter than its prior; it is never inverted here (the
second-level likelihood is evaluated through the reduction identity, as
``spm_dcm_peb`` does), and its negative eigenvalues are clipped to zero.
Parameters the first level did not learn about simply carry no
second-level information.

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
    n_clipped: int  # first-level information eigen-directions treated as uninformed (set to zero)


def peb(
    Ep: jnp.ndarray,
    Cp: jnp.ndarray,
    pE: jnp.ndarray,
    pC: jnp.ndarray,
    X: jnp.ndarray,
    gamma_prior_var: float = 4.0,
    n_iter: int = 32,
    shared_gamma: bool = False,
    cov_floor: float = 1e-6,
    info_threshold: float = 0.1,
) -> PEBResult:
    """Fit the second-level model.  ``Ep`` (n_sub, n_par), ``Cp`` (n_sub, n_par, n_par), ``X`` (n_sub, n_cov).

    The second-level likelihood of subject ``i`` as a function of its
    empirical prior ``N(X_i beta, Sigma)`` is the Bayesian-model-reduction
    identity (the evidence under that prior relative to the first-level
    prior), which is quadratic in ``beta``; so ``beta`` has a closed-form
    Gaussian posterior for every ``gamma`` and only ``gamma`` is iterated.
    The first-level information ``Lambda_i = Cp_i⁻¹ − pC⁻¹`` is never
    inverted; its negative eigenvalues (directions where the Laplace
    posterior is broader than the prior) are clipped to zero, and the number
    clipped is reported.  The prior on each row of ``beta`` is ``N(0, pC)``
    and ``gamma ~ N(0, gamma_prior_var)`` with ``Sigma = diag(pC) exp(−gamma)``;
    ``gamma`` has one entry per parameter, or a single entry with
    ``shared_gamma`` (one covariance component scaled by one log-precision,
    the ``spm_dcm_peb`` default, and far cheaper for many parameters: the
    Newton step needs a Hessian in ``gamma``).
    """
    Ep, Cp, pE, pC, X = (jnp.asarray(a, dtype=jnp.float64) for a in (Ep, Cp, pE, pC, X))
    n, p = Ep.shape
    c = X.shape[1]
    P0 = jnp.linalg.inv(pC)
    # first-level covariances saved in single precision (or from a Laplace step that
    # stopped early) can be slightly indefinite; symmetrise and floor their spectrum
    # at cov_floor × the largest eigenvalue before inverting
    Cp = 0.5 * (Cp + jnp.swapaxes(Cp, 1, 2))
    wC, VC = jnp.linalg.eigh(Cp)
    wC = jnp.maximum(wC, cov_floor * wC.max(axis=1, keepdims=True))
    Cp = jnp.einsum("nij,nj,nkj->nik", VC, wC, VC)
    Pp = jnp.linalg.inv(Cp)  # (n, p, p)
    w, V = jnp.linalg.eigh(Pp - P0[None])
    # a direction counts as informed only if the first level gained at least
    # `info_threshold` of the prior precision there: in a barely-informed
    # direction the posterior mean differs from the prior mean by optimisation
    # noise, and the implied data location Λ⁻¹h = O(noise / Λ) is huge, which
    # drives the between-subject variance to infinity
    prior_prec_dir = jnp.einsum("nik,ij,njk->nk", V, P0, V)  # vₖᵀ P0 vₖ per direction k
    keep = w > info_threshold * prior_prec_dir
    n_clipped = int(jnp.sum(~keep))
    Lam = jnp.einsum("nij,nj,nkj->nik", V, jnp.where(keep, w, 0.0), V)  # first-level information, PSD
    h = jnp.einsum("nij,nj->ni", Pp, Ep) - (P0 @ pE)[None]  # Pp Ep − P0 pE
    # the linear term must live in the retained directions: a direction with no
    # information cannot carry a location either
    h = jnp.einsum("nij,nj,nkj,nk->ni", V, keep.astype(h.dtype), V, h)
    # per-subject constant part of the reduction identity (independent of the empirical prior mean)
    const = 0.5 * jax.vmap(_logdet)(Cp * 0 + pC[None]) - 0.5 * jax.vmap(_logdet)(Cp) - 0.5 * (
        jnp.einsum("ni,nij,nj->n", Ep, Pp, Ep) - (pE @ P0 @ pE)
    )
    bE = jnp.zeros(c * p)
    bC = jnp.kron(jnp.eye(c), pC)
    bP = jnp.linalg.inv(bC)
    # mu_r,i = A_i vec(beta), vec row-major over (cov, par)
    A = jnp.einsum("nc,kq->nkcq", X, jnp.eye(p)).reshape(n, p, c * p)
    sig_base = jnp.diag(pC)

    n_gamma = 1 if shared_gamma else p

    def free_energy(gamma):
        Sigma = jnp.diag(sig_base * jnp.exp(-(gamma if not shared_gamma else jnp.repeat(gamma, p))))
        Pr = jnp.linalg.inv(Sigma)
        Q = Pr[None] + Lam  # (n, p, p)
        Qinv = jnp.linalg.inv(Q)
        # log evidence of subject i under prior N(mu_r, Sigma):
        #   const_i − ½ log|Sigma| − ½ log|Q| + ½ (Pr mu_r + h)ᵀ Q⁻¹ (Pr mu_r + h) − ½ mu_rᵀ Pr mu_r
        # quadratic in mu_r: precision  Pr − Pr Q⁻¹ Pr,  linear term  Pr Q⁻¹ h
        Prec = Pr[None] - jnp.einsum("ij,njk,kl->nil", Pr, Qinv, Pr)
        lin = jnp.einsum("ij,njk,nk->ni", Pr, Qinv, h)
        c0 = const - 0.5 * _logdet(Sigma) - 0.5 * jax.vmap(_logdet)(Q) + 0.5 * jnp.einsum("ni,nij,nj->n", h, Qinv, h)
        Pb = bP + jnp.einsum("nkq,nkl,nlr->qr", A, Prec, A)
        g = jnp.einsum("nkq,nk->q", A, lin)
        Eb = jnp.linalg.solve(Pb, g)
        F = (
            jnp.sum(c0)
            - 0.5 * _logdet(bC)
            - 0.5 * _logdet(Pb)
            + 0.5 * Eb @ Pb @ Eb
            - 0.5 * gamma @ gamma / gamma_prior_var
            - 0.5 * n_gamma * jnp.log(2 * jnp.pi * gamma_prior_var)
        )
        return F, (Eb, Pb, Sigma)

    f = jax.jit(lambda g: free_energy(g)[0])
    grad = jax.jit(jax.grad(f))
    hess = jax.jit(jax.hessian(f))
    gamma = jnp.zeros(n_gamma)
    F = f(gamma)
    for _ in range(n_iter):
        H = hess(gamma)
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
        done = float(F_new - F) < 1e-6
        gamma, F = g_new, F_new
        if done:
            break
    F, (Eb, Pb, Sigma) = free_energy(gamma)
    return PEBResult(Eb.reshape(c, p), jnp.linalg.inv(Pb), Sigma, gamma, F, bE, bC, X, n_clipped)


def bmr_effects(res: PEBResult, var_reduced: float = 1e-8, covariates: list[int] | None = None) -> np.ndarray:
    """F(full) − F(effect pinned to zero) for every covariate × parameter effect (n_cov, n_par).

    Positive values favour keeping the effect; ~3 nats ≈ posterior probability 0.95.
    ``covariates`` restricts the rows evaluated (others are NaN); each effect is
    one reduction of a ``(n_cov n_par)``-dimensional posterior.
    """
    c, p = res.Eb.shape
    mu = res.Eb.reshape(-1)
    out = np.full((c, p), np.nan)
    for ci in range(c) if covariates is None else covariates:
        for k in range(ci * p, (ci + 1) * p):
            Sr = res.bC.at[k, k].set(var_reduced)
            out.flat[k] = -float(log_evidence_reduce(res.bE, res.bC, mu, res.Cb, res.bE, Sr))
    return out


__all__ = ["PEBResult", "bmr_effects", "log_evidence_reduce", "peb"]
