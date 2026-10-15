"""Model evaluation: Bayes factors, fixed- and random-effects BMS, BMR, BMA.

References: Stephan et al. 2009 (random-effects BMS), Rigoux et al. 2014
(protected exceedance probabilities and the Bayesian omnibus risk),
Friston et al. 2016 (Bayesian model reduction), Penny et al. 2010 (BMA).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from scipy.special import digamma, gammaln


def log_bayes_factors(free_energies, reference=0):
    """``F_i - F_reference`` for a 1-D array of free energies."""
    F = np.asarray(free_energies, dtype=float)
    return F - F[reference]


def posterior_model_probabilities(free_energies):
    """Softmax of free energies (flat prior over models); a non-finite F gets probability zero."""
    F = np.asarray(free_energies, dtype=float)
    F = np.where(np.isfinite(F), F, -np.inf)
    if not np.isfinite(F).any():
        return np.full(F.shape, np.nan)
    z = F - F[np.isfinite(F)].max()
    p = np.exp(z)
    return p / p.sum()


def fixed_effects_bms(free_energies):
    """Group fixed effects: sum ``F`` over subjects (rows), then posterior probabilities over models (columns)."""
    F = np.asarray(free_energies, dtype=float)
    if F.ndim == 1:
        F = F[None, :]
    total = F.sum(axis=0)
    return total, posterior_model_probabilities(total)


class RFXResult(NamedTuple):
    alpha: np.ndarray                  # Dirichlet posterior
    expected_frequencies: np.ndarray   # r_k = alpha_k / sum alpha
    exceedance: np.ndarray             # phi_k
    protected_exceedance: np.ndarray   # phi_k (1 - BOR) + BOR / K
    bor: float                         # Bayesian omnibus risk p(H0 | y)
    posterior_responsibilities: np.ndarray  # g_nk


def _vb_dirichlet(F, alpha0, max_iter, tol):
    n, k = F.shape
    alpha = np.full(k, alpha0, dtype=float)
    for _ in range(max_iter):
        log_u = F + (digamma(alpha) - digamma(alpha.sum()))[None, :]
        log_u -= log_u.max(axis=1, keepdims=True)
        g = np.exp(log_u)
        g /= g.sum(axis=1, keepdims=True)
        beta = g.sum(axis=0)
        new = alpha0 + beta
        if np.max(np.abs(new - alpha)) < tol:
            alpha = new
            break
        alpha = new
    return alpha, g


def _dirichlet_vb_free_energy(F, alpha, g, alpha0):
    """Lower bound on the log evidence of the RFX model (Rigoux et al. 2014, Eq. A)."""
    n, k = F.shape
    a0 = np.full(k, alpha0)
    e_log_r = digamma(alpha) - digamma(alpha.sum())
    term = np.sum(g * (F + e_log_r[None, :] - np.log(np.maximum(g, 1e-300))))
    kl_dir = (gammaln(alpha.sum()) - np.sum(gammaln(alpha)) - gammaln(a0.sum()) + np.sum(gammaln(a0))
              + np.sum((alpha - a0) * e_log_r))
    return term - kl_dir


def random_effects_bms(free_energies, alpha0=1.0, n_samples=100_000, seed=0, max_iter=1000, tol=1e-8) -> RFXResult:
    """Random-effects BMS over a ``(n_subjects, n_models)`` free-energy matrix."""
    F = np.asarray(free_energies, dtype=float)
    if F.ndim != 2:
        raise ValueError("free_energies must be (n_subjects, n_models)")
    n, k = F.shape
    alpha, g = _vb_dirichlet(F, alpha0, max_iter, tol)
    r = alpha / alpha.sum()
    rng = np.random.default_rng(seed)
    draws = rng.dirichlet(alpha, size=n_samples)
    phi = np.bincount(draws.argmax(axis=1), minlength=k) / n_samples
    # Bayesian omnibus risk: evidence for H0 (all models equally frequent) vs H1 (RFX).
    f1 = _dirichlet_vb_free_energy(F, alpha, g, alpha0)
    f0 = np.sum(np.log(np.mean(np.exp(F - F.max(axis=1, keepdims=True)), axis=1)) + F.max(axis=1))
    bor = 1.0 / (1.0 + np.exp(f1 - f0))
    protected = phi * (1.0 - bor) + bor / k
    return RFXResult(alpha=alpha, expected_frequencies=r, exceedance=phi,
                     protected_exceedance=protected, bor=float(bor), posterior_responsibilities=g)


def bayesian_model_reduction(result, reduced_prior_mean, reduced_prior_cov):
    """``ln p(y | reduced prior) - ln p(y | full prior)`` and the reduced posterior.

    Analytic for Gaussian prior/posterior (Friston et al. 2016).  *result*
    is a :class:`~neurojax.bsd.inference.VLResult`; the reduced prior is on
    the same vector (set a parameter's prior mean and a tiny variance to
    switch it off).  Returns ``(delta_F, reduced_mean, reduced_cov)``.
    """
    mq, Sq = np.asarray(result.mean, float), np.asarray(result.cov, float)
    mp, Sp = np.asarray(result.prior_mean, float), np.asarray(result.prior_cov, float)
    mr, Sr = np.asarray(reduced_prior_mean, float), np.asarray(reduced_prior_cov, float)
    Pq, Pp, Pr = np.linalg.inv(Sq), np.linalg.inv(Sp), np.linalg.inv(Sr)
    Pe = Pq + Pr - Pp
    Se = np.linalg.inv(Pe)
    h = Pq @ mq + Pr @ mr - Pp @ mp
    me = Se @ h
    c = mq @ Pq @ mq + mr @ Pr @ mr - mp @ Pp @ mp
    ld = lambda P: np.linalg.slogdet(P)[1]
    dF = 0.5 * (ld(Pq) + ld(Pr) - ld(Pp) - ld(Pe)) + 0.5 * (h @ Se @ h - c)
    return float(dF), me, Se


def bayesian_model_average(results, free_energies=None, names=None):
    """Posterior-probability-weighted mean and covariance of named parameters shared across models.

    *results* are VLResults over models for one spectrum; parameters are
    matched by name, and a model lacking a name contributes nothing to it.
    Returns ``(names, mean, variance)`` with the between-model spread folded
    into the variance.
    """
    F = np.asarray([r.free_energy for r in results] if free_energies is None else free_energies, float)
    p = posterior_model_probabilities(F)
    if names is None:
        names = sorted({n for r in results for n in r.names})
    mean, var = np.zeros(len(names)), np.zeros(len(names))
    for j, name in enumerate(names):
        w, m, v = [], [], []
        for pk, r in zip(p, results):
            if name in r.names:
                i = r.names.index(name)
                w.append(pk); m.append(float(r.mean[i])); v.append(float(r.cov[i, i]))
        w = np.asarray(w) / max(sum(w), 1e-300)
        m, v = np.asarray(m), np.asarray(v)
        mean[j] = np.sum(w * m)
        var[j] = np.sum(w * (v + (m - mean[j]) ** 2))
    return list(names), mean, var
