"""Variational Laplace inversion of a spectral model.

The parameter vector is ``phi = [theta, log_precision]`` with Gaussian
priors on both (the log-precision hyperprior is centred on the residual
precision of a least-squares pre-fit, SD 1, which plays the role of DCM's
hyperprior).  The posterior mode is found by BFGS on the negative log
joint, the posterior covariance is the inverse negative Hessian at the mode
(Laplace), and the free energy is the Laplace approximation to the log
model evidence,

``F = ln p(y|phi*) + ln p(phi*) + (d/2) ln 2 pi + (1/2) ln |Sigma_q|``,

which is the variational free energy of a Gaussian ``q`` at convergence.
It is also reported as ``accuracy - complexity`` with the complexity the
KL divergence from prior to posterior, the decomposition DCM reports.

With ``likelihood="whittle"`` there is no precision hyperparameter and the
Welch segment count sets the noise.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from neurojax.bsd.likelihood import gaussian_log_likelihood, whittle_log_likelihood


class VLResult(NamedTuple):
    model: object
    names: tuple[str, ...]
    mean: jnp.ndarray          # posterior mean of phi
    cov: jnp.ndarray           # posterior covariance of phi
    prior_mean: jnp.ndarray
    prior_cov: jnp.ndarray
    free_energy: float
    log_likelihood: float
    accuracy: float
    complexity: float
    log_psd_fit: jnp.ndarray
    residual: jnp.ndarray
    converged: bool
    n_obs: int

    @property
    def theta(self):
        return self.mean[: self.model.n_params]

    @property
    def theta_cov(self):
        n = self.model.n_params
        return self.cov[:n, :n]

    def natural(self) -> dict:
        return self.model.natural(self.theta)


def _log_prior(phi, mu, sd):
    z = (phi - mu) / sd
    return -0.5 * jnp.sum(z ** 2) - jnp.sum(jnp.log(sd)) - 0.5 * phi.shape[0] * jnp.log(2.0 * jnp.pi)


def _kl_gaussians(mq, Sq, mp, Sp):
    """KL(q || p) for Gaussians (posterior from prior)."""
    d = mq.shape[0]
    Sp_inv = jnp.linalg.inv(Sp)
    diff = mq - mp
    _, ld_p = jnp.linalg.slogdet(Sp)
    _, ld_q = jnp.linalg.slogdet(Sq)
    return 0.5 * (jnp.trace(Sp_inv @ Sq) + diff @ Sp_inv @ diff - d + ld_p - ld_q)


def variational_laplace(model, freqs, log_psd, likelihood="gaussian", n_segments=None,
                        init=None, max_steps=2000, rtol=1e-6, atol=1e-8, restarts=0,
                        restart_scale=0.5, seed=0, hyper_sd=1.0, jitter=1e-8, prefit=True) -> VLResult:
    """Invert *model* on one log spectrum; see the module docstring."""
    f = jnp.asarray(freqs, dtype=float)
    y = jnp.asarray(log_psd, dtype=float)
    mu_t, sd_t = model.prior()
    n_t = model.n_params
    use_whittle = likelihood == "whittle"
    if use_whittle and n_segments is None:
        raise ValueError("whittle likelihood needs n_segments")
    if likelihood not in ("gaussian", "whittle"):
        raise ValueError("likelihood must be 'gaussian' or 'whittle'")

    # Hyperprior on the log precision from a quick least-squares pre-fit.
    def sse(theta, args):
        return jnp.sum((model.log_psd(f, theta) - y) ** 2)

    theta0 = mu_t if init is None else jnp.asarray(init, dtype=float)
    if prefit:
        pre = optx.minimise(sse, optx.BFGS(rtol=1e-4, atol=1e-6), theta0, max_steps=500, throw=False)
        pre_value = pre.value
    else:
        # mechanistic models with tight log-scaling priors: an unconstrained pre-fit wanders
        # tens of prior SDs away, so start the MAP search at the prior mean instead
        pre_value = theta0
    resid_var = jnp.maximum(sse(pre_value, None) / f.shape[0], 1e-6)
    lam0 = -jnp.log(resid_var)

    if use_whittle:
        mu, sd = mu_t, sd_t
    else:
        mu = jnp.concatenate([mu_t, jnp.array([lam0])])
        sd = jnp.concatenate([sd_t, jnp.array([hyper_sd])])

    def log_lik(phi):
        ls = model.log_psd(f, phi[:n_t])
        if use_whittle:
            return whittle_log_likelihood(ls, y, n_segments)
        return gaussian_log_likelihood(ls, y, phi[n_t])

    def neg_log_joint(phi, args):
        return -(log_lik(phi) + _log_prior(phi, mu, sd))

    def solve(start):
        sol = optx.minimise(neg_log_joint, optx.BFGS(rtol=rtol, atol=atol), start,
                            max_steps=max_steps, throw=False)
        return sol.value, -neg_log_joint(sol.value, None), sol.result == optx.RESULTS.successful

    start = jnp.concatenate([pre_value, jnp.array([lam0])]) if not use_whittle else pre_value
    if not bool(jnp.isfinite(start).all()):
        start = mu  # the pre-fit diverged; start from the prior mean
    best = solve(start)
    if not bool(jnp.isfinite(best[1])):
        best = solve(mu)
    if restarts:
        key = jax.random.PRNGKey(seed)
        for k in jax.random.split(key, restarts):
            cand = solve(start + restart_scale * sd * jax.random.normal(k, start.shape))
            if bool(jnp.isfinite(cand[1])) and (not bool(jnp.isfinite(best[1])) or cand[1] > best[1]):
                best = cand
    phi, lj, ok = best
    if not bool(jnp.isfinite(lj)):
        # Every start diverged: report a model that cannot explain the data rather than NaN.
        phi, lj, ok = mu, -jnp.inf, False

    H = jax.hessian(lambda p: neg_log_joint(p, None))(phi)
    H = jnp.where(jnp.isfinite(H), H, 0.0)
    H = 0.5 * (H + H.T)
    w, V = jnp.linalg.eigh(H)
    w = jnp.maximum(w, jitter)
    cov = (V * (1.0 / w)) @ V.T
    d = phi.shape[0]
    _, ld = jnp.linalg.slogdet(cov)
    F = lj + 0.5 * d * jnp.log(2.0 * jnp.pi) + 0.5 * ld
    F = jnp.where(jnp.isfinite(F), F, -jnp.inf)
    prior_cov = jnp.diag(sd ** 2)
    ll = log_lik(phi)
    kl = _kl_gaussians(phi, cov, mu, prior_cov)
    fit = model.log_psd(f, phi[:n_t])
    return VLResult(model=model, names=tuple(model.names) + (() if use_whittle else ("log_precision",)),
                    mean=phi, cov=cov, prior_mean=mu, prior_cov=prior_cov,
                    free_energy=float(F), log_likelihood=float(ll), accuracy=float(ll),
                    complexity=float(kl), log_psd_fit=fit, residual=y - fit,
                    converged=bool(ok), n_obs=int(f.shape[0]))
