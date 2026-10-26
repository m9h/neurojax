"""Tests for neurojax.inverse.peb -- parametric empirical Bayes over first-level
Laplace posteriors (Friston et al. 2016) with Bayesian model reduction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _first_level(rng, n_sub, n_par, beta, X, sd_between=0.1, sd_post=0.05):
    """Synthetic first-level posteriors: theta_i ~ N(X_i beta, sd_between²), observed with posterior sd."""
    theta = X @ beta + sd_between * rng.normal(size=(n_sub, n_par))
    Ep = theta + sd_post * rng.normal(size=(n_sub, n_par))
    Cp = np.stack([np.eye(n_par) * sd_post**2 for _ in range(n_sub)])
    return jnp.asarray(Ep), jnp.asarray(Cp)


class TestPEB:
    def test_recovers_group_mean_and_covariate_effect(self):
        from neurojax.inverse.peb import peb

        rng = np.random.default_rng(0)
        n_sub, n_par = 24, 3
        X = np.column_stack([np.ones(n_sub), np.repeat([0.0, 1.0], n_sub // 2)])  # intercept, drug
        beta = np.array([[0.1, -0.2, 0.0], [0.4, 0.0, 0.0]])  # drug raises parameter 0 by 0.4
        Ep, Cp = _first_level(rng, n_sub, n_par, beta, X)
        pE, pC = jnp.zeros(n_par), jnp.eye(n_par) / 16
        res = peb(Ep, Cp, pE, pC, jnp.asarray(X))
        assert res.Eb.shape == (2, n_par) and res.Cb.shape == (2 * n_par, 2 * n_par)
        np.testing.assert_allclose(np.asarray(res.Eb), beta, atol=0.08)
        sd = np.sqrt(np.diag(np.asarray(res.Cb))).reshape(2, n_par)
        assert abs(float(res.Eb[1, 0]) - 0.4) < 3 * sd[1, 0] + 0.02
        assert np.isfinite(float(res.F))

    def test_bmr_prunes_null_effect_and_keeps_real_one(self):
        from neurojax.inverse.peb import bmr_effects, peb

        rng = np.random.default_rng(1)
        n_sub, n_par = 24, 2
        X = np.column_stack([np.ones(n_sub), np.repeat([0.0, 1.0], n_sub // 2)])
        beta = np.array([[0.0, 0.0], [0.5, 0.0]])  # drug affects parameter 0 only
        Ep, Cp = _first_level(rng, n_sub, n_par, beta, X)
        res = peb(Ep, Cp, jnp.zeros(n_par), jnp.eye(n_par) / 16, jnp.asarray(X))
        dF = bmr_effects(res)  # (n_cov, n_par): F(full) − F(effect removed); > 0 means the effect is needed
        assert float(dF[1, 0]) > 3.0, dF
        assert float(dF[1, 1]) < 3.0, dF

    def test_between_subject_variance_is_estimated(self):
        from neurojax.inverse.peb import peb

        rng = np.random.default_rng(2)
        X = np.ones((30, 1))
        beta = np.zeros((1, 2))
        for sd_b in (0.05, 0.3):
            Ep, Cp = _first_level(rng, 30, 2, beta, X, sd_between=sd_b, sd_post=0.02)
            res = peb(Ep, Cp, jnp.zeros(2), jnp.eye(2) / 16, jnp.asarray(X))
            est = float(jnp.sqrt(jnp.mean(jnp.diag(res.Sigma))))
            assert abs(est - sd_b) / sd_b < 0.5, (sd_b, est)
