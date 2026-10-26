"""Tests for neurojax.inverse.peb -- parametric empirical Bayes over first-level
Laplace posteriors (Friston et al. 2016) with Bayesian model reduction."""

import jax
import jax.numpy as jnp
import numpy as np

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


class TestPEBPairedDesign:
    def test_subject_columns_plus_condition_effects_with_unidentified_directions(self):
        """20 subjects x 4 conditions; subject indicator columns + 3 condition columns.
        First-level posteriors are correlated and broader than the prior in one
        direction (so information eigenvalues get clipped); only condition 2
        shifts parameter 0."""
        from neurojax.inverse.peb import bmr_effects, peb

        rng = np.random.default_rng(7)
        n_sub, n_cond, p = 20, 4, 3
        pE, pC = np.zeros(p), np.eye(p) / 16
        subj = 0.15 * rng.normal(size=(n_sub, p))
        rows, Ep, Cp = [], [], []
        for s in range(n_sub):
            for c in range(n_cond):
                theta = subj[s].copy()
                if c == 2:
                    theta[0] += 0.4
                # posterior covariance: tight on params 0,1 (correlated), as broad as the prior on param 2
                R = np.array([[1.0, 0.6, 0.0], [0.6, 1.0, 0.0], [0.0, 0.0, 1.0]])
                sd = np.array([0.04, 0.04, 0.3])
                C = R * np.outer(sd, sd)
                Ep.append(theta + rng.multivariate_normal(np.zeros(p), C))
                Cp.append(C)
                rows.append((s, c))
        X = np.column_stack([[1.0 if s == k else 0.0 for s, _ in rows] for k in range(n_sub)]
                            + [[1.0 if c == k else 0.0 for _, c in rows] for k in (1, 2, 3)])
        res = peb(jnp.asarray(np.array(Ep)), jnp.asarray(np.array(Cp)), jnp.asarray(pE), jnp.asarray(pC), jnp.asarray(X))
        assert res.n_clipped > 0
        eff = np.asarray(res.Eb)[n_sub:]  # (3 conditions, p)
        sd = np.sqrt(np.diag(np.asarray(res.Cb))).reshape(res.Eb.shape)[n_sub:]
        assert abs(eff[1, 0] - 0.4) < 0.05, eff
        other = np.ones_like(eff, dtype=bool)
        other[1, 0] = False
        assert np.all(np.abs(eff[other]) < 0.06), eff
        assert abs(eff[1, 2]) < 1e-6, eff  # unidentified parameter: no effect can be inferred
        dF = bmr_effects(res)[n_sub:]
        assert dF[1, 0] > 10, dF
        assert np.all(dF[other] < 3), dF


def test_shared_gamma_matches_per_parameter_on_homogeneous_data():
    """With one between-subject log-precision the fit is close to the per-parameter one when all parameters share it."""
    from neurojax.inverse.peb import peb

    rng = np.random.default_rng(11)
    n, p = 24, 4
    pE, pC = jnp.zeros(p), jnp.eye(p) * 0.25
    X = jnp.asarray(np.column_stack([np.ones(n), rng.normal(size=n)]))
    beta = np.array([[0.2, -0.1, 0.0, 0.3], [0.4, 0.0, 0.0, 0.0]])
    theta = np.asarray(X) @ beta + rng.normal(size=(n, p)) * 0.2
    Cp = jnp.asarray(np.tile(np.eye(p) * 0.01, (n, 1, 1)))
    Ep = jnp.asarray(theta + rng.normal(size=(n, p)) * 0.1)
    full = peb(Ep, Cp, pE, pC, X)
    shared = peb(Ep, Cp, pE, pC, X, shared_gamma=True)
    assert shared.gamma.shape == (1,)
    assert np.allclose(np.asarray(full.Eb), np.asarray(shared.Eb), atol=0.05)
    assert np.allclose(np.sqrt(np.diag(np.asarray(shared.Sigma))), 0.2, atol=0.1)
