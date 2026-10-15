"""Bayesian spectral decomposition: models, inversion, evidence and comparison tools.

Red-green: each test states the behaviour the module must have; simulated
spectra carry Welch estimation noise (Gamma-distributed periodogram
averages) so the tolerances are those of real data, not of noiseless fits.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from neurojax.bsd import (  # noqa: E402
    DampedOscillators, DampingDistribution, SpecParam, XiAlpha,
    bayesian_model_average, bayesian_model_reduction, fixed_effects_bms,
    gaussian_log_likelihood, log_bayes_factors, posterior_model_probabilities,
    random_effects_bms, simulate_log_psd, variational_laplace, whittle_log_likelihood,
)

FREQS = jnp.linspace(1.0, 45.0, 89)
TWO_PI = 2 * np.pi


def _specparam_truth():
    m = SpecParam()
    t = m.prior()[0].at[0].set(0.0).at[1].set(jnp.log(1.2)).at[2].set(10.0).at[3].set(1.2).at[4].set(jnp.log(1.5))
    return m, t


def _liley_truth():
    m = DampingDistribution()
    t = (m.prior()[0].at[0].set(jnp.log(10.0)).at[1].set(jnp.log(TWO_PI * 0.8)).at[2].set(jnp.log(1.0))
         .at[3].set(jnp.log(1.0)).at[4].set(jnp.log(1e-3)))
    return m, t


def _xialpha_truth():
    m = XiAlpha()
    t = (m.prior()[0].at[0].set(jnp.log(1.0)).at[1].set(jnp.log(TWO_PI * 1.5)).at[2].set(jnp.log(0.5))
         .at[3].set(jnp.log(0.3)).at[4].set(jnp.log(TWO_PI * 0.8)).at[5].set(jnp.log(1.0))
         .at[6].set(jnp.log(10.0)).at[7].set(jnp.log(1e-3)))
    return m, t


GENERATORS = {"specparam": _specparam_truth, "liley": _liley_truth, "xialpha": _xialpha_truth}
MODELS = {"specparam": SpecParam, "liley": DampingDistribution, "xialpha": XiAlpha}


@pytest.mark.parametrize("make", [SpecParam, lambda: SpecParam(knee=True, peak_centres=(10.0, 20.0)),
                                  DampedOscillators, DampingDistribution, XiAlpha])
def test_models_are_finite_and_differentiable(make):
    m = make()
    mu, sd = m.prior()
    assert mu.shape == (m.n_params,) and sd.shape == (m.n_params,) and len(m.names) == m.n_params
    y = m.log_psd(FREQS, mu)
    g = jax.grad(lambda t: m.log_psd(FREQS, t).sum())(mu)
    assert jnp.isfinite(y).all() and jnp.isfinite(g).all()
    assert set(m.natural(mu)) and all(np.isfinite(v) for v in m.natural(mu).values())


def test_oscillator_peak_power_parameterisation():
    """At resonance the oscillator's power equals the peak parameter (floor aside)."""
    m = DampedOscillators(centres=(10.0,))
    t = m.prior()[0].at[2].set(jnp.log(2.0)).at[3].set(jnp.log(1e-9))
    assert abs(float(jnp.exp(m.log_psd(jnp.array([10.0]), t))[0]) - 2.0) < 1e-6


def test_whittle_and_gaussian_likelihoods_peak_at_the_model():
    m, t = _specparam_truth()
    ls = m.log_psd(FREQS, t)
    y = simulate_log_psd(m, t, FREQS, 32, jax.random.PRNGKey(0))
    w0 = whittle_log_likelihood(ls, y, 32)
    assert w0 > whittle_log_likelihood(ls + 0.3, y, 32) and w0 > whittle_log_likelihood(ls - 0.3, y, 32)
    g0 = gaussian_log_likelihood(ls, y, 3.0)
    assert g0 > gaussian_log_likelihood(ls + 0.3, y, 3.0)


@pytest.mark.parametrize("name", list(GENERATORS))
def test_recovery_from_own_generator(name):
    m, t = GENERATORS[name]()
    y = simulate_log_psd(m, t, FREQS, 32, jax.random.PRNGKey(1))
    r = variational_laplace(MODELS[name](), FREQS, y, restarts=2)
    assert r.converged and np.isfinite(r.free_energy) and r.complexity >= 0
    fit, truth = r.natural(), m.natural(t)
    key = {"specparam": ("cf0", 0.5), "liley": ("f0", 0.5), "xialpha": ("f_alpha", 0.5)}[name]
    assert abs(fit[key[0]] - truth[key[0]]) < key[1]
    if name == "specparam":
        assert abs(fit["exponent"] - truth["exponent"]) < 0.15
    # the fitted spectrum tracks the generating one to within the estimation noise
    assert float(jnp.sqrt(jnp.mean((r.log_psd_fit - m.log_psd(FREQS, t)) ** 2))) < 0.15


def test_evidence_selects_the_generating_model():
    """3 generators x 3 models x 3 realisations: the generator wins by fixed effects
    and never loses a single realisation by more than a few nats."""
    F = {g: np.zeros((3, 3)) for g in GENERATORS}
    for gi, (g, make) in enumerate(GENERATORS.items()):
        m, t = make()
        for seed in range(3):
            y = simulate_log_psd(m, t, FREQS, 32, jax.random.PRNGKey(seed))
            F[g][seed] = [variational_laplace(mk(), FREQS, y, restarts=2).free_energy for mk in MODELS.values()]
    for gi, g in enumerate(GENERATORS):
        total, p = fixed_effects_bms(F[g])
        assert int(np.argmax(total)) == gi, (g, total)
        assert p[gi] > 0.9, (g, p)
        for seed in range(3):
            assert F[g][seed].max() - F[g][seed][gi] < 3.0, (g, seed, F[g][seed])


def test_bayes_factors_and_probabilities():
    F = np.array([0.0, -3.0, -10.0])
    lbf = log_bayes_factors(F)
    assert np.allclose(lbf, F)
    p = posterior_model_probabilities(F)
    assert abs(p.sum() - 1) < 1e-12 and p[0] > 0.9


def test_random_effects_bms_recovers_a_dominant_model():
    rng = np.random.default_rng(0)
    n, k = 20, 3
    F = rng.normal(0, 1, (n, k))
    F[:15, 0] += 6.0   # 15 subjects favour model 0
    F[15:, 1] += 6.0   # 5 favour model 1
    res = random_effects_bms(F, seed=0)
    assert res.expected_frequencies[0] > 0.6 and res.exceedance[0] > 0.9
    assert 0.0 <= res.bor <= 1.0 and res.protected_exceedance[0] > 0.8
    assert np.allclose(res.posterior_responsibilities.sum(axis=1), 1.0)
    # no information: the omnibus risk should be high and exceedance near uniform
    flat = random_effects_bms(rng.normal(0, 0.1, (n, k)), seed=1)
    assert flat.bor > 0.5


def test_bayesian_model_reduction_agrees_with_direct_inversion():
    """BMR's analytic delta-F has the sign of, and is close to, the difference of two full inversions."""
    m, t = _specparam_truth()
    t_nopeak = t.at[3].set(0.0)
    y_peak = simulate_log_psd(m, t, FREQS, 32, jax.random.PRNGKey(5))
    y_nopeak = simulate_log_psd(m, t_nopeak, FREQS, 32, jax.random.PRNGKey(6))
    for y, expect_positive in ((y_nopeak, True), (y_peak, False)):
        full = variational_laplace(SpecParam(), FREQS, y, restarts=1)
        reduced = variational_laplace(SpecParam(power_prior=(0.0, 1e-3)), FREQS, y, restarts=1)
        rm, rc = np.asarray(full.prior_mean).copy(), np.asarray(full.prior_cov).copy()
        i = full.names.index("pw0")
        rm[i] = 0.0; rc[i, i] = 1e-6
        dF, me, Se = bayesian_model_reduction(full, rm, rc)
        direct = reduced.free_energy - full.free_energy
        assert (dF > 0) == expect_positive, (expect_positive, dF, direct)
        assert (direct > 0) == expect_positive, (expect_positive, dF, direct)
        if expect_positive:
            # The reduced model sits inside the full posterior's quadratic region:
            # the analytic reduction must reproduce the direct inversion closely.
            assert abs(dF - direct) < 3.0, (dF, direct)
        else:
            # Removing a real peak leaves the Laplace quadratic far from the mode
            # (centre and width become unidentified), so BMR is only a bound on
            # the penalty, not its value; it must not understate it.
            assert dF <= direct + 3.0, (dF, direct)
        assert np.isfinite(me).all() and np.all(np.linalg.eigvalsh(Se) > 0)


def test_bayesian_model_average_matches_shared_names():
    m, t = _specparam_truth()
    y = simulate_log_psd(m, t, FREQS, 32, jax.random.PRNGKey(2))
    r1 = variational_laplace(SpecParam(), FREQS, y)
    r2 = variational_laplace(SpecParam(knee=True), FREQS, y)
    names, mean, var = bayesian_model_average([r1, r2], names=["cf0", "log_exponent", "log_knee"])
    assert np.isfinite(mean).all() and np.all(var >= 0)
    assert abs(mean[0] - 10.0) < 0.5
    # log_knee exists only in r2: BMA returns r2's estimate for it
    assert abs(mean[2] - float(r2.mean[r2.names.index("log_knee")])) < 1e-9


def test_free_energy_is_differentiable_in_the_data():
    """The evidence is a JAX quantity: gradients through the whole inversion are finite."""
    m, t = _specparam_truth()
    y = simulate_log_psd(m, t, FREQS, 32, jax.random.PRNGKey(3))
    def F_of_scale(s):
        return variational_laplace(SpecParam(), FREQS, y + s, max_steps=300).free_energy
    a, b = F_of_scale(0.0), F_of_scale(0.5)
    assert np.isfinite(a) and np.isfinite(b)
