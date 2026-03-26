"""Tests for GaussianHMM — pure JAX HMM with Baum-Welch EM."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neurojax.models.hmm import (
    GaussianHMM,
    HMMConfig,
    _log_emission_matrix,
    _log_mvn_pdf,
    _viterbi,
    backward,
    e_step,
    forward,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_data():
    """Synthetic 2-state, 3-channel data with clear cluster structure."""
    key = jr.PRNGKey(42)
    k1, k2 = jr.split(key)
    # State 0: mean at [2, 0, 0]
    d0 = jr.normal(k1, (200, 3)) * 0.3 + jnp.array([2.0, 0.0, 0.0])
    # State 1: mean at [-2, 0, 0]
    d1 = jr.normal(k2, (200, 3)) * 0.3 + jnp.array([-2.0, 0.0, 0.0])
    # Alternate every 200 steps
    return [jnp.concatenate([d0, d1], axis=0)]  # (400, 3)


# ---------------------------------------------------------------------------
# Log MVN PDF
# ---------------------------------------------------------------------------

class TestLogMvnPdf:
    def test_shape(self):
        x = jnp.zeros(3)
        means = jnp.zeros((2, 3))
        prec = jnp.stack([jnp.eye(3)] * 2)
        log_dets = jnp.zeros(2)
        result = _log_mvn_pdf(x, means, prec, log_dets)
        assert result.shape == (2,)

    def test_peak_at_mean(self):
        """Log-density should be highest at the mean."""
        means = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        prec = jnp.stack([jnp.eye(2)] * 2)
        log_dets = jnp.zeros(2)
        # Evaluate at mean of state 0
        at_mean = _log_mvn_pdf(jnp.array([1.0, 0.0]), means, prec, log_dets)
        away = _log_mvn_pdf(jnp.array([5.0, 5.0]), means, prec, log_dets)
        assert float(at_mean[0]) > float(away[0])

    def test_finite(self):
        x = jr.normal(jr.PRNGKey(0), (5,))
        means = jr.normal(jr.PRNGKey(1), (3, 5))
        covs = jnp.stack([jnp.eye(5)] * 3)
        prec = jnp.linalg.inv(covs)
        L = jnp.linalg.cholesky(covs)
        log_dets = 2 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
        result = _log_mvn_pdf(x, means, prec, log_dets)
        assert jnp.all(jnp.isfinite(result))


# ---------------------------------------------------------------------------
# Forward / Backward
# ---------------------------------------------------------------------------

class TestForwardBackward:
    def test_forward_shape(self):
        log_B = jnp.zeros((10, 3))
        log_trans = jnp.log(jnp.ones((3, 3)) / 3)
        log_pi = jnp.log(jnp.ones(3) / 3)
        log_alpha, log_ll = forward(log_B, log_trans, log_pi)
        assert log_alpha.shape == (10, 3)
        assert log_ll.shape == ()

    def test_backward_shape(self):
        log_B = jnp.zeros((10, 3))
        log_trans = jnp.log(jnp.ones((3, 3)) / 3)
        log_beta = backward(log_B, log_trans)
        assert log_beta.shape == (10, 3)

    def test_forward_log_likelihood_finite(self):
        log_B = jr.normal(jr.PRNGKey(0), (50, 4)) * 0.1
        log_trans = jnp.log(jnp.ones((4, 4)) / 4)
        log_pi = jnp.log(jnp.ones(4) / 4)
        _, log_ll = forward(log_B, log_trans, log_pi)
        assert jnp.isfinite(log_ll)


# ---------------------------------------------------------------------------
# E-step
# ---------------------------------------------------------------------------

class TestEStep:
    def test_gamma_sums_to_one(self):
        """Posterior gamma should sum to 1 across states at each timestep."""
        log_B = jr.normal(jr.PRNGKey(0), (20, 3))
        log_trans = jnp.log(jnp.ones((3, 3)) / 3)
        log_pi = jnp.log(jnp.ones(3) / 3)
        gamma, _, _ = e_step(log_B, log_trans, log_pi)
        row_sums = gamma.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_gamma_nonnegative(self):
        log_B = jr.normal(jr.PRNGKey(1), (20, 4))
        log_trans = jnp.log(jnp.ones((4, 4)) / 4)
        log_pi = jnp.log(jnp.ones(4) / 4)
        gamma, _, _ = e_step(log_B, log_trans, log_pi)
        assert jnp.all(gamma >= -1e-7)

    def test_xi_shape(self):
        log_B = jnp.zeros((10, 2))
        log_trans = jnp.log(jnp.ones((2, 2)) / 2)
        log_pi = jnp.log(jnp.ones(2) / 2)
        _, xi, _ = e_step(log_B, log_trans, log_pi)
        assert xi.shape == (9, 2, 2)


# ---------------------------------------------------------------------------
# Viterbi
# ---------------------------------------------------------------------------

class TestViterbi:
    def test_shape(self):
        log_B = jnp.zeros((10, 3))
        log_trans = jnp.log(jnp.ones((3, 3)) / 3)
        log_pi = jnp.log(jnp.ones(3) / 3)
        states = _viterbi(log_B, log_trans, log_pi)
        assert states.shape == (10,)

    def test_single_timestep(self):
        log_B = jnp.array([[0.0, -10.0, -10.0]])
        log_pi = jnp.log(jnp.ones(3) / 3)
        log_trans = jnp.log(jnp.ones((3, 3)) / 3)
        states = _viterbi(log_B, log_trans, log_pi)
        assert states.shape == (1,)

    def test_deterministic_sequence(self):
        """With very strong emissions, Viterbi should mostly recover the right states."""
        T = 20
        log_B = jnp.full((T, 2), -100.0)
        # First 10 steps: state 0, last 10: state 1
        log_B = log_B.at[:10, 0].set(0.0)
        log_B = log_B.at[10:, 1].set(0.0)
        log_trans = jnp.log(jnp.array([[0.9, 0.1], [0.1, 0.9]]))
        log_pi = jnp.log(jnp.array([0.5, 0.5]))
        states = _viterbi(log_B, log_trans, log_pi)
        # Allow 1-step transition lag due to transition cost
        assert jnp.all(states[:9] == 0)
        assert jnp.all(states[11:] == 1)


# ---------------------------------------------------------------------------
# GaussianHMM class
# ---------------------------------------------------------------------------

class TestGaussianHMM:
    def test_construction(self):
        model = GaussianHMM(n_states=4, n_channels=5)
        assert model.config.n_states == 4
        assert model.config.n_channels == 5

    def test_init_params(self, simple_data):
        model = GaussianHMM(n_states=2, n_channels=3)
        model.init_params(simple_data)
        assert model.means.shape == (2, 3)
        assert model.covariances.shape == (2, 3, 3)
        assert model.trans_prob.shape == (2, 2)
        assert model.init_prob.shape == (2,)

    def test_trans_prob_rows_sum_to_one(self, simple_data):
        model = GaussianHMM(n_states=3, n_channels=3)
        model.init_params(simple_data)
        row_sums = model.trans_prob.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_fit_improves_likelihood(self, simple_data):
        model = GaussianHMM(n_states=2, n_channels=3)
        history = model.fit(simple_data, n_epochs=10, n_init=1)
        assert len(history) == 10
        # LL should generally improve (allow some noise)
        assert history[-1] > history[0]

    def test_infer_shape(self, simple_data):
        model = GaussianHMM(n_states=2, n_channels=3)
        model.fit(simple_data, n_epochs=5, n_init=1)
        gammas = model.infer(simple_data)
        assert len(gammas) == 1
        assert gammas[0].shape == (400, 2)

    def test_infer_sums_to_one(self, simple_data):
        model = GaussianHMM(n_states=2, n_channels=3)
        model.fit(simple_data, n_epochs=5, n_init=1)
        gammas = model.infer(simple_data)
        row_sums = gammas[0].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_decode_shape(self, simple_data):
        model = GaussianHMM(n_states=2, n_channels=3)
        model.fit(simple_data, n_epochs=5, n_init=1)
        sequences = model.decode(simple_data)
        assert len(sequences) == 1
        assert sequences[0].shape == (400,)

    def test_decode_valid_states(self, simple_data):
        model = GaussianHMM(n_states=2, n_channels=3)
        model.fit(simple_data, n_epochs=5, n_init=1)
        states = model.decode(simple_data)[0]
        assert jnp.all((states >= 0) & (states < 2))

    def test_stay_prob_config(self):
        config = HMMConfig(n_states=3, n_channels=5, stay_prob=0.95)
        model = GaussianHMM(config=config)
        data = [jr.normal(jr.PRNGKey(0), (100, 5))]
        model.init_params(data)
        # Diagonal should be 0.95
        diag = jnp.diag(model.trans_prob)
        np.testing.assert_allclose(diag, 0.95, atol=1e-6)

    def test_repr(self):
        model = GaussianHMM(n_states=8, n_channels=80)
        assert "n_states=8" in repr(model)
