"""Tests for M-DyNeMo — Multi-dynamic Network Modes (red-green TDD).

M-DyNeMo (Huang, Gohil & Woolrich 2025) separates *power* and *functional
connectivity* dynamics into two independent mode time courses:

    alpha (n_modes)      -> mixes mode means + standard deviations  -> D_t
    gamma (n_corr_modes) -> mixes mode correlation matrices         -> C_t
    Sigma_t = D_t @ C_t @ D_t,    x_t ~ N(m_t, Sigma_t)

This is the key extension over DyNeMo (single time course, single covariance).
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neurojax.models.mdynemo import MDyNeMo, MDyNeMoConfig


@pytest.fixture
def small_data():
    key = jr.PRNGKey(0)
    return [jr.normal(key, (600, 5))]


class TestConfig:
    def test_n_corr_modes_defaults_to_n_modes(self):
        c = MDyNeMoConfig(n_modes=4, n_channels=5)
        assert c.n_modes == 4
        assert c.n_corr_modes == 4

    def test_separate_corr_modes(self):
        c = MDyNeMoConfig(n_modes=4, n_corr_modes=6, n_channels=5)
        assert c.n_corr_modes == 6


class TestConstruction:
    def test_param_shapes(self):
        m = MDyNeMo(n_modes=4, n_corr_modes=3, n_channels=5)
        assert m.get_means().shape == (4, 5)
        assert m.get_stds().shape == (4, 5)
        assert m.get_corrs().shape == (3, 5, 5)

    def test_stds_positive(self):
        m = MDyNeMo(n_modes=4, n_channels=5)
        assert jnp.all(m.get_stds() > 0)

    def test_corrs_are_valid_correlation_matrices(self):
        m = MDyNeMo(n_modes=4, n_corr_modes=3, n_channels=5)
        C = np.asarray(m.get_corrs())
        for k in range(3):
            np.testing.assert_allclose(np.diag(C[k]), 1.0, atol=1e-5)
            np.testing.assert_allclose(C[k], C[k].T, atol=1e-5)
            assert np.linalg.eigvalsh(C[k]).min() > -1e-6


class TestInference:
    @pytest.fixture(scope="class")
    def fitted(self):
        m = MDyNeMo(n_modes=4, n_corr_modes=3, n_channels=5)
        key = jr.PRNGKey(0)
        m.fit([jr.normal(key, (600, 5))], n_epochs=2)
        return m

    def test_two_separate_time_courses(self, fitted, small_data):
        alpha, gamma = fitted.infer(small_data)[0]
        assert alpha.shape == (600, 4)
        assert gamma.shape == (600, 3)

    def test_time_courses_normalized(self, fitted, small_data):
        alpha, gamma = fitted.infer(small_data)[0]
        np.testing.assert_allclose(np.asarray(alpha).sum(1), 1.0, atol=1e-4)
        np.testing.assert_allclose(np.asarray(gamma).sum(1), 1.0, atol=1e-4)


class TestTraining:
    def test_fit_runs_and_improves(self, small_data):
        m = MDyNeMo(n_modes=4, n_corr_modes=3, n_channels=5)
        h = m.fit(small_data, n_epochs=12)
        assert len(h) == 12
        assert h[-1]["loss"] < h[0]["loss"]

    def test_reconstructed_covariance_is_spd(self, small_data):
        m = MDyNeMo(n_modes=4, n_corr_modes=3, n_channels=5)
        m.fit(small_data, n_epochs=3)
        Sigma = np.asarray(m.get_covariances(small_data[0]))  # (T, C, C)
        assert Sigma.shape == (600, 5, 5)
        for t in (0, 250, 599):
            assert np.linalg.eigvalsh(Sigma[t]).min() > 0


class TestSeparateDynamicsRecovery:
    """The defining property: alpha and gamma capture *independent* dynamics.

    Build data whose power (variance) switches on a different schedule than its
    correlation, and check the two recovered time courses are not collapsed
    onto each other.
    """

    def test_power_and_fc_time_courses_decouple(self):
        rng = np.random.default_rng(0)
        T, C = 1500, 4
        # Power regime flips every 250 samples; correlation regime every 375.
        x = np.zeros((T, C), np.float32)
        corr = np.eye(C)
        corr_b = np.full((C, C), 0.8) + 0.2 * np.eye(C)
        for t in range(T):
            scale = 2.0 if (t // 250) % 2 == 0 else 0.5
            cmat = corr if (t // 375) % 2 == 0 else corr_b
            x[t] = rng.multivariate_normal(np.zeros(C), (scale ** 2) * cmat)
        m = MDyNeMo(n_modes=3, n_corr_modes=3, n_channels=C)
        m.fit([jnp.asarray(x)], n_epochs=40)
        alpha, gamma = m.infer([jnp.asarray(x)])[0]
        a = np.asarray(alpha)
        g = np.asarray(gamma)
        # The dominant power mode should track the variance schedule; assert the
        # two time courses are not identical (independent dynamics recovered).
        corr_ag = np.corrcoef(a[:, 0], g[:, 0])[0, 1]
        assert abs(corr_ag) < 0.95, f"alpha/gamma collapsed (corr={corr_ag:.2f})"
