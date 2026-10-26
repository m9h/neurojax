"""Tests for neurojax.inverse.csd_dcm -- spectral (CSD) DCM in JAX.

Red-green: the forward model (linearised Jansen-Rit -> transfer function ->
cross-spectral density -> sensors) is checked against closed-form linear
algebra, and the variational-Laplace inverter is checked against a
closed-form Bayesian linear regression and against parameter recovery
on synthetic CSD data.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Linear state-space -> CSD
# ---------------------------------------------------------------------------


class TestTransferFunction:
    def test_white_noise_csd_integrates_to_lyapunov_covariance(self):
        """∫ H(ω) H(ω)^H dω / 2π == C P Cᵀ with A P + P Aᵀ + B Bᵀ = 0."""
        from neurojax.inverse.csd_dcm import transfer_function

        rng = np.random.default_rng(0)
        n = 4
        A = rng.normal(size=(n, n))
        A = A - (np.abs(np.linalg.eigvals(A).real).max() + 1.0) * np.eye(n)
        B = rng.normal(size=(n, 2))
        C = rng.normal(size=(3, n))
        P = scipy.linalg.solve_continuous_lyapunov(A, -B @ B.T)
        target = C @ P @ C.T

        omega = np.linspace(-400.0, 400.0, 160_001)
        H = transfer_function(jnp.asarray(A), jnp.asarray(B), jnp.asarray(C), jnp.asarray(omega))
        assert H.shape == (omega.size, 3, 2)
        S = jnp.einsum("fij,fkj->fik", H, jnp.conj(H))
        est = np.trapezoid(np.asarray(S), omega, axis=0) / (2 * np.pi)
        np.testing.assert_allclose(est.real, target, rtol=2e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# Jansen-Rit linearisation
# ---------------------------------------------------------------------------


class TestJansenRitLinear:
    def test_fixed_point_is_stationary(self):
        from neurojax.inverse.csd_dcm import (
            JansenRitSpec,
            jr_fixed_point,
            jr_vector_field,
        )

        spec = JansenRitSpec(n_regions=2, A_forward=jnp.array([[0.0, 0.0], [1.0, 0.0]]))
        theta = spec.default_theta()
        x0 = jr_fixed_point(spec, theta)
        assert x0.shape == (6 * spec.n_regions,)
        dx = jr_vector_field(spec, theta, x0, jnp.zeros(spec.n_regions))
        np.testing.assert_allclose(np.asarray(dx), 0.0, atol=1e-8)

    def test_linearisation_is_stable_and_has_alpha_peak(self):
        from neurojax.inverse.csd_dcm import (
            JansenRitSpec,
            jr_linearise,
            transfer_function,
        )

        spec = JansenRitSpec(n_regions=1)
        theta = spec.default_theta()
        A, B, C = jr_linearise(spec, theta)
        assert A.shape == (6, 6) and B.shape == (6, 1) and C.shape == (1, 6)
        assert np.all(np.linalg.eigvals(np.asarray(A)).real < 0)

        f = jnp.linspace(1.0, 45.0, 441)
        H = transfer_function(A, B, C, 2 * jnp.pi * f)
        psd = jnp.abs(H[:, 0, 0]) ** 2
        f_peak = float(f[jnp.argmax(psd)])
        assert 7.0 < f_peak < 14.0, f_peak

    def test_slower_inhibition_lowers_peak_frequency(self):
        """GABA-A potentiation (propofol) ~ longer inhibitory time constant -> slower rhythm."""
        from neurojax.inverse.csd_dcm import (
            JansenRitSpec,
            jr_linearise,
            transfer_function,
        )

        spec = JansenRitSpec(n_regions=1)
        f = jnp.linspace(1.0, 45.0, 441)

        def peak(theta):
            A, B, C = jr_linearise(spec, theta)
            H = transfer_function(A, B, C, 2 * jnp.pi * f)
            return float(f[jnp.argmax(jnp.abs(H[:, 0, 0]) ** 2)])

        base = spec.default_theta()
        slow = base._replace(tau_i=base.tau_i * 1.5)
        assert peak(slow) < peak(base)


# ---------------------------------------------------------------------------
# Full forward model: parameters -> sensor CSD
# ---------------------------------------------------------------------------


class TestSpectralDCMForward:
    def test_predict_shape_hermitian_and_positive(self):
        from neurojax.inverse.csd_dcm import JansenRitSpec, SpectralDCM

        spec = JansenRitSpec(n_regions=2, A_forward=jnp.array([[0.0, 0.0], [1.0, 0.0]]))
        L = jnp.array([[1.0, 0.2], [0.1, 1.0], [0.5, 0.5]])
        f = jnp.linspace(1.0, 45.0, 89)
        dcm = SpectralDCM(spec, leadfield=L, freqs=f)
        csd = dcm.predict(dcm.prior_mean)
        assert csd.shape == (89, 3, 3)
        np.testing.assert_allclose(np.asarray(csd), np.asarray(jnp.conj(jnp.swapaxes(csd, 1, 2))), atol=1e-12)
        assert np.all(np.asarray(jnp.real(jnp.diagonal(csd, axis1=1, axis2=2))) > 0)

    def test_gradient_through_forward_is_finite(self):
        from neurojax.inverse.csd_dcm import JansenRitSpec, SpectralDCM

        spec = JansenRitSpec(n_regions=1)
        dcm = SpectralDCM(spec, leadfield=jnp.eye(1), freqs=jnp.linspace(1.0, 45.0, 45))
        target = dcm.predict(dcm.prior_mean)

        def loss(theta):
            return jnp.sum(jnp.abs(dcm.predict(theta) - target) ** 2)

        g = jax.grad(loss)(dcm.prior_mean + 0.1)
        assert g.shape == dcm.prior_mean.shape
        assert np.all(np.isfinite(np.asarray(g)))
        assert float(jnp.linalg.norm(g)) > 0


# ---------------------------------------------------------------------------
# Variational Laplace
# ---------------------------------------------------------------------------


class TestVariationalLaplace:
    def test_matches_closed_form_bayesian_linear_regression(self):
        """For f(p) = X p with known noise precision the VL posterior is exact."""
        from neurojax.inverse.csd_dcm import variational_laplace

        rng = np.random.default_rng(1)
        X = jnp.asarray(rng.normal(size=(40, 3)))
        p_true = jnp.array([1.0, -2.0, 0.5])
        y = X @ p_true + 0.1 * jnp.asarray(rng.normal(size=40))
        pE = jnp.zeros(3)
        pC = jnp.eye(3) * 4.0

        res = variational_laplace(lambda p: X @ p, y, pE, pC, n_iter=32)

        # closed form with the *estimated* precision exp(h)
        prec = jnp.exp(res.h)
        post_prec = X.T @ X * prec + jnp.linalg.inv(pC)
        post_cov = jnp.linalg.inv(post_prec)
        post_mean = post_cov @ (X.T @ y * prec)
        np.testing.assert_allclose(np.asarray(res.Ep), np.asarray(post_mean), atol=1e-4)
        np.testing.assert_allclose(np.asarray(res.Cp), np.asarray(post_cov), atol=1e-4)
        assert abs(float(prec) - 100.0) / 100.0 < 0.5  # noise precision ~1/0.1²
        assert np.isfinite(float(res.F))
        assert res.F_trace[-1] >= res.F_trace[0]

    def test_free_energy_penalises_complexity(self):
        """A model with an unnecessary free parameter has lower F than the nested one."""
        from neurojax.inverse.csd_dcm import variational_laplace

        rng = np.random.default_rng(2)
        X = jnp.asarray(rng.normal(size=(60, 2)))
        y = X[:, 0] * 1.5 + 0.1 * jnp.asarray(rng.normal(size=60))
        small = variational_laplace(lambda p: X[:, :1] @ p, y, jnp.zeros(1), jnp.eye(1) * 4.0, n_iter=32)
        big = variational_laplace(lambda p: X @ p, y, jnp.zeros(2), jnp.eye(2) * 4.0, n_iter=32)
        assert float(small.F) > float(big.F)


# ---------------------------------------------------------------------------
# Parameter recovery on synthetic CSD
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_gaba():
    """One-region DCM; 'drug' lengthens the inhibitory time constant by 40 %."""
    from neurojax.inverse.csd_dcm import JansenRitSpec, SpectralDCM

    spec = JansenRitSpec(n_regions=1)
    f = jnp.linspace(1.0, 45.0, 89)
    dcm = SpectralDCM(spec, leadfield=jnp.eye(1), freqs=f)
    theta_true = dcm.prior_mean.at[dcm.index("tau_i")].set(jnp.log(1.4))
    rng = np.random.default_rng(3)
    csd = dcm.predict(theta_true)
    csd = csd * (1.0 + 0.03 * jnp.asarray(rng.normal(size=csd.shape)))
    return dcm, theta_true, csd


class TestRecovery:
    def test_recovers_inhibitory_time_constant(self, synthetic_gaba):
        dcm, theta_true, csd = synthetic_gaba
        res = dcm.invert(csd, n_iter=48)
        i = dcm.index("tau_i")
        err = float(jnp.abs(res.Ep[i] - theta_true[i]))
        sd = float(jnp.sqrt(res.Cp[i, i]))
        # tau_e and tau_i trade off against each other (both move the peak), so
        # the free model recovers tau_i to within ~0.1 log units; see the
        # fixed-tau_e test for the inverter's own accuracy.
        assert err < 0.15, (err, sd)
        assert err < 3 * sd + 0.05
        assert res.F_trace[-1] > res.F_trace[0]

    def test_recovers_tau_i_with_tau_e_known(self, synthetic_gaba):
        dcm, theta_true, csd = synthetic_gaba
        res = dcm.with_fixed("tau_e").invert(csd, n_iter=48)
        i = dcm.index("tau_i")
        err = float(jnp.abs(res.Ep[i] - theta_true[i]))
        assert err < 0.06, (err, float(jnp.sqrt(res.Cp[i, i])))

    def test_model_comparison_prefers_free_tau_i(self, synthetic_gaba):
        dcm, theta_true, csd = synthetic_gaba
        full = dcm.invert(csd, n_iter=48)
        reduced = dcm.with_fixed("tau_i").invert(csd, n_iter=48)
        assert float(full.F) > float(reduced.F)


class TestTwoRegionCoupling:
    def test_recovers_forward_connection_gain(self):
        from neurojax.inverse.csd_dcm import JansenRitSpec, SpectralDCM

        # default gain 100/s per unit firing rate: comparable to the mean input
        spec = JansenRitSpec(n_regions=2, A_forward=jnp.array([[0.0, 0.0], [100.0, 0.0]]))
        L = jnp.array([[1.0, 0.1], [0.1, 1.0], [0.6, 0.6]])
        dcm = SpectralDCM(spec, leadfield=L, freqs=jnp.linspace(1.0, 45.0, 89))
        i = dcm.index("A_forward", 1, 0)
        theta_true = dcm.prior_mean.at[i].set(jnp.log(2.0))
        rng = np.random.default_rng(4)
        csd = dcm.predict(theta_true)
        csd = csd * (1.0 + 0.03 * jnp.asarray(rng.normal(size=csd.shape)))  # noise-free data give unbounded precision
        res = dcm.invert(csd, n_iter=48)
        assert float(jnp.abs(res.Ep[i] - theta_true[i])) < 0.15


# ---------------------------------------------------------------------------
# Canonical microcircuit
# ---------------------------------------------------------------------------


class TestCMC:
    def test_stable_with_two_resonances(self):
        from neurojax.inverse.csd_dcm import CMCSpec, linearise

        spec = CMCSpec()
        A, _, _ = linearise(spec, spec.default_theta())
        ev = np.linalg.eigvals(np.asarray(A))
        assert np.all(ev.real < 0)
        modes = np.unique(np.round(np.abs(ev.imag) / (2 * np.pi), 1))
        modes = modes[modes > 0]
        assert np.any((modes > 4) & (modes < 13)), modes  # alpha-range mode
        assert np.any((modes > 13) & (modes < 30)), modes  # beta-range mode

    def test_slower_inhibition_slows_beta_mode(self):
        from neurojax.inverse.csd_dcm import CMCSpec, linearise, transfer_function

        spec = CMCSpec()
        f = jnp.linspace(1.0, 45.0, 441)

        def peak(theta):
            A, B, C = linearise(spec, theta)
            H = transfer_function(A, B, C, 2 * jnp.pi * f)
            return float(f[jnp.argmax(jnp.abs(H[:, 0, 0]) ** 2)])

        base = spec.default_theta()
        slow = base._replace(T=base.T.at[0, 2].multiply(1.5))
        assert peak(slow) < peak(base)

    def test_spectral_dcm_with_cmc(self):
        from neurojax.inverse.csd_dcm import CMCSpec, SpectralDCM

        spec = CMCSpec(n_regions=2, A_forward=jnp.array([[0.0, 0.0], [800.0, 0.0]]), A_backward=jnp.array([[0.0, 400.0], [0.0, 0.0]]))
        L = jnp.array([[1.0, 0.2], [0.1, 1.0]])
        dcm = SpectralDCM(spec, leadfield=L, freqs=jnp.linspace(1.0, 45.0, 45))
        assert dcm.index("T", 1, 2) == 1 * 4 + 2  # region 1, population ii
        assert dcm.index("A_backward", 0, 1) >= 0
        csd = dcm.predict(dcm.prior_mean)
        assert csd.shape == (45, 2, 2)
        assert np.all(np.isfinite(np.asarray(csd)))
        g = jax.grad(lambda p: jnp.sum(jnp.abs(dcm.predict(p)) ** 2))(dcm.prior_mean + 0.05)
        assert np.all(np.isfinite(np.asarray(g))) and float(jnp.linalg.norm(g)) > 0

    def test_recovers_inhibitory_time_constant(self):
        from neurojax.inverse.csd_dcm import CMCSpec, SpectralDCM

        dcm = SpectralDCM(CMCSpec(), leadfield=jnp.eye(1), freqs=jnp.linspace(1.0, 45.0, 89))
        i = dcm.index("T", 0, 2)
        theta_true = dcm.prior_mean.at[i].set(jnp.log(1.4))
        rng = np.random.default_rng(5)
        csd = dcm.predict(theta_true)
        csd = csd * (1.0 + 0.03 * jnp.asarray(rng.normal(size=csd.shape)))
        res = dcm.invert(csd, n_iter=48)
        err = float(jnp.abs(res.Ep[i] - theta_true[i]))
        assert err < 0.15, (err, float(jnp.sqrt(res.Cp[i, i])))

    def test_matches_spm_linearisation(self):
        """Jacobian spectrum, input and output matrices of the CMC at the prior equal
        SPM25's spm_fx_cmc / spm_gx_erp (exported with SPM-Python on 2026-10-25,
        tests/data/spm_cmc_prior_linearisation.npz; SPM orders states
        voltage/current per population, neurojax voltages then currents)."""
        from pathlib import Path

        from neurojax.inverse.csd_dcm import CMCSpec, linearise, transfer_function

        ref = np.load(Path(__file__).parent / "data" / "spm_cmc_prior_linearisation.npz")
        spec = CMCSpec()
        A, B, C = linearise(spec, spec.default_theta())
        ev = np.sort_complex(np.linalg.eigvals(np.asarray(A)))
        ev_spm = np.sort_complex(np.linalg.eigvals(ref["J"]))
        np.testing.assert_allclose(ev, ev_spm, rtol=1e-6, atol=1e-6)
        w = 2 * np.pi * jnp.asarray(ref["Hz"])
        H = np.asarray(transfer_function(A, B, C, w))[:, 0, 0]
        H_spm = np.array([ref["C"] @ np.linalg.solve(1j * x * np.eye(8) - ref["J"], ref["B"].reshape(-1, 1)) for x in np.asarray(w)]).ravel()
        np.testing.assert_allclose(H, H_spm, rtol=1e-6, atol=1e-12)
