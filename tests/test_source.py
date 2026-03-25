"""
Unit tests for neurojax.source subpackage.

Tests cover:
- LCMV Beamformer (beamformer.py)
- CHAMPAGNE source localization (champagne.py)
- ADMM inverse solver (inverse_scico.py)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")

# ---------------------------------------------------------------------------
# Fixtures: small synthetic forward models
# ---------------------------------------------------------------------------

@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def small_forward(rng_key):
    """10-sensor, 5-source forward model with synthetic data."""
    n_chan, n_src, n_time = 10, 5, 200
    k1, k2, k3 = jax.random.split(rng_key, 3)

    gain = jax.random.normal(k1, (n_chan, n_src))
    # Simulate source activity: only source 2 is active
    sources = jnp.zeros((n_src, n_time))
    sources = sources.at[2].set(jax.random.normal(k2, (n_time,)))
    noise = 0.1 * jax.random.normal(k3, (n_chan, n_time))
    data = gain @ sources + noise

    cov = (data @ data.T) / n_time
    return {
        "gain": gain,
        "cov": cov,
        "data": data,
        "sources": sources,
        "n_chan": n_chan,
        "n_src": n_src,
        "n_time": n_time,
    }


# ===================================================================
# LCMV Beamformer tests
# ===================================================================

class TestLCMVBeamformer:
    """Tests for neurojax.source.beamformer."""

    def test_import(self):
        """Module is importable and exposes expected functions."""
        from neurojax.source.beamformer import make_lcmv_filter, apply_lcmv, lcmv_power
        assert callable(make_lcmv_filter)
        assert callable(apply_lcmv)
        assert callable(lcmv_power)

    def test_weights_shape(self, small_forward):
        """Weights have shape (n_sources, n_chan)."""
        from neurojax.source.beamformer import make_lcmv_filter
        w = make_lcmv_filter(small_forward["cov"], small_forward["gain"])
        assert w.shape == (small_forward["n_src"], small_forward["n_chan"])

    def test_apply_lcmv_shape(self, small_forward):
        """apply_lcmv returns (n_sources, n_time)."""
        from neurojax.source.beamformer import make_lcmv_filter, apply_lcmv
        w = make_lcmv_filter(small_forward["cov"], small_forward["gain"])
        s_hat = apply_lcmv(small_forward["data"], w)
        assert s_hat.shape == (small_forward["n_src"], small_forward["n_time"])

    def test_unit_gain_constraint(self, small_forward):
        """LCMV weights satisfy unit-gain constraint: W @ G ≈ I."""
        from neurojax.source.beamformer import make_lcmv_filter
        gain = small_forward["gain"]
        cov = small_forward["cov"]
        w = make_lcmv_filter(cov, gain)
        product = w @ gain  # (n_src, n_src) should be close to I
        eye = jnp.eye(small_forward["n_src"])
        np.testing.assert_allclose(product, eye, atol=1e-4)

    def test_peak_at_active_source(self, small_forward):
        """Beamformer output power peaks at the active source (index 2)."""
        from neurojax.source.beamformer import make_lcmv_filter, apply_lcmv
        w = make_lcmv_filter(small_forward["cov"], small_forward["gain"])
        s_hat = apply_lcmv(small_forward["data"], w)
        power = jnp.mean(s_hat ** 2, axis=1)
        assert int(jnp.argmax(power)) == 2

    def test_regularization_positive(self, small_forward):
        """Non-negative reg keeps covariance invertible even if singular."""
        from neurojax.source.beamformer import make_lcmv_filter
        # Build a rank-deficient covariance
        cov = jnp.ones((10, 10))  # rank 1
        gain = small_forward["gain"]
        w = make_lcmv_filter(cov, gain, reg=0.1)
        assert jnp.all(jnp.isfinite(w))

    def test_lcmv_power_scalar(self, small_forward):
        """lcmv_power returns a scalar."""
        from neurojax.source.beamformer import lcmv_power
        p = lcmv_power(small_forward["cov"], small_forward["gain"])
        assert p.shape == ()
        assert jnp.isfinite(p)

    def test_lcmv_power_positive(self, small_forward):
        """Power is non-negative for valid covariance + gain."""
        from neurojax.source.beamformer import lcmv_power
        p = lcmv_power(small_forward["cov"], small_forward["gain"])
        assert float(p) >= 0.0

    def test_weights_finite_with_identity_cov(self, small_forward):
        """Identity covariance should produce valid weights."""
        from neurojax.source.beamformer import make_lcmv_filter
        cov = jnp.eye(small_forward["n_chan"])
        gain = small_forward["gain"]
        w = make_lcmv_filter(cov, gain, reg=0.0)
        assert jnp.all(jnp.isfinite(w))

    def test_jit_compilation(self, small_forward):
        """Functions are JIT-compiled (decorated with @jit)."""
        from neurojax.source.beamformer import make_lcmv_filter, apply_lcmv, lcmv_power
        # All three should be wrapped in JIT already; calling them twice
        # shouldn't raise errors
        cov = small_forward["cov"]
        gain = small_forward["gain"]
        w1 = make_lcmv_filter(cov, gain)
        w2 = make_lcmv_filter(cov, gain)
        np.testing.assert_allclose(w1, w2, atol=1e-6)


# ===================================================================
# CHAMPAGNE tests
# ===================================================================

class TestChampagne:
    """Tests for neurojax.source.champagne."""

    def test_import(self):
        """Module is importable and exposes expected functions."""
        from neurojax.source.champagne import champagne_solver, imaginary_coherence
        assert callable(champagne_solver)
        assert callable(imaginary_coherence)

    def test_champagne_output_shapes(self, small_forward):
        """champagne_solver returns (gamma, weights) with correct shapes."""
        from neurojax.source.champagne import champagne_solver
        gamma, weights = champagne_solver(
            small_forward["cov"],
            small_forward["gain"],
            max_iter=5,
        )
        assert gamma.shape == (small_forward["n_src"],)
        assert weights.shape == (small_forward["n_src"], small_forward["n_chan"])

    def test_champagne_gamma_positive(self, small_forward):
        """Estimated source powers (gamma) should be non-negative."""
        from neurojax.source.champagne import champagne_solver
        gamma, _ = champagne_solver(
            small_forward["cov"],
            small_forward["gain"],
            max_iter=10,
        )
        assert jnp.all(gamma >= 0)

    def test_champagne_peak_at_active_source(self, small_forward):
        """CHAMPAGNE gamma should peak at active source (index 2)."""
        from neurojax.source.champagne import champagne_solver
        gamma, _ = champagne_solver(
            small_forward["cov"],
            small_forward["gain"],
            max_iter=20,
        )
        assert int(jnp.argmax(gamma)) == 2

    def test_champagne_with_noise_cov(self, small_forward):
        """Explicit noise covariance should not crash."""
        from neurojax.source.champagne import champagne_solver
        noise_cov = 0.01 * jnp.eye(small_forward["n_chan"])
        gamma, weights = champagne_solver(
            small_forward["cov"],
            small_forward["gain"],
            noise_cov=noise_cov,
            max_iter=5,
        )
        assert jnp.all(jnp.isfinite(gamma))
        assert jnp.all(jnp.isfinite(weights))

    def test_champagne_weights_finite(self, small_forward):
        """Weights from CHAMPAGNE should be finite."""
        from neurojax.source.champagne import champagne_solver
        _, weights = champagne_solver(
            small_forward["cov"],
            small_forward["gain"],
            max_iter=5,
        )
        assert jnp.all(jnp.isfinite(weights))

    def test_imaginary_coherence_shape(self, rng_key):
        """imaginary_coherence returns (n_sources,)."""
        from neurojax.source.champagne import imaginary_coherence
        n_src, n_time = 5, 100
        # Complex analytic signal
        k1, k2 = jax.random.split(rng_key)
        real_part = jax.random.normal(k1, (n_src, n_time))
        imag_part = jax.random.normal(k2, (n_src, n_time))
        source_data = real_part + 1j * imag_part
        icoh = imaginary_coherence(source_data, ref_idx=0)
        assert icoh.shape == (n_src,)

    def test_imaginary_coherence_self_zero(self, rng_key):
        """Imaginary coherence of a source with itself should be ~0."""
        from neurojax.source.champagne import imaginary_coherence
        n_src, n_time = 5, 200
        k1, k2 = jax.random.split(rng_key)
        real_part = jax.random.normal(k1, (n_src, n_time))
        imag_part = jax.random.normal(k2, (n_src, n_time))
        source_data = real_part + 1j * imag_part

        icoh = imaginary_coherence(source_data, ref_idx=0)
        # Self-coherence is csd(0,0) = mean(|x0|^2), which is real => Im = 0
        np.testing.assert_allclose(float(icoh[0]), 0.0, atol=1e-5)

    def test_imaginary_coherence_bounded(self, rng_key):
        """Imaginary coherence values should be in [-1, 1]."""
        from neurojax.source.champagne import imaginary_coherence
        n_src, n_time = 5, 500
        k1, k2 = jax.random.split(rng_key)
        real_part = jax.random.normal(k1, (n_src, n_time))
        imag_part = jax.random.normal(k2, (n_src, n_time))
        source_data = real_part + 1j * imag_part
        icoh = imaginary_coherence(source_data, ref_idx=1)
        assert jnp.all(jnp.abs(icoh) <= 1.0 + 1e-6)


# ===================================================================
# ADMM Inverse Solver tests
# ===================================================================

class TestInverseADMM:
    """Tests for neurojax.source.inverse_scico."""

    def test_import(self):
        """Module is importable and exposes expected names."""
        from neurojax.source.inverse_scico import (
            InverseResult,
            soft_threshold,
            solve_inverse_admm,
            compute_resolution_matrix,
        )
        assert callable(soft_threshold)
        assert callable(solve_inverse_admm)
        assert callable(compute_resolution_matrix)

    def test_soft_threshold_basic(self):
        """Soft thresholding shrinks towards zero."""
        from neurojax.source.inverse_scico import soft_threshold
        x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        out = soft_threshold(x, 1.5)
        expected = jnp.array([-1.5, 0.0, 0.0, 0.0, 1.5])
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_soft_threshold_zero_threshold(self):
        """Zero threshold is identity."""
        from neurojax.source.inverse_scico import soft_threshold
        x = jnp.array([1.0, -2.0, 3.0])
        out = soft_threshold(x, 0.0)
        np.testing.assert_allclose(out, x, atol=1e-7)

    def test_soft_threshold_large_threshold(self):
        """Threshold larger than all values gives zeros."""
        from neurojax.source.inverse_scico import soft_threshold
        x = jnp.array([1.0, -2.0, 0.5])
        out = soft_threshold(x, 10.0)
        np.testing.assert_allclose(out, jnp.zeros(3), atol=1e-7)

    def test_solve_inverse_admm_output_type(self, small_forward):
        """solve_inverse_admm returns an InverseResult."""
        from neurojax.source.inverse_scico import solve_inverse_admm, InverseResult
        result = solve_inverse_admm(
            small_forward["data"],
            small_forward["gain"],
            maxiter=5,
        )
        assert isinstance(result, InverseResult)

    def test_solve_inverse_admm_shapes(self, small_forward):
        """Sources and residuals have correct shapes."""
        from neurojax.source.inverse_scico import solve_inverse_admm
        result = solve_inverse_admm(
            small_forward["data"],
            small_forward["gain"],
            maxiter=10,
        )
        n_src = small_forward["n_src"]
        n_time = small_forward["n_time"]
        n_chan = small_forward["n_chan"]
        assert result.sources.shape == (n_src, n_time)
        assert result.residuals.shape == (n_chan, n_time)

    def test_solve_inverse_admm_residual_consistency(self, small_forward):
        """Residuals = Y - L @ X_hat by definition."""
        from neurojax.source.inverse_scico import solve_inverse_admm
        Y = small_forward["data"]
        L = small_forward["gain"]
        result = solve_inverse_admm(Y, L, maxiter=10)
        expected_residuals = Y - L @ result.sources
        np.testing.assert_allclose(result.residuals, expected_residuals, atol=1e-5)

    def test_solve_inverse_admm_sparsity(self, small_forward):
        """L1 penalty should produce some near-zero entries."""
        from neurojax.source.inverse_scico import solve_inverse_admm
        result = solve_inverse_admm(
            small_forward["data"],
            small_forward["gain"],
            lambda_reg=1.0,
            maxiter=50,
        )
        # With high regularization, many entries should be near zero
        frac_near_zero = float(jnp.mean(jnp.abs(result.sources) < 1e-3))
        assert frac_near_zero > 0.1  # at least 10% are near zero

    def test_solve_inverse_admm_zero_reg(self, small_forward):
        """Zero regularization should give a least-squares-like solution."""
        from neurojax.source.inverse_scico import solve_inverse_admm
        result = solve_inverse_admm(
            small_forward["data"],
            small_forward["gain"],
            lambda_reg=0.0,
            maxiter=100,
        )
        # Residuals should be small when lambda=0
        residual_norm = float(jnp.linalg.norm(result.residuals))
        data_norm = float(jnp.linalg.norm(small_forward["data"]))
        assert residual_norm / data_norm < 0.5

    def test_solve_inverse_admm_finite(self, small_forward):
        """All outputs should be finite."""
        from neurojax.source.inverse_scico import solve_inverse_admm
        result = solve_inverse_admm(
            small_forward["data"],
            small_forward["gain"],
            maxiter=10,
        )
        assert jnp.all(jnp.isfinite(result.sources))
        assert jnp.all(jnp.isfinite(result.residuals))

    def test_compute_resolution_matrix_shape(self, small_forward):
        """Resolution matrix R = G @ L has shape (n_src, n_src)."""
        from neurojax.source.inverse_scico import compute_resolution_matrix
        n_src = small_forward["n_src"]
        L = small_forward["gain"]  # (n_chan, n_src)
        # G is an inverse operator (n_src, n_chan)
        G = jnp.linalg.pinv(L)
        # compute_resolution_matrix(L, G) does jnp.dot(G, L) -> (n_src, n_src)
        R = compute_resolution_matrix(L, G)
        assert R.shape == (n_src, n_src)

    def test_resolution_matrix_pinv_identity(self, small_forward):
        """R = G @ L approximates identity when G = pinv(L) and n_chan >= n_src."""
        from neurojax.source.inverse_scico import compute_resolution_matrix
        L = small_forward["gain"]  # (10, 5) -> n_chan > n_src
        G = jnp.linalg.pinv(L)    # (5, 10)
        R = compute_resolution_matrix(L, G)  # jnp.dot(G, L) -> (5, 5)
        np.testing.assert_allclose(R, jnp.eye(5), atol=1e-4)

    def test_inverse_result_namedtuple(self):
        """InverseResult has the expected fields."""
        from neurojax.source.inverse_scico import InverseResult
        r = InverseResult(
            sources=jnp.ones((3, 10)),
            residuals=jnp.zeros((5, 10)),
        )
        assert r.sources.shape == (3, 10)
        assert r.residuals.shape == (5, 10)
        assert r.resolution_matrix is None
