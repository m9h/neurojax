"""
Unit tests for neurojax.geometry subpackage.

Tests cover:
- BEM wrapper (bem.py) — import and function existence (requires MNE/FreeSurfer)
- BEM PINN solver (bem_jinns.py) — model creation and inference shapes
- Riemannian geometry (riemann.py) — log/exp maps, geodesic distance, Frechet mean
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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(123)


def _make_spd(key, n):
    """Create a random symmetric positive definite matrix of size (n, n)."""
    A = jax.random.normal(key, (n, n))
    return A @ A.T + 0.1 * jnp.eye(n)


@pytest.fixture
def spd_pair(rng_key):
    """Two small SPD matrices."""
    k1, k2 = jax.random.split(rng_key)
    A = _make_spd(k1, 4)
    B = _make_spd(k2, 4)
    return A, B


@pytest.fixture
def spd_batch(rng_key):
    """Batch of 6 SPD matrices of size 4x4."""
    keys = jax.random.split(rng_key, 6)
    return jnp.stack([_make_spd(k, 4) for k in keys])


# ===================================================================
# BEM wrapper tests (bem.py) — thin MNE wrapper, so just test import
# ===================================================================

class TestBEM:
    """Tests for neurojax.geometry.bem (MNE-based BEM wrapper)."""

    def test_import(self):
        """Module is importable and exposes expected functions."""
        from neurojax.geometry.bem import (
            make_bem_surfaces,
            coregister_montage,
            make_scalp_surfaces,
        )
        assert callable(make_bem_surfaces)
        assert callable(coregister_montage)
        assert callable(make_scalp_surfaces)

    def test_make_bem_surfaces_signature(self):
        """make_bem_surfaces accepts (subject, subjects_dir, overwrite)."""
        import inspect
        from neurojax.geometry.bem import make_bem_surfaces
        sig = inspect.signature(make_bem_surfaces)
        params = list(sig.parameters.keys())
        assert "subject" in params
        assert "subjects_dir" in params
        assert "overwrite" in params

    def test_coregister_montage_signature(self):
        """coregister_montage has the expected parameters."""
        import inspect
        from neurojax.geometry.bem import coregister_montage
        sig = inspect.signature(coregister_montage)
        params = list(sig.parameters.keys())
        assert "raw" in params
        assert "trans_path" in params
        assert "subject" in params

    def test_make_bem_surfaces_cache_check(self, tmp_path):
        """If BEM surface file exists, make_bem_surfaces returns early."""
        from neurojax.geometry.bem import make_bem_surfaces
        from pathlib import Path

        subject = "test_subj"
        subjects_dir = tmp_path
        bem_dir = subjects_dir / subject / "bem"
        bem_dir.mkdir(parents=True)
        (bem_dir / "outer_skin.surf").touch()

        # Should not raise (cache hit)
        make_bem_surfaces(subject, subjects_dir, overwrite=False)


# ===================================================================
# BEM PINN solver tests (bem_jinns.py)
# ===================================================================

class TestBemJinns:
    """Tests for neurojax.geometry.bem_jinns."""

    def test_import(self):
        """Module is importable and exposes expected names."""
        from neurojax.geometry.bem_jinns import create_bem_pinn, BemSolver
        assert callable(create_bem_pinn)

    def test_create_bem_pinn_output(self, rng_key):
        """create_bem_pinn returns an Equinox MLP-like module."""
        from neurojax.geometry.bem_jinns import create_bem_pinn
        import equinox as eqx
        model = create_bem_pinn(rng_key, input_dim=3, output_dim=1, hidden_width=32, depth=2)
        assert isinstance(model, eqx.Module)

    def test_create_bem_pinn_forward(self, rng_key):
        """Model can perform forward pass on a single 3D point."""
        from neurojax.geometry.bem_jinns import create_bem_pinn
        model = create_bem_pinn(rng_key, input_dim=3, output_dim=1, hidden_width=16, depth=2)
        point = jnp.array([0.1, 0.2, 0.3])
        out = model(point)
        assert out.shape == (1,)
        assert jnp.isfinite(out).all()

    def test_bem_solver_single_point(self, rng_key):
        """BemSolver can evaluate a single 3D point."""
        from neurojax.geometry.bem_jinns import create_bem_pinn, BemSolver
        model = create_bem_pinn(rng_key, input_dim=3, output_dim=1, hidden_width=16, depth=2)
        solver = BemSolver(model, sigma=0.3)
        point = jnp.array([0.0, 0.0, 0.0])
        out = solver(point)
        assert out.shape == (1,)

    def test_bem_solver_batch(self, rng_key):
        """BemSolver handles batched input (N, 3) -> (N, 1)."""
        from neurojax.geometry.bem_jinns import create_bem_pinn, BemSolver
        model = create_bem_pinn(rng_key, input_dim=3, output_dim=1, hidden_width=16, depth=2)
        solver = BemSolver(model, sigma=0.3)
        points = jax.random.normal(rng_key, (20, 3))
        out = solver(points)
        assert out.shape == (20, 1)
        assert jnp.all(jnp.isfinite(out))

    def test_bem_solver_sigma_attribute(self, rng_key):
        """BemSolver stores conductivity sigma."""
        from neurojax.geometry.bem_jinns import create_bem_pinn, BemSolver
        model = create_bem_pinn(rng_key)
        solver = BemSolver(model, sigma=0.5)
        assert solver.sigma == 0.5

    def test_bem_solver_default_sigma(self, rng_key):
        """Default sigma is 0.3."""
        from neurojax.geometry.bem_jinns import create_bem_pinn, BemSolver
        model = create_bem_pinn(rng_key)
        solver = BemSolver(model)
        assert solver.sigma == 0.3


# ===================================================================
# Riemannian geometry tests (riemann.py)
# ===================================================================

class TestRiemannianHelpers:
    """Tests for internal helpers _powm, _logm, _expm."""

    def test_powm_identity(self):
        """I^k = I for any k."""
        from neurojax.geometry.riemann import _powm
        eye = jnp.eye(4)
        for k in [0.5, 1.0, -0.5, 2.0]:
            result = _powm(eye, k)
            np.testing.assert_allclose(result, eye, atol=1e-5)

    def test_powm_inverse(self, spd_pair):
        """A^1 @ A^{-1} ≈ I."""
        from neurojax.geometry.riemann import _powm
        A, _ = spd_pair
        A1 = _powm(A, 1.0)
        Ainv = _powm(A, -1.0)
        product = A1 @ Ainv
        np.testing.assert_allclose(product, jnp.eye(4), atol=1e-4)

    def test_powm_sqrt(self, spd_pair):
        """A^{0.5} @ A^{0.5} ≈ A."""
        from neurojax.geometry.riemann import _powm
        A, _ = spd_pair
        sqrtA = _powm(A, 0.5)
        reconstructed = sqrtA @ sqrtA
        np.testing.assert_allclose(reconstructed, A, atol=1e-4)

    def test_logm_expm_roundtrip(self, spd_pair):
        """exp(log(A)) ≈ A for SPD A."""
        from neurojax.geometry.riemann import _logm, _expm
        A, _ = spd_pair
        logA = _logm(A)
        reconstructed = _expm(logA)
        np.testing.assert_allclose(reconstructed, A, atol=1e-4)

    def test_expm_identity(self):
        """exp(0) = I."""
        from neurojax.geometry.riemann import _expm
        zero = jnp.zeros((4, 4))
        result = _expm(zero)
        np.testing.assert_allclose(result, jnp.eye(4), atol=1e-6)

    def test_logm_identity(self):
        """log(I) = 0."""
        from neurojax.geometry.riemann import _logm
        eye = jnp.eye(4)
        result = _logm(eye)
        np.testing.assert_allclose(result, jnp.zeros((4, 4)), atol=1e-6)


class TestRiemannianDistance:
    """Tests for riemannian_distance."""

    def test_distance_self_zero(self, spd_pair):
        """Distance from A to A should be 0."""
        from neurojax.geometry.riemann import riemannian_distance
        A, _ = spd_pair
        d = riemannian_distance(A, A)
        np.testing.assert_allclose(float(d), 0.0, atol=1e-4)

    def test_distance_symmetric(self, spd_pair):
        """d(A, B) = d(B, A)."""
        from neurojax.geometry.riemann import riemannian_distance
        A, B = spd_pair
        d_ab = riemannian_distance(A, B)
        d_ba = riemannian_distance(B, A)
        np.testing.assert_allclose(float(d_ab), float(d_ba), atol=1e-4)

    def test_distance_non_negative(self, spd_pair):
        """Riemannian distance is non-negative."""
        from neurojax.geometry.riemann import riemannian_distance
        A, B = spd_pair
        d = riemannian_distance(A, B)
        assert float(d) >= -1e-6

    def test_distance_triangle_inequality(self, rng_key):
        """d(A, C) <= d(A, B) + d(B, C)."""
        from neurojax.geometry.riemann import riemannian_distance
        k1, k2, k3 = jax.random.split(rng_key, 3)
        A = _make_spd(k1, 3)
        B = _make_spd(k2, 3)
        C = _make_spd(k3, 3)
        d_ab = float(riemannian_distance(A, B))
        d_bc = float(riemannian_distance(B, C))
        d_ac = float(riemannian_distance(A, C))
        assert d_ac <= d_ab + d_bc + 1e-4

    def test_distance_scales_with_eigenvalues(self):
        """Scaled identity matrices: d(I, kI) = sqrt(n) * |log(k)|."""
        from neurojax.geometry.riemann import riemannian_distance
        n = 4
        k = 2.0
        I_n = jnp.eye(n)
        kI = k * I_n
        d = riemannian_distance(I_n, kI)
        expected = jnp.sqrt(n) * jnp.abs(jnp.log(k))
        np.testing.assert_allclose(float(d), float(expected), atol=1e-4)


class TestLogExpMap:
    """Tests for log_map and exp_map."""

    def test_log_exp_roundtrip(self, spd_pair):
        """exp_map(log_map(C, ref), ref) ≈ C."""
        from neurojax.geometry.riemann import log_map, exp_map
        C, C_ref = spd_pair
        tangent = log_map(C, C_ref)
        reconstructed = exp_map(tangent, C_ref)
        np.testing.assert_allclose(reconstructed, C, atol=1e-3)

    def test_exp_log_roundtrip(self, spd_pair):
        """log_map(exp_map(T, ref), ref) ≈ T for small T."""
        from neurojax.geometry.riemann import log_map, exp_map
        _, C_ref = spd_pair
        # Small tangent vector (symmetric matrix)
        T = 0.1 * jnp.eye(4)
        C = exp_map(T, C_ref)
        T_back = log_map(C, C_ref)
        np.testing.assert_allclose(T_back, T, atol=1e-3)

    def test_log_map_at_ref_is_zero(self, spd_pair):
        """log_map(ref, ref) = 0 matrix."""
        from neurojax.geometry.riemann import log_map
        A, _ = spd_pair
        tangent = log_map(A, A)
        np.testing.assert_allclose(tangent, jnp.zeros((4, 4)), atol=1e-4)

    def test_exp_map_zero_tangent(self, spd_pair):
        """exp_map(0, ref) = ref."""
        from neurojax.geometry.riemann import exp_map
        _, C_ref = spd_pair
        zero = jnp.zeros((4, 4))
        result = exp_map(zero, C_ref)
        np.testing.assert_allclose(result, C_ref, atol=1e-4)

    def test_tangent_is_symmetric(self, spd_pair):
        """log_map output should be symmetric."""
        from neurojax.geometry.riemann import log_map
        C, C_ref = spd_pair
        tangent = log_map(C, C_ref)
        np.testing.assert_allclose(tangent, tangent.T, atol=1e-5)


class TestCovarianceMean:
    """Tests for covariance_mean (Frechet mean)."""

    def test_mean_shape(self, spd_batch):
        """Frechet mean has the same shape as input matrices."""
        from neurojax.geometry.riemann import covariance_mean
        mean = covariance_mean(spd_batch)
        assert mean.shape == (4, 4)

    def test_mean_is_spd(self, spd_batch):
        """Frechet mean should be SPD."""
        from neurojax.geometry.riemann import covariance_mean
        mean = covariance_mean(spd_batch)
        eigvals = jnp.linalg.eigvalsh(mean)
        assert jnp.all(eigvals > 0)

    def test_mean_is_symmetric(self, spd_batch):
        """Frechet mean should be symmetric."""
        from neurojax.geometry.riemann import covariance_mean
        mean = covariance_mean(spd_batch)
        np.testing.assert_allclose(mean, mean.T, atol=1e-5)

    def test_mean_of_identical_matrices(self):
        """Mean of identical matrices is that matrix."""
        from neurojax.geometry.riemann import covariance_mean
        A = 2.0 * jnp.eye(3) + 0.5 * jnp.ones((3, 3))
        # Stack 5 identical copies
        batch = jnp.stack([A] * 5)
        mean = covariance_mean(batch)
        np.testing.assert_allclose(mean, A, atol=1e-4)

    def test_mean_of_two_scaled_identities(self):
        """Geometric mean of aI and bI should be sqrt(ab)*I."""
        from neurojax.geometry.riemann import covariance_mean
        a, b = 1.0, 4.0
        n = 3
        batch = jnp.stack([a * jnp.eye(n), b * jnp.eye(n)])
        mean = covariance_mean(batch)
        expected = jnp.sqrt(a * b) * jnp.eye(n)
        np.testing.assert_allclose(mean, expected, atol=1e-3)


class TestTangentSpaceVectorize:
    """Tests for tangent_space_vectorize."""

    def test_output_length(self):
        """For n x n matrix, output has n*(n+1)/2 elements."""
        from neurojax.geometry.riemann import tangent_space_vectorize
        n = 4
        T = jnp.eye(n)
        vec = tangent_space_vectorize(T)
        expected_len = n * (n + 1) // 2
        assert vec.shape == (expected_len,)

    def test_diagonal_preserved(self):
        """Diagonal elements appear unscaled in the vector."""
        from neurojax.geometry.riemann import tangent_space_vectorize
        n = 3
        T = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        vec = tangent_space_vectorize(T)
        # Upper triangle indices: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
        # For a diagonal matrix, off-diagonal entries are 0.
        # Diagonal positions in triu: (0,0)=idx0, (1,1)=idx3, (2,2)=idx5
        assert float(vec[0]) == pytest.approx(1.0)
        assert float(vec[3]) == pytest.approx(2.0)
        assert float(vec[5]) == pytest.approx(3.0)

    def test_off_diagonal_scaled(self):
        """Off-diagonal elements are scaled by sqrt(2)."""
        from neurojax.geometry.riemann import tangent_space_vectorize
        n = 3
        # Matrix with 1 on all entries
        T = jnp.ones((n, n))
        vec = tangent_space_vectorize(T)
        # triu indices for 3x3: (0,0),(0,1),(0,2),(1,1),(1,2),(2,2)
        # Diag: idx 0,3,5 -> value 1.0
        # Off-diag: idx 1,2,4 -> value 1.0 * sqrt(2)
        sqrt2 = float(jnp.sqrt(2.0))
        assert float(vec[0]) == pytest.approx(1.0)
        assert float(vec[1]) == pytest.approx(sqrt2)
        assert float(vec[2]) == pytest.approx(sqrt2)
        assert float(vec[3]) == pytest.approx(1.0)
        assert float(vec[4]) == pytest.approx(sqrt2)
        assert float(vec[5]) == pytest.approx(1.0)

    def test_norm_preservation(self, spd_pair):
        """Vectorization should preserve the Frobenius norm for symmetric matrices."""
        from neurojax.geometry.riemann import tangent_space_vectorize
        A, _ = spd_pair
        # Make symmetric tangent-like matrix
        T = (A + A.T) / 2.0
        vec = tangent_space_vectorize(T)
        # ||vec||_2 should equal ||T||_F for symmetric T
        np.testing.assert_allclose(
            float(jnp.linalg.norm(vec)),
            float(jnp.linalg.norm(T, ord="fro")),
            atol=1e-4,
        )


class TestMapTangentSpace:
    """Tests for map_tangent_space."""

    def test_output_shape(self, spd_batch):
        """Output is (n_matrices, n_features)."""
        from neurojax.geometry.riemann import map_tangent_space
        vecs = map_tangent_space(spd_batch)
        n_matrices = spd_batch.shape[0]
        n = spd_batch.shape[1]
        n_features = n * (n + 1) // 2
        assert vecs.shape == (n_matrices, n_features)

    def test_output_finite(self, spd_batch):
        """All output values should be finite."""
        from neurojax.geometry.riemann import map_tangent_space
        vecs = map_tangent_space(spd_batch)
        assert jnp.all(jnp.isfinite(vecs))

    def test_with_precomputed_mean(self, spd_batch):
        """Providing a precomputed mean should work."""
        from neurojax.geometry.riemann import map_tangent_space, covariance_mean
        mean = covariance_mean(spd_batch)
        vecs = map_tangent_space(spd_batch, mean=mean)
        assert vecs.shape[0] == spd_batch.shape[0]
        assert jnp.all(jnp.isfinite(vecs))
