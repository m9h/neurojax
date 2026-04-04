"""TDD tests for differentiable FEM forward model.

RED phase: tensor conductivity from DTI and subtraction method for
dipole singularity handling.

GREEN phase: implement in fem_forward.py.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Existing functionality tests (should be GREEN)
# ---------------------------------------------------------------------------

class TestElementStiffness:
    """P1 tetrahedral element stiffness matrix."""

    def test_stiffness_shape(self):
        from neurojax.geometry.fem_forward import tet_stiffness_element
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=jnp.float32)
        K = tet_stiffness_element(verts, 1.0)
        assert K.shape == (4, 4)

    def test_stiffness_symmetric(self):
        from neurojax.geometry.fem_forward import tet_stiffness_element
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=jnp.float32)
        K = tet_stiffness_element(verts, 1.0)
        np.testing.assert_allclose(K, K.T, atol=1e-6)

    def test_stiffness_row_sums_zero(self):
        """Laplacian property: rows sum to zero."""
        from neurojax.geometry.fem_forward import tet_stiffness_element
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=jnp.float32)
        K = tet_stiffness_element(verts, 1.0)
        np.testing.assert_allclose(K.sum(axis=1), 0.0, atol=1e-5)

    def test_stiffness_scales_with_sigma(self):
        from neurojax.geometry.fem_forward import tet_stiffness_element
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=jnp.float32)
        K1 = tet_stiffness_element(verts, 1.0)
        K2 = tet_stiffness_element(verts, 2.0)
        np.testing.assert_allclose(K2, 2.0 * K1, atol=1e-6)

    def test_stiffness_differentiable(self):
        from neurojax.geometry.fem_forward import tet_stiffness_element
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=jnp.float32)
        grad = jax.grad(lambda s: jnp.sum(tet_stiffness_element(verts, s) ** 2))(1.0)
        assert jnp.isfinite(grad)


class TestGlobalAssembly:
    """Global stiffness matrix assembly."""

    def test_assembly_shape(self):
        from neurojax.geometry.fem_forward import assemble_stiffness
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]], dtype=jnp.float32)
        elems = jnp.array([[0,1,2,3],[1,2,3,4]], dtype=jnp.int32)
        sigma = jnp.array([1.0, 1.0])
        K = assemble_stiffness(verts, elems, sigma)
        assert K.shape == (5, 5)

    def test_assembly_symmetric(self):
        from neurojax.geometry.fem_forward import assemble_stiffness
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]], dtype=jnp.float32)
        elems = jnp.array([[0,1,2,3],[1,2,3,4]], dtype=jnp.int32)
        sigma = jnp.array([1.0, 1.0])
        K = assemble_stiffness(verts, elems, sigma)
        np.testing.assert_allclose(K, K.T, atol=1e-5)


class TestDipoleSource:
    """Dipole right-hand side computation."""

    def test_dipole_rhs_shape(self):
        from neurojax.geometry.fem_forward import dipole_rhs
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=jnp.float32)
        elems = jnp.array([[0,1,2,3]], dtype=jnp.int32)
        f = dipole_rhs(verts, elems, jnp.array([0.2, 0.2, 0.2]), jnp.array([0,0,1.0]))
        assert f.shape == (4,)

    def test_dipole_rhs_sums_to_zero(self):
        """Current conservation: total injected current = 0."""
        from neurojax.geometry.fem_forward import dipole_rhs
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=jnp.float32)
        elems = jnp.array([[0,1,2,3]], dtype=jnp.int32)
        f = dipole_rhs(verts, elems, jnp.array([0.2, 0.2, 0.2]), jnp.array([0,0,1.0]))
        np.testing.assert_allclose(float(f.sum()), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# RED phase: tensor conductivity from DTI
# ---------------------------------------------------------------------------

class TestTensorConductivity:
    """Anisotropic conductivity from DTI eigenvectors.

    WHITE MATTER conductivity is anisotropic: ~1.7 S/m along fibers,
    ~0.1 S/m perpendicular. The conductivity tensor at each element is:

        sigma_tensor = sigma_par * e1 e1^T + sigma_perp * (I - e1 e1^T)

    where e1 is the primary eigenvector from DTI.
    """

    def test_tensor_element_stiffness_shape(self):
        """Tensor stiffness should be 4x4 like scalar."""
        from neurojax.geometry.fem_forward import tet_stiffness_tensor
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=jnp.float32)
        sigma_tensor = jnp.eye(3) * 0.3  # isotropic as tensor
        K = tet_stiffness_tensor(verts, sigma_tensor)
        assert K.shape == (4, 4)

    def test_tensor_isotropic_matches_scalar(self):
        """Isotropic tensor should give same result as scalar."""
        from neurojax.geometry.fem_forward import tet_stiffness_element, tet_stiffness_tensor
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=jnp.float32)
        K_scalar = tet_stiffness_element(verts, 0.3)
        K_tensor = tet_stiffness_tensor(verts, jnp.eye(3) * 0.3)
        np.testing.assert_allclose(K_tensor, K_scalar, atol=1e-5)

    def test_tensor_anisotropic_differs(self):
        """Anisotropic tensor should differ from isotropic."""
        from neurojax.geometry.fem_forward import tet_stiffness_tensor
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=jnp.float32)
        sigma_iso = jnp.eye(3) * 0.3
        sigma_aniso = jnp.diag(jnp.array([1.7, 0.1, 0.1]))  # fiber along x
        K_iso = tet_stiffness_tensor(verts, sigma_iso)
        K_aniso = tet_stiffness_tensor(verts, sigma_aniso)
        assert not jnp.allclose(K_iso, K_aniso)

    def test_tensor_symmetric(self):
        from neurojax.geometry.fem_forward import tet_stiffness_tensor
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=jnp.float32)
        sigma = jnp.diag(jnp.array([1.7, 0.1, 0.1]))
        K = tet_stiffness_tensor(verts, sigma)
        np.testing.assert_allclose(K, K.T, atol=1e-6)

    def test_tensor_row_sums_zero(self):
        from neurojax.geometry.fem_forward import tet_stiffness_tensor
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=jnp.float32)
        sigma = jnp.diag(jnp.array([1.7, 0.1, 0.1]))
        K = tet_stiffness_tensor(verts, sigma)
        np.testing.assert_allclose(K.sum(axis=1), 0.0, atol=1e-5)

    def test_tensor_from_dti(self):
        """Build conductivity tensor from DTI eigenvectors."""
        from neurojax.geometry.fem_forward import conductivity_tensor_from_dti
        # Primary eigenvector along x, FA = 0.8
        e1 = jnp.array([1.0, 0.0, 0.0])
        sigma_par = 1.7   # S/m along fiber
        sigma_perp = 0.1  # S/m perpendicular
        sigma = conductivity_tensor_from_dti(e1, sigma_par, sigma_perp)
        assert sigma.shape == (3, 3)
        # Should be diagonal with [1.7, 0.1, 0.1] for this eigenvector
        np.testing.assert_allclose(sigma[0, 0], sigma_par, atol=1e-5)
        np.testing.assert_allclose(sigma[1, 1], sigma_perp, atol=1e-5)
        np.testing.assert_allclose(sigma[2, 2], sigma_perp, atol=1e-5)

    def test_tensor_differentiable(self):
        from neurojax.geometry.fem_forward import tet_stiffness_tensor
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=jnp.float32)
        def loss(s_par):
            sigma = jnp.diag(jnp.array([s_par, 0.1, 0.1]))
            K = tet_stiffness_tensor(verts, sigma)
            return jnp.sum(K ** 2)
        grad = jax.grad(loss)(1.7)
        assert jnp.isfinite(grad)


# ---------------------------------------------------------------------------
# RED phase: subtraction method for dipole singularity
# ---------------------------------------------------------------------------

class TestSubtractionMethod:
    """Subtraction approach for dipole singularity (Fijee heritage).

    The direct FEM approach has a singularity at the dipole location.
    The subtraction method splits the potential into:

        phi = phi_0 + phi_corr

    where phi_0 is the analytical solution in a homogeneous sphere
    (singular but known analytically) and phi_corr is the correction
    due to conductivity inhomogeneity (smooth, solvable by FEM).

    This avoids mesh refinement around the dipole.
    """

    def test_analytical_dipole_potential_shape(self):
        """Compute phi_0 at mesh nodes from a dipole in homogeneous medium."""
        from neurojax.geometry.fem_forward import analytical_dipole_potential
        nodes = jnp.array([[0,0,0.05],[0.1,0,0],[0,0.1,0],[0,0,0.1]], dtype=jnp.float32)
        dipole_pos = jnp.array([0.0, 0.0, 0.0])
        dipole_mom = jnp.array([0.0, 0.0, 1.0])
        sigma_hom = 0.3
        phi_0 = analytical_dipole_potential(nodes, dipole_pos, dipole_mom, sigma_hom)
        assert phi_0.shape == (4,)
        assert jnp.all(jnp.isfinite(phi_0))

    def test_analytical_potential_decays_with_distance(self):
        """Potential should decay as 1/r^2 for a dipole."""
        from neurojax.geometry.fem_forward import analytical_dipole_potential
        dipole_pos = jnp.array([0.0, 0.0, 0.0])
        dipole_mom = jnp.array([0.0, 0.0, 1.0])
        near = jnp.array([[0, 0, 0.01]])
        far = jnp.array([[0, 0, 0.1]])
        phi_near = analytical_dipole_potential(near, dipole_pos, dipole_mom, 0.3)
        phi_far = analytical_dipole_potential(far, dipole_pos, dipole_mom, 0.3)
        # 10x further → ~100x smaller potential
        ratio = float(jnp.abs(phi_near[0]) / jnp.abs(phi_far[0]))
        assert ratio > 50  # approximately 100

    def test_subtraction_rhs_shape(self):
        """RHS for correction equation should have shape (n_nodes,)."""
        from neurojax.geometry.fem_forward import subtraction_rhs
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0.5]], dtype=jnp.float32)
        elems = jnp.array([[0,1,2,3],[1,2,3,4]], dtype=jnp.int32)
        sigma = jnp.array([0.3, 0.1])  # inhomogeneous
        sigma_hom = 0.3
        dipole_pos = jnp.array([0.3, 0.3, 0.3])
        dipole_mom = jnp.array([0.0, 0.0, 1.0])
        f_corr = subtraction_rhs(verts, elems, sigma, sigma_hom, dipole_pos, dipole_mom)
        assert f_corr.shape == (5,)

    def test_subtraction_zero_for_homogeneous(self):
        """If sigma is uniform, correction RHS should be zero."""
        from neurojax.geometry.fem_forward import subtraction_rhs
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=jnp.float32)
        elems = jnp.array([[0,1,2,3]], dtype=jnp.int32)
        sigma = jnp.array([0.3])
        sigma_hom = 0.3
        dipole_pos = jnp.array([0.2, 0.2, 0.2])
        dipole_mom = jnp.array([0.0, 0.0, 1.0])
        f_corr = subtraction_rhs(verts, elems, sigma, sigma_hom, dipole_pos, dipole_mom)
        np.testing.assert_allclose(f_corr, 0.0, atol=1e-4)

    def test_subtraction_differentiable(self):
        """Gradients should flow through the subtraction method."""
        from neurojax.geometry.fem_forward import subtraction_rhs
        verts = jnp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=jnp.float32)
        elems = jnp.array([[0,1,2,3]], dtype=jnp.int32)
        dipole_pos = jnp.array([0.2, 0.2, 0.2])
        dipole_mom = jnp.array([0.0, 0.0, 1.0])
        def loss(sigma_val):
            sigma = jnp.array([sigma_val])
            f = subtraction_rhs(verts, elems, sigma, 0.3, dipole_pos, dipole_mom)
            return jnp.sum(f ** 2)
        grad = jax.grad(loss)(0.1)
        assert jnp.isfinite(grad)
