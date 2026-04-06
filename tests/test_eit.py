"""
Tests for neurojax EIT (Electrical Impedance Tomography) module.

Tests cover:
- SCI head model loader (synthetic spherical model)
- Complete Electrode Model (CEM) assembly
- EIT forward solver
- Injection/measurement protocols
- Jacobian computation
- Inverse reconstruction (Tikhonov, NOSER, TV)
- Difference imaging
- JAX-differentiable forward
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest
import numpy as np
from numpy.testing import assert_allclose

import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spherical_head():
    """Small spherical head model for fast tests."""
    from neurojax.geometry.sci_head_model import make_spherical_head
    return make_spherical_head(
        radii=(60.0, 70.0, 78.0, 85.0),
        conductivities=(0.33, 1.79, 0.006, 0.43),
        n_electrodes=16,
        mesh_density=20.0,
    )


@pytest.fixture
def simple_protocol():
    """Adjacent protocol for 16 electrodes."""
    from neurojax.geometry.eit import adjacent_pattern
    return adjacent_pattern(16)


# ===================================================================
# SCI Head Model loader tests
# ===================================================================

class TestSCIHeadModel:
    """Tests for neurojax.geometry.sci_head_model."""

    def test_import(self):
        from neurojax.geometry.sci_head_model import (
            load_mesh, load_electrodes, make_spherical_head,
            HeadMesh, SCI_TISSUE, snap_electrodes_to_surface,
        )
        assert callable(load_mesh)
        assert callable(make_spherical_head)
        assert len(SCI_TISSUE) == 8

    def test_spherical_head_shape(self, spherical_head):
        h = spherical_head
        assert h.vertices.ndim == 2
        assert h.vertices.shape[1] == 3
        assert h.elements.ndim == 2
        assert h.elements.shape[1] == 4
        assert len(h.tissue_labels) == len(h.elements)
        assert len(h.conductivity) == len(h.elements)
        assert len(h.electrode_pos) == 16
        assert len(h.electrode_nodes) == 16

    def test_spherical_head_tissues(self, spherical_head):
        h = spherical_head
        unique_labels = np.unique(h.tissue_labels)
        # Should have at least 3 tissue types (coarse mesh may merge thin shells)
        assert len(unique_labels) >= 3
        # Conductivity should be positive for brain tissue
        brain_mask = h.tissue_labels == 0
        assert np.all(h.conductivity[brain_mask] > 0)

    def test_electrode_nodes_valid(self, spherical_head):
        h = spherical_head
        assert np.all(h.electrode_nodes >= 0)
        assert np.all(h.electrode_nodes < len(h.vertices))
        # Electrode positions should match node positions
        for i in range(len(h.electrode_nodes)):
            node_pos = h.vertices[h.electrode_nodes[i]]
            assert_allclose(h.electrode_pos[i], node_pos)

    def test_electrodes_on_outer_surface(self, spherical_head):
        h = spherical_head
        # Electrodes should be near the outer radius (85 mm)
        r = np.sqrt(np.sum(h.electrode_pos ** 2, axis=1))
        assert np.all(r > 60.0)  # not inside the brain

    def test_tissue_map(self, spherical_head):
        h = spherical_head
        assert 0 in h.tissue_map
        assert h.tissue_map[0] == "Brain"


# ===================================================================
# EIT Protocol tests
# ===================================================================

class TestProtocols:
    """Tests for EIT injection/measurement protocols."""

    def test_adjacent_pattern(self):
        from neurojax.geometry.eit import adjacent_pattern
        p = adjacent_pattern(16)
        assert p.injection.shape == (16, 16)
        assert p.description == "adjacent"
        # Each injection row should sum to 0 (current conservation)
        assert_allclose(p.injection.sum(axis=1), 0.0, atol=1e-10)
        # Measurements should also be differential (sum to 0)
        assert_allclose(p.measurement.sum(axis=1), 0.0, atol=1e-10)

    def test_opposite_pattern(self):
        from neurojax.geometry.eit import opposite_pattern
        p = opposite_pattern(16)
        assert p.injection.shape[0] == 8  # L/2 patterns
        assert p.injection.shape[1] == 16
        assert_allclose(p.injection.sum(axis=1), 0.0, atol=1e-10)

    def test_trigonometric_pattern(self):
        from neurojax.geometry.eit import trigonometric_pattern
        p = trigonometric_pattern(16)
        assert p.injection.shape == (15, 16)  # L-1 patterns
        # Patterns should be normalised
        assert np.all(np.max(np.abs(p.injection), axis=1) <= 1.0 + 1e-10)


# ===================================================================
# CEM Assembly tests
# ===================================================================

class TestCEMAssembly:
    """Tests for Complete Electrode Model FEM assembly."""

    def test_sparse_stiffness(self, spherical_head):
        from neurojax.geometry.eit import assemble_stiffness_sparse
        h = spherical_head
        K = assemble_stiffness_sparse(h.vertices, h.elements, h.conductivity)
        n = len(h.vertices)
        assert K.shape == (n, n)
        # Symmetric
        assert_allclose(K.toarray(), K.T.toarray(), atol=1e-10)
        # Positive semi-definite (eigenvalues >= 0)
        eigvals = np.linalg.eigvalsh(K.toarray())
        assert np.all(eigvals >= -1e-8)

    def test_cem_system_shape(self, spherical_head):
        from neurojax.geometry.eit import assemble_cem_system
        h = spherical_head
        system, _ = assemble_cem_system(
            h.vertices, h.elements, h.conductivity, h.electrode_nodes
        )
        n_total = len(h.vertices) + len(h.electrode_nodes)
        assert system.shape == (n_total, n_total)

    def test_cem_system_symmetric(self, spherical_head):
        from neurojax.geometry.eit import assemble_cem_system
        h = spherical_head
        system, _ = assemble_cem_system(
            h.vertices, h.elements, h.conductivity, h.electrode_nodes
        )
        dense = system.toarray()
        assert_allclose(dense, dense.T, atol=1e-10)


# ===================================================================
# Forward solver tests
# ===================================================================

class TestForwardSolver:
    """Tests for EIT forward solver."""

    def test_forward_runs(self, spherical_head, simple_protocol):
        from neurojax.geometry.eit import solve_eit_forward
        h = spherical_head
        result = solve_eit_forward(
            h.vertices, h.elements, h.conductivity,
            h.electrode_nodes, simple_protocol,
        )
        assert result.node_potentials.shape[0] == len(simple_protocol.injection)
        assert result.node_potentials.shape[1] == len(h.vertices)
        assert result.electrode_voltages.shape == (
            len(simple_protocol.injection), len(h.electrode_nodes)
        )
        assert len(result.measurements) == len(simple_protocol.measurement)

    def test_forward_current_conservation(self, spherical_head, simple_protocol):
        from neurojax.geometry.eit import solve_eit_forward
        h = spherical_head
        result = solve_eit_forward(
            h.vertices, h.elements, h.conductivity,
            h.electrode_nodes, simple_protocol,
        )
        # Electrode voltages should have approximately zero mean (ground)
        for p in range(len(simple_protocol.injection)):
            mean_v = np.mean(result.electrode_voltages[p])
            assert abs(mean_v) < 0.1, f"Pattern {p}: mean voltage {mean_v}"

    def test_forward_reciprocity(self, spherical_head):
        """Reciprocity: V measured at B due to I at A = V at A due to I at B."""
        from neurojax.geometry.eit import solve_eit_forward, EITProtocol
        h = spherical_head
        n_elec = len(h.electrode_nodes)

        # Pattern A: inject 0→1, measure 2→3
        inj_a = np.zeros((1, n_elec))
        inj_a[0, 0] = 1.0
        inj_a[0, 1] = -1.0
        meas_a = np.zeros((1, n_elec))
        meas_a[0, 2] = 1.0
        meas_a[0, 3] = -1.0

        # Pattern B: inject 2→3, measure 0→1
        inj_b = np.zeros((1, n_elec))
        inj_b[0, 2] = 1.0
        inj_b[0, 3] = -1.0
        meas_b = np.zeros((1, n_elec))
        meas_b[0, 0] = 1.0
        meas_b[0, 1] = -1.0

        prot_a = EITProtocol(inj_a, meas_a, "reciprocity_a")
        prot_b = EITProtocol(inj_b, meas_b, "reciprocity_b")

        result_a = solve_eit_forward(
            h.vertices, h.elements, h.conductivity,
            h.electrode_nodes, prot_a,
        )
        result_b = solve_eit_forward(
            h.vertices, h.elements, h.conductivity,
            h.electrode_nodes, prot_b,
        )

        # V_AB ≈ V_BA (reciprocity)
        assert_allclose(
            result_a.measurements, result_b.measurements,
            rtol=0.1,  # 10% tolerance for coarse mesh
        )

    def test_perturbation_changes_voltages(self, spherical_head, simple_protocol):
        from neurojax.geometry.eit import (
            solve_eit_forward, simulate_perturbation,
        )
        h = spherical_head

        # Reference
        result_ref = solve_eit_forward(
            h.vertices, h.elements, h.conductivity,
            h.electrode_nodes, simple_protocol,
        )

        # Add a conductive perturbation inside the brain
        sigma_pert = simulate_perturbation(
            h.vertices, h.elements, h.conductivity,
            perturbation_centre=np.array([0.0, 0.0, 0.0]),
            perturbation_radius=20.0,
            perturbation_value=1.0,  # much more conductive
        )
        result_pert = solve_eit_forward(
            h.vertices, h.elements, sigma_pert,
            h.electrode_nodes, simple_protocol,
        )

        # Voltages should differ
        delta = np.abs(result_pert.measurements - result_ref.measurements)
        assert np.max(delta) > 0, "Perturbation should change measurements"


# ===================================================================
# Jacobian tests
# ===================================================================

class TestJacobian:
    """Tests for Jacobian computation."""

    def test_jacobian_shape(self, spherical_head, simple_protocol):
        from neurojax.geometry.eit import solve_eit_forward, compute_jacobian
        h = spherical_head
        fwd = solve_eit_forward(
            h.vertices, h.elements, h.conductivity,
            h.electrode_nodes, simple_protocol,
        )
        J = compute_jacobian(
            h.vertices, h.elements, h.conductivity,
            h.electrode_nodes, simple_protocol,
            forward_result=fwd,
        )
        assert J.shape == (len(simple_protocol.measurement), len(h.elements))

    def test_jacobian_nonzero(self, spherical_head, simple_protocol):
        from neurojax.geometry.eit import solve_eit_forward, compute_jacobian
        h = spherical_head
        fwd = solve_eit_forward(
            h.vertices, h.elements, h.conductivity,
            h.electrode_nodes, simple_protocol,
        )
        J = compute_jacobian(
            h.vertices, h.elements, h.conductivity,
            h.electrode_nodes, simple_protocol,
            forward_result=fwd,
        )
        # Jacobian should have non-zero entries
        assert np.count_nonzero(J) > 0


# ===================================================================
# Inverse solver tests
# ===================================================================

class TestInverseSolver:
    """Tests for EIT inverse reconstruction."""

    def test_tikhonov_reconstruction(self, spherical_head, simple_protocol):
        from neurojax.geometry.eit import (
            solve_eit_forward, compute_jacobian,
            difference_eit, simulate_perturbation,
        )
        h = spherical_head

        # Reference forward
        fwd_ref = solve_eit_forward(
            h.vertices, h.elements, h.conductivity,
            h.electrode_nodes, simple_protocol,
        )

        # Jacobian at reference
        J = compute_jacobian(
            h.vertices, h.elements, h.conductivity,
            h.electrode_nodes, simple_protocol,
            forward_result=fwd_ref,
        )

        # Perturbed forward
        sigma_pert = simulate_perturbation(
            h.vertices, h.elements, h.conductivity,
            perturbation_centre=np.array([0.0, 0.0, 0.0]),
            perturbation_radius=20.0,
            perturbation_value=0.66,
        )
        fwd_pert = solve_eit_forward(
            h.vertices, h.elements, sigma_pert,
            h.electrode_nodes, simple_protocol,
        )

        # Reconstruct
        delta_sigma = difference_eit(
            fwd_ref.measurements, fwd_pert.measurements, J,
            method="tikhonov", alpha=0.1, normalise=False,
        )

        assert delta_sigma.shape == (len(h.elements),)
        # Reconstruction should be non-trivial
        assert np.std(delta_sigma) > 0

    def test_noser_reconstruction(self, spherical_head, simple_protocol):
        from neurojax.geometry.eit import (
            solve_eit_forward, compute_jacobian,
            solve_eit_inverse, simulate_perturbation,
        )
        h = spherical_head

        fwd_ref = solve_eit_forward(
            h.vertices, h.elements, h.conductivity,
            h.electrode_nodes, simple_protocol,
        )
        J = compute_jacobian(
            h.vertices, h.elements, h.conductivity,
            h.electrode_nodes, simple_protocol,
            forward_result=fwd_ref,
        )

        sigma_pert = simulate_perturbation(
            h.vertices, h.elements, h.conductivity,
            perturbation_centre=np.array([20.0, 0.0, 0.0]),
            perturbation_radius=15.0,
            perturbation_value=0.66,
        )
        fwd_pert = solve_eit_forward(
            h.vertices, h.elements, sigma_pert,
            h.electrode_nodes, simple_protocol,
        )

        delta_v = fwd_pert.measurements - fwd_ref.measurements
        delta_sigma = solve_eit_inverse(
            J, delta_v, method="noser", alpha=0.01,
            sigma_ref=h.conductivity,
        )
        assert delta_sigma.shape == (len(h.elements),)

    def test_tv_reconstruction(self, spherical_head, simple_protocol):
        from neurojax.geometry.eit import (
            solve_eit_forward, compute_jacobian,
            solve_eit_inverse, simulate_perturbation,
        )
        h = spherical_head

        fwd_ref = solve_eit_forward(
            h.vertices, h.elements, h.conductivity,
            h.electrode_nodes, simple_protocol,
        )
        J = compute_jacobian(
            h.vertices, h.elements, h.conductivity,
            h.electrode_nodes, simple_protocol,
            forward_result=fwd_ref,
        )

        sigma_pert = simulate_perturbation(
            h.vertices, h.elements, h.conductivity,
            perturbation_centre=np.array([0.0, 20.0, 0.0]),
            perturbation_radius=15.0,
            perturbation_value=0.66,
        )
        fwd_pert = solve_eit_forward(
            h.vertices, h.elements, sigma_pert,
            h.electrode_nodes, simple_protocol,
        )

        delta_v = fwd_pert.measurements - fwd_ref.measurements
        delta_sigma = solve_eit_inverse(
            J, delta_v, method="tv", alpha=0.01, max_iter=5,
        )
        assert delta_sigma.shape == (len(h.elements),)


# ===================================================================
# Element Laplacian tests
# ===================================================================

class TestElementLaplacian:
    """Tests for spatial regularisation Laplacian."""

    def test_laplacian_shape(self, spherical_head):
        from neurojax.geometry.eit import element_laplacian
        L = element_laplacian(spherical_head.elements)
        n = len(spherical_head.elements)
        assert L.shape == (n, n)

    def test_laplacian_symmetric(self, spherical_head):
        from neurojax.geometry.eit import element_laplacian
        L = element_laplacian(spherical_head.elements)
        assert_allclose(L, L.T, atol=1e-10)

    def test_laplacian_row_sum_zero(self, spherical_head):
        from neurojax.geometry.eit import element_laplacian
        L = element_laplacian(spherical_head.elements)
        # Graph Laplacian: each row sums to 0
        assert_allclose(L.sum(axis=1), 0.0, atol=1e-10)


# ===================================================================
# JAX differentiable forward tests
# ===================================================================

class TestJAXForward:
    """Tests for JAX-differentiable EIT forward."""

    def test_jax_forward_runs(self, spherical_head):
        """JAX forward should produce finite electrode voltages."""
        import jax.numpy as jnp
        from neurojax.geometry.eit import _eit_forward_jax
        h = spherical_head
        n_elec = len(h.electrode_nodes)

        inj = np.zeros(n_elec)
        inj[0] = 1.0
        inj[1] = -1.0

        v_jax = _eit_forward_jax(
            jnp.array(h.vertices),
            jnp.array(h.elements),
            jnp.array(h.conductivity),
            n_elec,
            jnp.array(h.electrode_nodes),
            jnp.array(inj),
        )

        assert v_jax.shape == (n_elec,)
        assert jnp.all(jnp.isfinite(v_jax)), "JAX forward produced non-finite values"

    def test_jax_grad_computes(self, spherical_head):
        """Gradient through JAX forward should compute without error."""
        import jax.numpy as jnp
        from neurojax.geometry.eit import _eit_forward_jax
        h = spherical_head
        n_elec = len(h.electrode_nodes)

        inj = np.zeros(n_elec)
        inj[0] = 1.0
        inj[1] = -1.0

        def loss_fn(sigma):
            v = _eit_forward_jax(
                jnp.array(h.vertices),
                jnp.array(h.elements),
                sigma, n_elec,
                jnp.array(h.electrode_nodes),
                jnp.array(inj),
            )
            return jnp.sum(v ** 2)

        sigma_jnp = jnp.array(h.conductivity)
        grad_sigma = jax.grad(loss_fn)(sigma_jnp)

        assert grad_sigma.shape == sigma_jnp.shape
        assert not jnp.any(jnp.isnan(grad_sigma))


# ===================================================================
# Perturbation simulation tests
# ===================================================================

class TestSimulatePerturbation:
    """Tests for conductivity perturbation simulation."""

    def test_perturbation_localised(self, spherical_head):
        from neurojax.geometry.eit import simulate_perturbation
        h = spherical_head
        sigma_pert = simulate_perturbation(
            h.vertices, h.elements, h.conductivity,
            perturbation_centre=np.array([0.0, 0.0, 0.0]),
            perturbation_radius=15.0,
            perturbation_value=1.0,
        )
        changed = sigma_pert != h.conductivity
        assert np.any(changed), "Some elements should change"
        assert not np.all(changed), "Not all elements should change"

    def test_perturbation_value(self, spherical_head):
        from neurojax.geometry.eit import simulate_perturbation
        h = spherical_head
        target_value = 2.5
        sigma_pert = simulate_perturbation(
            h.vertices, h.elements, h.conductivity,
            perturbation_centre=np.array([0.0, 0.0, 0.0]),
            perturbation_radius=10.0,
            perturbation_value=target_value,
        )
        changed_mask = sigma_pert != h.conductivity
        assert_allclose(sigma_pert[changed_mask], target_value)
