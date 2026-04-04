"""Differentiable FEM forward model for EEG/MEG in pure JAX.

Solves the quasi-static Laplace equation ∇·(σ∇φ) = -I for the
electric potential due to current dipole sources in a heterogeneous
volume conductor. The entire pipeline — stiffness assembly, solve,
leadfield extraction — is differentiable via JAX autodiff.

No external FEM framework required (no PETSc, FEniCS, or gmsh).
Uses tetrahedral elements with linear (P1) basis functions.

The forward model:
    K(σ) · φ = f(dipole)     [stiffness × potential = source]
    L[:, j] = φ_sensors       [leadfield column = sensor potentials]

Because K depends on σ, and σ can be parameterised by qMRI features,
gradients ∂L/∂σ flow through the entire computation.

References:
    Wolters et al. (2004) NeuroImage — FEM for EEG forward
    Vorwerk et al. (2014) NeuroImage — comparison BEM vs FEM
    Rullmann et al. (2009) IEEE TBME — anisotropic FEM
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional
from functools import partial


def tet_stiffness_element(vertices: jnp.ndarray,
                          sigma: float) -> jnp.ndarray:
    """Compute 4×4 element stiffness matrix for a tetrahedron.

    For linear (P1) basis functions on a tetrahedron with vertices
    v0, v1, v2, v3, the stiffness matrix is:

        K_ij = σ · V · (∇N_i · ∇N_j)

    where V is the element volume and ∇N_i are constant gradients.

    Args:
        vertices: (4, 3) tetrahedron vertex coordinates
        sigma: scalar conductivity for this element

    Returns:
        (4, 4) element stiffness matrix
    """
    # Edge vectors from v0
    d = vertices[1:] - vertices[0]  # (3, 3)

    # Volume = |det(d)| / 6
    det = jnp.linalg.det(d)
    vol = jnp.abs(det) / 6.0

    # Gradient of basis functions (constant per element for P1)
    # ∇N_i = cofactor_row_i / det
    # For a tet, the gradients are columns of inv(d)^T / det
    d_inv = jnp.linalg.inv(d)  # (3, 3)

    # Gradients of N1, N2, N3 (N0 = 1 - N1 - N2 - N3)
    grad_N = d_inv.T  # (3, 3) — each column is ∇N_{i+1}
    grad_N0 = -jnp.sum(grad_N, axis=1)  # ∇N_0

    # Assemble all 4 gradients: (4, 3)
    grads = jnp.vstack([grad_N0[None, :], grad_N.T])

    # Element stiffness: K_ij = σ · V · (∇N_i · ∇N_j)
    K_e = sigma * vol * (grads @ grads.T)

    return K_e


# Vectorised over all elements
_tet_stiffness_batch = jax.vmap(tet_stiffness_element, in_axes=(0, 0))


def assemble_stiffness(vertices: jnp.ndarray,
                       elements: jnp.ndarray,
                       sigma: jnp.ndarray) -> jnp.ndarray:
    """Assemble global stiffness matrix from tetrahedral elements.

    Args:
        vertices: (n_vertices, 3) node coordinates
        elements: (n_elements, 4) tetrahedral connectivity (vertex indices)
        sigma: (n_elements,) per-element conductivity

    Returns:
        (n_vertices, n_vertices) global stiffness matrix (dense)

    Note:
        For large meshes (>50K nodes), use sparse assembly instead.
        This dense version is simpler and fully JIT-compatible.
    """
    n_verts = vertices.shape[0]
    n_elems = elements.shape[0]

    # Gather element vertices: (n_elems, 4, 3)
    elem_verts = vertices[elements]

    # Compute all element stiffness matrices: (n_elems, 4, 4)
    K_local = _tet_stiffness_batch(elem_verts, sigma)

    # Assemble into global matrix
    K = jnp.zeros((n_verts, n_verts))
    for i in range(4):
        for j in range(4):
            K = K.at[elements[:, i], elements[:, j]].add(K_local[:, i, j])

    return K


@partial(jax.jit, static_argnums=(4,))
def solve_forward(vertices: jnp.ndarray,
                  elements: jnp.ndarray,
                  sigma: jnp.ndarray,
                  source_rhs: jnp.ndarray,
                  use_cg: bool = True) -> jnp.ndarray:
    """Solve Kφ = f for the electric potential.

    Args:
        vertices: (n_vertices, 3) node coordinates
        elements: (n_elements, 4) tetrahedral connectivity
        sigma: (n_elements,) per-element conductivity
        source_rhs: (n_vertices,) right-hand side (current source)
        use_cg: if True, use conjugate gradient; else direct solve

    Returns:
        (n_vertices,) potential at each node
    """
    K = assemble_stiffness(vertices, elements, sigma)

    # Regularise (pin reference potential)
    K = K + jnp.eye(K.shape[0]) * 1e-10

    if use_cg:
        phi, _ = jax.scipy.sparse.linalg.cg(
            lambda x: K @ x, source_rhs, maxiter=1000, tol=1e-8
        )
    else:
        phi = jnp.linalg.solve(K, source_rhs)

    return phi


def dipole_rhs(vertices: jnp.ndarray,
               elements: jnp.ndarray,
               dipole_pos: jnp.ndarray,
               dipole_mom: jnp.ndarray) -> jnp.ndarray:
    """Compute right-hand side for a current dipole source.

    A dipole at position p with moment q produces a source term:
        f_i = q · ∇N_i(p)
    evaluated in the element containing p.

    Args:
        vertices: (n_vertices, 3)
        elements: (n_elements, 4)
        dipole_pos: (3,) dipole position
        dipole_mom: (3,) dipole moment (nAm)

    Returns:
        (n_vertices,) right-hand side vector
    """
    n_verts = vertices.shape[0]

    # Find element containing the dipole (nearest element centroid)
    centroids = jnp.mean(vertices[elements], axis=1)  # (n_elem, 3)
    dists = jnp.sum((centroids - dipole_pos) ** 2, axis=1)
    elem_idx = jnp.argmin(dists)

    # Get element vertices and compute gradients
    ev = vertices[elements[elem_idx]]  # (4, 3)
    d = ev[1:] - ev[0]
    d_inv = jnp.linalg.inv(d)
    grad_N = d_inv.T  # (3, 3)
    grad_N0 = -jnp.sum(grad_N, axis=1)
    grads = jnp.vstack([grad_N0[None, :], grad_N.T])  # (4, 3)

    # f_i = q · ∇N_i
    f_local = grads @ dipole_mom  # (4,)

    # Scatter to global
    f = jnp.zeros(n_verts)
    f = f.at[elements[elem_idx]].add(f_local)
    return f


def compute_leadfield(vertices: jnp.ndarray,
                      elements: jnp.ndarray,
                      sigma: jnp.ndarray,
                      source_positions: jnp.ndarray,
                      sensor_indices: jnp.ndarray) -> jnp.ndarray:
    """Compute the full leadfield matrix via FEM.

    Solves the forward problem for each dipole source (3 orientations)
    and extracts the potential at sensor locations.

    Args:
        vertices: (n_vertices, 3) FEM mesh nodes
        elements: (n_elements, 4) tetrahedral connectivity
        sigma: (n_elements,) per-element conductivity
        source_positions: (n_sources, 3) dipole locations
        sensor_indices: (n_sensors,) vertex indices of sensors

    Returns:
        (n_sensors, n_sources * 3) leadfield matrix

    Note:
        Memory estimate: dense K is n_verts² × 8 bytes.
        For 10K nodes → 800 MB. For >50K, use sparse assembly.
    """
    import psutil
    n_verts = vertices.shape[0]
    mem_gb = n_verts ** 2 * 8 / 1e9
    avail_gb = psutil.virtual_memory().available / 1e9
    if mem_gb > avail_gb * 0.5:
        raise MemoryError(
            f"FEM stiffness matrix needs ~{mem_gb:.1f} GB "
            f"(available: {avail_gb:.1f} GB). Reduce mesh size."
        )

    n_sources = source_positions.shape[0]
    n_sensors = sensor_indices.shape[0]

    # Pre-assemble stiffness (reused for all sources)
    K = assemble_stiffness(vertices, elements, sigma)
    K = K + jnp.eye(n_verts) * 1e-10

    # Solve for each dipole (3 orientations)
    L = jnp.zeros((n_sensors, n_sources * 3))

    for i in range(n_sources):
        for ax in range(3):
            moment = jnp.zeros(3).at[ax].set(1.0)
            f = dipole_rhs(vertices, elements,
                           source_positions[i], moment)
            phi = jnp.linalg.solve(K, f)
            L = L.at[:, i * 3 + ax].set(phi[sensor_indices])

    return L


def sigma_from_qmri(t1_values: jnp.ndarray,
                     bpf_values: jnp.ndarray,
                     tissue_labels: jnp.ndarray,
                     params: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Differentiable conductivity model from qMRI features.

    Maps qMRI measurements to tissue conductivity. The mapping is
    parameterised so it can be optimised end-to-end.

    Args:
        t1_values: (n_elements,) T1 values in seconds
        bpf_values: (n_elements,) bound pool fraction
        tissue_labels: (n_elements,) integer tissue labels
        params: optional (n_params,) learnable parameters

    Returns:
        (n_elements,) conductivity in S/m
    """
    if params is None:
        # Default parameters: [σ_brain, σ_csf, σ_bone_scale, σ_scalp, bpf_exponent]
        params = jnp.array([0.33, 1.654, 0.01, 0.465, 4.0])

    sigma_brain, sigma_csf, sigma_bone_base, sigma_scalp, bpf_exp = params

    # Default: use label-based conductivity
    sigma = jnp.where(tissue_labels == 0, 0.0,       # unknown
            jnp.where(tissue_labels == 1, sigma_brain, # brain
            jnp.where(tissue_labels == 2, sigma_csf,   # CSF
            jnp.where(tissue_labels == 3,               # bone
                       sigma_bone_base * (1.0 - bpf_values) ** bpf_exp,
            jnp.where(tissue_labels == 4, sigma_scalp,  # scalp
                       0.25)))))  # other

    return jnp.clip(sigma, 0.001, 2.0)


# ---------------------------------------------------------------------------
# Tensor (anisotropic) conductivity from DTI
# ---------------------------------------------------------------------------

def conductivity_tensor_from_dti(e1: jnp.ndarray,
                                  sigma_par: float,
                                  sigma_perp: float) -> jnp.ndarray:
    """Build conductivity tensor from DTI primary eigenvector.

    sigma = sigma_par * e1 @ e1^T + sigma_perp * (I - e1 @ e1^T)

    Args:
        e1: (3,) primary eigenvector (fiber direction), unit vector
        sigma_par: conductivity along fiber (S/m), typically ~1.7
        sigma_perp: conductivity perpendicular (S/m), typically ~0.1

    Returns:
        (3, 3) symmetric positive-definite conductivity tensor
    """
    e1 = e1 / jnp.maximum(jnp.linalg.norm(e1), 1e-10)
    proj = jnp.outer(e1, e1)
    return sigma_par * proj + sigma_perp * (jnp.eye(3) - proj)


def tet_stiffness_tensor(vertices: jnp.ndarray,
                          sigma_tensor: jnp.ndarray) -> jnp.ndarray:
    """Element stiffness matrix with anisotropic conductivity tensor.

    K_ij = V * (∇N_i)^T @ σ @ (∇N_j)

    Args:
        vertices: (4, 3) tetrahedron vertex coordinates
        sigma_tensor: (3, 3) conductivity tensor for this element

    Returns:
        (4, 4) element stiffness matrix
    """
    d = vertices[1:] - vertices[0]
    det = jnp.linalg.det(d)
    vol = jnp.abs(det) / 6.0

    d_inv = jnp.linalg.inv(d)
    grad_N = d_inv.T  # (3, 3)
    grad_N0 = -jnp.sum(grad_N, axis=1)
    grads = jnp.vstack([grad_N0[None, :], grad_N.T])  # (4, 3)

    # K_ij = V * grads[i] @ sigma @ grads[j]
    sigma_grads = grads @ sigma_tensor  # (4, 3)
    K_e = vol * (sigma_grads @ grads.T)

    return K_e


# ---------------------------------------------------------------------------
# Subtraction method for dipole singularity (Fijee heritage)
# ---------------------------------------------------------------------------

def analytical_dipole_potential(nodes: jnp.ndarray,
                                dipole_pos: jnp.ndarray,
                                dipole_mom: jnp.ndarray,
                                sigma: float) -> jnp.ndarray:
    """Analytical potential of a current dipole in a homogeneous conductor.

    phi_0(r) = (1 / 4*pi*sigma) * (q · (r - r_d)) / |r - r_d|^3

    This is the singular primary potential used in the subtraction method.

    Args:
        nodes: (n_nodes, 3) evaluation points
        dipole_pos: (3,) dipole position
        dipole_mom: (3,) dipole moment (nAm)
        sigma: homogeneous conductivity (S/m)

    Returns:
        (n_nodes,) potential at each node
    """
    r = nodes - dipole_pos  # (n_nodes, 3)
    dist = jnp.sqrt(jnp.sum(r ** 2, axis=1))  # (n_nodes,)
    dist = jnp.maximum(dist, 1e-10)  # avoid division by zero

    # q · (r - r_d) / |r - r_d|^3
    qdotr = jnp.sum(r * dipole_mom, axis=1)  # (n_nodes,)
    phi = qdotr / (4.0 * jnp.pi * sigma * dist ** 3)

    return phi


def subtraction_rhs(vertices: jnp.ndarray,
                    elements: jnp.ndarray,
                    sigma: jnp.ndarray,
                    sigma_hom: float,
                    dipole_pos: jnp.ndarray,
                    dipole_mom: jnp.ndarray) -> jnp.ndarray:
    """Right-hand side for the correction equation in the subtraction method.

    The total potential is phi = phi_0 + phi_corr, where phi_0 is the
    analytical solution in a homogeneous conductor with sigma_hom. The
    correction phi_corr satisfies:

        K(sigma) @ phi_corr = -K(sigma - sigma_hom) @ phi_0

    This RHS is zero when sigma is uniform (no correction needed).

    Args:
        vertices: (n_vertices, 3)
        elements: (n_elements, 4)
        sigma: (n_elements,) per-element conductivity
        sigma_hom: reference homogeneous conductivity
        dipole_pos: (3,) dipole position
        dipole_mom: (3,) dipole moment

    Returns:
        (n_vertices,) right-hand side for correction equation
    """
    n_verts = vertices.shape[0]

    # Compute phi_0 at all nodes
    phi_0 = analytical_dipole_potential(vertices, dipole_pos, dipole_mom, sigma_hom)

    # Assemble K(sigma - sigma_hom)
    sigma_diff = sigma - sigma_hom
    K_diff = assemble_stiffness(vertices, elements, sigma_diff)

    # RHS = -K_diff @ phi_0
    return -K_diff @ phi_0
