"""Electrical Impedance Tomography (EIT) forward and inverse solvers.

Implements the Complete Electrode Model (CEM) for EIT on tetrahedral
head meshes. The same Laplace equation (∇·σ∇φ = 0) underlies both
EIT and EEG forward modeling — EIT additionally recovers σ from
boundary voltage measurements.

Forward model (CEM):
    ∇·(σ∇u) = 0                           in Ω
    u + z_l σ ∂u/∂n = U_l                 on electrode e_l
    ∫_{e_l} σ ∂u/∂n dS = I_l             current conservation
    σ ∂u/∂n = 0                           on ∂Ω \\ ∪e_l

FEM discretisation yields the augmented system:
    [A_σ + A_z,  A_w ] [φ]   [ 0 ]
    [ A_w^T,    A_d  ] [U] = [ I ]

Inverse problem:
    Reconstruct σ from boundary voltage measurements V using
    linearised Gauss-Newton with Tikhonov regularisation.

References:
    Somersalo E, Cheney M, Isaacson D (1992). Existence and
        uniqueness for electrode models for electric current
        computed tomography. SIAM J Appl Math.
    Vauhkonen M et al. (1999). A MATLAB package for the EIDORS
        project to reconstruct two- and three-dimensional EIT images.
    Adler A, Lionheart WRB (2006). Uses and abuses of EIDORS.
    Holder DS (2005). Electrical Impedance Tomography: Methods,
        History and Applications. IOP Publishing.
"""

import logging
from functools import partial
from typing import Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────

class EITProtocol(NamedTuple):
    """EIT measurement protocol (injection + measurement patterns)."""
    injection: np.ndarray      # (n_patterns, n_electrodes) current injection
    measurement: np.ndarray    # (n_measurements, n_electrodes) measurement
    description: str


class EITForwardResult(NamedTuple):
    """Result of EIT forward computation."""
    node_potentials: np.ndarray   # (n_patterns, n_nodes) potential field
    electrode_voltages: np.ndarray  # (n_patterns, n_electrodes) electrode voltages
    measurements: np.ndarray      # (n_measurements,) voltage differences
    protocol: EITProtocol


# ── Injection / measurement patterns ─────────────────────────────────

def adjacent_pattern(n_electrodes: int) -> EITProtocol:
    """Adjacent (Sheffield) drive/measurement protocol.

    Current injected between adjacent electrode pairs, voltage
    measured on all other adjacent pairs. Standard protocol for
    early EIT systems (e.g., Sheffield Mark I).

    Args:
        n_electrodes: number of electrodes

    Returns:
        EITProtocol with injection and measurement matrices
    """
    L = n_electrodes
    # Injection patterns: [+1, -1] on adjacent pairs
    injection = np.zeros((L, L))
    for i in range(L):
        injection[i, i] = 1.0
        injection[i, (i + 1) % L] = -1.0

    # Measurement: all adjacent pairs excluding drive electrodes
    meas_rows = []
    pattern_idx = []
    for p in range(L):
        drive_a, drive_b = p, (p + 1) % L
        for m in range(L):
            m_a, m_b = m, (m + 1) % L
            if m_a in (drive_a, drive_b) or m_b in (drive_a, drive_b):
                continue
            row = np.zeros(L)
            row[m_a] = 1.0
            row[m_b] = -1.0
            meas_rows.append(row)
            pattern_idx.append(p)

    measurement = np.array(meas_rows)
    return EITProtocol(injection, measurement, "adjacent")


def opposite_pattern(n_electrodes: int) -> EITProtocol:
    """Opposite (polar) drive/measurement protocol.

    Current injected between diametrically opposite electrodes.
    Better depth sensitivity than adjacent for brain EIT.

    Args:
        n_electrodes: number of electrodes

    Returns:
        EITProtocol with injection and measurement matrices
    """
    L = n_electrodes
    half = L // 2
    injection = np.zeros((half, L))
    for i in range(half):
        injection[i, i] = 1.0
        injection[i, i + half] = -1.0

    # Measure between all non-drive pairs
    meas_rows = []
    for p in range(half):
        drive_a, drive_b = p, p + half
        for m in range(L):
            m_next = (m + 1) % L
            if m in (drive_a, drive_b) or m_next in (drive_a, drive_b):
                continue
            row = np.zeros(L)
            row[m] = 1.0
            row[m_next] = -1.0
            meas_rows.append(row)

    measurement = np.array(meas_rows)
    return EITProtocol(injection, measurement, "opposite")


def trigonometric_pattern(n_electrodes: int) -> EITProtocol:
    """Trigonometric current patterns.

    Optimal current patterns that maximise distinguishability
    (Isaacson 1986). Uses sin/cos basis on the electrode ring.

    Args:
        n_electrodes: number of electrodes

    Returns:
        EITProtocol
    """
    L = n_electrodes
    n_modes = L - 1  # L-1 linearly independent patterns (sum = 0)
    injection = np.zeros((n_modes, L))

    for k in range(n_modes):
        freq = k // 2 + 1
        for el in range(L):
            angle = 2 * np.pi * freq * el / L
            if k % 2 == 0:
                injection[k, el] = np.cos(angle)
            else:
                injection[k, el] = np.sin(angle)

    # Normalise to unit current
    injection /= np.max(np.abs(injection), axis=1, keepdims=True)

    # Measurement: same patterns (self-adjoint)
    measurement = injection.copy()

    return EITProtocol(injection, measurement, "trigonometric")


# ── Sparse FEM assembly for CEM ──────────────────────────────────────

def _tet_stiffness_local(vertices: np.ndarray) -> np.ndarray:
    """Compute 4x4 element stiffness matrix (unit conductivity).

    Args:
        vertices: (4, 3) tetrahedron vertex coordinates

    Returns:
        (4, 4) element stiffness matrix for σ=1
    """
    d = vertices[1:] - vertices[0]  # (3, 3)
    det = np.linalg.det(d)
    vol = abs(det) / 6.0

    if abs(det) < 1e-15:
        return np.zeros((4, 4))

    d_inv = np.linalg.inv(d)
    grad_N = d_inv.T  # (3, 3)
    grad_N0 = -np.sum(grad_N, axis=1)
    grads = np.vstack([grad_N0[None, :], grad_N.T])  # (4, 3)

    return vol * (grads @ grads.T)


def _tet_stiffness_batch(
    elem_verts: np.ndarray,
) -> np.ndarray:
    """Batch-compute unit element stiffness matrices for all tetrahedra.

    Fully vectorised — no Python loops over elements.

    Args:
        elem_verts: (n_elements, 4, 3) vertex coordinates per element

    Returns:
        (n_elements, 4, 4) unit-conductivity element stiffness matrices
    """
    n = len(elem_verts)
    # Edge vectors from v0: (n, 3, 3)
    d = elem_verts[:, 1:] - elem_verts[:, 0:1]

    # Determinants and volumes: (n,)
    dets = np.linalg.det(d)
    vols = np.abs(dets) / 6.0

    # Inverse of edge matrix: (n, 3, 3)
    # Guard against degenerate elements
    dets_safe = np.where(np.abs(dets) < 1e-15, 1.0, dets)
    d_safe = d.copy()
    degenerate = np.abs(dets) < 1e-15
    # Replace degenerate elements with identity to avoid singular inverse
    d_safe[degenerate] = np.eye(3)
    d_inv = np.linalg.inv(d_safe)  # (n, 3, 3)

    # Gradients of N1, N2, N3: (n, 3, 3), each column = ∇N_{i+1}
    grad_N = np.transpose(d_inv, (0, 2, 1))  # (n, 3, 3)

    # ∇N_0 = -sum of other gradients: (n, 3)
    grad_N0 = -np.sum(grad_N, axis=2)  # (n, 3)

    # All 4 gradients: (n, 4, 3)
    grads = np.concatenate(
        [grad_N0[:, np.newaxis, :], np.transpose(grad_N, (0, 2, 1))],
        axis=1,
    )

    # Element stiffness: Ke_ij = V * (∇N_i · ∇N_j)
    # K = V * grads @ grads^T: (n, 4, 4)
    Ke = vols[:, np.newaxis, np.newaxis] * np.einsum(
        "nik,njk->nij", grads, grads
    )

    # Zero out degenerate elements
    Ke[degenerate] = 0.0

    return Ke


def assemble_stiffness_sparse(
    vertices: np.ndarray,
    elements: np.ndarray,
    sigma: np.ndarray,
) -> sparse.csr_matrix:
    """Assemble global stiffness matrix in sparse CSR format.

    Fully vectorised assembly — handles millions of elements
    without Python loops.

    Args:
        vertices: (n_vertices, 3) node coordinates
        elements: (n_elements, 4) tetrahedral connectivity
        sigma: (n_elements,) per-element conductivity

    Returns:
        (n_vertices, n_vertices) sparse CSR stiffness matrix
    """
    n_verts = len(vertices)
    n_elems = len(elements)

    # Gather element vertices: (n_elems, 4, 3)
    elem_verts = vertices[elements]

    # Batch stiffness: (n_elems, 4, 4)
    Ke_unit = _tet_stiffness_batch(elem_verts)

    # Scale by conductivity: (n_elems, 4, 4)
    Ke = sigma[:, np.newaxis, np.newaxis] * Ke_unit

    # Build COO arrays vectorised
    # Row/col indices: for each element, 4×4 = 16 entries
    ii = np.repeat(np.arange(4), 4)  # [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
    jj = np.tile(np.arange(4), 4)    # [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]

    rows = elements[:, ii].ravel()     # (n_elems * 16,)
    cols = elements[:, jj].ravel()     # (n_elems * 16,)
    vals = Ke[:, ii, jj].ravel()       # (n_elems * 16,)

    K = sparse.coo_matrix((vals, (rows, cols)), shape=(n_verts, n_verts))
    return K.tocsr()


def _find_boundary_faces(elements: np.ndarray) -> np.ndarray:
    """Find triangular faces on the mesh boundary.

    A face is on the boundary if it belongs to exactly one tetrahedron.

    Args:
        elements: (n_elements, 4) tetrahedral connectivity

    Returns:
        (n_boundary_faces, 3) boundary face vertex indices
    """
    # Each tet has 4 faces (opposite vertex)
    face_indices = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]

    face_count = {}
    face_list = []
    for e in range(len(elements)):
        for fi in face_indices:
            face = tuple(sorted(elements[e, list(fi)]))
            if face in face_count:
                face_count[face] += 1
            else:
                face_count[face] = 1
                face_list.append(face)

    boundary = [f for f in face_list if face_count[f] == 1]
    return np.array(boundary, dtype=np.int64)


def _triangle_area(v0, v1, v2):
    """Area of a triangle given three vertex coordinates."""
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


def assemble_cem_system(
    vertices: np.ndarray,
    elements: np.ndarray,
    sigma: np.ndarray,
    electrode_nodes: np.ndarray,
    contact_impedance: float = 0.01,
    electrode_area: Optional[np.ndarray] = None,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Assemble the Complete Electrode Model augmented system.

    Builds the block system:
        [A_σ + A_z,  A_w ] [φ]   [ 0 ]
        [ A_w^T,    A_d  ] [U] = [ I ]

    For point electrodes (single node per electrode), this simplifies to:
        A_z[k,k] += 1/z_l   for node k on electrode l
        A_w[k,l]  = 1/z_l   for node k on electrode l
        A_d[l,l]  = -1/z_l  (approximating electrode area as mesh element)

    Args:
        vertices: (n_vertices, 3)
        elements: (n_elements, 4)
        sigma: (n_elements,) per-element conductivity
        electrode_nodes: (n_electrodes,) node index per electrode
        contact_impedance: z_l in Ω·m² (default: 0.01)
        electrode_area: (n_electrodes,) area per electrode in mm²
            (estimated from surrounding elements if None)

    Returns:
        (system_matrix, ground_constraint):
            system_matrix: (n_nodes + n_elec, n_nodes + n_elec) sparse
            ground_constraint: index to pin reference
    """
    n_verts = len(vertices)
    n_elec = len(electrode_nodes)
    n_total = n_verts + n_elec

    # Volume stiffness
    K = assemble_stiffness_sparse(vertices, elements, sigma)

    # Estimate electrode area from surrounding elements if not provided
    if electrode_area is None:
        electrode_area = np.ones(n_elec) * 1.0  # default 1 mm²
        # Estimate from element volumes near each electrode
        for l in range(n_elec):
            node = electrode_nodes[l]
            # Find elements containing this node
            elem_mask = np.any(elements == node, axis=1)
            if np.any(elem_mask):
                # Average element face area ≈ (6V)^(2/3) / 4
                e_verts = vertices[elements[elem_mask]]
                d = e_verts[:, 1:] - e_verts[:, 0:1]
                vols = np.abs(np.linalg.det(d)) / 6.0
                avg_vol = np.mean(vols)
                # Surface element ≈ vol^(2/3)
                electrode_area[l] = max(avg_vol ** (2.0 / 3.0), 0.01)

    # Build CEM additions as COO entries
    z = contact_impedance
    cem_rows = []
    cem_cols = []
    cem_vals = []

    for l in range(n_elec):
        k = electrode_nodes[l]
        a_l = electrode_area[l]
        inv_z = a_l / z  # effective 1/(z/a_l)

        # A_z: add 1/z_l to diagonal of node k
        cem_rows.append(k)
        cem_cols.append(k)
        cem_vals.append(inv_z)

        # A_w: coupling node k ↔ electrode l (at index n_verts + l)
        cem_rows.append(k)
        cem_cols.append(n_verts + l)
        cem_vals.append(-inv_z)

        cem_rows.append(n_verts + l)
        cem_cols.append(k)
        cem_vals.append(-inv_z)

        # A_d: electrode self-coupling
        cem_rows.append(n_verts + l)
        cem_cols.append(n_verts + l)
        cem_vals.append(inv_z)

    # Build augmented sparse matrix
    # Start with K padded to (n_total, n_total)
    K_padded = sparse.block_diag([K, sparse.csr_matrix((n_elec, n_elec))])

    # Add CEM terms
    cem_coo = sparse.coo_matrix(
        (cem_vals, (cem_rows, cem_cols)), shape=(n_total, n_total)
    )
    system = (K_padded + cem_coo).tocsr()

    # Ground constraint: pin sum of electrode voltages = 0
    # Implemented by adding a small regularisation row
    ground_idx = n_verts  # first electrode DOF

    return system, ground_idx


def apply_ground_constraint(
    system: sparse.csr_matrix,
    n_verts: int,
    n_elec: int,
) -> sparse.csr_matrix:
    """Apply ground reference: Σ U_l = 0.

    Adds a Lagrange multiplier row/column or modifies the system
    to enforce the ground constraint.

    Here we use the simple approach of fixing the mean electrode
    voltage by adding a penalty term.
    """
    n_total = n_verts + n_elec
    # Add small coupling between all electrode DOFs to enforce mean = 0
    penalty = 1e-6
    rows = []
    cols = []
    vals = []
    for l in range(n_elec):
        for m in range(n_elec):
            rows.append(n_verts + l)
            cols.append(n_verts + m)
            vals.append(penalty / n_elec)

    ground = sparse.coo_matrix(
        (vals, (rows, cols)), shape=(n_total, n_total)
    )
    return (system + ground).tocsr()


# ── Forward solver ────────────────────────────────────────────────────

def solve_eit_forward(
    vertices: np.ndarray,
    elements: np.ndarray,
    sigma: np.ndarray,
    electrode_nodes: np.ndarray,
    protocol: EITProtocol,
    contact_impedance: float = 0.01,
) -> EITForwardResult:
    """Solve the EIT forward problem using the Complete Electrode Model.

    Args:
        vertices: (n_vertices, 3)
        elements: (n_elements, 4)
        sigma: (n_elements,) conductivity in S/m
        electrode_nodes: (n_electrodes,) node index per electrode
        protocol: injection/measurement patterns
        contact_impedance: electrode contact impedance in Ω·m²

    Returns:
        EITForwardResult with potentials, voltages, and measurements
    """
    n_verts = len(vertices)
    n_elec = len(electrode_nodes)
    n_total = n_verts + n_elec

    logger.info(
        f"EIT forward: {n_verts} nodes, {len(elements)} elements, "
        f"{n_elec} electrodes, {len(protocol.injection)} patterns"
    )

    # Assemble CEM system
    system, _ = assemble_cem_system(
        vertices, elements, sigma, electrode_nodes, contact_impedance
    )
    system = apply_ground_constraint(system, n_verts, n_elec)

    # Solve for each injection pattern
    n_patterns = len(protocol.injection)
    node_potentials = np.zeros((n_patterns, n_verts))
    electrode_voltages = np.zeros((n_patterns, n_elec))

    for p in range(n_patterns):
        # Build RHS: current injection at electrode DOFs
        rhs = np.zeros(n_total)
        rhs[n_verts:] = protocol.injection[p]

        # Solve
        x = spsolve(system, rhs)

        node_potentials[p] = x[:n_verts]
        electrode_voltages[p] = x[n_verts:]

    # Compute measurements: V = measurement_matrix @ electrode_voltages
    # Each row of measurement selects electrode voltage differences
    # Need to match measurements to their injection patterns
    measurements = _compute_measurements(
        electrode_voltages, protocol
    )

    return EITForwardResult(
        node_potentials=node_potentials,
        electrode_voltages=electrode_voltages,
        measurements=measurements,
        protocol=protocol,
    )


def _compute_measurements(
    electrode_voltages: np.ndarray,
    protocol: EITProtocol,
) -> np.ndarray:
    """Extract voltage measurements from electrode voltages.

    For adjacent protocol: each measurement row corresponds to a specific
    injection pattern. We compute the measurement as the dot product of
    the measurement row with the electrode voltages for the corresponding
    injection pattern.
    """
    n_patterns = len(protocol.injection)
    n_elec = electrode_voltages.shape[1]
    n_meas_per_pattern = len(protocol.measurement) // n_patterns

    if n_meas_per_pattern * n_patterns != len(protocol.measurement):
        # Non-uniform number of measurements per pattern
        # Compute all possible measurement-pattern combinations
        measurements = protocol.measurement @ electrode_voltages.T
        return measurements.ravel()

    measurements = np.zeros(len(protocol.measurement))
    for i, meas_row in enumerate(protocol.measurement):
        p = i // n_meas_per_pattern
        p = min(p, n_patterns - 1)
        measurements[i] = meas_row @ electrode_voltages[p]

    return measurements


# ── Jacobian computation ──────────────────────────────────────────────

def compute_jacobian(
    vertices: np.ndarray,
    elements: np.ndarray,
    sigma: np.ndarray,
    electrode_nodes: np.ndarray,
    protocol: EITProtocol,
    forward_result: Optional[EITForwardResult] = None,
    contact_impedance: float = 0.01,
) -> np.ndarray:
    """Compute the Jacobian ∂V/∂σ via the adjoint method.

    For the CEM, the Jacobian entry J[m, k] is:
        J[m, k] = -φ_inj^T (∂K/∂σ_k) φ_meas

    where φ_inj is the forward solution for the injection pattern and
    φ_meas is the solution for the measurement pattern (or vice versa).

    For P1 elements: ∂K/∂σ_k = K_k^{unit} (the unit-conductivity
    element stiffness matrix for element k).

    Args:
        vertices: (n_vertices, 3)
        elements: (n_elements, 4)
        sigma: (n_elements,) conductivity
        electrode_nodes: (n_electrodes,)
        protocol: EIT protocol
        forward_result: pre-computed forward result (avoids recomputation)
        contact_impedance: z in Ω·m²

    Returns:
        (n_measurements, n_elements) Jacobian matrix
    """
    n_verts = len(vertices)
    n_elec = len(electrode_nodes)
    n_elems = len(elements)
    n_total = n_verts + n_elec

    # Forward solve if not provided
    if forward_result is None:
        forward_result = solve_eit_forward(
            vertices, elements, sigma, electrode_nodes,
            protocol, contact_impedance
        )

    # We also need solutions for measurement patterns (adjoint fields)
    # Assemble system for measurement solve
    system, _ = assemble_cem_system(
        vertices, elements, sigma, electrode_nodes, contact_impedance
    )
    system = apply_ground_constraint(system, n_verts, n_elec)

    # Get unique measurement patterns for adjoint solves
    meas_patterns = protocol.measurement
    n_meas = len(meas_patterns)

    # Group unique measurement patterns
    unique_meas, inverse_idx = np.unique(
        meas_patterns, axis=0, return_inverse=True
    )

    # Solve adjoint for each unique measurement pattern
    adjoint_potentials = np.zeros((len(unique_meas), n_verts))
    for m in range(len(unique_meas)):
        rhs = np.zeros(n_total)
        rhs[n_verts:] = unique_meas[m]
        x = spsolve(system, rhs)
        adjoint_potentials[m] = x[:n_verts]

    # Batch-compute unit element stiffness: (n_elems, 4, 4)
    Ke_unit = _tet_stiffness_batch(vertices[elements])

    logger.info(f"Computing Jacobian: {n_meas} × {n_elems}")
    n_patterns = len(protocol.injection)
    n_meas_per_pattern = max(n_meas // max(n_patterns, 1), 1)

    # Map each measurement to its injection pattern index
    pattern_idx = np.minimum(
        np.arange(n_meas) // n_meas_per_pattern, n_patterns - 1
    )

    # Vectorised Jacobian: process per injection pattern to limit memory
    # Memory per pattern: O(n_meas_per_pattern × n_elems × 4) floats
    J = np.zeros((n_meas, n_elems))

    for p in range(n_patterns):
        mask = pattern_idx == p
        m_indices = np.where(mask)[0]
        if len(m_indices) == 0:
            continue

        # Injection potential at element nodes: (n_elems, 4)
        phi_inj_p = forward_result.node_potentials[p][elements]

        # Adjoint potentials for this pattern's measurements: (n_m, n_elems, 4)
        adj_idx = inverse_idx[m_indices]
        phi_meas_p = adjoint_potentials[adj_idx][:, elements]

        # J[m, k] = -φ_inj[k,i] Ke[k,i,j] φ_meas[m,k,j]
        # First: Ke @ phi_inj^T → (n_elems, 4) via einsum
        Ke_phi = np.einsum('kij,kj->ki', Ke_unit, phi_inj_p)  # (n_elems, 4)

        # Then: J[m, k] = -φ_meas[m,k,:] · Ke_phi[k,:]
        J[m_indices] = -np.einsum('mki,ki->mk', phi_meas_p, Ke_phi)

    return J


# ── Inverse solvers ──────────────────────────────────────────────────

def solve_eit_inverse(
    jacobian: np.ndarray,
    delta_v: np.ndarray,
    method: str = "tikhonov",
    alpha: float = 0.01,
    max_iter: int = 10,
    sigma_ref: Optional[np.ndarray] = None,
    laplacian: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reconstruct conductivity change from voltage difference.

    Solves the linearised inverse problem:
        δV ≈ J · δσ

    with regularisation to handle ill-posedness.

    Args:
        jacobian: (n_meas, n_elements) Jacobian matrix
        delta_v: (n_meas,) voltage difference (perturbed - reference)
        method: "tikhonov", "noser", or "tv" (total variation)
        alpha: regularisation parameter
        max_iter: iterations for iterative methods
        sigma_ref: (n_elements,) reference conductivity (for NOSER)
        laplacian: (n_elements, n_elements) spatial regularisation matrix

    Returns:
        (n_elements,) reconstructed conductivity change δσ
    """
    J = jacobian
    n_meas, n_elems = J.shape

    if method == "tikhonov":
        return _solve_tikhonov(J, delta_v, alpha, laplacian)
    elif method == "noser":
        return _solve_noser(J, delta_v, alpha, sigma_ref)
    elif method == "tv":
        return _solve_tv(J, delta_v, alpha, max_iter)
    else:
        raise ValueError(f"Unknown method: {method}. Use tikhonov/noser/tv.")


def _solve_tikhonov(
    J: np.ndarray,
    delta_v: np.ndarray,
    alpha: float,
    laplacian: Optional[np.ndarray] = None,
) -> np.ndarray:
    """One-step Gauss-Newton with Tikhonov regularisation.

        δσ = (J^T J + α R)^{-1} J^T δV

    where R = I (standard) or R = L^T L (smoothness prior).
    """
    n_elems = J.shape[1]

    if laplacian is not None:
        R = laplacian.T @ laplacian
    else:
        R = np.eye(n_elems)

    JtJ = J.T @ J
    Jtv = J.T @ delta_v
    A = JtJ + alpha * R
    delta_sigma = np.linalg.solve(A, Jtv)

    return delta_sigma


def _solve_noser(
    J: np.ndarray,
    delta_v: np.ndarray,
    alpha: float,
    sigma_ref: Optional[np.ndarray] = None,
) -> np.ndarray:
    """NOSER (Newton's One-Step Error Reconstructor).

    Uses diag(J^T J) as the regularisation matrix, weighted by the
    reference conductivity. This naturally adapts the regularisation
    to the sensitivity of each element.

        δσ = (J^T J + α diag(J^T J) / σ_ref)^{-1} J^T δV
    """
    n_elems = J.shape[1]
    JtJ = J.T @ J
    diag_JtJ = np.diag(JtJ)

    if sigma_ref is not None:
        # Weight by reference conductivity
        R = np.diag(diag_JtJ / np.maximum(sigma_ref, 1e-6))
    else:
        R = np.diag(diag_JtJ)

    # Add small diagonal to prevent singularity from zero-sensitivity elements
    R += 1e-10 * np.eye(n_elems)

    Jtv = J.T @ delta_v
    A = JtJ + alpha * R
    delta_sigma = np.linalg.solve(A, Jtv)

    return delta_sigma


def _solve_tv(
    J: np.ndarray,
    delta_v: np.ndarray,
    alpha: float,
    max_iter: int = 10,
    beta: float = 1e-6,
) -> np.ndarray:
    """Total Variation regularised reconstruction.

    Iteratively Reweighted Least Squares (IRLS) approximation:
        δσ^{k+1} = (J^T J + α W^k)^{-1} J^T δV

    where W^k = diag(1/|∇σ^k| + β) approximates the TV penalty.
    """
    n_elems = J.shape[1]
    JtJ = J.T @ J
    Jtv = J.T @ delta_v

    # Initial estimate: standard Tikhonov
    delta_sigma = np.linalg.solve(JtJ + alpha * np.eye(n_elems), Jtv)

    for iteration in range(max_iter):
        # Reweight: W = diag(1 / (|δσ| + β))
        weights = 1.0 / (np.abs(delta_sigma) + beta)
        W = np.diag(weights)
        A = JtJ + alpha * W
        delta_sigma_new = np.linalg.solve(A, Jtv)

        # Check convergence
        change = np.linalg.norm(delta_sigma_new - delta_sigma)
        change /= max(np.linalg.norm(delta_sigma), 1e-10)
        delta_sigma = delta_sigma_new

        if change < 1e-4:
            logger.info(f"TV converged at iteration {iteration + 1}")
            break

    return delta_sigma


# ── Difference imaging ────────────────────────────────────────────────

def difference_eit(
    v_reference: np.ndarray,
    v_perturbed: np.ndarray,
    jacobian: np.ndarray,
    method: str = "tikhonov",
    alpha: float = 0.01,
    normalise: bool = True,
    **kwargs,
) -> np.ndarray:
    """Time-difference EIT reconstruction.

    Reconstructs the conductivity change between a reference state
    and a perturbed state from their voltage measurements.

    Args:
        v_reference: (n_meas,) reference voltage measurements
        v_perturbed: (n_meas,) perturbed voltage measurements
        jacobian: (n_meas, n_elements) Jacobian at reference state
        method: reconstruction method
        alpha: regularisation parameter
        normalise: if True, normalise voltage difference by reference

    Returns:
        (n_elements,) reconstructed conductivity change
    """
    if normalise:
        # Normalised difference: (v_pert - v_ref) / v_ref
        denom = np.maximum(np.abs(v_reference), 1e-10)
        delta_v = (v_perturbed - v_reference) / denom
    else:
        delta_v = v_perturbed - v_reference

    return solve_eit_inverse(jacobian, delta_v, method=method,
                             alpha=alpha, **kwargs)


# ── JAX-differentiable forward (for small meshes / gradients) ─────────

def _eit_forward_jax(
    vertices: jnp.ndarray,
    elements: jnp.ndarray,
    sigma: jnp.ndarray,
    n_electrodes: int,
    electrode_nodes: jnp.ndarray,
    injection_pattern: jnp.ndarray,
    contact_impedance: float = 0.01,
) -> jnp.ndarray:
    """JAX-differentiable EIT forward for small meshes.

    Uses dense assembly (suited for meshes up to ~10K nodes).
    Fully differentiable via jax.grad for sensitivity analysis.

    Args:
        vertices: (n_verts, 3)
        elements: (n_elems, 4)
        sigma: (n_elems,) conductivity
        n_electrodes: number of electrodes (static)
        electrode_nodes: (n_elec,) node indices
        injection_pattern: (n_elec,) current pattern
        contact_impedance: z in Ω·m²

    Returns:
        (n_elec,) electrode voltages
    """
    from neurojax.geometry.fem_forward import assemble_stiffness

    n_verts = vertices.shape[0]
    n_total = n_verts + n_electrodes

    # Volume stiffness
    K = assemble_stiffness(vertices, elements, sigma)

    # Pad to augmented system size
    system = jnp.zeros((n_total, n_total))
    system = system.at[:n_verts, :n_verts].set(K)

    # CEM terms — vectorised via scatter
    inv_z = 1.0 / contact_impedance
    elec_idx = jnp.arange(n_electrodes)

    # A_z: diagonal at electrode nodes
    system = system.at[electrode_nodes, electrode_nodes].add(inv_z)
    # A_w: off-diagonal coupling
    system = system.at[electrode_nodes, n_verts + elec_idx].add(-inv_z)
    system = system.at[n_verts + elec_idx, electrode_nodes].add(-inv_z)
    # A_d: electrode self-coupling
    system = system.at[n_verts + elec_idx, n_verts + elec_idx].add(inv_z)

    # Ground constraint: penalise non-zero mean electrode voltage
    penalty = 1e-6
    ground_block = jnp.full((n_electrodes, n_electrodes), penalty / n_electrodes)
    system = system.at[n_verts:, n_verts:].add(ground_block)

    # Regularise for numerical stability
    system = system + 1e-8 * jnp.eye(n_total)

    # RHS
    rhs = jnp.zeros(n_total)
    rhs = rhs.at[n_verts:].set(injection_pattern)

    # Solve
    x = jnp.linalg.solve(system, rhs)

    return x[n_verts:]  # electrode voltages


def eit_voltage_map(
    vertices: jnp.ndarray,
    elements: jnp.ndarray,
    sigma: jnp.ndarray,
    electrode_nodes: jnp.ndarray,
    protocol: EITProtocol,
    contact_impedance: float = 0.01,
) -> jnp.ndarray:
    """Compute all EIT voltage measurements (JAX, differentiable).

    Convenience wrapper that loops over injection patterns and
    extracts measurements. Use for gradient-based optimisation
    on small meshes.

    Args:
        vertices, elements, sigma: mesh + conductivity
        electrode_nodes: (n_elec,) node indices
        protocol: injection/measurement patterns
        contact_impedance: contact impedance

    Returns:
        (n_measurements,) voltage measurements
    """
    n_elec = len(electrode_nodes)
    n_patterns = len(protocol.injection)
    n_meas_per_pattern = len(protocol.measurement) // n_patterns

    all_meas = []
    for p in range(n_patterns):
        v_elec = _eit_forward_jax(
            vertices, elements, sigma, n_elec,
            electrode_nodes, jnp.array(protocol.injection[p]),
            contact_impedance,
        )
        # Extract measurements for this pattern
        start = p * n_meas_per_pattern
        end = start + n_meas_per_pattern
        for m_idx in range(start, min(end, len(protocol.measurement))):
            meas_row = jnp.array(protocol.measurement[m_idx])
            all_meas.append(jnp.dot(meas_row, v_elec))

    return jnp.array(all_meas)


# ── Element-neighbour Laplacian for spatial regularisation ────────────

def element_laplacian(elements: np.ndarray) -> np.ndarray:
    """Compute element-adjacency Laplacian for spatial regularisation.

    Two elements are neighbours if they share a face (3 vertices).
    The Laplacian L encodes: L[i,i] = degree, L[i,j] = -1 if neighbours.

    Args:
        elements: (n_elements, 4) tetrahedral connectivity

    Returns:
        (n_elements, n_elements) Laplacian matrix
    """
    n_elems = len(elements)

    # Build face → element mapping
    face_to_elem = {}
    face_indices = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

    for e in range(n_elems):
        for fi in face_indices:
            face = tuple(sorted(elements[e, list(fi)]))
            if face in face_to_elem:
                face_to_elem[face].append(e)
            else:
                face_to_elem[face] = [e]

    # Build adjacency
    L = np.zeros((n_elems, n_elems))
    for face, elems in face_to_elem.items():
        if len(elems) == 2:
            e1, e2 = elems
            L[e1, e2] = -1.0
            L[e2, e1] = -1.0
            L[e1, e1] += 1.0
            L[e2, e2] += 1.0

    return L


# ── Utility: simulate perturbation ───────────────────────────────────

def simulate_perturbation(
    vertices: np.ndarray,
    elements: np.ndarray,
    sigma_background: np.ndarray,
    perturbation_centre: np.ndarray,
    perturbation_radius: float,
    perturbation_value: float,
) -> np.ndarray:
    """Create a conductivity distribution with a localised perturbation.

    Useful for simulating stroke, haemorrhage, or other focal changes.

    Args:
        vertices: (n_vertices, 3)
        elements: (n_elements, 4)
        sigma_background: (n_elements,) background conductivity
        perturbation_centre: (3,) centre of perturbation in mm
        perturbation_radius: radius in mm
        perturbation_value: conductivity of the perturbation in S/m

    Returns:
        (n_elements,) perturbed conductivity
    """
    centroids = np.mean(vertices[elements], axis=1)  # (n_elems, 3)
    dists = np.sqrt(np.sum(
        (centroids - perturbation_centre) ** 2, axis=1
    ))
    sigma_perturbed = sigma_background.copy()
    sigma_perturbed[dists <= perturbation_radius] = perturbation_value
    return sigma_perturbed
