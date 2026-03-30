"""Multiway tensor analysis for Magnetic Resonance Spectroscopy (MRS) data.

Applies tensor decomposition methods from chemometrics to MRS coil
combination, artifact rejection, and spectral unmixing.  MRS data is
naturally a 3-way tensor:

    (spectral_points x coils x averages)           — standard acquisition
    (spectral_points x coils x editing_conditions x averages)  — MEGA-PRESS

The algorithms here exploit that multilinear structure directly rather
than collapsing to matrix problems.

References:
    De Lathauwer, De Moor & Vandewalle (2000) "A multilinear singular
        value decomposition". SIAM J. Matrix Anal. Appl. 21(4):1253-1278.
    Harshman (1970) "Foundations of the PARAFAC procedure". UCLA Working
        Papers in Phonetics 16:1-84.
    Carroll & Chang (1970) "Analysis of individual differences in
        multidimensional scaling via an N-way generalization of
        Eckart-Young decomposition". Psychometrika 35:283-319.
    Tauler (1995) "Multivariate curve resolution applied to second order
        data". Chemometrics and Intelligent Laboratory Systems 30:133-146.
    Bro (1997) "PARAFAC: Tutorial and applications". Chemometrics and
        Intelligent Laboratory Systems 38:149-171.

All functions are pure (no I/O, no side-effects).  Uses NumPy for the
core linear algebra; results can be passed to JAX downstream.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd, norm


# =====================================================================
# Utility: tensor unfolding / refolding
# =====================================================================

def _unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Mode-*n* unfolding (matricisation) of a tensor.

    Rearranges the *mode*-th index to rows and Kronecker-products the
    remaining indices into columns, following the convention of
    De Lathauwer et al. (2000).

    Parameters
    ----------
    tensor : ndarray, shape (I_0, I_1, ..., I_{N-1})
    mode : int — which mode to unfold along.

    Returns
    -------
    matrix : ndarray, shape (I_mode, prod(I_n for n != mode))
    """
    return np.reshape(np.moveaxis(tensor, mode, 0),
                      (tensor.shape[mode], -1))


def _mode_dot(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    """Mode-*n* product of a tensor with a matrix.

    Contracts mode *mode* of *tensor* with the columns of *matrix*.
    Equivalent to multiplying the mode-*n* unfolding on the left by
    *matrix* and refolding.

    Parameters
    ----------
    tensor : ndarray, shape (..., I_mode, ...)
    matrix : ndarray, shape (J, I_mode)
    mode : int

    Returns
    -------
    result : ndarray — same order as *tensor* but with dimension *mode*
        replaced by J.
    """
    return np.tensordot(matrix, tensor, axes=([1], [mode]))  \
             if mode == 0 else \
           np.moveaxis(
               np.tensordot(matrix, tensor, axes=([1], [mode])),
               0, mode)


# =====================================================================
# 1. Tucker / HOSVD
# =====================================================================

def mrs_tucker_decomposition(data: np.ndarray, ranks: tuple) -> dict:
    """Tucker decomposition of a 3-D tensor via truncated HOSVD.

    The Higher-Order SVD (HOSVD) provides an optimal subspace per mode
    and a dense core tensor that captures inter-mode interactions.  For
    MRS data shaped (spectral_points, coils, averages) this extracts
    the dominant spectral, spatial (coil), and temporal (average)
    subspaces simultaneously.

    Algorithm (De Lathauwer et al. 2000, Sec. 4):
        1. Unfold the tensor along each mode *n*.
        2. Compute the truncated SVD of each unfolding to obtain the
           *R_n* dominant left singular vectors -> factor matrix U_n.
        3. Project the tensor onto all factor matrices to obtain the
           core tensor G = data x_1 U_1^T x_2 U_2^T x_3 U_3^T.

    Parameters
    ----------
    data : ndarray, shape (I, J, K)
        3-way MRS tensor.
    ranks : tuple of int, (R1, R2, R3)
        Truncation ranks for each mode.  Must satisfy R_n <= I_n.

    Returns
    -------
    dict with keys:
        'core'      : ndarray (R1, R2, R3) — core tensor.
        'factors'   : list of 3 ndarrays [(I, R1), (J, R2), (K, R3)]
                      — orthonormal factor (basis) matrices.
        'explained_variance' : float — fraction of the Frobenius norm
                      retained by the truncated decomposition.

    References
    ----------
    De Lathauwer, De Moor & Vandewalle (2000) SIAM J. Matrix Anal. 21(4).
    """
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D tensor, got {data.ndim}-D.")
    if len(ranks) != 3:
        raise ValueError("ranks must be a 3-tuple.")

    total_energy = norm(data) ** 2
    factors = []

    for mode, rank in enumerate(ranks):
        if rank > data.shape[mode]:
            raise ValueError(
                f"Rank {rank} exceeds mode-{mode} dimension {data.shape[mode]}.")
        unfolded = _unfold(data, mode)
        U, _, _ = svd(unfolded, full_matrices=False)
        factors.append(U[:, :rank])

    # Core tensor: project data onto all factor subspaces
    core = data.copy()
    for mode in range(3):
        core = _mode_dot(core, factors[mode].T, mode)

    explained = norm(core) ** 2 / total_energy if total_energy > 0 else 0.0

    return {
        "core": core,
        "factors": factors,
        "explained_variance": float(explained),
    }


# =====================================================================
# 2. PARAFAC / CP via Alternating Least Squares
# =====================================================================

def mrs_parafac(
    data: np.ndarray,
    n_components: int,
    n_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 0,
) -> dict:
    """PARAFAC (CP) decomposition using Alternating Least Squares.

    Decomposes a 3-way tensor into a sum of *R* rank-1 tensors:

        X ≈ sum_r  w_r * a_r (x) b_r (x) c_r

    where (x) denotes the outer product and w_r are scalar weights
    (norms absorbed from the factor columns).

    For MRS data this identifies independent source signals (spectral
    factor), their coil sensitivities (spatial factor), and their
    temporal evolution across averages (temporal factor).

    Algorithm — ALS (Harshman 1970, Bro 1997):
        For each mode *n* in turn, fix the other two factor matrices
        and solve the resulting linear least-squares problem.  Iterate
        until the relative change in reconstruction error is below *tol*
        or *n_iter* iterations are exhausted.

    Parameters
    ----------
    data : ndarray, shape (I, J, K)
        3-way MRS tensor.
    n_components : int
        Number of rank-1 components (R).
    n_iter : int
        Maximum number of ALS iterations.
    tol : float
        Convergence tolerance on relative reconstruction error change.
    seed : int
        Random seed for factor initialisation.

    Returns
    -------
    dict with keys:
        'weights'   : ndarray (R,) — component weights (descending).
        'factors'   : list of 3 ndarrays [(I, R), (J, R), (K, R)]
                      — normalised factor matrices.
        'n_iter'    : int — number of iterations run.
        'rec_error' : float — final relative reconstruction error.

    References
    ----------
    Harshman (1970) UCLA Working Papers in Phonetics 16.
    Bro (1997) Chemom. Intell. Lab. Syst. 38:149-171.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D tensor, got {data.ndim}-D.")

    R = n_components
    rng = np.random.RandomState(seed)
    I, J, K = data.shape

    # Initialise factor matrices with random normal entries
    A = rng.randn(I, R)
    B = rng.randn(J, R)
    C = rng.randn(K, R)

    data_norm = norm(data)
    prev_error = np.inf

    # Pre-compute mode unfoldings (these do not change)
    X0 = _unfold(data, 0)  # (I, J*K)
    X1 = _unfold(data, 1)  # (J, I*K)
    X2 = _unfold(data, 2)  # (K, I*J)

    for iteration in range(n_iter):
        # --- Update A ---
        # X_(0) ≈ A @ (C khat B)^T   where khat = Khatri-Rao product
        # _unfold(data,0) has shape (I, J*K) with col index = j*K + k,
        # so the Khatri-Rao needs rows ordered as (j, k) -> kron(B, C).
        CB = np.column_stack([np.kron(B[:, r], C[:, r]) for r in range(R)])
        A = X0 @ CB @ np.linalg.pinv(CB.T @ CB)

        # --- Update B ---
        # _unfold(data,1) has shape (J, I*K) with col index = i*K + k
        CA = np.column_stack([np.kron(A[:, r], C[:, r]) for r in range(R)])
        B = X1 @ CA @ np.linalg.pinv(CA.T @ CA)

        # --- Update C ---
        # _unfold(data,2) has shape (K, I*J) with col index = i*J + j
        BA = np.column_stack([np.kron(A[:, r], B[:, r]) for r in range(R)])
        C = X2 @ BA @ np.linalg.pinv(BA.T @ BA)

        # Convergence check: relative reconstruction error
        # Reconstruct from factors: sum_r A[:,r] (x) B[:,r] (x) C[:,r]
        rec = np.zeros_like(data)
        for r in range(R):
            rec += np.einsum('i,j,k->ijk', A[:, r], B[:, r], C[:, r])
        rec_error = norm(data - rec) / data_norm if data_norm > 0 else 0.0

        if abs(prev_error - rec_error) < tol:
            break
        prev_error = rec_error

    # Normalise columns and absorb norms into weights
    weights = np.empty(R)
    for r in range(R):
        nA = norm(A[:, r])
        nB = norm(B[:, r])
        nC = norm(C[:, r])
        w = nA * nB * nC
        weights[r] = w
        A[:, r] /= nA if nA > 0 else 1.0
        B[:, r] /= nB if nB > 0 else 1.0
        C[:, r] /= nC if nC > 0 else 1.0

    # Sort components by descending weight magnitude
    order = np.argsort(-np.abs(weights))
    weights = weights[order]
    A = A[:, order]
    B = B[:, order]
    C = C[:, order]

    return {
        "weights": weights,
        "factors": [A, B, C],
        "n_iter": iteration + 1,
        "rec_error": float(rec_error),
    }


# =====================================================================
# 3. Multivariate Curve Resolution — ALS (MCR-ALS)
# =====================================================================

def mrs_mcr_als(
    spectra: np.ndarray,
    n_components: int,
    n_iter: int = 200,
    tol: float = 1e-6,
    non_negative_spectra: bool = True,
    seed: int = 0,
) -> dict:
    """Multivariate Curve Resolution — Alternating Least Squares.

    Decomposes a (n_spectra, n_points) data matrix D into:

        D ≈ C @ S

    where C (n_spectra, n_components) are concentrations and
    S (n_components, n_points) are pure-component spectra.

    In MRS this separates overlapping metabolite resonances from a set
    of averaged spectra or resolves evolving lineshape artefacts.

    Constraints applied at each iteration:
        - Non-negativity on C (concentrations are physical quantities).
        - Optional non-negativity on S (appropriate for magnitude
          spectra; disable for complex/phase-sensitive data).

    Algorithm (Tauler 1995):
        1. Initialise S via truncated SVD of D.
        2. Alternate:
           a. C = D @ S^+ , clip negative entries.
           b. S = C^+ @ D , optionally clip negatives.
        3. Repeat until convergence.

    Parameters
    ----------
    spectra : ndarray, shape (n_spectra, n_points)
        Data matrix — rows are individual spectra (e.g., different
        averages, voxels, or editing conditions).
    n_components : int
        Number of pure components to resolve.
    n_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on relative residual change.
    non_negative_spectra : bool
        If True, enforce non-negativity on the resolved spectra S.
    seed : int
        Not used (deterministic SVD initialisation), kept for API
        symmetry with other decomposition functions.

    Returns
    -------
    dict with keys:
        'concentrations' : ndarray (n_spectra, n_components)
        'spectra'        : ndarray (n_components, n_points)
        'n_iter'         : int — iterations run.
        'residual'       : float — final relative Frobenius residual.

    References
    ----------
    Tauler (1995) Chemom. Intell. Lab. Syst. 30:133-146.
    De Juan & Tauler (2006) Crit. Rev. Anal. Chem. 36:163-176.
    """
    if spectra.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got {spectra.ndim}-D.")

    D = spectra.astype(np.float64)
    n_spectra, n_points = D.shape
    data_norm = norm(D)

    # Initialise S from the first n_components right singular vectors
    U, s, Vt = svd(D, full_matrices=False)
    S = np.diag(s[:n_components]) @ Vt[:n_components, :]

    prev_residual = np.inf

    for iteration in range(n_iter):
        # --- Solve for C given S ---
        # C = D @ S^T @ (S @ S^T)^{-1}   (least-squares)
        C = D @ S.T @ np.linalg.pinv(S @ S.T)
        C = np.maximum(C, 0.0)  # non-negativity constraint

        # --- Solve for S given C ---
        S = np.linalg.pinv(C.T @ C) @ C.T @ D
        if non_negative_spectra:
            S = np.maximum(S, 0.0)

        # Convergence
        residual = norm(D - C @ S) / data_norm if data_norm > 0 else 0.0
        if abs(prev_residual - residual) < tol:
            break
        prev_residual = residual

    return {
        "concentrations": C,
        "spectra": S,
        "n_iter": iteration + 1,
        "residual": float(residual),
    }


# =====================================================================
# 4. Optimal coil weights via rank-1 Tucker on the coil mode
# =====================================================================

def optimal_coil_weights_from_tucker(data: np.ndarray) -> np.ndarray:
    """Derive optimal coil-combination weights from a Tucker decomposition.

    Performs a rank-(full, 1, full) Tucker decomposition so that only the
    coil mode is compressed to rank 1.  The resulting single-column coil
    factor vector gives the relative sensitivity of each coil, which can
    be used as combination weights (analogous to first singular vector
    of the coil dimension).

    This is equivalent to computing the dominant left singular vector of
    the mode-1 (coil) unfolding.  Complex data is handled natively.

    Parameters
    ----------
    data : ndarray, shape (n_spectral, n_coils, n_averages)
        3-way MRS tensor (may be complex-valued).

    Returns
    -------
    weights : ndarray, shape (n_coils,)
        Complex coil-combination weights, normalised to unit norm.
        To combine: combined = np.tensordot(weights.conj(), data, axes=([0],[1]))
    """
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D tensor, got {data.ndim}-D.")

    # Mode-1 unfolding: (n_coils, n_spectral * n_averages)
    unfolded = _unfold(data, 1)

    # Dominant left singular vector of the coil unfolding
    U, _, _ = svd(unfolded, full_matrices=False)
    weights = U[:, 0]

    # Normalise to unit norm
    w_norm = norm(weights)
    if w_norm > 0:
        weights = weights / w_norm

    return weights


# =====================================================================
# 5. Artifact rejection via PARAFAC
# =====================================================================

def artifact_rejection_parafac(
    data: np.ndarray,
    n_signal: int = 3,
    n_total: int = 5,
    n_iter: int = 200,
    tol: float = 1e-7,
    seed: int = 42,
) -> np.ndarray:
    """Remove artifacts from MRS data using PARAFAC decomposition.

    Decomposes the 3-way MRS tensor into *n_total* PARAFAC components,
    then reconstructs using only the *n_signal* components with the
    largest weights (by magnitude).  Components with small or erratic
    weights typically correspond to motion artifacts, lipid
    contamination, or hardware instabilities.

    This follows the philosophy of tensor-based artifact rejection in
    chemometrics: genuine metabolite signals exhibit consistent coil and
    average profiles, whereas artefacts do not.

    Parameters
    ----------
    data : ndarray, shape (n_spectral, n_coils, n_averages)
        3-way MRS tensor.
    n_signal : int
        Number of signal (kept) components.
    n_total : int
        Total number of PARAFAC components to fit.  Must be >= n_signal.
    n_iter : int
        Maximum ALS iterations for PARAFAC.
    tol : float
        Convergence tolerance for PARAFAC.
    seed : int
        Random seed for PARAFAC initialisation.

    Returns
    -------
    cleaned : ndarray, shape (n_spectral, n_coils, n_averages)
        Artifact-rejected tensor, reconstructed from the *n_signal*
        dominant PARAFAC components.
    """
    if n_signal > n_total:
        raise ValueError(
            f"n_signal ({n_signal}) must be <= n_total ({n_total}).")
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D tensor, got {data.ndim}-D.")

    result = mrs_parafac(
        data, n_components=n_total, n_iter=n_iter, tol=tol, seed=seed)

    weights = result["weights"]
    A, B, C = result["factors"]

    # The components are already sorted by descending |weight| in mrs_parafac
    # Reconstruct from only the first n_signal components
    cleaned = np.zeros_like(data)
    for r in range(n_signal):
        cleaned += weights[r] * (
            A[:, r:r+1]
            * B[np.newaxis, :, r:r+1]
            * C[np.newaxis, np.newaxis, :, r]
        )

    return cleaned
