"""Minimum Norm Estimate (MNE) family — unified JAX implementation.

All MNE variants share the same inverse operator structure:
    x̂ = W @ y
    W = R @ G^T @ (G @ R @ G^T + λ · C)^{-1}

The variants differ only in:
    - R: source covariance prior (depth weighting)
    - Normalization: none (MNE), noise (dSPM), resolution (sLORETA/eLORETA)

This unified implementation makes depth weighting, orientation constraint,
and regularization all differentiable — enabling gradient-based optimization
of inverse parameters against known source configurations.

Implements: MNE, wMNE, dSPM, sLORETA, eLORETA

References
----------
Hämäläinen MS, Ilmoniemi RJ (1994). Interpreting magnetic fields of the
brain: minimum norm estimates. Med Biol Eng Comput 32(1):35-42.

Dale AM et al. (2000). Dynamic statistical parametric mapping: combining
fMRI and MEG for high-resolution imaging of cortical activity. Neuron
26(1):55-67.

Pascual-Marqui RD (2002). Standardized low-resolution brain electromagnetic
tomography (sLORETA). Methods Find Exp Clin Pharmacol 24D:5-12.

Pascual-Marqui RD (2007). Discrete, 3D distributed, linear imaging methods
of electric neuronal activity. arXiv:0710.3341 (eLORETA).
"""

from __future__ import annotations
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp


class InverseOperator(NamedTuple):
    """Precomputed inverse operator for fast application."""
    W: jnp.ndarray          # (n_sources, n_sensors) inverse kernel
    noise_norm: jnp.ndarray  # (n_sources,) normalization factors (for dSPM/sLORETA)
    method: str
    depth: float
    reg: float
    loose: float


@jax.jit
def compute_depth_prior(
    gain: jnp.ndarray,
    depth: float = 0.8,
    exp: float = 0.8,
) -> jnp.ndarray:
    """Compute depth weighting prior from the gain (leadfield) matrix.

    Parameters
    ----------
    gain : (n_sensors, n_sources) leadfield matrix.
    depth : float in [0, 1] — depth weighting strength.
        0 = no weighting (classical MNE, superficial bias)
        0.8 = standard depth weighting (wMNE/dSPM/sLORETA)
        1.0 = full depth compensation
    exp : float — exponent for the depth weighting.

    Returns
    -------
    R_diag : (n_sources,) — diagonal of source covariance prior.
    """
    # Column norms of gain matrix → proxy for source depth
    # Deeper sources have smaller gain norms
    col_norms = jnp.sqrt(jnp.sum(gain ** 2, axis=0))  # (n_sources,)
    col_norms = jnp.maximum(col_norms, 1e-10)

    if depth == 0:
        return jnp.ones_like(col_norms)

    # Depth weighting: R_ii = (||g_i||^exp)^(-2*depth)
    # This boosts deep sources (small ||g||) relative to superficial
    weights = col_norms ** (-exp * depth)
    # Normalize so mean weight = 1
    weights = weights / jnp.mean(weights)
    return weights


@jax.jit
def make_inverse_operator(
    gain: jnp.ndarray,
    noise_cov: jnp.ndarray,
    depth: float = 0.8,
    reg: float = 0.1,
    method: str = "dSPM",
) -> jnp.ndarray:
    """Compute the MNE inverse operator W.

    Parameters
    ----------
    gain : (n_sensors, n_sources) leadfield/gain matrix.
    noise_cov : (n_sensors, n_sensors) noise covariance.
    depth : float — depth weighting (0=none, 0.8=standard).
    reg : float — regularization parameter (fraction of max eigenvalue).
    method : str — "MNE", "dSPM", "sLORETA", "eLORETA".

    Returns
    -------
    W : (n_sources, n_sensors) inverse kernel.
    noise_norm : (n_sources,) normalization factors.
    """
    n_sensors, n_sources = gain.shape

    # Source prior (depth weighting)
    R_diag = compute_depth_prior(gain, depth)
    R = jnp.diag(R_diag)

    # Whiten: C^{-1/2}
    # For numerical stability, regularize noise covariance
    C_reg = noise_cov + reg * jnp.trace(noise_cov) / n_sensors * jnp.eye(n_sensors)
    eigvals, eigvecs = jnp.linalg.eigh(C_reg)
    eigvals = jnp.maximum(eigvals, 1e-10)
    C_inv_half = eigvecs @ jnp.diag(1.0 / jnp.sqrt(eigvals)) @ eigvecs.T

    # Whitened gain
    G_w = C_inv_half @ gain  # (n_sensors, n_sources)

    # Weighted gain
    G_wr = G_w @ jnp.diag(jnp.sqrt(R_diag))  # (n_sensors, n_sources)

    # SVD of weighted whitened gain
    U, S, Vh = jnp.linalg.svd(G_wr, full_matrices=False)

    # Regularized pseudoinverse: S / (S^2 + λ)
    # λ set as fraction of largest eigenvalue
    lambda_reg = reg * S[0] ** 2
    S_reg = S / (S ** 2 + lambda_reg)

    # Inverse kernel (before normalization)
    # W = R^{1/2} @ V @ diag(S_reg) @ U^T @ C^{-1/2}
    W = jnp.diag(jnp.sqrt(R_diag)) @ Vh.T @ jnp.diag(S_reg) @ U.T @ C_inv_half

    # Normalization
    noise_norm = jnp.ones(n_sources)

    if method == "dSPM":
        # dSPM: normalize by noise projection through inverse
        # noise_norm_i = sqrt(W_i @ C @ W_i^T)
        W_C_Wt_diag = jnp.sum((W @ noise_cov) * W, axis=1)
        noise_norm = jnp.sqrt(jnp.maximum(W_C_Wt_diag, 1e-20))

    elif method == "sLORETA":
        # sLORETA: normalize by resolution matrix diagonal
        # R_ii = W_i @ G_i (diagonal of resolution matrix)
        res_diag = jnp.sum(W * gain.T, axis=1)  # (n_sources,)
        noise_norm = jnp.sqrt(jnp.maximum(jnp.abs(res_diag), 1e-20))

    elif method == "eLORETA":
        # eLORETA: iterative weight adjustment for exact localization
        # Simplified: use sLORETA normalization with iterative refinement
        for _ in range(3):
            res_diag = jnp.sum(W * gain.T, axis=1)
            weight_update = 1.0 / jnp.sqrt(jnp.maximum(jnp.abs(res_diag), 1e-20))
            R_diag_new = R_diag * weight_update ** 2
            R_diag_new = R_diag_new / jnp.mean(R_diag_new)
            # Recompute with updated weights
            G_wr = G_w @ jnp.diag(jnp.sqrt(R_diag_new))
            U, S, Vh = jnp.linalg.svd(G_wr, full_matrices=False)
            S_reg = S / (S ** 2 + lambda_reg)
            W = jnp.diag(jnp.sqrt(R_diag_new)) @ Vh.T @ jnp.diag(S_reg) @ U.T @ C_inv_half
            R_diag = R_diag_new

        res_diag = jnp.sum(W * gain.T, axis=1)
        noise_norm = jnp.sqrt(jnp.maximum(jnp.abs(res_diag), 1e-20))

    return W, noise_norm


def apply_inverse(
    W: jnp.ndarray,
    noise_norm: jnp.ndarray,
    data: jnp.ndarray,
    method: str = "dSPM",
) -> jnp.ndarray:
    """Apply inverse operator to sensor data.

    Parameters
    ----------
    W : (n_sources, n_sensors) inverse kernel.
    noise_norm : (n_sources,) normalization.
    data : (n_sensors, n_timepoints) sensor data.
    method : str — normalization method.

    Returns
    -------
    source : (n_sources, n_timepoints) source estimates.
        For dSPM: z-scored (unitless statistical map).
        For sLORETA/eLORETA: standardized current density.
        For MNE: current density (Am).
    """
    source = W @ data  # (n_sources, n_timepoints)

    if method in ("dSPM", "sLORETA", "eLORETA"):
        source = source / noise_norm[:, None]

    return source


# ---------------------------------------------------------------------------
# Resolution matrix, Point Spread Function, Cross-Talk Function
# ---------------------------------------------------------------------------


@jax.jit
def resolution_matrix(W: jnp.ndarray, gain: jnp.ndarray) -> jnp.ndarray:
    """Compute the resolution matrix M = W @ G.

    M[i, j] = how much activity at source j leaks into the estimate at source i.
    Diagonal M[i, i] = how well source i is recovered (localization accuracy).
    Ideally M = I (perfect reconstruction).

    Parameters
    ----------
    W : (n_sources, n_sensors) inverse operator.
    gain : (n_sensors, n_sources) leadfield/forward model.

    Returns
    -------
    M : (n_sources, n_sources) resolution matrix.
    """
    return W @ gain


def point_spread_function(
    M: jnp.ndarray,
    source_idx: int,
) -> jnp.ndarray:
    """Point Spread Function for a single source.

    PSF(i) = column i of M = how a point source at location i
    spreads across the estimated source space.

    A narrow PSF means good spatial resolution at that location.
    A broad PSF means the inverse smears activity over many sources.

    Parameters
    ----------
    M : (n_sources, n_sources) resolution matrix.
    source_idx : int — index of the source of interest.

    Returns
    -------
    psf : (n_sources,) — spread pattern for source_idx.
    """
    return M[:, source_idx]


def cross_talk_function(
    M: jnp.ndarray,
    source_idx: int,
) -> jnp.ndarray:
    """Cross-Talk Function for a single source.

    CTF(i) = row i of M = which sources contribute to the
    estimate at location i.

    A narrow CTF means the estimate at source i is uncontaminated
    by distant sources. A broad CTF means strong cross-talk.

    Parameters
    ----------
    M : (n_sources, n_sources) resolution matrix.
    source_idx : int — index of the target location.

    Returns
    -------
    ctf : (n_sources,) — cross-talk pattern at source_idx.
    """
    return M[source_idx, :]


def resolution_metrics(M: jnp.ndarray) -> dict:
    """Compute summary metrics from the resolution matrix.

    Returns
    -------
    dict with:
        peak_localization_error: (n_sources,) — distance between peak of
            PSF and true source location (0 = perfect, in source indices)
        spatial_spread: (n_sources,) — width of PSF (sum of squared
            off-diagonal elements, smaller = more focal)
        relative_amplitude: (n_sources,) — M[i,i] diagonal, how much of
            the true source amplitude is recovered (1 = perfect)
        cross_talk_leakage: (n_sources,) — sum of squared CTF off-diagonal
            elements (smaller = less contamination from other sources)
    """
    n = M.shape[0]

    # Diagonal: self-recovery amplitude
    relative_amplitude = jnp.diag(M)

    # Peak localization error: where does PSF peak vs true location?
    peak_indices = jnp.argmax(jnp.abs(M), axis=0)  # per column (PSF)
    true_indices = jnp.arange(n)
    peak_localization_error = jnp.abs(peak_indices - true_indices)

    # Spatial spread: off-diagonal energy of PSF
    M_no_diag = M - jnp.diag(jnp.diag(M))
    spatial_spread = jnp.sum(M_no_diag ** 2, axis=0)  # per column (PSF)

    # Cross-talk leakage: off-diagonal energy of CTF
    cross_talk_leakage = jnp.sum(M_no_diag ** 2, axis=1)  # per row (CTF)

    return {
        "peak_localization_error": peak_localization_error,
        "spatial_spread": spatial_spread,
        "relative_amplitude": relative_amplitude,
        "cross_talk_leakage": cross_talk_leakage,
    }


def compare_inverse_methods(
    gain: jnp.ndarray,
    noise_cov: jnp.ndarray,
    methods: tuple[str, ...] = ("MNE", "dSPM", "sLORETA", "eLORETA"),
    depth: float = 0.8,
    reg: float = 0.1,
) -> dict[str, dict]:
    """Compare resolution properties across inverse methods.

    For each method, computes the resolution matrix and summary metrics.
    This answers: which method gives the best localization accuracy,
    least spatial spread, and least cross-talk at each source location?

    Returns
    -------
    dict keyed by method name → resolution_metrics dict.
    """
    results = {}
    for method in methods:
        W, norm = make_inverse_operator(gain, noise_cov, depth, reg, method)
        # For normalized methods, include normalization in resolution
        if method in ("dSPM", "sLORETA", "eLORETA"):
            W_norm = W / norm[:, None]
        else:
            W_norm = W
        M = resolution_matrix(W_norm, gain)
        metrics = resolution_metrics(M)
        metrics["resolution_matrix"] = M
        results[method] = metrics
    return results


def compute_all_variants(
    gain: jnp.ndarray,
    noise_cov: jnp.ndarray,
    data: jnp.ndarray,
    depths: tuple[float, ...] = (0.0, 0.5, 0.8, 1.0),
    regs: tuple[float, ...] = (0.05, 0.1, 0.2),
    methods: tuple[str, ...] = ("MNE", "dSPM", "sLORETA", "eLORETA"),
) -> dict[str, jnp.ndarray]:
    """Compute all MNE variants as a parameter sweep.

    Returns dict keyed by "{method}_depth{d}_reg{r}" → source estimate.
    """
    results = {}
    for method in methods:
        for depth in depths:
            for reg in regs:
                key = f"{method}_depth{depth}_reg{reg}"
                W, norm = make_inverse_operator(gain, noise_cov, depth, reg, method)
                source = apply_inverse(W, norm, data, method)
                results[key] = source
    return results
