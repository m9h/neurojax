"""Beamformer family — unified JAX implementation.

LCMV, Vector LCMV, SAM, DICS, weight-normalized variants, and
eigenspace projection. All differentiable via JAX.

The core equation:
    W = (G^T C^{-1} G)^{-1} G^T C^{-1}

Variants differ in:
    - Scalar vs vector (orientation handling)
    - Time-domain (LCMV) vs frequency-domain (DICS)
    - Weight normalization (NAI, unit-noise-gain)
    - Noise subspace projection (eigenspace)

References
----------
Van Veen BD et al. (1997). Localization of brain electrical activity
via linearly constrained minimum variance spatial filtering.
IEEE Trans Biomed Eng 44(9):867-880.

Sekihara K et al. (2001). Reconstructing spatio-temporal activities of
neural sources using an MEG vector beamformer technique.
IEEE Trans Biomed Eng 48(7):760-771.

Gross J et al. (2001). Dynamic imaging of coherent sources: studying
neural interactions in the human brain. PNAS 98(2):694-699.

Robinson SE, Vrba J (1999). Functional neuroimaging by synthetic
aperture magnetometry (SAM). In: Recent Advances in Biomagnetism.
"""

from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# LCMV (scalar, basic)
# ---------------------------------------------------------------------------

@jax.jit
def make_lcmv_filter(
    cov: jnp.ndarray,
    gain: jnp.ndarray,
    reg: float = 0.05,
) -> jnp.ndarray:
    """Scalar LCMV beamformer weights.

    W_i = (g_i^T C^{-1} g_i)^{-1} g_i^T C^{-1}

    Parameters
    ----------
    cov : (n_sensors, n_sensors) data covariance.
    gain : (n_sensors, n_sources) leadfield.
    reg : float — Tikhonov regularization.

    Returns
    -------
    W : (n_sources, n_sensors) spatial filter weights.
    """
    n_sensors = cov.shape[0]
    cov_reg = cov + reg * jnp.trace(cov) / n_sensors * jnp.eye(n_sensors)
    cov_inv = jnp.linalg.inv(cov_reg)

    numer = gain.T @ cov_inv  # (n_sources, n_sensors)
    denom = numer @ gain       # (n_sources, n_sources)
    return jnp.linalg.inv(denom) @ numer


@jax.jit
def apply_lcmv(data: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Apply beamformer: S = W @ X."""
    return weights @ data


# ---------------------------------------------------------------------------
# Weight normalization variants
# ---------------------------------------------------------------------------

@jax.jit
def neural_activity_index(
    weights: jnp.ndarray,
    noise_cov: jnp.ndarray,
) -> jnp.ndarray:
    """Neural Activity Index (NAI) normalization.

    Removes the depth bias from beamformer output by normalizing
    each source by the noise projected through its spatial filter.

    NAI_i = W_i @ C_data @ W_i^T / (W_i @ C_noise @ W_i^T)

    Parameters
    ----------
    weights : (n_sources, n_sensors).
    noise_cov : (n_sensors, n_sensors).

    Returns
    -------
    nai_norm : (n_sources,) — normalization factor per source.
    """
    noise_power = jnp.sum((weights @ noise_cov) * weights, axis=1)
    return jnp.sqrt(jnp.maximum(noise_power, 1e-20))


@jax.jit
def unit_noise_gain(
    weights: jnp.ndarray,
    noise_cov: jnp.ndarray,
) -> jnp.ndarray:
    """Unit-noise-gain normalization.

    Rescale weights so that noise power through each filter equals 1.
    Equivalent to NAI normalization applied to the weights themselves.

    Returns
    -------
    W_norm : (n_sources, n_sensors) — normalized weights.
    """
    nai = neural_activity_index(weights, noise_cov)
    return weights / nai[:, None]


# ---------------------------------------------------------------------------
# Vector LCMV (free orientation)
# ---------------------------------------------------------------------------

@jax.jit
def make_vector_lcmv_filter(
    cov: jnp.ndarray,
    gain: jnp.ndarray,
    reg: float = 0.05,
) -> jnp.ndarray:
    """Vector LCMV beamformer for free-orientation dipoles.

    gain has shape (n_sensors, n_sources * 3) where each source has
    3 orientation components (x, y, z). The filter is (n_sources * 3, n_sensors).

    The optimal dipole orientation is the eigenvector of
    (G^T C^{-1} G) corresponding to the maximum eigenvalue.

    Parameters
    ----------
    cov : (n_sensors, n_sensors).
    gain : (n_sensors, n_sources * 3) — 3 orientations per source.
    reg : float.

    Returns
    -------
    W : (n_sources, n_sensors) — scalar weights with optimal orientation.
    orientations : (n_sources, 3) — optimal dipole orientations.
    """
    n_sensors = cov.shape[0]
    n_sources3 = gain.shape[1]
    n_sources = n_sources3 // 3

    cov_reg = cov + reg * jnp.trace(cov) / n_sensors * jnp.eye(n_sensors)
    cov_inv = jnp.linalg.inv(cov_reg)

    W_all = []
    orientations = []

    for i in range(n_sources):
        G_i = gain[:, i * 3:(i + 1) * 3]  # (n_sensors, 3)
        # 3x3 beamformer matrix for this source
        A = G_i.T @ cov_inv @ G_i  # (3, 3)
        A_inv = jnp.linalg.inv(A + 1e-10 * jnp.eye(3))

        # Optimal orientation: eigenvector of A_inv with largest eigenvalue
        eigvals, eigvecs = jnp.linalg.eigh(A_inv)
        opt_ori = eigvecs[:, -1]  # (3,)
        orientations.append(opt_ori)

        # Scalar weight for this orientation
        g_opt = G_i @ opt_ori  # (n_sensors,)
        w_scalar = (g_opt @ cov_inv) / (g_opt @ cov_inv @ g_opt + 1e-10)
        W_all.append(w_scalar)

    return jnp.stack(W_all), jnp.stack(orientations)


# ---------------------------------------------------------------------------
# SAM (Synthetic Aperture Magnetometry)
# ---------------------------------------------------------------------------

@jax.jit
def sam_pseudo_z(
    cov_active: jnp.ndarray,
    cov_control: jnp.ndarray,
    gain: jnp.ndarray,
    reg: float = 0.05,
) -> jnp.ndarray:
    """SAM pseudo-Z statistic for source power mapping.

    Compares beamformer power between active and control conditions.

    Z_i = (P_active_i - P_control_i) / P_control_i

    Parameters
    ----------
    cov_active : (n_sensors, n_sensors) covariance during active window.
    cov_control : (n_sensors, n_sensors) covariance during control/baseline.
    gain : (n_sensors, n_sources).
    reg : float.

    Returns
    -------
    pseudo_z : (n_sources,) — SAM pseudo-Z statistic per source.
    """
    # Build filter from combined covariance (more stable)
    cov_combined = (cov_active + cov_control) / 2
    W = make_lcmv_filter(cov_combined, gain, reg)

    # Source power in each condition
    P_active = jnp.sum((W @ cov_active) * W, axis=1)
    P_control = jnp.sum((W @ cov_control) * W, axis=1)

    return (P_active - P_control) / jnp.maximum(P_control, 1e-20)


# ---------------------------------------------------------------------------
# DICS (Dynamic Imaging of Coherent Sources)
# ---------------------------------------------------------------------------

@jax.jit
def make_dics_filter(
    csd: jnp.ndarray,
    gain: jnp.ndarray,
    reg: float = 0.05,
) -> jnp.ndarray:
    """DICS beamformer for frequency-domain source localization.

    Same as LCMV but operates on the cross-spectral density (CSD)
    matrix at a specific frequency instead of the time-domain covariance.

    Parameters
    ----------
    csd : (n_sensors, n_sensors) complex — cross-spectral density at freq f.
    gain : (n_sensors, n_sources) leadfield.
    reg : float.

    Returns
    -------
    W : (n_sources, n_sensors) complex spatial filter.
    """
    n_sensors = csd.shape[0]
    csd_reg = csd + reg * jnp.trace(jnp.real(csd)) / n_sensors * jnp.eye(n_sensors)
    csd_inv = jnp.linalg.inv(csd_reg)

    numer = gain.T @ csd_inv
    denom = numer @ gain
    return jnp.linalg.inv(denom) @ numer


@jax.jit
def dics_power(
    csd: jnp.ndarray,
    gain: jnp.ndarray,
    reg: float = 0.05,
) -> jnp.ndarray:
    """DICS source power at a specific frequency.

    P_i = real(W_i @ CSD @ W_i^H)

    Returns
    -------
    power : (n_sources,) — source power at this frequency.
    """
    W = make_dics_filter(csd, gain, reg)
    return jnp.real(jnp.sum((W @ csd) * jnp.conj(W), axis=1))


@jax.jit
def dics_coherence(
    csd: jnp.ndarray,
    gain: jnp.ndarray,
    seed_idx: int,
    reg: float = 0.05,
) -> jnp.ndarray:
    """DICS source-space coherence from a seed source.

    Coh(seed, i) = |CSD_source(seed, i)|² / (P_seed · P_i)

    Parameters
    ----------
    csd : (n_sensors, n_sensors) complex CSD.
    gain : (n_sensors, n_sources) leadfield.
    seed_idx : int — index of the seed source.
    reg : float.

    Returns
    -------
    coherence : (n_sources,) — coherence with seed.
    """
    W = make_dics_filter(csd, gain, reg)
    # Source-space CSD
    csd_source = W @ csd @ jnp.conj(W.T)
    power = jnp.real(jnp.diag(csd_source))
    cross = csd_source[seed_idx, :]
    coh = jnp.abs(cross) ** 2 / (jnp.maximum(power[seed_idx] * power, 1e-20))
    return jnp.real(coh)


# ---------------------------------------------------------------------------
# Eigenspace LCMV (noise subspace projection)
# ---------------------------------------------------------------------------

@jax.jit
def make_eigenspace_lcmv_filter(
    cov: jnp.ndarray,
    gain: jnp.ndarray,
    noise_cov: jnp.ndarray,
    n_signal: int = 20,
    reg: float = 0.05,
) -> jnp.ndarray:
    """Eigenspace-projected LCMV beamformer.

    Projects out the noise subspace before beamforming, improving
    performance in low-SNR conditions.

    Parameters
    ----------
    cov : (n_sensors, n_sensors) data covariance.
    gain : (n_sensors, n_sources) leadfield.
    noise_cov : (n_sensors, n_sensors) noise covariance.
    n_signal : int — number of signal subspace dimensions to retain.
    reg : float.

    Returns
    -------
    W : (n_sources, n_sensors) spatial filter.
    """
    n_sensors = cov.shape[0]

    # Whitened covariance
    noise_reg = noise_cov + 1e-6 * jnp.trace(noise_cov) / n_sensors * jnp.eye(n_sensors)
    eigvals_n, eigvecs_n = jnp.linalg.eigh(noise_reg)
    W_noise = eigvecs_n @ jnp.diag(1.0 / jnp.sqrt(jnp.maximum(eigvals_n, 1e-10)))
    cov_white = W_noise.T @ cov @ W_noise

    # Signal subspace: top n_signal eigenvectors
    eigvals_s, eigvecs_s = jnp.linalg.eigh(cov_white)
    signal_space = eigvecs_s[:, -n_signal:]  # (n_sensors, n_signal)

    # Project gain into signal subspace
    P_signal = signal_space @ signal_space.T
    gain_projected = W_noise @ P_signal @ W_noise.T @ gain

    # Standard LCMV on projected data
    return make_lcmv_filter(cov, gain_projected, reg)


# ---------------------------------------------------------------------------
# LCMV power map (for scanning / source localization)
# ---------------------------------------------------------------------------

@jax.jit
def lcmv_power_map(
    cov: jnp.ndarray,
    gain: jnp.ndarray,
    reg: float = 0.05,
) -> jnp.ndarray:
    """Compute LCMV source power at each location.

    P_i = W_i @ C @ W_i^T

    Returns
    -------
    power : (n_sources,) source power estimate.
    """
    W = make_lcmv_filter(cov, gain, reg)
    return jnp.sum((W @ cov) * W, axis=1)
