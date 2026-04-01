"""LAURA (Local AUtoRegressive Average) source localization in JAX.

Distributed inverse solution using a biophysical spatial prior where
current density falls off as 1/d³ from each source point, motivated
by the physics of volume conduction (dipole fields decay as inverse
cube of distance).

This distinguishes LAURA from LORETA (which uses a Laplacian smoothness
prior with no biophysical justification) and from VARETA (which uses
data-driven adaptive resolution).

The inverse is:
    J = W @ L^T @ (L @ W @ L^T + λ @ C_noise)^{-1} @ Y

where W is the LAURA weight matrix with W_ij ∝ 1/||r_i - r_j||³.

References:
    Grave de Peralta Menendez R et al. (2001) NeuroImage 14:68-78
    Grave de Peralta Menendez R et al. (2004) IEEE TBME 51(1):73-82
    Michel CM et al. (2004) Clinical Neurophysiology 115:2195-2222
"""

import jax
import jax.numpy as jnp
from functools import partial


def laura_weight_matrix(positions: jnp.ndarray,
                         exponent: float = 3.0,
                         regularize: float = 1e-3) -> jnp.ndarray:
    """Compute the LAURA spatial weight matrix.

    W_ij = 1 / ||r_i - r_j||^exponent  for i ≠ j
    W_ii = 0  (no self-interaction)

    The exponent=3 corresponds to the biophysical dipole field decay.

    Args:
        positions: (n_sources, 3) array of source positions in mm
        exponent: distance exponent (default 3 for dipole physics)
        regularize: small constant to avoid division by zero

    Returns:
        (n_sources, n_sources) symmetric weight matrix
    """
    n = positions.shape[0]

    # Pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]  # (n, n, 3)
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + regularize ** 2)  # (n, n)

    # 1/d^exponent weights
    W = 1.0 / (dist ** exponent)

    # Zero diagonal (no self-interaction)
    W = W * (1.0 - jnp.eye(n))

    # Normalise rows to sum to 1 (makes it a proper averaging operator)
    row_sums = jnp.sum(W, axis=1, keepdims=True)
    W = W / jnp.maximum(row_sums, 1e-10)

    return W


@partial(jax.jit, static_argnums=(4, 5))
def laura(data: jnp.ndarray,
          gain: jnp.ndarray,
          positions: jnp.ndarray,
          noise_cov: jnp.ndarray,
          reg_param: float = 0.05,
          exponent: float = 3.0) -> jnp.ndarray:
    """LAURA inverse solution.

    J = W @ L^T @ (L @ W @ L^T + λ C)^{-1} @ Y

    Args:
        data: (n_sensors, n_times) sensor measurements
        gain: (n_sensors, n_sources) leadfield/gain matrix
        positions: (n_sources, 3) source positions in mm
        noise_cov: (n_sensors, n_sensors) noise covariance matrix
        reg_param: regularisation parameter λ (Tikhonov)
        exponent: distance exponent for weight matrix (3 = dipole)

    Returns:
        (n_sources, n_times) estimated source activity
    """
    n_sensors, n_sources = gain.shape

    # Compute LAURA weight matrix
    W = laura_weight_matrix(positions, exponent=exponent)

    # Source covariance prior: C_source = W (row-normalised)
    # The LAURA prior encodes that nearby sources covary
    # Convert to a proper covariance: W @ W^T (ensures PSD)
    C_source = W @ W.T + jnp.eye(n_sources) * 1e-6

    # Weighted gain
    LW = gain @ C_source  # (n_sensors, n_sources)

    # Data covariance in sensor space
    # (L @ C_source @ L^T + λ C_noise)
    sensor_cov = LW @ gain.T + reg_param * noise_cov

    # Inverse operator: K = C_source @ L^T @ sensor_cov^{-1}
    K = C_source @ gain.T @ jnp.linalg.inv(sensor_cov)

    # Apply to data
    J = K @ data

    return J
