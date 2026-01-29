"""
JAX-based LCMV Beamformer.

Linearly Constrained Minimum Variance (LCMV) Beamformer.
W = (G.T @ C^-1 @ G)^-1 @ G.T @ C^-1

References:
- Van Veen et al., 1997.
"""

import jax
import jax.numpy as jnp
from jax import jit

@jit
def make_lcmv_filter(cov: jnp.ndarray, gain: jnp.ndarray, reg: float = 0.05) -> jnp.ndarray:
    """
    Compute LCMV weights.
    
    Parameters
    ----------
    cov: (n_chan, n_chan) Data covariance.
    gain: (n_chan, n_sources) Forward solution (Leadfield).
    reg: Regularization factor (Tikhonov).
    
    Returns
    -------
    weights: (n_sources, n_chan)
    """
    n_chan = cov.shape[0]
    
    # 1. Regularize Covariance
    # C_reg = C + reg * trace(C)/C * I
    trace = jnp.trace(cov)
    eye = jnp.eye(n_chan)
    cov_reg = cov + reg * (trace / n_chan) * eye
    
    # 2. Invert Covariance
    # C_inv = inv(C_reg)
    cov_inv = jnp.linalg.inv(cov_reg)
    
    # 3. Compute Weights
    # Denominator: G.T @ C_inv @ G
    # Numerator: G.T @ C_inv
    
    numer = jnp.dot(gain.T, cov_inv) # (n_src, n_chan)
    
    denom = jnp.dot(numer, gain) # (n_src, n_src)
    
    # Invert Denominator
    # Ensure symmetry/stability
    denom_inv = jnp.linalg.inv(denom)
    
    weights = jnp.dot(denom_inv, numer) # (n_src, n_chan)
    
    return weights

@jit
def apply_lcmv(data: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """
    Apply weights to data.
    Linear projection: S = W @ X
    """
    return jnp.dot(weights, data)

@jit
def lcmv_power(cov: jnp.ndarray, gain: jnp.ndarray, reg: float = 0.05) -> jnp.ndarray:
    """
    Compute LCMV Power map directly (for scanning).
    P = tr((G^T C^-1 G)^-1)
    """
    n_chan = cov.shape[0]
    trace = jnp.trace(cov)
    eye = jnp.eye(n_chan)
    cov_reg = cov + reg * (trace / n_chan) * eye
    cov_inv = jnp.linalg.inv(cov_reg)
    
    # G.T @ C^-1 @ G
    inner = jnp.dot(jnp.dot(gain.T, cov_inv), gain)
    
    # Power is trace of inverse
    return jnp.trace(jnp.linalg.inv(inner))
