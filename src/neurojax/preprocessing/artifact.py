"""Artifact detection routines using Riemannian geometry."""
import jax.numpy as jnp
from jax import vmap
from neurojax.geometry.riemann import covariance_mean, riemannian_distance

def detect_artifacts_riemann(covariances: jnp.ndarray, n_std: float = 3.0) -> jnp.ndarray:
    """
    Detect artifact epochs using Minimum Distance to Riemannian Mean (MDRM).
    
    Args:
        covariances: (N, C, C) SPD matrices.
        n_std: Threshold in standard deviations for rejection.
        
    Returns:
        mask: (N,) boolean array where True indicates an artifact.
    """
    # 1. Compute Geometric Mean
    mean_cov = covariance_mean(covariances)
    
    # 2. Compute distances from each epoch to the mean
    distances = vmap(lambda c: riemannian_distance(c, mean_cov))(covariances)
    
    # 3. Z-score normalization of distances
    # Note: Distances are strictly positive. 
    # For robust rejection, we might want median/MAD, but standard is requested.
    dist_mean = jnp.mean(distances)
    dist_std = jnp.std(distances)
    z_scores = (distances - dist_mean) / (dist_std + 1e-8)
    
    # 4. Reject outliers
    return z_scores > n_std
