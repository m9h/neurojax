"""Riemannian geometry for SPD matrices."""
import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Optional, Tuple

def _powm(A: jnp.ndarray, k: float) -> jnp.ndarray:
    """Compute matrix power A^k for symmetric A."""
    eigvals, eigvecs = jnp.linalg.eigh(A)
    # Ensure positive eigenvalues for fractional powers
    eigvals = jnp.where(eigvals > 0, eigvals, 1e-14)
    return (eigvecs * (eigvals ** k)) @ eigvecs.T

def _logm(A: jnp.ndarray) -> jnp.ndarray:
    """Compute matrix logarithm for symmetric A."""
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.where(eigvals > 0, eigvals, 1e-14) # Numerical stability
    return (eigvecs * jnp.log(eigvals)) @ eigvecs.T

def _expm(A: jnp.ndarray) -> jnp.ndarray:
    """Compute matrix exponential for symmetric A."""
    eigvals, eigvecs = jnp.linalg.eigh(A)
    return (eigvecs * jnp.exp(eigvals)) @ eigvecs.T

@jit
def riemannian_distance(A: jnp.ndarray, B: jnp.ndarray) -> float:
    """
    Compute Riemannian distance between two SPD matrices.
    d_R(A, B) = || log(A^{-1/2} B A^{-1/2}) ||_F
    """
    inv_half_A = _powm(A, -0.5)
    term = inv_half_A @ B @ inv_half_A
    log_term = _logm(term)
    return jnp.linalg.norm(log_term, ord='fro')

@jit
def log_map(C: jnp.ndarray, C_ref: jnp.ndarray) -> jnp.ndarray:
    """
    Map covariance C to tangent space at C_ref.
    Log_{C_ref}(C) = C_ref^{1/2} log(C_ref^{-1/2} C C_ref^{-1/2}) C_ref^{1/2}
    """
    half_ref = _powm(C_ref, 0.5)
    inv_half_ref = _powm(C_ref, -0.5)
    inner = _logm(inv_half_ref @ C @ inv_half_ref)
    return half_ref @ inner @ half_ref

@jit
def exp_map(T: jnp.ndarray, C_ref: jnp.ndarray) -> jnp.ndarray:
    """
    Map tangent vector T back to manifold at C_ref.
    Exp_{C_ref}(T) = C_ref^{1/2} exp(C_ref^{-1/2} T C_ref^{-1/2}) C_ref^{1/2}
    """
    half_ref = _powm(C_ref, 0.5)
    inv_half_ref = _powm(C_ref, -0.5)
    inner = _expm(inv_half_ref @ T @ inv_half_ref)
    return half_ref @ inner @ half_ref

@jit
def _mean_loop_body(val):
    mean, covs, err, iter_idx = val
    # Project all covs to tangent space of current mean
    tangents = vmap(lambda c: log_map(c, mean))(covs)
    # Average tangent vectors
    avg_tangent = jnp.mean(tangents, axis=0)
    # Move mean along average tangent
    new_mean = exp_map(avg_tangent, mean)
    # Error is norm of average tangent step
    new_err = jnp.linalg.norm(avg_tangent)
    return new_mean, covs, new_err, iter_idx + 1

@jit
def _mean_cond_fun(val):
    mean, covs, err, iter_idx = val
    # Max iter hardcoded if not passed, but we pass it effectively via init
    return (err > 1e-6) & (iter_idx < 50)

def covariance_mean(covariances: jnp.ndarray, max_iter: int = 50) -> jnp.ndarray:
    """
    Compute Geometric Mean (Fréchet Mean) of covariances.
    Iteratively updates mean until convergence or max_iter.
    """
    # Initialize with Arithmetic mean
    mean = jnp.mean(covariances, axis=0)
    err = 1.0
    iter_idx = 0
    
    # We use a while_loop with iteration limit
    mean, _, final_err, _ = jax.lax.while_loop(
        _mean_cond_fun, 
        _mean_loop_body, 
        (mean, covariances, err, iter_idx)
    )
    return mean

@jit
def tangent_space_vectorize(T: jnp.ndarray) -> jnp.ndarray:
    """
    Vectorize a symmetric tangent matrix.
    Extracts upper triangle. Scales off-diagonals by sqrt(2) to preserve norm.
    """
    n = T.shape[0]
    # Indices for upper triangle
    triu_indices = jnp.triu_indices(n)
    vec = T[triu_indices]
    
    # Correction mask: off-diagonals need sqrt(2)
    # In triu_indices, diagonals satisfy row == col
    rows, cols = triu_indices
    correction = jnp.where(rows == cols, 1.0, jnp.sqrt(2.0))
    
    return vec * correction

def map_tangent_space(covariances: jnp.ndarray, mean: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Project covariances to tangent space vectors.
    """
    if mean is None:
        mean = covariance_mean(covariances)
    
    # Map to tangent matrices
    tangents = vmap(lambda c: log_map(c, mean))(covariances)
    
    # Vectorize
    vectors = vmap(tangent_space_vectorize)(tangents)
    return vectors
