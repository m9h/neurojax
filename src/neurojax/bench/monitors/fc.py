"""Functional connectivity computation in pure JAX.

Ports neurolib's func.fc (Pearson correlation of BOLD timeseries) and
matrix_correlation to JAX for GPU acceleration and differentiability.
"""

import jax
import jax.numpy as jnp


def fc(timeseries: jnp.ndarray) -> jnp.ndarray:
    """Compute functional connectivity matrix via Pearson correlation.

    Args:
        timeseries: (n_regions, n_timepoints) array of BOLD signals.

    Returns:
        (n_regions, n_regions) Pearson correlation matrix.
    """
    # Center each region's timeseries
    centered = timeseries - jnp.mean(timeseries, axis=1, keepdims=True)
    # Normalize — use jnp.maximum for grad-safe zero guard
    norms = jnp.sqrt(jnp.sum(centered**2, axis=1, keepdims=True))
    norms = jnp.maximum(norms, 1e-12)
    normalized = centered / norms
    # Correlation matrix = normalized @ normalized.T
    return normalized @ normalized.T


def fc_triu(timeseries: jnp.ndarray) -> jnp.ndarray:
    """Extract upper-triangular elements of the FC matrix.

    Args:
        timeseries: (n_regions, n_timepoints) array.

    Returns:
        1D array of upper-triangular FC values (excluding diagonal).
    """
    fc_matrix = fc(timeseries)
    n = fc_matrix.shape[0]
    rows, cols = jnp.triu_indices(n, k=1)
    return fc_matrix[rows, cols]


def matrix_correlation(mat1: jnp.ndarray, mat2: jnp.ndarray) -> float:
    """Pearson correlation between upper-triangular elements of two matrices.

    This is the standard metric for comparing simulated vs empirical FC.
    Uses masked sum instead of fancy indexing for JAX differentiability.

    Args:
        mat1: (N, N) matrix.
        mat2: (N, N) matrix.

    Returns:
        Scalar Pearson correlation coefficient.
    """
    n = mat1.shape[0]
    mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    n_elements = n * (n - 1) // 2

    # Compute masked means
    mean1 = jnp.sum(jnp.where(mask, mat1, 0.0)) / n_elements
    mean2 = jnp.sum(jnp.where(mask, mat2, 0.0)) / n_elements

    # Centered values (zero out non-mask elements)
    c1 = jnp.where(mask, mat1 - mean1, 0.0)
    c2 = jnp.where(mask, mat2 - mean2, 0.0)

    num = jnp.sum(c1 * c2)
    denom = jnp.sqrt(jnp.sum(c1**2) * jnp.sum(c2**2))
    denom = jnp.maximum(denom, 1e-12)
    return num / denom


def _pearson_r(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Pearson correlation coefficient between two 1D arrays."""
    x_centered = x - jnp.mean(x)
    y_centered = y - jnp.mean(y)
    num = jnp.sum(x_centered * y_centered)
    denom = jnp.sqrt(jnp.sum(x_centered**2) * jnp.sum(y_centered**2))
    # Use jnp.maximum for differentiable zero-guard
    denom = jnp.maximum(denom, 1e-12)
    return num / denom
