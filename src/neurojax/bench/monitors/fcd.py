"""Functional connectivity dynamics (FCD) computation in pure JAX.

Ports neurolib's FCD (moving-window correlation matrix) and KS distance
to JAX. FCD captures temporal fluctuations in FC by computing FC in
sliding windows and then correlating the windowed FC vectors.
"""

import jax
import jax.numpy as jnp

from neurojax.bench.monitors.fc import fc_triu


def fcd(
    timeseries: jnp.ndarray,
    window_size: int = 30,
    step_size: int = 5,
) -> jnp.ndarray:
    """Compute functional connectivity dynamics (FCD) matrix.

    For each sliding window, compute the upper-triangular FC vector.
    The FCD matrix is the Pearson correlation between all pairs of
    windowed FC vectors.

    Args:
        timeseries: (n_regions, n_timepoints) array of BOLD signals.
        window_size: Number of timepoints per window.
        step_size: Step between consecutive windows.

    Returns:
        (n_windows, n_windows) FCD matrix.
    """
    n_regions, n_time = timeseries.shape
    # Compute window start indices
    starts = jnp.arange(0, n_time - window_size + 1, step_size)
    n_windows = len(starts)

    # Extract FC vectors for each window
    def _window_fc(start):
        window = jax.lax.dynamic_slice(
            timeseries, (0, start), (n_regions, window_size)
        )
        return fc_triu(window)

    fc_vectors = jax.vmap(_window_fc)(starts)  # (n_windows, n_fc_elements)

    # FCD = correlation matrix of FC vectors
    centered = fc_vectors - jnp.mean(fc_vectors, axis=1, keepdims=True)
    norms = jnp.sqrt(jnp.sum(centered**2, axis=1, keepdims=True))
    norms = jnp.where(norms == 0, 1.0, norms)
    normalized = centered / norms
    return normalized @ normalized.T


def fcd_triu(
    timeseries: jnp.ndarray,
    window_size: int = 30,
    step_size: int = 5,
) -> jnp.ndarray:
    """Extract upper-triangular elements of the FCD matrix.

    Args:
        timeseries: (n_regions, n_timepoints) array.
        window_size: Window size for FCD.
        step_size: Step size for FCD.

    Returns:
        1D array of upper-triangular FCD values.
    """
    fcd_matrix = fcd(timeseries, window_size, step_size)
    n = fcd_matrix.shape[0]
    rows, cols = jnp.triu_indices(n, k=1)
    return fcd_matrix[rows, cols]


def fcd_ks_distance(
    ts1: jnp.ndarray,
    ts2: jnp.ndarray,
    window_size: int = 30,
    step_size: int = 5,
) -> float:
    """Kolmogorov-Smirnov distance between FCD distributions.

    Computes the KS statistic between the upper-triangular FCD elements
    of two timeseries. Lower values indicate more similar FCD dynamics.

    Args:
        ts1: (n_regions, n_timepoints) first timeseries.
        ts2: (n_regions, n_timepoints) second timeseries.
        window_size: Window size for FCD.
        step_size: Step size for FCD.

    Returns:
        Scalar KS distance in [0, 1].
    """
    fcd1 = fcd_triu(ts1, window_size, step_size)
    fcd2 = fcd_triu(ts2, window_size, step_size)
    return _ks_statistic(fcd1, fcd2)


def _ks_statistic(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Two-sample KS statistic (maximum CDF difference).

    Pure JAX implementation without scipy dependency.
    """
    # Combine and sort all values
    combined = jnp.concatenate([x, y])
    combined_sorted = jnp.sort(combined)

    n_x = x.shape[0]
    n_y = y.shape[0]

    # Compute empirical CDFs at each combined point
    # CDF_x(t) = (number of x_i <= t) / n_x
    # Using broadcasting: for each sorted value, count how many x values are <= it
    cdf_x = jnp.sum(x[:, None] <= combined_sorted[None, :], axis=0) / n_x
    cdf_y = jnp.sum(y[:, None] <= combined_sorted[None, :], axis=0) / n_y

    return jnp.max(jnp.abs(cdf_x - cdf_y))
