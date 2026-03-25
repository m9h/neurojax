"""Shared fixtures for bench tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def toy_connectome_4node():
    """4-node toy connectome with known connectivity and delays.

    Returns (weights, delays) where:
    - weights: symmetric 4x4 connectivity matrix (no self-connections)
    - delays: symmetric 4x4 delay matrix in ms
    """
    weights = jnp.array([
        [0.0, 0.5, 0.2, 0.0],
        [0.5, 0.0, 0.3, 0.1],
        [0.2, 0.3, 0.0, 0.4],
        [0.0, 0.1, 0.4, 0.0],
    ])
    delays = jnp.array([
        [0.0, 5.0, 10.0, 0.0],
        [5.0, 0.0, 3.0, 8.0],
        [10.0, 3.0, 0.0, 6.0],
        [0.0, 8.0, 6.0, 0.0],
    ])
    return weights, delays


@pytest.fixture
def synthetic_bold_4node():
    """Synthetic BOLD-like timeseries for 4 nodes, 200 timepoints.

    Generates correlated timeseries to produce non-trivial FC.
    """
    key = jax.random.PRNGKey(42)
    n_regions, n_time = 4, 200

    # Create correlated signals via mixing matrix
    keys = jax.random.split(key, 2)
    independent = jax.random.normal(keys[0], (n_regions, n_time))
    mixing = jnp.array([
        [1.0, 0.6, 0.2, 0.0],
        [0.6, 1.0, 0.4, 0.1],
        [0.2, 0.4, 1.0, 0.5],
        [0.0, 0.1, 0.5, 1.0],
    ])
    # Cholesky factor for generating correlated noise
    L = jnp.linalg.cholesky(mixing)
    correlated = L @ independent
    return correlated


@pytest.fixture
def identical_timeseries():
    """Two identical timeseries for testing zero-distance properties."""
    key = jax.random.PRNGKey(123)
    ts = jax.random.normal(key, (4, 100))
    return ts, ts.copy()


@pytest.fixture
def constant_timeseries():
    """Constant (zero-variance) timeseries for NaN-safety testing."""
    return jnp.ones((4, 100))
