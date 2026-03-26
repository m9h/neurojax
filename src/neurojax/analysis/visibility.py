"""Visibility graphs: convert time series to complex networks.

Inspired by pyunicorn's VisibilityGraph. Implements natural and
horizontal visibility graph construction, plus basic graph measures.

References
----------
Lacasa L et al. (2008). From time series to complex networks:
The visibility graph. PNAS 105(13):4972-4975.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def natural_visibility_graph(ts: jnp.ndarray) -> jnp.ndarray:
    """Construct a natural visibility graph from a 1D time series.

    Nodes i and j (i < j) are connected if for all k in (i, j):
        ts[k] < ts[i] + (ts[j] - ts[i]) * (k - i) / (j - i)

    Parameters
    ----------
    ts : (T,) time series.

    Returns
    -------
    adj : (T, T) symmetric binary adjacency matrix.
    """
    ts = np.asarray(ts)
    T = len(ts)
    adj = np.zeros((T, T), dtype=np.float32)

    for i in range(T):
        for j in range(i + 1, T):
            visible = True
            for k in range(i + 1, j):
                # Interpolated height at k
                height = ts[i] + (ts[j] - ts[i]) * (k - i) / (j - i)
                if ts[k] >= height:
                    visible = False
                    break
            if visible:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    return jnp.array(adj)


def horizontal_visibility_graph(ts: jnp.ndarray) -> jnp.ndarray:
    """Construct a horizontal visibility graph from a 1D time series.

    Nodes i and j (i < j) are connected if for all k in (i, j):
        ts[k] < min(ts[i], ts[j])

    Parameters
    ----------
    ts : (T,) time series.

    Returns
    -------
    adj : (T, T) symmetric binary adjacency matrix.
    """
    ts = np.asarray(ts)
    T = len(ts)
    adj = np.zeros((T, T), dtype=np.float32)

    for i in range(T):
        for j in range(i + 1, T):
            threshold = min(ts[i], ts[j])
            visible = True
            for k in range(i + 1, j):
                if ts[k] >= threshold:
                    visible = False
                    break
            if visible:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    return jnp.array(adj)


# ---------------------------------------------------------------------------
# Graph measures
# ---------------------------------------------------------------------------

def vg_degree(adj: jnp.ndarray) -> jnp.ndarray:
    """Degree of each node."""
    return jnp.sum(adj, axis=1)


def vg_mean_degree(adj: jnp.ndarray) -> float:
    """Average degree."""
    return float(jnp.mean(vg_degree(adj)))


def vg_clustering(adj: jnp.ndarray) -> jnp.ndarray:
    """Local clustering coefficient per node."""
    A = np.asarray(adj > 0.5, dtype=float)
    T = A.shape[0]
    cc = np.zeros(T)
    for i in range(T):
        neighbors = np.where(A[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            cc[i] = 0.0
            continue
        # Count edges among neighbors
        sub = A[np.ix_(neighbors, neighbors)]
        edges = np.sum(sub) / 2
        cc[i] = 2 * edges / (k * (k - 1))
    return jnp.array(cc)


def vg_degree_distribution(adj: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Degree histogram.

    Returns
    -------
    degrees : unique degree values
    counts : count per degree
    """
    deg = np.asarray(vg_degree(adj), dtype=int)
    unique, counts = np.unique(deg, return_counts=True)
    return jnp.array(unique), jnp.array(counts)


def vg_assortativity(adj: jnp.ndarray) -> float:
    """Degree-degree correlation (assortativity coefficient)."""
    A = np.asarray(adj > 0.5, dtype=float)
    deg = np.sum(A, axis=1)
    edges = np.argwhere(np.triu(A, k=1) > 0)
    if len(edges) == 0:
        return 0.0
    d_i = deg[edges[:, 0]]
    d_j = deg[edges[:, 1]]
    if np.std(d_i) == 0 or np.std(d_j) == 0:
        return 0.0
    return float(np.corrcoef(d_i, d_j)[0, 1])
