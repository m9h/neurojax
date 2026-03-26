"""Functional network construction: MI, cross-correlation, network measures.

Inspired by pyunicorn's CouplingAnalysis and Network classes.
Provides nonlinear coupling measures (mutual information) and
standard graph-theoretic measures for brain network characterization.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Mutual Information
# ---------------------------------------------------------------------------

def mutual_information(x: jnp.ndarray, y: jnp.ndarray, n_bins: int = 32) -> float:
    """Mutual information between two 1D signals using histogram binning.

    Parameters
    ----------
    x, y : (T,) signals.
    n_bins : int — number of histogram bins per dimension.

    Returns
    -------
    mi : float — mutual information in nats.
    """
    x_np = np.asarray(x).flatten()
    y_np = np.asarray(y).flatten()

    # 2D histogram
    hist_2d, _, _ = np.histogram2d(x_np, y_np, bins=n_bins)
    p_xy = hist_2d / hist_2d.sum()

    # Marginals
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mask = p_xy > 0
    outer = p_x[:, None] * p_y[None, :]
    mi = np.sum(p_xy[mask] * np.log(p_xy[mask] / outer[mask]))
    return float(mi)


def mutual_information_matrix(data: jnp.ndarray, n_bins: int = 32) -> jnp.ndarray:
    """MI matrix for all channel pairs.

    Parameters
    ----------
    data : (T, C) — multichannel time series.
    n_bins : int

    Returns
    -------
    mi_mat : (C, C) — symmetric MI matrix.
    """
    data_np = np.asarray(data)
    C = data_np.shape[1]
    mi_mat = np.zeros((C, C))
    for i in range(C):
        for j in range(i, C):
            mi = mutual_information(data_np[:, i], data_np[:, j], n_bins)
            mi_mat[i, j] = mi
            mi_mat[j, i] = mi
    return jnp.array(mi_mat)


# ---------------------------------------------------------------------------
# Lagged cross-correlation
# ---------------------------------------------------------------------------

def lagged_cross_correlation(
    x: jnp.ndarray, y: jnp.ndarray, max_lag: int = 50
) -> jnp.ndarray:
    """Cross-correlation at multiple lags.

    Parameters
    ----------
    x, y : (T,) signals.
    max_lag : int — maximum lag in samples.

    Returns
    -------
    cc : (2*max_lag + 1,) — correlation at lags [-max_lag, ..., +max_lag].
    """
    x = jnp.asarray(x) - jnp.mean(x)
    y = jnp.asarray(y) - jnp.mean(y)
    sx = jnp.sqrt(jnp.sum(x ** 2))
    sy = jnp.sqrt(jnp.sum(y ** 2))
    norm = jnp.maximum(sx * sy, 1e-12)

    cc = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = jnp.sum(x[:len(x) - lag] * y[lag:]) / norm
        else:
            c = jnp.sum(x[-lag:] * y[:len(y) + lag]) / norm
        cc.append(c)
    return jnp.array(cc)


def optimal_lag(
    x: jnp.ndarray, y: jnp.ndarray, max_lag: int = 50
) -> tuple[int, float]:
    """Lag with maximum absolute cross-correlation.

    Returns
    -------
    lag : int — optimal lag (negative = x leads y).
    correlation : float — correlation at optimal lag.
    """
    cc = lagged_cross_correlation(x, y, max_lag)
    idx = int(jnp.argmax(jnp.abs(cc)))
    lag = idx - max_lag
    return lag, float(cc[idx])


# ---------------------------------------------------------------------------
# Network measures
# ---------------------------------------------------------------------------

def degree(W: jnp.ndarray, threshold: float | None = None) -> jnp.ndarray:
    """Node degree from adjacency/weight matrix.

    Parameters
    ----------
    W : (N, N) — adjacency or weight matrix.
    threshold : float — binarize at this value before counting.

    Returns
    -------
    deg : (N,) — degree per node.
    """
    if threshold is not None:
        A = (jnp.abs(W) > threshold).astype(float)
    else:
        A = (jnp.abs(W) > 0).astype(float)
    # Remove self-connections
    A = A * (1 - jnp.eye(A.shape[0]))
    return jnp.sum(A, axis=1)


def threshold_matrix(W: jnp.ndarray, density: float = 0.1) -> jnp.ndarray:
    """Threshold weight matrix to keep top density fraction of connections.

    Parameters
    ----------
    W : (N, N) — weight matrix.
    density : float in (0, 1) — fraction of connections to keep.

    Returns
    -------
    A : (N, N) — binary adjacency matrix.
    """
    W_abs = jnp.abs(W)
    # Zero out diagonal
    W_abs = W_abs * (1 - jnp.eye(W.shape[0]))
    cutoff = jnp.percentile(W_abs[W_abs > 0], 100 * (1 - density))
    return (W_abs >= cutoff).astype(float)


def clustering_coefficient(W: jnp.ndarray) -> jnp.ndarray:
    """Local clustering coefficient per node.

    Parameters
    ----------
    W : (N, N) binary adjacency matrix.

    Returns
    -------
    cc : (N,) clustering coefficients.
    """
    A = np.asarray(W > 0.5, dtype=float)
    np.fill_diagonal(A, 0)
    N = A.shape[0]
    cc = np.zeros(N)
    for i in range(N):
        neighbors = np.where(A[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            continue
        sub = A[np.ix_(neighbors, neighbors)]
        triangles = np.sum(sub) / 2
        cc[i] = 2 * triangles / (k * (k - 1))
    return jnp.array(cc)


def global_clustering(W: jnp.ndarray) -> float:
    """Mean clustering coefficient."""
    return float(jnp.mean(clustering_coefficient(W)))


def characteristic_path_length(W: jnp.ndarray) -> float:
    """Average shortest path length via Floyd-Warshall.

    Parameters
    ----------
    W : (N, N) binary adjacency matrix.

    Returns
    -------
    L : float — average shortest path (inf if disconnected).
    """
    A = np.asarray(W > 0.5, dtype=float)
    np.fill_diagonal(A, 0)
    N = A.shape[0]
    # Initialize distance matrix
    dist = np.full((N, N), np.inf)
    dist[A > 0] = 1.0
    np.fill_diagonal(dist, 0)
    # Floyd-Warshall
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    # Average over connected pairs
    mask = (dist < np.inf) & (dist > 0)
    if not np.any(mask):
        return float("inf")
    return float(np.mean(dist[mask]))


def betweenness_centrality(W: jnp.ndarray) -> jnp.ndarray:
    """Node betweenness centrality (fraction of shortest paths through node).

    Uses Brandes' algorithm with BFS for unweighted graphs.
    """
    A = np.asarray(W > 0.5, dtype=float)
    np.fill_diagonal(A, 0)
    N = A.shape[0]
    bc = np.zeros(N)

    for s in range(N):
        # BFS from s
        stack = []
        predecessors = [[] for _ in range(N)]
        sigma = np.zeros(N)
        sigma[s] = 1.0
        dist = np.full(N, -1)
        dist[s] = 0
        queue = [s]

        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in np.where(A[v] > 0)[0]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        delta = np.zeros(N)
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]

    # Normalize
    if N > 2:
        bc /= (N - 1) * (N - 2)

    return jnp.array(bc)


def small_world_index(W: jnp.ndarray, n_random: int = 10) -> float:
    """Small-world index sigma = (C/C_rand) / (L/L_rand).

    Parameters
    ----------
    W : (N, N) binary adjacency matrix.
    n_random : int — number of random graphs for comparison.

    Returns
    -------
    sigma : float — small-world index (>1 indicates small-world).
    """
    C_real = global_clustering(W)
    L_real = characteristic_path_length(W)

    if L_real == float("inf") or L_real == 0:
        return 0.0

    A = np.asarray(W > 0.5, dtype=float)
    np.fill_diagonal(A, 0)
    n_edges = int(np.sum(A) / 2)
    N = A.shape[0]

    C_rands = []
    L_rands = []
    rng = np.random.default_rng(0)

    for _ in range(n_random):
        # Generate Erdos-Renyi random graph with same N and edge count
        rand_adj = np.zeros((N, N))
        edges_added = 0
        while edges_added < n_edges:
            i, j = rng.integers(0, N, size=2)
            if i != j and rand_adj[i, j] == 0:
                rand_adj[i, j] = 1.0
                rand_adj[j, i] = 1.0
                edges_added += 1
        C_rands.append(global_clustering(jnp.array(rand_adj)))
        L_rands.append(characteristic_path_length(jnp.array(rand_adj)))

    C_rand = np.mean(C_rands)
    L_rand = np.mean(L_rands)

    if C_rand == 0 or L_rand == 0:
        return 0.0

    return (C_real / C_rand) / (L_real / L_rand)
