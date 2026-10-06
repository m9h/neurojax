# © NeuroJAX developers
#
# License: BSD (3-clause)
"""Connectome harmonics — the graph-Laplacian eigenbasis of the brain network.

Connectome harmonics (Atasoy, Donnelly & Pearson 2016, *Nat Commun* 7:10340) are the
eigenvectors of the Laplacian of the connectome graph (local gray-matter mesh
adjacency + long-range white-matter tracts from DWI) — the brain's standing-wave
modes, ordered by spatial frequency (eigenvalue).  They are the natural anatomical
basis that diagonalises the dissipative (gradient) part of the Fokker–Planck flow:
each harmonic relaxes at its eigenvalue rate, and the solenoidal / rotational flow
(the stochastic resting cycle) is rotation *among* harmonics (see
``docs/CONNECTOME_HARMONICS_UNIFICATION.md``).

Differentiable in JAX via ``jnp.linalg.eigh``; works on any symmetric non-negative
weighted adjacency (region structural connectivity, surface mesh, or mesh + tracts).
"""

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
# Surface-scale graph construction (numpy / scipy.sparse, offline)
# ---------------------------------------------------------------------------

def mesh_edges(faces):
    """Unique undirected edges of a triangular surface mesh.

    Parameters
    ----------
    faces : (F, 3) integer vertex indices.

    Returns
    -------
    (E, 2) integer array with ``edges[:, 0] < edges[:, 1]``.
    """
    faces = np.asarray(faces)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)


def connectome_adjacency(n_vertices, mesh_edges, tract_edges=None):
    """Binary symmetric adjacency from local mesh edges + long-range tract edges.

    This is the Atasoy 2016 graph: surface-mesh neighbours plus vertex pairs
    joined by tractography streamline endpoints, binarised.

    Parameters
    ----------
    n_vertices : number of surface vertices N.
    mesh_edges : (E, 2) local edges (see :func:`mesh_edges`).
    tract_edges : optional (S, 2) streamline endpoint vertex pairs; duplicates
        and self-connections are ignored.

    Returns
    -------
    (N, N) ``scipy.sparse`` CSR matrix with entries in {0, 1} and zero diagonal.
    """
    edges = np.asarray(mesh_edges)
    if tract_edges is not None:
        edges = np.concatenate([edges, np.asarray(tract_edges)])
    edges = edges[edges[:, 0] != edges[:, 1]]
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_vertices, n_vertices))
    A = A.tocsr()
    A.data[:] = 1.0                                        # duplicates were summed
    return A


def _sparse_harmonics(A, n_modes, normalized):
    """Lanczos solver for the lowest Laplacian eigenpairs of a sparse graph."""
    if n_modes is None:
        raise ValueError("n_modes is required for a sparse adjacency (Lanczos solver)")
    A = sp.csr_matrix(A, dtype=np.float64)
    A = 0.5 * (A + A.T)
    A.setdiag(0.0)
    A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).ravel()
    if np.any(deg == 0):
        raise ValueError(
            f"{int(np.sum(deg == 0))} isolated vertices (zero degree); mask them out "
            "(e.g. medial wall) before computing harmonics.")
    if normalized:
        dinv = sp.diags(1.0 / np.sqrt(deg))
        # L = I - M; the smallest eigenvalues of L are the largest of M
        mu, evecs = eigsh(dinv @ A @ dinv, k=n_modes, which="LA")
        evals = 1.0 - mu
    else:
        L = sp.diags(deg) - A
        evals, evecs = eigsh(L, k=n_modes, sigma=-1e-3, which="LM")
    order = np.argsort(evals)
    evals, evecs = np.clip(evals[order], 0.0, None), evecs[:, order]
    return jnp.asarray(evals, jnp.float32), jnp.asarray(evecs, jnp.float32)


def connectome_harmonics(W, n_modes: int = None, normalized: bool = False):
    """Connectome harmonics = eigenvectors of the graph Laplacian of ``W``.

    Parameters
    ----------
    W : (N, N) symmetric non-negative weighted adjacency (self-loops ignored).
    n_modes : keep the ``n_modes`` lowest-frequency harmonics (default: all N).
    normalized : if True use the symmetric normalised Laplacian
        ``I − D^{-1/2} W D^{-1/2}`` (eigenvalues in [0, 2]); else the combinatorial
        Laplacian ``D − W``.

    Returns
    -------
    eigenvalues : (n_modes,) ascending — spatial frequencies (0 = constant mode).
    harmonics : (N, n_modes) orthonormal eigenvectors (columns), the spatial modes.

    Notes
    -----
    A ``scipy.sparse`` ``W`` (surface-scale graphs from
    :func:`connectome_adjacency`) is solved with Lanczos for the ``n_modes``
    lowest modes and is not differentiable; dense ``W`` uses ``jnp.linalg.eigh``.
    """
    if sp.issparse(W):
        return _sparse_harmonics(W, n_modes, normalized)
    W = jnp.asarray(W)
    W = 0.5 * (W + W.T)                                    # symmetrise
    W = W - jnp.diag(jnp.diag(W))                          # drop self-loops
    deg = jnp.sum(W, axis=1)
    if normalized:
        dinv = jnp.where(deg > 0, 1.0 / jnp.sqrt(deg), 0.0)
        L = jnp.eye(W.shape[0]) - (dinv[:, None] * W * dinv[None, :])
    else:
        L = jnp.diag(deg) - W
    L = 0.5 * (L + L.T)                                    # exact symmetry for eigh
    evals, evecs = jnp.linalg.eigh(L)
    if n_modes is not None:
        evals, evecs = evals[:n_modes], evecs[:, :n_modes]
    return evals, evecs


def project_harmonics(X, harmonics):
    """Project activity onto the harmonic basis -> harmonic-coefficient series.

    Parameters
    ----------
    X : (..., N) spatial activity (e.g. (T, N) parcel time series, or (N,)).
    harmonics : (N, n_modes) harmonics from :func:`connectome_harmonics`.

    Returns
    -------
    (..., n_modes) coefficients ``a = X · Φ`` (the harmonic latent for the dynamics).
    """
    return jnp.asarray(X) @ jnp.asarray(harmonics)


def harmonic_power_spectrum(X, harmonics):
    """Mean power per harmonic — the connectome-harmonic 'spectrum' of activity.

    Returns
    -------
    (n_modes,) mean squared harmonic coefficient over the leading axes of ``X``.
    """
    a = project_harmonics(X, harmonics)
    return jnp.mean(a ** 2, axis=tuple(range(a.ndim - 1)))


# ---------------------------------------------------------------------------
# Connectome-harmonic decomposition metrics (Atasoy 2017; Luppi 2023)
# ---------------------------------------------------------------------------

def harmonic_energy(alpha, eigenvalues):
    """Energy per harmonic, ``E_k = |a_k|^2 lambda_k^2`` (Atasoy et al. 2017).

    The ``lambda^2`` weighting means spatially white noise, whose ``|a_k|^2`` is
    flat, contributes energy that grows with wavenumber — report the unweighted
    power spectrum alongside it.
    """
    return alpha ** 2 * eigenvalues ** 2


def log_binned_spectrum(values, n_bins: int = 15, reduce: str = "mean"):
    """Aggregate a per-harmonic quantity into log-spaced wavenumber bins.

    Bin edges are integer harmonic indices, log-spaced and forced strictly
    increasing so every bin holds at least one harmonic (requires K >= n_bins).

    Returns
    -------
    binned : (..., n_bins)
    edges : (n_bins + 1,) integer edges; bin i covers harmonics [edges[i], edges[i+1]).
    """
    K = values.shape[-1]
    if K < n_bins:
        raise ValueError(f"need at least n_bins={n_bins} harmonics, got {K}")
    edges = np.round(np.logspace(0.0, np.log10(K), n_bins + 1)).astype(int)
    edges[0], edges[-1] = 0, K
    for i in range(1, n_bins):
        edges[i] = max(edges[i], edges[i - 1] + 1)
    for i in range(n_bins - 1, 0, -1):
        edges[i] = min(edges[i], edges[i + 1] - 1)
    bin_ids = np.searchsorted(edges, np.arange(K), side="right") - 1
    onehot = jnp.asarray(np.eye(n_bins)[bin_ids], dtype=values.dtype)   # (K, n_bins)
    binned = values @ onehot
    if reduce == "mean":
        binned = binned / onehot.sum(axis=0)
    elif reduce != "sum":
        raise ValueError(f"unknown reduce: {reduce!r}")
    return binned, jnp.asarray(edges)


def repertoire_entropy(power, eps: float = 1e-12):
    """Shannon entropy (nats) of the normalised distribution of power over harmonics.

    0 for a single active harmonic, ``log(K)`` for a flat spectrum. Scale
    invariant, and therefore unable to distinguish added high-wavenumber
    energy from lost low-wavenumber energy — use with :func:`log_binned_spectrum`.
    """
    p = power / (jnp.sum(power, axis=-1, keepdims=True) + eps)
    return -jnp.sum(jnp.where(p > 0, p * jnp.log(p + eps), 0.0), axis=-1)
