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
    """
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
