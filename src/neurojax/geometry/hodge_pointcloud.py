# © NeuroJAX developers
#
# License: BSD (3-clause)
#
# Algorithm adapted (re-implemented in JAX) from MARBLE's local tangent-space
# derivatives (Gosztolai et al., Nat. Methods 2024; github Dynamics-of-Neural-
# Systems-Lab/MARBLE, MIT-licensed): kNN manifold + local-PCA tangent frames +
# neighbour least-squares gradient.
"""Point-cloud Helmholtz-Hodge / flow operators (no surface mesh required).

The manifold is a kNN graph over a point cloud (e.g. the electrode cloud, per
Vinão-Carl/MARBLE).  Local tangent-space derivatives are fit by ridge-regularised
least-squares over each node's neighbours — the local Jacobian ``J`` such that
``F(x_j) − F(x_i) ≈ J (x_j − x_i)`` — from which ``divergence = tr(J)`` (sources/
sinks) and ``vorticity = n·curl(J)`` (vortices).  The point-cloud backend of
``neurojax.geometry.hodge``; drops into ``neurojax.analysis.routing``.
"""

import jax
import jax.numpy as jnp


def knn_graph(X, k=12):
    """Indices (n, k) of the k nearest neighbours of each point (self excluded)."""
    X = jnp.asarray(X)
    D = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    D = D + jnp.eye(X.shape[0]) * 1e30                  # exclude self
    return jax.lax.top_k(-D, k)[1]


def estimate_normals(X, nbr):
    """Per-point unit normal (smallest-variance local-PCA direction).  Sign/orientation
    is arbitrary — pass explicit normals where the vorticity sign must be consistent."""
    X = jnp.asarray(X)
    dp = X[nbr] - X[:, None, :]
    cov = jnp.einsum("nki,nkj->nij", dp, dp)
    _, V = jnp.linalg.eigh(cov)                         # ascending eigenvalues
    return V[:, :, 0]


def _local_fit(dp, dy, ridge):
    """Least-squares M (3, m): ``dy ≈ dp @ M`` (ridge-regularised normal equations)."""
    A = dp.T @ dp + ridge * jnp.eye(3)
    return jnp.linalg.solve(A, dp.T @ dy)


def point_gradient(X, nbr, f, ridge=1e-6):
    """Tangent gradient (n, 3) of a node scalar field ``f``."""
    X, f = jnp.asarray(X), jnp.asarray(f)
    dp = X[nbr] - X[:, None, :]
    dy = (f[nbr] - f[:, None])[..., None]
    M = jax.vmap(_local_fit, in_axes=(0, 0, None))(dp, dy, ridge)
    return M[..., 0]


def point_jacobian(X, nbr, F, ridge=1e-6):
    """Per-node Jacobian (n, 3, 3) of a vector field ``F`` (n, 3): F_a ≈ J_ab dx_b."""
    X, F = jnp.asarray(X), jnp.asarray(F)
    dp = X[nbr] - X[:, None, :]
    dF = F[nbr] - F[:, None, :]
    M = jax.vmap(_local_fit, in_axes=(0, 0, None))(dp, dF, ridge)   # dF ≈ dp @ M -> J = Mᵀ
    return jnp.transpose(M, (0, 2, 1))


def point_divergence(X, nbr, F, ridge=1e-6):
    """Vertex divergence ``tr(J)`` (n,) — sources positive, sinks negative."""
    return jnp.trace(point_jacobian(X, nbr, F, ridge), axis1=1, axis2=2)


def point_vorticity(X, nbr, F, normals, ridge=1e-6):
    """Scalar vorticity ``n·(∇×F)`` (n,) — CCW vortex positive."""
    J = point_jacobian(X, nbr, F, ridge)
    omega = jnp.stack([J[:, 2, 1] - J[:, 1, 2],
                       J[:, 0, 2] - J[:, 2, 0],
                       J[:, 1, 0] - J[:, 0, 1]], axis=1)            # axial vector of antisym(J)
    return jnp.sum(omega * jnp.asarray(normals), axis=1)


def point_phase_gradient(X, nbr, phase, normalize=True, ridge=1e-6):
    """Wrapping-robust unit phase-gradient field ``∇̂φ`` (n, 3) via ``Im(z̄ ∇z)``."""
    phase = jnp.asarray(phase)
    zr, zi = jnp.cos(phase), jnp.sin(phase)
    gzr = point_gradient(X, nbr, zr, ridge)
    gzi = point_gradient(X, nbr, zi, ridge)
    G = zr[:, None] * gzi - zi[:, None] * gzr
    if normalize:
        G = G / (jnp.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
    return G
