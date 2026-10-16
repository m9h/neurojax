# © NeuroJAX developers
#
# License: BSD (3-clause)
"""Differentiable Helmholtz-Hodge decomposition of a flow field on a triangle mesh.

A tangent vector field ``F`` on a surface splits as

    F = ∇α  (irrotational — sources/sinks)  ⊕  N×∇β  (solenoidal — vortices)  ⊕  h

(Helmholtz-Hodge).  ``∇·F`` isolates the sources/sinks (sinks = negative
divergence), the scalar ``curl`` isolates the vortices (signed by rotation sense),
and the harmonic remainder ``h`` is both div- and curl-free.  This is the
*physical-space* counterpart of the Langevin gradient/solenoidal drift split in
``jaxctrl._circulation`` (state space): vortices ≡ circulation/EPR, sources/sinks ≡
gradient relaxation — and the field's divergence and vorticity expand in the
connectome-harmonic (graph-Laplacian) eigenbasis.

Operators are discrete-exterior-calculus / FEM on a triangle mesh (vertex scalars,
per-face tangent vectors), fully differentiable in JAX (``jax.grad``/``vmap`` over
time and subjects).  Drives the Vinão-Carl cortical phase-flow routing analysis
(``F = ∇̂φ`` from instantaneous phase); neurojax owns the Hodge half.
"""

import jax.numpy as jnp


def _frames(V, F):
    """Per-face opposite-edge vectors, area, unit normal."""
    p = jnp.asarray(V)[jnp.asarray(F)]                 # (m, 3, 3): face, corner, xyz
    e0 = p[:, 2] - p[:, 1]                              # edge opposite corner 0
    e1 = p[:, 0] - p[:, 2]
    e2 = p[:, 1] - p[:, 0]
    nrm = jnp.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = 0.5 * jnp.linalg.norm(nrm, axis=1)
    N = nrm / (2.0 * area[:, None] + 1e-20)
    return (e0, e1, e2), area, N


def face_gradient(V, F, u):
    """Per-face gradient (m, 3) of a vertex scalar field ``u`` (n,)."""
    (e0, e1, e2), area, N = _frames(V, F)
    uf = jnp.asarray(u)[jnp.asarray(F)]                # (m, 3)
    g = (uf[:, 0:1] * jnp.cross(N, e0)
         + uf[:, 1:2] * jnp.cross(N, e1)
         + uf[:, 2:3] * jnp.cross(N, e2))
    return g / (2.0 * area[:, None] + 1e-20)


def phase_gradient(V, F, phase, normalize=True):
    """Phase-gradient flow field ``F = ∇φ`` on faces from a vertex phase field.

    Wrapping-robust via the analytic representation ``∇φ = Im(z̄ ∇z)``, ``z = e^{iφ}``
    (so 2π jumps don't corrupt the gradient).  With ``normalize`` (default) returns
    the **unit** field ``∇̂φ`` — the local wave-propagation direction whose
    divergence (sources/sinks) and curl (vortices) are the Vinão-Carl cortical
    routing maps.  Compose with :func:`divergence`, :func:`curl`,
    :func:`helmholtz_hodge`.
    """
    phase = jnp.asarray(phase)
    zr, zi = jnp.cos(phase), jnp.sin(phase)
    gzr, gzi = face_gradient(V, F, zr), face_gradient(V, F, zi)
    Ff = jnp.asarray(F)
    zr_f, zi_f = zr[Ff].mean(1)[:, None], zi[Ff].mean(1)[:, None]
    G = zr_f * gzi - zi_f * gzr                        # Im(z̄ ∇z) per face
    if normalize:
        G = G / (jnp.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
    return G


def divergence(V, F, X):
    """Vertex divergence (n,) of a per-face tangent vector field ``X`` (m, 3).

    Sources are positive, sinks negative."""
    (e0, e1, e2), area, N = _frames(V, F)
    F = jnp.asarray(F)
    n = jnp.asarray(V).shape[0]
    X = jnp.asarray(X)
    out = jnp.zeros(n)
    for k, e in enumerate((e0, e1, e2)):
        out = out.at[F[:, k]].add(-0.5 * jnp.sum(jnp.cross(N, e) * X, axis=1))
    return out


def curl(V, F, X):
    """Vertex scalar curl (n,) of ``X`` — vorticity, signed by rotation sense.

    ``curl(X) = −∇·(N×X)``; a counter-clockwise vortex gives positive curl."""
    (_, _, _), _, N = _frames(V, F)
    return -divergence(V, F, jnp.cross(N, jnp.asarray(X)))


def cotangent_laplacian(V, F):
    """Cotangent (FEM stiffness) Laplacian (n, n) = ∇·∇, consistent with the
    grad/div operators above."""
    (e0, e1, e2), area, N = _frames(V, F)
    F = jnp.asarray(F)
    n = jnp.asarray(V).shape[0]
    es = (e0, e1, e2)
    inv4A = 1.0 / (4.0 * area + 1e-20)
    L = jnp.zeros((n, n))
    for a in range(3):
        for b in range(3):
            w = jnp.sum(es[a] * es[b], axis=1) * inv4A
            L = L.at[F[:, a], F[:, b]].add(w)
    return L


def helmholtz_hodge(V, F, X):
    """Helmholtz-Hodge decomposition of a per-face flow field ``X`` (m, 3).

    Returns
    -------
    P : (m, 3) irrotational part ∇α (sources/sinks; ∇·P = ∇·X, curl-free).
    S : (m, 3) solenoidal part N×∇β (vortices; curl(S) = curl(X), div-free).
    H : (m, 3) harmonic remainder ``X − P − S`` (div- and curl-free).
    alpha, beta : (n,) the scalar potentials.
    div, vort : (n,) the divergence and curl (vorticity) fields of ``X``.
    """
    L = cotangent_laplacian(V, F)                      # stiffness; ∇·∇ = −L with sign-physical div
    div = divergence(V, F, X)
    vort = curl(V, F, X)
    alpha = jnp.linalg.lstsq(-L, div)[0]               # L singular (constants) -> min-norm
    beta = jnp.linalg.lstsq(-L, vort)[0]
    P = face_gradient(V, F, alpha)
    (_, _, _), _, N = _frames(V, F)
    S = jnp.cross(N, face_gradient(V, F, beta))
    H = jnp.asarray(X) - P - S
    return P, S, H, alpha, beta, div, vort
