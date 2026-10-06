"""Differentiable Helmholtz-Hodge decomposition on a triangle mesh.

The spatial-field sibling of the Langevin gradient/solenoidal split: a tangent
vector field F on a surface decomposes as F = ∇α (irrotational, sources/sinks) ⊕
N×∇β (solenoidal, vortices) ⊕ harmonic.  ∇·F isolates sources/sinks, the scalar
curl isolates vortices — the Vinão-Carl cortical phase-flow routing operators
(neurojax owns the differentiable Hodge half).
"""

import numpy as np
import jax.numpy as jnp

from neurojax.geometry.hodge import (
    face_gradient,
    divergence,
    curl,
    helmholtz_hodge,
    phase_gradient,
    _frames,
)


def flat_grid(k=20):
    """A flat triangulated unit square (z=0), CCW faces -> +z normal."""
    xs = np.linspace(0.0, 1.0, k)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(k * k)], axis=1).astype(np.float64)
    F = []
    for i in range(k - 1):
        for j in range(k - 1):
            a, b = i * k + j, i * k + (j + 1)
            c, d = (i + 1) * k + j, (i + 1) * k + (j + 1)
            F.append([a, b, c])
            F.append([b, d, c])
    return jnp.asarray(V), jnp.asarray(np.array(F))


def icosphere(n_sub=2):
    """A closed genus-0 sphere mesh (no boundary -> clean Hodge decomposition)."""
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t],
                  [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1],
                  [-t, 0, -1], [-t, 0, 1]], float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    Fc = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                   [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                   [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                   [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    for _ in range(n_sub):
        Vl = list(V)
        mids = {}

        def mid(a, b):
            key = tuple(sorted((a, b)))
            if key not in mids:
                m = V[a] + V[b]
                Vl.append(m / np.linalg.norm(m))
                mids[key] = len(Vl) - 1
            return mids[key]

        newF = []
        for a, b, c in Fc:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            newF += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        V, Fc = np.array(Vl), np.array(newF)
    return jnp.asarray(V), jnp.asarray(Fc)


def _centroids(V, F):
    return np.asarray(V)[np.asarray(F)].mean(1)


def _interior(V, lo=0.25, hi=0.75):
    Vn = np.asarray(V)
    return (Vn[:, 0] > lo) & (Vn[:, 0] < hi) & (Vn[:, 1] > lo) & (Vn[:, 1] < hi)


def test_gradient_of_linear_field():
    V, F = flat_grid(16)
    u = 2.0 * np.asarray(V)[:, 0] + 3.0 * np.asarray(V)[:, 1]      # ∇ = (2, 3)
    g = np.asarray(face_gradient(V, F, jnp.asarray(u)))
    np.testing.assert_allclose(g[:, :2].mean(0), [2.0, 3.0], atol=1e-4)


def test_source_has_positive_divergence():
    V, F = flat_grid(28)
    C = _centroids(V, F)
    X = np.stack([C[:, 0] - 0.5, C[:, 1] - 0.5, np.zeros(len(C))], axis=1)  # radial out
    d = np.asarray(divergence(V, F, jnp.asarray(X)))
    assert d[_interior(V)].mean() > 0                              # source ⇒ div > 0


def test_vortex_has_positive_curl():
    V, F = flat_grid(28)
    C = _centroids(V, F)
    X = np.stack([-(C[:, 1] - 0.5), C[:, 0] - 0.5, np.zeros(len(C))], axis=1)  # CCW
    c = np.asarray(curl(V, F, jnp.asarray(X)))
    assert c[_interior(V)].mean() > 0                             # CCW vortex ⇒ curl > 0


def test_phase_gradient_traveling_wave():
    # a planar travelling wave φ = k·x (wrapped to ±π) -> ∇̂φ points along k, even
    # across the 2π wraps (analytic-signal gradient)
    V, F = flat_grid(24)
    k = np.array([6.0, 0.0])
    phase = (np.asarray(V)[:, :2] @ k + np.pi) % (2 * np.pi) - np.pi    # wrapped
    Fhat = np.asarray(phase_gradient(V, F, jnp.asarray(phase)))         # unit field
    interior = _interior(V)[np.asarray(F)[:, 0]]                        # faces near interior verts
    mean_dir = Fhat[interior, :2].mean(0)
    mean_dir /= np.linalg.norm(mean_dir) + 1e-12
    np.testing.assert_allclose(mean_dir, [1.0, 0.0], atol=0.05)         # along +x despite wrapping


def test_curl_of_gradient_is_zero():
    V, F = flat_grid(20)
    u = np.sin(3.0 * np.asarray(V)[:, 0]) * np.cos(2.0 * np.asarray(V)[:, 1])
    g = face_gradient(V, F, jnp.asarray(u))
    c = np.asarray(curl(V, F, g))
    assert np.abs(c[_interior(V)]).max() < 0.05 * np.abs(np.asarray(g)).max()


def test_hodge_separates_and_reconstructs():
    # closed sphere -> no boundary, harmonic part ~0, decomposition is clean
    V, F = icosphere(3)
    (_, _, _), _, N = _frames(V, F)
    a = np.asarray(V)[:, 2]                                    # a smooth scalar (z-coord)
    Firr = face_gradient(V, F, jnp.asarray(a))                 # planted irrotational
    P, S, H, *_ = helmholtz_hodge(V, F, Firr)
    P, S, H = np.asarray(P), np.asarray(S), np.asarray(H)
    np.testing.assert_allclose(P + S + H, np.asarray(Firr), atol=1e-4)   # exact reconstruction
    assert np.linalg.norm(S) < 0.2 * np.linalg.norm(P)                   # mostly irrotational

    Fsol = jnp.cross(jnp.asarray(N), face_gradient(V, F, jnp.asarray(a)))  # planted solenoidal
    P2, S2, _, *_ = helmholtz_hodge(V, F, Fsol)
    assert np.linalg.norm(np.asarray(P2)) < 0.2 * np.linalg.norm(np.asarray(S2))
