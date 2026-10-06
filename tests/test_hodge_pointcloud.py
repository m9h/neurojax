"""Point-cloud Helmholtz-Hodge operators (MARBLE-style; MIT-licensed algorithm).

The manifold is a kNN graph over a point cloud (e.g. the electrode cloud, per
Vinão-Carl/MARBLE — no surface mesh required).  Local tangent-space derivatives are
fit by weighted least-squares over each node's neighbours (the local Jacobian),
giving the same divergence (sources/sinks) and vorticity (vortices) the mesh version
computes, on the *sensor* manifold.
"""

import numpy as np
import jax.numpy as jnp

from neurojax.geometry.hodge_pointcloud import (
    knn_graph,
    point_gradient,
    point_divergence,
    point_vorticity,
    point_phase_gradient,
)


def cloud(k=20):
    """A flat point cloud on z=0 (normals = +z), with a small jitter."""
    xs = np.linspace(0.0, 1.0, k)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(k * k)], axis=1).astype(np.float64)
    return jnp.asarray(P)


def _interior(P, lo=0.25, hi=0.75):
    Pn = np.asarray(P)
    return (Pn[:, 0] > lo) & (Pn[:, 0] < hi) & (Pn[:, 1] > lo) & (Pn[:, 1] < hi)


def test_point_gradient_linear():
    P = cloud(20)
    nbr = knn_graph(P, 10)
    f = 2.0 * np.asarray(P)[:, 0] + 3.0 * np.asarray(P)[:, 1]
    g = np.asarray(point_gradient(P, nbr, jnp.asarray(f)))
    np.testing.assert_allclose(g[_interior(P)].mean(0), [2.0, 3.0, 0.0], atol=1e-3)


def test_point_source_positive_divergence():
    P = cloud(22)
    nbr = knn_graph(P, 10)
    Pn = np.asarray(P)
    F = np.stack([Pn[:, 0] - 0.5, Pn[:, 1] - 0.5, np.zeros(len(Pn))], axis=1)  # radial
    d = np.asarray(point_divergence(P, nbr, jnp.asarray(F)))
    np.testing.assert_allclose(d[_interior(P)].mean(), 2.0, atol=0.3)          # div(x,y)=2


def test_point_vortex_positive_vorticity():
    P = cloud(22)
    nbr = knn_graph(P, 10)
    Pn = np.asarray(P)
    F = np.stack([-(Pn[:, 1] - 0.5), Pn[:, 0] - 0.5, np.zeros(len(Pn))], axis=1)  # CCW
    normals = np.tile([0.0, 0.0, 1.0], (len(Pn), 1))
    w = np.asarray(point_vorticity(P, nbr, jnp.asarray(F), jnp.asarray(normals)))
    np.testing.assert_allclose(w[_interior(P)].mean(), 2.0, atol=0.3)          # curl=2


def test_point_curl_of_gradient_zero():
    # linear scalar -> constant gradient (LS-exact) -> curl of a constant field is 0
    P = cloud(20)
    nbr = knn_graph(P, 10)
    Pn = np.asarray(P)
    f = 2.0 * Pn[:, 0] + 3.0 * Pn[:, 1]
    g = point_gradient(P, nbr, jnp.asarray(f))
    normals = jnp.asarray(np.tile([0.0, 0.0, 1.0], (len(Pn), 1)))
    w = np.asarray(point_vorticity(P, nbr, g, normals))
    assert np.abs(w[_interior(P)]).max() < 0.05 * np.linalg.norm([2.0, 3.0])


def test_point_phase_gradient_wave():
    P = cloud(24)
    nbr = knn_graph(P, 10)
    k = np.array([6.0, 0.0])
    phase = (np.asarray(P)[:, :2] @ k + np.pi) % (2 * np.pi) - np.pi          # wrapped
    G = np.asarray(point_phase_gradient(P, nbr, jnp.asarray(phase)))
    d = G[_interior(P), :2].mean(0)
    d /= np.linalg.norm(d) + 1e-12
    np.testing.assert_allclose(d, [1.0, 0.0], atol=0.05)
