# © NeuroJAX developers
#
# License: BSD (3-clause)
"""Phase-flow routing-mode analysis (Vinão-Carl cortical routing architecture).

Pipeline: a phase time-series ``φ(t)`` on a cortical mesh → per-time **divergence**
(sources/sinks) and **vorticity** (vortices) fields via the differentiable
Helmholtz-Hodge operators (``neurojax.geometry.hodge``) → low-frequency **routing
modes** (PCA of the field time-series) and their time activations → **rerouting
rates** (zero-crossings/s of the activations), the dynamic statistic that PLS'd
against regional grey-matter volume (r ≈ 0.89; Vinão-Carl et al. 2025).

The vorticity routing modes are the physical-space counterpart of the state-space
solenoidal circulation (``jaxctrl._circulation``); the mode spatial patterns project
onto the connectome-harmonic eigenbasis (``neurojax.spatial.connectome_harmonics``).
"""

import jax
import jax.numpy as jnp
import numpy as np

from neurojax.geometry.hodge import phase_gradient, divergence, curl
from neurojax.geometry.hodge_pointcloud import (
    point_phase_gradient,
    point_divergence,
    point_vorticity,
)


def routing_fields_pointcloud(X, nbr, phase_ts, normals):
    """Point-cloud backend of :func:`routing_fields` (MARBLE-style manifold, no mesh).

    ``X`` (n, 3) node positions, ``nbr`` (n, k) neighbour indices
    (:func:`~neurojax.geometry.hodge_pointcloud.knn_graph`), ``normals`` (n, 3).
    Returns (divergence, vorticity) fields, each (T, n) — the sensor-manifold routing
    maps exactly as Vinão-Carl compute them."""
    def one(phase):
        Fhat = point_phase_gradient(X, nbr, phase)
        return point_divergence(X, nbr, Fhat), point_vorticity(X, nbr, Fhat, normals)

    div, vort = jax.vmap(one)(jnp.asarray(phase_ts))
    return div, vort


def routing_fields(V, F, phase_ts):
    """Phase time-series (T, n_vert) -> (divergence, vorticity) fields, each (T, n_vert).

    For each timepoint: ``F = ∇̂φ`` (unit phase gradient) -> ``∇·F`` (sources/sinks)
    and ``curl(F)`` (vortices).  vmap-ed over time."""
    def one(phase):
        Fhat = phase_gradient(V, F, phase)
        return divergence(V, F, Fhat), curl(V, F, Fhat)

    div, vort = jax.vmap(one)(jnp.asarray(phase_ts))
    return div, vort


def routing_modes(field_ts, n_modes=25):
    """Routing modes = PCA of a field time-series (T, n).

    Returns
    -------
    modes : (n_modes, n) spatial patterns (right singular vectors), low-freq first.
    activations : (T, n_modes) time courses (``U·s``).
    var : (n_modes,) fraction of variance each mode explains.
    """
    X = jnp.asarray(field_ts)
    Xc = X - jnp.mean(X, axis=0, keepdims=True)
    U, s, Vt = jnp.linalg.svd(Xc, full_matrices=False)
    k = min(n_modes, s.shape[0])
    var = (s[:k] ** 2) / (jnp.sum(s ** 2) + 1e-30)
    return Vt[:k], U[:, :k] * s[:k], var


def rerouting_rate(activations, fs):
    """Zero-crossings per second of each routing-mode activation (T, n_modes).

    The Vinão-Carl 'rerouting rate' — how fast the routing reconfigures."""
    a = np.asarray(activations)
    a = a - a.mean(0, keepdims=True)
    crossings = np.sum(np.diff(np.sign(a), axis=0) != 0, axis=0)
    duration = a.shape[0] / float(fs)
    return crossings / duration


def phase_flow_routing(V, F, phase_ts, fs, n_modes=25):
    """End-to-end: phase time-series -> divergence & vorticity routing modes,
    activations, and rerouting rates.

    Returns a dict with ``div_modes/div_activ/div_var/div_rerouting`` and the
    matching ``vort_*`` entries."""
    div, vort = routing_fields(V, F, phase_ts)
    out = {}
    for name, field in (("div", div), ("vort", vort)):
        modes, activ, var = routing_modes(field, n_modes)
        out[f"{name}_modes"] = modes
        out[f"{name}_activ"] = activ
        out[f"{name}_var"] = var
        out[f"{name}_rerouting"] = rerouting_rate(activ, fs)
    return out
