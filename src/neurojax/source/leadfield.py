# © NeuroJAX developers
#
# License: BSD (3-clause)
"""Leadfield as a differentiable Lineax operator, with CiftiStorm-style QC.

CiftiStorm module 2 (Valdés-Sosa et al., *Front. Neurosci.* 2024) takes a
BEM/FEM forward solution and applies two *geometric corrections* before the
Bayesian inverse — a source-depth normalisation and a source-orientation
correction — then screens the leadfield (conditioning, dead sources, point
spread). Here both corrections are differentiable parameters of an
:class:`equinox.Module`, so ``jax.grad`` can tune them against sensor data,
and the corrected gain is exposed as a :mod:`lineax` linear operator for the
solvers in :mod:`neurojax.source`.

The forward solution itself comes from OpenMEEG / MNE (``mne.make_forward_solution``,
with ``solver="openmeeg"`` in ``mne.make_bem_solution`` when OpenMEEG is installed)
via :meth:`Leadfield.from_mne_forward`.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np

from neurojax.source.minimum_norm import resolution_matrix

__all__ = ["Leadfield", "leadfield_qc", "point_spread_width"]


class Leadfield(eqx.Module):
    """Free-orientation gain + differentiable depth / orientation corrections.

    Parameters
    ----------
    gain_free : (n_sensors, n_sources, 3) free-orientation gain (static geometry).
    normals : (n_sources, 3) unit cortical normals (static geometry).
    positions : (n_sources, 3) source positions in metres, or ``None``.
    depth : depth-normalisation exponent in [0, 1] — 0 leaves the forward
        model untouched, 1 scales every column to unit norm (full compensation).
        Trainable.
    tilt : (n_sources, 3) additive perturbation of the normals; the effective
        orientation is ``normalize(normals + tilt)``. Trainable, zero by default.
    """

    gain_free: jnp.ndarray
    normals: jnp.ndarray
    positions: jnp.ndarray | None
    depth: jnp.ndarray
    tilt: jnp.ndarray

    def __init__(self, gain_free, normals, positions=None, depth: float = 0.0,
                 tilt=None):
        self.gain_free = jnp.asarray(gain_free)
        self.normals = jnp.asarray(normals)
        self.positions = None if positions is None else jnp.asarray(positions)
        self.depth = jnp.asarray(depth, dtype=self.gain_free.dtype)
        self.tilt = (jnp.zeros_like(self.normals) if tilt is None
                     else jnp.asarray(tilt, dtype=self.normals.dtype))

    @property
    def n_sensors(self) -> int:
        return self.gain_free.shape[0]

    @property
    def n_sources(self) -> int:
        return self.gain_free.shape[1]

    def with_tilt(self, tilt) -> "Leadfield":
        return eqx.tree_at(lambda m: m.tilt, self, jnp.asarray(tilt, self.tilt.dtype))

    def orientations(self) -> jnp.ndarray:
        """(n_sources, 3) unit source orientations after the tilt correction."""
        o = jax.lax.stop_gradient(self.normals) + self.tilt
        return o / jnp.maximum(jnp.linalg.norm(o, axis=1, keepdims=True), 1e-12)

    def gain(self) -> jnp.ndarray:
        """(n_sensors, n_sources) fixed-orientation, depth-corrected gain.

        Depth correction scales column *j* by ``||g_j||^-depth``, renormalised
        so the mean column scale is one (the overall gain magnitude is kept).
        """
        G = jnp.einsum("mnk,nk->mn", jax.lax.stop_gradient(self.gain_free),
                       self.orientations())
        norms = jnp.maximum(jnp.linalg.norm(G, axis=0), 1e-12)
        w = norms ** (-self.depth)
        return G * (w / jnp.mean(w))[None, :]

    def operator(self) -> lx.AbstractLinearOperator:
        """The corrected gain as a Lineax operator (sources -> sensors)."""
        return lx.MatrixLinearOperator(self.gain())

    @classmethod
    def from_mne_forward(cls, fwd, depth: float = 0.0) -> "Leadfield":
        """Import an ``mne.Forward`` (BEM/OpenMEEG/sphere) keeping free orientations.

        Free-orientation forwards keep their (M, N, 3) gain and take the cortical
        normal from the third row of each source's local frame. Fixed-orientation
        forwards are expanded as ``gain[:, :, None] * nn`` so the tilt correction
        still acts on them.
        """
        import mne  # optional dependency

        sol = np.asarray(fwd["sol"]["data"], dtype=np.float32)
        rr = np.asarray(fwd["source_rr"], dtype=np.float32)
        n_src = fwd["nsource"]
        # the cortical normal lives in the source space (free-orientation
        # forwards carry an identity frame in ``source_nn``)
        normals = np.concatenate([s["nn"][s["vertno"]] for s in fwd["src"]]).astype(np.float32)
        if fwd["source_ori"] == mne.io.constants.FIFF.FIFFV_MNE_FREE_ORI:
            gain_free = sol.reshape(sol.shape[0], n_src, 3)
        else:
            fixed_nn = np.asarray(fwd["source_nn"], dtype=np.float32).reshape(n_src, 3)
            gain_free = sol[:, :, None] * fixed_nn[None, :, :]
            normals = fixed_nn
        return cls(gain_free, normals, rr, depth=depth)


def leadfield_qc(gain: jnp.ndarray, rank_tol: float | None = None,
                 outlier_z: float = 3.0) -> dict:
    """Leadfield screening: conditioning, numerical rank, dead/anomalous sources.

    Parameters
    ----------
    gain : (n_sensors, n_sources) fixed-orientation gain.
    rank_tol : singular values below ``rank_tol`` count as zero for the
        effective rank. Default is ``max(M, N) * eps(dtype) * s_max`` (the
        numpy ``matrix_rank`` rule); the value used is returned as ``rank_tol``.
    outlier_z : sources whose log column norm is more than ``outlier_z``
        robust standard deviations (1.4826 * MAD) from the median are flagged.

    Returns
    -------
    dict with ``singular_values``, ``condition_number``, ``effective_rank``,
    ``rank_tol``, ``column_norms``, ``depth_outliers`` (int indices),
    ``n_sensors``, ``n_sources``.
    """
    G = np.asarray(gain)
    s = np.linalg.svd(G, compute_uv=False)
    if rank_tol is None:
        rank_tol = float(max(G.shape) * np.finfo(G.dtype).eps * s[0])
    norms = np.linalg.norm(G, axis=0)
    log_n = np.log(np.maximum(norms, np.finfo(G.dtype).tiny))
    med = np.median(log_n)
    mad = 1.4826 * np.median(np.abs(log_n - med))
    z = (log_n - med) / max(mad, 1e-12)
    return {
        "singular_values": s,
        "condition_number": float(s[0] / s[-1]) if s[-1] > 0 else np.inf,
        "effective_rank": int(np.sum(s > rank_tol)),
        "rank_tol": rank_tol,
        "column_norms": norms,
        "depth_outliers": np.flatnonzero(np.abs(z) > outlier_z),
        "n_sensors": G.shape[0],
        "n_sources": G.shape[1],
    }


def point_spread_width(W: jnp.ndarray, gain: jnp.ndarray,
                       positions: jnp.ndarray) -> jnp.ndarray:
    """Spatial dispersion of each source's point-spread function, in metres.

    ``width_j = sqrt(sum_i M_ij^2 ||r_i - r_j||^2 / sum_i M_ij^2)`` with
    ``M = W @ gain`` the resolution matrix (Molins et al. 2008; Hauk et al. 2011).
    Zero for a perfect inverse, growing with the smearing of the inverse.
    """
    M = resolution_matrix(W, gain)
    r = jnp.asarray(positions)
    d2 = jnp.sum((r[:, None, :] - r[None, :, :]) ** 2, axis=-1)  # (N_i, N_j)
    p = M ** 2
    return jnp.sqrt(jnp.sum(p * d2, axis=0) / jnp.maximum(jnp.sum(p, axis=0), 1e-30))
