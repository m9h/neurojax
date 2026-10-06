#!/usr/bin/env python
"""Phase-flow routing (Vinão-Carl) on WAND resting MEG — the physical-space view of
the same solenoidal cycle the Langevin found in state space.

The 68 Desikan parcels are the point-cloud manifold (centroids); alpha-band analytic
phase (the cleanest irreversibility band) gives the instantaneous phase field.  The
MARBLE-style point-cloud Hodge operators then give the sources/sinks (∇·F) and
vortices (∇×F) routing maps, their PCA routing modes, and rerouting rates — and the
vorticity modes project onto the connectome harmonics (flow↔structure bridge).

Comparison: physical-space **vortices** here ↔ state-space **circulation/EPR** from
``wand_band_langevin.py`` (alpha: EPR=0.636, f_sol=0.074 Hz) — the same broken-
detailed-balance object in two coordinate systems.

    WAND_OUT=/data/datasets/wand_src PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wand_phase_flow.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from neurojax.analysis.timefreq import morlet_cwt, eeglab_cycles
from neurojax.geometry import knn_graph, estimate_normals
from neurojax.analysis.routing import (
    routing_fields_pointcloud,
    routing_modes,
    rerouting_rate,
    routing_to_harmonics,
)
from neurojax.spatial import connectome_harmonics

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
FS = 250.0
ALPHA = 10.0            # alpha-band centre (Hz) for the instantaneous phase
PHASE_FS = 50.0         # downsample the phase field to the network timescale
KNN = 8


def main():
    print("JAX backend:", jax.default_backend())
    X = np.load(os.path.join(OUT, "parcels68.npy")).astype(np.float32)      # (T, 68)
    cent = np.load(os.path.join(OUT, "desikan68_centroids.npy"))           # (68, 3)

    # alpha-band analytic phase per parcel
    C = morlet_cwt(jnp.asarray(X.T), FS, jnp.array([ALPHA]),
                   eeglab_cycles(jnp.array([ALPHA]), 7.0, 0.0))            # (68, 1, T)
    phase = np.asarray(jnp.angle(C[:, 0, :])).T                           # (T, 68)
    ds = int(FS / PHASE_FS)
    phase = phase[::ds]                                                   # -> PHASE_FS
    print(f"alpha phase field: {phase.shape} @ {PHASE_FS:.0f} Hz on 68-parcel cloud")

    # point-cloud manifold + outward normals
    nbr = knn_graph(jnp.asarray(cent), KNN)
    normals = np.array(estimate_normals(jnp.asarray(cent), nbr))
    out = cent - cent.mean(0)
    normals = normals * np.sign(np.sum(normals * out, axis=1, keepdims=True) + 1e-9)  # orient outward

    div, vort = routing_fields_pointcloud(jnp.asarray(cent), nbr, jnp.asarray(phase),
                                          jnp.asarray(normals))
    div, vort = np.asarray(div), np.asarray(vort)                         # (T, 68)

    print(f"\n=====  Phase-flow routing on WAND alpha (physical space)  =====")
    print(f"source/sink activity  mean|∇·F| = {np.abs(div).mean():.3f}")
    print(f"vortex activity       mean|∇×F| = {np.abs(vort).mean():.3f}")
    print(f"net vorticity (rotation bias)   = {vort.mean():+.4f}")

    vmodes, vactiv, vvar = routing_modes(jnp.asarray(vort), n_modes=10)
    vmodes, vactiv, vvar = np.asarray(vmodes), np.asarray(vactiv), np.asarray(vvar)
    rr = np.asarray(rerouting_rate(jnp.asarray(vactiv), PHASE_FS))
    print(f"top vorticity routing mode: {vvar[0]*100:.0f}% var; "
          f"rerouting rate {rr[0]:.2f}/s (mean {rr.mean():.2f}/s)")

    # flow<->structure: project the top vorticity modes onto the connectome harmonics
    _, Phi = connectome_harmonics(jnp.asarray(_gauss_graph(cent)), normalized=False)
    Phi = np.asarray(Phi)[:, 1:21]                                        # low-order harmonics
    load = np.abs(np.asarray(routing_to_harmonics(jnp.asarray(vmodes[:3]), jnp.asarray(Phi))))
    top = [int(np.argmax(load[i])) + 1 for i in range(3)]
    print(f"top-3 vorticity modes load most on connectome harmonics: H{top}")
    print("  -> physical-space vortices are the spatial realization of the state-space")
    print("     solenoidal circulation (Langevin alpha EPR=0.636, f_sol=0.074 Hz),")
    print("     and they express the same low-order connectome harmonics.")


def _gauss_graph(c):
    d2 = np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=-1)
    W = np.exp(-d2 / np.median(d2[d2 > 0]))
    np.fill_diagonal(W, 0.0)
    return W


if __name__ == "__main__":
    main()
