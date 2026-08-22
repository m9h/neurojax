#!/usr/bin/env python
"""JAX cortical-mesh wave analysis of real WH alpha-band source data, with
null-testing so the metrics are interpretable:

  - spatial smoothing of the analytic field (suppresses phase-noise singularities),
  - amplitude-thresholded singularity counts (only genuinely oscillating cortex),
  - a SPATIAL-SHUFFLE null (permute vertices -> destroy spatial phase structure)
    to give the chance level of PGD and singularity counts.

A travelling/rotating-wave structure is only credible if the real metrics exceed
the shuffle null. Run after wh_mesh_waves_prep.py:
    WH_OUT=/data/datasets/wh_mesh PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wh_mesh_waves.py
"""

import os

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from neurojax.analysis.waves import (
    face_amplitude,
    generalized_phase,
    mesh_phase_gradient,
    mesh_phase_gradient_directionality,
    phase_singularity_charge,
    smooth_field_mesh,
)

OUT = os.environ.get("WH_OUT", "/data/datasets/wh_mesh")
NSMOOTH = int(os.environ.get("WH_SMOOTH", "3"))
N_NULL_FRAMES = 200


def main():
    print("JAX backend:", jax.default_backend())
    verts = jnp.asarray(np.load(os.path.join(OUT, "mesh_vertices.npy")))
    faces = jnp.asarray(np.load(os.path.join(OUT, "mesh_faces.npy")))
    x = np.load(os.path.join(OUT, "source_alpha.npy"))
    print(f"Mesh: {verts.shape[0]} verts, {faces.shape[0]} faces | source {x.shape} "
          f"| smoothing passes={NSMOOTH}")

    phase, amp = generalized_phase(jnp.asarray(x), fs=100.0, neg_freq_correction=False)
    V = jnp.exp(1j * phase).T          # (T, n_verts)
    A = amp.T                          # (T, n_verts)

    def metrics(Vt, at):
        Vs = smooth_field_mesh(Vt, faces, n_iter=NSMOOTH)
        g = mesh_phase_gradient(Vs, verts, faces)
        pgd = mesh_phase_gradient_directionality(g, verts, faces)
        ch = phase_singularity_charge(Vs, faces)
        fa = face_amplitude(at, faces)
        thr = jnp.percentile(fa, 75.0)
        n_sig = jnp.sum((jnp.abs(ch) > 0.5) & (fa > thr))
        return pgd, n_sig

    pgd, n_sig = jax.lax.map(lambda a: metrics(*a), (V, A))
    pgd, n_sig = np.asarray(pgd), np.asarray(n_sig)

    # Spatial-shuffle null on a random subset of frames.
    T = V.shape[0]
    idx = np.asarray(jr.choice(jr.PRNGKey(0), T, (N_NULL_FRAMES,), replace=False))
    perm = jr.permutation(jr.PRNGKey(1), verts.shape[0])
    Vn, An = V[idx][:, perm], A[idx][:, perm]
    pgd_n, nsig_n = jax.lax.map(lambda a: metrics(*a), (Vn, An))
    pgd_n, nsig_n = np.asarray(pgd_n), np.asarray(nsig_n)

    print("\n=====  Mesh travelling-wave metrics on real WH alpha MEG (smoothed)  =====")
    print(f"  frames analysed                     : {len(pgd)}")
    print(f"  PGD            real median {np.median(pgd):.3f}  |  null median {np.median(pgd_n):.3f}"
          f"  (null 95th {np.percentile(pgd_n, 95):.3f})")
    print(f"  amp-thresholded singularities/frame : real {n_sig.mean():.1f}  |  null {nsig_n.mean():.1f}")
    real_above = float(np.mean(pgd > np.percentile(pgd_n, 95)))
    print(f"  frames with PGD above null-95th     : {100*real_above:.1f}%")
    verdict = ("ABOVE chance" if (np.median(pgd) > np.percentile(pgd_n, 95)
                                  or n_sig.mean() > 1.5 * nsig_n.mean())
               else "NOT above chance")
    print(f"  verdict: spatial wave structure is {verdict} vs the shuffle null")


if __name__ == "__main__":
    main()
