#!/usr/bin/env python
"""Apply the JAX cortical-mesh wave operators to real WH alpha-band source data:
quantify travelling-wave coherence (PGD) and rotational phase singularities per
time frame on the cortical surface.

Descriptive only — apparent MEG waves can arise from source leakage or two beating
dipoles (PMC7615062); this measures the wave statistics, it does not assert a
mechanism. Run after wh_mesh_waves_prep.py:
    WH_OUT=/data/datasets/wh_mesh PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wh_mesh_waves.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from neurojax.analysis.waves import (
    generalized_phase,
    mesh_phase_gradient,
    mesh_phase_gradient_directionality,
    phase_singularity_charge,
)

OUT = os.environ.get("WH_OUT", "/data/datasets/wh_mesh")


def main():
    print("JAX backend:", jax.default_backend())
    verts = jnp.asarray(np.load(os.path.join(OUT, "mesh_vertices.npy")))
    faces = jnp.asarray(np.load(os.path.join(OUT, "mesh_faces.npy")))
    x = np.load(os.path.join(OUT, "source_alpha.npy"))  # (n_verts, n_times)
    print(f"Mesh: {verts.shape[0]} verts, {faces.shape[0]} faces | source {x.shape}")

    # Narrowband (alpha) analytic phase per vertex; unit-amplitude analytic field.
    phase, _ = generalized_phase(jnp.asarray(x), fs=100.0, neg_freq_correction=False)
    V = jnp.exp(1j * phase).T  # (n_times, n_verts)

    def frame_metrics(Vt):
        g = mesh_phase_gradient(Vt, verts, faces)
        pgd = mesh_phase_gradient_directionality(g, verts, faces)
        charge = phase_singularity_charge(Vt, faces)
        n_sing = jnp.sum(jnp.abs(charge) > 0.5)
        return pgd, n_sing

    pgd, n_sing = jax.lax.map(frame_metrics, V)
    pgd = np.asarray(pgd)
    n_sing = np.asarray(n_sing)

    print("\n=========  Mesh travelling-wave metrics on real WH alpha MEG  =========")
    print(f"  frames analysed                 : {len(pgd)}")
    print(f"  phase-gradient directionality   : median {np.median(pgd):.3f}, "
          f"90th pct {np.percentile(pgd, 90):.3f}")
    print(f"  frames with coherent wave (PGD>0.5): {100*np.mean(pgd>0.5):.1f}%")
    print(f"  phase singularities / frame     : mean {n_sing.mean():.1f}, "
          f"max {int(n_sing.max())}")
    print(f"  frames with >=1 singularity     : {100*np.mean(n_sing>=1):.1f}%")
    print("  (descriptive; source-leakage / two-dipole nulls not yet ruled out)")


if __name__ == "__main__":
    main()
