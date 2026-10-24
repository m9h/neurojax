#!/usr/bin/env python
"""Surface Laplace-Beltrami eigenmodes (Pang 2023 geometric harmonics) -> Desikan-68.

The proper geometric basis (vs the coarse centroid-Gaussian graph): per-hemisphere
cortical-surface LBO eigenmodes via `lapy`, aggregated to the 68 Desikan regions in
exact `parcels68` order, so the MEG can project onto them.  Hemispheres are separate
surfaces (no cortical-surface connection across the midline), so each mode lives on
one hemisphere — the honest geometric-surface structure (cf. the structural connectome,
which bridges hemispheres via the corpus callosum).

Runs in `.venv-oracle` (lapy + MNE/nibabel for the fsaverage surface/annot).
Saves `desikan68_surfaceLBO.npz` (Phi 68×n DC-free, evals) for the broadband analysis.

    .venv-oracle/bin/python scripts/real_data/wand_surface_lbo.py
"""

import os

import numpy as np
import mne
from mne.datasets import fetch_fsaverage
from lapy import TriaMesh, Solver

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
N_MODES = 30                # LBO modes per hemisphere


def main():
    fs = fetch_fsaverage(verbose="ERROR")
    sd = os.path.dirname(fs)
    names = [n for n in open(os.path.join(OUT, "desikan68_names.txt")).read().split("\n") if n]
    idx = {n: i for i, n in enumerate(names)}                  # parcels68 order
    labels = [l for l in mne.read_labels_from_annot(
        "fsaverage", "aparc", subjects_dir=sd, verbose="ERROR")
        if "unknown" not in l.name]

    cols, evals = [], []
    for hemi in ("lh", "rh"):
        v, f = mne.read_surface(os.path.join(sd, "fsaverage", "surf", f"{hemi}.white"))
        ev, evec = Solver(TriaMesh(v, f)).eigs(k=N_MODES)      # (N,), (n_vert, N)
        for m in range(N_MODES):
            col = np.zeros(68)
            for l in labels:
                if l.hemi == hemi:
                    col[idx[l.name]] = float(np.mean(evec[l.vertices, m]))
            cols.append(col)
            evals.append(float(ev[m]))
    Phi = np.array(cols).T                                     # (68, 2*N_MODES)
    evals = np.array(evals)
    order = np.argsort(evals)
    Phi, evals = Phi[:, order], evals[order]
    Phi, evals = Phi[:, 2:], evals[2:]                         # drop the 2 hemisphere DC modes

    np.savez(os.path.join(OUT, "desikan68_surfaceLBO.npz"), Phi=Phi.astype(np.float32), evals=evals)
    print(f"surface-LBO region harmonics: Phi {Phi.shape}, λ1..5 = {np.round(evals[:5], 4)}")


if __name__ == "__main__":
    main()
