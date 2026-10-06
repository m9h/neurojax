#!/usr/bin/env python
"""Source-localize real WH MEG to the cortical MESH (vertex-level, left hemi) in
the alpha band, and save the mesh + source time series so the JAX mesh wave
operators (analysis/waves.py) can look for travelling / rotating waves.

Unlike wh_source_prep_oracle.py (which parcellates to 68 labels for the HMM),
this keeps the full decimated source mesh so phase gradients and singularities
can be computed on the surface.

Run in the osl/MNE env:
    .venv-oracle/bin/python -u scripts/real_data/wh_mesh_waves_prep.py
"""

import os
import os.path as op

import mne
import numpy as np
from mne.beamformer import apply_lcmv_raw, make_lcmv
from mne.datasets import fetch_fsaverage

OUT = os.environ.get("WH_OUT", "/data/datasets/wh_mesh")
BASE = os.environ.get("WH_BASE", "/data/datasets/ds000117-download")
SUBJECT = os.environ.get("WH_SUBJECT", "sub-05")
RUN = int(os.environ.get("WH_RUN", "1"))
CROP = float(os.environ.get("WH_CROP", "60.0"))  # seconds (bounded for memory)
LO, HI = 8.0, 12.0  # alpha band — classic MEG travelling-wave band


def main():
    os.makedirs(OUT, exist_ok=True)
    fs_dir = fetch_fsaverage(verbose="ERROR")
    subjects_dir = op.dirname(fs_dir)
    src_fname = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
    bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

    fif = (f"{BASE}/{SUBJECT}/ses-meg/meg/"
           f"{SUBJECT}_ses-meg_task-facerecognition_run-{RUN:02d}_meg.fif")
    raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
    raw.pick("mag")
    raw.crop(tmax=CROP)
    raw.filter(LO, HI, verbose="ERROR")
    raw.resample(100.0, verbose="ERROR")

    coreg = mne.coreg.Coregistration(raw.info, "fsaverage", subjects_dir, fiducials="estimated")
    coreg.fit_fiducials(verbose="ERROR")
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose="ERROR")
    fwd = mne.make_forward_solution(raw.info, coreg.trans, src_fname, bem,
                                    meg=True, eeg=False, verbose="ERROR")
    src = fwd["src"]

    data_cov = mne.compute_raw_covariance(raw, verbose="ERROR")
    filters = make_lcmv(raw.info, fwd, data_cov, reg=0.05,
                        pick_ori="max-power", weight_norm="unit-noise-gain", verbose="ERROR")
    stc = apply_lcmv_raw(raw, filters, verbose="ERROR")

    # Left hemisphere: used vertices, positions, decimated triangulation.
    lh = src[0]
    vertno = lh["vertno"]
    rr = lh["rr"][vertno] * 1000.0          # m -> mm
    remap = -np.ones(lh["np"], int)
    remap[vertno] = np.arange(len(vertno))
    faces = remap[lh["use_tris"]]
    faces = faces[(faces >= 0).all(axis=1)]  # keep fully-mapped triangles
    lh_data = stc.data[: len(vertno)].astype(np.float32)  # (n_lh_vertices, n_times)

    np.save(op.join(OUT, "mesh_vertices.npy"), rr.astype(np.float32))
    np.save(op.join(OUT, "mesh_faces.npy"), faces.astype(np.int32))
    np.save(op.join(OUT, "source_alpha.npy"), lh_data)
    print(f"Saved LH mesh: {rr.shape[0]} verts, {faces.shape[0]} faces; "
          f"source {lh_data.shape} (alpha {LO}-{HI} Hz, {CROP:.0f}s @ 100 Hz)", flush=True)


if __name__ == "__main__":
    main()
