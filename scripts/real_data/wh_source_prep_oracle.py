#!/usr/bin/env python
"""Source-space, MULTI-SUBJECT Wakeman-Henson prep + osl-dynamics HMM oracle.

Proper osl-dynamics-style analysis: MEG source reconstruction (fsaverage
template coregistration -> LCMV beamformer -> cortical parcellation), across
several subjects concatenated as group sessions, then osl-dynamics TDE-PCA prep
and HMM.  Saves prepared data + oracle gamma + state network covariances for the
JAX comparison (wh_jax_compare.py).

Run (unbuffered, osl env):
    .venv-oracle/bin/python -u scripts/real_data/wh_source_prep_oracle.py
"""

import os
import os.path as op
import sys
import types

fsl = types.ModuleType("fsl")
fsl.wrappers = types.ModuleType("fsl.wrappers")
sys.modules["fsl"] = fsl
sys.modules["fsl.wrappers"] = fsl.wrappers

import mne
import numpy as np
from mne.beamformer import apply_lcmv_raw, make_lcmv
from mne.datasets import fetch_fsaverage

OUT = os.environ.get("WH_OUT", "/data/datasets/wh_src")
BASE = os.environ.get("WH_BASE", "/data/datasets/ds000117-download")
SUBJECTS = os.environ.get("WH_SUBJECTS", "sub-05,sub-06,sub-07").split(",")
RUNS = [int(r) for r in os.environ.get("WH_RUNS", "1,2").split(",")]
N_STATES = int(os.environ.get("WH_N_STATES", "6"))


def source_parcels(raw, fwd, src, labels):
    """LCMV beamformer -> parcellated label time course (n_times, n_labels)."""
    data_cov = mne.compute_raw_covariance(raw, verbose="ERROR")
    filters = make_lcmv(
        raw.info, fwd, data_cov, reg=0.05,
        pick_ori="max-power", weight_norm="unit-noise-gain", verbose="ERROR",
    )
    stc = apply_lcmv_raw(raw, filters, verbose="ERROR")
    ltc = mne.extract_label_time_course(
        stc, labels, src, mode="pca_flip", verbose="ERROR"
    )  # (n_labels, n_times)
    del stc
    return ltc.T.astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    fs_dir = fetch_fsaverage(verbose="ERROR")
    subjects_dir = op.dirname(fs_dir)
    src_fname = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
    bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
    src = mne.read_source_spaces(src_fname, verbose="ERROR")
    labels = [
        l for l in mne.read_labels_from_annot(
            "fsaverage", "aparc", subjects_dir=subjects_dir, verbose="ERROR"
        )
        if "unknown" not in l.name
    ]
    print(f"fsaverage: {len(src)} src spaces, {len(labels)} aparc labels", flush=True)

    sessions = []
    for subj in SUBJECTS:
        fwd = None
        for run in RUNS:
            fif = (
                f"{BASE}/{subj}/ses-meg/meg/"
                f"{subj}_ses-meg_task-facerecognition_run-{run:02d}_meg.fif"
            )
            raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
            raw.pick("mag")
            raw.filter(1.0, 45.0, verbose="ERROR")
            raw.resample(100.0, verbose="ERROR")  # 100 Hz keeps source recon tractable

            if fwd is None:  # coreg + forward once per subject (head pos constant)
                coreg = mne.coreg.Coregistration(
                    raw.info, "fsaverage", subjects_dir, fiducials="estimated"
                )
                coreg.fit_fiducials(verbose="ERROR")
                coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose="ERROR")
                fwd = mne.make_forward_solution(
                    raw.info, coreg.trans, src_fname, bem,
                    meg=True, eeg=False, verbose="ERROR",
                )
                print(f"{subj}: coreg+forward done", flush=True)

            parcels = source_parcels(raw, fwd, src, labels)
            sessions.append(parcels)
            print(f"{subj} run-{run:02d}: source parcels {parcels.shape}", flush=True)

    from osl_dynamics.data import Data

    data = Data(sessions)
    data.prepare(
        {"tde_pca": {"n_embeddings": 15, "n_pca_components": 40}, "standardize": {}}
    )
    prepared = np.concatenate([np.asarray(data[i]) for i in range(len(sessions))], axis=0)
    np.save(op.join(OUT, "prepared.npy"), prepared)
    print(f"Prepared (source TDE-PCA): {prepared.shape} "
          f"from {len(SUBJECTS)} subjects x {len(RUNS)} runs", flush=True)

    from osl_dynamics.models.hmm import Config, Model

    config = Config(
        n_states=N_STATES, n_channels=prepared.shape[1], sequence_length=200,
        learn_means=False, learn_covariances=True, learn_trans_prob=True,
        batch_size=32, learning_rate=0.01, n_epochs=40,
    )
    model = Model(config)
    model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)
    print("Fitting osl-dynamics HMM oracle on real source-space MEG ...", flush=True)
    model.fit(data)

    gamma = model.get_alpha(data)
    gamma = np.concatenate([np.asarray(g) for g in gamma]) if isinstance(gamma, list) else np.asarray(gamma)
    np.save(op.join(OUT, "oracle_gamma.npy"), gamma)
    np.save(op.join(OUT, "oracle_covariances.npy"), np.asarray(model.get_covariances()))
    print(f"Oracle done: gamma {gamma.shape}", flush=True)
    print(f"Oracle fractional occupancy: {np.round(gamma.mean(0), 3)}", flush=True)


if __name__ == "__main__":
    main()
