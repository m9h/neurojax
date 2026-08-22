#!/usr/bin/env python
"""Source-localize Wakeman-Henson MEG (OpenNeuro ds000117) to the Desikan-68
parcellation and fit a K=12 HMM -- the WH counterpart of `wand_source_prep.py`,
needed to cross-check WH's discrete TINDA cycle (`wh_k12_cycle.py`) against the
same continuous Langevin/EPR + connectome-harmonic + transfer-entropy battery
already run on WAND.

`wh_source_prep_oracle.py` never persisted the raw 68-region parcel time series
(only the TDE-PCA `prepared.npy`), so this is a fresh, self-consistent run:
parcels68 (raw, standardized) + prepared (TDE-PCA) + oracle_gamma/oracle_covariances
all derived from the exact same session list, written to a NEW output dir so the
existing `/data/datasets/wh_src8` cache (basis of the already-published
S=+0.065, z=56.6 result) is left untouched.

Run in the osl/MNE env:
    .venv-oracle/bin/python -u scripts/real_data/wh_source_parcels_prep.py
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

OUT = os.environ.get("WH_OUT", "/data/datasets/wh_src8_full")
BASE = os.environ.get("WH_BASE", "/data/datasets/ds000117-download")
SUBJECTS = os.environ.get(
    "WH_SUBJECTS",
    "sub-01,sub-02,sub-03,sub-04,sub-05,sub-06,sub-07,sub-08",
).split(",")
RUNS = [int(r) for r in os.environ.get("WH_RUNS", "1,2").split(",")]
N_STATES = int(os.environ.get("WH_N_STATES", "12"))


def source_parcels(raw, fwd, src, labels):
    data_cov = mne.compute_raw_covariance(raw, verbose="ERROR")
    filters = make_lcmv(raw.info, fwd, data_cov, reg=0.05,
                        pick_ori="max-power", weight_norm="unit-noise-gain", verbose="ERROR")
    stc = apply_lcmv_raw(raw, filters, verbose="ERROR")
    ltc = mne.extract_label_time_course(stc, labels, src, mode="pca_flip", verbose="ERROR")
    del stc
    return ltc.T.astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    fs_dir = fetch_fsaverage(verbose="ERROR")
    subjects_dir = op.dirname(fs_dir)
    src_fname = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
    bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
    src = mne.read_source_spaces(src_fname, verbose="ERROR")
    labels = [l for l in mne.read_labels_from_annot(
        "fsaverage", "aparc", subjects_dir=subjects_dir, verbose="ERROR")
        if "unknown" not in l.name]
    print(f"fsaverage src + {len(labels)} Desikan labels", flush=True)
    print(f"subjects: {SUBJECTS} | runs: {RUNS}", flush=True)

    sessions = []
    for subj in SUBJECTS:
        try:
            fwd = None
            for run in RUNS:
                fif = (f"{BASE}/{subj}/ses-meg/meg/"
                      f"{subj}_ses-meg_task-facerecognition_run-{run:02d}_meg.fif")
                raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
                raw.pick("mag")
                raw.filter(1.0, 45.0, verbose="ERROR")
                raw.resample(100.0, verbose="ERROR")

                if fwd is None:
                    coreg = mne.coreg.Coregistration(raw.info, "fsaverage", subjects_dir,
                                                     fiducials="estimated")
                    coreg.fit_fiducials(verbose="ERROR")
                    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose="ERROR")
                    fwd = mne.make_forward_solution(raw.info, coreg.trans, src_fname, bem,
                                                    meg=True, eeg=False, verbose="ERROR")
                    print(f"{subj}: coreg+forward done", flush=True)

                parcels = source_parcels(raw, fwd, src, labels)
                sessions.append(parcels)
                print(f"{subj} run-{run:02d}: source parcels {parcels.shape}", flush=True)
        except Exception as e:
            print(f"{subj}: SKIPPED ({type(e).__name__}: {e})", flush=True)

    from osl_dynamics.data import Data

    # Raw parcels (standardized) -- for connectome-harmonic projection + Langevin
    data_raw = Data([s for s in sessions])
    data_raw.prepare({"standardize": {}})
    parcels68 = np.concatenate([np.asarray(data_raw[i]) for i in range(len(sessions))], axis=0)
    np.save(op.join(OUT, "parcels68.npy"), parcels68)

    # TDE-PCA for the HMM
    data = Data([s for s in sessions])
    data.prepare({"tde_pca": {"n_embeddings": 15, "n_pca_components": 40}, "standardize": {}})
    prepared = np.concatenate([np.asarray(data[i]) for i in range(len(sessions))], axis=0)
    np.save(op.join(OUT, "prepared.npy"), prepared)
    print(f"parcels68 {parcels68.shape} | prepared {prepared.shape} from "
          f"{len(sessions)} sessions", flush=True)

    from osl_dynamics.models.hmm import Config, Model
    config = Config(n_states=N_STATES, n_channels=prepared.shape[1], sequence_length=200,
                    learn_means=False, learn_covariances=True, learn_trans_prob=True,
                    batch_size=32, learning_rate=0.01, n_epochs=40)
    model = Model(config)
    model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)
    print(f"fitting K={N_STATES} HMM on WH source MEG ...", flush=True)
    model.fit(data)
    gamma = model.get_alpha(data)
    gamma = np.concatenate([np.asarray(g) for g in gamma]) if isinstance(gamma, list) else np.asarray(gamma)
    np.save(op.join(OUT, "oracle_gamma.npy"), gamma)
    np.save(op.join(OUT, "oracle_covariances.npy"), np.asarray(model.get_covariances()))
    print(f"HMM done: gamma {gamma.shape} | occ {np.round(np.sort(gamma.mean(0))[::-1], 3)}",
          flush=True)


if __name__ == "__main__":
    main()
