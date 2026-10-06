#!/usr/bin/env python
"""Source-localize WAND resting-state CTF MEG (10 subjects) to the Desikan-68
parcellation, then fit a K=12 HMM — the Leg-A/C input for the three-way
cross-test (TINDA cycle + CEBRA/DYSCO governing equation + waves).

fsaverage template coregistration (only sub-08033 has individual FreeSurfer).
Saves: parcels68 (concatenated, standardized — for CEBRA/DYSCO), prepared
(TDE-PCA — for the HMM), oracle_gamma (K=12), oracle_covariances.

Run in the osl/MNE env:
    .venv-oracle/bin/python -u scripts/real_data/wand_source_prep.py
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

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
BASE = os.environ.get("WAND_BASE", "/data/raw/wand")
N_SUBJECTS = int(os.environ.get("WAND_N", "10"))
CROP = float(os.environ.get("WAND_CROP", "360"))   # s (6 min keeps it tractable)
N_STATES = int(os.environ.get("WAND_N_STATES", "12"))


def resting_subjects(n):
    import glob
    fs = sorted(glob.glob(f"{BASE}/sub-*/ses-01/meg/sub-*_ses-01_task-resting.ds"))
    subs = [op.basename(f).split("_")[0] for f in fs]
    return subs[:n]


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

    subjects = resting_subjects(N_SUBJECTS)
    print(f"subjects: {subjects}", flush=True)

    sessions = []
    for subj in subjects:
        try:
            ds = f"{BASE}/{subj}/ses-01/meg/{subj}_ses-01_task-resting.ds"
            raw = mne.io.read_raw_ctf(ds, preload=True, verbose="ERROR")
            raw.apply_gradient_compensation(3)          # CTF noise reduction
            raw.pick("mag")
            raw.crop(tmax=min(CROP, raw.times[-1]))
            raw.filter(1.0, 45.0, verbose="ERROR")
            raw.resample(250.0, verbose="ERROR")
            coreg = mne.coreg.Coregistration(raw.info, "fsaverage", subjects_dir, fiducials="estimated")
            coreg.fit_fiducials(verbose="ERROR")
            coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose="ERROR")
            fwd = mne.make_forward_solution(raw.info, coreg.trans, src_fname, bem,
                                            meg=True, eeg=False, verbose="ERROR")
            parcels = source_parcels(raw, fwd, src, labels)
            sessions.append(parcels)
            print(f"{subj}: source parcels {parcels.shape}", flush=True)
        except Exception as e:
            print(f"{subj}: SKIPPED ({type(e).__name__}: {e})", flush=True)

    from osl_dynamics.data import Data

    # Raw parcels (standardized) for CEBRA/DYSCO
    data_raw = Data([s for s in sessions])
    data_raw.prepare({"standardize": {}})
    parcels68 = np.concatenate([np.asarray(data_raw[i]) for i in range(len(sessions))], axis=0)
    np.save(op.join(OUT, "parcels68.npy"), parcels68)

    # TDE-PCA for the HMM
    data = Data([s for s in sessions])
    data.prepare({"tde_pca": {"n_embeddings": 15, "n_pca_components": 40}, "standardize": {}})
    prepared = np.concatenate([np.asarray(data[i]) for i in range(len(sessions))], axis=0)
    np.save(op.join(OUT, "prepared.npy"), prepared)
    print(f"parcels68 {parcels68.shape} | prepared {prepared.shape} from {len(sessions)} subjects", flush=True)

    from osl_dynamics.models.hmm import Config, Model
    config = Config(n_states=N_STATES, n_channels=prepared.shape[1], sequence_length=200,
                    learn_means=False, learn_covariances=True, learn_trans_prob=True,
                    batch_size=32, learning_rate=0.01, n_epochs=40)
    model = Model(config)
    model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)
    print(f"fitting K={N_STATES} HMM on WAND resting source MEG ...", flush=True)
    model.fit(data)
    gamma = model.get_alpha(data)
    gamma = np.concatenate([np.asarray(g) for g in gamma]) if isinstance(gamma, list) else np.asarray(gamma)
    np.save(op.join(OUT, "oracle_gamma.npy"), gamma)
    np.save(op.join(OUT, "oracle_covariances.npy"), np.asarray(model.get_covariances()))
    print(f"HMM done: gamma {gamma.shape} | occ {np.round(np.sort(gamma.mean(0))[::-1],3)}", flush=True)


if __name__ == "__main__":
    main()
