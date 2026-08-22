#!/usr/bin/env python
"""Prepare real Wakeman-Henson MEG (Woolrich osl toolbox-paper data) and fit the
osl-dynamics HMM oracle — MULTI-RUN for a stable, non-degenerate HMM.

Concatenates several runs (default all 6 of one subject) as osl-dynamics
sessions, prepares them identically (sensor-level TDE-PCA + standardize), and
fits the osl-dynamics HMM.  Saves the prepared data + oracle gamma + oracle
state covariances (the "network maps") for comparison in wh_jax_compare.py.

Run in the osl-dynamics env:
    .venv-oracle/bin/python scripts/real_data/wh_prep_oracle.py
"""

import os
import sys
import types

fsl = types.ModuleType("fsl")
fsl.wrappers = types.ModuleType("fsl.wrappers")
sys.modules["fsl"] = fsl
sys.modules["fsl.wrappers"] = fsl.wrappers

import mne
import numpy as np

OUT = os.environ.get("WH_OUT", "/data/datasets/wh_real")
BASE = os.environ.get("WH_BASE", "/data/datasets/ds000117-download")
SUBJECT = os.environ.get("WH_SUBJECT", "sub-05")
RUNS = [int(r) for r in os.environ.get("WH_RUNS", "1,2,3,4,5,6").split(",")]
N_STATES = int(os.environ.get("WH_N_STATES", "6"))


def main():
    os.makedirs(OUT, exist_ok=True)

    arrays = []
    for run in RUNS:
        fif = (
            f"{BASE}/{SUBJECT}/ses-meg/meg/"
            f"{SUBJECT}_ses-meg_task-facerecognition_run-{run:02d}_meg.fif"
        )
        raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
        raw.pick("mag")
        raw.filter(1.0, 45.0, verbose="ERROR")
        raw.resample(250.0, verbose="ERROR")
        arrays.append(raw.get_data().T.astype(np.float32))
        print(f"  {SUBJECT} run-{run:02d}: {arrays[-1].shape}")
    print(f"Loaded {len(arrays)} runs, {sum(a.shape[0] for a in arrays)} total samples")

    from osl_dynamics.data import Data

    data = Data(arrays)
    data.prepare(
        {"tde_pca": {"n_embeddings": 15, "n_pca_components": 40}, "standardize": {}}
    )
    sessions = [np.asarray(data[i]) for i in range(len(arrays))]
    prepared = np.concatenate(sessions, axis=0)
    np.save(os.path.join(OUT, "prepared.npy"), prepared)
    np.save(os.path.join(OUT, "session_lengths.npy"), np.array([s.shape[0] for s in sessions]))
    print(f"Prepared (TDE-PCA) data: {prepared.shape} across {len(sessions)} sessions")

    from osl_dynamics.models.hmm import Config, Model

    config = Config(
        n_states=N_STATES,
        n_channels=prepared.shape[1],
        sequence_length=200,
        learn_means=False,
        learn_covariances=True,
        learn_trans_prob=True,
        batch_size=32,
        learning_rate=0.01,
        n_epochs=40,
    )
    model = Model(config)
    model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)
    print("Fitting osl-dynamics HMM oracle on real multi-run MEG ...")
    model.fit(data)

    gamma = model.get_alpha(data)
    gamma = np.concatenate([np.asarray(g) for g in gamma]) if isinstance(gamma, list) else np.asarray(gamma)
    np.save(os.path.join(OUT, "oracle_gamma.npy"), gamma)
    np.save(os.path.join(OUT, "oracle_covariances.npy"), np.asarray(model.get_covariances()))
    np.save(os.path.join(OUT, "oracle_trans_prob.npy"), np.asarray(model.get_trans_prob()))
    print(f"Oracle done: gamma {gamma.shape}")
    print(f"Oracle fractional occupancy: {np.round(gamma.mean(0), 3)}")


if __name__ == "__main__":
    main()
