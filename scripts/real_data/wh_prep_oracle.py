#!/usr/bin/env python
"""Prepare real Wakeman-Henson MEG (Woolrich osl toolbox-paper data) and fit the
osl-dynamics HMM oracle.

Sensor-level dynamics (à la Gohil/Woolrich 2025 "Canonical HMM"): pick
magnetometers, band-pass, downsample, then osl-dynamics TDE-PCA + standardize —
the exact preparation osl-dynamics uses.  The prepared array is saved so the
JAX models (wh_jax_compare.py) consume *identical* data, and the osl-dynamics
HMM is fit as the comparison oracle.

Run in the osl-dynamics env:
    .venv-oracle/bin/python scripts/real_data/wh_prep_oracle.py
"""

import os
import sys
import types

# Stub FSL (not needed for sensor-level HMM).
fsl = types.ModuleType("fsl")
fsl.wrappers = types.ModuleType("fsl.wrappers")
sys.modules["fsl"] = fsl
sys.modules["fsl.wrappers"] = fsl.wrappers

import mne
import numpy as np

OUT = os.environ.get("WH_OUT", "/data/datasets/wh_real")
FIF = os.environ.get(
    "WH_FIF",
    "/data/datasets/ds000117-download/sub-05/ses-meg/meg/"
    "sub-05_ses-meg_task-facerecognition_run-01_meg.fif",
)
N_STATES = int(os.environ.get("WH_N_STATES", "6"))


def main():
    os.makedirs(OUT, exist_ok=True)

    # --- Preprocess real MEG ---
    raw = mne.io.read_raw_fif(FIF, preload=True, verbose="ERROR")
    raw.pick("mag")                      # 102 magnetometers
    raw.filter(1.0, 45.0, verbose="ERROR")
    raw.resample(250.0, verbose="ERROR")  # osl-dynamics-typical sampling
    X = raw.get_data().T.astype(np.float32)  # (n_times, 102)
    print(f"Raw MEG prepared: {X.shape} @ 250 Hz")

    # --- osl-dynamics preparation: TDE-PCA + standardize ---
    from osl_dynamics.data import Data

    data = Data([X])
    data.prepare(
        {
            "tde_pca": {"n_embeddings": 15, "n_pca_components": 40},
            "standardize": {},
        }
    )
    prepared = np.asarray(data[0])
    np.save(os.path.join(OUT, "prepared.npy"), prepared)
    print(f"Prepared (TDE-PCA) data: {prepared.shape} -> saved")

    # --- osl-dynamics HMM oracle (covariance-only, the Vidaurre/Woolrich setup) ---
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
    print("Fitting osl-dynamics HMM oracle on real MEG ...")
    model.fit(data)

    gamma = model.get_alpha(data)
    if isinstance(gamma, list):
        gamma = gamma[0]
    np.save(os.path.join(OUT, "oracle_gamma.npy"), np.asarray(gamma))
    covs = model.get_covariances()
    np.save(os.path.join(OUT, "oracle_covariances.npy"), np.asarray(covs))
    np.save(os.path.join(OUT, "oracle_trans_prob.npy"), np.asarray(model.get_trans_prob()))
    print(f"Oracle done: gamma {np.asarray(gamma).shape}, covs {np.asarray(covs).shape}")
    print(f"Oracle fractional occupancy: {np.round(np.asarray(gamma).mean(0), 3)}")


if __name__ == "__main__":
    main()
