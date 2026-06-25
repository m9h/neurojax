#!/usr/bin/env python
"""Fit a K=12 HMM on the real WH source-space data and run TINDA — to obtain a
non-trivial structured network cycle (the paper used K=12).

Refits on the already-prepared (source TDE-PCA) data from
wh_source_prep_oracle.py.  Run in the osl env:
    .venv-oracle/bin/python -u scripts/real_data/wh_k12_cycle.py
"""

import os
import sys
import types

fsl = types.ModuleType("fsl")
fsl.wrappers = types.ModuleType("fsl.wrappers")
sys.modules["fsl"] = fsl
sys.modules["fsl.wrappers"] = fsl.wrappers

import numpy as np
from osl_dynamics.analysis import tinda as T

OUT = os.environ.get("WH_OUT", "/data/datasets/wh_src")
K = int(os.environ.get("WH_N_STATES", "12"))


def cycle_strength(states, k):
    oh = np.zeros((len(states), k), float)
    oh[np.arange(len(states)), states] = 1.0
    fo, _, _ = T.tinda(oh)
    order = list(np.asarray(T.optimise_sequence(fo)).ravel())
    angles = T.circle_angles(order)
    asym = np.nanmean(fo[:, :, 0, :] - fo[:, :, 1, :], axis=-1)
    return order, float(np.nanmean(T.compute_cycle_strength(angles, asym)))


def main():
    X = np.load(os.path.join(OUT, "prepared.npy")).astype(np.float32)
    print(f"Prepared source data: {X.shape}", flush=True)

    from osl_dynamics.data import Data
    from osl_dynamics.models.hmm import Config, Model

    data = Data([X])
    config = Config(
        n_states=K, n_channels=X.shape[1], sequence_length=200,
        learn_means=False, learn_covariances=True, learn_trans_prob=True,
        batch_size=32, learning_rate=0.01, n_epochs=40,
    )
    model = Model(config)
    model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)
    print(f"Fitting K={K} HMM ...", flush=True)
    model.fit(data)

    gamma = model.get_alpha(data)
    gamma = np.concatenate([np.asarray(g) for g in gamma]) if isinstance(gamma, list) else np.asarray(gamma)
    np.save(os.path.join(OUT, f"oracle_gamma_k{K}.npy"), gamma)
    states = gamma.argmax(1)
    order, S = cycle_strength(states, K)
    print(f"\n=========  TINDA cycle on real WH source MEG, K={K}  =========", flush=True)
    print(f"  fractional occupancy: {np.round(np.sort(gamma.mean(0))[::-1], 3)}", flush=True)
    print(f"  cycle order : {order}", flush=True)
    print(f"  cycle strength S = {S:+.4f}  (S>0 = consistent directional cycle)", flush=True)


if __name__ == "__main__":
    main()
