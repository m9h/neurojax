#!/usr/bin/env python
"""Fit the osl-dynamics DyNeMo oracle on the already-prepared real WH MEG, so the
JAX DyNeMo can be compared against it (wh_dynemo_compare.py).

Consumes WH_OUT/prepared.npy produced by wh_prep_oracle.py.

Run in the osl env:
    .venv-oracle/bin/python -u scripts/real_data/wh_dynemo_oracle.py
"""

import os
import sys
import types

fsl = types.ModuleType("fsl")
fsl.wrappers = types.ModuleType("fsl.wrappers")
sys.modules["fsl"] = fsl
sys.modules["fsl.wrappers"] = fsl.wrappers

import numpy as np

OUT = os.environ.get("WH_OUT", "/data/datasets/wh_real")
N = int(os.environ.get("WH_N_MODES", "6"))


def main():
    X = np.load(os.path.join(OUT, "prepared.npy")).astype(np.float32)
    print(f"Prepared data: {X.shape}", flush=True)

    from osl_dynamics.data import Data
    from osl_dynamics.models.dynemo import Config, Model

    data = Data([X])
    config = Config(
        n_modes=N,
        n_channels=X.shape[1],
        sequence_length=200,
        inference_n_units=64,
        inference_normalization="layer",
        model_n_units=64,
        model_normalization="layer",
        learn_means=False,
        learn_covariances=True,
        do_kl_annealing=True,
        kl_annealing_curve="tanh",
        kl_annealing_sharpness=10,
        n_kl_annealing_epochs=10,
        batch_size=32,
        learning_rate=0.01,
        n_epochs=40,
        init_method="random_subset",
        n_init=3,
        n_init_epochs=1,
        init_take=0.25,
    )
    model = Model(config)
    print("Fitting osl-dynamics DyNeMo oracle ...", flush=True)
    model.fit(data)

    covs = np.asarray(model.get_covariances())          # (n_modes, C, C)
    alpha = model.get_alpha(data)
    alpha = np.concatenate([np.asarray(a) for a in alpha]) if isinstance(alpha, list) else np.asarray(alpha)
    np.save(os.path.join(OUT, "oracle_dynemo_covariances.npy"), covs)
    np.save(os.path.join(OUT, "oracle_dynemo_alpha.npy"), alpha)
    print(f"Oracle DyNeMo done: covs {covs.shape}, alpha {alpha.shape}", flush=True)
    print(f"Mean mode activation: {np.round(alpha.mean(0), 3)}", flush=True)


if __name__ == "__main__":
    main()
