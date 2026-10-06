#!/usr/bin/env python
"""K=12 HMM + TINDA structured network cycle on real WH source-space MEG, with a
block-shuffle significance null.  Reproduces Woolrich's "structured cycles" (van
Es et al. 2025) on our data.

Uses an existing oracle_gamma.npy if it already has K states (e.g. produced by
wh_source_prep_oracle.py with WH_N_STATES=12); otherwise refits the HMM on the
prepared source data.

    WH_OUT=/data/datasets/wh_src8 .venv-oracle/bin/python -u scripts/real_data/wh_k12_cycle.py
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

OUT = os.environ.get("WH_OUT", "/data/datasets/wh_src8")
K = int(os.environ.get("WH_N_STATES", "12"))


def cycle_S(seq, k):
    oh = np.zeros((len(seq), k), float)
    oh[np.arange(len(seq)), seq] = 1.0
    fo, _, _ = T.tinda(oh)
    order = np.asarray(T.optimise_sequence(fo)).ravel()
    asym = np.nanmean(fo[:, :, 0, :] - fo[:, :, 1, :], axis=-1)
    S = float(np.nanmean(T.compute_cycle_strength(T.circle_angles(list(order)), asym)))
    return list(order), S


def get_states():
    g_path = os.path.join(OUT, "oracle_gamma.npy")
    if os.path.exists(g_path):
        g = np.load(g_path)
        if g.shape[1] == K:
            return g.argmax(1)
    # else refit
    from osl_dynamics.data import Data
    from osl_dynamics.models.hmm import Config, Model

    X = np.load(os.path.join(OUT, "prepared.npy")).astype(np.float32)
    data = Data([X])
    config = Config(n_states=K, n_channels=X.shape[1], sequence_length=200,
                    learn_means=False, learn_covariances=True, learn_trans_prob=True,
                    batch_size=32, learning_rate=0.01, n_epochs=40)
    model = Model(config)
    model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)
    model.fit(data)
    g = model.get_alpha(data)
    g = np.concatenate([np.asarray(a) for a in g]) if isinstance(g, list) else np.asarray(g)
    return g.argmax(1)


def main():
    st = get_states()
    order, real = cycle_S(st, K)

    # Block-shuffle null: permute ~1000-sample blocks (kills long-range ordering).
    rng = np.random.default_rng(0)
    bs = 1000
    nb = len(st) // bs
    nulls = []
    for _ in range(20):
        perm = rng.permutation(nb)
        nulls.append(cycle_S(st[: nb * bs].reshape(nb, bs)[perm].ravel(), K)[1])
    nulls = np.array(nulls)
    z = (real - nulls.mean()) / (nulls.std() + 1e-9)

    print(f"\n=====  TINDA structured cycle on real WH source MEG (K={K})  =====")
    print(f"  samples {len(st)} | cycle order {order}")
    print(f"  cycle strength S = {real:+.4f}")
    print(f"  block-shuffle null: {nulls.mean():+.4f} +- {nulls.std():.4f}")
    print(f"  z vs null = {z:.1f}  -> {'SIGNIFICANT cycle' if z > 3 else 'not significant'}")


if __name__ == "__main__":
    main()
