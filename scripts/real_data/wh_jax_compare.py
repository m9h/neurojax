#!/usr/bin/env python
"""Run the JAX dynamics models on real Wakeman-Henson MEG and compare the JAX
HMM to the osl-dynamics oracle (state segmentation on identical prepared data).

Run in the JAX/GPU env (after wh_prep_oracle.py):
    PYTHONPATH=src .venv-models/bin/python scripts/real_data/wh_jax_compare.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from neurojax.models import DyNeMo, GaussianHMM, HMMConfig, MDyNeMo

OUT = os.environ.get("WH_OUT", "/data/datasets/wh_real")
N = int(os.environ.get("WH_N_STATES", "6"))


def greedy_label_accuracy(a, b, k):
    """Best-permutation timepoint agreement between two label sequences."""
    conf = np.zeros((k, k))
    for i in range(k):
        bi = b[a == i]
        for j in range(k):
            conf[i, j] = np.sum(bi == j)
    taken, used, total = set(), set(), 0.0
    order = np.dstack(np.unravel_index(np.argsort(-conf, axis=None), conf.shape))[0]
    for i, j in order:
        if i not in used and j not in taken:
            used.add(i)
            taken.add(j)
            total += conf[i, j]
    return total / len(a)


def main():
    print("JAX backend:", jax.default_backend(), "| device:", jax.devices()[0])
    X = np.load(os.path.join(OUT, "prepared.npy")).astype(np.float32)
    print(f"Real prepared MEG: {X.shape}")
    data = [jnp.asarray(X)]

    # --- JAX HMM (covariance-only, matching the oracle) ---
    hmm = GaussianHMM(config=HMMConfig(n_states=N, n_channels=X.shape[1], learn_means=False))
    hmm.fit(data, n_epochs=40, n_init=3, standardize=False)
    jax_states = np.asarray(hmm.decode(data)[0])
    np.save(os.path.join(OUT, "jax_hmm_states.npy"), jax_states)

    # --- Oracle comparison (align lengths: oracle gamma is trimmed to a whole
    #     number of sequence_length blocks, the JAX decode is full-length) ---
    oracle_gamma = np.load(os.path.join(OUT, "oracle_gamma.npy"))
    oracle_states = oracle_gamma.argmax(1)
    n = min(len(jax_states), len(oracle_states))
    jax_states, oracle_states = jax_states[:n], oracle_states[:n]
    jax_fo = np.bincount(jax_states, minlength=N) / len(jax_states)
    acc = greedy_label_accuracy(oracle_states, jax_states, N)
    print("\n================  JAX HMM vs osl-dynamics oracle (real MEG)  ================")
    print(f"  state-segmentation agreement : {acc:.3f}")
    print(f"  JAX    fractional occupancy  : {np.round(np.sort(jax_fo), 3)}")
    print(f"  oracle fractional occupancy  : {np.round(np.sort(oracle_gamma.mean(0)), 3)}")

    # --- JAX DyNeMo + M-DyNeMo also run on the same real data ---
    dyn = DyNeMo(n_modes=N, n_channels=X.shape[1])
    h_dyn = dyn.fit(data, n_epochs=30)
    alpha = np.asarray(dyn.infer(data)[0])
    print("\n  DyNeMo   : loss %.3f -> %.3f | mean alpha occ %s"
          % (h_dyn[0]["loss"], h_dyn[-1]["loss"], np.round(alpha.mean(0), 3)))

    md = MDyNeMo(n_modes=N, n_corr_modes=N, n_channels=X.shape[1])
    h_md = md.fit(data, n_epochs=30)
    a_md, g_md = md.infer(data)[0]
    corr_ag = np.corrcoef(np.asarray(a_md)[:, 0], np.asarray(g_md)[:, 0])[0, 1]
    print("  M-DyNeMo : loss %.3f -> %.3f | power/FC time-course corr=%.2f (decoupled)"
          % (h_md[0]["loss"], h_md[-1]["loss"], corr_ag))


if __name__ == "__main__":
    main()
