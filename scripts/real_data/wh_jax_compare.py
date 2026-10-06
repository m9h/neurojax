#!/usr/bin/env python
"""Run the JAX dynamics models on real (multi-run) Wakeman-Henson MEG and compare
the JAX HMM to the osl-dynamics oracle.

Headline metric is the **state network-map (covariance) correlation** — the
standard osl-dynamics way to compare HMM runs — matched state-to-state, plus the
resulting per-timepoint segmentation agreement.

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


def _vec_upper(covs):
    iu = np.triu_indices(covs.shape[-1])
    return np.stack([c[iu] for c in covs])


def _match(sim):
    """Greedy global-max matching of rows->cols of a similarity matrix."""
    k = sim.shape[0]
    match, taken = -np.ones(k, int), set()
    order = np.dstack(np.unravel_index(np.argsort(-sim, axis=None), sim.shape))[0]
    for i, j in order:
        if match[i] == -1 and j not in taken:
            match[i] = j
            taken.add(j)
    return match


def _seg_acc(a, b, k):
    conf = np.zeros((k, k))
    for i in range(k):
        bi = b[a == i]
        for j in range(k):
            conf[i, j] = np.sum(bi == j)
    m = _match(conf)
    return sum(conf[i, m[i]] for i in range(k)) / len(a)


def main():
    print("JAX backend:", jax.default_backend(), "| device:", jax.devices()[0])
    X = np.load(os.path.join(OUT, "prepared.npy")).astype(np.float32)
    print(f"Real prepared multi-run MEG: {X.shape}")
    data = [jnp.asarray(X)]

    hmm = GaussianHMM(config=HMMConfig(n_states=N, n_channels=X.shape[1], learn_means=False))
    hmm.fit(data, n_epochs=40, n_init=3, standardize=False)
    jax_covs = np.asarray(hmm.covariances)            # (N, C, C)
    jax_states = np.asarray(hmm.decode(data)[0])
    np.save(os.path.join(OUT, "jax_hmm_covariances.npy"), jax_covs)
    np.save(os.path.join(OUT, "jax_hmm_states.npy"), jax_states)

    oracle_covs = np.load(os.path.join(OUT, "oracle_covariances.npy"))
    oracle_gamma = np.load(os.path.join(OUT, "oracle_gamma.npy"))
    oracle_states = oracle_gamma.argmax(1)

    # --- Network-map (covariance) similarity, state-matched ---
    jv, ov = _vec_upper(jax_covs), _vec_upper(oracle_covs)
    sim = np.corrcoef(np.vstack([ov, jv]))[:N, N:]    # oracle x jax
    match = _match(sim)                               # oracle i -> jax match[i]
    net_corr = np.array([sim[i, match[i]] for i in range(N)])

    # --- Segmentation agreement under the covariance-based matching ---
    n = min(len(jax_states), len(oracle_states))
    inv = np.argsort(match)                            # jax label -> oracle label
    seg = np.mean(inv[jax_states[:n]] == oracle_states[:n])

    print("\n==========  JAX HMM vs osl-dynamics oracle — REAL multi-run MEG  ==========")
    print(f"  state network-map correlation (matched): mean {net_corr.mean():.3f}")
    print(f"      per-state: {np.round(np.sort(net_corr)[::-1], 3)}")
    print(f"  segmentation agreement (cov-matched)   : {seg:.3f}")
    print(f"  JAX    occupancy: {np.round(np.sort(np.bincount(jax_states, minlength=N)/len(jax_states)), 3)}")
    print(f"  oracle occupancy: {np.round(np.sort(oracle_gamma.mean(0)), 3)}")

    # --- DyNeMo + M-DyNeMo on the same real data ---
    dyn = DyNeMo(n_modes=N, n_channels=X.shape[1])
    h_dyn = dyn.fit(data, n_epochs=30)
    md = MDyNeMo(n_modes=N, n_corr_modes=N, n_channels=X.shape[1])
    h_md = md.fit(data, n_epochs=30)
    a_md, g_md = md.infer(data)[0]
    corr_ag = np.corrcoef(np.asarray(a_md)[:, 0], np.asarray(g_md)[:, 0])[0, 1]
    print(f"\n  DyNeMo   : ELBO {h_dyn[0]['loss']:.2f} -> {h_dyn[-1]['loss']:.2f}")
    print(f"  M-DyNeMo : ELBO {h_md[0]['loss']:.2f} -> {h_md[-1]['loss']:.2f} | "
          f"power/FC corr={corr_ag:.2f} (decoupled)")


if __name__ == "__main__":
    main()
