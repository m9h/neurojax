#!/usr/bin/env python
"""Compare the JAX DyNeMo to the osl-dynamics DyNeMo oracle on real WH MEG.

DyNeMo modes are soft/overlapping, so (as for the HMM) we match modes by
network-map (covariance) similarity and report the matched correlation — the
identifiable, standard osl-dynamics comparison.

Run in the JAX/GPU env (after wh_dynemo_oracle.py):
    WH_OUT=/data/datasets/wh_real PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wh_dynemo_compare.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from neurojax.models import DyNeMo, DyNeMoConfig

OUT = os.environ.get("WH_OUT", "/data/datasets/wh_real")
N = int(os.environ.get("WH_N_MODES", "6"))


def _vec_upper(covs):
    iu = np.triu_indices(covs.shape[-1])
    return np.stack([c[iu] for c in covs])


def _match(sim):
    k = sim.shape[0]
    match, taken = -np.ones(k, int), set()
    for i, j in np.dstack(np.unravel_index(np.argsort(-sim, axis=None), sim.shape))[0]:
        if match[i] == -1 and j not in taken:
            match[i] = j
            taken.add(j)
    return match


def main():
    print("JAX backend:", jax.default_backend(), "| device:", jax.devices()[0])
    X = np.load(os.path.join(OUT, "prepared.npy")).astype(np.float32)
    print(f"Real prepared MEG: {X.shape}")
    data = [jnp.asarray(X)]

    dyn = DyNeMo(config=DyNeMoConfig(
        n_modes=N, n_channels=X.shape[1], learn_means=False, n_epochs=40,
    ))
    h = dyn.fit(data, n_epochs=40)
    jax_covs = np.asarray(dyn.get_covariances())          # (N, C, C)
    jax_alpha = np.asarray(dyn.infer(data)[0])

    oracle_covs = np.load(os.path.join(OUT, "oracle_dynemo_covariances.npy"))
    oracle_alpha = np.load(os.path.join(OUT, "oracle_dynemo_alpha.npy"))

    jv, ov = _vec_upper(jax_covs), _vec_upper(oracle_covs)
    sim = np.corrcoef(np.vstack([ov, jv]))[:N, N:]        # oracle x jax
    match = _match(sim)
    net_corr = np.array([sim[i, match[i]] for i in range(N)])

    print("\n========  JAX DyNeMo vs osl-dynamics DyNeMo oracle — REAL MEG  ========")
    print(f"  ELBO: {h[0]['loss']:.2f} -> {h[-1]['loss']:.2f}")
    print(f"  mode network-map correlation (matched): mean {net_corr.mean():.3f}")
    print(f"      per-mode: {np.round(np.sort(net_corr)[::-1], 3)}")
    print(f"  JAX    mean mode activation: {np.round(np.sort(jax_alpha.mean(0))[::-1], 3)}")
    print(f"  oracle mean mode activation: {np.round(np.sort(oracle_alpha.mean(0))[::-1], 3)}")


if __name__ == "__main__":
    main()
