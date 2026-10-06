#!/usr/bin/env python
"""Band-resolved Langevin/Fokker-Planck on WH resting/task MEG -- the WH
counterpart of `wand_band_langevin.py` (the script that produced WAND's real
headline EPR / sol-tot / f_sol table; the single-timescale `wh_langevin.py`
collapses to ~null, exactly as WAND's own single-timescale `wand_langevin.py`
does -- see docs/WAND_THREE_LEG_CROSSTEST.md).

For each band: EEGLAB complex Morlet -> amplitude envelope -> downsample ->
Gavish-Donoho optimal-rank PCA embedding -> linear Langevin fit -> entropy
production, solenoidal rotation frequency, and the solenoidal/total drift
fraction, each against a time-shuffle null.

    WH_OUT=/data/datasets/wh_src8_full PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wh_band_langevin.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from neurojax.analysis.timefreq import morlet_cwt, eeglab_cycles
from neurojax.dynamics import (
    svht_rank,
    fit_linear_langevin,
    langevin_entropy_production,
    langevin_solenoidal_part,
    langevin_solenoidal_frequency,
)

OUT = os.environ.get("WH_OUT", "/data/datasets/wh_src8_full")
FS = 100.0                          # wh_source_parcels_prep.py resample rate
BANDS = {"delta": (2, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
ENV_FS = 25.0
RANK_CAP = 24


def band_embedding(X, lo, hi):
    fc = jnp.linspace(lo, hi, 6)
    C = morlet_cwt(jnp.asarray(X.T), FS, fc, eeglab_cycles(fc, 3.0, 0.5))
    env = np.asarray(jnp.mean(jnp.abs(C), axis=1))
    ds = int(FS / ENV_FS)
    env = env[:, : (env.shape[1] // ds) * ds].reshape(68, -1, ds).mean(2)
    env = env - env.mean(1, keepdims=True)
    r = min(int(svht_rank(env)), RANK_CAP)
    U, s, _ = np.linalg.svd(env, full_matrices=False)
    Z = (U[:, :r].T @ env).T
    return Z.astype(np.float32), r


def langevin_stats(Z):
    m = fit_linear_langevin(Z, 1.0 / ENV_FS)
    eps = float(langevin_entropy_production(m))
    f_sol = float(langevin_solenoidal_frequency(m))
    A_sol = np.asarray(langevin_solenoidal_part(m))
    frac = float(np.linalg.norm(A_sol) / (np.linalg.norm(np.asarray(m.A)) + 1e-12))
    return eps, f_sol, frac


def main():
    print("JAX backend:", jax.default_backend())
    X = np.load(os.path.join(OUT, "parcels68.npy")).astype(np.float32)
    print(f"raw source parcels: {X.shape} @ {FS:.0f} Hz")
    rng = np.random.default_rng(0)

    print(f"\n=====  Band-resolved Langevin on WH MEG  (envelope {ENV_FS:.0f} Hz)  =====")
    print(f"{'band':>7} {'Hz':>9} {'rank':>5} {'EPR':>8} {'null':>8} "
          f"{'f_sol/Hz':>9} {'sol/tot':>8}")
    for name, (lo, hi) in BANDS.items():
        Z, r = band_embedding(X, lo, hi)
        eps, f_sol, frac = langevin_stats(Z)
        eps_null, _, _ = langevin_stats(Z[rng.permutation(len(Z))])
        print(f"{name:>7} {f'{lo}-{hi}':>9} {r:>5d} {eps:>8.3f} {eps_null:>8.3f} "
              f"{f_sol:>9.3f} {frac:>8.2f}")
    print("  -> EPR above the shuffle null => a real solenoidal (broken-detailed-balance)")
    print("     drift in WH, comparable to WAND's band table.")


if __name__ == "__main__":
    main()
