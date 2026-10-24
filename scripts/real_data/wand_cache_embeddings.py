#!/usr/bin/env python
"""Cache the per-band Morlet-envelope PCA embeddings to .npy.

The band embeddings are built with JAX/Morlet on the GPU (``.venv-models``); the
recurrence/determinism analysis runs under pyunicorn in the CPU oracle
(``.venv-oracle``), which has no JAX.  This bridges the two: build once, save
``band_emb_{name}.npy`` (T, r) for each band so the oracle can load them.

    WAND_OUT=/data/datasets/wand_src PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wand_cache_embeddings.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from neurojax.analysis.timefreq import morlet_cwt, eeglab_cycles
from neurojax.dynamics import svht_rank

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
FS = 250.0
BANDS = {"delta": (2, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
ENV_FS = 50.0
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
    return ((U[:, :r].T @ env).T).astype(np.float32), r


def main():
    print("JAX backend:", jax.default_backend())
    X = np.load(os.path.join(OUT, "parcels68.npy")).astype(np.float32)
    for name, (lo, hi) in BANDS.items():
        Z, r = band_embedding(X, lo, hi)
        path = os.path.join(OUT, f"band_emb_{name}.npy")
        np.save(path, Z)
        print(f"{name:>7}  ({lo}-{hi} Hz)  rank={r:>2d}  Z={Z.shape}  -> {path}")


if __name__ == "__main__":
    main()
