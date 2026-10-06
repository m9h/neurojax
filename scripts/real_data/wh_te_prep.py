#!/usr/bin/env python
"""Build + cache the low-order connectome-harmonic coefficient time series for
Wakeman-Henson (WH) -- the WH counterpart of `wand_te_prep.py`, reusing WAND's
geometric harmonic basis (same fsaverage Desikan-68 template space, so
`desikan68_centroids.npy` is directly shared -- WH has no structural connectome
of its own).

Geometric harmonics on the Desikan-68 centroid graph -> project the WH source
parcels (`wh_source_parcels_prep.py` output) -> analytic envelope -> downsample.
Saves harmonic_coeffs.npy (T, N_HARM) for `wh_jax_te.py` / `wand_jax_te.py`
pointed at WH_OUT.

    WH_OUT=/data/datasets/wh_src8_full WAND_OUT=/data/datasets/wand_src \\
      PYTHONPATH=src .venv-models/bin/python scripts/real_data/wh_te_prep.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from neurojax.spatial import connectome_harmonics, project_harmonics

OUT = os.environ.get("WH_OUT", "/data/datasets/wh_src8_full")
CENTROIDS_FROM = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
FS = 100.0                       # WH source parcels rate (wh_source_parcels_prep.py)
TE_FS = 25.0
N_HARM = 15


def analytic_env(x):
    T = x.shape[0]
    X = jnp.fft.fft(x, axis=0)
    h = jnp.zeros(T).at[0].set(1.0).at[1:(T + 1) // 2].set(2.0)
    h = h.at[T // 2].set(1.0) if T % 2 == 0 else h
    return jnp.abs(jnp.fft.ifft(X * h[:, None], axis=0))


def main():
    print("JAX backend:", jax.default_backend())
    cent = np.load(os.path.join(CENTROIDS_FROM, "desikan68_centroids.npy"))
    d2 = np.sum((cent[:, None, :] - cent[None, :, :]) ** 2, axis=-1)
    W = np.exp(-d2 / np.median(d2[d2 > 0]))
    np.fill_diagonal(W, 0.0)
    _, Phi = connectome_harmonics(jnp.asarray(W), normalized=False)
    Phi_lo = np.asarray(Phi)[:, 1:N_HARM + 1]

    X = np.load(os.path.join(OUT, "parcels68.npy")).astype(np.float32)
    a = np.asarray(project_harmonics(jnp.asarray(X), jnp.asarray(Phi_lo)))
    env = np.asarray(analytic_env(jnp.asarray(a)))
    ds = int(FS / TE_FS)
    env = env[: (len(env) // ds) * ds].reshape(-1, ds, N_HARM).mean(1)
    env = (env - env.mean(0)) / (env.std(0) + 1e-9)
    np.save(os.path.join(OUT, "harmonic_coeffs.npy"), env.astype(np.float64))
    print(f"WH harmonic coeffs {env.shape} @ {TE_FS:.0f} Hz -> {OUT}/harmonic_coeffs.npy")


if __name__ == "__main__":
    main()
