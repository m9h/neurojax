#!/usr/bin/env python
"""Band-resolved non-equilibrium analysis of WAND resting MEG, on the RAW source
parcels (high-frequency preserved — not the smoothed HMM gamma).

For each band: EEGLAB-style complex Morlet -> amplitude envelope -> whiten by the
Donoho-shrunk instantaneous covariance -> measure time-reversal irreversibility
(INSIDEOUT-style lagged-covariance asymmetry; the model-free entropy-production
proxy, Deco 2022 / Tewarie 2023). Tests whether the irreversibility the
single-timescale / smoothed analysis collapsed is in fact frequency-resolved.

    WAND_OUT=/data/datasets/wand_src PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wand_timefreq_irrev.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from neurojax.analysis.timefreq import morlet_cwt, eeglab_cycles
from neurojax.dynamics import shrink_covariance

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
FS = 250.0
BANDS = {"delta": (2, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
ENV_FS = 50.0                       # envelope downsample
LAG_MS = 120.0                      # network-timescale lag


def irreversibility(E, lag, rng):
    """Whitened lagged-covariance asymmetry of envelopes E (n_parcel, n_time)."""
    E = E - E.mean(1, keepdims=True)
    n = E.shape[1]
    S0 = np.asarray(shrink_covariance(jnp.asarray((E @ E.T) / n), n, "frobenius"))
    w, V = np.linalg.eigh(S0)
    Wmat = V @ np.diag(1.0 / np.sqrt(np.clip(w, 1e-6, None))) @ V.T   # Σ^{-1/2}
    Ew = Wmat @ E
    L = (Ew[:, :-lag] @ Ew[:, lag:].T) / (n - lag)
    asym = np.linalg.norm(L - L.T)          # antisymmetric (lead-lag) magnitude
    # time-shuffle null (destroys lagged structure -> ~0)
    Esh = E[:, rng.permutation(n)]
    Ews = Wmat @ (Esh - Esh.mean(1, keepdims=True))
    Ln = (Ews[:, :-lag] @ Ews[:, lag:].T) / (n - lag)
    asym_null = np.linalg.norm(Ln - Ln.T)
    return asym, asym_null


def main():
    print("JAX backend:", jax.default_backend())
    X = np.load(os.path.join(OUT, "parcels68.npy")).astype(np.float32)  # (T, 68)
    print(f"raw source parcels: {X.shape} @ {FS:.0f} Hz")
    rng = np.random.default_rng(0)
    ds = int(FS / ENV_FS)
    lag = max(1, int(LAG_MS / 1000.0 * ENV_FS))

    print(f"\n=====  Band-resolved irreversibility on WAND resting MEG (lag {LAG_MS:.0f} ms)  =====")
    print(f"{'band':>7} {'Hz':>9} {'irrev':>8} {'null':>8} {'ratio':>7}")
    for name, (lo, hi) in BANDS.items():
        fc = jnp.linspace(lo, hi, 6)
        C = morlet_cwt(jnp.asarray(X.T), FS, fc, eeglab_cycles(fc, 3.0, 0.5))  # (68, 6, T)
        env = np.asarray(jnp.mean(jnp.abs(C), axis=1))                          # (68, T)
        env = env[:, : (env.shape[1] // ds) * ds].reshape(68, -1, ds).mean(2)   # -> ENV_FS
        irr, null = irreversibility(env, lag, rng)
        print(f"{name:>7} {f'{lo}-{hi}':>9} {irr:>8.4f} {null:>8.4f} {irr/(null+1e-9):>7.2f}")
    print("  -> frequency-resolved irreversibility (entropy-production proxy);")
    print("     single-timescale / smoothed analysis averages this away")


if __name__ == "__main__":
    main()
