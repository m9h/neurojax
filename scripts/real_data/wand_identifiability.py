#!/usr/bin/env python
"""Donoho-Tanner identifiability check for the sparse WAND drift fits.

The band-resolved Langevin (``wand_band_langevin.py``) and the SINDy ring fits are
sparse regressions: recover a ``k``-sparse coefficient vector from an ``n×N``
library (``n`` time points, ``N`` candidate terms).  The Donoho-Tanner *weak*
phase transition — via the statistical dimension of the ℓ1 descent cone
(Amelunxen-Lotz-McCoy-Tropp 2014, matching Donoho-Tanner 2009) — says when
ℓ1/STLSQ can recover the support at all: identifiable iff ``n > N·δ(k/N)``.  This
is the identifiability counterpart of the Gavish-Donoho rank selection already
used on the estimation side (``svht_rank``).

Per band: EEGLAB complex Morlet -> amplitude envelope -> optimal-rank PCA (svht) ->
degree-2 polynomial drift fit (SINDy/STLSQ on the 1st Kramers-Moyal moment; the
threshold is reported) -> count the active support ``k`` -> Donoho-Tanner regime.
The headline is the **minimum window length** that still identifies the sparse
drift — the actionable floor for a windowed-SINDy analysis.  With the full 6-min
record every band is hugely oversampled (delta = n/N >> 1), so identifiability is
*not* the binding constraint here; the check quantifies exactly how much headroom
there is and how short a segment the sparse support could be recovered from.

Caveat: the theory assumes a (near-)Gaussian / rotationally-invariant design.
Polynomial-feature libraries are strongly *correlated*, so this is the optimistic
/ information-theoretic bound — a correlated design needs strictly more samples.

    WAND_OUT=/data/datasets/wand_src PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wand_identifiability.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from neurojax.analysis.timefreq import morlet_cwt, eeglab_cycles
from neurojax.dynamics import (
    svht_rank,
    SINDyOptimizer,
    polynomial_library,
    donoho_tanner_regime,
    donoho_tanner_threshold,
)

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
FS = 250.0
BANDS = {"delta": (2, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
ENV_FS = 50.0                       # envelope downsample (network timescale)
RANK_CAP = 24                       # guard the fit against over-ranking
STLSQ_THRESH = 0.05                 # reported: STLSQ coefficient threshold (z-scored PCs)


def band_embedding(X, lo, hi):
    """Morlet envelope of one band -> unit-variance optimal-rank PCA scores."""
    fc = jnp.linspace(lo, hi, 6)
    C = morlet_cwt(jnp.asarray(X.T), FS, fc, eeglab_cycles(fc, 3.0, 0.5))   # (68,6,T)
    env = np.asarray(jnp.mean(jnp.abs(C), axis=1))                          # (68, T)
    ds = int(FS / ENV_FS)
    env = env[:, : (env.shape[1] // ds) * ds].reshape(68, -1, ds).mean(2)   # -> ENV_FS
    env = env - env.mean(1, keepdims=True)
    r = min(int(svht_rank(env)), RANK_CAP)                                  # Gavish-Donoho
    U, s, _ = np.linalg.svd(env, full_matrices=False)
    Z = (U[:, :r].T @ env).T                                                # (T, r) PC scores
    Z = Z / (Z.std(0, keepdims=True) + 1e-12)                              # unit variance
    return Z.astype(np.float32), r


def sparse_drift_support(Z):
    """Degree-2 SINDy drift on the 1st Kramers-Moyal moment -> (P, k).

    ``P`` = polynomial library width; ``k`` = worst-case active support over the
    output equations (the binding regression for Donoho-Tanner)."""
    Xs = Z[:-1]
    dZ = (Z[1:] - Z[:-1]) * ENV_FS                          # ~ first KM moment (drift)
    lib = lambda A: polynomial_library(A, degree=2)
    Xi = np.asarray(SINDyOptimizer(threshold=STLSQ_THRESH, max_iter=10).fit(
        jnp.asarray(Xs), jnp.asarray(dZ), lib))
    P = Xi.shape[0]
    k = int(np.max(np.sum(np.abs(Xi) > 0, axis=0)))        # nonzeros in the densest eqn
    return P, k


def main():
    print("JAX backend:", jax.default_backend())
    X = np.load(os.path.join(OUT, "parcels68.npy")).astype(np.float32)   # (T, 68)
    print(f"raw source parcels: {X.shape} @ {FS:.0f} Hz")
    print(f"STLSQ threshold (reported): {STLSQ_THRESH}")
    print(f"Donoho-Tanner weak anchor rho_W(0.5) = {donoho_tanner_threshold(0.5):.4f}"
          "  (literature 0.385)")

    print(f"\n=====  Donoho-Tanner identifiability of the sparse band drifts  =====")
    print(f"{'band':>7} {'rank':>5} {'P':>5} {'k':>4} {'n':>8} "
          f"{'delta':>9} {'ident':>6} {'min_win/s':>10} {'headroom':>9}")
    for name, (lo, hi) in BANDS.items():
        Z, r = band_embedding(X, lo, hi)
        P, k = sparse_drift_support(Z)
        n = len(Z) - 1
        reg = donoho_tanner_regime(n, P, k)
        min_win_s = reg.min_measurements / ENV_FS
        print(f"{name:>7} {r:>5d} {P:>5d} {k:>4d} {n:>8d} "
              f"{reg.delta:>9.1f} {str(reg.identifiable):>6} "
              f"{min_win_s:>10.2f} {reg.headroom:>9.0f}")
    print("  -> delta = n/N (>1 ⇔ oversampled); min_win/s = shortest segment that")
    print("     DT-identifies the k-sparse drift at ENV_FS (the windowed-SINDy floor).")
    print("  Caveat: polynomial libraries are correlated, so DT is the optimistic")
    print("  bound -- a correlated design needs strictly more samples.")


if __name__ == "__main__":
    main()
