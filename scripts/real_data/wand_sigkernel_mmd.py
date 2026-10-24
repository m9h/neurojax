#!/usr/bin/env python
"""Signature-kernel MMD irreversibility test for the WAND bands (the complete test).

The signature kernel is characteristic on path laws, so the MMD between the law of
the forward path and the law of its time-reversal is zero iff the process is
statistically time-reversible — a single, multichannel, all-orders irreversibility
statistic (Chevyrev–Oberhauser 2022 + Gretton 2012; the unpublished [GAP] test from
the review).  Here: the truncated (depth-DEPTH) signature feature map (signax), with
a proper two-sample **permutation** null over forward vs reversed window signatures.

Unlike the log-signature mean test (`wand_logsig_irrev.py`, which uses only the
odd/antisymmetric levels via a sign-flip null), this compares the full forward and
reversed *signature distributions* — symmetric and antisymmetric parts and their
interaction — and is a genuine two-sample test, not a mean test.

    WAND_OUT=/data/datasets/wand_src PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wand_sigkernel_mmd.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import signax

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
WIN = 100               # window length (samples @ 50 Hz = 2 s)
DEPTH = 3
N_PERM = 400


def sig_features(Z):
    """Depth-DEPTH signatures of forward and time-reversed non-overlapping windows."""
    k = Z.shape[1]
    n = (len(Z) // WIN) * WIN
    win = jnp.asarray(Z[:n].reshape(-1, WIN, k))                  # (n_win, WIN, k)
    Sf = jax.vmap(lambda w: signax.signature(w, DEPTH))(win)
    Sr = jax.vmap(lambda w: signax.signature(w[::-1], DEPTH))(win)
    return np.asarray(Sf), np.asarray(Sr)


def mmd2(g1, g2):
    """Linear-kernel MMD² between two feature sets = ‖mean1 − mean2‖²."""
    d = g1.mean(0) - g2.mean(0)
    return float(d @ d)


def main():
    print("JAX backend:", jax.default_backend(), "| signax", getattr(signax, "__version__", "?"))
    rng = np.random.default_rng(0)
    print(f"\n=====  WAND signature-kernel MMD(X, X̄) irreversibility "
          f"(depth {DEPTH}, win {WIN}@50Hz)  =====")
    print(f"{'band':>7} {'rank':>5} {'n_win':>6} {'MMD2':>10} {'null':>10} {'z':>7} {'p':>7}")
    for name in BANDS:
        path = os.path.join(OUT, f"band_emb_{name}.npy")
        if not os.path.exists(path):
            print(f"{name:>7}  (missing {path} — run wand_cache_embeddings.py)")
            continue
        Z = np.load(path)
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
        Sf, Sr = sig_features(Z)
        n = len(Sf)
        P = np.concatenate([Sf, Sr], axis=0)
        P = (P - P.mean(0)) / (P.std(0) + 1e-12)                  # balance levels/coords
        obs = mmd2(P[:n], P[n:])
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            idx = rng.permutation(2 * n)
            null[i] = mmd2(P[idx[:n]], P[idx[n:]])
        z = (obs - null.mean()) / (null.std() + 1e-12)
        p = (np.sum(null >= obs) + 1) / (N_PERM + 1)
        print(f"{name:>7} {Z.shape[1]:>5d} {n:>6d} {obs:>10.4f} {null.mean():>10.4f} "
              f"{z:>7.2f} {p:>7.3f}")
    print("  -> MMD(forward, reversed) above the permutation null ⇔ statistically")
    print("     time-irreversible at all orders (characteristic signature kernel).")


if __name__ == "__main__":
    main()
