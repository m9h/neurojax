#!/usr/bin/env python
"""Higher-order (nonlinear) irreversibility of the WAND bands via log-signatures.

The level-2 antisymmetric signature (Lévy area) is the *linear* circulation — the
same object as the Langevin α* and the lagged-covariance asymmetry.  The log-
signature negates uniformly under time reversal (every Lie level is time-odd), so
its higher levels carry *nonlinear* path asymmetry that the Gaussian/linear
estimators cannot see.

Test: window each band path, compute the depth-3 log-signature (signax) per window,
and split into levels.  For a time-reversible process the windowed log-sigs are
symmetric under negation, so E[logsig_level] = 0.  The observed ‖mean_level‖ is
ranked against a **sign-flip null** (randomly negate each window's log-sig and
recompute the mean) — a self-contained, exact reversible null requiring no surrogate
synthesis.  Level 2 is the linear circulation (a sanity check that should be
significant); **level 3 significant ⇒ genuine nonlinear irreversibility beyond the
Gaussian/linear story** — independent evidence the other axes can't provide.

    WAND_OUT=/data/datasets/wand_src PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wand_logsig_irrev.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import signax

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
WIN = 100               # window length (samples @ 50 Hz = 2 s); short -> stable signature
DEPTH = 3
N_NULL = 400


def witt_bounds(k):
    """Cumulative log-signature level boundaries for k channels (Witt dims, l≤3)."""
    d1, d2, d3 = k, (k * k - k) // 2, (k ** 3 - k) // 3
    return [(0, d1), (d1, d1 + d2), (d1 + d2, d1 + d2 + d3)]


def windowed_logsig(Z):
    """Non-overlapping depth-DEPTH log-signatures of Z (T, k) -> (n_win, logsig_dim)."""
    k = Z.shape[1]
    n = (len(Z) // WIN) * WIN
    W = jnp.asarray(Z[:n].reshape(-1, WIN, k))                # (n_win, WIN, k)
    return np.asarray(jax.vmap(lambda w: signax.logsignature(w, DEPTH))(W))


def signflip_test(L, rng):
    """Observed per-coordinate mean and the sign-flip null ensemble of means."""
    means = L.mean(0)
    flips = rng.choice([-1.0, 1.0], size=(N_NULL, L.shape[0]))
    null_means = np.einsum("ni,ij->nj", flips, L) / L.shape[0]   # (N_NULL, dim)
    return means, null_means


def main():
    print("JAX backend:", jax.default_backend(), "| signax", getattr(signax, "__version__", "?"))
    rng = np.random.default_rng(0)
    print(f"\n=====  WAND higher-order irreversibility (log-sig depth {DEPTH}, "
          f"win {WIN}@50Hz)  =====")
    print(f"{'band':>7} {'rank':>5} {'lvl2_z':>7} {'lvl2_p':>7} {'lvl3_z':>7} {'lvl3_p':>7}")
    for name in BANDS:
        path = os.path.join(OUT, f"band_emb_{name}.npy")
        if not os.path.exists(path):
            print(f"{name:>7}  (missing {path} — run wand_cache_embeddings.py)")
            continue
        Z = np.load(path)
        k = Z.shape[1]
        L = windowed_logsig(Z)                                   # (n_win, dim)
        bounds = witt_bounds(k)
        means, null_means = signflip_test(L, rng)
        row = [f"{name:>7}", f"{k:>5d}"]
        for lvl in (1, 2):                                       # level 2 and 3 (0-indexed 1,2)
            lo, hi = bounds[lvl]
            obs = np.linalg.norm(means[lo:hi])
            nrm = np.linalg.norm(null_means[:, lo:hi], axis=1)
            z = (obs - nrm.mean()) / (nrm.std() + 1e-12)
            p = (np.sum(nrm >= obs) + 1) / (len(nrm) + 1)
            row += [f"{z:>7.2f}", f"{p:>7.3f}"]
        print(" ".join(row))
    print("  -> lvl2 = linear circulation (sanity); lvl3 significant ⇒ NONLINEAR")
    print("     irreversibility beyond the Gaussian/linear story (sign-flip null).")


if __name__ == "__main__":
    main()
