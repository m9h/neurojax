#!/usr/bin/env python
"""JAX-native multivariate transfer entropy on the WAND connectome harmonics — the
fast, GPU, differentiable replacement for the IDTxl oracle (Gaussian/linear regime).

Same question as ``wand_transfer_entropy.py`` (does "who drives whom" among the
low-order harmonics form a directed loop = the rotation direction?), but the whole
network + a circular-shift surrogate null run on the GB10 in seconds instead of the
IDTxl greedy-selection minutes.

    WAND_OUT=/data/datasets/wand_src PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wand_jax_te.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from neurojax.dynamics import mvte_matrix

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
N_SURR = 100
MAXLAG = 2


def main():
    print("JAX backend:", jax.default_backend())
    a = np.load(os.path.join(OUT, "harmonic_coeffs.npy"))             # (T, n)
    T, n = a.shape
    A = jnp.asarray(a)
    print(f"harmonic coeffs {a.shape}; JAX multivariate conditional TE on {n} harmonics")

    M = np.asarray(mvte_matrix(A))                                    # (n, n) conditional TE

    # circular-shift surrogate null: each column independently shifted (kills cross-node
    # coupling, keeps autocorrelation), recompute the whole MV-TE matrix.  Loop over
    # surrogates (one ~2GB matrix at a time; XLA-cached so each is fast) — vmapping the
    # outer dim too would materialise all of them at once and OOM.
    idx0 = jnp.arange(T)

    @jax.jit
    def surrogate_mvte(key):
        shifts = jax.random.randint(key, (n,), MAXLAG + 1, T - MAXLAG - 1)
        sh = jnp.take_along_axis(A, (idx0[:, None] - shifts[None, :]) % T, axis=0)
        return mvte_matrix(sh)

    null = np.stack([np.asarray(surrogate_mvte(k))
                     for k in jax.random.split(jax.random.PRNGKey(0), N_SURR)])
    thresh = np.percentile(null, 99)                                 # p<0.01 network threshold
    sig = (M > thresh) & (~np.eye(n, dtype=bool))

    print(f"\n=====  JAX multivariate TE — directed coupling among harmonics  =====")
    print(f"{int(sig.sum())} significant edges (p<0.01 vs {N_SURR} circular-shift surrogates; "
          f"density {sig.sum()/(n*(n-1)):.2f})")
    outdeg, indeg = sig.sum(1), sig.sum(0)
    drivers = np.argsort(-outdeg)[:4]
    recv = np.argsort(-indeg)[:4]
    print(f"top drivers (out-degree): {[f'H{d+1}({int(outdeg[d])})' for d in drivers]}")
    print(f"top receivers (in-degree): {[f'H{r+1}({int(indeg[r])})' for r in recv]}")
    Ms = M.copy(); Ms[~sig] = 0
    se = np.dstack(np.unravel_index(np.argsort(Ms.ravel())[::-1], Ms.shape))[0][:6]
    print("strongest significant directed edges (TE, nats):")
    for s, t in se:
        if Ms[s, t] > 0:
            print(f"    H{s+1:>2d} -> H{t+1:>2d}   TE={M[s, t]:.4f}")

    # directed cycles via matrix powers: tr(A^k)>0 => a length-k closed walk
    Ab = sig.astype(int)
    cyc = [k for k in range(2, n + 1) if np.trace(np.linalg.matrix_power(Ab, k)) > 0]
    print(f"\ndirected cycles present at lengths: {cyc[:6]} "
          f"({'YES' if cyc else 'none'} — a loop = the rotational/circulation structure)")
    print("  -> a directed loop among low-order harmonics is the TE signature of the")
    print("     solenoidal cycle (cross-validates the Langevin circulation).")


if __name__ == "__main__":
    main()
