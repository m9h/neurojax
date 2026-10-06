#!/usr/bin/env python
"""Band-resolved Langevin/Fokker-Planck on WAND resting MEG — the MODEL-BASED
counterpart of ``wand_timefreq_irrev.py``.

The model-free ‖L−Lᵀ‖ asymmetry showed irreversibility is frequency-resolved and
theta-dominant (theta ratio 2.70 -> gamma 1.21), but it cannot say *why*: a raw
lead-lag asymmetry conflates a genuine solenoidal (broken-detailed-balance) drift
with a plain amplitude/diffusion effect. A fitted linear Langevin separates them:

    dz = A z dt + sqrt(2D) dW
    gradient (reversible) drift   A_rev = -D Σ⁻¹
    solenoidal (irreversible)     A_sol = A + D Σ⁻¹     (cyclic probability current)
    entropy production rate        Ṡ = tr(A_sol Σ A_solᵀ D⁻¹) ≥ 0

For each band: EEGLAB complex Morlet -> amplitude envelope -> downsample ->
Gavish-Donoho optimal-rank PCA embedding (``svht_rank``; reported) -> linear
Langevin fit -> entropy production, solenoidal rotation frequency, and the
solenoidal/total drift fraction, each against a time-shuffle null. Ṡ is invariant
under any invertible linear change of coordinates, so the PCA embedding gives the
same entropy production as full whitening — the only reported choice is the rank.

    WAND_OUT=/data/datasets/wand_src PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wand_band_langevin.py
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

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
FS = 250.0
BANDS = {"delta": (2, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
ENV_FS = 50.0                       # envelope downsample (network timescale)
RANK_CAP = 24                       # guard the Langevin fit against over-ranking


def band_embedding(X, lo, hi):
    """Morlet envelope of one band -> optimal-rank PCA scores Z (T, r), rank r."""
    fc = jnp.linspace(lo, hi, 6)
    C = morlet_cwt(jnp.asarray(X.T), FS, fc, eeglab_cycles(fc, 3.0, 0.5))   # (68,6,T)
    env = np.asarray(jnp.mean(jnp.abs(C), axis=1))                          # (68, T)
    ds = int(FS / ENV_FS)
    env = env[:, : (env.shape[1] // ds) * ds].reshape(68, -1, ds).mean(2)   # -> ENV_FS
    env = env - env.mean(1, keepdims=True)
    r = min(int(svht_rank(env)), RANK_CAP)                                  # Gavish-Donoho
    U, s, _ = np.linalg.svd(env, full_matrices=False)
    Z = (U[:, :r].T @ env).T                                                # (T, r) PC scores
    return Z.astype(np.float32), r


def langevin_stats(Z):
    """Entropy production, solenoidal rotation freq, ‖A_sol‖/‖A‖ for trajectory Z."""
    m = fit_linear_langevin(Z, 1.0 / ENV_FS)
    eps = float(langevin_entropy_production(m))
    f_sol = float(langevin_solenoidal_frequency(m))
    A_sol = np.asarray(langevin_solenoidal_part(m))
    frac = float(np.linalg.norm(A_sol) / (np.linalg.norm(np.asarray(m.A)) + 1e-12))
    return eps, f_sol, frac


def main():
    print("JAX backend:", jax.default_backend())
    X = np.load(os.path.join(OUT, "parcels68.npy")).astype(np.float32)   # (T, 68)
    print(f"raw source parcels: {X.shape} @ {FS:.0f} Hz")
    rng = np.random.default_rng(0)

    print(f"\n=====  Band-resolved Langevin on WAND resting MEG  (envelope {ENV_FS:.0f} Hz)  =====")
    print(f"{'band':>7} {'Hz':>9} {'rank':>5} {'EPR':>8} {'null':>8} "
          f"{'f_sol/Hz':>9} {'sol/tot':>8}")
    for name, (lo, hi) in BANDS.items():
        Z, r = band_embedding(X, lo, hi)
        eps, f_sol, frac = langevin_stats(Z)
        eps_null, _, _ = langevin_stats(Z[rng.permutation(len(Z))])       # shuffle -> ~0 EPR
        print(f"{name:>7} {f'{lo}-{hi}':>9} {r:>5d} {eps:>8.3f} {eps_null:>8.3f} "
              f"{f_sol:>9.3f} {frac:>8.2f}")
    print("  -> EPR above the shuffle null ⇔ a real solenoidal (broken-detailed-balance)")
    print("     drift; f_sol is the frequency of the stochastic cycle DMD/SINDy/DYSCO miss.")


if __name__ == "__main__":
    main()
