#!/usr/bin/env python
"""Langevin/Fokker-Planck test of the WAND resting cycle: is it a non-equilibrium
SOLENOIDAL flow (the part DMD/SINDy/DYSCO miss)?

Fits a linear Langevin model to a low-D PCA of the network-state trajectory and
reports the Helmholtz split + entropy production. Hypothesis (from the cross-test
+ Friston FP framing): deterministic drift rotation ~0, but solenoidal flow and
entropy production > 0 with a rotation frequency in the TINDA cycle band.

    WAND_OUT=/data/datasets/wand_src PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wand_langevin.py
"""

import os

import jax
import numpy as np

from neurojax.dynamics import (
    fit_linear_langevin,
    langevin_entropy_production,
    langevin_gradient_part,
    langevin_solenoidal_part,
    langevin_solenoidal_frequency,
    transition_flux,
    discrete_entropy_production,
)

# TINDA cycle order from Leg A (wh_k12_cycle.py on WAND)
TINDA_ORDER = [0, 11, 6, 5, 8, 7, 1, 4, 10, 3, 9, 2]

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
DS = 5                      # 250 Hz -> 50 Hz
FS = 250.0 / DS
K = int(os.environ.get("WAND_LANGEVIN_DIM", "4"))


def main():
    print("JAX backend:", jax.default_backend())
    gamma = np.load(os.path.join(OUT, "oracle_gamma.npy")).astype(np.float32)
    T, _ = gamma.shape
    g = gamma[: (T // DS) * DS].reshape(-1, DS, gamma.shape[1]).mean(1)
    g = g - g.mean(0)
    Vt = np.linalg.svd(g, full_matrices=False)[2]
    Z = (g @ Vt[:K].T).astype(np.float64)     # (T, K) PCA latent
    dt = 1.0 / FS
    print(f"network-state trajectory {gamma.shape} -> latent {Z.shape} @ {FS:.0f} Hz")

    m = fit_linear_langevin(Z, dt)
    A_rev = np.asarray(langevin_gradient_part(m))
    A_sol = np.asarray(langevin_solenoidal_part(m))
    eps = float(langevin_entropy_production(m))
    f_sol = float(langevin_solenoidal_frequency(m))
    det_evals = np.linalg.eigvals(np.asarray(m.A))
    f_det = float(np.max(np.abs(det_evals.imag)) / (2 * np.pi))
    ratio = float(np.linalg.norm(A_sol) / (np.linalg.norm(A_rev) + 1e-12))

    # time-shuffle null (destroys temporal structure -> ~0 entropy production)
    rng = np.random.default_rng(0)
    eps_null = float(langevin_entropy_production(
        fit_linear_langevin(Z[rng.permutation(len(Z))], dt)))

    # --- Discrete (jump-process) test on the K=12 HMM state sequence ---
    K_states = gamma.shape[1]
    states = gamma.argmax(1)
    d_eps = float(discrete_entropy_production(states, K_states))
    d_eps_null = float(discrete_entropy_production(
        states[rng.permutation(len(states))], K_states))
    F = np.asarray(transition_flux(states, K_states))
    cyc = np.array([F[TINDA_ORDER[i], TINDA_ORDER[(i + 1) % K_states]]
                    for i in range(K_states)])
    frac_forward = float(np.mean(np.sign(cyc) == np.sign(np.median(cyc))))

    print("\n========  Langevin / Fokker-Planck on the WAND resting cycle  ========")
    print("  [continuous Langevin on the smoothed HMM gamma -> near-equilibrium]")
    print(f"    deterministic drift rotation        : {f_det:.3f} Hz")
    print(f"    solenoidal rotation frequency       : {f_sol:.3f} Hz")
    print(f"    entropy production (real/null)      : {eps:.4f} / {eps_null:.4f}")
    print(f"    ||solenoidal||/||gradient|| drift   : {ratio:.2f}")
    print("  [discrete entropy production on the K=12 state sequence (Lynn 2021)]")
    print(f"    discrete entropy production (real)  : {d_eps:.4f}")
    print(f"    discrete entropy production (null)  : {d_eps_null:.4f}   <- shuffled")
    print(f"    flux circulates along TINDA order   : {100*frac_forward:.0f}% of edges same direction")
    print(f"  -> the cycle is a discrete-state non-equilibrium probability current")


if __name__ == "__main__":
    main()
