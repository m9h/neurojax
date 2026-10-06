#!/usr/bin/env python
"""Langevin/Fokker-Planck test of the WH resting/task cycle: is the TINDA cycle
found by `wh_k12_cycle.py` also a non-equilibrium SOLENOIDAL flow in the
continuous state-space sense (the same test WAND's cycle passes)?

WH counterpart of `wand_langevin.py` -- same battery, own oracle_gamma.npy and
own TINDA order (derived here from `wh_k12_cycle.py`'s output, not WAND's).

    WH_OUT=/data/datasets/wh_src8_full PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wh_langevin.py [TINDA_ORDER...]
"""

import os
import sys

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

OUT = os.environ.get("WH_OUT", "/data/datasets/wh_src8_full")
DS = 4                      # 100 Hz -> 25 Hz
FS = 100.0 / DS
K = int(os.environ.get("WH_LANGEVIN_DIM", "4"))


def main():
    print("JAX backend:", jax.default_backend())
    gamma = np.load(os.path.join(OUT, "oracle_gamma.npy")).astype(np.float32)
    T, K_states = gamma.shape

    if len(sys.argv) > 1:
        tinda_order = [int(x) for x in sys.argv[1:]]
    else:
        tinda_order = list(range(K_states))
        print("  (no TINDA_ORDER given on the command line -- run wh_k12_cycle.py "
              "first and pass its `order`; using identity order as a placeholder)")

    g = gamma[: (T // DS) * DS].reshape(-1, DS, K_states).mean(1)
    g = g - g.mean(0)
    Vt = np.linalg.svd(g, full_matrices=False)[2]
    Z = (g @ Vt[:K].T).astype(np.float64)
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

    rng = np.random.default_rng(0)
    eps_null = float(langevin_entropy_production(
        fit_linear_langevin(Z[rng.permutation(len(Z))], dt)))

    states = gamma.argmax(1)
    d_eps = float(discrete_entropy_production(states, K_states))
    d_eps_null = float(discrete_entropy_production(
        states[rng.permutation(len(states))], K_states))
    F = np.asarray(transition_flux(states, K_states))
    cyc = np.array([F[tinda_order[i], tinda_order[(i + 1) % K_states]]
                    for i in range(K_states)])
    frac_forward = float(np.mean(np.sign(cyc) == np.sign(np.median(cyc))))

    print("\n========  Langevin / Fokker-Planck on the WH resting cycle  ========")
    print("  [continuous Langevin on the smoothed HMM gamma]")
    print(f"    deterministic drift rotation        : {f_det:.3f} Hz")
    print(f"    solenoidal rotation frequency       : {f_sol:.3f} Hz")
    print(f"    entropy production (real/null)      : {eps:.4f} / {eps_null:.4f}")
    print(f"    ||solenoidal||/||gradient|| drift   : {ratio:.2f}")
    print("  [discrete entropy production on the K states (Lynn 2021)]")
    print(f"    discrete entropy production (real)  : {d_eps:.4f}")
    print(f"    discrete entropy production (null)  : {d_eps_null:.4f}   <- shuffled")
    print(f"    flux circulates along TINDA order   : {100*frac_forward:.0f}% of edges same direction")


if __name__ == "__main__":
    main()
