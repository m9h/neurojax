#!/usr/bin/env python
"""Reproduce Woolrich's structured network CYCLE (TINDA) on our real WH source-
space HMM states — for both the osl-dynamics oracle and the JAX HMM — to check
the cyclic ordering and cycle strength agree.

TINDA (van Es et al. 2025): inter-visit interval FO asymmetry -> directed graph
-> optimal cyclic ordering on the unit circle + cycle strength S.

Run in the osl env:
    .venv-oracle/bin/python scripts/real_data/wh_tinda_cycle.py
"""

import os
import sys
import types

fsl = types.ModuleType("fsl")
fsl.wrappers = types.ModuleType("fsl.wrappers")
sys.modules["fsl"] = fsl
sys.modules["fsl.wrappers"] = fsl.wrappers

import numpy as np
from osl_dynamics.analysis import tinda as T

OUT = os.environ.get("WH_OUT", "/data/datasets/wh_src")


def one_hot(states, k):
    oh = np.zeros((len(states), k), float)
    oh[np.arange(len(states)), states] = 1.0
    return oh


def cycle(states, k):
    fo, _, _ = T.tinda(one_hot(states, k))
    order = list(np.asarray(T.optimise_sequence(fo)).ravel())
    angles = T.circle_angles(order)
    asym = np.nanmean(fo[:, :, 0, :] - fo[:, :, 1, :], axis=-1)  # (k, k)
    strength = float(np.nanmean(T.compute_cycle_strength(angles, asym)))
    return order, strength


def main():
    oracle = np.load(os.path.join(OUT, "oracle_gamma.npy")).argmax(1)
    jax_st = np.load(os.path.join(OUT, "jax_hmm_states.npy"))
    n = min(len(oracle), len(jax_st))
    oracle, jax_st = oracle[:n], jax_st[:n]
    k = int(max(oracle.max(), jax_st.max())) + 1

    o_order, o_strength = cycle(oracle, k)
    j_order, j_strength = cycle(jax_st, k)

    print("===========  TINDA network cycle on real WH source-space MEG  ===========")
    print(f"  osl-dynamics oracle : cycle order {o_order}  | strength S = {o_strength:+.3f}")
    print(f"  JAX HMM             : cycle order {j_order}  | strength S = {j_strength:+.3f}")
    print("  (S>0 = consistent directional cycle; sign/order are arbitrary up to")
    print("   rotation & reflection of the circular layout)")


if __name__ == "__main__":
    main()
