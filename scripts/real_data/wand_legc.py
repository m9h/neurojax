#!/usr/bin/env python
"""Leg C of the three-way cross-test on WAND resting MEG: run CEBRA + DYSCO on
the HMM network-state trajectory and ask whether the discrete TINDA cycle is a
continuous ring / limit cycle.

Operates on the K=12 state-probability trajectory (gamma), downsampled to the
network timescale, so CEBRA/DYSCO see the slow network dynamics whose cyclic
ordering TINDA characterizes.

    WAND_OUT=/data/datasets/wand_src PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wand_legc.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from neurojax.dynamics import CEBRA, DYSCO, SINDyOptimizer, KoopmanEstimator
from jaxctrl import polynomial_library

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
DS = 5            # 250 Hz -> 50 Hz (network timescale)
FS = 250.0 / DS


def main():
    print("JAX backend:", jax.default_backend())
    gamma = np.load(os.path.join(OUT, "oracle_gamma.npy")).astype(np.float32)  # (T, K)
    T, K = gamma.shape
    g = gamma[: (T // DS) * DS].reshape(-1, DS, K).mean(1)  # downsample
    g = (g - g.mean(0)) / (g.std(0) + 1e-6)
    print(f"network-state trajectory: {gamma.shape} -> downsampled {g.shape} @ {FS:.0f} Hz")
    X = jnp.asarray(g)

    # --- CEBRA: does the state trajectory embed as a ring? ---
    cebra = CEBRA(out_dim=2, temperature=0.05, n_steps=2000, batch_size=512, key=jax.random.PRNGKey(0))
    cebra.fit(X)
    e = np.asarray(cebra.transform(X))
    phi = np.arctan2(e[:, 1], e[:, 0])
    bins = np.histogram(phi, bins=12, range=(-np.pi, np.pi))[0]
    coverage = float(np.mean(bins > 0))  # fraction of the circle visited

    # --- DYSCO: is the latent flow a limit cycle, and at what frequency? ---
    dysco = DYSCO(latent_dim=2, library_degree=1, dt=1.0 / FS, n_steps=5000,
                  batch_size=512, key=jax.random.PRNGKey(0))
    dysco.fit(X)
    A = np.asarray(dysco.linear_part())
    evals = np.linalg.eigvals(A)
    is_oscillatory = bool(np.any(np.abs(evals.imag) > 1e-3))
    freq_hz = float(np.max(np.abs(evals.imag)) / (2 * np.pi))
    period_ms = 1000.0 / freq_hz if freq_hz > 0 else np.inf

    print("\n=========  Leg C on WAND network-state trajectory  =========")
    print(f"  CEBRA 2-D embedding circle coverage : {coverage:.2f} (1.0 = full ring)", flush=True)

    # The cycle is recovered as a RING by CEBRA; estimate its rotation frequency
    # ON THE RING (the correct coordinates), three ways. Subsample for the dense
    # solvers; the embedding is unit-norm so its dynamics are ~pure rotation.
    sub = max(1, e.shape[0] // 20000)
    es = e[::sub].astype(np.float32)
    dt_e = sub / FS

    # (i) phase-slope: net angular velocity of the ring traversal
    uphi = np.unwrap(np.arctan2(e[:, 1], e[:, 0]))
    freq_phase = abs(float(np.polyfit(np.arange(len(uphi)) / FS, uphi, 1)[0])) / (2 * np.pi)

    # (ii) SINDy linear flow on the ring
    de = np.gradient(es, dt_e, axis=0).astype(np.float32)
    Xi = np.asarray(SINDyOptimizer(threshold=0.0).fit(
        jnp.asarray(es), jnp.asarray(de), lambda x: polynomial_library(x, 1)))
    freq_s = float(np.max(np.abs(np.linalg.eigvals(Xi[1:3].T).imag)) / (2 * np.pi))

    # (iii) DMD / Koopman on the ring (features, samples)
    _, ev_dmd, _ = KoopmanEstimator().fit(jnp.asarray(es[:-1].T), jnp.asarray(es[1:].T))
    freq_dmd = float(np.max(np.abs(np.angle(np.asarray(ev_dmd)))) / (2 * np.pi * dt_e))

    print(f"  --- cycle rotation frequency on the CEBRA ring ---")
    print(f"  phase-slope                         : {freq_phase:.2f} Hz  (period {1000/freq_phase if freq_phase>0 else float('inf'):.0f} ms)")
    print(f"  SINDy  (linear flow on ring)        : {freq_s:.2f} Hz  (period {1000/freq_s if freq_s>0 else float('inf'):.0f} ms)")
    print(f"  DMD    (Koopman on ring)            : {freq_dmd:.2f} Hz  (period {1000/freq_dmd if freq_dmd>0 else float('inf'):.0f} ms)")
    print(f"  DYSCO (self-supervised latent) eig {np.round(evals,3)} | oscillatory: {is_oscillatory}")
    print(f"  -> compare to the TINDA cycle (van Es 2025: 300-1000 ms / ~1-3 Hz)")
    np.save(os.path.join(OUT, "legc_cebra_embedding.npy"), e)


if __name__ == "__main__":
    main()
