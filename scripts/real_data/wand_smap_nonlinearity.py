#!/usr/bin/env python
"""S-map nonlinearity test for the WAND bands — the EDM positive determinism probe.

Complements the RQA determinism axis with the Sugihara (1994) S-map θ test, which
the research flagged as the cleanest "deterministic nonlinear skeleton" detector and
a JAX-port candidate.  θ=0 is a single global linear map (≡ DMD / linear SINDy /
the Langevin drift); if forecast skill *rises* for θ>0 the dynamics are
state-dependent (nonlinear deterministic).  The gain Δρ = ρ(θ*) − ρ(0) is ranked
against an IAAFT-surrogate floor (same Julia oracle): Δρ ≈ surrogate ⇒ a global
linear map suffices ⇒ confirms the linear/stochastic reading; Δρ ≫ surrogate ⇒
hidden nonlinear determinism the linear methods missed.

    WAND_OUT=/data/datasets/wand_src PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wand_smap_nonlinearity.py
"""

import os
import subprocess

import jax
import numpy as np

from neurojax.dynamics import smap_nonlinearity

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
SCR = os.environ.get("SCRATCH", "/tmp/claude-1000/-home-mhough-dev-neurojax/"
                     "a44a9232-d44b-4b8b-8812-560682c446fa/scratchpad")
ORACLE = os.path.join(os.path.dirname(__file__), "oracle_surrogates")
JULIA = os.path.expanduser("~/.juliaup/bin/julia")
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
DS = 5                  # 50 Hz -> 10 Hz
THETAS = np.array([0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0])
N_SURR = 20


def dgain(Z):
    """Δρ = ρ(θ*) − ρ(0): the S-map nonlinearity gain (also returns ρ(0))."""
    rho = np.asarray(smap_nonlinearity(Z, THETAS, lag=1, max_pts=2000))
    return float(rho.max() - rho[0]), float(rho[0])


def iaaft(Z, n, seed):
    fin, fout = os.path.join(SCR, "sm_in.npy"), os.path.join(SCR, "sm_out.npy")
    np.save(fin, Z.astype(np.float64))
    subprocess.run([JULIA, "--startup-file=no",
                    os.path.join(ORACLE, "gen_surrogates.jl"),
                    fin, fout, str(n), "iaaft", str(seed)],
                   check=True, capture_output=True,
                   env=dict(os.environ, JULIA_PROJECT=ORACLE))
    return np.load(fout)


def main():
    print("JAX backend:", jax.default_backend())
    print(f"\n=====  WAND S-map nonlinearity (θ test, lag 1 @ {50//DS} Hz)  =====")
    print(f"{'band':>7} {'rank':>5} {'rho0':>7} {'dRho':>7} {'surr_dRho':>10} {'z':>6}")
    for name in BANDS:
        path = os.path.join(OUT, f"band_emb_{name}.npy")
        if not os.path.exists(path):
            print(f"{name:>7}  (missing {path} — run wand_cache_embeddings.py)")
            continue
        Z = np.load(path)[::DS]
        dr, rho0 = dgain(Z)
        S = iaaft(Z, N_SURR, 0)
        surr = np.array([dgain(S[i])[0] for i in range(len(S))])
        z = (dr - surr.mean()) / (surr.std() + 1e-12)
        print(f"{name:>7} {Z.shape[1]:>5d} {rho0:>7.3f} {dr:>7.4f} "
              f"{surr.mean():>10.4f} {z:>6.2f}")
    print("  -> dRho ≈ surr_dRho (low z) ⇔ a global linear map suffices ⇒ confirms")
    print("     the linear/stochastic reading; dRho ≫ surr ⇒ nonlinear determinism.")


if __name__ == "__main__":
    main()
