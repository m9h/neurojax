#!/usr/bin/env python
"""Leg B — connectome-harmonic dynamics of WAND resting MEG (broadband, no bands).

Geometric eigenmodes first (Pang 2023 sibling of Atasoy 2016): a Desikan-68 region
graph weighted by centroid proximity (Gaussian, σ = median pairwise distance) →
connectome harmonics (graph-Laplacian eigenbasis, ordered by spatial frequency).
The broadband source MEG is projected onto the low-order harmonics, and the dynamics
are run **in the harmonic basis without any EEG band-filtering** — the eigenvalue
spectrum is the structure-derived "frequency" axis that replaces 2–4 / 4–8 / … Hz.

Central test (the doc): is the resting cycle a **solenoidal rotation among a small
set of low-order connectome harmonics**?  Project → broadband analytic envelope →
linear Langevin → the irreversible-circulation matrix α* = A_sol·Σ in the harmonic
basis; its largest antisymmetric entries name the harmonic *pair* the probability
current rotates between.

    WAND_OUT=/data/datasets/wand_src PYTHONPATH=src \\
      .venv-models/bin/python scripts/real_data/wand_connectome_harmonics.py
"""

import os
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from neurojax.spatial import connectome_harmonics, project_harmonics
from neurojax.dynamics import (
    fit_linear_langevin,
    langevin_entropy_production,
    langevin_solenoidal_part,
    langevin_solenoidal_frequency,
    solenoidal_circulation,
)

OUT = os.environ.get("WAND_OUT", "/data/datasets/wand_src")
SCR = os.environ.get("SCRATCH", "/tmp/claude-1000/-home-mhough-dev-neurojax/"
                     "a44a9232-d44b-4b8b-8812-560682c446fa/scratchpad")
ORACLE = os.path.join(os.path.dirname(__file__), "oracle_surrogates")
JULIA = os.path.expanduser("~/.juliaup/bin/julia")
FS = 250.0
ENV_FS = 50.0
N_HARM = 20            # low-order harmonics kept (excl. the constant/DC mode)
N_SURR = 40


def gaussian_graph(centroids):
    """Geometric region graph: W_ij = exp(-d_ij² / 2σ²), σ = median pairwise distance."""
    d2 = np.sum((centroids[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    sigma2 = np.median(d2[d2 > 0])
    W = np.exp(-d2 / sigma2)
    np.fill_diagonal(W, 0.0)
    return W, float(np.sqrt(sigma2))


def analytic_env(x):
    """Broadband amplitude envelope (|analytic signal|) along time (axis 0)."""
    T = x.shape[0]
    X = jnp.fft.fft(x, axis=0)
    h = jnp.zeros(T).at[0].set(1.0).at[1:(T + 1) // 2].set(2.0)
    h = h.at[T // 2].set(1.0) if T % 2 == 0 else h
    return jnp.abs(jnp.fft.ifft(X * h[:, None], axis=0))


def main():
    print("JAX backend:", jax.default_backend())
    cent = np.load(os.path.join(OUT, "desikan68_centroids.npy"))      # (68, 3)
    W, sigma = gaussian_graph(cent)
    evals, Phi = connectome_harmonics(jnp.asarray(W), normalized=False)
    evals, Phi = np.asarray(evals), np.asarray(Phi)
    print(f"geometric graph: 68 Desikan regions, Gaussian σ={sigma*1000:.1f} mm")
    print(f"harmonic eigenvalues (spatial freq): λ1..5 = "
          f"{np.round(evals[1:6], 3)} ... λ68 = {evals[-1]:.2f}")

    X = np.load(os.path.join(OUT, "parcels68.npy")).astype(np.float32)  # (T, 68)
    Phi_lo = Phi[:, 1:N_HARM + 1]                                      # drop DC mode
    a = np.asarray(project_harmonics(jnp.asarray(X), jnp.asarray(Phi_lo)))  # (T, N_HARM)
    env = np.asarray(analytic_env(jnp.asarray(a)))                     # broadband envelope
    ds = int(FS / ENV_FS)
    env = env[: (len(env) // ds) * ds].reshape(-1, ds, N_HARM).mean(1)  # -> ENV_FS
    env = env - env.mean(0)
    power = (env ** 2).mean(0)

    m = fit_linear_langevin(env, 1.0 / ENV_FS)
    eps = float(langevin_entropy_production(m))
    fsol = float(langevin_solenoidal_frequency(m))
    A_sol = np.asarray(langevin_solenoidal_part(m))
    alpha = np.asarray(solenoidal_circulation(jnp.asarray(A_sol), jnp.asarray(m.Sigma)))

    print(f"\n=====  Connectome-harmonic dynamics (broadband, {N_HARM} low-order modes)  =====")
    print(f"harmonic power (top 6 modes by power): "
          f"{[(int(i) + 1, round(float(power[i]), 2)) for i in np.argsort(power)[::-1][:6]]}")
    print(f"broadband EPR = {eps:.3f}   solenoidal rotation f_sol = {fsol:.3f} Hz")

    # which harmonic PAIR the probability current rotates between (largest |α*_ij|)
    au = np.abs(np.triu(alpha, 1))
    pairs = np.dstack(np.unravel_index(np.argsort(au.ravel())[::-1], au.shape))[0][:5]
    print("top rotational harmonic pairs (mode_i, mode_j, |α*|):")
    for i, j in pairs:
        print(f"    H{i+1:>2d} <-> H{j+1:>2d}   |α*|={au[i, j]:.4f}  "
              f"(λ={evals[i+1]:.2f},{evals[j+1]:.2f})")
    com = np.sum(au * (np.arange(N_HARM)[:, None] + np.arange(N_HARM)[None, :] + 2) / 2) / au.sum()
    print(f"  -> circulation centre-of-mass at harmonic order ≈ {com:.1f} "
          f"(of {N_HARM}); low ⇒ the cycle is rotation among LOW-order connectome harmonics.")

    # significance: broadband EPR vs a reversible IAAFT null (spectrum+marginal matched)
    fin, fout = os.path.join(SCR, "ch_in.npy"), os.path.join(SCR, "ch_out.npy")
    np.save(fin, env.astype(np.float64))
    subprocess.run([JULIA, "--startup-file=no",
                    os.path.join(ORACLE, "gen_surrogates.jl"), fin, fout,
                    str(N_SURR), "iaaft", "0"],
                   check=True, capture_output=True,
                   env=dict(os.environ, JULIA_PROJECT=ORACLE))
    S = np.load(fout)
    surr = np.array([float(langevin_entropy_production(
        fit_linear_langevin(S[i], 1.0 / ENV_FS))) for i in range(len(S))])
    z = (eps - surr.mean()) / (surr.std() + 1e-12)
    p = (np.sum(surr >= eps) + 1) / (len(surr) + 1)
    print(f"\nreversible IAAFT null: EPR={eps:.3f} vs null {surr.mean():.3f}±{surr.std():.3f}"
          f"  z={z:.2f}  p={p:.3f}  (harmonic-basis broken detailed balance)")


if __name__ == "__main__":
    main()
