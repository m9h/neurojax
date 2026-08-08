#!/usr/bin/env python3
"""Replicate the mouse arm of Berjaga-Buisan et al. (2026) with open EBRAINS data.

Tests the paper's Figure 2B claim: FDT violations, computed from *spontaneous* activity via a
fitted OU model, discriminate graded isoflurane anesthesia (they report Spearman rho = 0.885).

The PCI arm (Fig 2A/2C) is NOT reproducible here: the EBRAINS release contains only spontaneous
recordings (all files `stim-SPN`; the Stim event channel is present but empty), and PCI requires
evoked responses to cortical perturbation.

    python -m neurojax.thermo.replicate_ebrains --restarts 4 --steps 300
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from .ebrains import fetch, list_recordings, load_smr, preprocess
from .fdt import entropy_production, fdt_violation, time_reversal_asymmetry
from .gec import empirical_observables, fit_gec

CACHE = os.environ.get("EBRAINS_CACHE", "/tmp/ebdata")


def analyze(path, restarts, steps, lag_s=0.1, lr=2e-2, seed=0):
    X, fs = load_smr(path)
    Xf, fs2 = preprocess(X, fs)
    lag = max(1, int(round(lag_s * fs2)))
    FC, FS = empirical_observables(jnp.asarray(Xf), lag)
    tau = lag / fs2
    n = FC.shape[0]
    Q = jnp.eye(n)
    taus = jnp.linspace(0.02, 0.4, 6)

    # all restarts as ONE vmapped computation (the axis the reference farms out to SLURM)
    from .gec import fit_gec_restarts
    As, finals = fit_gec_restarts(FC, FS, tau, jax.random.PRNGKey(seed),
                                  n_restarts=restarts, n_steps=steps, lr=lr)
    finals = np.asarray(finals)
    finals = np.where(np.isfinite(finals), finals, np.inf)
    k = int(np.argmin(finals))
    L, A = float(finals[k]), As[k]
    return dict(
        loss=L,
        fdt_violation=float(fdt_violation(A, Q, taus)),
        entropy_production=float(entropy_production(A, Q)),
        time_rev_asym=float(time_reversal_asymmetry(A, Q, tau)),
        emp_fs_asym=float(jnp.linalg.norm(FS - FS.T) / jnp.linalg.norm(FS)),
        n_channels=int(n), fs=float(fs2), tau_ms=float(tau * 1000),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(CACHE, "fdt_results.json"))
    a = ap.parse_args()

    recs = list_recordings()
    print(f"{len(recs)} recordings (8 mice x 3 isoflurane levels)", flush=True)
    rows = []
    for i, (sub, iso, path) in enumerate(recs, 1):
        t0 = time.time()
        local = fetch(path, CACHE)
        try:
            r = analyze(local, a.restarts, a.steps)
        except Exception as e:
            print(f"  [{i}/{len(recs)}] sub-{sub} ISO{iso:03d}  FAILED {e!r}", flush=True)
            continue
        r.update(subject=sub, iso=iso)
        rows.append(r)
        print(f"  [{i}/{len(recs)}] sub-{sub} ISO{iso:03d}  FDT={r['fdt_violation']:.4f} "
              f"S={r['entropy_production']:.2f} loss={r['loss']:.1f} ({time.time()-t0:.0f}s)",
              flush=True)

    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}  ({len(rows)}/{len(recs)} succeeded)")
    summarize(rows)


def summarize(rows):
    from scipy.stats import spearmanr, wilcoxon
    import collections
    # rank isoflurane level within each mouse: 0 = lightest, 2 = deepest
    bysub = collections.defaultdict(list)
    for r in rows:
        bysub[r["subject"]].append(r)
    ranks, viols, sers = [], [], []
    for s, rs in bysub.items():
        for k, r in enumerate(sorted(rs, key=lambda x: x["iso"])):
            r["depth_rank"] = k
            ranks.append(k); viols.append(r["fdt_violation"]); sers.append(r["entropy_production"])
    print(f"\n{'depth':>8}{'n':>4}{'FDT violation (mean+-sd)':>30}{'entropy prod':>18}")
    for k, name in enumerate(("light", "mid", "deep")):
        v = [x for x, rk in zip(viols, ranks) if rk == k]
        e = [x for x, rk in zip(sers, ranks) if rk == k]
        if v:
            print(f"{name:>8}{len(v):>4}{np.mean(v):>20.4f} +- {np.std(v):<6.4f}{np.mean(e):>14.2f}")
    if len(set(ranks)) > 1:
        rho, p = spearmanr(ranks, viols)
        rho_s, p_s = spearmanr(ranks, sers)
        print(f"\nSpearman(anesthesia depth, FDT violation) = {rho:+.3f}  p={p:.4g}")
        print(f"Spearman(anesthesia depth, entropy prod.)  = {rho_s:+.3f}  p={p_s:.4g}")
        print("\npaper (Fig 2B): FDT violations DECREASE with anesthesia depth, rho = 0.885")
        print(f"our sign: {'MATCHES (negative rho = decreasing)' if rho < 0 else 'OPPOSITE'}")


if __name__ == "__main__":
    main()
