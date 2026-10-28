#!/usr/bin/env python
"""Sensitivity of the benchmark to low-frequency handling × epoch length.

    python scripts/consciousness_bench/run_sensitivity.py --dataset propofol-chennu2016 --cond-a baseline --cond-b moderate

Recordings are loaded *without* the 0.5 Hz high-pass (only the 45 Hz
low-pass and average reference) so every alternative in PREPROCS starts
from the same broadband data. Chennu 2016 propofol was already 0.5-45 Hz
filtered by the authors, so for that dataset cutoffs below 0.5 Hz are moot.
Writes $CB_OUT/<dataset>/sensitivity_<a>_vs_<b>.csv and a figure.
"""

import argparse
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neurojax.analysis.nltsa import MEASURES
from neurojax.bench.sensitivity import PREPROCS, sensitivity_table
from neurojax.io.consciousness_bench import LOADERS

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=list(LOADERS))
ap.add_argument("--cond-a", required=True)
ap.add_argument("--cond-b", required=True)
ap.add_argument("--n-channels", type=int, default=10)
ap.add_argument("--total-s", type=float, default=60.0)
ap.add_argument("--epochs", default="2,5,10,30")
ap.add_argument("--preprocs", default="all")
ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
args = ap.parse_args()

out = Path(os.environ.get("CB_OUT", "/data/datasets/consciousness-bench")) / args.dataset
loader, _ = LOADERS[args.dataset]
t0 = time.time()
recs = loader(l_freq=None)
for r in recs:
    idx = np.linspace(0, r.data.shape[0] - 1, args.n_channels).round().astype(int)
    r.data, r.channels = r.data[idx], [r.channels[i] for i in idx]
print(f"[load] {len(recs)} recordings (no HPF), {time.time()-t0:.0f}s", flush=True)
pre = PREPROCS if args.preprocs == "all" else {k: PREPROCS[k] for k in args.preprocs.split(",")}
eps = tuple(float(e) for e in args.epochs.split(","))

t0 = time.time()
T = sensitivity_table(recs, MEASURES, args.cond_a, args.cond_b, pre, eps, args.total_s, n_jobs=args.n_jobs)
tag = f"{args.cond_a}_vs_{args.cond_b}"
T.to_csv(out / f"sensitivity_{tag}.csv", index=False)
print(f"[sensitivity] {len(T)} rows in {time.time()-t0:.0f}s; reference {T.attrs['reference']}", flush=True)

D = T.pivot_table(index="measure", columns=["preproc", "epoch_s"], values="cohen_d")
R = T.pivot_table(index="measure", columns=["preproc", "epoch_s"], values="rho_ref")
pd.set_option("display.width", 250)
print("\nCohen's d by preprocessing (epoch 10 s):")
print(D.xs(10.0, axis=1, level="epoch_s").to_string(float_format="%.2f"))
print("\nCohen's d by epoch length (hpf_0.5):")
print(D["hpf_0.5"].to_string(float_format="%.2f"))
print("\nrank stability of subject changes vs reference (rho), epoch 10 s:")
print(R.xs(10.0, axis=1, level="epoch_s").to_string(float_format="%.2f"))
summ = T.groupby(["preproc", "epoch_s"]).agg(mean_abs_d=("cohen_d", lambda v: np.nanmean(np.abs(v))),
                                             mean_rho=("rho_ref", "mean"), mantel=("mantel_ref", "first")).reset_index()
print("\nper-setting summary:")
print(summ.to_string(index=False, float_format="%.2f"))

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for ax, M, t in zip(axes, (D, R), ("Cohen's d", "Spearman rho of subject changes vs reference")):
    im = ax.imshow(M.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-max(1, np.nanmax(np.abs(M.to_numpy()))) if t.startswith("C") else -1,
                   vmax=max(1, np.nanmax(np.abs(M.to_numpy()))) if t.startswith("C") else 1)
    ax.set_yticks(range(len(M.index))); ax.set_yticklabels(M.index, fontsize=7)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([f"{p}\n{e:g}s" for p, e in M.columns], rotation=90, fontsize=6)
    ax.set_title(f"{t}: {args.dataset} {args.cond_b} − {args.cond_a}", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7)
fig.tight_layout()
fig.savefig(out / f"sensitivity_{tag}.png", dpi=130)
print(f"[done] -> {out}/sensitivity_{tag}.csv")
