#!/usr/bin/env python
"""Consciousness complexity benchmark driver.

    python scripts/consciousness_bench/run_bench.py --dataset propofol-chennu2016 \
        [--measures lz76_mean,permutation_entropy] [--epoch-s 10] [--max-epochs 12] \
        [--n-channels 20] [--n-surrogates 20] [--n-jobs 20] [--no-decompose]

Writes to $CB_OUT (default /data/datasets/consciousness-bench)/<dataset>/:
scores.csv, contrast_<a>_vs_<b>.csv, decomposition_<a>_vs_<b>.csv, and a
params.json naming every estimator/parameter used.
"""

import argparse
import json
import os
import time
from pathlib import Path

import jax
import numpy as np

from neurojax.analysis.nltsa import MEASURES
from neurojax.analysis.spectral_baselines import SPECTRAL_MEASURES
from neurojax.bench.consciousness_bench import contrast, decompose_contrast, score_recordings
from neurojax.io.consciousness_bench import LOADERS

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=list(LOADERS))
ap.add_argument("--measures", default="all", help="all | nltsa | spectral | comma list")
ap.add_argument("--tag", default="", help="suffix for output files, e.g. spectral")
ap.add_argument("--epoch-s", type=float, default=10.0)
ap.add_argument("--max-epochs", type=int, default=12)
ap.add_argument("--n-channels", type=int, default=0, help="evenly spaced channel subset (0 = all)")
ap.add_argument("--n-surrogates", type=int, default=20)
ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
ap.add_argument("--subjects", default="")
ap.add_argument("--no-decompose", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

out = Path(os.environ.get("CB_OUT", "/data/datasets/consciousness-bench")) / args.dataset
out.mkdir(parents=True, exist_ok=True)
ALL = {**MEASURES, **SPECTRAL_MEASURES}
measures = {"all": ALL, "nltsa": MEASURES, "spectral": SPECTRAL_MEASURES}.get(
    args.measures) or {k: ALL[k] for k in args.measures.split(",")}
tag = f"_{args.tag}" if args.tag else ""
loader, contrasts = LOADERS[args.dataset]

t0 = time.time()
kw = dict(subjects=args.subjects.split(",")) if args.subjects else {}
recs = loader(**kw)
if args.n_channels:
    for r in recs:
        idx = np.linspace(0, r.data.shape[0] - 1, args.n_channels).round().astype(int)
        r.data, r.channels = r.data[idx], [r.channels[i] for i in idx]
print(f"[load] {len(recs)} recordings, {sorted({r.condition for r in recs})}, "
      f"{recs[0].data.shape[0]} ch @ {recs[0].fs} Hz, {time.time()-t0:.0f}s", flush=True)

json.dump(dict(vars(args), measures={k: dict(family=m.family, params=m.params, reference=m.reference,
                                            multivariate=m.multivariate) for k, m in measures.items()},
               n_recordings=len(recs), fs=recs[0].fs, n_channels=recs[0].data.shape[0]),
          open(out / f"params{tag}.json", "w"), indent=1, default=str)

t0 = time.time()
df = score_recordings(recs, measures, args.epoch_s, args.max_epochs, n_jobs=args.n_jobs)
df.to_csv(out / f"scores{tag}.csv", index=False)
print(f"[score] {len(df)} rows in {time.time()-t0:.0f}s", flush=True)

for a, b in contrasts:
    if not ({a, b} <= set(df.condition)):
        continue
    c = contrast(df, a, b)
    c.to_csv(out / f"contrast_{a}_vs_{b}{tag}.csv", index=False)
    print(f"\n[contrast] {b} - {a} (n={int(c.n.max())}):")
    print(c[["measure", "cohen_d", "p_wilcoxon", "auc", "frac_increase"]].to_string(index=False, float_format="%.3f"))
    if args.no_decompose:
        continue
    t0 = time.time()
    d = decompose_contrast(recs, measures, a, b, jax.random.PRNGKey(args.seed), args.epoch_s,
                           args.max_epochs, args.n_surrogates, n_jobs=args.n_jobs)
    d.to_csv(out / f"decomposition_{a}_vs_{b}{tag}.csv", index=False)
    d.attrs["per_subject"].to_csv(out / f"decomposition_{a}_vs_{b}_per_subject{tag}.csv", index=False)
    print(f"[decompose] {b} - {a}, {args.n_surrogates} surrogates, {time.time()-t0:.0f}s:")
    print(d[["measure", "total", "spectral", "phasic", "spectral_fraction", "phasic_z"]]
          .to_string(index=False, float_format="%.3f"))
print(f"[done] -> {out}")
