#!/usr/bin/env python3
"""Cohort-level WAND MEG directed connectivity.

Runs the individual recon (run_wand_recon.py) on every FreeSurfer-complete WAND
subject that also has resting MEG -- one at a time (memory-considerate on the
shared DGX), idempotent on the per-subject ``~/wand_recon_<sub>.npz`` -- then
aggregates into a GROUP directed-connectivity map: the mean PDC across subjects
plus the mean leakage, and the most consistent leakage-clean directed edges.

Usage:  python run_wand_cohort.py            # all ready subjects
        python run_wand_cohort.py --no-run   # only aggregate existing npz
The FS cohort builds via the Legion queue (reference_wand_freesurfer_recipe);
re-run this as more subjects complete -- it just picks up the new ones.
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

FS = "/data/raw/wand/derivatives/freesurfer"
RECON = os.path.expanduser("~/dev/neurojax/scripts/run_wand_recon.py")
PY = os.path.expanduser("~/dev/neurojax/.venv-models/bin/python")


def ready_subjects():
    subs = []
    for done in sorted(glob.glob(f"{FS}/sub-*_ses-02/scripts/recon-all.done")):
        s = os.path.basename(os.path.dirname(os.path.dirname(done))).replace("_ses-02", "")
        ds = f"/data/raw/wand/{s}/ses-01/meg/{s}_ses-01_task-resting.ds"
        if os.path.isdir(ds):
            subs.append(s)
    return subs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-run", action="store_true", help="aggregate only")
    a = ap.parse_args()
    subs = ready_subjects()
    print(f"{len(subs)} FS+restingMEG subjects: {subs}", flush=True)

    if not a.no_run:
        env = {**os.environ, "PYTHONNOUSERSITE": "1"}
        for s in subs:
            out = os.path.expanduser(f"~/wand_recon_{s}.npz")
            if os.path.exists(out):
                print(f"  {s}: npz exists, skipping", flush=True)
                continue
            print(f"  {s}: running individual recon ...", flush=True)
            subprocess.run([PY, RECON, s], check=False, env=env)

    pdcs, leaks, parcels, freqs, used = [], [], None, None, []
    for s in subs:
        out = os.path.expanduser(f"~/wand_recon_{s}.npz")
        if not os.path.exists(out):
            continue
        d = np.load(out, allow_pickle=True)
        pdcs.append(d["pdc"]); leaks.append(d["leakage"])
        parcels, freqs = d["parcels"], d["freqs"]; used.append(s)
    if not pdcs:
        sys.exit("no per-subject recon outputs found")

    G, L = np.mean(pdcs, 0), np.mean(leaks, 0)
    names = list(parcels)
    alpha = (freqs >= 8) & (freqs <= 12)
    P = G[alpha].mean(0); np.fill_diagonal(P, 0.0)
    idx = np.dstack(np.unravel_index(np.argsort(P, axis=None)[::-1], P.shape))[0]
    print(f"\nGROUP (n={len(used)}) top alpha-band directed edges (leakage-clean):",
          flush=True)
    shown = 0
    for i, j in idx:
        if L[i, j] > 0.7:
            continue
        print(f"  {names[j]:>22} -> {names[i]:<22} PDC={P[i, j]:.3f} leak={L[i, j]:.2f}",
              flush=True)
        shown += 1
        if shown >= 12:
            break
    outp = os.path.expanduser("~/wand_cohort_connectivity.npz")
    np.savez_compressed(outp, pdc=G, leakage=L, freqs=freqs, parcels=parcels,
                        n=len(used), subjects=np.array(used))
    print(f"\n[done] group map (n={len(used)}) -> {outp}", flush=True)


if __name__ == "__main__":
    main()
