#!/usr/bin/env python3
"""Compare HBN's 2018 FreeSurfer v6.0.0 against a modern FreeSurfer (via fMRIPrep).

Quantifies how much the stale precomputed FS6 differs from a fresh recon-all, on
the morphometry that EMEG Recon (source space, parcellation) and structural
brain-age care about: per-region cortical thickness + surface area (aparc.stats)
and subcortical volumes (aseg.stats). Region-level stats are directly comparable
across FS versions without surface re-registration.

Usage (after the Legion fMRIPrep run finishes):
    python compare_fs6_vs_modern.py \
        --old-root  ~/hbn_fmriprep/freesurfer6 \
        --new-root  ~/hbn_fmriprep/derivatives/sourcedata/freesurfer \
        --subjects  NDARAD481FXF NDARAE199TDD \
        --out       fs6_vs_modern.csv

Old FS6 subject dir = <old-root>/<NDAR>; fMRIPrep FS dir = <new-root>/sub-<NDAR>.
Deps: numpy only.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


def _parse_aparc(path: str) -> dict[str, dict[str, float]]:
    """region -> {area, grayvol, thickness} from an [lh|rh].aparc.stats."""
    out: dict[str, dict[str, float]] = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.split()
            # StructName NumVert SurfArea GrayVol ThickAvg ThickStd ...
            out[f[0]] = {
                "area": float(f[2]),
                "grayvol": float(f[3]),
                "thickness": float(f[4]),
            }
    return out


def _parse_aseg(path: str) -> dict[str, float]:
    """structure -> Volume_mm3 from aseg.stats."""
    out: dict[str, float] = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.split()
            # Index SegId NVoxels Volume_mm3 StructName ...
            out[f[4]] = float(f[3])
    return out


def _stats_dir(root: str, sub: str, new: bool) -> str:
    return os.path.join(root, f"sub-{sub}" if new else sub, "stats")


def _agg(old: dict, new: dict, key=None):
    """Paired (old, new) arrays over shared regions; key extracts a scalar."""
    shared = sorted(set(old) & set(new))
    o = np.array([old[r][key] if key else old[r] for r in shared], float)
    n = np.array([new[r][key] if key else new[r] for r in shared], float)
    return shared, o, n


def _summary(o, n):
    if len(o) < 2:
        return {"n": len(o), "r": float("nan"), "mean_abs_pct": float("nan")}
    r = float(np.corrcoef(o, n)[0, 1])
    pct = 100.0 * np.abs(n - o) / np.maximum(np.abs(o), 1e-9)
    return {"n": len(o), "r": r, "mean_abs_pct": float(np.mean(pct))}


def compare_subject(old_root, new_root, sub):
    so, sn = _stats_dir(old_root, sub, False), _stats_dir(new_root, sub, True)
    aparc_o = {**_parse_aparc(f"{so}/lh.aparc.stats"), **{f"rh_{k}": v for k, v in _parse_aparc(f"{so}/rh.aparc.stats").items()}}
    aparc_n = {**_parse_aparc(f"{sn}/lh.aparc.stats"), **{f"rh_{k}": v for k, v in _parse_aparc(f"{sn}/rh.aparc.stats").items()}}
    aseg_o, aseg_n = _parse_aseg(f"{so}/aseg.stats"), _parse_aseg(f"{sn}/aseg.stats")

    rows = []
    _, to, tn = _agg(aparc_o, aparc_n, "thickness")
    _, ao, an = _agg(aparc_o, aparc_n, "area")
    vregions, vo, vn = _agg(aseg_o, aseg_n)
    res = {
        "subject": sub,
        "thickness": _summary(to, tn),
        "area": _summary(ao, an),
        "aseg_volume": _summary(vo, vn),
    }
    for r, oo, nn in zip(*_agg(aparc_o, aparc_n, "thickness")):
        rows.append((sub, "thickness", r, oo, nn))
    for r, oo, nn in zip(vregions, vo, vn):
        rows.append((sub, "aseg_volume", r, oo, nn))
    return res, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-root", required=True)
    p.add_argument("--new-root", required=True)
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--out", default="fs6_vs_modern.csv")
    a = p.parse_args()

    all_rows = []
    print(f"{'subject':16} {'metric':12} {'n':>4} {'r(old,new)':>11} {'mean|%diff|':>11}")
    for sub in a.subjects:
        res, rows = compare_subject(a.old_root, a.new_root, sub)
        all_rows += rows
        for m in ("thickness", "area", "aseg_volume"):
            s = res[m]
            print(f"{sub:16} {m:12} {s['n']:>4} {s['r']:>11.3f} {s['mean_abs_pct']:>10.2f}%")

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "metric", "region", "fs6", "modern"])
        w.writerows(all_rows)
    print(f"\nper-region table -> {a.out}")
    print("Interpretation: r near 1 + small mean|%diff| => FS6 adequate for EMEG "
          "Recon; large thickness/area drift => prefer modern recon for the cohort.")


if __name__ == "__main__":
    main()
