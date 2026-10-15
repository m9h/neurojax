#!/usr/bin/env python3
"""Summarise bsd_batch.py output: model selection per condition, parameters, paired contrasts.

    python scripts/bsd_summarise.py results/bsd/hbn_rest.json
    python scripts/bsd_summarise.py results/bsd/hbn_ss.json --ssvep
"""

from __future__ import annotations

import argparse
import json
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from neurojax.bsd import random_effects_bms, fixed_effects_bms  # noqa: E402

LINEAR = {"pw0", "pw1", "offset", "cf0", "cf1"}

PARAM_KEYS = {
    "specparam": ("exponent", "cf0", "pw0", "bw0"),
    "specparam_knee": ("exponent", "knee", "cf0", "pw0"),
    "specparam2": ("exponent", "knee", "cf0", "pw0", "cf1", "pw1"),
    "liley": ("f0", "gamma_median", "spread"),
    "liley_mix": ("f_alpha", "gamma_alpha", "peak_alpha", "f_bg", "gamma_bg_median", "spread_bg"),
    "xialpha": ("f_alpha", "lam_alpha", "nu_alpha", "lam_xi", "nu_xi"),
}


def models_of(rows):
    return [k for k in rows[0]["fits"] if isinstance(rows[0]["fits"][k], dict) and "F" in rows[0]["fits"][k]]


def F_matrix(rows, models):
    F = np.array([[r["fits"][m]["F"] for m in models] for r in rows], dtype=float)
    F = np.where(np.isfinite(F), F, -1e9)
    return F


def summarise_condition(rows, models, label):
    F = F_matrix(rows, models)
    wins = np.bincount(F.argmax(axis=1), minlength=len(models))
    total, p_ffx = fixed_effects_bms(F)
    line = f"{label:22s} n={len(rows):3d} wins " + " ".join(f"{m}:{w}" for m, w in zip(models, wins))
    if len(rows) >= 3:
        rfx = random_effects_bms(F, n_samples=20000)
        line += " | RFX xp " + " ".join(f"{m}:{x:.2f}" for m, x in zip(models, rfx.protected_exceedance)) + f" BOR {rfx.bor:.2f}"
    line += " | FFX p " + " ".join(f"{m}:{x:.2f}" for m, x in zip(models, p_ffx))
    print(line)


def param_table(rows, model, label):
    keys = PARAM_KEYS.get(model, ())
    vals = {k: [r["fits"][model]["params"].get(k, np.nan) for r in rows if np.isfinite(r["fits"][model]["F"])] for k in keys}
    print(f"  {model:14s} {label:12s} " + "  ".join(f"{k}={np.nanmedian(v):.3g} [{np.nanpercentile(v,25):.3g},{np.nanpercentile(v,75):.3g}]" for k, v in vals.items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--ssvep", action="store_true")
    p.add_argument("--pair", nargs=2, default=None, help="two condition labels to contrast within subject (e.g. EC EO)")
    a = p.parse_args()
    rows = json.load(open(a.path))
    rows = [r for r in rows if "fits" in r]
    models = models_of(rows)
    conds = sorted({r["condition"] for r in rows})
    print(f"{a.path}: {len(rows)} rows, {len({r['subject'] for r in rows})} subjects, conditions {conds}\nmodels {models}\n")
    print("== model selection ==")
    summarise_condition(rows, models, "all")
    for c in conds:
        summarise_condition([r for r in rows if r["condition"] == c], models, c)
    print("\n== parameters (median [IQR]) ==")
    for c in conds:
        sub = [r for r in rows if r["condition"] == c]
        for m in models:
            param_table(sub, m, c)
    pair = a.pair or (("EC", "EO") if {"EC", "EO"} <= set(conds) else None)
    if pair:
        A, B = pair
        print(f"\n== within-subject {A} minus {B} ==")
        subs = sorted({r["subject"] for r in rows})
        for m in models:
            for k in PARAM_KEYS.get(m, ()):
                d = []
                for s in subs:
                    ra = [r for r in rows if r["subject"] == s and r["condition"] == A]
                    rb = [r for r in rows if r["subject"] == s and r["condition"] == B]
                    if ra and rb and np.isfinite(ra[0]["fits"][m]["F"]) and np.isfinite(rb[0]["fits"][m]["F"]):
                        va, vb = ra[0]["fits"][m]["params"][k], rb[0]["fits"][m]["params"][k]
                        # positive-definite parameters are contrasted as log ratios
                        d.append(np.log(va / vb) if (k not in LINEAR and va > 0 and vb > 0) else va - vb)
                if d:
                    d = np.asarray(d); t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan
                    unit = "diff" if k in LINEAR else "log-ratio"
                    print(f"  {m:14s} {k:16s} {unit:9s} mean {d.mean():+.3g} sd {d.std(ddof=1):.3g} n={len(d)} t={t:+.2f}")
        dF = []
        for s in subs:
            ra = [r for r in rows if r["subject"] == s and r["condition"] == A]
            rb = [r for r in rows if r["subject"] == s and r["condition"] == B]
            if ra and rb:
                dF.append([ra[0]["fits"][m]["F"] - rb[0]["fits"][m]["F"] for m in models])
        if dF:
            dF = np.asarray(dF); print("  mean dF per model:", " ".join(f"{m}:{v:+.1f}" for m, v in zip(models, np.nanmean(dF, axis=0))))
    if a.ssvep:
        print("\n== steady-state response ==")
        # pooled tag detection: mean SNR spectrum over all foreground>0 rows
        f = np.asarray(rows[0]["freqs"])
        acc = np.zeros_like(f)
        n = 0
        for r in rows:
            if "fg0_" in r["condition"] or r["condition"].endswith("_fg0"):
                continue
            y = np.exp(np.asarray(r["log_psd"]))
            snr = np.ones_like(y)
            for i in range(7, len(f) - 7):
                snr[i] = y[i] / np.r_[y[i-7:i-1], y[i+2:i+8]].mean()
            acc += np.log(snr); n += 1
        acc /= max(n, 1)
        top = np.argsort(-acc)[:5]
        print("  pooled SNR peaks (Hz, mean log SNR):", [(round(float(f[i]), 2), round(float(acc[i]), 3)) for i in top])
        by = {}
        for r in rows:
            by.setdefault(r["condition"], []).append(r.get("ssvep_snr", np.nan))
        for c, v in sorted(by.items()):
            print(f"  {c:12s} per-row tag SNR median {np.nanmedian(v):.2f} (n={len(v)})")


if __name__ == "__main__":
    main()
