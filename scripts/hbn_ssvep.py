#!/usr/bin/env python3
"""Steady-state response in the HBN surround-suppression task: contrast response and surround suppression.

The four foreground discs flash; the phase-locked (trial-averaged) spectrum
shows the tag at 25 Hz with a harmonic at 50 Hz in subjects with a clear
response, and 60 Hz line noise everywhere, so the evoked spectrum is
computed on 60 Hz-notched, average-referenced posterior channels and the
response is the SNR of the 25 Hz bin (and 50 Hz) against its flanks.  Rows
are subject x condition pooled over the two runs.

    python scripts/hbn_ssvep.py --root ~/Workspace/data/ds005505 --out results/bsd/hbn_ssvep.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

PICKS = ["E70", "E75", "E83", "E71", "E76", "E74", "E82"]
TAGS = (25.0, 50.0)


def snr_at(f, S, f0, half=1, flank=8, gap=2):
    i = int(np.argmin(np.abs(f - f0)))
    sig = S[i - half:i + half + 1].mean()
    fl = np.r_[S[i - half - gap - flank:i - half - gap], S[i + half + gap + 1:i + half + gap + 1 + flank]].mean()
    return float(sig / fl), float(sig)


def main():
    import mne
    mne.set_log_level("ERROR")
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    trials = {}
    for setf in sorted(glob.glob(os.path.join(a.root, "sub-*", "eeg", "*task-surroundSupp_run-*_eeg.set"))):
        sub = os.path.basename(setf).split("_")[0]
        ev = list(csv.DictReader(open(setf.replace("_eeg.set", "_events.tsv")), delimiter="\t"))
        raw = mne.io.read_raw_eeglab(setf, preload=True)
        raw.set_eeg_reference("average", projection=False)
        raw.notch_filter([60.0], verbose=False)
        fs = raw.info["sfreq"]
        x = raw.get_data(picks=[c for c in PICKS if c in raw.ch_names]).mean(axis=0)
        n = int(2.0 * fs)
        for e in ev:
            if e["value"] != "stim_ON":
                continue
            key = f"bg{e['background']}_fg{e['foreground_contrast']}"
            s = int((float(e["onset"]) + 0.4) * fs)  # skip the onset transient
            seg = x[s:s + n]
            if seg.size == n:
                trials.setdefault((sub, key), []).append(seg - seg.mean())
    f = np.fft.rfftfreq(n, 1 / fs)
    rows = []
    for (sub, key), segs in sorted(trials.items()):
        X = np.stack(segs)
        win = np.hanning(n)
        evoked = np.abs(np.fft.rfft(X.mean(axis=0) * win)) ** 2
        induced = (np.abs(np.fft.rfft(X * win, axis=1)) ** 2).mean(axis=0)
        row = {"subject": sub, "condition": key, "background": key[2], "fg_contrast": float(key.split("_fg")[1]), "n_trials": len(segs)}
        for tag in TAGS:
            se, pe = snr_at(f, evoked, tag); si, pi = snr_at(f, induced, tag)
            row[f"evoked_snr_{int(tag)}"] = se; row[f"evoked_power_{int(tag)}"] = pe
            row[f"induced_snr_{int(tag)}"] = si
        # alpha power during stimulation, for the induced side
        band = (f >= 8) & (f <= 13)
        row["alpha_power"] = float(np.log(induced[band].mean()))
        rows.append(row)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=0)
    # condition table
    conds = sorted({r["condition"] for r in rows}, key=lambda c: (c[2], float(c.split("_fg")[1])))
    print(f"{'condition':12s} n   evoked SNR25 median [IQR]   evoked SNR50   induced SNR25   trials")
    for c in conds:
        sub = [r for r in rows if r["condition"] == c]
        e25 = np.array([r["evoked_snr_25"] for r in sub]); e50 = np.array([r["evoked_snr_50"] for r in sub]); i25 = np.array([r["induced_snr_25"] for r in sub])
        print(f"{c:12s} {len(sub):2d}   {np.median(e25):5.2f} [{np.percentile(e25,25):4.2f},{np.percentile(e25,75):5.2f}]      {np.median(e50):5.2f}          {np.median(i25):5.2f}        {int(np.median([r['n_trials'] for r in sub]))}")
    # suppression index per subject at full contrast: SNR(bg0_fg1) / SNR(bg1_fg1)
    subs = sorted({r["subject"] for r in rows})
    si = []
    for s in subs:
        g = {r["condition"]: r for r in rows if r["subject"] == s}
        if "bg0_fg1" in g and "bg1_fg1" in g:
            si.append(np.log(g["bg0_fg1"]["evoked_snr_25"] / g["bg1_fg1"]["evoked_snr_25"]))
    si = np.array(si)
    print(f"\nsurround suppression log(SNR no-bg / SNR bg) at full contrast: mean {si.mean():+.2f} sd {si.std(ddof=1):.2f} n={len(si)} t={si.mean()/(si.std(ddof=1)/np.sqrt(len(si))):+.2f}")
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
