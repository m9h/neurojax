#!/usr/bin/env python3
"""First REAL individualized HBN EMEG source-connectivity map.

sub-NDARAD481FXF resting-state EEG -> fsaverage BEM forward -> MNE inverse ->
aparc parcel time courses -> neurojax directed_source_connectivity (PDC/DTF +
leakage). Template (fsaverage) head model for this first pass; the individual FS
(validated FS6-adequate) is the refinement.
"""
import os
import sys

import numpy as np
import jax.numpy as jnp
import mne
from mne.datasets import fetch_fsaverage

sys.path.insert(0, os.path.expanduser("~/dev/neurojax/src"))
from neurojax.analysis.source_connectivity import directed_source_connectivity

mne.set_log_level("WARNING")
# arg1: HBN RestingState .set ; arg2: output dir (default: alongside the .set)
SET = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/hbn_eeg/sub-NDARAD481FXF_task-RestingState_eeg.set")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(SET))

print("[1] read EEG", flush=True)
raw = mne.io.read_raw_eeglab(SET, preload=True)
raw.pick_types(eeg=True)
print(f"    {len(raw.ch_names)} ch, sfreq={raw.info['sfreq']}, "
      f"{raw.times[-1]:.0f}s; first ch: {raw.ch_names[:3]}", flush=True)
# HBN = EGI 128 net + Cz reference (129 ch); the 129 montage includes Cz.
raw.set_montage("GSN-HydroCel-129", match_case=False, on_missing="raise")
n_pos = sum(1 for d in raw.info["dig"] or [] if d["kind"] == 3)
print(f"    montage set; {n_pos} electrode positions", flush=True)
raw.set_eeg_reference("average", projection=True)
raw.filter(1.0, 45.0)
raw.resample(125)
raw.crop(tmax=min(120.0, raw.times[-1]))  # 2 min is plenty for resting connectivity

print("[2] fsaverage forward", flush=True)
fs_dir = fetch_fsaverage(verbose=False)
subjects_dir = os.path.dirname(fs_dir)
src = mne.read_source_spaces(f"{fs_dir}/bem/fsaverage-ico-5-src.fif")
bem = f"{fs_dir}/bem/fsaverage-5120-5120-5120-bem-sol.fif"
fwd = mne.make_forward_solution(raw.info, trans="fsaverage", src=src, bem=bem,
                                eeg=True, meg=False, verbose=False)
print(f"    forward: {fwd['nsource']} sources x {fwd['nchan']} sensors", flush=True)

print("[3] MNE inverse -> aparc parcels", flush=True)
cov = mne.make_ad_hoc_cov(raw.info)
inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, verbose=False)
stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=1.0 / 9.0,
                                         method="MNE", verbose=False)
labels = mne.read_labels_from_annot("fsaverage", "aparc", subjects_dir=subjects_dir,
                                    verbose=False)
labels = [l for l in labels if "unknown" not in l.name.lower()]
ltc = mne.extract_label_time_course(stc, labels, src, mode="mean_flip", verbose=False)
print(f"    {ltc.shape[0]} parcels x {ltc.shape[1]} samples", flush=True)

print("[4] neurojax directed source connectivity (PDC/DTF)", flush=True)
fs = raw.info["sfreq"]
freqs = jnp.linspace(1.0, 45.0, 45)
out = directed_source_connectivity(jnp.asarray(ltc), freqs, fs=fs, order=6)
pdc = np.asarray(out["pdc"])  # (45, n_parcels, n_parcels)

# alpha-band (8-12 Hz) mean directed connectivity, strongest source->target edges
alpha = (np.asarray(freqs) >= 8) & (np.asarray(freqs) <= 12)
P = pdc[alpha].mean(0)
np.fill_diagonal(P, 0.0)
names = [l.name for l in labels]
idx = np.dstack(np.unravel_index(np.argsort(P, axis=None)[::-1], P.shape))[0][:8]
print("    top alpha-band directed edges (source -> target, PDC):", flush=True)
for i, j in idx:
    print(f"      {names[j]:>22} -> {names[i]:<22} {P[i, j]:.3f}", flush=True)

outpath = os.path.join(OUT, "hbn_source_connectivity.npz")
np.savez_compressed(outpath, pdc=pdc, dtf=np.asarray(out["dtf"]),
                    freqs=np.asarray(freqs), parcels=np.array(names))
print(f"[done] -> {outpath}", flush=True)
