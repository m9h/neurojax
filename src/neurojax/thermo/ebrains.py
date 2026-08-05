"""Loader for the EBRAINS mouse ECoG dataset used by Berjaga-Buisan et al. (2026).

Dataset: "Propagation modes of slow waves in mouse cortex" (Sanchez-Vives, IDIBAPS / HBP)
         https://doi.org/10.25493/WKA8-Q4T   CC BY-NC-SA 4.0, free access
Contents: 8 mice x 3 isoflurane levels = 24 recordings, 32-channel ECoG, ~5 kHz, ~228 s,
          Spike2 (.smr). ALL recordings are spontaneous (`stim-SPN`; the Stim event channel is
          present but empty) -- so FDT violations are computable, PCI is not.

Preprocessing follows the paper: delta band (0.5-4 Hz) for the FDT analysis, then downsampling.
"""
from __future__ import annotations

import os
import re
import urllib.request

import numpy as np

DATASET_ID = "7866daf2-7064-4fa0-b6a2-0b1c899ba35f"
BASE = f"https://data-proxy.ebrains.eu/api/v1/public/datasets/{DATASET_ID}"


def list_recordings():
    """Return [(subject, iso_level, remote_path)] for the 24 spontaneous recordings."""
    import json
    with urllib.request.urlopen(f"{BASE}?limit=500", timeout=60) as r:
        objs = json.load(r)["objects"]
    out = []
    for o in objs:
        m = re.search(r"sub-(\d+).*?ana-ISO(\d+)_stim-SPN\.smr$", o["name"])
        if m:
            out.append((int(m.group(1)), int(m.group(2)), o["name"]))
    return sorted(out)


def fetch(remote_path, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, os.path.basename(remote_path))
    if not os.path.exists(local):
        urllib.request.urlretrieve(f"{BASE}/{remote_path}", local)
    return local


def load_smr(path):
    """Return (X, fs) with X of shape (n_channels, n_samples)."""
    import neo
    blk = neo.io.Spike2IO(path, try_signal_grouping=False).read_block(lazy=False)
    sigs = blk.segments[0].analogsignals
    fs = float(sigs[0].sampling_rate)
    arrs = [np.asarray(s.magnitude).ravel() for s in sigs]
    n = min(a.size for a in arrs)          # channels can differ by a sample
    X = np.stack([a[:n] for a in arrs])
    return X.astype("float64"), fs


def preprocess(X, fs, band=(0.5, 4.0), fs_target=50.0):
    """Downsample, then delta-band filter, then z-score (the paper's FDT pipeline).

    Order matters: a 3rd-order Butterworth at 0.5 Hz with fs = 5 kHz has normalized frequency
    ~2e-4 and is numerically unstable in transfer-function form (it returns NaN). Decimating first
    (with `decimate`'s built-in anti-aliasing) puts the band edges in a well-conditioned range, and
    second-order-sections are used for the band-pass itself.
    """
    from scipy.signal import butter, sosfiltfilt, decimate
    Xd = np.asarray(X, dtype="float64")
    while fs / fs_target >= 4:                     # decimate in stages of <=4 for stability
        Xd = decimate(Xd, 4, axis=1, zero_phase=True)
        fs /= 4
    q = int(round(fs / fs_target))
    if q > 1:
        Xd = decimate(Xd, q, axis=1, zero_phase=True)
        fs /= q
    sos = butter(3, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band", output="sos")
    Xf = sosfiltfilt(sos, Xd, axis=1)
    Xf = (Xf - Xf.mean(axis=1, keepdims=True)) / (Xf.std(axis=1, keepdims=True) + 1e-12)
    return Xf, fs
