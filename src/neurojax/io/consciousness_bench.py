# © NeuroJAX developers
#
# License: BSD (3-clause)
"""Loaders for the consciousness complexity benchmark datasets.

All loaders return a list of :class:`neurojax.bench.consciousness_bench.Recording`
with ``data`` as (C, T) float32 at the native (or requested) sampling rate,
EEG channels only, average-referenced. Missing datasets raise
``FileNotFoundError`` — tests skip on that. Root: ``$CB_ROOT`` or
``/data/raw/consciousness-bench`` (see ``docs/CONSCIOUSNESS_BENCH.md``).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import mne
import numpy as np
import scipy.io as sio

from neurojax.bench.consciousness_bench import Recording

__all__ = ["cb_root", "load_chennu_propofol", "load_dmt", "load_psilocybin_microdose",
           "load_sarasso_anesthesia", "LOADERS"]

CHENNU_LEVELS = {1: "baseline", 2: "mild", 3: "moderate", 4: "recovery"}


def cb_root() -> Path:
    return Path(os.environ.get("CB_ROOT", "/data/raw/consciousness-bench"))


def _finish(raw: mne.io.BaseRaw, resample: float | None, l_freq=0.5, h_freq=45.0) -> tuple[np.ndarray, float]:
    raw.pick("eeg")
    raw.load_data()
    raw.filter(l_freq, h_freq, verbose=False)
    if resample and raw.info["sfreq"] != resample:
        raw.resample(resample, verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw.get_data().astype(np.float32), float(raw.info["sfreq"])


# --------------------------------------------------------------------------
# Chennu et al. 2016 propofol sedation (Cambridge repository 1810/252736)
# --------------------------------------------------------------------------

def load_chennu_propofol(root: Path | None = None, resample: float | None = 250.0,
                         subjects: list[str] | None = None) -> list[Recording]:
    """20 subjects × {baseline, mild, moderate, recovery}; 91-channel EEGLAB
    epochs (10 s, already 0.5-45 Hz filtered and artefact-rejected) are
    concatenated into one (C, T) array per recording. ``datainfo.mat`` gives
    the sedation level, propofol plasma concentration, reaction time and hits."""
    d = (root or cb_root()) / "propofol-chennu2016" / "unzipped" / "Sedation-RestingState"
    info_path = d / "datainfo.mat"
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    m = sio.loadmat(info_path, squeeze_me=True, struct_as_record=False)
    info = m["datainfo"]
    recs = []
    for row in np.atleast_1d(info):
        name = str(row[0]).strip()
        level = int(row[1])
        subject = name.split(" ")[0].split("-")[0]
        if subjects and subject not in subjects:
            continue
        set_path = d / f"{name}.set"
        if not set_path.exists():
            continue
        ep = mne.io.read_epochs_eeglab(set_path, verbose=False)
        ep.pick("eeg")
        X = ep.get_data()  # (E, C, T)
        fs = float(ep.info["sfreq"])
        data = np.concatenate(list(X), axis=1)
        raw = mne.io.RawArray(data, mne.create_info(ep.ch_names, fs, "eeg"), verbose=False)
        data, fs = _finish(raw, resample)
        recs.append(Recording("propofol-chennu2016", subject, CHENNU_LEVELS[level], data, fs,
                              list(raw.ch_names),
                              dict(propofol_ugml=float(row[2]), rt_ms=float(row[3]), hits=float(row[4]),
                                   file=name, epoch_s=X.shape[2] / fs)))
    return recs


# --------------------------------------------------------------------------
# Pallavicini et al. 2021 inhaled DMT (Zenodo 3992359)
# --------------------------------------------------------------------------

_DMT_RE = re.compile(r"^[sS](\d+)[-_](DMT|EC|EO|Ec)\.(bdf|xdf)$")


def load_dmt(root: Path | None = None, resample: float | None = 250.0,
             subjects: list[str] | None = None) -> list[Recording]:
    """35 subjects × {EC (eyes closed rest), EO, DMT}; BDF files (two XDF
    recordings are skipped unless ``pyxdf`` is installed)."""
    d = (root or cb_root()) / "dmt-pallavicini2021"
    files = sorted(p for p in d.glob("*") if _DMT_RE.match(p.name)) if d.exists() else []
    if not files:
        raise FileNotFoundError(d)
    recs = []
    for p in files:
        sid, cond, ext = _DMT_RE.match(p.name).groups()
        subject = f"S{int(sid):02d}"
        if subjects and subject not in subjects:
            continue
        if ext == "xdf":
            continue  # pyxdf not a dependency; 2 of 105 files
        raw = mne.io.read_raw_bdf(p, preload=True, verbose=False)
        raw.pick("eeg")
        data, fs = _finish(raw, resample)
        recs.append(Recording("dmt-pallavicini2021", subject, cond.upper(), data, fs, list(raw.ch_names),
                              dict(file=p.name)))
    return recs


# --------------------------------------------------------------------------
# Cavanna et al. 2022 psilocybin microdosing (Zenodo 5745892, Data.rar)
# --------------------------------------------------------------------------

def load_psilocybin_microdose(root: Path | None = None, resample: float | None = 250.0,
                              subjects: list[str] | None = None) -> list[Recording]:
    """Placebo vs 0.5 g psilocybe, eyes-closed rest. Walks the extracted
    archive for EEG files whose path names the condition; the archive layout
    is read at load time (see ``meta['file']``)."""
    d = (root or cb_root()) / "psilocybin-microdose-cavanna2022" / "unrar"
    files = sorted(p for p in d.rglob("*") if p.suffix.lower() in (".bdf", ".edf", ".set", ".fif", ".vhdr"))
    if not files:
        raise FileNotFoundError(d)
    recs = []
    for p in files:
        s = str(p).lower()
        cond = "psilocybin" if re.search(r"psilo|active|dose|mush", s) else "placebo" if "placebo" in s else None
        if cond is None:
            continue
        msub = re.search(r"(?:sub|s|subject)[-_]?(\d+)", p.stem.lower())
        subject = f"S{int(msub.group(1)):02d}" if msub else p.stem
        if subjects and subject not in subjects:
            continue
        if p.suffix.lower() == ".set":
            raw = mne.io.read_raw_eeglab(p, preload=True, verbose=False)
        elif p.suffix.lower() == ".vhdr":
            raw = mne.io.read_raw_brainvision(p, preload=True, verbose=False)
        else:
            raw = mne.io.read_raw(p, preload=True, verbose=False)
        data, fs = _finish(raw, resample)
        recs.append(Recording("psilocybin-microdose-cavanna2022", subject, cond, data, fs,
                              list(raw.ch_names), dict(file=str(p.relative_to(d)))))
    return recs


# --------------------------------------------------------------------------
# Sarasso 2015 / Maschke 2024 propofol-xenon-ketamine (Zenodo 806176, restricted)
# --------------------------------------------------------------------------

def load_sarasso_anesthesia(root: Path | None = None, resample: float | None = 250.0) -> list[Recording]:
    """Wake vs drug for propofol, xenon, ketamine (N=5 each). Expects the
    requested archive unpacked under ``anesthesia-sarasso2015/<drug>/<subject>/``
    with one readable EEG file per state whose name contains ``wake`` or the
    drug name; the exact layout is only known after access is granted."""
    d = (root or cb_root()) / "anesthesia-sarasso2015"
    files = sorted(p for p in d.rglob("*") if p.suffix.lower() in (".set", ".edf", ".bdf", ".fif", ".vhdr", ".mat"))
    if not files:
        raise FileNotFoundError(d)
    recs = []
    for p in files:
        parts = [q.lower() for q in p.relative_to(d).parts]
        drug = next((x for x in ("propofol", "xenon", "ketamine") if any(x in q for q in parts)), None)
        if drug is None or p.suffix.lower() == ".mat":
            continue
        cond = "wake" if "wake" in p.name.lower() or "baseline" in p.name.lower() else drug
        subject = f"{drug}-{p.parent.name}"
        raw = mne.io.read_raw(p, preload=True, verbose=False)
        data, fs = _finish(raw, resample)
        recs.append(Recording("anesthesia-sarasso2015", subject, cond, data, fs, list(raw.ch_names),
                              dict(drug=drug, file=str(p.relative_to(d)))))
    return recs


LOADERS = {
    "propofol-chennu2016": (load_chennu_propofol, [("baseline", "mild"), ("baseline", "moderate"),
                                                   ("baseline", "recovery")]),
    "dmt-pallavicini2021": (load_dmt, [("EC", "DMT"), ("EC", "EO")]),
    "psilocybin-microdose-cavanna2022": (load_psilocybin_microdose, [("placebo", "psilocybin")]),
    "anesthesia-sarasso2015": (load_sarasso_anesthesia, [("wake", "propofol"), ("wake", "xenon"),
                                                         ("wake", "ketamine")]),
}
