#!/usr/bin/env python3
"""Fit the competing spectral models to recorded EEG, one JSON row per run and condition.

    python scripts/bsd_batch.py hbn-rest --root ~/Workspace/data/ds005505 --out results/bsd/hbn_rest.json
    python scripts/bsd_batch.py hbn-ss   --root ~/Workspace/data/ds005505 --out results/bsd/hbn_ss.json
    python scripts/bsd_batch.py lemon    --root ~/Workspace/data/lemon/preprocessed --out results/bsd/lemon.json
    python scripts/bsd_batch.py ds005620 --root ~/Workspace/data/ds005620 --out results/bsd/propofol_awakening.json
    python scripts/bsd_batch.py nimh     --root ~/Workspace/data/nimh_bolt --out results/bsd/nimh.json
    python scripts/bsd_batch.py chennu   --root ~/Workspace/data/chennu_propofol/bids --out results/bsd/chennu_propofol.json

Every adapter yields (subject, run, condition, fs, data[channels, time], channel
names, extra).  The common path is: average reference over the cap, pick a
posterior set, Welch spectra 2-45 Hz with 2 s Hann segments, the mean log
spectrum over the picked channels, then variational Laplace for SpecParam,
DampingDistribution and XiAlpha with two restarts each.  For driven
paradigms the steady-state line and its second harmonic are masked out of
the fit and reported separately as a signal-to-noise ratio against the
neighbouring bins.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from neurojax.bsd import (  # noqa: E402
    DampedMixture, DampedMixture2, DampingDistribution, SpecParam, XiAlpha, posterior_model_probabilities, variational_laplace,
)

POSTERIOR_1010 = ["O1", "O2", "Oz", "PO3", "PO4", "POz", "PO7", "PO8", "P3", "P4", "Pz"]
POSTERIOR_EGI = ["E70", "E75", "E83", "E71", "E76", "E74", "E82"]  # O1, Oz, O2 and neighbours
MODELS = {"specparam": SpecParam, "specparam_knee": lambda **kw: SpecParam(knee=True, **kw),
          "specparam2": lambda peak_centres=(10.0,), **kw: SpecParam(knee=True, peak_centres=(peak_centres[0], 20.0), **kw),
          "liley": DampingDistribution, "liley_mix": DampedMixture, "liley_mix2": DampedMixture2, "xialpha": XiAlpha}


# ---------------------------------------------------------------------------
# spectra and fits
# ---------------------------------------------------------------------------

def welch_log_psd(x, fs, segment_s=2.0, f_lo=2.0, f_hi=45.0):
    from scipy.signal import welch
    nper = int(round(segment_s * fs))
    f, p = welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, window="hann", axis=-1)
    m = (f >= f_lo) & (f <= f_hi)
    n_seg = max(1, x.shape[-1] // nper)
    return f[m], np.log(p[..., m]), n_seg


def mean_log_psd(data, fs, **kw):
    """Mean over channels of the log Welch spectrum (log of the channel-mean power)."""
    f, lp, n = welch_log_psd(data, fs, **kw)
    return f, np.log(np.mean(np.exp(lp), axis=0)), n


def fit_models(freqs, log_psd, mask=None, restarts=2, centre=10.0):
    """Fit every model to the spectrum after removing its median log power.

    Recorded spectra sit at ln(V^2/Hz) ~ -25, far outside every prior; the
    models' power parameters are therefore relative to the median log power,
    which is returned as ``log_power_offset`` so absolute values can be
    restored.  The evidence is unchanged by this shift up to the prior term.
    """
    f = np.asarray(freqs); y = np.asarray(log_psd)
    if mask is not None:
        f, y = f[mask], y[mask]
    y0 = float(np.median(y)); y = y - y0
    out = {"log_power_offset": y0}
    for name, make in MODELS.items():
        m = make(peak_centres=(centre,)) if name.startswith("specparam") else make(centre=centre)
        t = time.time()
        r = variational_laplace(m, jnp.asarray(f), jnp.asarray(y), restarts=restarts)
        out[name] = {"F": r.free_energy, "accuracy": r.accuracy, "complexity": r.complexity,
                     "converged": r.converged, "params": r.natural(),
                     "rmse": float(np.sqrt(np.mean(np.asarray(r.residual) ** 2))), "seconds": round(time.time() - t, 1)}
    best_rmse = min(out[k]["rmse"] for k in MODELS)
    for k in MODELS:
        out[k]["collapsed"] = bool(out[k]["rmse"] > 10.0 * best_rmse + 0.5)
        if out[k]["collapsed"]:
            out[k]["F"] = float("-inf")
    F = [out[k]["F"] for k in MODELS]
    p = posterior_model_probabilities(F)
    out["p_model"] = {k: float(v) for k, v in zip(MODELS, p)}
    out["winner"] = list(MODELS)[int(np.argmax(F))]
    return out


def refit_models(row, names, restarts=2):
    """Refit the named models into an existing row from its stored spectrum and re-rank every model."""
    f = np.asarray(row["freqs"]); y = np.asarray(row["log_psd"]) - row["fits"]["log_power_offset"]
    mask = None
    if "ssvep_hz" in row:
        mask = np.ones_like(f, dtype=bool)
        for h in (1, 2):
            mask &= np.abs(f - h * row["ssvep_hz"]) > 0.75
        f, y = f[mask], y[mask]
    centre = row["alpha_centre_prior"]
    out = row["fits"]
    for name in names:
        make = MODELS[name]
        m = make(peak_centres=(centre,)) if name.startswith("specparam") else make(centre=centre)
        t = time.time()
        r = variational_laplace(m, jnp.asarray(f), jnp.asarray(y), restarts=restarts)
        out[name] = {"F": r.free_energy, "accuracy": r.accuracy, "complexity": r.complexity,
                     "converged": r.converged, "params": r.natural(),
                     "rmse": float(np.sqrt(np.mean(np.asarray(r.residual) ** 2))), "seconds": round(time.time() - t, 1)}
    present = [k for k in MODELS if k in out]
    best_rmse = min(out[k]["rmse"] for k in present)
    for k in present:
        out[k]["collapsed"] = bool(out[k]["rmse"] > 10.0 * best_rmse + 0.5)
        if out[k]["collapsed"]:
            out[k]["F"] = float("-inf")
    F = [out[k]["F"] for k in present]
    p = posterior_model_probabilities(F)
    out["p_model"] = {k: float(v) for k, v in zip(present, p)}
    out["winner"] = present[int(np.argmax(F))]
    return out


def alpha_centre(freqs, log_psd):
    """Prior centre for the alpha peak: the 6-14 Hz maximum of the detrended log spectrum."""
    f = np.asarray(freqs); y = np.asarray(log_psd)
    m = (f >= 6) & (f <= 14)
    slope = np.polyfit(np.log(f), y, 1)
    resid = y - np.polyval(slope, np.log(f))
    return float(f[m][np.argmax(resid[m])])


def ssvep_snr(freqs, log_psd, f_lo=3.0, f_hi=45.0, half_width_bins=1, flank_bins=6):
    """Tag frequency and SNR: power at a bin over the mean of flanking bins, searched over the band."""
    f = np.asarray(freqs); p = np.exp(np.asarray(log_psd))
    best = (None, 0.0)
    for i in range(flank_bins + half_width_bins, len(f) - flank_bins - half_width_bins):
        if not (f_lo <= f[i] <= f_hi):
            continue
        sig = p[i - half_width_bins:i + half_width_bins + 1].mean()
        flank = np.r_[p[i - flank_bins - half_width_bins:i - half_width_bins], p[i + half_width_bins + 1:i + half_width_bins + 1 + flank_bins]].mean()
        snr = sig / flank
        if snr > best[1]:
            best = (float(f[i]), float(snr))
    return best


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def _mne():
    import mne
    mne.set_log_level("ERROR")
    return mne


ALL_CHANNELS = False   # set by --reduce hosvd: adapters then return the whole cap


def _prep(raw, picks_pref, resample=250.0):
    """Average reference over all EEG channels, pick the posterior set (or keep the cap), resample."""
    mne = _mne()
    raw.pick("eeg") if "eeg" in raw.get_channel_types() else None
    raw.load_data()
    raw.set_eeg_reference("average", projection=False, verbose=False)
    if raw.info["sfreq"] > resample:
        raw.resample(resample, verbose=False)
    if ALL_CHANNELS:
        return raw.get_data(), float(raw.info["sfreq"]), list(raw.ch_names)
    names = [c for c in picks_pref if c in raw.ch_names]
    if not names:
        raise RuntimeError(f"none of {picks_pref[:4]}... in {raw.ch_names[:8]}")
    return raw.get_data(picks=names), float(raw.info["sfreq"]), names


def _read_events(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def iter_hbn_rest(root):
    mne = _mne()
    for setf in sorted(glob.glob(os.path.join(root, "sub-*", "eeg", "*task-RestingState_eeg.set"))):
        sub = os.path.basename(setf).split("_")[0]
        ev = _read_events(setf.replace("_eeg.set", "_events.tsv"))
        raw = mne.io.read_raw_eeglab(setf, preload=True, verbose=False)
        data, fs, names = _prep(raw, POSTERIOR_EGI)
        # eyes-open / eyes-closed blocks from the instruction markers
        marks = [(float(e["onset"]), e["value"]) for e in ev if e["value"] in ("instructed_toOpenEyes", "instructed_toCloseEyes")]
        segs = {"EO": [], "EC": []}
        for (t0, v), (t1, _) in zip(marks[:-1], marks[1:]):
            segs["EO" if v == "instructed_toOpenEyes" else "EC"].append((t0 + 1.0, t1))
        for cond, spans in segs.items():
            chunks = [data[:, int(a * fs):int(b * fs)] for a, b in spans if b - a > 4]
            if chunks:
                yield sub, "RestingState", cond, fs, np.concatenate(chunks, axis=1), names, {"n_blocks": len(chunks)}


def iter_hbn_ss(root):
    mne = _mne()
    for setf in sorted(glob.glob(os.path.join(root, "sub-*", "eeg", "*task-surroundSupp_run-*_eeg.set"))):
        sub = os.path.basename(setf).split("_")[0]
        run = os.path.basename(setf).split("_")[2]
        ev = _read_events(setf.replace("_eeg.set", "_events.tsv"))
        raw = mne.io.read_raw_eeglab(setf, preload=True, verbose=False)
        data, fs, names = _prep(raw, POSTERIOR_EGI)
        trials = {}
        for e in ev:
            if e["value"] != "stim_ON":
                continue
            key = f"bg{e['background']}_fg{e['foreground_contrast']}"
            t0 = float(e["onset"]); dur = float(e["duration"]) if e["duration"] not in ("n/a", "") else 2.4
            trials.setdefault(key, []).append(data[:, int((t0 + 0.3) * fs):int((t0 + dur) * fs)])
        for cond, eps in sorted(trials.items()):
            n = min(ep.shape[1] for ep in eps)
            yield sub, run, cond, fs, np.stack([ep[:, :n] for ep in eps]), names, {"n_trials": len(eps), "epoched": True}


def iter_lemon(root):
    mne = _mne()
    for setf in sorted(glob.glob(os.path.join(root, "sub-*_E[CO].set"))):
        base = os.path.basename(setf)[:-4]
        sub, cond = base.split("_")
        raw = mne.io.read_raw_eeglab(setf, preload=True, verbose=False)
        data, fs, names = _prep(raw, POSTERIOR_1010)
        yield sub, "rest", cond, fs, data, names, {}


def iter_ds005620(root):
    mne = _mne()
    for vhdr in sorted(glob.glob(os.path.join(root, "sub-*", "eeg", "*_eeg.vhdr"))):
        base = os.path.basename(vhdr)[:-9]
        parts = dict(p.split("-", 1) for p in base.split("_") if "-" in p)
        sub = parts["sub"]; task = parts.get("task", ""); acq = parts.get("acq", ""); run = parts.get("run", "1")
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {base}: {exc}"); continue
        data, fs, names = _prep(raw, POSTERIOR_1010)
        cond = f"{task}-{acq}" if task == "awake" else task
        yield sub, run, cond, fs, data, names, {}


def iter_nimh(root):
    from scipy.io import loadmat
    for mat in sorted(glob.glob(os.path.join(root, "*EEG_pp.mat"))):
        base = os.path.basename(mat).replace("_EEG_pp.mat", "")
        d = loadmat(mat, squeeze_me=True, struct_as_record=False)
        eeg = d.get("EEG")
        if eeg is None:
            continue
        data = np.asarray(eeg.data, dtype=float)
        if data.ndim == 3:
            data = data.reshape(data.shape[0], -1)
        labels = [str(c.labels) for c in np.atleast_1d(eeg.chanlocs)]
        fs = float(eeg.srate)
        keep = [i for i, l in enumerate(labels) if l.upper() not in ("ECG", "EKG")]
        data = data[keep]; labels = [labels[i] for i in keep]
        data = data - data.mean(axis=0, keepdims=True)
        picks = [i for i, l in enumerate(labels) if l in POSTERIOR_1010]
        if not picks:
            picks = list(range(len(labels)))
        sub, _, run_code = base.split("-")[0], None, "-".join(base.split("-")[1:])
        yield sub, run_code, "rest" if "ecr" in base else "deep-breath" if "adb" in base else "cue", fs, data[picks], [labels[i] for i in picks], {}


def iter_chennu(root):
    """Chennu et al. 2016 propofol sedation (FieldTrip BIDS copy): 91 EGI channels, 250 Hz, four runs per subject.

    The run's sedation level and plasma propofol concentration come from
    ``sub-XX_scans.tsv``; the EEG is 10-s cleaned epochs concatenated, so a
    2-s Welch segment never straddles more than one boundary.
    """
    mne = _mne()
    for sdir in sorted(glob.glob(os.path.join(root, "sub-*"))):
        sub = os.path.basename(sdir)
        scans = {}
        for row in _read_events(os.path.join(sdir, f"{sub}_scans.tsv")):
            scans[os.path.basename(row["filename"])] = row
        for vhdr in sorted(glob.glob(os.path.join(sdir, "eeg", "*_eeg.vhdr"))):
            meta = scans.get(os.path.basename(vhdr), {})
            run = os.path.basename(vhdr).split("_run-")[1].split("_")[0]
            try:
                raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
            except Exception as exc:  # noqa: BLE001
                print(f"skip {vhdr}: {exc}"); continue
            data, fs, names = _prep(raw, POSTERIOR_1010 + ["E71", "E72", "E76", "E77", "E84"])
            cond = meta.get("sedation", f"run-{run}").replace(" ", "_")
            extra = {k: (float(meta[k]) if meta.get(k, "n/a") not in ("n/a", "") else None)
                     for k in ("concentration", "reactiontime", "correctresponses") if k in meta}
            yield sub, run, cond, fs, data, names, extra


def iter_ds004541(root, max_window_s=300.0):
    """ds004541 (Saini Ferron et al., CC0): EEG-fNIRS during general anaesthesia, 58 EEG channels, 1 kHz, EDF.

    The anaesthetic agent is not documented in the release (README "in preparation",
    participants.tsv all n/a), so the conditions are purely procedural, cut from the
    event markers: ``baseline`` = up to ``max_window_s`` before ``start`` (drug
    delivery), ``induction`` = start..loc (loss of consciousness), ``maintenance`` =
    loc..end (end of delivery; contains tetanic stimuli), ``emergence`` = end..roc
    (return of consciousness).  Channels flagged bad in channels.tsv are dropped
    before the average reference.
    """
    mne = _mne()
    for edf in sorted(glob.glob(os.path.join(root, "sub-*", "ses-*", "eeg", "*_eeg.edf"))):
        base = os.path.basename(edf)[:-8]
        sub = base.split("_")[0]
        ev = {r["trial_type"]: float(r["onset"]) for r in _read_events(edf.replace("_eeg.edf", "_events.tsv"))
              if r["trial_type"] in ("start", "loc", "end", "roc")}
        if "start" not in ev:
            print(f"skip {base}: no start marker"); continue
        bad = [r["name"] for r in _read_events(edf.replace("_eeg.edf", "_channels.tsv")) if r.get("status") == "bad"]
        try:
            raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {base}: {exc}"); continue
        raw.drop_channels([c for c in bad if c in raw.ch_names])
        raw.drop_channels([c for c in raw.ch_names if raw.get_channel_types(picks=[c])[0] != "eeg"])
        t_end = raw.times[-1]
        windows = {"baseline": (max(0.0, ev["start"] - max_window_s), ev["start"])}
        if "loc" in ev:
            windows["induction"] = (ev["start"], ev["loc"])
            if "end" in ev:
                windows["maintenance"] = (ev["loc"], min(ev["end"], ev["loc"] + max_window_s))
                if "roc" in ev:
                    windows["emergence"] = (max(ev["end"], ev["roc"] - max_window_s), ev["roc"])
        for i, (cond, (t0, t1)) in enumerate(windows.items(), 1):
            t1 = min(t1, t_end)
            if t1 - t0 < 20.0:
                print(f"skip {base} {cond}: {t1 - t0:.0f} s"); continue
            seg = raw.copy().crop(t0, t1)
            data, fs, names = _prep(seg, POSTERIOR_1010 + ["PO3", "PO4", "POz", "P1", "P2"])
            yield sub, str(i), cond, fs, data, names, {"t0": t0, "t1": t1, "n_bad": len(bad)}


ADAPTERS = {"hbn-rest": iter_hbn_rest, "hbn-ss": iter_hbn_ss, "lemon": iter_lemon,
            "ds005620": iter_ds005620, "nimh": iter_nimh, "chennu": iter_chennu,
            "ds004541": iter_ds004541}


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=sorted(ADAPTERS))
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="stop after this many rows (0 = all)")
    p.add_argument("--reduce", choices=["posterior", "hosvd", "parafac", "parafac-weights"], default="posterior",
                   help="posterior: fixed posterior-channel mean; hosvd: leading channel factor of the log-spectrum tensor; parafac: one row per non-negative space-frequency-time component (rank --rank) of the whole-cap power tensor; parafac-weights: the alpha component's spatial factor weights the whole-cap mean log spectrum")
    p.add_argument("--rank", type=int, default=3)
    p.add_argument("--refit", nargs="+", default=None, metavar="MODEL",
                   help="refit only these models into the rows already in --out (no data read) and re-rank")
    a = p.parse_args()
    if a.refit:
        rows = json.load(open(a.out))
        seen = set(); kept = []
        for r in rows:   # drop exact duplicate rows left by an earlier run
            key = (r["subject"], r["run"], r["condition"])
            if key not in seen:
                seen.add(key); kept.append(r)
        for i, r in enumerate(kept):
            refit_models(r, a.refit, restarts=a.restarts)
            print(f"{r['subject']} {r['run']} {r['condition']}: winner {r['fits']['winner']} "
                  + " ".join(f"{k}={r['fits'][k]['F']:.1f}" for k in a.refit), flush=True)
            json.dump(kept, open(a.out, "w"))
        print(f"refit {a.refit} in {len(kept)} rows of {a.out}")
        return
    global ALL_CHANNELS
    ALL_CHANNELS = a.reduce in ("hosvd", "parafac", "parafac-weights")

    rows = []
    if os.path.exists(a.out):
        rows = json.load(open(a.out))
    done = {(r["subject"], r["run"], r["condition"]) for r in rows}
    def expand(sub, run, cond, fs, data, names, extra):
        """Yield (cond, log_psd, n_seg, extra) items for one run; several for a PARAFAC reduction."""
        if a.reduce == "parafac" and not extra.get("epoched"):
            from neurojax.bsd.multiway import parafac_components
            f, comps, fit, n_seg = parafac_components(data * 1e6, fs, rank=a.rank)
            for k, c in enumerate(comps):
                top = np.argsort(-c["spatial"])[:8]
                ex = dict(extra); ex.update({"reduce": "parafac", "component": k, "rank": a.rank, "parafac_fit": fit,
                                             "power_share": c["power_share"], "alpha_fraction": c["alpha_fraction"],
                                             "spatial_top": {names[i]: round(float(c["spatial"][i]), 4) for i in top},
                                             "temporal_cv": float(c["temporal"].std() / c["temporal"].mean())})
                yield f"{cond}::comp{k}", f, c["log_psd"], n_seg, None, ex
            return
        yield cond, None, None, None, None, extra

    for sub, run, cond0, fs, data, names, extra0 in ADAPTERS[a.dataset](a.root):
      for cond, f_pf, lp_pf, n_pf, _, extra in expand(sub, run, cond0, fs, data, names, extra0):
        if (sub, run, cond) in done:
            continue
        if f_pf is not None:
            f, lp, n_seg, mask = f_pf, lp_pf, n_pf, None
        elif extra.get("epoched"):
            # one periodogram per trial, averaged: n_segments = n_trials
            from scipy.signal import periodogram
            n = data.shape[-1]
            f, pxx = periodogram(data, fs=fs, window="hann", axis=-1)
            m = (f >= 2.0) & (f <= 45.0)
            f = f[m]; pxx = pxx[..., m].mean(axis=0)  # mean over trials -> (ch, f)
            lp = np.log(pxx.mean(axis=0)); n_seg = data.shape[0]
            tag, snr = ssvep_snr(f, lp)
            mask = np.ones_like(f, dtype=bool)
            for h in (1, 2):
                mask &= np.abs(f - h * tag) > 0.75
            extra.update({"ssvep_hz": tag, "ssvep_snr": snr, "mask_excluded_bins": int((~mask).sum())})
        elif a.reduce == "parafac-weights":
            from neurojax.bsd.multiway import parafac_components, spectral_tensor
            f, comps, fit, n_seg = parafac_components(data * 1e6, fs, rank=a.rank)
            alpha = max(comps, key=lambda c: c["alpha_fraction"] * c["power_share"])
            wgt = alpha["spatial"]
            f2, T, n_seg = spectral_tensor(data, fs)
            lp = np.einsum("c,cfs->f", wgt, T) / T.shape[2]; mask = None; f = f2
            top = np.argsort(-wgt)[:8]
            extra.update({"reduce": "parafac-weights", "parafac_fit": fit, "alpha_fraction": alpha["alpha_fraction"],
                          "power_share": alpha["power_share"], "channel_weights": {names[i]: round(float(wgt[i]), 4) for i in top}})
        elif a.reduce == "hosvd":
            from neurojax.bsd.multiway import hosvd_spectrum, spectral_tensor
            f, T, n_seg = spectral_tensor(data, fs)
            red = hosvd_spectrum(T)
            lp = red["log_psd"]; mask = None
            top = np.argsort(-red["channel_weights"])[:8]
            extra.update({"reduce": "hosvd", "channel_weights": {names[i]: round(float(red["channel_weights"][i]), 4) for i in top},
                          "energy_channel_1": red["energy_channel_1"], "energy_segment_1": red["energy_segment_1"]})
        else:
            f, lp, n_seg = mean_log_psd(data, fs)
            mask = None
        centre = alpha_centre(f, lp)
        fits = fit_models(f, lp, mask=mask, restarts=a.restarts, centre=centre)
        row = {"dataset": a.dataset, "subject": sub, "run": run, "condition": cond, "fs": fs,
               "channels": names, "n_segments": int(n_seg), "duration_s": float(data.shape[-1] / fs) if not extra.get("epoched") else float(data.shape[0] * data.shape[-1] / fs),
               "alpha_centre_prior": centre, "freqs": f.tolist(), "log_psd": lp.tolist(), "fits": fits, **extra}
        rows.append(row); done.add((sub, run, cond))
        print(f"{sub} {run} {cond}: n_seg {n_seg} winner {fits['winner']} p={ {k: round(v,2) for k,v in fits['p_model'].items()} } "
              f"alpha~{centre:.1f} Hz" + (f" ssvep {extra['ssvep_hz']:.1f} Hz snr {extra['ssvep_snr']:.1f}" if "ssvep_hz" in extra else ""), flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w"))
        if a.limit and len(rows) >= a.limit:
            break
      if a.limit and len(rows) >= a.limit:
          break
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
