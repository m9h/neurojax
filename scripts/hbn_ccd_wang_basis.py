"""CCD SSVEP: Wang-2000-style trilinear basis built on the EVOKED tensor, then single trials projected onto it.

Variants compared on the same cleaned epochs (bad channels/trials removed, common good channels):
  clean      : no denoising
  raw_t99    : Tucker 99% energy on single-trial tensor (trials x ch x time)   [earlier, wrong regime]
  group_e99  : basis from group evoked tensor (subjects x ch x time), 99% energy per mode; trials projected on ch x time bases
  group_e999 : same at 99.9%
  self_e99   : basis from the subject's own sub-average tensor (5 sub-averages x ch x time), 99%
Metrics: evoked SNR (spectrum of the trial average) and single-trial SNR (median over trials) at 20/25/40/50 Hz, posterior picks.
"""
import glob, os, csv, json
import numpy as np, mne
mne.set_log_level("ERROR")
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from neurojax.bsd.multiway import tucker_energy_denoise
from neurojax.analysis.tensor import hosvd, unfold

root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Workspace/data/ds005505")
PICKS = ["E70", "E75", "E83", "E71", "E76", "E74", "E82"]
TAGS = (20.0, 25.0, 40.0, 50.0)

def snr_at(f, S, f0, half=1, gap=2, flank=8):
    i = int(np.argmin(np.abs(f - f0))); sig = S[i-half:i+half+1].mean()
    fl = np.r_[S[i-half-gap-flank:i-half-gap], S[i+half+gap+1:i+half+gap+1+flank]].mean(); return sig / fl

subs = {}
for setf in sorted(glob.glob(os.path.join(root, "sub-*", "eeg", "*task-contrastChangeDetection_run-1_eeg.set"))):
    sub = os.path.basename(setf).split("_")[0]
    ev = list(csv.DictReader(open(setf.replace("_eeg.set", "_events.tsv")), delimiter="\t"))
    raw = mne.io.read_raw_eeglab(setf, preload=True); raw.filter(1.0, 100.0, verbose=False); raw.notch_filter([60.0], verbose=False)
    X = raw.get_data(); fs = raw.info["sfreq"]
    v = X.var(axis=1); med = np.median(v); good = (v < 5 * med) & (v > med / 20)
    raw.info["bads"] = [c for c, g in zip(raw.ch_names, good) if not g]
    raw.interpolate_bads(reset_bads=True, verbose=False)   # spherical spline from the EGI positions
    X = raw.get_data(); Xg = X - X.mean(axis=0, keepdims=True)   # average reference over the full interpolated cap
    good = np.ones(len(raw.ch_names), dtype=bool)
    n = int(2.0 * fs); eps = []
    for e in ev:
        if e["value"] == "contrastTrial_start":
            s = int((float(e["onset"]) + 0.5) * fs); seg = Xg[:, s:s+n]
            if seg.shape[1] == n: eps.append(seg - seg.mean(axis=1, keepdims=True))
    if len(eps) < 10: continue
    E = np.stack(eps); amp = np.abs(E).max(axis=(1, 2)); keep = amp < np.median(amp) * 3
    subs[sub] = dict(E=E[keep], good=good, fs=fs, names=raw.ch_names)
    print(f"{sub}: interpolated {len(raw.info['bads']) if raw.info['bads'] else 0} bad, trials {keep.sum()}/{len(eps)}", flush=True)

common = np.logical_and.reduce([d["good"] for d in subs.values()])
names = [c for c, g in zip(next(iter(subs.values()))["names"], common) if g]
pick = [names.index(c) for c in PICKS if c in names]
print(f"common good channels {common.sum()}/129, posterior picks kept {len(pick)}/{len(PICKS)}")
fs = next(iter(subs.values()))["fs"]; n = int(2.0 * fs); f = np.fft.rfftfreq(n, 1/fs); win = np.hanning(n)

def measures(T):
    x = T[:, pick, :].mean(axis=1)                       # trials x time, posterior mean
    evoked = np.abs(np.fft.rfft(x.mean(axis=0) * win))**2
    single = np.abs(np.fft.rfft(x * win, axis=-1))**2    # trials x f
    return ({int(t): round(float(snr_at(f, evoked, t)), 2) for t in TAGS},
            {int(t): round(float(np.median([snr_at(f, s, t) for s in single])), 2) for t in TAGS})

def energy_ranks(X, energy):
    ranks = []
    for mode in range(X.ndim):
        s = np.linalg.svd(np.asarray(unfold(X, mode)), compute_uv=False)
        c = np.cumsum(s**2) / np.sum(s**2); ranks.append(int(np.searchsorted(c, energy) + 1))
    return ranks

def project(E, Uc, Ut):
    Pc = Uc @ Uc.T; Pt = Ut @ Ut.T
    return np.einsum("ij,njk,kl->nil", Pc, E, Pt)

# group evoked tensor
A = np.stack([d["E"][:, common, :].mean(axis=0) for d in subs.values()])   # subjects x ch x time
bases = {}
for en in (0.99, 0.999):
    rk = energy_ranks(A, en); core, facs = hosvd(A, ranks=rk)
    bases[en] = (np.asarray(facs[1]), np.asarray(facs[2]), rk)
    print(f"group evoked basis {en}: ranks (subj, ch, time) = {rk}")

rows = []
for sub, d in subs.items():
    Ec = d["E"][:, common, :]
    res = {"subject": sub, "n": int(Ec.shape[0])}
    res["clean"] = measures(Ec)
    D, rk = tucker_energy_denoise(Ec, 0.99); res["raw_t99"] = measures(D); res["raw_t99_ranks"] = rk
    for en, key in ((0.99, "group_e99"), (0.999, "group_e999")):
        Uc, Ut, rk = bases[en]; res[key] = measures(project(Ec, Uc, Ut))
    # self basis: 5 sub-averages
    idx = np.array_split(np.random.default_rng(0).permutation(Ec.shape[0]), 5)
    S = np.stack([Ec[i].mean(axis=0) for i in idx]); rk = energy_ranks(S, 0.99); core, facs = hosvd(S, ranks=rk)
    res["self_e99"] = measures(project(Ec, np.asarray(facs[1]), np.asarray(facs[2]))); res["self_e99_ranks"] = rk
    rows.append(res)
    print(f"{sub}: " + " | ".join(f"{k} ev{res[k][0][20]}/{res[k][0][40]} st{res[k][1][20]}/{res[k][1][40]}" for k in ("clean", "raw_t99", "group_e99", "group_e999", "self_e99")), flush=True)

print("\nMEDIANS over subjects (evoked SNR / single-trial SNR):")
for k in ("clean", "raw_t99", "group_e99", "group_e999", "self_e99"):
    print(f"  {k:11s} " + "  ".join(f"{t}Hz ev {np.median([r[k][0][t] for r in rows]):.2f} st {np.median([r[k][1][t] for r in rows]):.2f}" for t in (20, 25, 40, 50)))
json.dump({"common_channels": int(common.sum()), "group_ranks": {str(k): v[2] for k, v in bases.items()}, "rows": rows},
          open("results/bsd/hbn_ccd_wang_basis_interp.json", "w"), indent=1)
