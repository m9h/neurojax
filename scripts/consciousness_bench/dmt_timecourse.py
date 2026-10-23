#!/usr/bin/env python
"""DMT time course: measures and TDE-HMM state occupancy along the drug effect.

Pallavicini et al. 2021 protocol: 5 min eyes-closed (EC) and eyes-open
baselines, then the DMT recording starts *at exhalation* and runs until the
participant reports return to baseline (6 ± 1.4 min). So t = 0 is drug
onset and the whole DMT file is the acute window + tail.

    python scripts/consciousness_bench/dmt_timecourse.py [--subjects S01,S02] [--win 10] [--step 5]

Per subject: sliding-window values of every registered measure (10
channels), z-scored against that subject's EC windows; TDE-HMM (15 lags,
PCA, K states) fitted on pooled EC + DMT data of all subjects (all 24
channels), giving per-window fractional occupancy, mean lifetime and
switching rate; a Bateman effect curve y(t) = b + A (e^{-t/τe} − e^{-t/τa})
fitted per subject × feature, reporting time-to-peak, amplitude and R²;
onset / offset by threshold crossing of the group z. Outputs to
$CB_OUT/dmt-pallavicini2021/timecourse/.
"""

import argparse
import os
import time
from pathlib import Path

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from neurojax.analysis.nltsa import MEASURES
from neurojax.analysis.spectral_baselines import SPECTRAL_MEASURES
from neurojax.bench.consciousness_bench import score_epochs
from neurojax.io.consciousness_bench import load_dmt
from neurojax.models.hmm import GaussianHMM, HMMConfig

ap = argparse.ArgumentParser()
ap.add_argument("--subjects", default="")
ap.add_argument("--win", type=float, default=10.0)
ap.add_argument("--step", type=float, default=5.0)
ap.add_argument("--n-channels", type=int, default=10)
ap.add_argument("--n-states", type=int, default=6)
ap.add_argument("--n-embeddings", type=int, default=15)
ap.add_argument("--n-pca", type=int, default=30)
ap.add_argument("--hmm-epochs", type=int, default=15)
ap.add_argument("--measures", default="all")
ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
ap.add_argument("--max-t", type=float, default=480.0, help="seconds after exhalation to keep")
args = ap.parse_args()

out = Path(os.environ.get("CB_OUT", "/data/datasets/consciousness-bench")) / "dmt-pallavicini2021" / "timecourse"
out.mkdir(parents=True, exist_ok=True)
ALL = {**MEASURES, **SPECTRAL_MEASURES}
measures = ALL if args.measures == "all" else {k: ALL[k] for k in args.measures.split(",")}

t0 = time.time()
recs = load_dmt(subjects=args.subjects.split(",") if args.subjects else None)
by = {}
for r in recs:
    by.setdefault(r.subject, {})[r.condition] = r
subjects = sorted(s for s, c in by.items() if "EC" in c and "DMT" in c)
fs = recs[0].fs
print(f"[load] {len(subjects)} subjects with EC+DMT, fs={fs}, {time.time()-t0:.0f}s", flush=True)


def windows(X, win, step, max_t=None):
    n, h = int(win * fs), int(step * fs)
    T = X.shape[1] if max_t is None else min(X.shape[1], int(max_t * fs))
    starts = range(0, T - n + 1, h)
    return [(s / fs, X[:, s:s + n]) for s in starts]


# ---------------------------------------------------------------- measures
jobs, idx = [], []
for s in subjects:
    for cond in ("EC", "DMT"):
        X = by[s][cond].data
        chan = np.linspace(0, X.shape[0] - 1, args.n_channels).round().astype(int)
        for t, W in windows(X[chan], args.win, args.step, args.max_t if cond == "DMT" else None):
            jobs.append(W)
            idx.append((s, cond, t))
t0 = time.time()
vals = score_epochs(jobs, fs, measures, n_jobs=args.n_jobs)
M = pd.DataFrame(idx, columns=["subject", "condition", "t"])
for m in measures:
    M[m] = [v[m] for v in vals]
M.to_csv(out / "windows_measures.csv", index=False)
print(f"[measures] {len(M)} windows × {len(measures)} measures in {time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- TDE-HMM
def tde(X, n_emb):
    """(C, T) -> (T', C*n_emb) lagged copies, lags centred on 0."""
    h = n_emb // 2
    T = X.shape[1] - 2 * h
    return np.concatenate([X[:, h + k: h + k + T].T for k in range(-h, h + 1)], axis=1)


t0 = time.time()
seqs, keys = [], []
for s in subjects:
    for cond in ("EC", "DMT"):
        X = by[s][cond].data.astype(np.float64)
        X = X / X.std(1, keepdims=True)
        seqs.append(tde(X, args.n_embeddings))
        keys.append((s, cond))
pooled = np.concatenate(seqs, 0)
mu = pooled.mean(0)
U, S, Vt = np.linalg.svd(pooled[::10] - mu, full_matrices=False)
P = Vt[: args.n_pca].T
expl = float((S[: args.n_pca] ** 2).sum() / (S ** 2).sum())
seqs = [jax.numpy.asarray((x - mu) @ P, dtype=jax.numpy.float32) for x in seqs]
hmm = GaussianHMM(HMMConfig(n_states=args.n_states, n_channels=args.n_pca, covariance_type="full"))
ll = hmm.fit(seqs, n_epochs=args.hmm_epochs, n_init=3, key=jax.random.PRNGKey(0))
gammas = [np.asarray(g) for g in hmm.infer(seqs)]
print(f"[hmm] TDE {args.n_embeddings} lags, PCA {args.n_pca} ({expl:.0%} var), K={args.n_states}, "
      f"final LL/sample {ll[-1]/len(pooled):.3f}, {time.time()-t0:.0f}s", flush=True)

rows = []
h = args.n_embeddings // 2
for (s, cond), g in zip(keys, gammas):
    st = g.argmax(1)
    T = len(st)
    n, hop = int(args.win * fs), int(args.step * fs)
    Tmax = T if cond == "EC" else min(T, int(args.max_t * fs))
    for s0 in range(0, Tmax - n + 1, hop):
        seg = st[s0:s0 + n]
        occ = np.bincount(seg, minlength=args.n_states) / n
        switches = int(np.sum(np.diff(seg) != 0))
        row = dict(subject=s, condition=cond, t=(s0 + h) / fs, switching_rate=switches / args.win,
                   mean_lifetime=float(n / (switches + 1) / fs))
        row.update({f"occ_{k}": occ[k] for k in range(args.n_states)})
        rows.append(row)
H = pd.DataFrame(rows)
H.to_csv(out / "windows_hmm.csv", index=False)
np.savez(out / "hmm_params.npz", means=np.asarray(hmm.means), pca=P, pca_mean=mu, trans=np.asarray(hmm.trans_prob)
         if hasattr(hmm, "trans_prob") else np.zeros(0), ll=np.asarray(ll))

# ---------------------------------------------------------------- z vs EC baseline, group curves
feat_cols = list(measures) + [c for c in H.columns if c.startswith("occ_")] + ["switching_rate", "mean_lifetime"]
W = M.merge(H, on=["subject", "condition", "t"], how="outer")
Z = W.copy()
# per-subject EC SD with a floor at 20 % of the pooled EC SD, so features that are
# nearly constant in a subject's baseline (e.g. an unvisited HMM state) do not blow up
pooled_sd = W[W.condition == "EC"][feat_cols].std(ddof=1)
for s in subjects:
    ec = W[(W.subject == s) & (W.condition == "EC")]
    mu_s = ec[feat_cols].mean()
    sd_s = np.maximum(ec[feat_cols].std(ddof=1).fillna(0), 0.2 * pooled_sd).replace(0, np.nan)
    sel = W.subject == s
    Z.loc[sel, feat_cols] = (W.loc[sel, feat_cols] - mu_s) / sd_s
Z.to_csv(out / "windows_z_vs_EC.csv", index=False)
D = Z[Z.condition == "DMT"].copy()
D["t_bin"] = (D.t // 30 * 30).astype(int)
G = D.groupby("t_bin")[feat_cols].agg(["mean", "sem", "count"])
G.to_csv(out / "group_curves_30s.csv")

# ---------------------------------------------------------------- Bateman PD fits
def bateman(t, b, A, ta, te):
    return b + A * (np.exp(-t / te) - np.exp(-t / ta))


def fit_bateman(t, y):
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 8:
        return dict(R2=np.nan)
    try:
        p, _ = curve_fit(bateman, t, y, p0=[y[0], (y.max() - y[0]) * 2, 30.0, 150.0],
                         bounds=([-np.inf, -np.inf, 2.0, 10.0], [np.inf, np.inf, 300.0, 2000.0]), maxfev=4000)
    except Exception:
        return dict(R2=np.nan)
    b, A, ta, te = p
    yhat = bateman(t, *p)
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    tmax = np.log(te / ta) * ta * te / (te - ta) if te != ta else ta
    return dict(b=b, A=A, tau_abs=ta, tau_elim=te, t_peak=tmax, R2=r2)


pd_rows = []
for s in subjects:
    d = D[D.subject == s]
    for f in feat_cols:
        r = fit_bateman(d.t.to_numpy(float), d[f].to_numpy(float))
        pd_rows.append(dict(subject=s, feature=f, **r))
PD = pd.DataFrame(pd_rows)
PD.to_csv(out / "bateman_fits.csv", index=False)
good = PD[PD.R2 > 0.3]
summ = good.groupby("feature").agg(n_fit=("R2", "count"), median_R2=("R2", "median"), t_peak_median=("t_peak", "median"),
                                   t_peak_iqr=("t_peak", lambda v: np.subtract(*np.percentile(v, [75, 25]))),
                                   A_median=("A", "median")).reset_index()
summ["frac_subjects_fit"] = summ.n_fit / len(subjects)
summ = summ.sort_values("frac_subjects_fit", ascending=False)
summ.to_csv(out / "bateman_summary.csv", index=False)

# onset / offset on the group z curve (|mean z| > 1 for the first/last time)
on_rows = []
for f in feat_cols:
    g = D.groupby("t")[f].mean()
    above = np.abs(g.to_numpy()) > 1.0
    if above.any():
        tt = g.index.to_numpy()
        on_rows.append(dict(feature=f, onset_s=float(tt[np.argmax(above)]), offset_s=float(tt[len(above) - 1 - np.argmax(above[::-1])]),
                            peak_z=float(g.to_numpy()[np.nanargmax(np.abs(g.to_numpy()))])))
ON = pd.DataFrame(on_rows).sort_values("peak_z", key=np.abs, ascending=False)
ON.to_csv(out / "onset_offset.csv", index=False)

pd.set_option("display.width", 220)
print("\n[bateman] features by fraction of subjects with R² > 0.3:")
print(summ.head(20).to_string(index=False, float_format="%.2f"))
print("\n[onset/offset] |group z| > 1:")
print(ON.head(20).to_string(index=False, float_format="%.1f"))

# ---------------------------------------------------------------- figure
show = [m for m in ("specparam_alpha_power", "specparam_exponent", "relpow_delta", "lz76_mean", "mse_sum", "permutation_entropy",
                    "pair_correlation", "rn_transitivity", "dfa") if m in feat_cols]
occs = [c for c in feat_cols if c.startswith("occ_")] + ["switching_rate"]
fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
tb = G.index.to_numpy() + 15
for m in show:
    axes[0].errorbar(tb, G[(m, "mean")], G[(m, "sem")], label=m, capsize=2)
axes[0].axhline(0, color="k", lw=0.5); axes[0].set_ylabel("z vs own EC baseline"); axes[0].legend(fontsize=7, ncol=3)
axes[0].set_title(f"DMT time course, n={len(subjects)} (t=0 exhalation; {args.win:g} s windows, {args.step:g} s step)")
for m in occs:
    axes[1].errorbar(tb, G[(m, "mean")], G[(m, "sem")], label=m, capsize=2)
axes[1].axhline(0, color="k", lw=0.5); axes[1].set_ylabel("TDE-HMM occupancy / switching, z vs EC"); axes[1].set_xlabel("s after exhalation")
axes[1].legend(fontsize=7, ncol=4)
fig.tight_layout(); fig.savefig(out / "timecourse.png", dpi=130)
print(f"[done] -> {out}")
