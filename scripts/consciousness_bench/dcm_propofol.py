"""Spectral DCM on the Chennu 2016 propofol data (first real-data test of the GABA-A handle).

Per subject: the leading spatial mode (PCA of the baseline recording, same
weights for every sedation level) gives one sensor series per condition; a
multitaper CSD averaged over 10 s epochs (1-45 Hz, 0.5 Hz steps) is
inverted with a one-region ``SpectralDCM`` (Jansen-Rit, or canonical microcircuit with ``--model cmc``).  Outputs under
``$CB_OUT/propofol-chennu2016/dcm/``:

- ``posteriors.csv``  one row per subject x condition x parameter
- ``fits.csv``        free energy, noise precision, scale, starts
- ``spectra.csv``     observed vs predicted log power per frequency
- ``contrasts.csv``   paired contrast of each parameter vs baseline

Usage::

    python scripts/consciousness_bench/dcm_propofol.py [--subjects 02 03] [--n-jobs 20]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

jax.config.update("jax_enable_x64", True)

from neurojax.analysis.multitaper import multitaper_cpsd  # noqa: E402
from neurojax.bench.consciousness_bench import epoch_recording  # noqa: E402
from neurojax.inverse.csd_dcm import CMCSpec, JansenRitSpec, SpectralDCM  # noqa: E402
from neurojax.io.consciousness_bench import load_chennu_propofol  # noqa: E402

FREQS = np.arange(1.0, 45.5, 0.5)
CONDITIONS = ["baseline", "mild", "moderate", "recovery"]


def leading_mode(data: np.ndarray) -> np.ndarray:
    """Unit-norm channel weights of the first principal component of (C, T) data."""
    X = data - data.mean(axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(X.T, full_matrices=False)
    w = Vt[0]
    return w * np.sign(w[np.argmax(np.abs(w))])


def mode_csd(data: np.ndarray, fs: float, w: np.ndarray, epoch_s: float = 10.0) -> np.ndarray:
    """Multitaper CSD (n_freq, 1, 1) of the spatial mode, averaged over epochs and interpolated to FREQS."""
    from neurojax.bench.consciousness_bench import Recording

    rec = Recording("x", "x", "x", data, fs)
    ep = epoch_recording(rec, epoch_s=epoch_s)  # (E, C, T)
    series = np.einsum("c,ect->et", w, ep)  # (E, T)
    acc = None
    for s in series:
        cpsd, f = multitaper_cpsd(jnp.asarray(s[:, None]), fs=fs, bandwidth=1.0)
        acc = np.asarray(cpsd) if acc is None else acc + np.asarray(cpsd)
    acc = acc / len(series)
    f = np.asarray(f)
    out = np.interp(FREQS, f, acc[:, 0, 0].real)
    return out[:, None, None].astype(complex)


def fit_subject(subject: str, recs: list, n_iter: int, n_starts: int, model: str = "jr", out: Path | None = None) -> tuple[list, list, list]:
    spec = JansenRitSpec(n_regions=1) if model == "jr" else CMCSpec(n_regions=1)
    dcm = SpectralDCM(spec, leadfield=jnp.eye(1), freqs=jnp.asarray(FREQS))
    by_cond = {r.condition: r for r in recs}
    if "baseline" not in by_cond:
        return [], [], []
    w = leading_mode(by_cond["baseline"].data)
    post, fits, spectra = [], [], []
    for cond in CONDITIONS:
        if cond not in by_cond:
            continue
        r = by_cond[cond]
        csd = mode_csd(r.data, r.fs, w)
        t0 = time.time()
        res = dcm.invert(jnp.asarray(csd), n_iter=n_iter, n_starts=n_starts)
        dt = time.time() - t0
        sd = np.sqrt(np.diag(np.asarray(res.Cp)))
        if out is not None:
            np.savez(out / f"posterior_{subject}_{cond}.npz", Ep=np.asarray(res.Ep), Cp=np.asarray(res.Cp), pE=np.asarray(dcm.prior_mean),
                     pC=np.asarray(dcm.prior_cov), names=dcm.names, F=float(res.F))
        for name, m, s in zip(dcm.names, np.asarray(res.Ep), sd):
            post.append(dict(subject=subject, condition=cond, propofol_ugml=r.meta.get("propofol_ugml"),
                             param=name, mean=float(m), sd=float(s)))
        fits.append(dict(subject=subject, condition=cond, propofol_ugml=r.meta.get("propofol_ugml"),
                         F=float(res.F), h=float(res.h), scale=float(res.scale), n_iter=len(res.F_trace),
                         F_starts=" ".join(f"{x:.1f}" for x in res.F_starts), seconds=dt,
                         n_epochs=int(r.data.shape[1] // int(10 * r.fs))))
        obs = np.log(csd[:, 0, 0].real * res.scale)
        pred = np.log(np.asarray(res.prediction)[:, 0, 0].real * res.scale)
        for f, o, p in zip(FREQS, obs, pred):
            spectra.append(dict(subject=subject, condition=cond, f=float(f), observed=float(o), predicted=float(p)))
        key = "tau_i[0]" if model == "jr" else "T[0, 2]"
        print(f"{subject} {cond:9s} F={float(res.F):8.1f} {key}={dict(zip(dcm.names, np.asarray(res.Ep)))[key]:+.3f} {dt:5.1f}s", flush=True)
    return post, fits, spectra


def contrasts(post: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wide = post.pivot_table(index=["subject", "param"], columns="condition", values="mean")
    for cond in CONDITIONS[1:]:
        if cond not in wide:
            continue
        for param, g in wide.groupby(level="param"):
            g = g.dropna(subset=["baseline", cond])
            if len(g) < 3:
                continue
            diff = (g[cond] - g["baseline"]).to_numpy()
            d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
            try:
                p = stats.wilcoxon(diff).pvalue
            except ValueError:
                p = np.nan
            rows.append(dict(contrast=f"{cond}-baseline", param=param, n=len(g), mean_diff=diff.mean(),
                             cohen_d=d, p_wilcoxon=p))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--n-jobs", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=64)
    ap.add_argument("--n-starts", type=int, default=4)
    ap.add_argument("--model", choices=["jr", "cmc"], default="jr", help="column model; outputs go to dcm/ (jr) or dcm_cmc/")
    ap.add_argument("--out", default=os.environ.get("CB_OUT", "/data/datasets/consciousness-bench"))
    a = ap.parse_args()
    out = Path(a.out) / "propofol-chennu2016" / ("dcm" if a.model == "jr" else "dcm_cmc")
    out.mkdir(parents=True, exist_ok=True)

    recs = load_chennu_propofol(subjects=a.subjects)
    subjects = sorted({r.subject for r in recs})
    print(f"{len(recs)} recordings, {len(subjects)} subjects", flush=True)
    results = Parallel(n_jobs=a.n_jobs)(
        delayed(fit_subject)(s, [r for r in recs if r.subject == s], a.n_iter, a.n_starts, a.model, out) for s in subjects
    )
    post = pd.DataFrame([x for r in results for x in r[0]])
    fits = pd.DataFrame([x for r in results for x in r[1]])
    spectra = pd.DataFrame([x for r in results for x in r[2]])
    post.to_csv(out / "posteriors.csv", index=False)
    fits.to_csv(out / "fits.csv", index=False)
    spectra.to_csv(out / "spectra.csv", index=False)
    con = contrasts(post)
    con.to_csv(out / "contrasts.csv", index=False)
    if con.empty:
        print("fewer than 3 subjects: no contrasts")
    else:
        print(con.sort_values(["contrast", "cohen_d"]).to_string(index=False))


if __name__ == "__main__":
    main()
