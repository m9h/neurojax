# © NeuroJAX developers
#
# License: BSD (3-clause)
"""Benchmark harness: complexity / NLTSA measures across states of consciousness.

Modelled on the McGill/Milan/Liège comparison of criticality and complexity
measures under propofol, xenon and ketamine (Maschke et al. 2024, *Commun.
Biol.*) and extended to psychedelic EEG (DMT, psilocybin). Each measure in
:data:`neurojax.analysis.nltsa.MEASURES` is scored per epoch and channel, then
summarised per subject × condition; :func:`contrast` gives paired effect
sizes, and :func:`decompose_contrast` splits every measure's between-condition
change into the part reproduced by phase-randomised surrogates (spectral) and
the remainder (phasic; Mediano et al. 2020) — the control the bare measure
comparisons lack.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import jax
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu, wilcoxon

import jax.numpy as jnp

from neurojax.analysis.nltsa import MEASURES, Measure
from neurojax.analysis.nltsa_batched import BATCHED, accelerator_available, score_batched
from neurojax.analysis.surrogates import multivariate_phase_surrogate

__all__ = ["Recording", "epoch_recording", "score_epochs", "score_recording", "score_recordings",
           "spectral_explained", "contrast", "decompose_contrast"]


@dataclass
class Recording:
    dataset: str
    subject: str
    condition: str
    data: np.ndarray  # (C, T)
    fs: float
    channels: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def epoch_recording(rec: Recording, epoch_s: float = 10.0, max_epochs: int | None = None) -> np.ndarray:
    """Non-overlapping (E, C, T_epoch) epochs; a trailing partial epoch is dropped
    and, with ``max_epochs``, epochs are taken evenly across the recording."""
    n = int(round(epoch_s * rec.fs))
    E = rec.data.shape[1] // n
    ep = rec.data[:, : E * n].reshape(rec.data.shape[0], E, n).transpose(1, 0, 2)
    if max_epochs is not None and E > max_epochs:
        ep = ep[np.linspace(0, E - 1, max_epochs).round().astype(int)]
    return ep


def _score_epoch(X, fs, measures):
    out = {}
    for name, m in measures.items():
        if m.multivariate:
            out[name] = m(X, fs)
        else:
            out[name] = float(np.nanmean([m(X[c], fs) for c in range(X.shape[0])]))  # one flat channel must not void the epoch
    return out


def score_epochs(epochs, fs: float, measures: dict[str, Measure] | None = None, n_jobs: int = 1,
                 batched: bool | None = None) -> list[dict]:
    """One measure dict per (C, T) epoch. With ``batched`` (default: whenever the
    default JAX device is an accelerator) the O(T²) measures in
    :data:`nltsa_batched.BATCHED` are scored on the device over all epochs ×
    channels in one call and the rest go through joblib on the CPU; the joblib
    workers are pinned to the CPU backend so they do not each claim the GPU."""
    measures = measures or MEASURES
    epochs = list(epochs)
    if batched is None:
        batched = accelerator_available()
    names_b = [n for n in measures if n in BATCHED] if batched else []
    rest = {n: m for n, m in measures.items() if n not in names_b}
    rows = [dict() for _ in epochs]
    if names_b and epochs and accelerator_available():
        os.environ["JAX_PLATFORMS"] = "cpu"  # inherited by loky workers spawned from here on; this process keeps its device

    def cpu_part():
        if rest and epochs:
            return Parallel(n_jobs=n_jobs)(delayed(_score_epoch)(X, fs, rest) for X in epochs)
        return [{} for _ in epochs]

    # the joblib pool and the device run concurrently (XLA releases the GIL while it waits)
    with ThreadPoolExecutor(1) as pool:
        fut = pool.submit(cpu_part)
        if names_b and epochs:
            by_shape: dict[tuple, list[int]] = {}
            for i, X in enumerate(epochs):
                by_shape.setdefault(tuple(np.shape(X)), []).append(i)
            for (C, T), idx in by_shape.items():
                stack = np.stack([epochs[i] for i in idx]).reshape(-1, T)  # (E·C, T)
                out = score_batched(stack, fs, {n: measures[n] for n in names_b})
                for n, v in out.items():
                    v = np.nanmean(v.reshape(len(idx), C), axis=1)  # channel mean, as _score_epoch
                    for j, i in enumerate(idx):
                        rows[i][n] = float(v[j])
        for r, v in zip(rows, fut.result()):
            r.update(v)
    return rows


def score_recording(rec: Recording, measures: dict[str, Measure] | None = None, epoch_s: float = 10.0,
                    max_epochs: int | None = None, n_jobs: int = 1, batched: bool | None = None) -> dict[str, np.ndarray]:
    """Per-epoch values (E,) of each measure; univariate measures averaged over channels."""
    measures = measures or MEASURES
    ep = epoch_recording(rec, epoch_s, max_epochs)
    rows = score_epochs(ep, rec.fs, measures, n_jobs, batched)
    return {k: np.array([r[k] for r in rows]) for k in measures}


def score_recordings(recordings: list[Recording], measures=None, epoch_s=10.0, max_epochs=None,
                      n_jobs: int = 1, batched: bool | None = None) -> pd.DataFrame:
    """Long table: dataset, subject, condition, measure, value (mean over epochs), sd, n_epochs.
    All recordings' epochs go through one :func:`score_epochs` call so the device
    sees one large batch and the joblib pool one long job list."""
    measures = measures or MEASURES
    epochs, owner = [], []
    for i, rec in enumerate(recordings):
        for X in epoch_recording(rec, epoch_s, max_epochs):
            epochs.append(X)
            owner.append(i)
    fss = {r.fs for r in recordings}
    assert len(fss) <= 1, f"mixed sampling rates: {fss}"
    vals = score_epochs(epochs, fss.pop() if fss else 0.0, measures, n_jobs, batched)
    owner = np.asarray(owner)
    rows = []
    for i, rec in enumerate(recordings):
        mine = [vals[j] for j in np.flatnonzero(owner == i)]
        for k in measures:
            v = np.array([r[k] for r in mine]) if mine else np.array([])
            rows.append(dict(dataset=rec.dataset, subject=rec.subject, condition=rec.condition,
                             measure=k, value=float(np.nanmean(v)) if v.size else np.nan,
                             sd=float(np.nanstd(v)) if v.size else np.nan,
                             n_epochs=int(np.sum(np.isfinite(v)))))
    return pd.DataFrame(rows)


def contrast(df: pd.DataFrame, cond_a: str, cond_b: str) -> pd.DataFrame:
    """Per measure, ``cond_b`` − ``cond_a`` across subjects: paired Cohen's d
    (mean diff / SD of diffs), Wilcoxon signed-rank p, AUC (unpaired
    Mann-Whitney), fraction of subjects increasing. Subjects missing either
    condition are dropped (``n`` reports the pairs used)."""
    rows = []
    for meas, g in df.groupby("measure"):
        a = g[g.condition == cond_a].set_index("subject")["value"]
        b = g[g.condition == cond_b].set_index("subject")["value"]
        common = a.index.intersection(b.index)
        a, b = a.loc[common].to_numpy(), b.loc[common].to_numpy()
        diff = b - a
        n = len(diff)
        d = float(diff.mean() / diff.std(ddof=1)) if n > 1 and diff.std(ddof=1) > 0 else np.nan
        p = float(wilcoxon(diff).pvalue) if n > 1 and np.any(diff != 0) else np.nan
        u = mannwhitneyu(b, a).statistic if n > 0 else np.nan
        rows.append(dict(measure=meas, n=n, mean_a=a.mean(), mean_b=b.mean(), mean_diff=diff.mean(),
                         cohen_d=d, p_wilcoxon=p, auc=float(u / (n * n)) if n else np.nan,
                         frac_increase=float(np.mean(diff > 0)) if n else np.nan))
    return pd.DataFrame(rows).sort_values("cohen_d", key=np.abs, ascending=False).reset_index(drop=True)


def decompose_contrast(recordings: list[Recording], measures, cond_a: str, cond_b: str, key: jax.Array,
                       epoch_s: float = 10.0, max_epochs: int | None = None, n_surrogates: int = 20,
                       channels: list[int] | None = None, n_jobs: int = 1) -> pd.DataFrame:
    """Spectral/phasic decomposition of each measure's ``cond_b − cond_a``
    change, per subject, then averaged: the feature is the epoch- and
    channel-mean measure of a recording, and surrogates are Prichard-Theiler
    multivariate phase surrogates of the whole recording (per-channel spectra
    and cross-spectra kept; phase structure and nonstationarity destroyed).
    ``spectral_fraction`` = spectral / total.
    Restrict ``channels`` for the O(T²) measures."""
    measures = measures or MEASURES
    by_subj: dict[str, dict[str, Recording]] = {}
    for r in recordings:
        by_subj.setdefault(r.subject, {})[r.condition] = r
    pairs = [(s, c[cond_a], c[cond_b]) for s, c in by_subj.items() if cond_a in c and cond_b in c]
    sel = slice(None) if channels is None else channels

    # One flat job list over (subject, condition, original + surrogates, epoch) so
    # the parallel pool is saturated; a per-subject loop would only parallelise
    # over the handful of epochs of one recording.
    jobs, index = [], []
    for s, ra, rb in pairs:
        key, ka, kb = jax.random.split(key, 3)
        for cond, rec, kc in ((cond_a, ra, ka), (cond_b, rb, kb)):
            X = rec.data[sel]
            # Prichard-Theiler surrogates: one phase rotation shared by all channels,
            # so cross-spectra survive and the multivariate measures see a fair null
            variants = [X] + [np.asarray(multivariate_phase_surrogate(jnp.asarray(X.T), k)).T
                              for k in jax.random.split(kc, n_surrogates)]
            for v, Xv in enumerate(variants):
                for X_ep in epoch_recording(Recording("", s, cond, Xv, rec.fs), epoch_s, max_epochs):
                    jobs.append((X_ep, rec.fs))
                    index.append((s, cond, v))
    fss = {fs for _, fs in jobs}
    assert len(fss) <= 1, f"mixed sampling rates in one decomposition: {fss}"
    vals = score_epochs([X for X, _ in jobs], fss.pop() if fss else 0.0, measures, n_jobs)
    names = list(measures)
    table = pd.DataFrame(index, columns=["subject", "condition", "variant"])
    for m in names:
        table[m] = [v[m] for v in vals]
    feat = table.groupby(["subject", "condition", "variant"])[names].mean()  # epoch mean

    rows = []
    for s, _, _ in pairs:
        fa, fb = feat.loc[(s, cond_a)], feat.loc[(s, cond_b)]
        oa, ob = fa.loc[0].to_numpy(), fb.loc[0].to_numpy()
        sa, sb = fa.loc[1:].to_numpy(), fb.loc[1:].to_numpy()
        total = ob - oa
        spectral = sb.mean(0) - sa.mean(0)
        va, vb = sa.var(0, ddof=1), sb.var(0, ddof=1)
        spectral_std = np.sqrt((va + vb) / n_surrogates)
        phasic_std = np.sqrt((va + vb) * (1.0 + 1.0 / n_surrogates))  # as complexity.spectral_phase_decomposition
        for i, m in enumerate(names):
            rows.append(dict(subject=s, measure=m, total=total[i], spectral=spectral[i],
                             phasic=total[i] - spectral[i], spectral_std=spectral_std[i],
                             phasic_std=phasic_std[i]))
    per = pd.DataFrame(rows)
    out = per.groupby("measure").agg(n=("subject", "count"), total=("total", "mean"),
                                     spectral=("spectral", "mean"), phasic=("phasic", "mean"),
                                     phasic_std=("phasic_std", lambda v: float(np.sqrt(np.mean(np.square(v)) / len(v)))),
                                     spectral_std=("spectral_std", lambda v: float(np.sqrt(np.mean(np.square(v)) / len(v))))
                                     ).reset_index()
    out["spectral_fraction"] = out["spectral"] / out["total"]
    out["phasic_z"] = out["phasic"] / out["phasic_std"]
    out.attrs["per_subject"] = per
    return out


def spectral_explained(scores: pd.DataFrame, spectral_scores: pd.DataFrame, cond_a: str, cond_b: str,
                       predictors=("specparam_exponent", "specparam_offset", "specparam_alpha_power",
                                   "relpow_delta", "relpow_theta", "relpow_beta")) -> pd.DataFrame:
    """How much of each NLTSA measure is linear in the standard spectral
    parameters. Per measure, over all subject × condition rows: ``r2`` of an
    OLS fit on ``predictors`` (z-scored, intercept), then the paired
    ``cond_b − cond_a`` Cohen's d of the measure (``d_raw``) and of its
    residual after that fit (``d_residual``). ``d_residual`` ≈ 0 means FOOOF
    and band powers already carry the contrast."""
    wide = scores.pivot_table(index=["subject", "condition"], columns="measure", values="value")
    spec = spectral_scores.pivot_table(index=["subject", "condition"], columns="measure", values="value")
    preds = [p for p in predictors if p in spec.columns]
    wide = wide[[c for c in wide.columns if c in MEASURES]]  # a combined scores table also holds the predictors
    both = wide.join(spec[preds], how="inner").dropna(subset=preds)
    Z = (both[preds] - both[preds].mean()) / both[preds].std(ddof=1)
    A = np.column_stack([np.ones(len(Z)), Z.to_numpy()])
    rows = []
    for m in wide.columns:
        y = both[m].to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() < len(preds) + 3:
            continue
        beta, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
        resid = np.full_like(y, np.nan)
        resid[ok] = y[ok] - A[ok] @ beta
        r2 = 1 - np.nanvar(resid) / np.nanvar(y)
        tmp = both[[m]].copy()
        tmp["resid"] = resid
        d_raw = _paired_d(tmp[m], cond_a, cond_b)
        d_res = _paired_d(tmp["resid"], cond_a, cond_b)
        rows.append(dict(measure=m, n=int(ok.sum()), r2=float(r2), d_raw=d_raw, d_residual=d_res,
                         **{f"beta_{p}": float(b) for p, b in zip(preds, beta[1:])}))
    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)


def _paired_d(series: pd.Series, cond_a: str, cond_b: str) -> float:
    a = series.xs(cond_a, level="condition")
    b = series.xs(cond_b, level="condition")
    common = a.index.intersection(b.index)
    diff = (b.loc[common] - a.loc[common]).dropna()
    return float(diff.mean() / diff.std(ddof=1)) if len(diff) > 1 and diff.std(ddof=1) > 0 else np.nan
