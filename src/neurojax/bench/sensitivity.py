# © NeuroJAX developers
#
# License: BSD (3-clause)
"""Sensitivity of the complexity benchmark to low-frequency handling and
epoch length.

Most NLTSA measures are threshold-crossing counts or scaling-range fits, so
the high-pass cutoff, the detrending method and the epoch length fix a
large part of the number before any drug does. :data:`PREPROCS` holds the
alternatives — Butterworth high-pass at several cutoffs, linear detrending,
EMD (drop the IMFs whose mean instantaneous frequency is below a cutoff plus
the residue; Huang 1998), and Hankel-DMD low-mode removal (fit exact DMD on
a delay-embedded channel, drop the modes whose frequency is below the cutoff;
Schmid 2010, Brunton 2017) — and :func:`sensitivity_table` scores the same
contrast under every (preprocessing × epoch length) and reports per measure
the effect size and the rank stability of the subject-level changes against
a reference setting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, hilbert, sosfiltfilt
from scipy.stats import spearmanr

from neurojax.bench.consciousness_bench import Recording, contrast, score_recordings
from neurojax.bench.measure_overlap import compare_overlap, delta_table, overlap_matrix

__all__ = ["PREPROCS", "hpf", "emd_detrend", "dmd_lowmode_removal", "sensitivity_table"]


def hpf(X: np.ndarray, fs: float, f_cut: float, order: int = 4) -> np.ndarray:
    sos = butter(order, f_cut, btype="high", fs=fs, output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def emd_detrend(X: np.ndarray, fs: float, f_cut: float = 0.5, max_imf: int = 12) -> np.ndarray:
    """Remove, per channel, the EMD residue and every IMF whose mean
    instantaneous frequency (Hilbert) is below ``f_cut`` Hz."""
    from PyEMD import EMD

    out = np.empty_like(X, dtype=np.float32)
    for c in range(X.shape[0]):
        x = np.asarray(X[c], float)
        imfs = EMD(spline_kind="cubic")(x, max_imf=max_imf)
        keep = np.zeros_like(x)
        for imf in imfs:
            ph = np.unwrap(np.angle(hilbert(imf)))
            f_mean = np.abs(np.diff(ph)).mean() * fs / (2 * np.pi)
            if f_mean >= f_cut:
                keep += imf
        out[c] = keep
    return out


def dmd_lowmode_removal(X: np.ndarray, fs: float, f_cut: float = 0.5, fs_low: float = 8.0, rank: int = 60,
                        delay: int | None = None, win_s: float = 15.0, noise_factor: float = 1.5) -> np.ndarray:
    """Hankel (delay-embedded) exact DMD low-mode removal per channel.

    Each ``win_s`` window is linearly detrended, decimated to ``fs_low``
    (anti-aliased) and delay-embedded with ``delay`` = 2·fs_low/f_cut samples
    (two periods of the cutoff; shorter delays cannot resolve slow modes).
    Exact DMD with SVD rank = #singular values above ``noise_factor`` × median
    (noise floor), capped at ``rank``; modes with |frequency| < ``f_cut`` Hz and
    |λ|^m within [0.1, 10] over the window (neither decayed nor explosive)
    are reconstructed with closed-form optimal amplitudes (Jovanović, Schmid
    & Nichols 2014), interpolated back to ``fs`` and subtracted together with
    the linear trend."""
    from scipy.interpolate import CubicSpline
    from scipy.signal import decimate

    q = max(1, int(round(fs / fs_low)))
    fs_l = fs / q
    delay = int(2 * fs_l / f_cut) if delay is None else delay
    n = int(win_s * fs)
    out = np.empty_like(X, dtype=np.float32)
    for c in range(X.shape[0]):
        x = np.asarray(X[c], float)
        res = np.empty_like(x)
        for s0 in range(0, len(x), n):
            seg = x[s0:s0 + n]
            if len(seg) < 2 * n // 3 and s0 > 0:  # short tail: merge into previous window's treatment
                seg_d = detrend(seg)
                res[s0:s0 + n] = seg_d
                continue
            seg_d = detrend(seg)
            xl = decimate(seg_d, q, ftype="fir", zero_phase=True) if q > 1 else seg_d
            t_low = np.arange(len(xl)) * q / fs
            t_full = np.arange(len(seg)) / fs
            m = len(xl) - delay + 1
            if m < 2 * delay:
                res[s0:s0 + n] = seg_d
                continue
            H = np.stack([xl[i:i + m] for i in range(delay)])
            X1, X2 = H[:, :-1], H[:, 1:]
            U, S, Vh = np.linalg.svd(X1, full_matrices=False)
            r = max(1, min(rank, int(np.sum(S > noise_factor * np.median(S)))))
            U, S, Vh = U[:, :r], S[:r], Vh[:r]
            A = U.conj().T @ X2 @ Vh.conj().T / S
            lam, W = np.linalg.eig(A)
            Phi = X2 @ Vh.conj().T / S @ W
            freq = np.abs(np.angle(lam)) * fs_l / (2 * np.pi)
            growth = np.abs(lam) ** m
            low = (freq < f_cut) & (growth > 0.1) & (growth < 10)
            if not low.any():
                res[s0:s0 + n] = seg_d
                continue
            Vand = lam[:, None] ** np.arange(m)[None, :]
            P = (Phi.conj().T @ Phi) * np.conj(Vand @ Vand.conj().T)
            qv = np.conj(np.diag(Vand @ H.conj().T @ Phi))
            bamp = np.linalg.lstsq(P, qv, rcond=None)[0]
            low_sig = (Phi[:, low] @ (bamp[low, None] * Vand[low])).real
            acc, cnt = np.zeros(len(xl)), np.zeros(len(xl))
            for i in range(delay):
                acc[i:i + m] += low_sig[i]
                cnt[i:i + m] += 1
            slow = acc / np.maximum(cnt, 1)
            res[s0:s0 + n] = seg_d - CubicSpline(t_low, slow)(np.clip(t_full, t_low[0], t_low[-1]))
        out[c] = res
    return out


PREPROCS = {
    "demean": lambda X, fs: (X - X.mean(-1, keepdims=True)).astype(np.float32),
    "detrend_linear": lambda X, fs: detrend(X, axis=-1).astype(np.float32),
    "hpf_0.1": lambda X, fs: hpf(X, fs, 0.1),
    "hpf_0.5": lambda X, fs: hpf(X, fs, 0.5),
    "hpf_1.0": lambda X, fs: hpf(X, fs, 1.0),
    "hpf_2.0": lambda X, fs: hpf(X, fs, 2.0),
    "emd_0.5": lambda X, fs: emd_detrend(X, fs, 0.5),
    "dmd_0.5": lambda X, fs: dmd_lowmode_removal(X, fs, 0.5),
}


def sensitivity_table(recordings: list[Recording], measures, cond_a: str, cond_b: str,
                      preprocs: dict | None = None, epoch_lengths=(2.0, 5.0, 10.0, 30.0), total_s: float = 60.0,
                      reference=("hpf_0.5", 10.0), n_jobs: int = 1) -> pd.DataFrame:
    """Score ``cond_b − cond_a`` under every preprocessing × epoch length
    (``total_s`` seconds of data per recording in every setting). Returns a
    long table with, per (preproc, epoch_s, measure): the paired Cohen's d,
    Wilcoxon p, AUC, ``rho_ref`` = Spearman rank correlation of the
    subject-level changes with the reference setting, and ``mantel_ref`` =
    Mantel r of the measure-overlap matrix against the reference setting."""
    preprocs = preprocs or PREPROCS
    deltas, overlaps, rows = {}, {}, []
    for pname, pf in preprocs.items():
        pre = [Recording(r.dataset, r.subject, r.condition, pf(r.data, r.fs), r.fs, r.channels, r.meta)
               for r in recordings]
        for ep in epoch_lengths:
            df = score_recordings(pre, measures, epoch_s=ep, max_epochs=max(1, int(total_s // ep)), n_jobs=n_jobs)
            c = contrast(df, cond_a, cond_b).set_index("measure")
            deltas[(pname, ep)] = delta_table(df, cond_a, cond_b)
            overlaps[(pname, ep)] = overlap_matrix(df, cond_a, cond_b)
            for m in c.index:
                rows.append(dict(preproc=pname, epoch_s=ep, measure=m, cohen_d=c.loc[m, "cohen_d"],
                                 p_wilcoxon=c.loc[m, "p_wilcoxon"], auc=c.loc[m, "auc"], n=c.loc[m, "n"]))
    T = pd.DataFrame(rows)
    ref = reference if reference in deltas else next(iter(deltas))
    dref, oref = deltas[ref], overlaps[ref]
    rho, mantel = [], {}
    for (pname, ep) in deltas:
        mantel[(pname, ep)] = compare_overlap(oref, overlaps[(pname, ep)], n_perm=200)["mantel_r"]
    for _, r in T.iterrows():
        d = deltas[(r.preproc, r.epoch_s)]
        common = d.index.intersection(dref.index)
        if r.measure in d.columns and r.measure in dref.columns and len(common) > 3:
            rho.append(spearmanr(d.loc[common, r.measure], dref.loc[common, r.measure], nan_policy="omit").statistic)
        else:
            rho.append(np.nan)
    T["rho_ref"] = rho
    T["mantel_ref"] = [mantel[(p, e)] for p, e in zip(T.preproc, T.epoch_s)]
    T.attrs["reference"] = ref
    T.attrs["deltas"] = deltas
    return T
