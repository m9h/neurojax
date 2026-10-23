# © NeuroJAX developers
#
# License: BSD (3-clause)
"""Nonlinear time-series and complexity measures with a named registry.

This is the measure set of the McGill/Milan/Liège comparison of resting EEG
under propofol, xenon and ketamine (Maschke et al. 2024, *Commun. Biol.*
7:946: Lempel-Ziv, Katz FD, multiscale entropy, DFA on band envelopes,
aperiodic slope, 0-1 chaos test, largest Lyapunov exponent, pair correlation
function, avalanche branching ratio), plus the complexity-entropy causal plane
of Newman et al. 2025 (permutation entropy × Jensen-Shannon statistical
complexity), and the standard univariate NLTSA measures those papers cite
(Higuchi/Petrosian FD, Hjorth, sample/SVD/spectral entropy, correlation
dimension, RQA determinism).

Every measure is a plain function ``f(x, fs=..., **params) -> float`` on a
(T,) signal (or (C, T) for ``multivariate`` ones); :data:`MEASURES` registers
each with its family, defaults and reference so a benchmark reports exactly
which estimator and parameters produced a number. Pair with
:func:`neurojax.analysis.complexity.spectral_phase_decomposition` to split any
of them into spectral and phasic parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache, partial
from itertools import permutations
from math import factorial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch

from neurojax.analysis.complexity import lz_complexity
from neurojax.analysis.entropy import _chebyshev_count, _embed, spectral_entropy, svd_entropy

__all__ = [
    "FAMILIES", "MEASURES", "Measure", "compute_measures",
    "permutation_entropy", "statistical_complexity", "sample_entropy",
    "multiscale_entropy", "katz_fd", "higuchi_fd", "petrosian_fd", "dfa",
    "dfa_envelope", "aperiodic_exponent", "hjorth", "chaos_01",
    "lyapunov_rosenstein", "correlation_dimension", "rqa_determinism", "rqa_profile",
    "pair_correlation", "branching_ratio",
]

FAMILIES = ("lempel-ziv", "ordinal", "entropy", "fractal", "scaling", "spectral",
            "chaos", "recurrence", "criticality")

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


def _bandpass(x, fs, lo, hi, order=4):
    sos = butter(order, [lo, min(hi, 0.49 * fs)], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


# --------------------------------------------------------------------------
# ordinal patterns
# --------------------------------------------------------------------------

def _ordinal_distribution(x, m=3, tau=1):
    """Probability of each of the m! ordinal patterns (Bandt & Pompe 2002)."""
    x = np.asarray(x, dtype=float)
    emb = np.asarray(_embed(jnp.asarray(x), m, tau))
    ranks = np.argsort(emb, axis=1)
    ids = np.ravel_multi_index(ranks.T, (m,) * m)
    counts = np.bincount(ids, minlength=m ** m).astype(float)
    # keep the m! permutation ids (the other m^m - m! bins are zero by construction)
    perm_ids = np.ravel_multi_index(np.array(list(permutations(range(m)))).T, (m,) * m)
    return counts[perm_ids] / counts.sum()


def permutation_entropy(x, m: int = 3, tau: int = 1, **_) -> float:
    """Normalised permutation entropy H / log(m!) (Bandt & Pompe 2002)."""
    p = _ordinal_distribution(x, m, tau)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(factorial(m)))


def statistical_complexity(x, m: int = 3, tau: int = 1, **_) -> float:
    """Jensen-Shannon statistical complexity C = Q_J[P, P_e] · H[P]
    (Rosso et al. 2007 PRL; the y-axis of the complexity-entropy causal plane)."""
    p = _ordinal_distribution(x, m, tau)
    n = len(p)
    pe = np.full(n, 1.0 / n)

    def S(q):
        q = q[q > 0]
        return -(q * np.log(q)).sum()

    h = S(p) / np.log(n)
    js = S((p + pe) / 2) - S(p) / 2 - S(pe) / 2
    q0 = -2.0 / (((n + 1) / n) * np.log(n + 1) - 2 * np.log(2 * n) + np.log(n))
    return float(q0 * js * h)


# --------------------------------------------------------------------------
# sample / multiscale entropy
# --------------------------------------------------------------------------

@partial(jax.jit, static_argnums=1)
def _sampen(x, m, r):
    e1 = _embed(x, m, 1)[:-1]
    e2 = _embed(x, m + 1, 1)
    b = jnp.sum(_chebyshev_count(e1, r) - 1)
    a = jnp.sum(_chebyshev_count(e2, r) - 1)
    return -jnp.log(jnp.clip(a, 1e-10) / jnp.clip(b, 1e-10))


def sample_entropy(x, m: int = 2, r: float = 0.2, **_) -> float:
    """Sample entropy (Richman & Moorman 2000), tolerance ``r`` × SD."""
    x = jnp.asarray(x, dtype=jnp.float32)
    return float(_sampen(x, m, r * jnp.std(x)))


def multiscale_entropy(x, scales: int = 20, m: int = 2, r: float = 0.15, **_) -> np.ndarray:
    """Sample entropy of the coarse-grained series at scales 1..``scales``
    (Costa et al. 2002); ``r`` is fixed from the SD of the *original* series."""
    x = np.asarray(x, dtype=np.float32)
    tol = r * x.std()
    out = np.empty(scales)
    for s in range(1, scales + 1):
        n = len(x) // s
        cg = x[: n * s].reshape(n, s).mean(1)
        out[s - 1] = float(_sampen(jnp.asarray(cg), m, tol))
    return out


def mse_sum(x, scales: int = 20, m: int = 2, r: float = 0.15, **_) -> float:
    """Multiscale entropy summed over scales (the scalar used by Maschke 2024)."""
    return float(np.nansum(multiscale_entropy(x, scales, m, r)))


# --------------------------------------------------------------------------
# fractal dimensions
# --------------------------------------------------------------------------

def katz_fd(x, **_) -> float:
    """Katz (1988) fractal dimension in the amplitude-only convention used by
    antropy / NeuroKit (L = Σ|Δx|, d = max|x − x₀|, n = number of steps), so
    it is scale-free; a monotone ramp gives 1."""
    x = np.asarray(x, dtype=float)
    dx = np.abs(np.diff(x))
    L = dx.sum()
    d = np.max(np.abs(x - x[0]))
    n = len(dx)
    return float(np.log10(n) / (np.log10(n) + np.log10(d / L)))


def higuchi_fd(x, k_max: int = 10, **_) -> float:
    """Higuchi (1988) fractal dimension; white noise ≈ 2, Brownian ≈ 1.5."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    ks, Ls = [], []
    for k in range(1, k_max + 1):
        Lm = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            Lm.append(np.sum(np.abs(np.diff(x[idx]))) * (N - 1) / ((len(idx) - 1) * k) / k)
        ks.append(k)
        Ls.append(np.mean(Lm))
    slope = np.polyfit(np.log(1.0 / np.array(ks)), np.log(Ls), 1)[0]
    return float(slope)


def petrosian_fd(x, **_) -> float:
    """Petrosian (1995) fractal dimension from sign changes of the derivative."""
    x = np.asarray(x, dtype=float)
    d = np.diff(x)
    n_delta = np.sum(d[1:] * d[:-1] < 0)
    n = len(x)
    return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta))))


# --------------------------------------------------------------------------
# scaling
# --------------------------------------------------------------------------

def dfa(x, min_win: int = 4, max_win: int | None = None, n_scales: int = 20, **_) -> float:
    """Detrended fluctuation analysis exponent (Peng et al. 1994), linear
    detrending, log-spaced windows from ``min_win`` to ``max_win`` (T/4)."""
    x = np.asarray(x, dtype=float)
    y = np.cumsum(x - x.mean())
    T = len(y)
    max_win = max_win or T // 4
    wins = np.unique(np.logspace(np.log10(min_win), np.log10(max_win), n_scales).astype(int))
    F = []
    for w in wins:
        n = T // w
        seg = y[: n * w].reshape(n, w)
        t = np.arange(w)
        A = np.vstack([t, np.ones(w)]).T
        coef, *_ = np.linalg.lstsq(A, seg.T, rcond=None)
        resid = seg - (A @ coef).T
        F.append(np.sqrt(np.mean(resid ** 2)))
    return float(np.polyfit(np.log(wins), np.log(F), 1)[0])


def dfa_envelope(x, fs: float, band: str = "alpha", min_s: float = 1.0, max_s: float = 20.0, **_) -> float:
    """DFA exponent of the Hilbert amplitude envelope of a band (Maschke 2024:
    scales 1-20 s on delta..gamma envelopes; long-range temporal correlations)."""
    lo, hi = BANDS[band]
    env = np.abs(hilbert(_bandpass(np.asarray(x, float), fs, lo, hi)))
    return dfa(env, min_win=int(min_s * fs), max_win=min(int(max_s * fs), len(env) // 4))


def aperiodic_exponent(x, fs: float, f_range=(1.0, 40.0), nperseg_s: float = 2.0, **_) -> float:
    """Aperiodic (1/f) exponent from a log-log linear fit of the Welch PSD over
    ``f_range``: no peak model, so oscillatory peaks bias it (use specparam via
    :mod:`neurojax.analysis.specparam_analysis` for the peak-aware fit)."""
    x = np.asarray(x, dtype=float)
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(nperseg_s * fs)))
    sel = (f >= f_range[0]) & (f <= f_range[1])
    slope = np.polyfit(np.log10(f[sel]), np.log10(p[sel]), 1)[0]
    return float(-slope)


# --------------------------------------------------------------------------
# Hjorth
# --------------------------------------------------------------------------

def hjorth(x, **_) -> tuple[float, float]:
    """Hjorth (1970) mobility and complexity."""
    x = np.asarray(x, dtype=float)
    dx, ddx = np.diff(x), np.diff(np.diff(x))
    mob = np.sqrt(dx.var() / x.var())
    comp = np.sqrt(ddx.var() / dx.var()) / mob
    return float(mob), float(comp)


# --------------------------------------------------------------------------
# chaos
# --------------------------------------------------------------------------

def chaos_01(x, n_c: int = 100, seed: int = 0, **_) -> float:
    """Modified 0-1 test for chaos (Gottwald & Melbourne 2009): median over
    ``n_c`` random c ∈ (π/5, 4π/5) of the correlation between n and the
    mean-square displacement of the translation variables. ≈0 regular, ≈1 chaotic."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    rng = np.random.RandomState(seed)
    cs = rng.uniform(np.pi / 5, 4 * np.pi / 5, n_c)
    n_cut = N // 10
    ns = np.arange(1, n_cut + 1)
    j = np.arange(1, N + 1)
    ex2 = np.mean(x) ** 2
    Ks = np.empty(n_c)
    for i, c in enumerate(cs):
        p = np.cumsum(x * np.cos(j * c))
        q = np.cumsum(x * np.sin(j * c))
        M = np.array([np.mean((p[n:] - p[:-n]) ** 2 + (q[n:] - q[:-n]) ** 2) for n in ns])
        D = M - ex2 * (1 - np.cos(ns * c)) / (1 - np.cos(c))
        Ks[i] = np.corrcoef(ns, D)[0, 1]
    return float(np.median(Ks))


def lyapunov_rosenstein(x, fs: float, m: int = 2, tau: int = 1, theiler: int | None = None,
                        k_max: int = 20, **_) -> float:
    """Largest Lyapunov exponent (Rosenstein et al. 1993), per second.
    Nearest neighbours outside a Theiler window (default: fs/10 samples),
    mean log divergence over ``k_max`` steps, slope by least squares."""
    x = np.asarray(x, dtype=float)
    theiler = int(fs / 10) if theiler is None else theiler
    emb = np.asarray(_embed(jnp.asarray(x), m, tau))
    n = len(emb) - k_max
    E = emb[:n]
    D = np.sqrt(((E[:, None, :] - E[None, :, :]) ** 2).sum(-1))
    idx = np.arange(n)
    D[np.abs(idx[:, None] - idx[None, :]) <= theiler] = np.inf
    nn = np.argmin(D, axis=1)
    ks = np.arange(k_max + 1)
    div = np.empty(k_max + 1)
    floor = 1e-12 * (x.std() + 1e-30)  # exact recurrences (periodic signals) stay at a constant floor
    for k in ks:
        d = np.linalg.norm(emb[idx + k] - emb[nn + k], axis=1)
        div[k] = np.mean(np.log(np.maximum(d, floor)))
    return float(np.polyfit(ks, div, 1)[0] * fs)


def correlation_dimension(x, m: int = 2, tau: int = 1, n_r: int = 10, **_) -> float:
    """Grassberger-Procaccia correlation dimension: slope of log C(r) vs log r
    for r between the 5th and 50th percentiles of pairwise distances."""
    emb = np.asarray(_embed(jnp.asarray(np.asarray(x, float)), m, tau))
    D = np.sqrt(((emb[:, None, :] - emb[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(emb), 1)
    d = D[iu]
    d = d[d > 0]  # quantised / flat stretches give exact ties; a zero radius breaks the log-log fit
    if d.size < 10:
        return float("nan")
    lo, hi = np.percentile(d, 5), np.percentile(d, 50)
    if not hi > lo:
        return float("nan")
    rs = np.logspace(np.log10(lo), np.log10(hi), n_r)
    C = np.array([np.mean(d < r) for r in rs])
    return float(np.polyfit(np.log(rs), np.log(C), 1)[0])


def _run_lengths_along_axis0(B: np.ndarray) -> np.ndarray:
    """Lengths of all runs of True down the columns of a boolean matrix."""
    P = np.zeros((B.shape[0] + 2, B.shape[1]), dtype=np.int8)
    P[1:-1] = B
    d = np.diff(P, axis=0)
    starts, ends = np.nonzero(d.T == 1), np.nonzero(d.T == -1)
    return ends[1] - starts[1]


def _diagonal_line_points(R: np.ndarray, l_min: int) -> tuple[int, int]:
    """(recurrent points on diagonal lines of length >= l_min, recurrent points),
    main diagonal excluded. Diagonals of R are laid out as columns of a skewed
    (T, 2T-1) array so run lengths come from one vectorised diff."""
    lengths, total = _diagonal_lines(R)
    return int(lengths[lengths >= l_min].sum()), total


def _diagonal_lines(R: np.ndarray) -> tuple[np.ndarray, int]:
    T = R.shape[0]
    R = R.astype(bool) & ~np.eye(T, dtype=bool)
    rows = np.arange(T)[:, None]
    B = np.zeros((T, 2 * T - 1), dtype=bool)
    B[rows, np.arange(T)[None, :] - rows + T - 1] = R
    return _run_lengths_along_axis0(B), int(R.sum())


def _recurrence_matrix(x, m, tau, recurrence_rate):
    emb = np.asarray(_embed(jnp.asarray(np.asarray(x, float)), m, tau))
    D = np.sqrt(((emb[:, None, :] - emb[None, :, :]) ** 2).sum(-1))
    # numpy percentile (introselect) rather than jnp's full sort: 1 s -> 50 ms at T=2500
    return D <= np.percentile(D, recurrence_rate * 100)


@lru_cache(maxsize=64)
def _rqa_cached(x_bytes: bytes, n: int, m: int, tau: int, recurrence_rate: float, l_min: int, v_min: int) -> dict:
    x = np.frombuffer(x_bytes, dtype=np.float64, count=n)
    R = _recurrence_matrix(x, m, tau, recurrence_rate)
    T = R.shape[0]
    diag, total = _diagonal_lines(R)
    dl = diag[diag >= l_min]
    vert = _run_lengths_along_axis0(R)
    vl = vert[vert >= v_min]
    white = _run_lengths_along_axis0(~R)
    pd_ = np.bincount(dl) / len(dl) if len(dl) else np.array([1.0])
    pd_ = pd_[pd_ > 0]
    # recurrence network: R without self-loops as adjacency (Donges et al. 2011)
    A = (R & ~np.eye(T, dtype=bool)).astype(np.float32)
    k = A.sum(1)
    A2 = A @ A
    triangles = float((A2 * A).sum())  # 6 × number of triangles
    triples = float((k * (k - 1)).sum())
    local_c = np.divide(np.einsum("ij,ij->i", A2, A), k * (k - 1), out=np.zeros(T), where=k > 1)
    return dict(
        recurrence_rate=float(total / (T * (T - 1))),
        determinism=float(dl.sum() / total) if total else 0.0,
        laminarity=float(vl.sum() / vert.sum()) if vert.sum() else 0.0,
        trapping_time=float(vl.mean()) if len(vl) else 0.0,
        mean_diag_length=float(dl.mean()) if len(dl) else 0.0,
        max_diag_length=float(dl.max()) if len(dl) else 0.0,
        diag_entropy=float(-(pd_ * np.log(pd_)).sum()),
        mean_recurrence_time=float(white.mean()) if len(white) else 0.0,
        rn_transitivity=float(triangles / triples) if triples else 0.0,
        rn_global_clustering=float(local_c.mean()),
    )


def rqa_profile(x, m: int = 3, tau: int = 1, recurrence_rate: float = 0.05, l_min: int = 2, v_min: int = 2,
                **_) -> dict:
    """Recurrence quantification (Marwan et al. 2007) and recurrence-network
    measures (Donges, Marwan, Kurths 2011/2012) of one epoch at a fixed
    recurrence rate: determinism, laminarity, trapping time, mean / max
    diagonal line length, diagonal-line entropy, mean recurrence time (mean
    white vertical line), network transitivity and global clustering.
    Vectorised line counting; conventions match pyunicorn (main diagonal
    excluded from diagonal lines; vertical lines include it)."""
    x = np.ascontiguousarray(np.asarray(x, np.float64))
    return _rqa_cached(x.tobytes(), len(x), m, tau, float(recurrence_rate), l_min, v_min)


def rqa_determinism(x, m: int = 3, tau: int = 1, recurrence_rate: float = 0.05,
                    l_min: int = 2, **_) -> float:
    """RQA determinism at a fixed recurrence rate (Marwan et al. 2007): share
    of recurrent points (off the main diagonal) on diagonal lines >= l_min."""
    return rqa_profile(x, m, tau, recurrence_rate, l_min)["determinism"]


def _rqa(key):
    return lambda x, fs, **kw: rqa_profile(x, **kw)[key]


# --------------------------------------------------------------------------
# multivariate criticality measures
# --------------------------------------------------------------------------

def pair_correlation(X, fs: float, band=(8.0, 13.0), **_) -> float:
    """Pair correlation function PCF = N · Var_t |R(t)| of the Kuramoto order
    parameter of band-limited Hilbert phases (Yoon et al. 2015); peaks at the
    synchronisation transition."""
    X = np.asarray(X, float)
    ph = np.angle(hilbert(_bandpass(X, fs, *band), axis=-1))
    R = np.abs(np.mean(np.exp(1j * ph), axis=0))
    return float(X.shape[0] * np.var(R))


def branching_ratio(X, fs: float, threshold: float = 2.0, bin_ms: float = 4.0, **_) -> float:
    """Avalanche branching ratio σ = Σ n(t+1) / Σ n(t) over consecutive
    non-empty time bins (Beggs & Plenz 2003); events are |z| crossings of
    ``threshold`` SD per channel, binned at ``bin_ms``."""
    X = np.asarray(X, float)
    z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    ev = (np.abs(z[:, 1:]) >= threshold) & (np.abs(z[:, :-1]) < threshold)
    w = max(1, int(round(bin_ms * fs / 1000)))
    nb = ev.shape[1] // w
    n = ev[:, : nb * w].reshape(X.shape[0], nb, w).any(-1).sum(0)
    act = (n[:-1] > 0) & (n[1:] > 0)
    if act.sum() == 0:
        return 0.0
    return float(n[1:][act].sum() / n[:-1][act].sum())


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Measure:
    name: str
    fn: Callable
    family: str
    reference: str
    multivariate: bool = False
    params: dict = field(default_factory=dict)

    def __call__(self, x, fs: float) -> float:
        return float(self.fn(x, fs=fs, **self.params))


def _reg(*ms) -> dict[str, Measure]:
    return {m.name: m for m in ms}


MEASURES: dict[str, Measure] = _reg(
    Measure("lz76_mean", lambda x, fs: lz_complexity(x, "mean", "analytic", "lz76"),
            "lempel-ziv", "Lempel & Ziv 1976; Kaspar & Schuster 1987; Maschke 2024 (mean binarisation)"),
    Measure("lz76_envelope", lambda x, fs: lz_complexity(x, "envelope-median", "analytic", "lz76"),
            "lempel-ziv", "Schartner 2015/2017 (Hilbert envelope, median)"),
    Measure("lz78_median", lambda x, fs: lz_complexity(x, "median", "analytic", "lz78"),
            "lempel-ziv", "Lempel & Ziv 1978"),
    Measure("permutation_entropy", permutation_entropy, "ordinal", "Bandt & Pompe 2002", params=dict(m=3, tau=1)),
    Measure("statistical_complexity", statistical_complexity, "ordinal",
            "Rosso 2007; Newman 2025 complexity-entropy causal plane", params=dict(m=3, tau=1)),
    Measure("sample_entropy", sample_entropy, "entropy", "Richman & Moorman 2000", params=dict(m=2, r=0.2)),
    Measure("mse_sum", mse_sum, "entropy", "Costa 2002; Maschke 2024 (sum over scales)",
            params=dict(scales=20, m=2, r=0.15)),
    Measure("spectral_entropy", lambda x, fs: spectral_entropy(jnp.asarray(x)[None], fs=fs)[0],
            "entropy", "Inouye 1991"),
    Measure("svd_entropy", lambda x, fs: svd_entropy(jnp.asarray(x)[None])[0], "entropy", "Roberts 1999"),
    Measure("katz_fd", katz_fd, "fractal", "Katz 1988; Maschke 2024"),
    Measure("higuchi_fd", higuchi_fd, "fractal", "Higuchi 1988", params=dict(k_max=10)),
    Measure("petrosian_fd", petrosian_fd, "fractal", "Petrosian 1995"),
    Measure("dfa", dfa, "scaling", "Peng 1994"),
    Measure("dfa_alpha_envelope", dfa_envelope, "scaling", "Linkenkaer-Hansen 2001; Maschke 2024",
            params=dict(band="alpha", min_s=1.0, max_s=20.0)),
    Measure("aperiodic_exponent", aperiodic_exponent, "spectral", "Donoghue 2020 (log-log fit variant)",
            params=dict(f_range=(1.0, 40.0))),
    Measure("hjorth_mobility", lambda x, fs: hjorth(x)[0], "spectral", "Hjorth 1970"),
    Measure("hjorth_complexity", lambda x, fs: hjorth(x)[1], "spectral", "Hjorth 1970"),
    Measure("chaos_01", chaos_01, "chaos", "Gottwald & Melbourne 2009; Toker 2022; Maschke 2024"),
    Measure("lyapunov", lyapunov_rosenstein, "chaos", "Rosenstein 1993; Maschke 2024 (m=2, tau=1)",
            params=dict(m=2, tau=1, k_max=20)),
    Measure("correlation_dimension", correlation_dimension, "chaos", "Grassberger & Procaccia 1983",
            params=dict(m=5, tau=1)),
    Measure("rqa_determinism", rqa_determinism, "recurrence", "Marwan 2007",
            params=dict(m=3, tau=1, recurrence_rate=0.05)),
    Measure("rqa_laminarity", _rqa("laminarity"), "recurrence", "Marwan 2007 (pyunicorn convention)",
            params=dict(m=3, tau=1, recurrence_rate=0.05, v_min=2)),
    Measure("rqa_trapping_time", _rqa("trapping_time"), "recurrence", "Marwan 2007",
            params=dict(m=3, tau=1, recurrence_rate=0.05, v_min=2)),
    Measure("rqa_mean_diag_length", _rqa("mean_diag_length"), "recurrence", "Marwan 2007",
            params=dict(m=3, tau=1, recurrence_rate=0.05, l_min=2)),
    Measure("rqa_diag_entropy", _rqa("diag_entropy"), "recurrence", "Marwan 2007",
            params=dict(m=3, tau=1, recurrence_rate=0.05, l_min=2)),
    Measure("rqa_mean_recurrence_time", _rqa("mean_recurrence_time"), "recurrence", "Ngamga 2007; pyunicorn",
            params=dict(m=3, tau=1, recurrence_rate=0.05)),
    Measure("rn_transitivity", _rqa("rn_transitivity"), "recurrence", "Donges, Marwan, Kurths 2011 recurrence network",
            params=dict(m=3, tau=1, recurrence_rate=0.05)),
    Measure("rn_global_clustering", _rqa("rn_global_clustering"), "recurrence", "Donges 2011; Marwan 2009",
            params=dict(m=3, tau=1, recurrence_rate=0.05)),
    Measure("pair_correlation", pair_correlation, "criticality", "Yoon 2015; Maschke 2024 (alpha)",
            multivariate=True, params=dict(band=(8.0, 13.0))),
    Measure("branching_ratio", branching_ratio, "criticality", "Beggs & Plenz 2003; Maschke 2024",
            multivariate=True, params=dict(threshold=2.0, bin_ms=4.0)),
)


def compute_measures(X, fs: float, measures: dict[str, Measure] | None = None) -> dict[str, float]:
    """All registered measures on one (C, T) epoch: univariate measures are
    averaged over channels, multivariate ones see the full array."""
    X = np.asarray(X, float)
    measures = measures or MEASURES
    out = {}
    for name, m in measures.items():
        if m.multivariate:
            out[name] = m(X, fs)
        else:
            out[name] = float(np.mean([m(X[c], fs) for c in range(X.shape[0])]))
    return out
