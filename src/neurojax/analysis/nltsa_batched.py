# © NeuroJAX developers
#
# License: BSD (3-clause)
"""Batched (vmap / accelerator) versions of the expensive NLTSA measures.

The scalar functions in :mod:`neurojax.analysis.nltsa` are numpy, one channel
at a time, and are the reference (checked against pyunicorn / nolds). Four of
them dominate the benchmark cost because they build an N×N distance matrix
per channel-epoch (recurrence profile, Rosenstein Lyapunov, correlation
dimension, sample / multiscale entropy) or loop over 100 test frequencies
(0-1 chaos). This module
re-expresses them as jitted functions over a (B, T) stack so a GPU scores all
channels × epochs of a recording in one call. Results agree with the scalar
versions to float32 (test_nltsa_batched.py); recurrence thresholds are
percentiles of the float32 distance matrix, so a few ties can differ.

Memory: every item needs several N×N float32 buffers (distances, recurrence
matrix, adjacency products, skewed diagonals). :func:`chunk_size` estimates
this and chunks the batch against the device's memory; the estimate is
printed once so nothing is hidden.
"""

from __future__ import annotations

import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from neurojax.analysis.entropy import _embed

__all__ = ["BATCHED", "batched_rqa_profile", "batched_lyapunov", "batched_correlation_dimension",
           "batched_chaos_01", "batched_sample_entropy", "batched_mse_sum", "score_batched", "chunk_size",
           "accelerator_available"]

RQA_KEYS = ("recurrence_rate", "determinism", "laminarity", "trapping_time", "mean_diag_length",
            "max_diag_length", "diag_entropy", "mean_recurrence_time", "rn_transitivity", "rn_global_clustering")


def accelerator_available() -> bool:
    return jax.devices()[0].platform != "cpu"


def _device_bytes() -> float:
    """Usable bytes on the default device (GPU memory limit, else host RAM)."""
    try:
        st = jax.devices()[0].memory_stats() or {}
        if "bytes_limit" in st:
            return float(st["bytes_limit"])
    except Exception:  # noqa: BLE001  — CPU backends have no memory_stats
        pass
    import psutil
    return float(psutil.virtual_memory().available)


def chunk_size(T: int, m: int, per_item_buffers: int = 16, max_chunk: int = 16, fraction: float = 0.5) -> int:
    """Largest batch chunk such that ``per_item_buffers`` N×N float32 buffers per
    item use at most ``fraction`` of device memory. N = T − (m−1)·tau."""
    per_item = per_item_buffers * 4.0 * T * T
    n = int(fraction * _device_bytes() // max(per_item, 1))
    return int(max(1, min(max_chunk, n)))


def _map(f, X, chunk):
    """vmap ``f`` over the leading axis of ``X`` in chunks of ``chunk`` (lax.map batch_size)."""
    X = jnp.asarray(X, jnp.float32)
    if chunk is None:
        chunk = chunk_size(X.shape[1], 1)
    chunk = int(min(chunk, X.shape[0]))
    with jax.default_matmul_precision("highest"):
        return jax.lax.map(f, X, batch_size=chunk)


def _dist(emb):
    """Euclidean distances between embedded points, (N, N) float32, accumulated per
    embedding coordinate so no (N, N, m) intermediate is materialised."""
    N, m = emb.shape
    D2 = jnp.zeros((N, N), jnp.float32)
    for j in range(m):
        d = emb[:, j][:, None] - emb[:, j][None, :]
        D2 = D2 + d * d
    return jnp.sqrt(D2)


def _run_length_hist(B):
    """Histogram h[L] (L = 0..T) of lengths of True runs down the columns of a
    boolean (T, K) matrix. Runs end where a True is followed by False (or the
    bottom); their length is the row distance to the last run start. The
    histogram is taken by sorting the lengths and counting with searchsorted:
    a scatter-add would send every non-end cell to bin 0, and on a GPU that
    many colliding atomics is slower than the whole numpy pipeline."""
    T = B.shape[0]
    P = jnp.concatenate([jnp.zeros((1, B.shape[1]), bool), B, jnp.zeros((1, B.shape[1]), bool)], 0)
    d = jnp.diff(P.astype(jnp.int8), axis=0)  # (T+1, K): +1 at a run start row, −1 one past its end
    rows = jnp.arange(T + 1, dtype=jnp.int32)[:, None]
    last_start = jax.lax.cummax(jnp.where(d == 1, rows, -1), axis=0)
    length = jnp.where(d == -1, rows - last_start, 0).ravel()
    srt = jnp.sort(length)
    edges = jnp.arange(T + 2, dtype=jnp.int32)  # bin L covers [L, L+1)
    h = jnp.diff(jnp.searchsorted(srt, edges, method="scan")).astype(jnp.float32)
    return h.at[0].set(0.0)


def _line_stats(h, l_min):
    L = jnp.arange(h.shape[0], dtype=jnp.float32)
    keep = (L >= l_min)
    hk = h * keep
    n = hk.sum()
    total = (hk * L).sum()
    mean = jnp.where(n > 0, total / jnp.maximum(n, 1), 0.0)
    mx = jnp.where(n > 0, (L * (hk > 0)).max(), 0.0)
    p = jnp.where(n > 0, hk / jnp.maximum(n, 1), 0.0)
    ent = -(jnp.where(p > 0, p * jnp.log(jnp.where(p > 0, p, 1.0)), 0.0)).sum()
    ent = jnp.where(n > 0, ent, 0.0)  # numpy version: one line of p=1 when there are no lines
    return dict(n=n, total=total, mean=mean, max=mx, entropy=ent)


def _rqa_one(x, m, tau, recurrence_rate, l_min, v_min):
    emb = _embed(x, m, tau)
    N = emb.shape[0]
    D = _dist(emb)
    thr = jnp.percentile(D, recurrence_rate * 100)
    R = D <= thr
    eye = jnp.eye(N, dtype=bool)
    Roff = R & ~eye
    total = Roff.sum().astype(jnp.float32)
    # diagonals as columns of a skewed (N, 2N−1) array: column k holds diagonal j − i = k − (N−1)
    i = jnp.arange(N)[:, None]
    k = jnp.arange(2 * N - 1)[None, :]
    j = i + k - (N - 1)
    valid = (j >= 0) & (j < N)
    S = jnp.take_along_axis(Roff, jnp.clip(j, 0, N - 1), axis=1) & valid
    dl = _line_stats(_run_length_hist(S), l_min)
    vert_h = _run_length_hist(R)
    vl = _line_stats(vert_h, v_min)
    vert_sum = (vert_h * jnp.arange(N + 1)).sum()
    white = _line_stats(_run_length_hist(~R), 1)
    A = Roff.astype(jnp.float32)
    deg = A.sum(1)
    A2 = A @ A
    tri_i = (A2 * A).sum(1)
    triangles = tri_i.sum()
    triples = (deg * (deg - 1)).sum()
    local_c = jnp.where(deg > 1, tri_i / jnp.maximum(deg * (deg - 1), 1), 0.0)
    return dict(
        recurrence_rate=total / (N * (N - 1)),
        determinism=jnp.where(total > 0, dl["total"] / jnp.maximum(total, 1), 0.0),
        laminarity=jnp.where(vert_sum > 0, vl["total"] / jnp.maximum(vert_sum, 1), 0.0),
        trapping_time=vl["mean"],
        mean_diag_length=dl["mean"],
        max_diag_length=dl["max"],
        diag_entropy=dl["entropy"],
        mean_recurrence_time=white["mean"],
        rn_transitivity=jnp.where(triples > 0, triangles / jnp.maximum(triples, 1), 0.0),
        rn_global_clustering=local_c.mean(),
    )


def batched_rqa_profile(X, m: int = 3, tau: int = 1, recurrence_rate: float = 0.05, l_min: int = 2,
                        v_min: int = 2, chunk: int | None = None, **_) -> dict:
    """:func:`nltsa.rqa_profile` for every row of ``X`` (B, T); dict of (B,) arrays."""
    f = partial(_rqa_one, m=m, tau=tau, recurrence_rate=recurrence_rate, l_min=l_min, v_min=v_min)
    return _map(f, X, chunk)


def _lyap_one(x, fs, m, tau, theiler, k_max):
    emb = _embed(x, m, tau)
    n = emb.shape[0] - k_max
    E = emb[:n]
    D = _dist(E)
    idx = jnp.arange(n)
    D = jnp.where(jnp.abs(idx[:, None] - idx[None, :]) <= theiler, jnp.inf, D)
    nn = jnp.argmin(D, axis=1)
    floor = 1e-12 * (jnp.std(x) + 1e-30)
    ks = jnp.arange(k_max + 1, dtype=jnp.float32)
    div = jnp.stack([jnp.mean(jnp.log(jnp.maximum(jnp.linalg.norm(emb[idx + k] - emb[nn + k], axis=1), floor)))
                     for k in range(k_max + 1)])
    slope = jnp.sum((ks - ks.mean()) * (div - div.mean())) / jnp.sum((ks - ks.mean()) ** 2)
    return slope * fs


def batched_lyapunov(X, fs: float, m: int = 2, tau: int = 1, theiler: int | None = None, k_max: int = 20,
                     chunk: int | None = None, **_):
    """:func:`nltsa.lyapunov_rosenstein` for every row of ``X`` (B, T) → (B,)."""
    theiler = int(fs / 10) if theiler is None else theiler
    return _map(partial(_lyap_one, fs=fs, m=m, tau=tau, theiler=theiler, k_max=k_max), X, chunk)


def _cd_one(x, m, tau, n_r):
    emb = _embed(x, m, tau)
    N = emb.shape[0]
    D = _dist(emb)
    use = jnp.triu(jnp.ones((N, N), bool), 1) & (D > 0)
    n_use = use.sum()
    Dn = jnp.where(use, D, jnp.nan)
    lo, hi = jnp.nanpercentile(Dn, 5), jnp.nanpercentile(Dn, 50)
    rs = jnp.logspace(jnp.log10(lo), jnp.log10(hi), n_r)
    C = jnp.stack([jnp.sum(use & (D < r)) / jnp.maximum(n_use, 1) for r in rs])
    lx, ly = jnp.log(rs), jnp.log(C)
    slope = jnp.sum((lx - lx.mean()) * (ly - ly.mean())) / jnp.sum((lx - lx.mean()) ** 2)
    return jnp.where((n_use >= 10) & (hi > lo), slope, jnp.nan)


def batched_correlation_dimension(X, m: int = 2, tau: int = 1, n_r: int = 10, chunk: int | None = None, **_):
    """:func:`nltsa.correlation_dimension` for every row of ``X`` (B, T) → (B,)."""
    return _map(partial(_cd_one, m=m, tau=tau, n_r=n_r), X, chunk)


def _chaos_one(x, cs, n_cut):
    N = x.shape[0]
    j = jnp.arange(1, N + 1, dtype=jnp.float32)
    p = jnp.cumsum(x[None, :] * jnp.cos(j[None, :] * cs[:, None]), axis=1)  # (n_c, N)
    q = jnp.cumsum(x[None, :] * jnp.sin(j[None, :] * cs[:, None]), axis=1)
    ns = jnp.arange(1, n_cut + 1, dtype=jnp.float32)
    ex2 = jnp.mean(x) ** 2

    def msd(n):
        valid = (jnp.arange(N) < N - n)[None, :]
        dp = jnp.roll(p, -n, axis=1) - p
        dq = jnp.roll(q, -n, axis=1) - q
        return jnp.sum(jnp.where(valid, dp * dp + dq * dq, 0.0), axis=1) / (N - n)

    M = jax.vmap(msd)(jnp.arange(1, n_cut + 1)).T  # (n_c, n_cut)
    Dc = M - ex2 * (1 - jnp.cos(ns[None, :] * cs[:, None])) / (1 - jnp.cos(cs))[:, None]
    nc = ns - ns.mean()
    Dm = Dc - Dc.mean(1, keepdims=True)
    K = (Dm * nc[None, :]).sum(1) / jnp.sqrt((Dm * Dm).sum(1) * (nc * nc).sum())
    return jnp.median(K)


def batched_chaos_01(X, n_c: int = 100, seed: int = 0, chunk: int | None = None, **_):
    """:func:`nltsa.chaos_01` for every row of ``X`` (B, T) → (B,). Same c draws
    (RandomState(seed)); the mean-square displacement at lag n is taken over a
    same as the scalar version (masked roll instead of a variable-length slice)."""
    X = np.asarray(X, np.float32)
    cs = jnp.asarray(np.random.RandomState(seed).uniform(np.pi / 5, 4 * np.pi / 5, n_c), jnp.float32)
    return _map(partial(_chaos_one, cs=cs, n_cut=X.shape[1] // 10), X, chunk)


def _cheb_count_minus_self(emb, r):
    """Number of other embedded vectors within Chebyshev distance r of each one,
    summed; the (N, N) max-distance matrix is accumulated one coordinate at a
    time so no (N, N, m) intermediate exists."""
    N, m = emb.shape
    Dm = jnp.zeros((N, N), jnp.float32)
    for j in range(m):
        Dm = jnp.maximum(Dm, jnp.abs(emb[:, j][:, None] - emb[:, j][None, :]))
    return jnp.sum(Dm <= r) - N


def _sampen_one(x, m, r):
    e1 = _embed(x, m, 1)[:-1]
    e2 = _embed(x, m + 1, 1)
    b = _cheb_count_minus_self(e1, r)
    a = _cheb_count_minus_self(e2, r)
    return -jnp.log(jnp.clip(a, 1e-10) / jnp.clip(b, 1e-10))


def _mse_one(x, scales, m, r):
    tol = r * jnp.std(x)
    vals = []
    for s in range(1, scales + 1):
        n = x.shape[0] // s
        cg = x[: n * s].reshape(n, s).mean(1)
        vals.append(_sampen_one(cg, m, tol))
    return jnp.nansum(jnp.stack(vals))


def batched_sample_entropy(X, m: int = 2, r: float = 0.2, chunk: int | None = None, **_):
    """:func:`nltsa.sample_entropy` for every row of ``X`` (B, T) → (B,)."""
    return _map(lambda x: _sampen_one(x, m, r * jnp.std(x)), X, chunk)


def batched_mse_sum(X, scales: int = 20, m: int = 2, r: float = 0.15, chunk: int | None = None, **_):
    """:func:`nltsa.mse_sum` for every row of ``X`` (B, T) → (B,)."""
    return _map(partial(_mse_one, scales=scales, m=m, r=r), X, chunk)


# registry: measure name -> (batched fn, output key or None, params that select the shared computation)
BATCHED: dict[str, tuple] = {
    "lyapunov": (batched_lyapunov, None),
    "correlation_dimension": (batched_correlation_dimension, None),
    "chaos_01": (batched_chaos_01, None),
    "sample_entropy": (batched_sample_entropy, None),
    "mse_sum": (batched_mse_sum, None),
    **{f"rqa_{k}": (batched_rqa_profile, k) for k in
       ("determinism", "laminarity", "trapping_time", "mean_diag_length", "diag_entropy", "mean_recurrence_time")},
    "rn_transitivity": (batched_rqa_profile, "rn_transitivity"),
    "rn_global_clustering": (batched_rqa_profile, "rn_global_clustering"),
}


def score_batched(X, fs: float, measures: dict, chunk: int | None = None) -> dict[str, np.ndarray]:
    """Score every measure of ``measures`` that has a batched implementation on
    the (B, T) stack ``X``; measures sharing a batched function and parameters
    (the RQA family) are computed once. Returns name -> (B,) float64 array."""
    X = np.asarray(X, np.float32)
    out, cache = {}, {}
    for name, meas in measures.items():
        if name not in BATCHED:
            continue
        fn, key = BATCHED[name]
        params = dict(meas.params)
        if key is not None:  # RQA family: one profile per parameter set
            params = {"l_min": 2, "v_min": 2, **params}
            ck = (fn.__name__, tuple(sorted(params.items())))
            if ck not in cache:
                cache[ck] = fn(X, chunk=chunk, **params)
            out[name] = np.asarray(cache[ck][key], np.float64)
        else:
            out[name] = np.asarray(fn(X, fs=fs, chunk=chunk, **params), np.float64)
    return out
