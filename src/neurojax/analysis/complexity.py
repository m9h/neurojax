# © NeuroJAX developers
#
# License: BSD (3-clause)
"""Lempel-Ziv complexity and its spectral / phasic decomposition.

Lempel-Ziv complexity of a binarised signal is a detector, not an explanation:
its value is largely set by the threshold-crossing rate, hence by the power
spectrum (less alpha, flatter aperiodic slope or more broadband power all raise
it mechanically). Mediano, Rosas, Barrett & Bor (arXiv:2009.10015; *Phys. Rev.
Lett.*) split a between-condition difference of any non-linear feature into a
*spectral* part, reproduced by phase-randomised surrogates that keep the
amplitude spectrum, and a residual *phasic* part that depends on the phase
structure the surrogates destroy. :func:`spectral_phase_decomposition` is that
split for an arbitrary feature; :func:`lz_complexity` names every estimator
choice (LZ76 vs LZ78, envelope vs raw binarisation, normaliser) explicitly
because the same word covers different estimators across papers.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from neurojax.analysis.analytic import envelope
from neurojax.analysis.surrogates import phase_randomized_surrogate, shuffle_surrogate

__all__ = ["lz76", "lz78", "lz_complexity", "spectral_phase_decomposition"]


def lz76(s) -> int:
    """Lempel-Ziv 1976 complexity (number of words) of a binary sequence.

    Kaspar & Schuster (1987) algorithm; ``0001101001000101`` parses into 6 words.
    """
    s = np.asarray(s).astype(np.uint8).tobytes()
    n = len(s)
    if n == 0:
        return 0
    i, k, l = 0, 1, 1
    c, k_max = 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            k_max = max(k, k_max)
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c


def lz78(s) -> int:
    """Lempel-Ziv 1978 complexity: size of the incremental-parsing dictionary."""
    s = np.asarray(s).astype(np.uint8).tobytes()
    words, w = set(), b""
    for ch in s:
        w += bytes([ch])
        if w not in words:
            words.add(w)
            w = b""
    return len(words) + (1 if w else 0)


_BINARISE = ("envelope-median", "median", "mean")
_NORMALISE = (None, "analytic", "shuffle", "phase")
_METHOD = {"lz76": lz76, "lz78": lz78}


def _binarise(x: jnp.ndarray, how: str) -> np.ndarray:
    """(T,) -> (T,) uint8. ``how`` picks the signal and the threshold."""
    if how == "envelope-median":
        a = envelope(x)
        return np.asarray(a > jnp.median(a)).astype(np.uint8)
    if how == "median":
        return np.asarray(x > jnp.median(x)).astype(np.uint8)
    if how == "mean":
        return np.asarray(x > jnp.mean(x)).astype(np.uint8)
    raise ValueError(f"binarise must be one of {_BINARISE}, got {how!r}")


def _lz_1d(x, binarise, method):
    return float(_METHOD[method](_binarise(x, binarise)))


def lz_complexity(
    x: jnp.ndarray,
    binarise: str = "envelope-median",
    normalise: Optional[str] = "analytic",
    method: str = "lz76",
    key: Optional[jax.Array] = None,
    n_surrogates: int = 20,
) -> jnp.ndarray:
    """Lempel-Ziv complexity of a signal with every estimator choice named.

    Parameters
    ----------
    x : (T,) or (T, C) time series; channels are scored independently.
    binarise : ``"envelope-median"`` (Hilbert envelope thresholded at its
        median — the usual EEG choice), ``"median"`` or ``"mean"`` of the raw signal.
    normalise : ``None`` returns the raw word count; ``"analytic"`` divides by
        the random-sequence asymptote ``T / log2(T)``; ``"shuffle"`` divides by
        the mean over shuffled surrogates (removes amplitude-distribution effects
        only); ``"phase"`` divides by the mean over phase-randomised surrogates
        (removes spectral effects as well — the ratio of two biased estimators,
        so its variance is larger).
    method : ``"lz76"`` (Kaspar-Schuster) or ``"lz78"`` (dictionary size).
    key, n_surrogates : for the surrogate normalisers.

    Returns
    -------
    scalar for (T,), or (C,) for (T, C).
    """
    if binarise not in _BINARISE:
        raise ValueError(f"binarise must be one of {_BINARISE}, got {binarise!r}")
    if normalise not in _NORMALISE:
        raise ValueError(f"normalise must be one of {_NORMALISE}, got {normalise!r}")
    if method not in _METHOD:
        raise ValueError(f"method must be one of {tuple(_METHOD)}, got {method!r}")
    x = jnp.asarray(x)
    squeeze = x.ndim == 1
    X = x[:, None] if squeeze else x
    T, C = X.shape
    raw = np.array([_lz_1d(X[:, c], binarise, method) for c in range(C)])

    if normalise is None:
        out = raw
    elif normalise == "analytic":
        out = raw / (T / np.log2(T))
    else:
        if key is None:
            raise ValueError(f"normalise={normalise!r} needs a PRNG key")
        surr_fn = shuffle_surrogate if normalise == "shuffle" else phase_randomized_surrogate
        keys = jax.random.split(key, n_surrogates)
        ref = np.mean([[_lz_1d(surr_fn(X, k)[:, c], binarise, method) for c in range(C)]
                       for k in keys], axis=0)
        out = raw / ref
    out = jnp.asarray(out)
    return out[0] if squeeze else out


def spectral_phase_decomposition(
    feature: Callable[[jnp.ndarray], jnp.ndarray],
    x_a: jnp.ndarray,
    x_b: jnp.ndarray,
    key: jax.Array,
    n_surrogates: int = 100,
) -> dict:
    """Split ``feature(x_b) - feature(x_a)`` into spectral and phasic parts.

    Mediano, Rosas, Barrett & Bor (arXiv:2009.10015): with phase-randomised
    surrogates ``s_a``, ``s_b`` that keep each condition's amplitude spectrum,

        spectral = E[feature(s_b)] - E[feature(s_a)]
        phasic   = total - spectral

    The spectral part is what a linear Gaussian process with the same spectra
    would show; the phasic part is what depends on phase structure.
    ``spectral_std`` is the standard error of the surrogate-mean difference;
    ``phasic_std`` is the null spread of the phasic part (each observed feature
    is one draw from its surrogate distribution under the null, so this is the
    per-draw spread of both conditions, not the error of the means). The
    phasic part against ``phasic_std`` is what sizes the argument.

    Parameters
    ----------
    feature : callable (T,) or (T, C) -> scalar or (C,) feature.
    x_a, x_b : the two conditions, same shape convention as ``feature`` expects.
    key, n_surrogates : phase-randomised surrogates per condition.

    Returns
    -------
    dict with ``total``, ``spectral``, ``phasic``, ``spectral_std``,
    ``phasic_std``, ``feature_a``, ``feature_b``, and the per-surrogate feature
    values ``surrogate_a``, ``surrogate_b`` (n_surrogates, ...).
    """
    ka, kb = jax.random.split(key)
    fa, fb = np.asarray(feature(x_a)), np.asarray(feature(x_b))
    sa = np.stack([np.asarray(feature(phase_randomized_surrogate(x_a, k)))
                   for k in jax.random.split(ka, n_surrogates)])
    sb = np.stack([np.asarray(feature(phase_randomized_surrogate(x_b, k)))
                   for k in jax.random.split(kb, n_surrogates)])
    total = fb - fa
    spectral = sb.mean(0) - sa.mean(0)
    va, vb = sa.var(0, ddof=1), sb.var(0, ddof=1)
    # spectral part: standard error of the difference of surrogate means.
    # phasic part: under the null (no phase structure) feature(x) is itself one
    # draw from the surrogate distribution, so its spread is the per-draw
    # variance of both conditions plus the error of the surrogate means.
    spectral_std = np.sqrt(va / n_surrogates + vb / n_surrogates)
    phasic_std = np.sqrt((va + vb) * (1.0 + 1.0 / n_surrogates))
    return {
        "total": total,
        "spectral": spectral,
        "phasic": total - spectral,
        "spectral_std": spectral_std,
        "phasic_std": phasic_std,
        "feature_a": fa,
        "feature_b": fb,
        "surrogate_a": sa,
        "surrogate_b": sb,
    }
