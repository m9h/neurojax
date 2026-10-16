# © NeuroJAX developers
#
# License: BSD (3-clause)
"""AFRICA-style automatic artefact-IC classification and rejection.

OSL's ``osl_africa`` runs ICA, identifies artefactual independent components
(ocular, cardiac, line-noise), and back-projects the remainder. neurojax already
provides the decomposition (:class:`neurojax.preprocessing.ica.FastICA`, whose
``components_`` are the IC time courses and ``mixing_`` the topographies); this
module is the missing classify-and-reject layer.

Components are flagged by:

* **EOG/ECG correlation** — ``|Pearson(component, reference)|`` against an eye or
  cardiac reference channel (:func:`correlation_scores`);
* **line-noise power** — fraction of a component's spectrum within a narrow band
  around the mains frequency (:func:`line_noise_scores`).

Flagged components are zeroed and the data reconstructed
(:func:`reject_components`). The heuristics are montage-free (what AFRICA uses
for MEG); for EEG, ``mne_icalabel``'s ICLabel is an alternative classifier that
can supply the indices passed to :func:`reject_components`.
"""

import jax.numpy as jnp

__all__ = [
    "correlation_scores",
    "line_noise_scores",
    "classify_artifact_components",
    "reject_components",
]


def correlation_scores(sources: jnp.ndarray, reference: jnp.ndarray) -> jnp.ndarray:
    """Absolute Pearson correlation of each component with a reference signal.

    Parameters
    ----------
    sources : (n_components, n_samples) — IC time courses.
    reference : (n_samples,) — EOG or ECG reference channel.

    Returns
    -------
    scores : (n_components,) in [0, 1].
    """
    r = reference - jnp.mean(reference)
    S = sources - jnp.mean(sources, axis=1, keepdims=True)
    num = S @ r
    den = jnp.sqrt(jnp.sum(S ** 2, axis=1) * jnp.sum(r ** 2))
    return jnp.abs(num / jnp.maximum(den, 1e-20))


def line_noise_scores(sources: jnp.ndarray, fs: float,
                      line_freq: float = 50.0, bw: float = 1.0) -> jnp.ndarray:
    """Fraction of each component's power within ``±bw`` of the mains frequency.

    Returns
    -------
    scores : (n_components,) in [0, 1]; high => line-noise dominated.
    """
    n = sources.shape[1]
    spec = jnp.abs(jnp.fft.rfft(sources, axis=1)) ** 2
    freqs = jnp.fft.rfftfreq(n, d=1.0 / fs)
    band = (freqs >= line_freq - bw) & (freqs <= line_freq + bw)
    total = jnp.sum(spec, axis=1)
    in_band = jnp.sum(spec * band[None, :], axis=1)
    return in_band / jnp.maximum(total, 1e-20)


def classify_artifact_components(sources, fs, *, eog=None, ecg=None,
                                 corr_thresh: float = 0.4, line_freq=None,
                                 line_thresh: float = 0.3) -> dict:
    """Flag ocular / cardiac / line-noise components by the heuristics above.

    Returns a dict with per-type scores (when the relevant reference / line
    frequency is supplied) and ``"artifacts"``: the sorted union of flagged
    component indices, ready for :func:`reject_components`.
    """
    out: dict = {}
    flagged: set[int] = set()
    if eog is not None:
        oc = correlation_scores(sources, eog)
        out["ocular"] = oc
        flagged |= {int(i) for i in jnp.where(oc > corr_thresh)[0]}
    if ecg is not None:
        ca = correlation_scores(sources, ecg)
        out["cardiac"] = ca
        flagged |= {int(i) for i in jnp.where(ca > corr_thresh)[0]}
    if line_freq is not None:
        ln = line_noise_scores(sources, fs, line_freq=line_freq)
        out["line"] = ln
        flagged |= {int(i) for i in jnp.where(ln > line_thresh)[0]}
    out["artifacts"] = sorted(flagged)
    return out


def reject_components(sources: jnp.ndarray, mixing: jnp.ndarray,
                      reject_idx, mean: jnp.ndarray | None = None) -> jnp.ndarray:
    """Reconstruct the sensor data with the flagged components zeroed.

    ``X_clean = mixing @ (sources with reject rows set to 0) [+ mean]``.

    Parameters
    ----------
    sources : (n_components, n_samples).
    mixing : (n_features, n_components) — IC topographies.
    reject_idx : sequence of component indices to remove.
    mean : (n_features, 1) or None — added back if the ICA centred the data.

    Returns
    -------
    cleaned : (n_features, n_samples).
    """
    idx = jnp.asarray(list(reject_idx), dtype=jnp.int32)
    kept = sources.at[idx].set(0.0)
    cleaned = mixing @ kept
    if mean is not None:
        cleaned = cleaned + mean
    return cleaned
