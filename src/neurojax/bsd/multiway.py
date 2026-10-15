"""Multiway (tensor) reduction of EEG spectra and epochs for the spectral models.

Two uses, both HOSVD-based and pure JAX (via jaxctrl):

* :func:`spectral_tensor` / :func:`hosvd_spectrum`: build the
  ``(channels, frequencies, segments)`` tensor of log Welch spectra for a
  run and reduce it by HOSVD.  The leading channel-mode factor is a
  data-driven spatial weighting (replacing a fixed posterior-channel
  average), the leading segment-mode factor says how stationary the run
  is, and the core's energy along each mode gives a multilinear rank.
  The spectrum handed to the models is the channel-factor-weighted mean
  log spectrum, so the models see one spectrum per run as before but the
  weights are learned from that run.
* :func:`denoise_epochs`: Marchenko-Pastur-thresholded HOSVD on a
  ``(trials, channels, time)`` tensor, the multiway NORDIC idea applied
  to epoched EEG before spectra are taken (per-mode truncation with the
  noise level estimated from the temporal first difference).
"""

from __future__ import annotations

import numpy as np
from jaxctrl import hosvd, tensor_fold as fold, tensor_unfold as unfold


def spectral_tensor(data, fs, segment_s=2.0, f_lo=2.0, f_hi=45.0):
    """``(channels, freqs, segments)`` log power from non-overlapping Hann segments."""
    from scipy.signal import periodogram
    data = np.asarray(data, dtype=float)
    nper = int(round(segment_s * fs))
    n_seg = data.shape[-1] // nper
    segs = data[:, : n_seg * nper].reshape(data.shape[0], n_seg, nper)
    f, p = periodogram(segs, fs=fs, window="hann", axis=-1)
    m = (f >= f_lo) & (f <= f_hi)
    return f[m], np.log(np.maximum(np.transpose(p[..., m], (0, 2, 1)), 1e-300)), n_seg


def hosvd_spectrum(tensor, ranks=(1, None, 1), centre=False):
    """Leading-factor reduction of a ``(channels, freqs, segments)`` log-spectrum tensor.

    Returns a dict with the channel weights (sign fixed so their mean is
    positive, normalised to sum to one), the segment weights, the weighted
    mean log spectrum, the fraction of energy captured by the leading
    channel and segment factors, and the per-mode singular values.  The
    weights are taken from the alpha-band-emphasised tensor: log spectra are
    shifted to be positive so the leading factor is the dominant spatial
    pattern rather than the sign-ambiguous first principal deviation.
    """
    T = np.asarray(tensor, dtype=float)
    # With centre=False the leading channel factor is the dominant spatial pattern of
    # the spectra themselves (posterior for an alpha-dominated run); with centre=True
    # the common spectral shape is removed first and the leading factors describe
    # the dominant *deviation*, which on resting EEG is a near-global power drift.
    R = T - T.mean(axis=(0, 2), keepdims=True) if centre else T - T.min() + 1e-6
    sv = []
    for mode in range(3):
        s = np.linalg.svd(np.asarray(unfold(R, mode)), compute_uv=False)
        sv.append(s)
    core, factors = hosvd(R, ranks=[r if r is not None else T.shape[i] for i, r in enumerate(ranks)])
    w_ch = np.asarray(factors[0][:, 0]); w_ch = w_ch * np.sign(w_ch.mean() or 1.0)
    w_seg = np.asarray(factors[2][:, 0]); w_seg = w_seg * np.sign(w_seg.mean() or 1.0)
    w_ch_pos = np.maximum(w_ch, 0); w_ch_pos = w_ch_pos / max(w_ch_pos.sum(), 1e-12)
    spectrum = np.einsum("c,cfs->f", w_ch_pos, T) / T.shape[2]
    return {"channel_weights": w_ch_pos, "segment_weights": w_seg, "log_psd": spectrum,
            "energy_channel_1": float(sv[0][0] ** 2 / (sv[0] ** 2).sum()),
            "energy_segment_1": float(sv[2][0] ** 2 / (sv[2] ** 2).sum()),
            "singular_values": [s.tolist() for s in sv]}


def _mp_truncate(M, sigma):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    thr = sigma * (np.sqrt(M.shape[0]) + np.sqrt(M.shape[1]))
    k = int((s > thr).sum())
    k = max(k, 1)
    return (U[:, :k] * s[:k]) @ Vt[:k], k


def denoise_epochs(epochs, sigma=None):
    """HOSVD with per-mode Marchenko-Pastur truncation on ``(trials, channels, time)``.

    Sequential truncated HOSVD (the multiway NORDIC baseline); *sigma* is
    the noise standard deviation, estimated from the median absolute
    deviation of the temporal first difference when not given.  Returns
    the denoised tensor and the rank kept per mode.
    """
    X = np.asarray(epochs, dtype=float)
    if sigma is None:
        d = np.diff(X, axis=-1)
        sigma = float(np.median(np.abs(d - np.median(d)))) / 0.6745 / np.sqrt(2.0)
    out = X; kept = []
    for mode in range(3):
        M = np.asarray(unfold(out, mode))
        Mt, k = _mp_truncate(M, sigma)
        out = np.asarray(fold(Mt, mode, out.shape)); kept.append(k)
    return out, kept
