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


# ---------------------------------------------------------------------------
# Non-negative PARAFAC (space x frequency x time) for EEG power tensors
# ---------------------------------------------------------------------------

def nn_parafac(tensor, rank, n_iter=300, seed=0, tol=1e-6):
    """Non-negative canonical polyadic decomposition by HALS (Cichocki & Phan 2009).

    ``tensor[c, f, s] ~ sum_r A[c, r] B[f, r] C[s, r]`` with all factors
    non-negative, the Miwakeichi et al. (2004) space-frequency-time
    decomposition of EEG power.  Columns are normalised so the spatial and
    temporal factors have unit sum and the spectral factor carries the scale.
    Returns ``(factors, fit)`` with ``fit = 1 - ||T - T_hat|| / ||T||``.
    """
    T = np.asarray(tensor, dtype=float)
    if (T < 0).any():
        raise ValueError("nn_parafac needs a non-negative tensor (use power, not log power)")
    rng = np.random.default_rng(seed)
    shape = T.shape
    factors = [rng.uniform(0.1, 1.0, size=(n, rank)) for n in shape]
    unf = [np.asarray(unfold(T, m)) for m in range(3)]
    norm_T = np.linalg.norm(T)
    prev = np.inf
    for it in range(n_iter):
        for m in range(3):
            others = [factors[k] for k in range(3) if k != m]
            # Khatri-Rao product of the other factors (order matches unfolding)
            kr = np.einsum("ir,jr->ijr", others[0], others[1]).reshape(-1, rank)
            G = kr.T @ kr                      # (rank, rank)
            MtK = unf[m] @ kr                  # (n_m, rank)
            A = factors[m]
            for r in range(rank):
                num = MtK[:, r] - A @ G[:, r] + A[:, r] * G[r, r]
                A[:, r] = np.maximum(num / max(G[r, r], 1e-12), 1e-12)
            factors[m] = A
        if it % 10 == 0:
            That = np.einsum("cr,fr,sr->cfs", *factors)
            err = np.linalg.norm(T - That) / norm_T
            if abs(prev - err) < tol:
                break
            prev = err
    # normalise: space and time factors to unit sum, scale into the spectral factor
    for m in (0, 2):
        s = factors[m].sum(axis=0, keepdims=True)
        factors[m] = factors[m] / s
        factors[1] = factors[1] * s
    That = np.einsum("cr,fr,sr->cfs", *factors)
    fit = 1.0 - np.linalg.norm(T - That) / norm_T
    order = np.argsort(-factors[1].sum(axis=0))
    factors = [f[:, order] for f in factors]
    return factors, float(fit)


def parafac_components(data, fs, rank=3, segment_s=2.0, f_lo=2.0, f_hi=45.0, n_iter=300):
    """Space-frequency-time PARAFAC of a run's Welch power tensor.

    Returns ``freqs`` and a list of components, each with ``spatial`` weights
    over channels (sum 1), ``log_psd`` (the spectral signature, in log power,
    directly usable by the spectral models), ``temporal`` weights over
    segments (sum 1), ``power_share`` (its fraction of total reconstructed
    power) and the alpha-band fraction of its spectral signature.
    """
    f, logT, n_seg = spectral_tensor(data, fs, segment_s, f_lo, f_hi)
    T = np.exp(logT)
    factors, fit = nn_parafac(T, rank, n_iter=n_iter)
    A, B, C = factors
    total = B.sum()
    comps = []
    band = (f >= 8) & (f <= 13)
    for r in range(rank):
        comps.append({"spatial": A[:, r], "log_psd": np.log(np.maximum(B[:, r], 1e-300)), "temporal": C[:, r],
                      "power_share": float(B[:, r].sum() / total), "alpha_fraction": float(B[band, r].sum() / B[:, r].sum())})
    return f, comps, fit, n_seg


def tucker_energy_denoise(epochs, energy=0.99):
    """Truncated HOSVD on ``(trials, channels, time)`` keeping the smallest rank per mode that
    retains *energy* of that mode's singular-value spectrum (the "trilinear basis" denoiser).

    Returns the reconstruction from the retained basis and the ranks kept.
    """
    X = np.asarray(epochs, dtype=float)
    ranks = []
    for mode in range(3):
        s = np.linalg.svd(np.asarray(unfold(X, mode)), compute_uv=False)
        c = np.cumsum(s ** 2) / np.sum(s ** 2)
        ranks.append(int(np.searchsorted(c, energy) + 1))
    core, factors = hosvd(X, ranks=ranks)
    out = np.asarray(core)
    for mode, U in enumerate(factors):
        out = np.asarray(fold(np.asarray(U) @ np.asarray(unfold(out, mode)), mode, tuple(out.shape[:mode]) + (U.shape[0],) + tuple(out.shape[mode + 1:])))
    return out, ranks


def evoked_trilinear_basis(evoked, energy=0.99):
    """Wang, Begleiter & Porjesz (2000) trilinear basis: HOSVD of an *evoked* tensor
    ``(groups, channels, time)`` (subjects, conditions or sub-averages along the first mode),
    keeping the smallest rank per mode that retains *energy* of that mode's spectrum.

    Returns ``(U_channels, U_time, ranks)``.  Build the basis on trial averages, where the
    retained energy is signal, and project single trials with :func:`project_trials`; applying
    the energy cut to raw single-trial epochs instead keeps artefact directions (see
    ``docs/bsd_results.md``).
    """
    X = np.asarray(evoked, dtype=float)
    ranks = []
    for mode in range(X.ndim):
        s = np.linalg.svd(np.asarray(unfold(X, mode)), compute_uv=False)
        c = np.cumsum(s ** 2) / np.sum(s ** 2)
        ranks.append(int(np.searchsorted(c, energy) + 1))
    _, factors = hosvd(X, ranks=ranks)
    return np.asarray(factors[1]), np.asarray(factors[2]), ranks


def project_trials(epochs, u_channels, u_time):
    """Project ``(trials, channels, time)`` epochs onto the channel and time subspaces of a trilinear basis."""
    E = np.asarray(epochs, dtype=float)
    pc = u_channels @ u_channels.T
    pt = u_time @ u_time.T
    return np.einsum("ij,njk,kl->nil", pc, E, pt)
