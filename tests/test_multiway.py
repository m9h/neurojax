"""Multiway reductions for EEG spectra: tensor construction, HOSVD weights, PARAFAC recovery, denoising."""

import numpy as np
import pytest

from neurojax.bsd.multiway import denoise_epochs, hosvd_spectrum, nn_parafac, parafac_components, spectral_tensor


def test_spectral_tensor_shape_and_band():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 250 * 30))
    f, T, n = spectral_tensor(x, 250.0, segment_s=2.0, f_lo=2.0, f_hi=45.0)
    assert T.shape == (6, f.size, n) and n == 15 and f[0] >= 2.0 and f[-1] <= 45.0
    assert np.isfinite(T).all()


def test_nn_parafac_recovers_planted_space_frequency_time_components():
    rng = np.random.default_rng(1)
    C, F, S = 12, 40, 30
    space = np.zeros((C, 2)); space[:4, 0] = 1.0; space[8:, 1] = 1.0
    f = np.linspace(2, 45, F)
    spec = np.stack([np.exp(-((f - 10) / 1.5) ** 2) + 0.05, 1.0 / f], axis=1)
    time = np.stack([1 + 0.5 * np.sin(np.arange(S) / 4), np.ones(S)], axis=1)
    T = np.einsum("cr,fr,sr->cfs", space, spec, time)
    T = T * np.exp(0.1 * rng.normal(size=T.shape))
    factors, fit = nn_parafac(T, rank=2, n_iter=500)
    assert fit > 0.9
    A, B, Cf = factors
    # one recovered spatial factor loads on the first block, the other on the last
    a0 = A[:4].sum(axis=0) / A.sum(axis=0); a1 = A[8:].sum(axis=0) / A.sum(axis=0)
    assert max(a0) > 0.9 and max(a1) > 0.9 and np.argmax(a0) != np.argmax(a1)
    # the alpha-like spectral factor peaks near 10 Hz
    alpha_r = int(np.argmax(a0))
    assert abs(f[np.argmax(B[:, alpha_r])] - 10.0) < 2.0


def test_parafac_components_api():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(8, 250 * 40))
    x[:3] += 3 * np.sin(2 * np.pi * 10 * np.arange(x.shape[1]) / 250.0)   # posterior-like alpha on 3 channels
    f, comps, fit, n = parafac_components(x, 250.0, rank=2)
    assert len(comps) == 2 and all(abs(c["spatial"].sum() - 1) < 1e-6 for c in comps)
    alpha = max(comps, key=lambda c: c["alpha_fraction"])
    assert alpha["alpha_fraction"] > 0.3 and alpha["spatial"][:3].sum() > 0.6
    assert all(np.isfinite(c["log_psd"]).all() for c in comps)


def test_hosvd_spectrum_returns_normalised_weights():
    rng = np.random.default_rng(3)
    f, T, n = spectral_tensor(rng.normal(size=(5, 250 * 20)), 250.0)
    r = hosvd_spectrum(T)
    assert abs(float(r["channel_weights"].sum()) - 1) < 1e-5 and r["log_psd"].shape == (f.size,)
    assert 0 < r["energy_channel_1"] <= 1


def test_denoise_epochs_reduces_error_on_low_rank_signal():
    rng = np.random.default_rng(4)
    t = np.arange(400) / 250.0
    sig = np.einsum("i,j,k->ijk", 1 + 0.2 * rng.normal(size=20), rng.normal(size=8), np.sin(2 * np.pi * 10 * t))
    noisy = sig + 0.5 * rng.normal(size=sig.shape)
    den, kept = denoise_epochs(noisy)
    assert np.linalg.norm(den - sig) < np.linalg.norm(noisy - sig)
    assert all(k >= 1 for k in kept)
