"""Batched (vmap, accelerator) recurrence / chaos measures must reproduce the
scalar numpy implementations in nltsa.py, which are themselves checked
against pyunicorn / nolds in test_nltsa_oracles.py."""

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from neurojax.analysis import nltsa
from neurojax.analysis.nltsa_batched import (BATCHED, batched_chaos_01, batched_correlation_dimension,
                                             batched_lyapunov, batched_rqa_profile, score_batched)

FS = 250.0
T = 1500


def _signals():
    rng = np.random.default_rng(1)
    white = rng.normal(size=T)
    sos = butter(4, [1, 40], btype="band", fs=FS, output="sos")
    eeg = sosfiltfilt(sos, np.cumsum(rng.normal(size=T + 500)))[500:]
    eeg = eeg / eeg.std()
    x = 0.4
    logi = np.empty(T)
    for i in range(T):
        x = 3.9 * x * (1 - x)
        logi[i] = x
    sine = np.sin(2 * np.pi * 10 * np.arange(T) / FS) + 0.05 * rng.normal(size=T)
    return np.stack([white, eeg, logi - logi.mean(), sine])


@pytest.fixture(scope="module")
def X():
    return _signals()


def test_rqa_profile_matches_numpy(X):
    out = batched_rqa_profile(X, m=3, tau=1, recurrence_rate=0.05, l_min=2, v_min=2)
    for b in range(X.shape[0]):
        ref = nltsa.rqa_profile(X[b], m=3, tau=1, recurrence_rate=0.05, l_min=2, v_min=2)
        for k, v in ref.items():
            assert float(out[k][b]) == pytest.approx(v, rel=2e-2, abs=2e-3), (k, b)


def test_lyapunov_matches_numpy(X):
    out = batched_lyapunov(X, FS, m=2, tau=1, k_max=20)
    for b in range(X.shape[0]):
        ref = nltsa.lyapunov_rosenstein(X[b], FS, m=2, tau=1, k_max=20)
        assert float(out[b]) == pytest.approx(ref, rel=2e-2, abs=0.5), b


def test_correlation_dimension_matches_numpy(X):
    out = batched_correlation_dimension(X, m=5, tau=1)
    for b in range(X.shape[0]):
        ref = nltsa.correlation_dimension(X[b], m=5, tau=1)
        assert float(out[b]) == pytest.approx(ref, rel=2e-2, abs=0.02), b
    assert np.isnan(float(batched_correlation_dimension(np.zeros((1, T)), m=5, tau=1)[0]))


def test_chaos_01_matches_numpy(X):
    out = batched_chaos_01(X)
    for b in range(X.shape[0]):
        ref = nltsa.chaos_01(X[b])
        assert float(out[b]) == pytest.approx(ref, abs=0.02), b


def test_score_batched_covers_registry_measures(X):
    names = [n for n in nltsa.MEASURES if n in BATCHED]
    assert {"lyapunov", "correlation_dimension", "chaos_01", "rqa_determinism", "rn_transitivity"} <= set(names)
    ms = {n: nltsa.MEASURES[n] for n in names}
    out = score_batched(X, FS, ms)
    assert set(out) == set(names)
    for n in names:
        assert out[n].shape == (X.shape[0],)
        for b in range(X.shape[0]):
            ref = ms[n](X[b], FS)
            if np.isnan(ref):
                assert np.isnan(out[n][b])
            else:
                assert float(out[n][b]) == pytest.approx(ref, rel=2e-2, abs=0.02), (n, b)


def test_chunking_gives_same_answer(X):
    full = batched_rqa_profile(X, chunk=None)["determinism"]
    chunked = batched_rqa_profile(X, chunk=1)["determinism"]
    np.testing.assert_allclose(np.asarray(full), np.asarray(chunked), rtol=1e-6)
