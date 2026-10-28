"""Tests for low-frequency handling alternatives and the sensitivity table."""

import numpy as np
import pytest

from neurojax.analysis.nltsa import MEASURES
from neurojax.bench.consciousness_bench import Recording
from neurojax.bench.sensitivity import PREPROCS, dmd_lowmode_removal, emd_detrend, hpf, sensitivity_table

FS = 100.0
T = np.arange(6000) / FS
rng = np.random.RandomState(0)
SLOW = 3.0 * np.sin(2 * np.pi * 0.05 * T) + 0.002 * T ** 1.0  # drift + 0.05 Hz wander
FAST = np.sin(2 * np.pi * 10 * T) + 0.3 * rng.randn(len(T))
X = (SLOW + FAST)[None].astype(np.float32)


def _slow_power_left(Y):
    f = np.fft.rfftfreq(Y.shape[-1], 1 / FS)
    P = np.abs(np.fft.rfft(Y - Y.mean(-1, keepdims=True), axis=-1)) ** 2
    return P[..., (f > 0) & (f < 0.3)].sum() / P[..., f >= 0.3].sum()


@pytest.mark.parametrize("fn", [lambda X, fs: hpf(X, fs, 0.5), emd_detrend, dmd_lowmode_removal])
def test_slow_component_removed_fast_kept(fn):
    pytest.importorskip("PyEMD")
    Y = fn(X, FS)
    assert Y.shape == X.shape and Y.dtype == np.float32
    assert _slow_power_left(Y) < 0.05 * _slow_power_left(X)
    # 10 Hz content preserved within 15 % (edges excluded)
    sl = slice(500, -500)
    r = np.corrcoef(Y[0, sl], FAST[sl])[0, 1]
    assert r > 0.9


def test_sensitivity_table_shape():
    pytest.importorskip("PyEMD")
    recs = []
    for s in range(5):
        for cond, r in (("a", 0.97), ("b", 0.6)):
            a1, a2 = 2 * r * np.cos(2 * np.pi * 10 / FS), -(r ** 2)
            x = rng.randn(2, 3200)
            for t in range(2, 3200):
                x[:, t] += a1 * x[:, t - 1] + a2 * x[:, t - 2]
            recs.append(Recording("synth", f"s{s}", cond, x[:, 200:].astype(np.float32), FS))
    meas = {k: MEASURES[k] for k in ("lz76_mean", "dfa", "hjorth_mobility")}
    pre = {k: PREPROCS[k] for k in ("hpf_0.5", "hpf_2.0", "dmd_0.5")}
    T = sensitivity_table(recs, meas, "a", "b", preprocs=pre, epoch_lengths=(5.0, 10.0), total_s=30.0)
    assert set(T.columns) >= {"preproc", "epoch_s", "measure", "cohen_d", "rho_ref", "mantel_ref"}
    assert len(T) == 3 * 2 * 3
    ref = T[(T.preproc == "hpf_0.5") & (T.epoch_s == 10.0)]
    assert np.allclose(ref.rho_ref, 1.0) and np.allclose(ref.mantel_ref, 1.0)
