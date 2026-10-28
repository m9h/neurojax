"""Tests for the specparam / band-power baselines."""

import numpy as np
import pytest

pytest.importorskip("specparam")

from neurojax.analysis.spectral_baselines import (
    SPECTRAL_MEASURES,
    alpha_peak_frequency,
    band_powers,
    specparam_features,
    spectral_edge,
)

FS = 250.0
rng = np.random.RandomState(0)
T = np.arange(20000) / FS
# 1/f^1.5 background + 10 Hz alpha
_w = np.fft.rfft(rng.randn(20000))
_f = np.fft.rfftfreq(20000, 1 / FS)
_w[1:] /= _f[1:] ** 0.75
PINK = np.fft.irfft(_w, n=20000)
PINK /= PINK.std()
X = PINK + 1.5 * np.sin(2 * np.pi * 10.0 * T)


def test_specparam_recovers_exponent_and_alpha_peak():
    sp = specparam_features(X, FS)
    assert abs(sp["exponent"] - 1.5) < 0.3
    assert abs(sp["alpha_cf"] - 10.0) < 0.5
    assert sp["alpha_power"] > 0.3 and sp["r_squared"] > 0.9
    sp0 = specparam_features(PINK, FS)
    assert sp0["alpha_power"] == 0.0 and np.isnan(sp0["alpha_cf"])


def test_band_powers_sum_to_one_and_alpha_dominates():
    bp = band_powers(X, FS)
    assert abs(sum(v for k, v in bp.items() if k.startswith("relpow_")) - 1.0) < 0.05
    assert bp["relpow_alpha"] == max(v for k, v in bp.items() if k.startswith("relpow_"))


def test_edge_and_peak():
    assert 9.5 <= alpha_peak_frequency(X, FS) <= 10.5
    assert spectral_edge(X, FS) < spectral_edge(rng.randn(20000), FS)


def test_registry_runs():
    for m in SPECTRAL_MEASURES.values():
        v = m(X[:2500], FS)
        assert np.isfinite(v) or m.name == "specparam_alpha_cf"
    assert len(SPECTRAL_MEASURES) >= 12
