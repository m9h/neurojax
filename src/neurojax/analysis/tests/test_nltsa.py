"""Tests for neurojax.analysis.nltsa — the measure registry behind the
consciousness complexity benchmark (Maschke et al. 2024 Comms Biol measure
set + the complexity-entropy causal plane of Newman et al. 2025)."""

import jax
import numpy as np
import pytest

from neurojax.analysis import nltsa
from neurojax.analysis.nltsa import (
    MEASURES,
    aperiodic_exponent,
    branching_ratio,
    chaos_01,
    compute_measures,
    correlation_dimension,
    dfa,
    higuchi_fd,
    hjorth,
    katz_fd,
    lyapunov_rosenstein,
    multiscale_entropy,
    pair_correlation,
    permutation_entropy,
    petrosian_fd,
    statistical_complexity,
)

FS = 250.0
rng = np.random.RandomState(0)
WHITE = rng.randn(5000)
BROWN = np.cumsum(rng.randn(5000))
T = np.arange(5000) / FS
SINE = np.sin(2 * np.pi * 10.0 * T)


def _logistic(n, r=4.0, x0=0.3):
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x


class TestOrdinal:
    def test_permutation_entropy_bounds(self):
        assert permutation_entropy(np.arange(1000.0), m=3) == pytest.approx(0.0)
        assert 0.97 < permutation_entropy(WHITE, m=3) <= 1.0
        assert permutation_entropy(SINE, m=3) < permutation_entropy(WHITE, m=3)

    def test_statistical_complexity_is_zero_at_extremes(self):
        # Rosso et al. 2007: C = 0 for both a uniform pattern distribution
        # (white noise) and a delta distribution (monotone ramp).
        assert statistical_complexity(WHITE, m=3) < 0.02
        assert statistical_complexity(np.arange(1000.0), m=3) == pytest.approx(0.0)
        assert statistical_complexity(_logistic(5000), m=4) > 0.1


class TestFractal:
    def test_katz_line_is_one(self):
        assert katz_fd(np.arange(500.0)) == pytest.approx(1.0, abs=1e-6)
        assert katz_fd(WHITE) > katz_fd(SINE)

    def test_higuchi_white_noise_near_two(self):
        assert 1.85 < higuchi_fd(WHITE, k_max=10) < 2.05
        assert 1.4 < higuchi_fd(BROWN, k_max=10) < 1.6

    def test_petrosian_range(self):
        assert 1.0 < petrosian_fd(WHITE) < 1.1
        assert petrosian_fd(WHITE) > petrosian_fd(SINE)


class TestScaling:
    def test_dfa_white_vs_brown(self):
        assert abs(dfa(WHITE) - 0.5) < 0.12
        assert abs(dfa(BROWN) - 1.5) < 0.15

    def test_aperiodic_exponent_brownian_is_two(self):
        assert abs(aperiodic_exponent(BROWN, fs=FS, f_range=(1, 40)) - 2.0) < 0.3
        assert abs(aperiodic_exponent(WHITE, fs=FS, f_range=(1, 40))) < 0.3


class TestHjorth:
    def test_sine_mobility_and_complexity(self):
        mob, comp = hjorth(SINE)
        assert mob == pytest.approx(2 * np.pi * 10.0 / FS, rel=0.02)
        assert comp == pytest.approx(1.0, rel=0.02)
        assert hjorth(WHITE)[0] > mob


class TestChaos:
    def test_chaos_01_regular_vs_noise(self):
        assert chaos_01(SINE) < 0.2
        assert chaos_01(WHITE) > 0.8

    def test_lyapunov_logistic_positive_sine_small(self):
        lam = lyapunov_rosenstein(_logistic(3000), fs=1.0, m=2, tau=1, theiler=5, k_max=5)
        assert 0.3 < lam < 1.0  # true value ln 2 = 0.69
        assert lyapunov_rosenstein(SINE, fs=FS, m=2, tau=1, theiler=25, k_max=10) < 0.3 * FS

    def test_correlation_dimension_logistic(self):
        d2 = correlation_dimension(_logistic(3000), m=2, tau=1)
        assert 0.7 < d2 < 1.3


class TestRQA:
    def test_determinism_matches_reference_line_counter(self):
        from neurojax.analysis.nltsa import _diagonal_line_points, rqa_determinism
        from neurojax.analysis.recurrence import determinism
        rng_ = np.random.RandomState(3)
        R = (rng_.rand(60, 60) < 0.2).astype(np.float32)
        R = np.maximum(R, R.T)
        np.fill_diagonal(R, 1)
        on, tot = _diagonal_line_points(R, 2)
        assert on / tot == pytest.approx(float(determinism(R, l_min=2)))
        assert rqa_determinism(SINE[:1500]) > rqa_determinism(WHITE[:1500])


class TestEntropyScales:
    def test_multiscale_entropy_white_noise_decreases(self):
        mse = multiscale_entropy(WHITE, scales=10, m=2, r=0.15)
        assert mse.shape == (10,)
        assert mse[0] > mse[-1]
        assert np.all(np.isfinite(mse))


class TestMultivariate:
    def test_pair_correlation_sync_vs_independent(self):
        t = T
        key = jax.random.PRNGKey(0)
        noise = np.asarray(jax.random.normal(key, (8, len(t))))
        sync = np.sin(2 * np.pi * 10 * t)[None] + 0.1 * noise
        phases = rng.rand(8, 1) * 2 * np.pi
        indep = np.sin(2 * np.pi * 10 * t[None] + phases) + 0.1 * noise
        assert pair_correlation(sync, fs=FS) < pair_correlation(indep, fs=FS)

    def test_branching_ratio_finite(self):
        X = rng.randn(16, 5000)
        sigma = branching_ratio(X, fs=FS, threshold=2.0, bin_ms=8)
        assert np.isfinite(sigma) and sigma >= 0


class TestRegistry:
    def test_every_measure_runs_on_ar2(self):
        X = np.zeros((4, 2500))
        e = rng.randn(4, 3000)
        a1, a2 = 2 * 0.95 * np.cos(2 * np.pi * 10 / FS), -(0.95 ** 2)
        for t in range(2, 3000):
            e[:, t] += a1 * e[:, t - 1] + a2 * e[:, t - 2]
        X = e[:, 500:]
        out = compute_measures(X, fs=FS)
        assert set(out) == set(MEASURES)
        for k, v in out.items():
            assert np.isfinite(v), k
        names = [m.name for m in MEASURES.values()]
        assert len(names) == len(set(names)) and len(names) >= 20

    def test_measures_declare_family_and_reference(self):
        for m in MEASURES.values():
            assert m.family in nltsa.FAMILIES
            assert m.reference
            assert isinstance(m.multivariate, bool)
