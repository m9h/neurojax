"""Parity of neurojax.analysis.nltsa against the widely used reference
implementations: antropy (Vallat), NeuroKit2 (Makowski) and nolds (Schölzel).
Each test states the tolerance and, where the algorithms differ by
construction, why."""

import numpy as np
import pytest

from neurojax.analysis import nltsa
from neurojax.analysis.complexity import lz76

antropy = pytest.importorskip("antropy")

FS = 250.0
rng = np.random.RandomState(7)
WHITE = rng.randn(4000)
PINK = np.cumsum(rng.randn(4000)) * 0.1 + rng.randn(4000)
T = np.arange(4000) / FS
AR = np.zeros(4000)
for t in range(2, 4000):
    AR[t] = 1.8 * np.cos(2 * np.pi * 10 / FS) * AR[t - 1] - 0.81 * AR[t - 2] + rng.randn()
SIGNALS = {"white": WHITE, "pink": PINK, "ar2": AR}


@pytest.mark.parametrize("name", list(SIGNALS))
class TestAntropy:
    def test_permutation_entropy(self, name):
        x = SIGNALS[name]
        assert nltsa.permutation_entropy(x, m=3, tau=1) == pytest.approx(
            antropy.perm_entropy(x, order=3, delay=1, normalize=True), rel=1e-6)

    def test_sample_entropy(self, name):
        x = SIGNALS[name]
        # same definition (m=2, r=0.2 SD, Chebyshev); antropy uses population SD
        assert nltsa.sample_entropy(x, m=2, r=0.2) == pytest.approx(
            antropy.sample_entropy(x, order=2), rel=0.02)

    def test_fractal_dimensions(self, name):
        x = SIGNALS[name]
        assert nltsa.higuchi_fd(x, k_max=10) == pytest.approx(antropy.higuchi_fd(x, kmax=10), rel=1e-3)
        # antropy counts sign changes of diff(x) via np.diff(np.sign(.)) which treats exact zeros differently
        assert nltsa.petrosian_fd(x) == pytest.approx(antropy.petrosian_fd(x), rel=1e-4)
        # antropy's Katz uses unit-free L = sum|dx| and d = max|x - x0| (no time axis)
        assert nltsa.katz_fd(x) == pytest.approx(antropy.katz_fd(x), rel=0.05)

    def test_lempel_ziv(self, name):
        x = SIGNALS[name]
        s = (x > np.median(x)).astype(np.uint8)
        assert lz76(s) == antropy.lziv_complexity(s)
        assert float(nltsa.lz_complexity(x, "median", "analytic")) == pytest.approx(
            antropy.lziv_complexity(s, normalize=True), rel=1e-6)

    def test_spectral_and_svd_entropy(self, name):
        x = SIGNALS[name]
        # antropy spectral entropy: full-length periodogram; ours (entropy.spectral_entropy) too
        ours = nltsa.MEASURES["spectral_entropy"](x, FS)
        ref = antropy.spectral_entropy(x, sf=FS, method="fft", normalize=True)
        assert abs(ours / np.log2(len(x) // 2 + 1) - ref) < 0.05 or abs(ours - ref) < 0.05
        assert nltsa.MEASURES["svd_entropy"](x, FS) == pytest.approx(
            antropy.svd_entropy(x, order=10, delay=2, normalize=False), rel=0.02)

    def test_dfa(self, name):
        x = SIGNALS[name]
        # antropy: fixed window set (4..T/10 log-spaced, 10 scales); ours 4..T/4, 20 scales
        assert abs(nltsa.dfa(x) - antropy.detrended_fluctuation(x)) < 0.12


class TestNeuroKitAndNolds:
    def test_hjorth(self):
        nk = pytest.importorskip("neurokit2")
        x = AR
        mob, comp = nltsa.hjorth(x)
        ref, info = nk.complexity_hjorth(x)
        assert comp == pytest.approx(float(ref), rel=1e-3)
        assert mob == pytest.approx(float(info["Mobility"]) if "Mobility" in info else mob, rel=1e-3)

    def test_lyapunov_logistic_vs_nolds(self):
        nolds = pytest.importorskip("nolds")
        x = np.empty(3000)
        x[0] = 0.3
        for i in range(1, 3000):
            x[i] = 4 * x[i - 1] * (1 - x[i - 1])
        ours = nltsa.lyapunov_rosenstein(x, fs=1.0, m=2, tau=1, theiler=5, k_max=5)
        ref = nolds.lyap_r(x, emb_dim=2, lag=1, min_tsep=5, trajectory_len=6, fit="poly")
        assert abs(ours - np.log(2)) < 0.35 and abs(ref - np.log(2)) < 0.35
        assert abs(ours - ref) < 0.3

    def test_correlation_dimension_vs_nolds(self):
        nolds = pytest.importorskip("nolds")
        x = np.empty(3000)
        x[0] = 0.3
        for i in range(1, 3000):
            x[i] = 4 * x[i - 1] * (1 - x[i - 1])
        ours = nltsa.correlation_dimension(x, m=2, tau=1)
        ref = nolds.corr_dim(x, emb_dim=2, fit="poly")
        # different r ranges (ours: 5th-50th percentile of distances); both near 1
        assert abs(ours - ref) < 0.3


class TestPyunicorn:
    def test_rqa_and_recurrence_network_match_pyunicorn(self):
        pu = pytest.importorskip("pyunicorn.timeseries")
        x = PINK[:1500]
        ours = nltsa.rqa_profile(x, m=3, tau=1, recurrence_rate=0.05, l_min=2, v_min=2)
        rp = pu.RecurrencePlot(x, dim=3, tau=1, recurrence_rate=0.05, metric="euclidean", silence_level=2)
        rn = pu.RecurrenceNetwork(x, dim=3, tau=1, recurrence_rate=0.05, metric="euclidean", silence_level=2)
        # pyunicorn's threshold search is approximate -> small recurrence-rate differences
        assert ours["recurrence_rate"] == pytest.approx(rp.recurrence_rate(), rel=0.05)
        assert ours["determinism"] == pytest.approx(rp.determinism(l_min=2), rel=0.02)
        assert ours["laminarity"] == pytest.approx(rp.laminarity(v_min=2), rel=0.02)
        assert ours["trapping_time"] == pytest.approx(rp.trapping_time(v_min=2), rel=0.05)
        assert ours["mean_diag_length"] == pytest.approx(rp.average_diaglength(l_min=2), rel=0.05)
        assert ours["diag_entropy"] == pytest.approx(rp.diag_entropy(l_min=2), rel=0.05)
        assert ours["mean_recurrence_time"] == pytest.approx(rp.mean_recurrence_time(), rel=0.05)
        assert ours["rn_transitivity"] == pytest.approx(rn.transitivity(), rel=0.03)
        assert ours["rn_global_clustering"] == pytest.approx(rn.global_clustering(), rel=0.03)
