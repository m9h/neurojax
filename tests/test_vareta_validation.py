"""Tests for VARETA vs MNE sLORETA/dSPM source localization validation.

Red-green TDD on synthetic data with known source locations.
vareta(data, gain, noise_cov, n_iter) → (source, source_variance, evidence)
"""
import numpy as np
import pytest


def _make_forward_and_data(n_channels=30, n_sources=50, n_times=1000,
                           active_sources=None, snr=5.0, seed=42):
    rng = np.random.default_rng(seed)
    if active_sources is None:
        active_sources = [20]

    L = rng.standard_normal((n_channels, n_sources)).astype(np.float64)
    S = np.zeros((n_sources, n_times), dtype=np.float64)
    for idx in active_sources:
        S[idx] = rng.standard_normal(n_times)

    signal = L @ S
    noise_std = np.std(signal) / snr
    noise = noise_std * rng.standard_normal((n_channels, n_times))
    data = signal + noise
    noise_cov = np.eye(n_channels) * noise_std ** 2
    return L, data, S, noise_cov


@pytest.fixture
def single_dipole():
    return _make_forward_and_data(active_sources=[20])

@pytest.fixture
def distributed_sources():
    return _make_forward_and_data(active_sources=[10, 25, 40])

@pytest.fixture
def low_snr():
    return _make_forward_and_data(active_sources=[20], snr=1.0)


def _run_vareta(L, data, noise_cov, n_iter=20):
    from neurojax.source.vareta import vareta
    import jax.numpy as jnp
    source, source_var, evidence = vareta(
        jnp.array(data), jnp.array(L), jnp.array(noise_cov), n_iter=n_iter)
    return np.array(source), np.array(source_var), np.array(evidence)


class TestVARETABasic:
    def test_import(self):
        from neurojax.source.vareta import vareta
        assert callable(vareta)

    def test_output_shape(self, single_dipole):
        L, data, _, noise_cov = single_dipole
        source, svar, evidence = _run_vareta(L, data, noise_cov, n_iter=5)
        assert source.shape == (50, 1000)
        assert svar.shape == (50,)

    def test_output_finite(self, single_dipole):
        L, data, _, noise_cov = single_dipole
        source, svar, evidence = _run_vareta(L, data, noise_cov, n_iter=5)
        assert np.all(np.isfinite(source))
        assert np.all(np.isfinite(svar))


class TestSingleDipoleLocalization:
    def test_vareta_finds_peak(self, single_dipole):
        L, data, _, noise_cov = single_dipole
        source, _, _ = _run_vareta(L, data, noise_cov, n_iter=20)
        power = np.mean(source ** 2, axis=1)
        peak = np.argmax(power)
        assert abs(peak - 20) <= 3, f"VARETA peak at {peak}, expected near 20"

    def test_pseudoinverse_finds_peak(self, single_dipole):
        L, data, _, _ = single_dipole
        pinv_est = np.linalg.pinv(L) @ data
        power = np.mean(pinv_est ** 2, axis=1)
        peak = np.argmax(power)
        assert abs(peak - 20) <= 3, f"Pseudoinverse peak at {peak}, expected near 20"


class TestDistributedSources:
    def test_vareta_recovers_multiple_sources(self, distributed_sources):
        L, data, _, noise_cov = distributed_sources
        source, _, _ = _run_vareta(L, data, noise_cov, n_iter=20)
        power = np.mean(source ** 2, axis=1)
        top10 = np.argsort(power)[-10:]
        for active in [10, 25, 40]:
            assert any(abs(t - active) <= 3 for t in top10), \
                f"Source {active} not in top 10: {sorted(top10)}"


class TestLowSNR:
    def test_vareta_finite_at_low_snr(self, low_snr):
        L, data, _, noise_cov = low_snr
        source, svar, _ = _run_vareta(L, data, noise_cov, n_iter=10)
        assert np.all(np.isfinite(source))

    def test_vareta_still_localizes_at_low_snr(self, low_snr):
        L, data, _, noise_cov = low_snr
        source, _, _ = _run_vareta(L, data, noise_cov, n_iter=30)
        power = np.mean(source ** 2, axis=1)
        peak = np.argmax(power)
        assert abs(peak - 20) <= 10, f"Low SNR: peak at {peak}, expected near 20"


class TestSourceMapCorrelation:
    def test_vareta_pseudoinverse_correlation(self, single_dipole):
        L, data, _, noise_cov = single_dipole
        source, _, _ = _run_vareta(L, data, noise_cov, n_iter=20)
        vareta_power = np.mean(source ** 2, axis=1)

        pinv_est = np.linalg.pinv(L) @ data
        pinv_power = np.mean(pinv_est ** 2, axis=1)

        r = np.corrcoef(vareta_power, pinv_power)[0, 1]
        assert r > 0.3, f"Weak correlation: r={r:.3f}"
