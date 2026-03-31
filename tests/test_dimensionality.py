"""Tests for dimensionality estimation: neurojax Laplace vs MNE ICA.

Validates the PCA rank=80 choice for TDE+PCA pipeline.
"""
import numpy as np
import pytest
import mne


def _make_low_rank_data(n_features, n_samples, true_rank, noise=0.01, seed=42):
    """Create data with known intrinsic dimensionality."""
    rng = np.random.default_rng(seed)
    sources = rng.standard_normal((true_rank, n_samples))
    mixing = rng.standard_normal((n_features, true_rank))
    data = mixing @ sources + noise * rng.standard_normal((n_features, n_samples))
    return data.astype(np.float32)


@pytest.fixture
def rank3_data():
    return _make_low_rank_data(30, 5000, 3)

@pytest.fixture
def rank10_data():
    return _make_low_rank_data(30, 5000, 10)

@pytest.fixture
def rank30_data():
    return _make_low_rank_data(50, 5000, 30)


class TestNeurojaxLaplace:
    def test_returns_positive(self, rank3_data):
        from neurojax.preprocessing.ica_comparison import estimate_dim_neurojax
        dim = estimate_dim_neurojax(rank3_data)
        assert dim.n_components > 0

    def test_rank3_detected(self, rank3_data):
        from neurojax.preprocessing.ica_comparison import estimate_dim_neurojax
        dim = estimate_dim_neurojax(rank3_data)
        assert 2 <= dim.n_components <= 6, \
            f"Expected ~3, got {dim.n_components}"

    def test_rank10_detected(self, rank10_data):
        from neurojax.preprocessing.ica_comparison import estimate_dim_neurojax
        dim = estimate_dim_neurojax(rank10_data)
        assert 7 <= dim.n_components <= 15, \
            f"Expected ~10, got {dim.n_components}"

    def test_monotonic_with_true_rank(self):
        from neurojax.preprocessing.ica_comparison import estimate_dim_neurojax
        dims = []
        for rank in [3, 10, 20]:
            data = _make_low_rank_data(40, 5000, rank, seed=rank)
            dim = estimate_dim_neurojax(data)
            dims.append(dim.n_components)
        assert dims[0] < dims[1] < dims[2], \
            f"Expected monotonic, got {dims}"


class TestMNEICADimensionality:
    def _make_raw(self, n_ch=30, duration=30, sfreq=250):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((n_ch, int(duration * sfreq))) * 1e-13
        info = mne.create_info([f"MEG{i:04d}" for i in range(n_ch)], sfreq, ["mag"] * n_ch)
        return mne.io.RawArray(data, info, verbose=False)

    def test_auto_components(self):
        from neurojax.preprocessing.ica_comparison import run_mne_ica
        raw = self._make_raw()
        _, info = run_mne_ica(raw, n_components=None)
        assert info["n_components"] > 0
        assert info["n_components"] <= 30

    def test_fixed_components(self):
        from neurojax.preprocessing.ica_comparison import run_mne_ica
        raw = self._make_raw()
        _, info = run_mne_ica(raw, n_components=10)
        assert info["n_components"] == 10


class TestCrossMethodAgreement:
    def test_both_methods_similar_on_rank10(self):
        """neurojax Laplace and MNE ICA should roughly agree on rank."""
        from neurojax.preprocessing.ica_comparison import estimate_dim_neurojax, run_mne_ica
        data = _make_low_rank_data(30, 5000, 10)
        dim_nj = estimate_dim_neurojax(data)

        info = mne.create_info([f"MEG{i:04d}" for i in range(30)], 250.0, ["mag"] * 30)
        raw = mne.io.RawArray(data * 1e-13, info, verbose=False)
        _, mne_info = run_mne_ica(raw, n_components=None)

        # Should agree within +/- 8 components
        diff = abs(dim_nj.n_components - mne_info["n_components"])
        assert diff < 8, \
            f"Too different: neurojax={dim_nj.n_components}, MNE={mne_info['n_components']}"


class TestTDEPCARank:
    def test_rank80_sufficient_for_68_parcels_15_lags(self):
        """With 68 parcels × 15 lags = 1020 TDE features, rank=80 should
        capture the dominant structure if true rank is modest."""
        from neurojax.preprocessing.ica_comparison import estimate_dim_neurojax
        # Simulate TDE-like data: 68 channels, 15 lags → 1020 features
        # with true rank ~20 (typical for resting MEG)
        data = _make_low_rank_data(1020, 3000, 20, noise=0.1, seed=42)
        dim = estimate_dim_neurojax(data[:100])  # subsample features for speed
        # Rank=80 should be well above the estimated intrinsic dim
        assert dim.n_components <= 80, \
            f"Estimated dim={dim.n_components} > 80, rank=80 may be insufficient"

    def test_real_parcellated_data(self):
        """If sub-08033 parcellated data exists, estimate its dimensionality."""
        import os
        path = "/data/raw/wand/derivatives/neurojax-meg/sub-08033/ses-01/source/parcellated_desikan.npy"
        if not os.path.exists(path):
            pytest.skip("No parcellated data available")

        from neurojax.preprocessing.ica_comparison import estimate_dim_neurojax
        parc = np.load(path)  # (T, 68)
        assert parc.shape[1] == 68
        # Transpose to (features, samples) for dim estimation
        dim = estimate_dim_neurojax(parc.T.astype(np.float32))
        assert 5 <= dim.n_components <= 68, \
            f"Unexpected dim={dim.n_components} for 68-parcel MEG"
