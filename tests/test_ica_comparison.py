"""Tests for multi-oracle ICA comparison.

Red-green TDD: validates NIfTI bridge, component matching, and oracle agreement.
"""
import numpy as np
import pytest
import mne


@pytest.fixture
def synthetic_meg_30ch():
    """30-channel synthetic MEG with 3 known sources."""
    sfreq = 250.0
    n_times = int(30 * sfreq)
    n_channels = 30
    n_sources = 3

    rng = np.random.default_rng(42)

    # 3 independent sources
    t = np.arange(n_times) / sfreq
    s1 = np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
    s2 = np.sin(2 * np.pi * 22 * t)  # 22 Hz beta
    s3 = rng.standard_normal(n_times) * 0.3  # noise source

    sources = np.array([s1, s2, s3])  # (3, n_times)

    # Random mixing matrix
    A = rng.standard_normal((n_channels, n_sources))
    data = A @ sources + rng.standard_normal((n_channels, n_times)) * 0.1

    data *= 1e-13  # MEG scale

    ch_names = [f"MEG{i:04d}" for i in range(n_channels)]
    info = mne.create_info(ch_names, sfreq, ["mag"] * n_channels)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw, A, sources


class TestMEGToNIfTI:
    def test_creates_valid_nifti(self, tmp_path, synthetic_meg_30ch):
        from neurojax.preprocessing.ica_comparison import meg_to_nifti
        raw, _, _ = synthetic_meg_30ch
        data = raw.get_data()
        out_path = str(tmp_path / "test.nii.gz")
        meg_to_nifti(data, raw.info["sfreq"], out_path)
        assert (tmp_path / "test.nii.gz").exists()

    def test_nifti_shape_correct(self, tmp_path, synthetic_meg_30ch):
        import nibabel as nib
        from neurojax.preprocessing.ica_comparison import meg_to_nifti
        raw, _, _ = synthetic_meg_30ch
        data = raw.get_data()
        out_path = str(tmp_path / "test.nii.gz")
        meg_to_nifti(data, raw.info["sfreq"], out_path)
        img = nib.load(out_path)
        # Shape should be (n_channels, 1, 1, n_times)
        assert img.shape == (30, 1, 1, data.shape[1])

    def test_nifti_data_preserved(self, tmp_path, synthetic_meg_30ch):
        import nibabel as nib
        from neurojax.preprocessing.ica_comparison import meg_to_nifti
        raw, _, _ = synthetic_meg_30ch
        data = raw.get_data()
        out_path = str(tmp_path / "test.nii.gz")
        meg_to_nifti(data, raw.info["sfreq"], out_path)
        img = nib.load(out_path)
        recovered = img.get_fdata().squeeze()
        # float32 precision
        np.testing.assert_allclose(recovered, data, rtol=1e-5)


class TestComponentMatching:
    def test_identical_matrices_perfect_match(self):
        from neurojax.preprocessing.ica_comparison import match_components
        rng = np.random.default_rng(42)
        A = rng.standard_normal((30, 5))
        corr, matching = match_components(A, A)
        assert corr > 0.99

    def test_permuted_columns_still_match(self):
        from neurojax.preprocessing.ica_comparison import match_components
        rng = np.random.default_rng(42)
        A = rng.standard_normal((30, 5))
        B = A[:, [3, 1, 4, 0, 2]]  # permuted
        corr, matching = match_components(A, B)
        assert corr > 0.99

    def test_sign_flipped_columns_match(self):
        from neurojax.preprocessing.ica_comparison import match_components
        rng = np.random.default_rng(42)
        A = rng.standard_normal((30, 5))
        B = A.copy()
        B[:, 0] *= -1
        B[:, 3] *= -1
        corr, matching = match_components(A, B)
        assert corr > 0.99  # absolute correlation handles sign flips

    def test_different_n_components(self):
        from neurojax.preprocessing.ica_comparison import match_components
        rng = np.random.default_rng(42)
        A = rng.standard_normal((30, 5))
        B = rng.standard_normal((30, 3))
        corr, matching = match_components(A, B)
        assert matching.shape == (3, 2)  # min(5, 3) = 3 pairs

    def test_random_matrices_low_correlation(self):
        from neurojax.preprocessing.ica_comparison import match_components
        rng = np.random.default_rng(42)
        A = rng.standard_normal((100, 5))
        B = rng.standard_normal((100, 5))
        corr, _ = match_components(A, B)
        assert corr < 0.5  # random matrices shouldn't match well


class TestDimensionalityEstimation:
    def test_neurojax_estimate_returns_positive(self, synthetic_meg_30ch):
        from neurojax.preprocessing.ica_comparison import estimate_dim_neurojax
        raw, _, _ = synthetic_meg_30ch
        data = raw.get_data()
        dim = estimate_dim_neurojax(data)
        assert dim.n_components > 0
        assert dim.n_components <= data.shape[0]
        assert dim.method == "neurojax_laplace"

    def test_neurojax_finds_low_rank_data(self):
        """Data with 3 true sources should estimate ~3 dimensions."""
        from neurojax.preprocessing.ica_comparison import estimate_dim_neurojax
        rng = np.random.default_rng(42)
        # 3 sources, 30 channels, high SNR
        sources = rng.standard_normal((3, 5000))
        mixing = rng.standard_normal((30, 3))
        data = mixing @ sources + rng.standard_normal((30, 5000)) * 0.01
        dim = estimate_dim_neurojax(data)
        assert 2 <= dim.n_components <= 6, \
            f"Expected ~3 components, got {dim.n_components}"


class TestMNEICA:
    def test_mne_ica_runs(self, synthetic_meg_30ch):
        from neurojax.preprocessing.ica_comparison import run_mne_ica
        raw, _, _ = synthetic_meg_30ch
        ica, info = run_mne_ica(raw, n_components=5)
        assert info["n_components"] == 5
        assert isinstance(info["ecg_indices"], list)

    def test_mne_ica_auto_components(self, synthetic_meg_30ch):
        from neurojax.preprocessing.ica_comparison import run_mne_ica
        raw, _, _ = synthetic_meg_30ch
        ica, info = run_mne_ica(raw, n_components=None)
        assert info["n_components"] > 0
        assert info["n_components"] <= 30
