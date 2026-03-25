"""Tests for BIDSConnectomeLoader and ConnectomeData."""

import jax.numpy as jnp
import numpy as np
import pytest

from neurojax.io.connectome import BIDSConnectomeLoader, ConnectomeData


# ---------------------------------------------------------------------------
# Fixtures: create a minimal BIDS derivatives tree on disk
# ---------------------------------------------------------------------------

@pytest.fixture
def bids_derivatives(tmp_path):
    """Create a fake BIDS derivatives directory with QSIRecon outputs."""
    deriv = tmp_path / "derivatives"

    # QSIRecon: sub-01, two atlases
    qsi_dwi = deriv / "qsirecon" / "sub-01" / "dwi"
    qsi_dwi.mkdir(parents=True)

    n = 10  # 10 regions for Schaefer200 (toy)
    rng = np.random.default_rng(42)
    weights = rng.random((n, n))
    weights = (weights + weights.T) / 2  # symmetric
    np.fill_diagonal(weights, 0)

    lengths = rng.random((n, n)) * 100  # mm
    lengths = (lengths + lengths.T) / 2
    np.fill_diagonal(lengths, 0)

    np.savetxt(
        qsi_dwi / "sub-01_space-T1w_atlas-Schaefer200_desc-sift2_connectivity.csv",
        weights, delimiter=","
    )
    np.savetxt(
        qsi_dwi / "sub-01_space-T1w_atlas-Schaefer200_desc-meanlength_connectivity.csv",
        lengths, delimiter=","
    )

    # Second atlas: DKT (different size)
    n2 = 8
    weights2 = rng.random((n2, n2))
    weights2 = (weights2 + weights2.T) / 2
    np.fill_diagonal(weights2, 0)
    np.savetxt(
        qsi_dwi / "sub-01_space-T1w_atlas-DKT_desc-count_connectivity.csv",
        weights2, delimiter=","
    )

    # dmipy-jax: axon delay matrix for Schaefer200
    dmipy_dwi = deriv / "dmipy-jax" / "sub-01" / "dwi"
    dmipy_dwi.mkdir(parents=True)

    delays = lengths / 7.0  # velocity = 7 mm/ms
    np.savetxt(
        dmipy_dwi / "sub-01_space-T1w_atlas-Schaefer200_desc-axondelay_connectivity.csv",
        delays, delimiter=","
    )

    return deriv, weights, lengths, delays


@pytest.fixture
def bids_with_session(tmp_path):
    """Create BIDS derivatives with session structure."""
    deriv = tmp_path / "derivatives"
    qsi_dwi = deriv / "qsirecon" / "sub-02" / "ses-01" / "dwi"
    qsi_dwi.mkdir(parents=True)

    n = 6
    weights = np.eye(n) * 0  # zeros on diagonal
    weights[0, 1] = weights[1, 0] = 0.5
    np.savetxt(
        qsi_dwi / "sub-02_ses-01_space-T1w_atlas-Gordon_desc-sift2_connectivity.csv",
        weights, delimiter=","
    )
    return deriv


# ---------------------------------------------------------------------------
# ConnectomeData tests
# ---------------------------------------------------------------------------

class TestConnectomeData:
    def test_for_adapter_with_delays(self):
        """When delays are provided, for_adapter returns them directly."""
        w = jnp.eye(4)
        d = jnp.ones((4, 4)) * 5.0
        data = ConnectomeData(
            weights=w, delays=d, fiber_lengths=None,
            atlas="test", n_regions=4, region_labels=None, subject="01"
        )
        w_out, d_out = data.for_adapter()
        assert jnp.allclose(w_out, w)
        assert jnp.allclose(d_out, d)

    def test_for_adapter_from_fiber_lengths(self):
        """When delays missing but fiber lengths available, estimate delays."""
        w = jnp.eye(4)
        fl = jnp.ones((4, 4)) * 70.0  # 70 mm
        data = ConnectomeData(
            weights=w, delays=None, fiber_lengths=fl,
            atlas="test", n_regions=4, region_labels=None, subject="01"
        )
        w_out, d_out = data.for_adapter()
        # 70mm / 7mm/ms = 10ms
        assert jnp.allclose(d_out, 10.0)

    def test_for_adapter_no_delays_no_lengths(self):
        """When no delay info available, return zero delays."""
        w = jnp.eye(4)
        data = ConnectomeData(
            weights=w, delays=None, fiber_lengths=None,
            atlas="test", n_regions=4, region_labels=None, subject="01"
        )
        _, d_out = data.for_adapter()
        assert jnp.allclose(d_out, 0.0)
        assert d_out.shape == (4, 4)


# ---------------------------------------------------------------------------
# BIDSConnectomeLoader tests
# ---------------------------------------------------------------------------

class TestAvailableAtlases:
    def test_discovers_atlases(self, bids_derivatives):
        deriv, *_ = bids_derivatives
        loader = BIDSConnectomeLoader(deriv, "01")
        atlases = loader.available_atlases()
        assert "Schaefer200" in atlases
        assert "DKT" in atlases

    def test_sorted(self, bids_derivatives):
        deriv, *_ = bids_derivatives
        loader = BIDSConnectomeLoader(deriv, "01")
        atlases = loader.available_atlases()
        assert atlases == sorted(atlases)

    def test_missing_subject_empty(self, bids_derivatives):
        deriv, *_ = bids_derivatives
        loader = BIDSConnectomeLoader(deriv, "99")
        assert loader.available_atlases() == []


class TestLoadWeights:
    def test_loads_schaefer200(self, bids_derivatives):
        deriv, weights, *_ = bids_derivatives
        loader = BIDSConnectomeLoader(deriv, "01", atlas="Schaefer200")
        data = loader.load()
        assert data.n_regions == 10
        assert data.atlas == "Schaefer200"
        assert data.subject == "01"
        assert data.weights.shape == (10, 10)

    def test_normalized_by_default(self, bids_derivatives):
        deriv, *_ = bids_derivatives
        loader = BIDSConnectomeLoader(deriv, "01", atlas="Schaefer200")
        data = loader.load()
        assert float(jnp.max(data.weights)) == pytest.approx(1.0)

    def test_unnormalized(self, bids_derivatives):
        deriv, weights, *_ = bids_derivatives
        loader = BIDSConnectomeLoader(
            deriv, "01", atlas="Schaefer200", normalize_weights=False
        )
        data = loader.load()
        # Raw values should match original
        assert float(jnp.max(data.weights)) == pytest.approx(
            float(np.max(weights)), rel=1e-5
        )

    def test_loads_dkt(self, bids_derivatives):
        deriv, *_ = bids_derivatives
        loader = BIDSConnectomeLoader(deriv, "01", atlas="DKT")
        data = loader.load()
        assert data.n_regions == 8
        assert data.atlas == "DKT"

    def test_missing_atlas_raises(self, bids_derivatives):
        deriv, *_ = bids_derivatives
        loader = BIDSConnectomeLoader(deriv, "01", atlas="Nonexistent")
        with pytest.raises(FileNotFoundError, match="Nonexistent"):
            loader.load()


class TestLoadFiberLengths:
    def test_loads_lengths(self, bids_derivatives):
        deriv, _, lengths, _ = bids_derivatives
        loader = BIDSConnectomeLoader(deriv, "01", atlas="Schaefer200")
        data = loader.load()
        assert data.fiber_lengths is not None
        assert data.fiber_lengths.shape == (10, 10)

    def test_no_lengths_for_dkt(self, bids_derivatives):
        deriv, *_ = bids_derivatives
        loader = BIDSConnectomeLoader(deriv, "01", atlas="DKT")
        data = loader.load()
        assert data.fiber_lengths is None


class TestLoadDelays:
    def test_loads_dmipy_delays(self, bids_derivatives):
        deriv, *_ = bids_derivatives
        loader = BIDSConnectomeLoader(deriv, "01", atlas="Schaefer200")
        data = loader.load()
        assert data.delays is not None
        assert data.delays.shape == (10, 10)

    def test_no_delays_for_dkt(self, bids_derivatives):
        deriv, *_ = bids_derivatives
        loader = BIDSConnectomeLoader(deriv, "01", atlas="DKT")
        data = loader.load()
        assert data.delays is None


class TestSessions:
    def test_session_path(self, bids_with_session):
        loader = BIDSConnectomeLoader(
            bids_with_session, "02", atlas="Gordon", session="01"
        )
        data = loader.load()
        assert data.n_regions == 6
        assert data.atlas == "Gordon"


class TestLoadAtlasOverride:
    def test_load_with_atlas_override(self, bids_derivatives):
        deriv, *_ = bids_derivatives
        loader = BIDSConnectomeLoader(deriv, "01", atlas="Schaefer200")
        # Override at load time
        data = loader.load(atlas="DKT")
        assert data.atlas == "DKT"
        assert data.n_regions == 8


class TestForAdapterIntegration:
    def test_full_pipeline(self, bids_derivatives):
        """Load and convert to adapter-ready format."""
        deriv, *_ = bids_derivatives
        loader = BIDSConnectomeLoader(deriv, "01", atlas="Schaefer200")
        data = loader.load()
        weights, delays = data.for_adapter()
        assert weights.shape == (10, 10)
        assert delays.shape == (10, 10)
        assert jnp.all(jnp.isfinite(weights))
        assert jnp.all(jnp.isfinite(delays))


class TestLoadGroup:
    def test_loads_multiple_subjects(self, tmp_path):
        """load_group returns data for multiple subjects."""
        deriv = tmp_path / "derivatives"
        n = 5
        rng = np.random.default_rng(0)

        for sub_id in ["01", "02", "03"]:
            qsi_dwi = deriv / "qsirecon" / f"sub-{sub_id}" / "dwi"
            qsi_dwi.mkdir(parents=True)
            w = rng.random((n, n))
            np.savetxt(
                qsi_dwi / f"sub-{sub_id}_atlas-Schaefer200_desc-sift2_connectivity.csv",
                w, delimiter=","
            )

        loader = BIDSConnectomeLoader(deriv, "01", atlas="Schaefer200")
        results = loader.load_group(["01", "02", "03"])
        assert len(results) == 3
        assert all(r.subject in ["01", "02", "03"] for r in results)

    def test_skips_missing_subjects(self, bids_derivatives):
        deriv, *_ = bids_derivatives
        loader = BIDSConnectomeLoader(deriv, "01", atlas="Schaefer200")
        with pytest.warns(UserWarning, match="Skipping sub-99"):
            results = loader.load_group(["01", "99"])
        assert len(results) == 1
        assert results[0].subject == "01"


class TestNpyFormat:
    def test_loads_npy_matrix(self, tmp_path):
        """Supports .npy format for connectivity matrices."""
        deriv = tmp_path / "derivatives"
        qsi_dwi = deriv / "qsirecon" / "sub-01" / "dwi"
        qsi_dwi.mkdir(parents=True)

        w = np.random.default_rng(0).random((5, 5))
        np.save(
            qsi_dwi / "sub-01_atlas-Test_desc-sift2_connectivity.npy", w
        )

        loader = BIDSConnectomeLoader(deriv, "01", atlas="Test")
        data = loader.load()
        assert data.n_regions == 5


class TestTsvFormat:
    def test_loads_tsv_matrix(self, tmp_path):
        """Supports .tsv format for connectivity matrices."""
        deriv = tmp_path / "derivatives"
        qsi_dwi = deriv / "qsirecon" / "sub-01" / "dwi"
        qsi_dwi.mkdir(parents=True)

        w = np.random.default_rng(0).random((5, 5))
        np.savetxt(
            qsi_dwi / "sub-01_atlas-Test_desc-sift2_connectivity.tsv",
            w, delimiter="\t"
        )

        loader = BIDSConnectomeLoader(deriv, "01", atlas="Test")
        data = loader.load()
        assert data.n_regions == 5
