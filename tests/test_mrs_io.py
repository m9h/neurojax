"""TDD tests for native Siemens TWIX reader (no spec2nii dependency).

Reads Siemens TWIX .dat files directly using mapVBVD, returning a
standardized MRSData object with FID data and metadata.

Test data: Big GABA S5 MEGA-PRESS TWIX file.
Tests are skipped if the file is not available.

References:
    Mikkelsen et al. (2017) Big GABA: Edited MR spectroscopy at 24
    research sites. NeuroImage 159:32-45.
"""
import numpy as np
import pytest

from neurojax.analysis.mrs_io import read_twix, MRSData


# ---------------------------------------------------------------------------
# Test data path
# ---------------------------------------------------------------------------

TWIX_PATH = "/data/datasets/big_gaba/S5/S5_MP/S01/S01_GABA_68.dat"

import os
DATA_AVAILABLE = os.path.isfile(TWIX_PATH)
skip_no_data = pytest.mark.skipif(
    not DATA_AVAILABLE, reason=f"TWIX file not found: {TWIX_PATH}"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReadTwixReturnsMRSData:
    """read_twix returns an MRSData with required attributes."""

    @skip_no_data
    def test_read_twix_returns_mrsdata(self):
        mrd = read_twix(TWIX_PATH)
        assert isinstance(mrd, MRSData)
        assert hasattr(mrd, 'data')
        assert hasattr(mrd, 'dwell_time')
        assert hasattr(mrd, 'centre_freq')
        assert hasattr(mrd, 'n_coils')
        assert hasattr(mrd, 'n_averages')
        # Dwell time is positive float
        assert isinstance(mrd.dwell_time, float)
        assert mrd.dwell_time > 0
        # Centre freq is positive (Hz)
        assert isinstance(mrd.centre_freq, float)
        assert mrd.centre_freq > 0
        # Coils and averages are positive ints
        assert isinstance(mrd.n_coils, int)
        assert mrd.n_coils >= 1
        assert isinstance(mrd.n_averages, int)
        assert mrd.n_averages >= 1


class TestReadTwixShape:
    """FID shape follows (n_spec, n_coils, n_dyn) or
    (n_spec, n_coils, n_edit, n_dyn) for MEGA-PRESS data."""

    @skip_no_data
    def test_read_twix_shape(self):
        mrd = read_twix(TWIX_PATH)
        fid = mrd.data
        assert fid.ndim >= 3, f"Expected at least 3 dims, got {fid.ndim}"
        n_spec = fid.shape[0]
        assert n_spec > 0, "n_spec must be positive"
        # Second dimension should match n_coils
        assert fid.shape[1] == mrd.n_coils, (
            f"Axis 1 ({fid.shape[1]}) should equal n_coils ({mrd.n_coils})"
        )
        # For MEGA-PRESS, expect 4D: (n_spec, n_coils, n_edit, n_dyn)
        # or 3D: (n_spec, n_coils, n_dyn)
        if fid.ndim == 4:
            n_edit = fid.shape[2]
            assert n_edit in (2, 4), (
                f"Edit dim should be 2 (MEGA) or 4 (HERMES), got {n_edit}"
            )


class TestReadTwixMetadata:
    """Extracts TE, TR, and field strength from TWIX header."""

    @skip_no_data
    def test_read_twix_metadata(self):
        mrd = read_twix(TWIX_PATH)
        # TE in ms (MEGA-PRESS typically 68 ms)
        assert isinstance(mrd.te, float)
        assert 10 < mrd.te < 500, f"TE={mrd.te} ms out of range"
        # TR in ms
        assert isinstance(mrd.tr, float)
        assert 500 < mrd.tr < 10000, f"TR={mrd.tr} ms out of range"
        # Field strength in Tesla (3T expected for Big GABA)
        assert isinstance(mrd.field_strength, float)
        assert 1.0 < mrd.field_strength < 12.0, (
            f"B0={mrd.field_strength} T out of range"
        )


class TestReadTwixWaterRef:
    """Loads water reference if available in the TWIX file."""

    @skip_no_data
    def test_read_twix_water_ref(self):
        mrd = read_twix(TWIX_PATH, load_water_ref=True)
        # Water ref may or may not be in the file; if present,
        # it should be an ndarray
        if mrd.water_ref is not None:
            assert isinstance(mrd.water_ref, np.ndarray)
            assert np.iscomplexobj(mrd.water_ref)
            assert mrd.water_ref.ndim >= 1


class TestMRSDataToNumpy:
    """MRSData.data returns a complex numpy array."""

    @skip_no_data
    def test_mrsdata_to_numpy(self):
        mrd = read_twix(TWIX_PATH)
        arr = mrd.data
        assert isinstance(arr, np.ndarray)
        assert np.iscomplexobj(arr), "FID data should be complex"


class TestReadTwixNonexistent:
    """read_twix raises FileNotFoundError for missing file."""

    def test_read_twix_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            read_twix("/nonexistent/path/to/fake.dat")
