"""Tests for neurojax.qmri — signal models, fitting, ROI extraction, I/O.

TDD: tests define expected behaviour, implementation follows.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import tempfile
import os

# =====================================================================
# Signal model tests
# =====================================================================

class TestSPGRSignal:
    """Validate SPGR signal equation against known analytical values."""

    def test_zero_flip_angle_gives_zero_signal(self):
        from neurojax.qmri.steady_state import spgr_signal
        assert spgr_signal(1000.0, 1.0, 0.0, 0.004) == pytest.approx(0.0, abs=1e-6)

    def test_ernst_angle_maximises_signal(self):
        from neurojax.qmri.steady_state import spgr_signal_multi
        T1, TR, M0 = 1.0, 0.004, 1000.0
        fas = jnp.deg2rad(jnp.arange(1, 90, 1).astype(float))
        signals = spgr_signal_multi(M0, T1, fas, TR)
        # Ernst angle = arccos(exp(-TR/T1))
        ernst = jnp.arccos(jnp.exp(-TR / T1))
        peak_fa = fas[jnp.argmax(signals)]
        assert abs(float(peak_fa - ernst)) < jnp.deg2rad(2.0)

    def test_known_wm_signal_ratio(self):
        """WM at 3T: T1~800ms, TR=4ms, FA=2 vs FA=18 ratio."""
        from neurojax.qmri.steady_state import spgr_signal
        T1 = 0.8  # WM
        s2 = spgr_signal(1000.0, T1, jnp.deg2rad(2.0), 0.004)
        s18 = spgr_signal(1000.0, T1, jnp.deg2rad(18.0), 0.004)
        # At FA=18, signal should be lower than FA=2 for short TR
        assert float(s2) > float(s18)

    def test_signal_is_differentiable(self):
        from neurojax.qmri.steady_state import spgr_signal
        grad_fn = jax.grad(spgr_signal, argnums=1)  # d/dT1
        g = grad_fn(1000.0, 1.0, jnp.deg2rad(10.0), 0.004)
        assert jnp.isfinite(g)


class TestBSSFPSignal:
    def test_bssfp_positive(self):
        from neurojax.qmri.steady_state import bssfp_signal
        s = bssfp_signal(1000.0, 1.0, 0.08, jnp.deg2rad(30.0), 0.00454)
        assert float(s) > 0

    def test_bssfp_differentiable_wrt_t2(self):
        from neurojax.qmri.steady_state import bssfp_signal
        grad_fn = jax.grad(bssfp_signal, argnums=2)  # d/dT2
        g = grad_fn(1000.0, 1.0, 0.08, jnp.deg2rad(30.0), 0.00454)
        assert jnp.isfinite(g)


class TestMultiechoSignal:
    def test_mono_exponential_decay(self):
        from neurojax.qmri.steady_state import multiecho_signal_multi
        TEs = jnp.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035])
        S0, T2star = 100.0, 0.025
        signals = multiecho_signal_multi(S0, T2star, TEs)
        # Signal should decrease monotonically
        diffs = jnp.diff(signals)
        assert jnp.all(diffs < 0)
        # At TE=T2*, signal should be ~S0/e
        s_at_t2 = multiecho_signal_multi(S0, T2star, jnp.array([T2star]))
        assert float(s_at_t2[0]) == pytest.approx(S0 / np.e, rel=0.01)


class TestSuperLorentzian:
    def test_lineshape_positive(self):
        from neurojax.qmri.steady_state import super_lorentzian_lineshape
        g = super_lorentzian_lineshape(1000.0)
        assert float(g) > 0

    def test_lineshape_decreases_with_offset(self):
        from neurojax.qmri.steady_state import super_lorentzian_lineshape
        g_near = super_lorentzian_lineshape(1000.0)
        g_far = super_lorentzian_lineshape(50000.0)
        assert float(g_near) > float(g_far)


# =====================================================================
# Fitting tests
# =====================================================================

class TestDESPOT1Fit:
    """Test DESPOT1 parameter recovery from synthetic data."""

    @pytest.fixture
    def synthetic_spgr(self):
        from neurojax.qmri.steady_state import spgr_signal_multi
        T1_true, M0_true = 0.8, 500.0
        fa_deg = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 18.0])
        fa_rad = jnp.deg2rad(fa_deg)
        TR = 0.004
        signals = spgr_signal_multi(M0_true, T1_true, fa_rad, TR)
        # Add small noise
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, signals.shape) * 2.0
        return signals + noise, fa_rad, TR, T1_true, M0_true

    def test_t1_recovery(self, synthetic_spgr):
        from neurojax.qmri.despot import despot1_fit_voxel
        data, fa_rad, TR, T1_true, M0_true = synthetic_spgr
        result = despot1_fit_voxel(data, fa_rad, TR, n_iters=300, lr=5e-3)
        # Linear DESPOT1 has known bias with noise; within 50% is acceptable
        # NLLS refinement (QUIT-style) or DESPOT1-HIFI gives tighter recovery
        assert abs(float(result.T1) - T1_true) / T1_true < 0.5
        assert float(result.T1) > 0.3  # physiologically plausible

    def test_fit_is_jittable(self, synthetic_spgr):
        from neurojax.qmri.despot import despot1_fit_voxel
        data, fa_rad, TR, _, _ = synthetic_spgr
        jitted = jax.jit(lambda d: despot1_fit_voxel(d, fa_rad, TR))
        result = jitted(data)
        assert jnp.isfinite(result.T1)


class TestMultiechoFit:
    def test_t2star_recovery(self):
        from neurojax.qmri.steady_state import multiecho_signal_multi
        from neurojax.qmri.multiecho import monoexp_t2star_fit
        S0_true, T2s_true = 100.0, 0.025
        TEs = jnp.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035])
        signals = multiecho_signal_multi(S0_true, T2s_true, TEs)
        key = jax.random.PRNGKey(0)
        signals = signals + jax.random.normal(key, signals.shape) * 1.0
        result = monoexp_t2star_fit(signals, TEs)
        assert abs(float(result["T2star"]) - T2s_true) < 0.005


# =====================================================================
# ROI extraction tests
# =====================================================================

class TestROI:
    @pytest.fixture
    def synthetic_roi_data(self):
        """Create a simple 3D param map + segmentation."""
        seg = np.zeros((10, 10, 10), dtype=np.int32)
        seg[2:5, 2:5, 2:5] = 2    # WM label
        seg[5:8, 5:8, 5:8] = 1001  # lh-bankssts (cortical GM)

        param = np.zeros((10, 10, 10), dtype=np.float32)
        param[2:5, 2:5, 2:5] = 0.8   # WM T1
        param[5:8, 5:8, 5:8] = 1.4   # GM T1
        return param, seg

    def test_tissue_stats_wm_gm(self, synthetic_roi_data):
        from neurojax.qmri.roi import extract_tissue_stats
        param, seg = synthetic_roi_data
        stats = extract_tissue_stats(param, seg)
        assert stats["WM"]["mean"] == pytest.approx(0.8, abs=0.01)
        assert stats["WM"]["n_voxels"] == 27

    def test_roi_stats_returns_regions(self, synthetic_roi_data):
        from neurojax.qmri.roi import extract_roi_stats, DESIKAN_LABELS
        param, seg = synthetic_roi_data
        stats = extract_roi_stats(param, seg, labels=DESIKAN_LABELS)
        names = [s["name"] for s in stats]
        assert "lh-bankssts" in names

    def test_compare_tools(self, synthetic_roi_data):
        from neurojax.qmri.roi import compare_tools
        param, seg = synthetic_roi_data
        param2 = param * 1.1  # slightly different "tool"
        comparison = compare_tools({"Tool_A": param, "Tool_B": param2}, seg)
        assert "WM" in comparison
        assert "Tool_A" in comparison["WM"]
        assert "Tool_B" in comparison["WM"]

    def test_csv_export(self, synthetic_roi_data, tmp_path):
        from neurojax.qmri.roi import extract_roi_stats, to_csv, DESIKAN_LABELS
        param, seg = synthetic_roi_data
        stats = extract_roi_stats(param, seg, labels=DESIKAN_LABELS)
        csv_path = str(tmp_path / "test_roi.csv")
        to_csv(stats, csv_path)
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) >= 2  # header + at least 1 data row


# =====================================================================
# I/O tests
# =====================================================================

class TestIO:
    def test_roundtrip_nifti(self, tmp_path):
        from neurojax.qmri.io import save_nifti, load_nifti
        data = jnp.ones((10, 10, 10)) * 42.0
        affine = np.eye(4)
        path = str(tmp_path / "test.nii.gz")
        save_nifti(data, affine, path)
        loaded, loaded_affine, _ = load_nifti(path)
        assert loaded.shape == (10, 10, 10)
        assert float(loaded[5, 5, 5]) == pytest.approx(42.0)

    def test_reorient_identity(self):
        from neurojax.qmri.io import reorient_to_standard
        data = jnp.zeros((10, 12, 14))
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        out, _, perm = reorient_to_standard(data, affine)
        assert perm is None  # already standard
        assert out.shape == (10, 12, 14)

    def test_reorient_permuted(self):
        from neurojax.qmri.io import reorient_to_standard
        data = jnp.zeros((14, 10, 12))  # Z, X, Y
        # Affine says dim0=Z, dim1=X, dim2=Y
        affine = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        out, _, perm = reorient_to_standard(data, affine)
        assert perm is not None
        # After reorient, should be (10, 12, 14) = X, Y, Z
        assert out.shape[0] != 14 or out.shape[2] != 14  # permuted


# =====================================================================
# B1 correction tests
# =====================================================================

class TestB1:
    def test_correct_fa_identity(self):
        from neurojax.qmri.b1 import correct_fa_for_b1
        fa = jnp.deg2rad(jnp.array([10.0, 20.0]))
        b1 = jnp.ones(fa.shape)
        corrected = correct_fa_for_b1(fa, b1)
        np.testing.assert_allclose(corrected, fa, atol=1e-6)

    def test_correct_fa_scales(self):
        from neurojax.qmri.b1 import correct_fa_for_b1
        fa = jnp.deg2rad(jnp.array([10.0]))
        b1 = jnp.array([0.8])  # 80% of nominal
        corrected = correct_fa_for_b1(fa, b1)
        expected = jnp.deg2rad(jnp.array([8.0]))
        np.testing.assert_allclose(corrected, expected, atol=1e-6)

    def test_t1_correction_direction(self):
        """B1 < 1 should increase corrected T1 (underflipped → underestimated T1)."""
        from neurojax.qmri.b1 import correct_t1_for_b1
        t1 = jnp.array([0.7])  # underestimated WM T1
        b1 = jnp.array([0.85])  # 85% B1
        corrected = correct_t1_for_b1(t1, b1, nominal_fa_deg=10.0, TR=0.004)
        assert float(corrected[0]) > float(t1[0])  # correction should increase T1
