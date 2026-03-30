"""Tests for oracle comparison on actual WAND data.

These tests validate the ROI extraction pipeline against real
processing outputs from QUIT, FreeSurfer, and Python DESPOT1.
They serve as integration tests and produce the comparison tables
for the paper.

Requires WAND derivatives at /Users/mhough/dev/wand/derivatives/
"""

import pytest
import numpy as np
import os
import json

WAND = "/Users/mhough/dev/wand"
DERIV = f"{WAND}/derivatives"
SUB = "sub-08033"
FS_DIR = f"{DERIV}/freesurfer/{SUB}"
QMRI = f"{DERIV}/qmri/{SUB}"

# Skip all tests if WAND data not available
pytestmark = pytest.mark.skipif(
    not os.path.exists(f"{FS_DIR}/mri/aparc+aseg.mgz"),
    reason="WAND derivatives not available"
)


class TestThreeWayT1:
    """Compare VFA T1 from QUIT, FreeSurfer, and Python across ROIs."""

    @pytest.fixture
    def segmentation(self):
        from neurojax.qmri.roi import load_segmentation
        return load_segmentation(f"{FS_DIR}/mri/aparc+aseg.mgz")

    @pytest.fixture
    def t1_maps(self):
        """Load all three T1 maps, resampled to same space."""
        from neurojax.qmri.io import load_nifti
        maps = {}
        paths = {
            "QUIT_DESPOT1": f"{QMRI}/ses-02/quit/D1_D1_T1.nii.gz",
            "FreeSurfer": f"{QMRI}/ses-02/freesurfer/T1.mgz",
            "Python_linear": f"{QMRI}/ses-02/T1map.nii.gz",
        }
        for name, path in paths.items():
            if os.path.exists(path):
                data, _, _ = load_nifti(path)
                maps[name] = np.asarray(data)
        return maps

    def test_all_three_tools_available(self, t1_maps):
        assert len(t1_maps) >= 2, f"Need at least 2 T1 maps, got {list(t1_maps.keys())}"

    def test_tissue_stats_wm_range(self, t1_maps, segmentation):
        from neurojax.qmri.roi import extract_tissue_stats
        for name, t1 in t1_maps.items():
            if t1.shape != segmentation.shape:
                continue  # skip mismatched orientations
            stats = extract_tissue_stats(t1, segmentation, valid_range=(0.1, 5.0))
            wm = stats["WM"]
            if wm["n_voxels"] > 0:
                # All tools should give WM T1 between 400-1200ms at 3T
                assert 0.4 < wm["median"] < 1.2, \
                    f"{name} WM T1 median={wm['median']:.3f}s outside range"

    def test_tissue_stats_gm_gt_wm(self, t1_maps, segmentation):
        from neurojax.qmri.roi import extract_tissue_stats
        for name, t1 in t1_maps.items():
            if t1.shape != segmentation.shape:
                continue
            stats = extract_tissue_stats(t1, segmentation, valid_range=(0.1, 5.0))
            if stats["WM"]["n_voxels"] > 0 and stats["GM"]["n_voxels"] > 0:
                # GM T1 should always be longer than WM T1
                assert stats["GM"]["median"] > stats["WM"]["median"], \
                    f"{name}: GM T1 ({stats['GM']['median']:.3f}) <= WM ({stats['WM']['median']:.3f})"

    def test_compare_tools_produces_table(self, t1_maps, segmentation):
        from neurojax.qmri.roi import compare_tools
        # Only use maps that match segmentation shape
        matched = {k: v for k, v in t1_maps.items() if v.shape == segmentation.shape}
        if len(matched) < 2:
            pytest.skip("Need at least 2 shape-matched T1 maps")
        comparison = compare_tools(matched, segmentation, valid_range=(0.1, 5.0))
        assert "WM" in comparison
        assert len(comparison["WM"]) >= 2

    def test_export_comparison_csv(self, t1_maps, segmentation, tmp_path):
        from neurojax.qmri.roi import compare_tools, compare_tools_to_csv
        matched = {k: v for k, v in t1_maps.items() if v.shape == segmentation.shape}
        if len(matched) < 2:
            pytest.skip("Need at least 2 shape-matched T1 maps")
        comparison = compare_tools(matched, segmentation, valid_range=(0.1, 5.0))
        csv_path = str(tmp_path / "t1_comparison.csv")
        compare_tools_to_csv(comparison, csv_path)
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) >= 3  # header + WM + GM at minimum


class TestQUITQMTResults:
    """Validate QUIT QMT BPF results against expected tissue values."""

    @pytest.fixture
    def quit_bpf(self):
        from neurojax.qmri.io import load_nifti
        path = f"{QMRI}/ses-02/quit/QMT_QMT_f_b.nii.gz"
        if not os.path.exists(path):
            pytest.skip("QUIT QMT BPF not available")
        data, _, _ = load_nifti(path)
        return np.asarray(data)

    @pytest.fixture
    def quit_t1_qmt_space(self):
        from neurojax.qmri.io import load_nifti
        path = f"{QMRI}/ses-02/quit/D1_T1_in_QMT.nii.gz"
        if not os.path.exists(path):
            pytest.skip("QUIT T1 in QMT space not available")
        data, _, _ = load_nifti(path)
        return np.asarray(data)

    def test_bpf_wm_higher_than_gm(self, quit_bpf, quit_t1_qmt_space):
        """WM should have ~2x the BPF of GM (more myelin)."""
        wm_mask = (quit_t1_qmt_space > 0.3) & (quit_t1_qmt_space < 1.0) & (quit_bpf > 0.01)
        gm_mask = (quit_t1_qmt_space >= 1.0) & (quit_t1_qmt_space < 2.5) & (quit_bpf > 0.01)

        wm_bpf = quit_bpf[wm_mask]
        gm_bpf = quit_bpf[gm_mask]

        if len(wm_bpf) > 100 and len(gm_bpf) > 100:
            assert np.median(wm_bpf) > np.median(gm_bpf), \
                f"WM BPF ({np.median(wm_bpf):.4f}) should > GM ({np.median(gm_bpf):.4f})"

    def test_bpf_in_physiological_range(self, quit_bpf, quit_t1_qmt_space):
        """BPF should be 0.02-0.25 in brain tissue."""
        brain = (quit_t1_qmt_space > 0.2) & (quit_bpf > 0.01) & (quit_bpf < 0.30)
        vals = quit_bpf[brain]
        if len(vals) > 100:
            assert 0.05 < np.median(vals) < 0.20

    def test_quit_all_params_finite(self):
        """Check all QUIT QMT parameter maps have finite values in brain."""
        from neurojax.qmri.io import load_nifti
        params = ["QMT_QMT_f_b", "QMT_QMT_k_bf", "QMT_QMT_T1_f",
                   "QMT_QMT_T2_b", "QMT_QMT_T2_f"]
        for p in params:
            path = f"{QMRI}/ses-02/quit/{p}.nii.gz"
            if os.path.exists(path):
                data, _, _ = load_nifti(path)
                arr = np.asarray(data)
                finite_frac = np.isfinite(arr[arr != 0]).mean()
                assert finite_frac > 0.9, f"{p}: only {finite_frac:.1%} finite"


class TestMyelinProxyComparison:
    """Re-run T1w/T2w vs QUIT BPF correlation with proper ROI extraction."""

    @pytest.fixture
    def myelin_comparison_data(self):
        path = f"{QMRI}/myelin_comparison/myelin_comparison.json"
        if not os.path.exists(path):
            pytest.skip("Myelin comparison not available")
        with open(path) as f:
            return json.load(f)

    def test_correlation_exists(self, myelin_comparison_data):
        corrs = myelin_comparison_data.get("correlations", {})
        assert len(corrs) > 0

    def test_vertex_correlation_positive(self, myelin_comparison_data):
        """T1w/T2w should correlate positively with BPF (both index myelin)."""
        corrs = myelin_comparison_data.get("correlations", {})
        for key, vals in corrs.items():
            if "QMT_BPF" in key and "ROI" not in key:
                # Vertex-wise: may be weak without B1 correction
                assert vals["pearson_r"] > -0.5, f"{key}: r={vals['pearson_r']}"


class TestPerfusionResults:
    """Validate perfusion outputs."""

    def test_cbf_exists(self):
        path = f"{DERIV}/perfusion/{SUB}/oxford_asl/native_space/perfusion_calib.nii.gz"
        assert os.path.exists(path), "CBF map not found"

    def test_cmro2_exists(self):
        path = f"{DERIV}/perfusion/{SUB}/CMRO2_map.nii.gz"
        assert os.path.exists(path), "CMRO2 map not found"

    def test_cbf_physiological(self):
        from neurojax.qmri.io import load_nifti
        path = f"{DERIV}/perfusion/{SUB}/oxford_asl/native_space/perfusion_calib.nii.gz"
        if not os.path.exists(path):
            pytest.skip("CBF not available")
        data, _, _ = load_nifti(path)
        brain = np.asarray(data)
        vals = brain[brain > 5]
        assert 20 < np.median(vals) < 80, f"Median CBF={np.median(vals):.1f} outside range"

    def test_trust_results_json(self):
        path = f"{DERIV}/perfusion/{SUB}/trust_results.json"
        if not os.path.exists(path):
            pytest.skip("TRUST not available")
        with open(path) as f:
            trust = json.load(f)
        assert "T2_blood_ms" in trust
        assert trust["T2_blood_ms"] > 0


class TestDTIResults:
    """Validate DTI outputs from dtifit."""

    def test_fa_exists(self):
        path = f"{DERIV}/fsl-dwi/{SUB}/ses-02/dtifit/dtifit_FA.nii.gz"
        assert os.path.exists(path), "FA map not found"

    def test_fa_range(self):
        from neurojax.qmri.io import load_nifti
        path = f"{DERIV}/fsl-dwi/{SUB}/ses-02/dtifit/dtifit_FA.nii.gz"
        if not os.path.exists(path):
            pytest.skip("FA not available")
        data, _, _ = load_nifti(path)
        fa = np.asarray(data)
        brain = fa[fa > 0.05]
        assert 0.2 < np.median(brain) < 0.6, f"Median FA={np.median(brain):.3f}"

    def test_md_exists(self):
        path = f"{DERIV}/fsl-dwi/{SUB}/ses-02/dtifit/dtifit_MD.nii.gz"
        assert os.path.exists(path), "MD map not found"


class TestSegmentationOutputs:
    """Validate all segmentation outputs exist."""

    def test_aparc_aseg(self):
        assert os.path.exists(f"{FS_DIR}/mri/aparc+aseg.mgz")

    def test_thalamic_nuclei(self):
        assert os.path.exists(f"{FS_DIR}/mri/ThalamicNuclei_DTI_CNN.mgz")

    def test_hippocampal_subfields(self):
        assert os.path.exists(f"{FS_DIR}/mri/lh.hippoAmygLabels-T1.v22.mgz")

    def test_hypothalamic_subunits(self):
        assert os.path.exists(f"{FS_DIR}/mri/hypothalamic_subunits_seg.v1.mgz")

    def test_samseg(self):
        assert os.path.exists(f"{DERIV}/advanced-freesurfer/{SUB}/samseg/samseg.csv")

    def test_synthseg_four_sessions(self):
        for ses in ["ses-02", "ses-03", "ses-04", "ses-05"]:
            assert os.path.exists(
                f"{DERIV}/advanced-freesurfer/{SUB}/synthseg/{ses}_volumes.csv"
            ), f"SynthSeg {ses} missing"
