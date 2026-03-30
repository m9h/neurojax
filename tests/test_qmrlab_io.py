"""TDD tests for qMRLab Octave I/O — diagnosing and fixing save failures.

RED: These tests define what working qMRLab I/O looks like.
Each test isolates one aspect of the Octave ↔ NIfTI pipeline.
"""

import pytest
import numpy as np
import subprocess
import os
import tempfile
import json


OCTAVE = "octave-cli"
QMRLAB_PATH = os.path.expanduser("~/dev/qMRLab")
WAND = "/Users/mhough/dev/wand"
SUB = "sub-08033"
QMRI = f"{WAND}/derivatives/qmri/{SUB}/ses-02"

pytestmark = pytest.mark.skipif(
    not os.path.exists(QMRLAB_PATH),
    reason="qMRLab not installed"
)


def run_octave(script, timeout=60):
    """Run Octave script and return (stdout, stderr, returncode)."""
    result = subprocess.run(
        [OCTAVE, "--eval", f"addpath(genpath('{QMRLAB_PATH}')); {script}"],
        capture_output=True, text=True, timeout=timeout
    )
    return result.stdout, result.stderr, result.returncode


class TestOctaveLoadNii:
    """Can Octave load WAND NIfTI files with load_nii?"""

    def test_load_nii_available(self):
        out, err, rc = run_octave("which load_nii; disp('ok')")
        assert "ok" in out, f"load_nii not found: {err}"

    def test_load_qmt_mtoff(self):
        nii = f"{WAND}/{SUB}/ses-02/anat/{SUB}_ses-02_mt-off_part-mag_QMT.nii.gz"
        out, err, rc = run_octave(
            f"nii = load_nii('{nii}'); disp(size(nii.img)); disp('ok')"
        )
        assert "ok" in out, f"Failed to load QMT: {err[-200:]}"

    def test_load_quit_t1(self):
        nii = f"{QMRI}/quit/D1_T1_in_QMT.nii.gz"
        if not os.path.exists(nii):
            pytest.skip("QUIT T1 not available")
        out, err, rc = run_octave(
            f"nii = load_nii('{nii}'); disp(size(nii.img)); disp('ok')"
        )
        assert "ok" in out, f"Failed to load T1: {err[-200:]}"


class TestOctaveSaveNii:
    """Can Octave save NIfTI files with save_nii?"""

    def test_save_nii_roundtrip(self, tmp_path):
        out_file = str(tmp_path / "test_save.nii.gz")
        out, err, rc = run_octave(f"""
            nii = struct();
            nii.img = single(randn(10, 10, 10));
            nii.hdr.dime.datatype = 16;
            nii.hdr.dime.bitpix = 32;
            nii.hdr.dime.dim = [3, 10, 10, 10, 1, 1, 1, 1];
            nii.hdr.dime.pixdim = [1, 1, 1, 1, 0, 0, 0, 0];
            nii.hdr.hist.srow_x = [1, 0, 0, 0];
            nii.hdr.hist.srow_y = [0, 1, 0, 0];
            nii.hdr.hist.srow_z = [0, 0, 1, 0];
            nii.hdr.hist.sform_code = 1;
            save_nii(nii, '{out_file}');
            disp('saved');
        """)
        assert "saved" in out, f"save_nii failed: {err[-200:]}"
        assert os.path.exists(out_file), "Output file not created"

    def test_save_nii_from_reference(self, tmp_path):
        """Save using a reference NIfTI header (the actual use case)."""
        ref = f"{WAND}/{SUB}/ses-02/anat/{SUB}_ses-02_mt-off_part-mag_QMT.nii.gz"
        out_file = str(tmp_path / "test_ref_save.nii.gz")
        out, err, rc = run_octave(f"""
            ref = load_nii('{ref}');
            out = ref;
            out.img = single(randn(size(ref.img)));
            out.hdr.dime.datatype = 16;
            out.hdr.dime.bitpix = 32;
            save_nii(out, '{out_file}');
            disp('saved');
        """)
        assert "saved" in out, f"save from ref failed: {err[-200:]}"
        assert os.path.exists(out_file)

    def test_save_nii_from_quit_reference(self, tmp_path):
        """Save using QUIT T1 as reference (different orientation from QMT)."""
        ref = f"{QMRI}/quit/D1_T1_in_QMT.nii.gz"
        if not os.path.exists(ref):
            pytest.skip("QUIT T1 not available")
        out_file = str(tmp_path / "test_quit_ref.nii.gz")
        out, err, rc = run_octave(f"""
            ref = load_nii('{ref}');
            out = ref;
            out.img = single(randn(size(ref.img)));
            out.hdr.dime.datatype = 16;
            out.hdr.dime.bitpix = 32;
            save_nii(out, '{out_file}');
            disp('saved');
        """)
        assert "saved" in out, f"save from QUIT ref failed: {err[-200:]}"


class TestOctaveMatSave:
    """Fallback: save results as .mat when NIfTI fails."""

    def test_mat_save_struct(self, tmp_path):
        out_file = str(tmp_path / "test_results.mat")
        out, err, rc = run_octave(f"""
            results.F = single(randn(10, 10, 10));
            results.kf = single(randn(10, 10, 10));
            results.T1f = single(randn(10, 10, 10));
            save('{out_file}', '-struct', 'results');
            disp('saved');
        """)
        assert "saved" in out, f"mat save failed: {err[-200:]}"
        assert os.path.exists(out_file)

    def test_mat_load_in_python(self, tmp_path):
        """Verify Python can read the .mat saved by Octave."""
        import scipy.io
        out_file = str(tmp_path / "test_py_read.mat")
        run_octave(f"""
            results.BPF = single(rand(5, 5, 5) * 0.2);
            results.T1 = single(rand(5, 5, 5) * 2.0);
            save('{out_file}', '-struct', 'results');
        """)
        data = scipy.io.loadmat(out_file)
        assert "BPF" in data
        assert data["BPF"].shape == (5, 5, 5)
        assert data["BPF"].max() <= 0.2


class TestParallelTempDir:
    """Verify parallel qMRLab runs don't conflict on temp files."""

    def test_separate_workdirs(self, tmp_path):
        """Two Octave processes in separate dirs don't conflict."""
        dir1 = tmp_path / "slab0"
        dir2 = tmp_path / "slab1"
        dir1.mkdir()
        dir2.mkdir()

        script = """
            cd('{workdir}');
            data = randn(5, 5);
            save('FitTempResults.mat', 'data');
            pause(1);
            assert(exist('FitTempResults.mat', 'file') == 2);
            disp('ok');
        """
        p1 = subprocess.Popen(
            [OCTAVE, "--eval", script.format(workdir=str(dir1))],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        p2 = subprocess.Popen(
            [OCTAVE, "--eval", script.format(workdir=str(dir2))],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out1, _ = p1.communicate(timeout=30)
        out2, _ = p2.communicate(timeout=30)

        assert "ok" in out1
        assert "ok" in out2
        assert (dir1 / "FitTempResults.mat").exists()
        assert (dir2 / "FitTempResults.mat").exists()


class TestQMRLabFitSmallVolume:
    """Run qMRLab qmt_spgr on a tiny volume to verify the full pipeline."""

    def test_fit_single_slice(self, tmp_path):
        """Fit just 1 slice to verify qMRLab works end-to-end in Octave."""
        out_file = str(tmp_path / "FitResults.mat")
        script = f"""
            addpath(genpath('{QMRLAB_PATH}'));
            pkg load struct; pkg load io; pkg load statistics; pkg load optim; pkg load image;

            % Load one slice of real data
            anat_dir = '{WAND}/{SUB}/ses-02/anat';
            mtoff = load_nii(fullfile(anat_dir, '{SUB}_ses-02_mt-off_part-mag_QMT.nii.gz'));
            mt_off = double(mtoff.img(:,:,64));

            tags = {{'flip-1_mt-1','flip-1_mt-2','flip-2_mt-1','flip-2_mt-2','flip-2_mt-3','flip-2_mt-4','flip-2_mt-5','flip-2_mt-6','flip-3_mt-1'}};
            mt_data = zeros([size(mt_off), 9]);
            for i = 1:9
                tmp = load_nii(fullfile(anat_dir, ['{SUB}_ses-02_' tags{{i}} '_part-mag_QMT.nii.gz']));
                mt_data(:,:,i) = double(tmp.img(:,:,64));
            end

            t1_nii = load_nii('{QMRI}/quit/D1_T1_in_QMT.nii.gz');
            t1_slice = double(t1_nii.img(:,:,64));

            mask = (t1_slice > 0.3) & (t1_slice < 3.0) & (mt_off > 100);
            fprintf('Mask voxels: %d\\n', sum(mask(:)));

            Model = qmt_spgr;
            Model.Prot.MTdata.Mat = [332 56360; 332 1000; 628 47180; 628 12060; 628 2750; 628 2770; 628 2790; 628 2890; 333 1000];
            Model.Prot.TimingTable.Mat = [5, 0.055];

            % Reshape for 2D slice
            mt_3d = reshape(mt_data, [size(mt_off, 1), size(mt_off, 2), 1, 9]);
            data.MTdata = mt_3d;
            data.R1map = reshape(1.0 ./ max(t1_slice, 0.01), [size(mt_off, 1), size(mt_off, 2), 1]);
            data.Mask = reshape(mask, [size(mt_off, 1), size(mt_off, 2), 1]);

            cd('{str(tmp_path)}');
            FitResults = FitData(data, Model, 0);
            save('{out_file}', '-struct', 'FitResults');
            fprintf('Saved to %s\\n', '{out_file}');

            if isfield(FitResults, 'F')
                F = FitResults.F;
                valid = F(mask & isfinite(F) & F > 0 & F < 0.3);
                fprintf('BPF median: %.4f, n=%d\\n', median(valid), length(valid));
            end
            disp('DONE');
        """
        result = subprocess.run(
            [OCTAVE, "--eval", script],
            capture_output=True, text=True, timeout=600
        )
        assert "DONE" in result.stdout, f"qMRLab fit failed:\n{result.stderr[-500:]}"
        assert os.path.exists(out_file), "FitResults.mat not saved"

        # Verify in Python
        import scipy.io
        data = scipy.io.loadmat(out_file)
        assert "F" in data or "BPF" in data or len(data) > 3, \
            f"FitResults.mat has unexpected keys: {list(data.keys())}"
