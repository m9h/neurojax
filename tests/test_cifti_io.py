"""Tests for neurojax.io.cifti — CIFTI-2 dense series / scalar I/O."""

import os

import nibabel as nib
import numpy as np
import pytest
from nibabel.cifti2 import cifti2_axes as axes

from neurojax.io.cifti import (
    CiftiSeries,
    cortex_to_surface,
    load_dtseries,
    save_dscalar,
    surface_vertex_mask,
)

N_LEFT, N_RIGHT = 10, 12          # mesh vertices per hemisphere
KEEP_LEFT = np.array([0, 1, 2, 5, 6, 7, 8])      # medial wall = 3, 4, 9
KEEP_RIGHT = np.array([0, 2, 3, 4, 5, 6, 7, 8, 10, 11])
N_SUB = 5                         # a subcortical voxel structure
T, TR = 7, 1.5


def _brain_axis():
    bm_l = axes.BrainModelAxis.from_mask(
        np.isin(np.arange(N_LEFT), KEEP_LEFT), name="CortexLeft")
    bm_r = axes.BrainModelAxis.from_mask(
        np.isin(np.arange(N_RIGHT), KEEP_RIGHT), name="CortexRight")
    bm_s = axes.BrainModelAxis.from_mask(
        np.ones((N_SUB, 1, 1), bool), name="Thalamus_Left", affine=np.eye(4))
    return bm_l + bm_r + bm_s


@pytest.fixture
def dtseries(tmp_path):
    """Synthetic dtseries with cortex-left, cortex-right and one voxel structure."""
    n = len(KEEP_LEFT) + len(KEEP_RIGHT) + N_SUB
    data = np.arange(T * n, dtype=np.float32).reshape(T, n)
    hdr = nib.cifti2.Cifti2Header.from_axes((axes.SeriesAxis(0, TR, T), _brain_axis()))
    path = tmp_path / "synthetic.dtseries.nii"
    nib.cifti2.Cifti2Image(data, header=hdr).to_filename(path)
    return path, data


class TestLoad:
    def test_cortex_only_by_default(self, dtseries):
        path, data = dtseries
        s = load_dtseries(path)
        assert isinstance(s, CiftiSeries)
        assert s.data.shape == (T, len(KEEP_LEFT) + len(KEEP_RIGHT))
        np.testing.assert_array_equal(s.data, data[:, : len(KEEP_LEFT) + len(KEEP_RIGHT)])
        assert s.tr == pytest.approx(TR)

    def test_structure_bookkeeping(self, dtseries):
        s = load_dtseries(dtseries[0])
        assert s.structures["CIFTI_STRUCTURE_CORTEX_LEFT"] == slice(0, len(KEEP_LEFT))
        assert s.n_mesh_vertices == {
            "CIFTI_STRUCTURE_CORTEX_LEFT": N_LEFT,
            "CIFTI_STRUCTURE_CORTEX_RIGHT": N_RIGHT,
        }
        np.testing.assert_array_equal(
            s.vertex_indices["CIFTI_STRUCTURE_CORTEX_RIGHT"], KEEP_RIGHT)

    def test_include_subcortical(self, dtseries):
        s = load_dtseries(dtseries[0], structures=None)
        assert s.data.shape[1] == len(KEEP_LEFT) + len(KEEP_RIGHT) + N_SUB
        assert "CIFTI_STRUCTURE_THALAMUS_LEFT" in s.structures


class TestSurfaceMapping:
    def test_vertex_mask_matches_medial_wall(self, dtseries):
        s = load_dtseries(dtseries[0])
        mask = surface_vertex_mask(s)
        assert mask.shape == (N_LEFT + N_RIGHT,)
        assert mask.sum() == len(KEEP_LEFT) + len(KEEP_RIGHT)
        assert not mask[3] and not mask[4] and not mask[9]       # left wall
        assert not mask[N_LEFT + 1] and not mask[N_LEFT + 9]     # right wall

    def test_cortex_to_surface_round_trip(self, dtseries):
        s = load_dtseries(dtseries[0])
        full = cortex_to_surface(s.data, s)
        assert full.shape == (T, N_LEFT + N_RIGHT)
        mask = surface_vertex_mask(s)
        assert np.all(np.isnan(full[:, ~mask]))
        np.testing.assert_array_equal(full[:, mask], s.data)
        # left vertex 5 is the 4th kept left vertex
        np.testing.assert_array_equal(full[:, 5], s.data[:, 3])


class TestSave:
    def test_dscalar_round_trip(self, dtseries, tmp_path):
        s = load_dtseries(dtseries[0])
        maps = np.random.default_rng(0).normal(size=(3, s.data.shape[1])).astype(np.float32)
        out = tmp_path / "maps.dscalar.nii"
        save_dscalar(out, maps, s, names=["a", "b", "c"])
        img = nib.load(out)
        assert img.shape == (3, s.data.shape[1])
        np.testing.assert_allclose(img.get_fdata(), maps)
        assert list(img.header.get_axis(0).name) == ["a", "b", "c"]
        assert img.header.get_axis(1).nvertices["CIFTI_STRUCTURE_CORTEX_LEFT"] == N_LEFT


REAL = "/data/datasets/ds006072/NON_BIDS/ciftis/sub-1_After1_upck_faln_dbnd_xr3d_dc_atl.ctx.dtseries.nii"


@pytest.mark.skipif(not os.path.exists(REAL), reason="ds006072 CIFTIs not staged")
def test_real_fslr_32k():
    s = load_dtseries(REAL)
    assert s.data.shape == (1490, 59412)
    assert s.n_mesh_vertices["CIFTI_STRUCTURE_CORTEX_LEFT"] == 32492
    assert surface_vertex_mask(s).sum() == 59412
    assert s.tr == pytest.approx(1.761)


class TestConnectomes:
    """pconn / dconn writers on the same grayordinate contract (step 7)."""

    def _parcels(self):
        return [("lh.A", {"CIFTI_STRUCTURE_CORTEX_LEFT": np.array([0, 1, 2])}),
                ("lh.B", {"CIFTI_STRUCTURE_CORTEX_LEFT": np.array([3, 5, 7])}),
                ("rh.A", {"CIFTI_STRUCTURE_CORTEX_RIGHT": np.array([2, 3, 4, 5])})]

    def test_pconn_roundtrip(self, tmp_path):
        from neurojax.io.cifti import load_pconn, save_pconn
        parcels = self._parcels()
        C = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]], np.float32)
        path = tmp_path / "test.pconn.nii"
        save_pconn(path, C, parcels, n_mesh_vertices={
            "CIFTI_STRUCTURE_CORTEX_LEFT": N_LEFT, "CIFTI_STRUCTURE_CORTEX_RIGHT": N_RIGHT})
        img = nib.load(str(path))
        assert img.nifti_header.get_intent()[0] == "ConnParcels"
        for i in range(2):
            assert isinstance(img.header.get_axis(i), axes.ParcelsAxis)
        names, M = load_pconn(path)
        assert names == ["lh.A", "lh.B", "rh.A"]
        np.testing.assert_allclose(M, C)
        ax = img.header.get_axis(0)
        np.testing.assert_array_equal(ax.vertices[2]["CIFTI_STRUCTURE_CORTEX_RIGHT"], [2, 3, 4, 5])
        assert ax.nvertices["CIFTI_STRUCTURE_CORTEX_LEFT"] == N_LEFT

    def test_pconn_rejects_shape_mismatch(self, tmp_path):
        from neurojax.io.cifti import save_pconn
        with pytest.raises(ValueError):
            save_pconn(tmp_path / "bad.pconn.nii", np.eye(2, dtype=np.float32), self._parcels())

    def test_pconn_stack_of_frequencies(self, tmp_path):
        """A (F, P, P) stack is written as F pconn files, one per frequency."""
        from neurojax.io.cifti import load_pconn, save_pconn
        C = np.stack([np.eye(3, dtype=np.float32) * (f + 1) for f in range(2)])
        paths = save_pconn(tmp_path / "band.pconn.nii", C, self._parcels(), names=["8Hz", "9Hz"])
        assert [p.name for p in paths] == ["band_8Hz.pconn.nii", "band_9Hz.pconn.nii"]
        np.testing.assert_allclose(load_pconn(paths[1])[1], 2 * np.eye(3))

    def test_dconn_on_grayordinates(self, dtseries, tmp_path):
        from neurojax.io.cifti import save_dconn
        s = load_dtseries(dtseries[0])
        G = s.data.shape[1]
        C = np.corrcoef(np.random.RandomState(0).randn(G, T)).astype(np.float32)
        path = tmp_path / "test.dconn.nii"
        save_dconn(path, C, s)
        img = nib.load(str(path))
        assert img.nifti_header.get_intent()[0] == "ConnDense"
        assert img.shape == (G, G)
        assert len(img.header.get_axis(0)) == G and isinstance(img.header.get_axis(1), axes.BrainModelAxis)
        np.testing.assert_allclose(np.asarray(img.dataobj), C)
