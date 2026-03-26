"""Tests for source space decimation (_get_ico_surface, setup_source_space)."""

import struct
import jax.numpy as jnp
import numpy as np
import pytest

from neurojax.geometry.source_space import _get_ico_surface, setup_source_space


def _write_fs_surface(path, vertices, faces):
    """Write a minimal FreeSurfer surface file."""
    with open(path, "wb") as f:
        f.write(b"\xff\xff\xfe")
        f.write(b"test\n\n")
        f.write(struct.pack(">ii", len(vertices), len(faces)))
        for v in vertices:
            f.write(struct.pack(">fff", *v))
        for face in faces:
            f.write(struct.pack(">iii", *face))


class TestGetIcoSurface:
    def test_grade0_vertex_count(self):
        verts, faces = _get_ico_surface(0)
        assert verts.shape == (12, 3)

    def test_grade0_face_count(self):
        verts, faces = _get_ico_surface(0)
        assert faces.shape == (20, 3)

    def test_unit_sphere(self):
        verts, _ = _get_ico_surface(0)
        norms = np.linalg.norm(verts, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_valid_face_indices(self):
        verts, faces = _get_ico_surface(0)
        assert np.all(faces >= 0)
        assert np.all(faces < len(verts))

    def test_euler_characteristic(self):
        """V - E + F = 2 for a closed surface."""
        verts, faces = _get_ico_surface(0)
        V = len(verts)
        F = len(faces)
        # Count edges from faces
        edges = set()
        for f in faces:
            for i in range(3):
                e = tuple(sorted([int(f[i]), int(f[(i + 1) % 3])]))
                edges.add(e)
        E = len(edges)
        assert V - E + F == 2


class TestSetupSourceSpace:
    @pytest.fixture
    def fs_tree(self, tmp_path):
        """Create a minimal FreeSurfer subject directory."""
        sub_dir = tmp_path / "subjects" / "fsaverage" / "surf"
        sub_dir.mkdir(parents=True)

        # Create sphere and white surfaces for both hemispheres
        # Sphere: 12 vertices on unit sphere * 100 (FS convention)
        ico_v, ico_f = _get_ico_surface(0)
        sphere_verts = ico_v * 100.0  # FS sphere has radius ~100

        # White surface: same topology but slightly different coords
        white_verts = ico_v * 105.0

        for hemi in ["lh", "rh"]:
            _write_fs_surface(
                str(sub_dir / f"{hemi}.sphere"),
                sphere_verts.tolist(), ico_f.tolist()
            )
            _write_fs_surface(
                str(sub_dir / f"{hemi}.white"),
                white_verts.tolist(), ico_f.tolist()
            )

        return str(tmp_path / "subjects"), "fsaverage"

    def test_returns_both_hemispheres(self, fs_tree):
        subj_dir, subj = fs_tree
        spaces = setup_source_space(subj_dir, subj, grade=0)
        assert "lh" in spaces
        assert "rh" in spaces

    def test_required_keys(self, fs_tree):
        subj_dir, subj = fs_tree
        spaces = setup_source_space(subj_dir, subj, grade=0)
        for hemi in ["lh", "rh"]:
            assert "vertno" in spaces[hemi]
            assert "rr" in spaces[hemi]
            assert "nn" in spaces[hemi]
            assert "n_use" in spaces[hemi]

    def test_rr_shape(self, fs_tree):
        subj_dir, subj = fs_tree
        spaces = setup_source_space(subj_dir, subj, grade=0)
        for hemi in ["lh", "rh"]:
            n_use = spaces[hemi]["n_use"]
            assert spaces[hemi]["rr"].shape == (n_use, 3)

    def test_nn_unit_length(self, fs_tree):
        subj_dir, subj = fs_tree
        spaces = setup_source_space(subj_dir, subj, grade=0)
        for hemi in ["lh", "rh"]:
            nn = np.array(spaces[hemi]["nn"])
            norms = np.linalg.norm(nn, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_jax_arrays(self, fs_tree):
        subj_dir, subj = fs_tree
        spaces = setup_source_space(subj_dir, subj, grade=0)
        assert isinstance(spaces["lh"]["rr"], jnp.ndarray)
        assert isinstance(spaces["lh"]["vertno"], jnp.ndarray)

    def test_missing_hemisphere_skipped(self, tmp_path):
        """If one hemisphere is missing, it should be skipped."""
        sub_dir = tmp_path / "subjects" / "sub01" / "surf"
        sub_dir.mkdir(parents=True)
        # Only create lh files
        ico_v, ico_f = _get_ico_surface(0)
        _write_fs_surface(
            str(sub_dir / "lh.sphere"),
            (ico_v * 100).tolist(), ico_f.tolist()
        )
        _write_fs_surface(
            str(sub_dir / "lh.white"),
            (ico_v * 105).tolist(), ico_f.tolist()
        )
        spaces = setup_source_space(str(tmp_path / "subjects"), "sub01", grade=0)
        assert "lh" in spaces
        assert "rh" not in spaces
