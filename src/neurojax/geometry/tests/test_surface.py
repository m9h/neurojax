"""Tests for FreeSurfer surface IO (read_surface, read_curv)."""

import struct
import jax.numpy as jnp
import numpy as np
import pytest

from neurojax.geometry.surface import read_curv, read_surface


def _write_freesurfer_surface(path, vertices, faces):
    """Write a FreeSurfer triangle surface file."""
    n_verts = len(vertices)
    n_faces = len(faces)
    with open(path, "wb") as f:
        # Magic number for triangle format
        f.write(b"\xff\xff\xfe")
        # Creator string + two newlines
        f.write(b"created by test\n\n")
        # Vertex and face counts (big-endian int32)
        f.write(struct.pack(">ii", n_verts, n_faces))
        # Vertices (big-endian float32)
        for v in vertices:
            f.write(struct.pack(">fff", *v))
        # Faces (big-endian int32)
        for face in faces:
            f.write(struct.pack(">iii", *face))


def _write_freesurfer_curv(path, values):
    """Write a FreeSurfer curvature file."""
    n = len(values)
    with open(path, "wb") as f:
        f.write(b"\xff\xff\xff")  # magic
        f.write(struct.pack(">i", n))  # n_vertices
        f.write(struct.pack(">i", 0))  # n_faces (unused)
        f.write(struct.pack(">i", 1))  # vals_per_vertex
        for v in values:
            f.write(struct.pack(">f", v))


class TestReadSurface:
    def test_tetrahedron_shapes(self, tmp_path):
        verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        path = tmp_path / "lh.white"
        _write_freesurfer_surface(str(path), verts, faces)
        v, f = read_surface(str(path))
        assert v.shape == (4, 3)
        assert f.shape == (4, 3)

    def test_vertex_values(self, tmp_path):
        verts = [[1.5, -2.3, 0.7], [0.0, 0.0, 0.0]]
        faces = [[0, 1, 0]]
        path = tmp_path / "test.surf"
        _write_freesurfer_surface(str(path), verts, faces)
        v, _ = read_surface(str(path))
        np.testing.assert_allclose(v[0], [1.5, -2.3, 0.7], atol=1e-5)

    def test_face_indices(self, tmp_path):
        verts = [[0, 0, 0]] * 5
        faces = [[0, 1, 2], [2, 3, 4]]
        path = tmp_path / "test.surf"
        _write_freesurfer_surface(str(path), verts, faces)
        _, f = read_surface(str(path))
        assert int(f[1, 2]) == 4

    def test_returns_jax_arrays(self, tmp_path):
        verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        faces = [[0, 1, 2]]
        path = tmp_path / "test.surf"
        _write_freesurfer_surface(str(path), verts, faces)
        v, f = read_surface(str(path))
        assert isinstance(v, jnp.ndarray)
        assert isinstance(f, jnp.ndarray)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read_surface("/nonexistent/lh.white")


class TestReadCurv:
    def test_roundtrip(self, tmp_path):
        values = [0.1, -0.5, 1.2, 0.0]
        path = tmp_path / "lh.curv"
        _write_freesurfer_curv(str(path), values)
        curv = read_curv(str(path))
        assert curv.shape == (4,)
        np.testing.assert_allclose(curv, values, atol=1e-5)

    def test_returns_jax_array(self, tmp_path):
        _write_freesurfer_curv(str(tmp_path / "c.curv"), [1.0, 2.0])
        curv = read_curv(str(tmp_path / "c.curv"))
        assert isinstance(curv, jnp.ndarray)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read_curv("/nonexistent/lh.curv")
