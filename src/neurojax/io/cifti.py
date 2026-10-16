# © NeuroJAX developers
#
# License: BSD (3-clause)
"""CIFTI-2 dense series / scalar I/O on the HCP fs_LR grayordinate contract.

A dense CIFTI stores *grayordinates*: cortical surface vertices with the medial
wall removed (59,412 of 2 x 32,492 at fs_LR 32k) plus subcortical voxels. This
module keeps the bookkeeping needed to go back and forth between grayordinates
and full surface meshes, which is what surface graph operators (e.g.
:mod:`neurojax.spatial.harmonics`) need. The same contract is used by HCP
megconnectome / CiftiStorm outputs, so EEG/MEG source estimates and BOLD share
one surface space.
"""

from typing import NamedTuple, Optional, Sequence

import nibabel as nib
import numpy as np
from nibabel.cifti2 import cifti2_axes as axes

CORTEX = ("CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT")


class CiftiSeries(NamedTuple):
    """A dense CIFTI matrix with its grayordinate bookkeeping.

    Attributes:
        data: (T, G) values over the selected grayordinates.
        structures: structure name -> slice into the G axis of ``data``.
        vertex_indices: surface structure name -> (n,) mesh vertex ids.
        n_mesh_vertices: surface structure name -> full mesh vertex count.
        tr: sampling step of the series axis in seconds (None for scalars).
        brain_axis: the nibabel BrainModelAxis of the selected grayordinates.
    """

    data: np.ndarray
    structures: dict
    vertex_indices: dict
    n_mesh_vertices: dict
    tr: Optional[float]
    brain_axis: axes.BrainModelAxis


def load_dtseries(path, structures: Optional[Sequence[str]] = CORTEX) -> CiftiSeries:
    """Load a dense CIFTI (dtseries / dscalar) restricted to selected structures.

    Args:
        path: ``*.dtseries.nii`` or ``*.dscalar.nii`` file.
        structures: structure names to keep, in this order. Default is the two
            cortical surfaces; ``None`` keeps every structure in file order.

    Returns:
        :class:`CiftiSeries` with ``data`` of shape (T, G).
    """
    img = nib.load(str(path))
    series_axis, brain_axis = (img.header.get_axis(i) for i in range(2))
    present = {name: (sl, bm) for name, sl, bm in brain_axis.iter_structures()}
    if structures is None:
        structures = list(present)
    missing = [s for s in structures if s not in present]
    if missing:
        raise ValueError(f"structures not in {path}: {missing}; have {list(present)}")

    full = np.asarray(img.dataobj, dtype=np.float32)
    blocks, out_structures, vertex_indices, bm_parts, start = [], {}, {}, [], 0
    for name in structures:
        sl, bm = present[name]
        blocks.append(full[:, sl])
        out_structures[name] = slice(start, start + len(bm))
        if bm.surface_mask.all():
            vertex_indices[name] = np.asarray(bm.vertex)
        bm_parts.append(bm)
        start += len(bm)

    merged = bm_parts[0]
    for bm in bm_parts[1:]:
        merged = merged + bm
    n_mesh = {k: v for k, v in brain_axis.nvertices.items() if k in vertex_indices}
    tr = float(series_axis.step) if isinstance(series_axis, axes.SeriesAxis) else None
    return CiftiSeries(np.concatenate(blocks, axis=1), out_structures, vertex_indices,
                       n_mesh, tr, merged)


def _surface_offsets(series: CiftiSeries):
    """Per-surface offset into a concatenated full-mesh (left then right) array."""
    offsets, total = {}, 0
    for name in CORTEX:
        if name in series.n_mesh_vertices:
            offsets[name] = total
            total += series.n_mesh_vertices[name]
    return offsets, total


def surface_vertex_mask(series: CiftiSeries) -> np.ndarray:
    """Boolean mask over the concatenated full meshes of the vertices with data.

    False entries are the medial wall (and any other vertices the CIFTI omits).
    """
    offsets, total = _surface_offsets(series)
    mask = np.zeros(total, dtype=bool)
    for name, off in offsets.items():
        mask[off + series.vertex_indices[name]] = True
    return mask


def cortex_to_surface(values, series: CiftiSeries, fill=np.nan) -> np.ndarray:
    """Scatter cortical grayordinate values onto the full concatenated meshes.

    Args:
        values: (..., G_cortex) values ordered as the cortical structures of
            ``series`` (left then right).
        series: bookkeeping from :func:`load_dtseries`.
        fill: value for vertices without data (medial wall).

    Returns:
        (..., N_left + N_right) array.
    """
    values = np.asarray(values)
    offsets, total = _surface_offsets(series)
    out = np.full(values.shape[:-1] + (total,), fill, dtype=values.dtype)
    for name, off in offsets.items():
        sl = series.structures[name]
        out[..., off + series.vertex_indices[name]] = values[..., sl]
    return out


def save_dscalar(path, maps, series: CiftiSeries, names: Optional[Sequence[str]] = None):
    """Write (K, G) maps as a ``*.dscalar.nii`` on the grayordinates of ``series``.

    Useful for viewing harmonics or per-vertex statistics in Connectome Workbench.
    """
    maps = np.asarray(maps, dtype=np.float32)
    if maps.ndim == 1:
        maps = maps[None]
    if maps.shape[1] != len(series.brain_axis):
        raise ValueError(
            f"maps have {maps.shape[1]} grayordinates, series has {len(series.brain_axis)}")
    names = list(names) if names is not None else [f"map{i}" for i in range(len(maps))]
    hdr = nib.cifti2.Cifti2Header.from_axes((axes.ScalarAxis(names), series.brain_axis))
    nib.cifti2.Cifti2Image(maps, header=hdr).to_filename(str(path))


def _parcels_axis(parcels, n_mesh_vertices: Optional[dict]) -> axes.ParcelsAxis:
    names = [p[0] for p in parcels]
    vertices = [{k: np.asarray(v, dtype=int) for k, v in p[1].items()} for p in parcels]
    voxels = [np.zeros((0, 3), dtype=int) for _ in parcels]
    ax = axes.ParcelsAxis(names, voxels, vertices)
    # nibabel needs the mesh size of every surface structure; without it, fall
    # back to the largest vertex id seen (Workbench then needs a matching mesh)
    for v in vertices:
        for k, idx in v.items():
            ax.nvertices[k] = max(ax.nvertices.get(k, 0), int(idx.max()) + 1 if idx.size else 0)
    if n_mesh_vertices is not None:
        ax.nvertices.update(n_mesh_vertices)
    return ax


def save_pconn(path, matrix, parcels, names: Optional[Sequence[str]] = None,
               n_mesh_vertices: Optional[dict] = None):
    """Write a parcellated connectome as ``*.pconn.nii`` (ParcelsAxis x ParcelsAxis).

    Args:
        path: output file. For a (F, P, P) stack, one file per slice is written
            with ``names[f]`` (or the index) inserted before ``.pconn.nii``.
        matrix: (P, P) or (F, P, P) real connectivity.
        parcels: sequence of ``(name, {structure: vertex ids})`` on the full
            surface mesh — the HCP/megconnectome parcel contract.
        names: labels for the F slices of a stack.
        n_mesh_vertices: structure name -> full mesh vertex count, written into
            the axis so Workbench can relate parcels to a surface.

    Returns:
        the written path, or the list of paths for a stack.
    """
    from pathlib import Path

    matrix = np.asarray(matrix, dtype=np.float32)
    P = len(parcels)
    if matrix.ndim == 3:
        names = list(names) if names is not None else [str(f) for f in range(len(matrix))]
        path = Path(path)
        stem = path.name[: -len(".pconn.nii")] if path.name.endswith(".pconn.nii") else path.stem
        return [save_pconn(path.with_name(f"{stem}_{n}.pconn.nii"), m, parcels,
                           n_mesh_vertices=n_mesh_vertices) for n, m in zip(names, matrix)]
    if matrix.shape != (P, P):
        raise ValueError(f"matrix is {matrix.shape}, expected ({P}, {P}) for {P} parcels")
    ax = _parcels_axis(parcels, n_mesh_vertices)
    hdr = nib.cifti2.Cifti2Header.from_axes((ax, ax))
    img = nib.cifti2.Cifti2Image(matrix, header=hdr)
    img.nifti_header.set_intent("NIFTI_INTENT_CONNECTIVITY_PARCELLATED")
    img.to_filename(str(path))
    return Path(path)


def load_pconn(path):
    """Read a ``*.pconn.nii``: returns ``(parcel names, (P, P) matrix)``."""
    img = nib.load(str(path))
    return list(img.header.get_axis(0).name), np.asarray(img.dataobj, dtype=np.float32)


def save_dconn(path, matrix, series: CiftiSeries):
    """Write a dense (G, G) connectome on the grayordinates of ``series`` as ``*.dconn.nii``."""
    matrix = np.asarray(matrix, dtype=np.float32)
    G = len(series.brain_axis)
    if matrix.shape != (G, G):
        raise ValueError(f"matrix is {matrix.shape}, series has {G} grayordinates")
    hdr = nib.cifti2.Cifti2Header.from_axes((series.brain_axis, series.brain_axis))
    img = nib.cifti2.Cifti2Image(matrix, header=hdr)
    img.nifti_header.set_intent("NIFTI_INTENT_CONNECTIVITY_DENSE")
    img.to_filename(str(path))
