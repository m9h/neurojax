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
