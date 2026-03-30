"""NIfTI/MGZ I/O helpers for qMRI data.

Bridges nibabel ↔ JAX arrays with orientation handling.
"""

import nibabel as nib
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Optional


def load_nifti(path: str):
    """Load NIfTI or MGZ file, return JAX array + affine + header.

    Args:
        path: Path to .nii.gz, .nii, or .mgz file

    Returns:
        (data, affine, header) where data is jnp.ndarray
    """
    img = nib.load(str(path))
    data = jnp.array(img.get_fdata(dtype=np.float32))
    affine = np.array(img.affine)
    return data, affine, img.header


def save_nifti(data, affine: np.ndarray, path: str,
               header=None):
    """Save JAX or numpy array as NIfTI.

    Args:
        data: Array (JAX or numpy)
        affine: 4x4 affine matrix
        path: Output path (.nii.gz)
        header: Optional nibabel header
    """
    arr = np.asarray(data, dtype=np.float32)
    img = nib.Nifti1Image(arr, affine, header)
    nib.save(img, str(path))


def load_multi_volume(path: str, volumes: Optional[list] = None):
    """Load 4D NIfTI with optional volume selection.

    Args:
        path: Path to 4D NIfTI
        volumes: List of volume indices to load, or None for all

    Returns:
        (data, affine, header) — data is (X, Y, Z, N) JAX array
    """
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if volumes is not None:
        data = data[..., volumes]
    return jnp.array(data), np.array(img.affine), img.header


def reorient_to_standard(data: jnp.ndarray, affine: np.ndarray):
    """Reorient volume so array dimensions correspond to R-L, A-P, I-S.

    Args:
        data: 3D or 4D array
        affine: 4x4 affine matrix

    Returns:
        (reoriented_data, new_affine, permutation)
    """
    abs_aff = np.abs(affine[:3, :3])
    # For each array dim, which anatomical axis dominates?
    axis_map = np.argmax(abs_aff, axis=0)

    if np.array_equal(axis_map, [0, 1, 2]):
        return data, affine, None

    perm = np.argsort(axis_map)
    if data.ndim == 4:
        data_reoriented = jnp.transpose(data, (*perm, 3))
    else:
        data_reoriented = jnp.transpose(data, perm)

    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, perm]
    new_affine[:3, 3] = affine[:3, 3]

    return data_reoriented, new_affine, perm
