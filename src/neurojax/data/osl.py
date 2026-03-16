"""Helper functions for discovering osl-ephys output files.

osl-ephys produces standardised directory structures for preprocessed and
source-reconstructed data.  The helpers here walk those structures and return
sorted lists of file paths that can be fed directly into ``Data(...)``.

Typical layouts::

    preproc_dir/
        sub-001/ses-01/sub-001_run-01_preproc-raw.fif
        sub-001/ses-01/sub-001_run-02_preproc-raw.fif
        sub-002/ses-01/sub-002_run-01_preproc-raw.fif

    recon_dir/
        sub-001/sflip_parc-raw.fif   (or sflip_parc.npy)
        sub-002/sflip_parc-raw.fif
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


def find_subjects(base_dir: str | os.PathLike) -> List[str]:
    """List subject directories under *base_dir*.

    Directories whose name starts with ``sub-`` are returned, sorted
    lexicographically.

    Parameters
    ----------
    base_dir : str or path-like
        Top-level directory containing subject folders.

    Returns
    -------
    list of str
        Absolute paths to subject directories.
    """
    base = Path(base_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Directory not found: {base}")
    subjects = sorted(
        str(p) for p in base.iterdir() if p.is_dir() and p.name.startswith("sub-")
    )
    return subjects


def find_parcellated_files(
    recon_dir: str | os.PathLike,
    format: str = "npy",
) -> List[str]:
    """Find sign-flipped parcellated source data files.

    Parameters
    ----------
    recon_dir : str or path-like
        Source reconstruction output directory (contains ``sub-XXX/``
        subdirectories).
    format : ``"npy"`` or ``"fif"``
        Which file variant to look for.  ``"npy"`` matches
        ``sflip_parc.npy``; ``"fif"`` matches ``sflip_parc-raw.fif``.

    Returns
    -------
    list of str
        Sorted absolute paths to the discovered files.

    Raises
    ------
    ValueError
        If *format* is not ``"npy"`` or ``"fif"``.
    FileNotFoundError
        If *recon_dir* does not exist.
    """
    base = Path(recon_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Directory not found: {base}")

    if format == "npy":
        filename = "sflip_parc.npy"
    elif format == "fif":
        filename = "sflip_parc-raw.fif"
    else:
        raise ValueError(f"format must be 'npy' or 'fif', got '{format}'")

    files = sorted(str(p) for p in base.rglob(filename))
    return files


def find_preprocessed_files(
    preproc_dir: str | os.PathLike,
) -> List[str]:
    """Find all preprocessed ``.fif`` files in an osl-ephys preproc directory.

    Searches recursively for files matching the pattern
    ``*_preproc-raw.fif``.

    Parameters
    ----------
    preproc_dir : str or path-like
        Preprocessing output directory.

    Returns
    -------
    list of str
        Sorted absolute paths to the discovered files.

    Raises
    ------
    FileNotFoundError
        If *preproc_dir* does not exist.
    """
    base = Path(preproc_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Directory not found: {base}")

    files = sorted(str(p) for p in base.rglob("*_preproc-raw.fif"))
    return files
