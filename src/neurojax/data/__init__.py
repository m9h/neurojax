"""Data loading and preparation utilities for neurojax.

Provides an osl-dynamics-compatible :class:`Data` class and stand-alone
preparation functions for time-delay embedding and PCA.

Quick start::

    from neurojax.data import Data

    data = Data("/path/to/recon_dir")
    data.prepare({"tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
                   "standardize": {}})
    prepared = data.prepared_data  # list of JAX arrays
"""

from neurojax.data.loading import Data, prepare_pca, prepare_tde
from neurojax.data.osl import (
    find_parcellated_files,
    find_preprocessed_files,
    find_subjects,
)

__all__ = [
    "Data",
    "prepare_tde",
    "prepare_pca",
    "find_subjects",
    "find_parcellated_files",
    "find_preprocessed_files",
]
