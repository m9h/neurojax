"""osl-dynamics-compatible data loading and preparation for JAX.

This module provides :class:`Data`, a JAX-native reimplementation of the
osl-dynamics ``Data`` class.  It loads ``.npy``, ``.mat``, and ``.fif``
files, stores them as JAX arrays with shape ``(n_samples, n_channels)``,
and supports the same ``prepare(methods)`` API for time-delay embedding,
PCA, standardisation, etc.

Example
-------
>>> from neurojax.data import Data
>>> data = Data(["/path/to/sub-001.npy", "/path/to/sub-002.npy"])
>>> data.prepare({"tde_pca": {"n_embeddings": 15, "n_pca_components": 80}})
>>> data.prepared_data  # list of JAX arrays, one per subject
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stand-alone preparation functions
# ---------------------------------------------------------------------------


def prepare_tde(
    x: jax.Array,
    n_embeddings: int = 15,
) -> jax.Array:
    """Time-delay embedding (TDE).

    For an input of shape ``(T, C)`` and *n_embeddings* lags, the output has
    shape ``(T - n_embeddings + 1, C * n_embeddings)`` where each row is the
    concatenation of ``n_embeddings`` consecutive windows.

    Parameters
    ----------
    x : jax.Array, shape (n_samples, n_channels)
        Input time series.
    n_embeddings : int
        Number of lagged copies (including lag-0).

    Returns
    -------
    jax.Array, shape (n_samples - n_embeddings + 1, n_channels * n_embeddings)
    """
    if n_embeddings < 1:
        raise ValueError(f"n_embeddings must be >= 1, got {n_embeddings}")
    if n_embeddings == 1:
        return x

    T, C = x.shape
    T_out = T - n_embeddings + 1
    if T_out <= 0:
        raise ValueError(
            f"n_embeddings ({n_embeddings}) must be less than n_samples ({T})"
        )

    # Build index array for all lagged windows
    indices = jnp.arange(n_embeddings)[None, :] + jnp.arange(T_out)[:, None]
    # indices shape: (T_out, n_embeddings)
    # x[indices] shape: (T_out, n_embeddings, C)
    embedded = x[indices]
    # Reshape to (T_out, n_embeddings * C)
    return embedded.reshape(T_out, n_embeddings * C)


def prepare_pca(
    x: jax.Array,
    n_pca_components: int,
) -> jax.Array:
    """SVD-based PCA dimensionality reduction.

    Data is mean-centred before computing the SVD.  The top
    *n_pca_components* principal components are retained.

    Parameters
    ----------
    x : jax.Array, shape (n_samples, n_features)
        Input data.
    n_pca_components : int
        Number of principal components to keep.

    Returns
    -------
    jax.Array, shape (n_samples, n_pca_components)
    """
    if n_pca_components < 1:
        raise ValueError(
            f"n_pca_components must be >= 1, got {n_pca_components}"
        )

    n_features = x.shape[1]
    if n_pca_components > n_features:
        raise ValueError(
            f"n_pca_components ({n_pca_components}) > n_features ({n_features})"
        )

    mean = jnp.mean(x, axis=0)
    x_centred = x - mean

    # Economy SVD — only need the first n_pca_components
    U, S, Vh = jnp.linalg.svd(x_centred, full_matrices=False)
    # Project onto top components: X_pca = U[:, :k] * S[:k]
    return U[:, :n_pca_components] * S[:n_pca_components]


def _standardize(x: jax.Array) -> jax.Array:
    """Z-score each channel (zero mean, unit variance).

    Parameters
    ----------
    x : jax.Array, shape (n_samples, n_channels)

    Returns
    -------
    jax.Array, same shape as *x*.
    """
    mean = jnp.mean(x, axis=0)
    std = jnp.std(x, axis=0)
    # Avoid division by zero for constant channels
    std = jnp.where(std == 0, 1.0, std)
    return (x - mean) / std


def _amplitude_envelope(x: jax.Array) -> jax.Array:
    """Compute the amplitude envelope (absolute value of the Hilbert transform).

    Uses ``jnp.fft`` to compute the analytic signal.

    Parameters
    ----------
    x : jax.Array, shape (n_samples, n_channels)

    Returns
    -------
    jax.Array, same shape as *x*.
    """
    T = x.shape[0]
    X_fft = jnp.fft.fft(x, axis=0)

    # Build the multiplier for the analytic signal:
    #   h[0] = 1, h[1:T//2] = 2, h[T//2] = 1 (if T even), rest = 0
    h = jnp.zeros(T)
    h = h.at[0].set(1.0)
    if T % 2 == 0:
        h = h.at[1 : T // 2].set(2.0)
        h = h.at[T // 2].set(1.0)
    else:
        h = h.at[1 : (T + 1) // 2].set(2.0)

    analytic = jnp.fft.ifft(X_fft * h[:, None], axis=0)
    return jnp.abs(analytic).real


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------


def _load_npy(path: str) -> jax.Array:
    """Load a ``.npy`` file and return a JAX array (n_samples, n_channels)."""
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[:, None]
    return jnp.array(arr)


def _load_mat(path: str, data_field: str = "X") -> jax.Array:
    """Load a ``.mat`` file and return a JAX array (n_samples, n_channels).

    Uses ``scipy.io.loadmat``; the variable named *data_field* is extracted.
    """
    from scipy.io import loadmat

    mat = loadmat(path)
    if data_field not in mat:
        available = [k for k in mat if not k.startswith("__")]
        raise KeyError(
            f"Field '{data_field}' not found in {path}. "
            f"Available fields: {available}"
        )
    arr = np.asarray(mat[data_field], dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return jnp.array(arr)


def _load_fif(path: str, picks: str = "misc") -> jax.Array:
    """Load a ``.fif`` file via MNE and return a JAX array (n_samples, n_channels).

    For parcellated source data the channel type is typically ``"misc"``.
    """
    import mne

    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    data = raw.get_data(picks=picks)  # (n_channels, n_samples)
    return jnp.array(data.T)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


class Data:
    """osl-dynamics-compatible data container backed by JAX arrays.

    Parameters
    ----------
    inputs : str, path-like, list of str/path-like, or list of np.ndarray/jax.Array
        What to load.

        * **Directory path** -- all ``.npy`` files found (recursively) in the
          directory are loaded and sorted lexicographically.
        * **List of file paths** -- each file is loaded individually.
          Supported extensions: ``.npy``, ``.mat``, ``.fif``.
        * **List of arrays** -- used directly (converted to JAX arrays).

    data_field : str, optional
        Variable name to extract from ``.mat`` files (default ``"X"``).
    picks : str, optional
        Channel selection string passed to MNE when reading ``.fif`` files
        (default ``"misc"``).
    """

    def __init__(
        self,
        inputs: Union[
            str,
            os.PathLike,
            Sequence[Union[str, os.PathLike, np.ndarray, jax.Array]],
        ],
        data_field: str = "X",
        picks: str = "misc",
    ) -> None:
        self.data_field = data_field
        self.picks = picks

        self._raw_data: List[jax.Array] = self._load_inputs(inputs)
        self._prepared_data: Optional[List[jax.Array]] = None

        n_subjects = len(self._raw_data)
        shapes = [d.shape for d in self._raw_data]
        logger.info("Loaded %d dataset(s): %s", n_subjects, shapes)

    # -- public properties ---------------------------------------------------

    @property
    def raw_data(self) -> List[jax.Array]:
        """Raw (unprocessed) data -- list of arrays, one per subject/session."""
        return self._raw_data

    @property
    def prepared_data(self) -> List[jax.Array]:
        """Prepared data (after :meth:`prepare`).  Falls back to raw data."""
        if self._prepared_data is not None:
            return self._prepared_data
        return self._raw_data

    @property
    def n_subjects(self) -> int:
        """Number of loaded datasets (subjects / sessions)."""
        return len(self._raw_data)

    @property
    def n_channels(self) -> int:
        """Number of channels in the (first) raw dataset."""
        return self._raw_data[0].shape[1]

    # -- loading -------------------------------------------------------------

    def _load_inputs(self, inputs) -> List[jax.Array]:
        """Dispatch to the appropriate loader."""
        # Single directory path
        if isinstance(inputs, (str, os.PathLike)):
            p = Path(inputs)
            if p.is_dir():
                return self._load_directory(p)
            else:
                return [self._load_file(str(p))]

        # Sequence of arrays or paths
        if not isinstance(inputs, (list, tuple)):
            raise TypeError(
                f"inputs must be a path, list of paths, or list of arrays; "
                f"got {type(inputs)}"
            )

        if len(inputs) == 0:
            raise ValueError("inputs is empty")

        first = inputs[0]

        # List of arrays
        if isinstance(first, (np.ndarray, jax.Array)):
            return [self._ensure_jax(a) for a in inputs]

        # List of file paths
        return [self._load_file(str(f)) for f in inputs]

    def _load_directory(self, directory: Path) -> List[jax.Array]:
        """Load all ``.npy`` files from a directory, sorted."""
        files = sorted(directory.rglob("*.npy"))
        if not files:
            raise FileNotFoundError(
                f"No .npy files found in {directory}"
            )
        logger.info("Found %d .npy files in %s", len(files), directory)
        return [_load_npy(str(f)) for f in files]

    def _load_file(self, path: str) -> jax.Array:
        """Load a single file based on its extension."""
        ext = Path(path).suffix.lower()
        if ext == ".npy":
            return _load_npy(path)
        elif ext == ".mat":
            return _load_mat(path, data_field=self.data_field)
        elif ext == ".fif":
            return _load_fif(path, picks=self.picks)
        else:
            raise ValueError(f"Unsupported file extension: '{ext}' ({path})")

    @staticmethod
    def _ensure_jax(arr) -> jax.Array:
        """Convert an array-like to a JAX array with shape (T, C)."""
        x = jnp.asarray(arr)
        if x.ndim == 1:
            x = x[:, None]
        return x

    # -- preparation ---------------------------------------------------------

    def prepare(
        self,
        methods: Dict[str, dict],
    ) -> None:
        """Apply preparation steps to each dataset.

        Follows the osl-dynamics convention: *methods* is an ordered dict
        whose keys name the operations and whose values are keyword argument
        dicts for that operation.

        Supported method keys
        ---------------------
        ``"tde"``
            Time-delay embedding.  Kwargs: ``n_embeddings`` (int).
        ``"pca"``
            PCA dimensionality reduction.  Kwargs: ``n_pca_components`` (int).
        ``"tde_pca"``
            Combined TDE then PCA.  Kwargs: ``n_embeddings`` (int),
            ``n_pca_components`` (int).
        ``"standardize"``
            Z-score each channel.  No kwargs needed.
        ``"amplitude_envelope"``
            Hilbert amplitude envelope.  No kwargs needed.

        Parameters
        ----------
        methods : dict
            Ordered mapping of ``{method_name: {**kwargs}}``.

        Examples
        --------
        >>> data.prepare({
        ...     "tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
        ...     "standardize": {},
        ... })
        """
        prepared: List[jax.Array] = list(self._raw_data)

        for method_name, kwargs in methods.items():
            logger.info("Applying '%s' with %s", method_name, kwargs)
            prepared = [
                self._apply_method(x, method_name, kwargs) for x in prepared
            ]

        self._prepared_data = prepared
        shapes = [d.shape for d in prepared]
        logger.info("Preparation complete. Shapes: %s", shapes)

    @staticmethod
    def _apply_method(
        x: jax.Array,
        method_name: str,
        kwargs: dict,
    ) -> jax.Array:
        """Apply a single preparation method to one array."""

        if method_name == "tde":
            return prepare_tde(x, **kwargs)

        elif method_name == "pca":
            return prepare_pca(x, **kwargs)

        elif method_name == "tde_pca":
            n_embeddings = kwargs.get("n_embeddings", 15)
            n_pca_components = kwargs.get("n_pca_components", 80)
            x = prepare_tde(x, n_embeddings=n_embeddings)
            x = prepare_pca(x, n_pca_components=n_pca_components)
            return x

        elif method_name == "standardize":
            return _standardize(x)

        elif method_name == "amplitude_envelope":
            return _amplitude_envelope(x)

        else:
            raise ValueError(f"Unknown preparation method: '{method_name}'")

    # -- dunder helpers ------------------------------------------------------

    def __repr__(self) -> str:
        shapes = [d.shape for d in self._raw_data]
        return f"Data(n_subjects={self.n_subjects}, raw_shapes={shapes})"

    def __len__(self) -> int:
        return self.n_subjects

    def __getitem__(self, idx: int) -> jax.Array:
        """Index into prepared (or raw) data."""
        return self.prepared_data[idx]
