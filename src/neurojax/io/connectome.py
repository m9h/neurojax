"""BIDSConnectomeLoader — load structural connectivity from BIDS derivatives.

Loads connectivity matrices (Cmat) from QSIRecon and optionally conduction
delay matrices (Dmat) from dmipy-jax AxonDelayMapper. Atlas-agnostic: works
with any parcellation fMRIPrep/QSIRecon produces (Schaefer, DKT, Gordon, etc.).

Expected BIDS derivatives layout::

    derivatives/
    ├── qsirecon/
    │   └── sub-01/dwi/
    │       ├── sub-01_..._atlas-Schaefer200_desc-sift2_connectivity.csv
    │       └── sub-01_..._atlas-Schaefer200_desc-meanlength_connectivity.csv
    └── dmipy-jax/          (optional)
        └── sub-01/dwi/
            └── sub-01_..._atlas-Schaefer200_desc-axondelay_connectivity.csv

Usage::

    loader = BIDSConnectomeLoader("derivatives", "01", atlas="Schaefer200")
    data = loader.load()
    # data.weights → (N, N) JAX array, data.delays → (N, N) or None
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np


@dataclass
class ConnectomeData:
    """Structural connectivity data for whole-brain modeling.

    Attributes
    ----------
    weights : jnp.ndarray
        Structural connectivity matrix of shape (n_regions, n_regions).
        Typically streamline counts from QSIRecon, normalized to [0, 1].
    delays : jnp.ndarray or None
        Conduction delay matrix of shape (n_regions, n_regions) in ms.
        None if delay data is unavailable.
    fiber_lengths : jnp.ndarray or None
        Fiber length matrix of shape (n_regions, n_regions) in mm.
        None if length data is unavailable.
    atlas : str
        Atlas name (e.g., "Schaefer200", "DKT", "Gordon").
    n_regions : int
        Number of brain regions.
    region_labels : list[str] or None
        Region label names if available from atlas metadata.
    subject : str
        BIDS subject ID (without "sub-" prefix).
    """

    weights: jnp.ndarray
    delays: Optional[jnp.ndarray]
    fiber_lengths: Optional[jnp.ndarray]
    atlas: str
    n_regions: int
    region_labels: Optional[list[str]]
    subject: str

    def for_adapter(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (weights, delays) ready for VbjaxFitnessAdapter.

        If delays are unavailable, estimates them from fiber lengths
        assuming 7 m/s conduction velocity (cortical default).
        If neither delays nor fiber lengths are available, returns
        zero delays.

        Returns
        -------
        weights : jnp.ndarray, shape (n_regions, n_regions)
        delays : jnp.ndarray, shape (n_regions, n_regions) in ms
        """
        if self.delays is not None:
            return self.weights, self.delays
        if self.fiber_lengths is not None:
            # Default cortical conduction velocity: ~7 m/s = 7 mm/ms
            velocity_mm_per_ms = 7.0
            delays = self.fiber_lengths / velocity_mm_per_ms
            return self.weights, delays
        return self.weights, jnp.zeros_like(self.weights)


class BIDSConnectomeLoader:
    """Load structural connectivity from BIDS derivatives.

    Discovers and loads connectivity matrices produced by QSIRecon
    (structural connectivity, fiber lengths) and optionally dmipy-jax
    (axon-caliber-derived conduction delays).

    Parameters
    ----------
    derivatives_dir : str or Path
        Path to the BIDS derivatives directory containing ``qsirecon/``
        and optionally ``dmipy-jax/`` subdirectories.
    subject : str
        Subject ID (without "sub-" prefix), e.g., "01" or "NDARAB123".
    atlas : str, optional
        Atlas name to load. Must match the ``atlas-<name>`` BIDS entity
        in the filenames. Default: "Schaefer200".
    session : str or None, optional
        Session ID (without "ses-" prefix) if the dataset uses sessions.
    normalize_weights : bool, optional
        If True, normalize the weight matrix by its maximum value.
        Default: True.

    Examples
    --------
    >>> loader = BIDSConnectomeLoader("derivatives", "01")
    >>> loader.available_atlases()
    ['DKT', 'Gordon', 'Schaefer200', 'Schaefer400']
    >>> data = loader.load(atlas="Schaefer200")
    >>> weights, delays = data.for_adapter()
    """

    def __init__(
        self,
        derivatives_dir: str | Path,
        subject: str,
        atlas: str = "Schaefer200",
        session: str | None = None,
        normalize_weights: bool = True,
    ):
        self.derivatives_dir = Path(derivatives_dir)
        self.subject = subject
        self.atlas = atlas
        self.session = session
        self.normalize_weights = normalize_weights

        # Build subject paths
        sub_dir = f"sub-{subject}"
        ses_dir = f"ses-{session}" if session else None

        self._qsirecon_dwi = self._build_subdir("qsirecon", sub_dir, ses_dir, "dwi")
        self._dmipy_dwi = self._build_subdir("dmipy-jax", sub_dir, ses_dir, "dwi")

    def _build_subdir(
        self, pipeline: str, sub_dir: str, ses_dir: str | None, modality: str
    ) -> Path:
        """Build the path to a BIDS derivatives modality directory."""
        base = self.derivatives_dir / pipeline / sub_dir
        if ses_dir:
            base = base / ses_dir
        return base / modality

    def available_atlases(self) -> list[str]:
        """Discover which atlases have connectivity matrices.

        Scans the QSIRecon derivatives directory for files matching
        the ``atlas-<name>`` BIDS entity pattern.

        Returns
        -------
        list of str
            Sorted list of atlas names found.
        """
        atlases = set()
        if self._qsirecon_dwi.exists():
            for f in self._qsirecon_dwi.glob("*_connectivity.*"):
                name = f.name
                if "atlas-" in name:
                    # Extract atlas name from BIDS entity
                    start = name.index("atlas-") + 6
                    rest = name[start:]
                    # Atlas name ends at next underscore or file extension
                    end = len(rest)
                    for sep in ("_", "."):
                        idx = rest.find(sep)
                        if idx != -1:
                            end = min(end, idx)
                    atlases.add(rest[:end])
        return sorted(atlases)

    def _find_connectivity_file(
        self,
        directory: Path,
        atlas: str,
        desc: str,
        extensions: tuple[str, ...] = (".csv", ".tsv", ".txt", ".npy"),
    ) -> Path | None:
        """Find a connectivity matrix file matching atlas and desc entities."""
        if not directory.exists():
            return None
        for ext in extensions:
            pattern = f"*atlas-{atlas}*desc-{desc}*connectivity{ext}"
            matches = list(directory.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _load_matrix(self, path: Path) -> np.ndarray:
        """Load a connectivity matrix from file."""
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.load(path)
        elif suffix in (".csv", ".tsv", ".txt"):
            delimiter = "\t" if suffix == ".tsv" else ","
            try:
                return np.loadtxt(path, delimiter=delimiter)
            except ValueError:
                # May have a header row — skip it
                return np.loadtxt(path, delimiter=delimiter, skiprows=1)
        else:
            raise ValueError(f"Unsupported connectivity file format: {suffix}")

    def _load_labels(self, atlas: str) -> list[str] | None:
        """Try to load region labels for the atlas.

        Looks for a TSV/CSV file with atlas labels in the QSIRecon
        derivatives or a standard atlas lookup.
        """
        if not self._qsirecon_dwi.exists():
            return None
        for ext in (".tsv", ".csv", ".txt"):
            pattern = f"*atlas-{atlas}*labels{ext}"
            matches = list(self._qsirecon_dwi.glob(pattern))
            if matches:
                try:
                    with open(matches[0]) as f:
                        lines = f.read().strip().split("\n")
                    # Skip header if present
                    if lines and any(
                        h in lines[0].lower() for h in ("label", "name", "region")
                    ):
                        lines = lines[1:]
                    return [line.strip().split("\t")[-1].split(",")[-1] for line in lines]
                except Exception:
                    return None
        return None

    def load(self, atlas: str | None = None) -> ConnectomeData:
        """Load connectivity data for a given atlas.

        Parameters
        ----------
        atlas : str or None
            Atlas to load. If None, uses the default atlas from __init__.

        Returns
        -------
        ConnectomeData
            Structural connectivity with weights, optional delays/lengths.

        Raises
        ------
        FileNotFoundError
            If no connectivity matrix found for the specified atlas.
        """
        atlas = atlas or self.atlas

        # Load structural connectivity (weights)
        # Try common QSIRecon desc values: sift2, siftcount, count
        weights_path = None
        for desc in ("sift2", "siftcount", "count"):
            weights_path = self._find_connectivity_file(
                self._qsirecon_dwi, atlas, desc
            )
            if weights_path is not None:
                break

        if weights_path is None:
            raise FileNotFoundError(
                f"No connectivity matrix found for atlas={atlas} in "
                f"{self._qsirecon_dwi}. "
                f"Available atlases: {self.available_atlases()}"
            )

        weights = self._load_matrix(weights_path)
        if self.normalize_weights and weights.max() > 0:
            weights = weights / weights.max()

        # Load fiber lengths (optional)
        fiber_lengths = None
        lengths_path = self._find_connectivity_file(
            self._qsirecon_dwi, atlas, "meanlength"
        )
        if lengths_path is not None:
            fiber_lengths = self._load_matrix(lengths_path)

        # Load conduction delays from dmipy-jax (optional)
        delays = None
        delays_path = self._find_connectivity_file(
            self._dmipy_dwi, atlas, "axondelay"
        )
        if delays_path is not None:
            delays = self._load_matrix(delays_path)

        # Load region labels (optional)
        labels = self._load_labels(atlas)

        n_regions = weights.shape[0]

        return ConnectomeData(
            weights=jnp.array(weights, dtype=jnp.float32),
            delays=jnp.array(delays, dtype=jnp.float32) if delays is not None else None,
            fiber_lengths=jnp.array(fiber_lengths, dtype=jnp.float32)
                if fiber_lengths is not None else None,
            atlas=atlas,
            n_regions=n_regions,
            region_labels=labels,
            subject=self.subject,
        )

    def load_group(
        self,
        subjects: list[str],
        atlas: str | None = None,
    ) -> list[ConnectomeData]:
        """Load connectivity for multiple subjects.

        Parameters
        ----------
        subjects : list of str
            Subject IDs (without "sub-" prefix).
        atlas : str or None
            Atlas to use. If None, uses default.

        Returns
        -------
        list of ConnectomeData
            One per subject. Subjects with missing data are skipped
            with a warning.
        """
        results = []
        atlas = atlas or self.atlas
        for sub in subjects:
            loader = BIDSConnectomeLoader(
                self.derivatives_dir,
                sub,
                atlas=atlas,
                session=self.session,
                normalize_weights=self.normalize_weights,
            )
            try:
                results.append(loader.load(atlas))
            except FileNotFoundError as e:
                import warnings
                warnings.warn(f"Skipping sub-{sub}: {e}", stacklevel=2)
        return results
