"""RegionalParameterSpace — per-region heterogeneous parameter support.

Maps a mix of global parameters (one value for all brain regions) and
regional parameters (one value per region) to/from a flat 1-D array
suitable for black-box and gradient-based optimizers.

Layout of the flat array:
    [global_0, global_1, ..., regional_0_r0, regional_0_r1, ..., regional_0_rN,
     regional_1_r0, ..., regional_1_rN, ...]

This is a standalone mapping module. It does not run simulations —
a future integration step will wire it into VbjaxFitnessAdapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RegionalParameterSpace:
    """Defines which brain-model parameters are global vs per-region.

    Parameters
    ----------
    global_params : dict[str, tuple[float, float]]
        Parameter names mapped to (lower_bound, upper_bound).
        One scalar value shared across all regions.
    regional_params : dict[str, tuple[float, float]]
        Parameter names mapped to (lower_bound, upper_bound).
        One value per region (N values total per parameter).
    n_regions : int
        Number of brain regions in the network.
    """

    global_params: dict[str, tuple[float, float]]
    regional_params: dict[str, tuple[float, float]]
    n_regions: int

    # Deterministic ordering — set in __post_init__
    _global_names: list[str] = field(default_factory=list, repr=False, init=False)
    _regional_names: list[str] = field(default_factory=list, repr=False, init=False)

    def __post_init__(self) -> None:
        if self.n_regions < 1:
            raise ValueError("n_regions must be >= 1")
        if not self.global_params and not self.regional_params:
            raise ValueError("Must define at least one parameter (global or regional)")
        # Fix ordering for reproducible flat-array layout
        self._global_names = sorted(self.global_params.keys())
        self._regional_names = sorted(self.regional_params.keys())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """Total number of scalar values in the flat array."""
        return len(self._global_names) + len(self._regional_names) * self.n_regions

    @property
    def param_names(self) -> list[str]:
        """Human-readable names matching each index in the flat array."""
        names: list[str] = list(self._global_names)
        for rp in self._regional_names:
            names.extend(f"{rp}_{i}" for i in range(self.n_regions))
        return names

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bound arrays matching the flat-array layout.

        Returns
        -------
        (lower, upper) : tuple of 1-D float64 arrays, each of length n_params.
        """
        lower: list[float] = []
        upper: list[float] = []

        for name in self._global_names:
            lo, hi = self.global_params[name]
            lower.append(lo)
            upper.append(hi)

        for name in self._regional_names:
            lo, hi = self.regional_params[name]
            lower.extend([lo] * self.n_regions)
            upper.extend([hi] * self.n_regions)

        return np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)

    # ------------------------------------------------------------------
    # Flat-array <-> parameter dict
    # ------------------------------------------------------------------

    def to_flat_array(self, params: dict[str, np.ndarray | float]) -> np.ndarray:
        """Pack a parameter dict into a 1-D array.

        Parameters
        ----------
        params : dict
            Keys are parameter names. Global params have scalar values;
            regional params have 1-D arrays of length ``n_regions``.

        Returns
        -------
        1-D float64 array of length ``n_params``.
        """
        parts: list[np.ndarray] = []

        for name in self._global_names:
            parts.append(np.atleast_1d(np.asarray(params[name], dtype=np.float64)))

        for name in self._regional_names:
            arr = np.asarray(params[name], dtype=np.float64)
            if arr.shape != (self.n_regions,):
                raise ValueError(
                    f"Regional parameter '{name}' must have length {self.n_regions}, "
                    f"got shape {arr.shape}"
                )
            parts.append(arr)

        return np.concatenate(parts)

    def from_flat_array(self, flat: np.ndarray) -> dict[str, np.ndarray]:
        """Unpack a 1-D array into a parameter dict.

        Parameters
        ----------
        flat : 1-D array of length ``n_params``.

        Returns
        -------
        dict with scalar values for global params and 1-D arrays for
        regional params.

        Raises
        ------
        ValueError
            If ``flat`` has the wrong length.
        """
        flat = np.asarray(flat, dtype=np.float64)
        if flat.shape != (self.n_params,):
            raise ValueError(
                f"Expected flat array of length {self.n_params}, got {flat.shape}"
            )

        result: dict[str, np.ndarray] = {}
        idx = 0

        for name in self._global_names:
            result[name] = flat[idx]
            idx += 1

        for name in self._regional_names:
            result[name] = flat[idx : idx + self.n_regions].copy()
            idx += self.n_regions

        return result

    def default_flat_array(self) -> np.ndarray:
        """Create a flat array with each value at the midpoint of its bounds.

        Useful for initializing optimizers.
        """
        lower, upper = self.bounds
        return (lower + upper) / 2.0
