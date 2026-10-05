"""Portable timestamped scalar observations (NPZ, no pickle).

Time is seconds in one explicitly documented recording clock. Each stream
keeps its own sampling grid; validity is never inferred from zero padding.
The file contract is consumed by vpjax without depending on neurojax.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = "neurophys-observations/1"


@dataclass
class Observation:
    time_s: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    unit: str
    metadata: dict

    def __post_init__(self):
        self.time_s = np.asarray(self.time_s, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        self.valid = np.asarray(self.valid, dtype=bool)
        if (
            self.time_s.ndim != 1
            or self.values.shape != self.time_s.shape
            or self.valid.shape != self.time_s.shape
        ):
            raise ValueError("time_s, values and valid must be matching 1D arrays")
        if not np.isfinite(self.time_s).all() or np.any(np.diff(self.time_s) <= 0):
            raise ValueError("timestamps must be finite and strictly increasing")
        self.valid = self.valid & np.isfinite(self.values)
        if not self.unit:
            raise ValueError("an explicit unit is required")


def save_observations(path: str | Path, streams: dict[str, Observation]):
    """Write a self-contained file, preserving masks, units and provenance."""
    arrays = {}
    manifest = {"schema": SCHEMA, "streams": []}
    for i, (name, stream) in enumerate(streams.items()):
        key = f"s{i}"
        manifest["streams"].append(
            {"name": name, "key": key, "unit": stream.unit, "metadata": stream.metadata}
        )
        arrays[f"{key}_time_s"] = stream.time_s
        arrays[f"{key}_values"] = stream.values
        arrays[f"{key}_valid"] = stream.valid
    arrays["manifest"] = np.array(json.dumps(manifest, allow_nan=False))
    with Path(path).open("wb") as file:
        np.savez_compressed(file, **arrays)


def load_observations(path: str | Path) -> dict[str, Observation]:
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        if manifest["schema"] != SCHEMA:
            raise ValueError("unsupported observation schema")
        return {
            item["name"]: Observation(
                data[item["key"] + "_time_s"],
                data[item["key"] + "_values"],
                data[item["key"] + "_valid"],
                item["unit"],
                item["metadata"],
            )
            for item in manifest["streams"]
        }
