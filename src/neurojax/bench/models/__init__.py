"""Neural mass models for whole-brain simulation benchmarks."""

from neurojax.bench.models.rww import (
    RWWTheta,
    rww_default_theta,
    rww_dfun,
    rww_transfer_E,
    rww_transfer_I,
)

__all__ = [
    "RWWTheta",
    "rww_default_theta",
    "rww_dfun",
    "rww_transfer_E",
    "rww_transfer_I",
]
