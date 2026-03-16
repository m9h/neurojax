"""Dynamical systems identification — thin wrappers around jaxctrl.

The core SINDy and Koopman algorithms live in ``jaxctrl._sysid``.
This module re-exports them and provides neuroscience-specific helpers.
"""

from jaxctrl import (
    KoopmanEstimator,
    SINDyOptimizer,
    fourier_library,
    polynomial_library,
)

__all__ = [
    "SINDyOptimizer",
    "KoopmanEstimator",
    "polynomial_library",
    "fourier_library",
]
