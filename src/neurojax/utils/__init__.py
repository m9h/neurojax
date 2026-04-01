"""Shared utility functions for the neurojax package.

This sub-package collects helpers that are used across multiple neurojax
modules but are not specific to any single analysis domain.

Modules:
    bridge: Convert between MNE-Python data structures and JAX arrays,
        bridging the MNE ecosystem (I/O, preprocessing, visualisation)
        with JAX-based computation (GPU-accelerated modelling, autodiff).

See Also:
    neurojax.io.bridge: Lower-level MNE <-> JAX converters (round-trip).
"""
