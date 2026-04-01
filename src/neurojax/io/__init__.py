"""I/O helpers — load, save, and convert neuroimaging data.

This sub-package provides lightweight converters and readers that sit
between vendor / standard file formats and the JAX array world used
throughout neurojax.

Modules:
    bridge: Minimal converters between MNE-Python objects and JAX
        arrays, supporting round-trip workflows where preprocessing
        is done in MNE and modelling in JAX.

See Also:
    neurojax.utils.bridge: Extended MNE-to-JAX converter with channel
        picking and sample-range slicing.
    neurojax.analysis.mrs_io: Siemens TWIX reader for MRS data.
"""
