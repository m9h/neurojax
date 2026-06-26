"""CEBRA-style contrastive latent embedding — re-exported from jaxctrl.

The implementation lives in :mod:`jaxctrl` (general differentiable system
identification, alongside SINDy/Koopman); NeuroJAX re-exports it here so it can be
applied to neuroimaging latent dynamics — e.g. embedding source-reconstructed MEG
and checking whether the HMM+TINDA structured cycle appears as a ring / limit
cycle.  See ``docs/THREE_LEGS_NEURAL_DYNAMICS.md``.

This mirrors :mod:`neurojax.dynamics.sindy`, which re-exports jaxctrl's SINDy.
"""

from jaxctrl import CEBRA, ContrastiveEncoder, info_nce

__all__ = ["CEBRA", "ContrastiveEncoder", "info_nce"]
