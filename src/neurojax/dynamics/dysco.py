"""DYSCO — governing equations from latent dynamics — re-exported from jaxctrl.

Implementation lives in :mod:`jaxctrl` (`_dysco`): a contrastive encoder + a
SINDy-parameterized latent flow trained JEPA-style, recovering the latent
trajectory AND its governing equation ż = Θ(z)·Ξ (Muratore & Mathis,
arXiv:2606.13260).  NeuroJAX re-exports it to apply to source-MEG latent
dynamics — e.g. testing whether the HMM+TINDA structured cycle is a continuous
limit-cycle attractor.  See ``docs/THREE_LEGS_NEURAL_DYNAMICS.md``.
"""

from jaxctrl import DYSCO, LatentFlow

__all__ = ["DYSCO", "LatentFlow"]
