"""Inverse modelling: Bayesian inversion of generative models of M/EEG.

* :mod:`neurojax.inverse.csd_dcm` — spectral DCM (DCM for cross-spectral
  density): a Jansen-Rit network linearised at its fixed point, mapped
  through a leadfield to a sensor CSD and inverted by variational Laplace
  with a free-energy bound for model comparison.

Source localisation lives in :mod:`neurojax.source`; data-driven dynamics
identification in :mod:`neurojax.dynamics`.
"""

from neurojax.inverse.csd_dcm import (
    JansenRitSpec,
    JRTheta,
    SpectralDCM,
    SpectralDCMResult,
    VLResult,
    jr_fixed_point,
    jr_linearise,
    jr_vector_field,
    model_posterior,
    transfer_function,
    variational_laplace,
)

__all__ = [
    "JRTheta",
    "JansenRitSpec",
    "SpectralDCM",
    "SpectralDCMResult",
    "VLResult",
    "jr_fixed_point",
    "jr_linearise",
    "jr_vector_field",
    "model_posterior",
    "transfer_function",
    "variational_laplace",
]
