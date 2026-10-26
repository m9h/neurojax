"""Inverse modelling: Bayesian inversion of generative models of M/EEG.

* :mod:`neurojax.inverse.csd_dcm` — spectral DCM (DCM for cross-spectral
  density): a Jansen-Rit network linearised at its fixed point, mapped
  through a leadfield to a sensor CSD and inverted by variational Laplace
  with a free-energy bound for model comparison.

* :mod:`neurojax.inverse.peb` — parametric empirical Bayes over first-level
  posteriors (group / study / drug effects on model parameters) with
  Bayesian model reduction.

Source localisation lives in :mod:`neurojax.source`; data-driven dynamics
identification in :mod:`neurojax.dynamics`.
"""

from neurojax.inverse.csd_dcm import (
    CMCSpec,
    CMCTheta,
    JansenRitSpec,
    JRTheta,
    SpectralDCM,
    SpectralDCMResult,
    VLResult,
    fixed_point,
    jr_fixed_point,
    jr_linearise,
    jr_vector_field,
    linearise,
    model_posterior,
    transfer_function,
    variational_laplace,
)
from neurojax.inverse.peb import PEBResult, bmr_effects, log_evidence_reduce, peb

__all__ = [
    "CMCSpec",
    "CMCTheta",
    "JRTheta",
    "JansenRitSpec",
    "SpectralDCM",
    "SpectralDCMResult",
    "VLResult",
    "fixed_point",
    "jr_fixed_point",
    "jr_linearise",
    "jr_vector_field",
    "linearise",
    "model_posterior",
    "transfer_function",
    "variational_laplace",
    "PEBResult",
    "bmr_effects",
    "log_evidence_reduce",
    "peb",
]
