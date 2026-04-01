"""Inverse modelling utilities for neurojax.

This sub-package provides tools for solving inverse problems on neural
data — estimating model parameters, source configurations, or latent
states from observed measurements (MEG/EEG/fMRI).

Planned capabilities:

* **Source localisation** — beamforming and minimum-norm estimates
  implemented as differentiable JAX transforms so that source
  reconstructions can be embedded inside end-to-end gradient pipelines.
* **Parameter estimation** — gradient-based or sampling-based inference
  of neural-mass / neural-field model parameters given time-series
  observations, complementing the forward simulation in
  :mod:`neurojax.models`.
* **Bayesian inversion** — variational and MCMC wrappers that leverage
  JAX's autodiff for efficient posterior computation (cf. Dynamic Causal
  Modelling; Friston et al., 2003).

This module is currently a scaffold; concrete implementations will be
added as the inverse-problem API stabilises.

See Also:
    neurojax.models: Forward neural-mass and neural-field models.
    neurojax.dynamics: Data-driven dynamics identification (SINDy, DMD).
    neurojax.source: Source-space analysis utilities.
"""
