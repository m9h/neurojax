"""Bayesian spectral decomposition: competing parametric spectral models under one evidence.

Three generative hypotheses for the same resting power spectrum, each a
differentiable JAX model of ``ln S(f)``:

* :class:`SpecParam` — the phenomenological aperiodic-plus-Gaussian-peaks
  parameterisation (Donoghue et al. 2020), with a fixed or knee aperiodic.
* :class:`DampedOscillators`, :class:`DampingDistribution` and :class:`DampedMixture` — Liley's
  hypothesis that the spectrum is a superposition of stochastically driven
  damped oscillators, the 1/f background arising from a distribution of
  damping rates (Muthukumaraswamy & Liley 2018; Liley & Muthukumaraswamy 2019).
* :class:`XiAlpha` — the Valdes-Sosa hypothesis of two separate processes,
  an aperiodic ξ Matérn process and an α frequency-shifted Matérn process
  (Pascual-Marqui, Valdes-Sosa & Alvarez-Amador 1988; ξ-α NET, Garcia Reyes
  et al. 2026).

Inversion is variational Laplace with Gaussian priors on unconstrained
parameters and a log-precision hyperparameter on the log-power residual,
the scheme of DCM (Friston et al. 2007) and of SPM's BSD toolbox (Medrano
et al. 2025).  :mod:`neurojax.bsd.compare` supplies the evaluation tools:
log Bayes factors and posterior model probabilities, fixed- and
random-effects Bayesian model selection with protected exceedance
probabilities (Stephan et al. 2009; Rigoux et al. 2014), Bayesian model
reduction (Friston et al. 2016) and Bayesian model averaging.

Everything here is pure JAX and differentiable; ``jax.grad`` flows through
the free energy.  Use float64 (``jax.config.update("jax_enable_x64", True)``)
for evidence computations: the Hessian determinant is the quantity model
comparison rests on and single precision is not enough for it.
"""

from neurojax.bsd.compare import (
    bayesian_model_average,
    bayesian_model_reduction,
    fixed_effects_bms,
    log_bayes_factors,
    posterior_model_probabilities,
    random_effects_bms,
)
from neurojax.bsd.inference import VLResult, variational_laplace
from neurojax.bsd.likelihood import gaussian_log_likelihood, whittle_log_likelihood
from neurojax.bsd.models import (
    DampedMixture,
    DampedOscillators,
    DampingDistribution,
    SpecParam,
    SpectralModel,
    XiAlpha,
)
from neurojax.bsd.simulate import psd_from_timeseries, simulate_log_psd

__all__ = [
    "SpectralModel", "SpecParam", "DampedOscillators", "DampingDistribution", "DampedMixture", "XiAlpha",
    "gaussian_log_likelihood", "whittle_log_likelihood",
    "VLResult", "variational_laplace",
    "log_bayes_factors", "posterior_model_probabilities", "fixed_effects_bms",
    "random_effects_bms", "bayesian_model_reduction", "bayesian_model_average",
    "simulate_log_psd", "psd_from_timeseries",
]
