"""SMNI / Canonical Momenta Indicators / PATHINT — re-exported from smni-cmi.

Lester Ingber's Statistical Mechanics of Neocortical Interactions (SMNI; Ingber
1997, *Phys. Rev. E* 55:4578) casts neocortical field dynamics as a short-time
path integral.  The differentiable JAX implementation lives in the peer package
:mod:`smni_cmi`; NeuroJAX re-exports it so SMNI sits alongside the Langevin,
SINDy, Koopman and DYSCO members of the dynamics hub.

Why it belongs here — it is the *canonical-momenta / path-integral* view of the
same Fokker-Planck dynamics that :mod:`neurojax.dynamics.langevin` estimates
data-drivenly:

* **Canonical Momenta Indicators (CMI)** — ``Π = Σ⁻¹(Ṁ − g(M))``, the conjugate
  momentum to the SMNI drift ``g(M)``.  The Langevin drift split ``(Γ+Q)∇log p``
  (Friston) and Ingber's ``g(M)`` are the same object; CMI is its Legendre
  conjugate (:func:`canonical_momenta`, :func:`momentum_magnitude`).
* **PATHINT** (:mod:`pathint`) — deterministic propagation of a probability
  density by folding the short-time kernel: the path-integral solution of the
  Fokker-Planck equation that ``langevin`` characterises locally.
* **action / nonlinear / coherence** — the SMNI Lagrangian with CMI as ``∂L/∂q̇``
  by autodiff (:mod:`action`), the nonlinear (tanh) drift fit
  (:mod:`nonlinear`), and density-matrix coherence (:mod:`coherence`).

Typical use: fit the drift on source-MEG / connectome-harmonic latent
trajectories and read off CMI as a physics-grounded complement to the
data-driven estimators in this package::

    from neurojax.dynamics import fit_linear_drift, canonical_momenta
    params = fit_linear_drift(M)            # M: (trials, channels, time)
    cmi = canonical_momenta(M, params)      # Π = Σ⁻¹(Ṁ − g(M))

See Also:
    neurojax.dynamics.langevin: data-driven drift+diffusion split with entropy
        production — the empirical counterpart of the SMNI drift here.
"""

from smni_cmi.model import (
    DT,
    FS_HZ,
    SMNIDrift,
    canonical_momenta,
    drift,
    fit_linear_drift,
    momentum_magnitude,
    smni_log_likelihood,
    velocity,
)
from smni_cmi.fitting import FitResult, MLEConfig, fit_mle
from smni_cmi import action, coherence, nonlinear, pathint

__all__ = [
    "DT",
    "FS_HZ",
    "SMNIDrift",
    "canonical_momenta",
    "drift",
    "fit_linear_drift",
    "momentum_magnitude",
    "smni_log_likelihood",
    "velocity",
    "FitResult",
    "MLEConfig",
    "fit_mle",
    "action",
    "coherence",
    "nonlinear",
    "pathint",
]
