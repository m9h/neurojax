"""Dynamical systems identification for neural time-series.

This sub-package provides data-driven methods for characterising the
dynamics of brain signals (MEG, EEG, LFP).  The core algorithms are
thin wrappers around the ``jaxctrl`` library's system-identification
routines, augmented with neuroscience-specific windowed analyses.

Methods:
    * **SINDy** (:class:`SINDyOptimizer`, :func:`polynomial_library`,
      :func:`fourier_library`) — Sparse Identification of Nonlinear
      Dynamics (Brunton et al., 2016).  Discovers parsimonious ODE
      models from time-series data.
    * **Koopman / DMD** (:class:`KoopmanEstimator`) — Dynamic Mode
      Decomposition for linear approximation of nonlinear dynamics
      (Schmid, 2010; Brunton et al., 2021).
    * **Windowed analysis** (:func:`windowed_sindy`,
      :func:`windowed_dmd`, :func:`windowed_signatures`) — sliding-window
      wrappers that track how dynamical features (Jacobian eigenvalues,
      DMD frequencies, log-signature geometry) evolve over time.  Change
      points in these features can be compared against HMM / DyNeMo
      state transitions.

Modules:
    sindy: SINDy re-exports from jaxctrl.
    koopman: Koopman/DMD re-exports from jaxctrl.
    windowed: Windowed systems-identification for MEG dynamics comparison.

References:
    Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016).
        Discovering governing equations from data by sparse
        identification of nonlinear dynamical systems.
        *PNAS*, 113(15), 3932-3937.
    Schmid, P. J. (2010). Dynamic mode decomposition of numerical and
        experimental data. *J. Fluid Mech.*, 656, 5-28.

See Also:
    neurojax.models: Forward neural-mass and neural-field models whose
        parameters can be estimated from the dynamics identified here.
    neurojax.analysis.rough: Log-signature and rough-path utilities
        used by :func:`windowed_signatures`.
"""

from jaxctrl import (
    KoopmanEstimator,
    SINDyOptimizer,
    fourier_library,
    polynomial_library,
)

from neurojax.dynamics.windowed import (
    windowed_sindy,
    windowed_dmd,
    windowed_signatures,
    WindowedSINDyResult,
    WindowedDMDResult,
    WindowedSignatureResult,
)

__all__ = [
    "SINDyOptimizer",
    "KoopmanEstimator",
    "polynomial_library",
    "fourier_library",
    "windowed_sindy",
    "windowed_dmd",
    "windowed_signatures",
    "WindowedSINDyResult",
    "WindowedDMDResult",
    "WindowedSignatureResult",
]
