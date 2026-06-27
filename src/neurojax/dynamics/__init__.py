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
    smni: Ingber SMNI / Canonical Momenta Indicators / PATHINT re-exports from
        the smni-cmi peer package (optional; inert when not installed).

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

# SINDy/Koopman/windowed wrap jaxctrl; keep them optional so the contrastive
# (CEBRA) leg, which only needs equinox, imports without jaxctrl installed.
# SINDy / Koopman / windowed / CEBRA all live in jaxctrl; keep them optional so
# neurojax still imports in a lean env without jaxctrl installed.
try:
    from jaxctrl import (
        KoopmanEstimator,
        SINDyOptimizer,
        fourier_library,
        polynomial_library,
        svht_rank,
        svht_denoise,
        optimal_shrinkage_denoise,
        shrink_covariance,
        l1_statistical_dimension,
        donoho_tanner_threshold,
        donoho_tanner_regime,
        DonohoTannerRegime,
        CEBRA,
        ContrastiveEncoder,
        info_nce,
        DYSCO,
        LatentFlow,
        LinearLangevin,
        fit_linear_langevin,
        langevin_gradient_part,
        langevin_solenoidal_part,
        langevin_entropy_production,
        langevin_solenoidal_frequency,
        transition_flux,
        discrete_entropy_production,
    )
    from neurojax.dynamics.windowed import (
        windowed_sindy,
        windowed_dmd,
        windowed_signatures,
        WindowedSINDyResult,
        WindowedDMDResult,
        WindowedSignatureResult,
    )
    _HAS_JAXCTRL = True
except ImportError:  # jaxctrl not installed (e.g. lean GPU env)
    _HAS_JAXCTRL = False

# SMNI / CMI / PATHINT — Ingber's path-integral statistical mechanics, the
# canonical-momenta view of the Langevin/Fokker-Planck dynamics above.  Optional
# peer package (smni-cmi); keep inert when absent, like the jaxctrl block.
try:
    from neurojax.dynamics import smni as smni
    from neurojax.dynamics.smni import (
        DT,
        FS_HZ,
        SMNIDrift,
        canonical_momenta,
        drift,
        fit_linear_drift,
        momentum_magnitude,
        smni_log_likelihood,
        velocity,
        FitResult,
        MLEConfig,
        fit_mle,
        action,
        coherence,
        nonlinear,
        pathint,
    )
    _HAS_SMNI = True
except ImportError:  # smni-cmi not installed
    _HAS_SMNI = False

__all__ = []
if _HAS_JAXCTRL:
    __all__ += [
        "SINDyOptimizer",
        "KoopmanEstimator",
        "polynomial_library",
        "fourier_library",
        "svht_rank",
        "svht_denoise",
        "optimal_shrinkage_denoise",
        "shrink_covariance",
        "l1_statistical_dimension",
        "donoho_tanner_threshold",
        "donoho_tanner_regime",
        "DonohoTannerRegime",
        "CEBRA",
        "ContrastiveEncoder",
        "info_nce",
        "DYSCO",
        "LatentFlow",
        "LinearLangevin",
        "fit_linear_langevin",
        "langevin_gradient_part",
        "langevin_solenoidal_part",
        "langevin_entropy_production",
        "langevin_solenoidal_frequency",
        "transition_flux",
        "discrete_entropy_production",
        "windowed_sindy",
        "windowed_dmd",
        "windowed_signatures",
        "WindowedSINDyResult",
        "WindowedDMDResult",
        "WindowedSignatureResult",
    ]

if _HAS_SMNI:
    __all__ += [
        "smni",
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
