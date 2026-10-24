"""NEOBA oscillatory brain-age features — re-exported from the neoba peer package.

NEOBA (Hu, Valdés-Sosa et al. 2025, *Front. Aging Neurosci.*) describes each
recording by interpretable *oscillatory* features and predicts brain age from
them — a capability OSL / osl-dynamics does not provide. The differentiable
clean-room implementation lives in the peer package :mod:`neoba` (github.com/
m9h/neoba); NeuroJAX re-exports it so brain-age featurisation sits alongside the
spectral / connectivity / state-network analyses in :mod:`neurojax.analysis`.

Two feature families (the canonical NEOBA spec uses both):

* **OSF — Oscillatory Spectral Features** (:func:`compute_osf_table`,
  :func:`extract_features` with ``include_osf=True``): per-channel aperiodic
  (specparam offset + exponent), periodic (dominant-peak centre/power/bandwidth),
  relative band power, and the NEOBA power ratios.
* **ODC — Oscillatory Dynamic Connectivity** (:func:`compute_odc`,
  ``include_odc=True``): montage-independent regression of each OSF on the others
  across electrodes, with the paper's Σ|PCC| sparse-group-lasso group weights
  (:class:`SparseGroupLasso`).

Heads: :func:`make_ridge_regressor` (recommended at small N) and the paper's
:func:`make_neoba_regressor` (FCNN); :func:`make_bias_corrected_regressor` adds
Cole-style age-bias recalibration. On LEMON (n=120, 10-fold) the OSF/ODC +
RidgeCV path reaches MAE ~10.4 yr / R² ~0.54.

Example::

    from neurojax.analysis.neoba import extract_features, make_ridge_regressor
    X, names = extract_features(recordings, sfreq, include_osf=True, include_odc=True)
    reg = make_ridge_regressor().fit(X_train, age_train)

See Also:
    neurojax.analysis.spectral: the multitaper / Welch spectra NEOBA's OSFs build
        on; :func:`welch_psd` here is the package's lightweight PSD helper.
"""

from neoba import (
    CANONICAL_BANDS,
    FIT_RANGE,
    N_ODC,
    N_OSF,
    N_XSPEC,
    OSF_GROUPS,
    OSF_NAMES,
    RATIO_PAIRS,
    BiasCorrectedRegressor,
    SparseGroupLasso,
    compute_cross_spectral,
    compute_odc,
    compute_osf_table,
    compute_osf_vector,
    extract_features,
    fit_spectra,
    group_sizes,
    make_bias_corrected_regressor,
    make_neoba_regressor,
    make_ridge_regressor,
    osf_spec,
    welch_psd,
)

__all__ = [
    "CANONICAL_BANDS",
    "RATIO_PAIRS",
    "FIT_RANGE",
    "OSF_NAMES",
    "OSF_GROUPS",
    "N_OSF",
    "N_ODC",
    "N_XSPEC",
    "compute_osf_table",
    "compute_osf_vector",
    "osf_spec",
    "compute_odc",
    "SparseGroupLasso",
    "group_sizes",
    "fit_spectra",
    "welch_psd",
    "extract_features",
    "make_neoba_regressor",
    "make_ridge_regressor",
    "BiasCorrectedRegressor",
    "make_bias_corrected_regressor",
    "compute_cross_spectral",
]
