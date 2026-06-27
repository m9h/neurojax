"""Bridge: NEOBA oscillatory brain-age features re-exported through neurojax.analysis.

NEOBA (Hu, Valdés-Sosa et al. 2025) is the interpretable oscillatory brain-age
baseline that OSL / osl-dynamics lacks; the differentiable implementation lives
in the peer package :mod:`neoba`, re-exported here as ``neurojax.analysis.neoba``.
This pins the re-export surface plus a minimal functional check (OSF extraction
on a tiny synthetic cohort -> feature matrix). Skipped when the optional peer
package is absent.
"""

import numpy as np
import pytest

pytest.importorskip("neoba")  # optional peer package; bridge inert without it

import neurojax.analysis.neoba as nb

_REEXPORTED = (
    "extract_features",
    "make_ridge_regressor",
    "make_neoba_regressor",
    "make_bias_corrected_regressor",
    "compute_odc",
    "compute_osf_table",
    "osf_spec",
    "SparseGroupLasso",
    "group_sizes",
    "welch_psd",
    "compute_cross_spectral",
    "N_OSF",
    "N_ODC",
    "CANONICAL_BANDS",
)


def test_neoba_symbols_reexported():
    for name in _REEXPORTED:
        assert hasattr(nb, name), f"neurojax.analysis.neoba is missing {name}"
        assert name in nb.__all__, f"{name} not advertised in __all__"


def test_osf_spec_is_consistent():
    # pure (no data): the OSF name/group vectors must be parallel and sized N_OSF.
    names, groups = nb.osf_spec("fixed")
    assert len(names) == len(groups) == nb.N_OSF


def test_extract_features_through_bridge():
    # tiny synthetic cohort: 2 subjects, each (n_epochs, n_channels, n_samples)
    rng = np.random.default_rng(0)
    sfreq = 200.0
    t = np.arange(400) / sfreq
    recordings = []
    for _ in range(2):
        # broadband (1/f-ish) noise + a 10 Hz bump so specparam has a real peak
        base = rng.standard_normal((2, 3, 400))
        base += np.sin(2 * np.pi * 10 * t)[None, None, :]
        recordings.append(base)
    X, names = nb.extract_features(
        recordings, sfreq, include_osf=True, include_odc=False
    )
    assert X.shape[0] == 2
    assert X.shape[1] == len(names)
    assert np.all(np.isfinite(X))
