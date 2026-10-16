"""Bridge: EEG/MEG foundation-model adapters + FMScope identity-trap diagnostics.

Re-exports the emeg-fm peer package through ``neurojax.bench.foundation_models``:
the HuggingFace EEG-FM adapters (REVE/LaBraM via make_hf_encoder), the FMScope
subject-axis LEACE erasure / identity-trap audit, the MOABB cohort builder, and
the linear SVM probe. These are capabilities OSL has no equivalent of. Torch is
imported lazily inside emeg-fm, so the bridge stays import-light; this test only
exercises the numpy/sklearn-clean LEACE path. Skipped when the peer package is
absent.
"""

import numpy as np
import pytest

pytest.importorskip("emeg_fm")
pytest.importorskip("fmscope")

import fmscope.diagnostics.erasure as _erasure

import neurojax.bench.foundation_models as fm

_REEXPORTED = (
    "adapters",
    "moabb_cohort",
    "erasure",
    "audit",
    "svm_probe",
    "make_hf_encoder",
    "build_moabb_cohort",
    "subject_axis_erasure",
    "audit_cell",
    "whiten",
)


def test_fm_symbols_reexported():
    for name in _REEXPORTED:
        assert hasattr(fm, name), f"neurojax.bench.foundation_models missing {name}"
        assert name in fm.__all__, f"{name} not advertised in __all__"


def test_reexport_is_identity():
    # the bridged objects must BE the emeg-fm objects, not copies
    assert fm.whiten is _erasure.whiten
    assert fm.erasure is _erasure


def test_leace_whitening_through_bridge():
    # whiten returns Σ^{-1/2}; applying it must whiten the data to ~identity cov.
    rng = np.random.default_rng(0)
    A = rng.standard_normal((200, 5))
    X = A @ rng.standard_normal((5, 5))  # correlated features
    mu, Xc, W, W_plus, cond = fm.whiten(X, shrinkage=False)
    white = Xc @ W
    cov = white.T @ white / len(white)
    assert np.allclose(cov, np.eye(5), atol=1e-6)
