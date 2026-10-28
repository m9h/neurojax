"""Bridge: Ingber SMNI / CMI / PATHINT physics re-exported through neurojax.dynamics.

``smni-cmi`` (Lester Ingber's Statistical Mechanics of Neocortical Interactions)
is the path-integral / canonical-momenta member of the Fokker-Planck family that
``neurojax.dynamics`` already hosts via the Langevin re-exports — CMI is the
conjugate momentum to the Langevin drift, and PATHINT folds the same short-time
kernel.  This test pins the re-export surface plus a minimal functional
round-trip (fit drift -> canonical momenta) so the hub stays honest.

The bridge is a no-op when the optional peer package is absent (mirrors the
jaxctrl-optional dynamics re-exports), so the whole module is skipped then.
"""

import pytest

pytest.importorskip("smni_cmi")  # optional peer package; bridge inert without it

import jax
import jax.numpy as jnp

import neurojax.dynamics as nd

_REEXPORTED = (
    "canonical_momenta",
    "fit_linear_drift",
    "drift",
    "velocity",
    "momentum_magnitude",
    "smni_log_likelihood",
    "SMNIDrift",
    "fit_mle",
    "action",
    "pathint",
    "nonlinear",
    "coherence",
)


def test_smni_symbols_reexported():
    for name in _REEXPORTED:
        assert hasattr(nd, name), f"neurojax.dynamics is missing {name}"
        assert name in nd.__all__, f"{name} not advertised in __all__"


def test_smni_submodule_aliases_top_level():
    from neurojax.dynamics import smni

    assert smni.canonical_momenta is nd.canonical_momenta
    assert smni.fit_linear_drift is nd.fit_linear_drift


def test_cmi_roundtrip_through_bridge():
    # tiny synthetic trajectory: trials x channels x time
    key = jax.random.PRNGKey(0)
    M = jax.random.normal(key, (4, 6, 64))
    params = nd.fit_linear_drift(M)
    cmi = nd.canonical_momenta(M, params)
    assert cmi.shape == M.shape
    assert jnp.all(jnp.isfinite(cmi))
    assert jnp.all(jnp.isfinite(nd.momentum_magnitude(cmi)))
