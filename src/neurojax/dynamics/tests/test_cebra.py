"""Integration test: NeuroJAX re-exports jaxctrl's CEBRA.

The method (and its known-answer ring tests) lives in jaxctrl; here we just
check the re-export works and fits/transforms neuroimaging-shaped data. (Skips
cleanly if jaxctrl is not installed, like the SINDy re-export.)
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

cebra = pytest.importorskip("neurojax.dynamics.cebra")
CEBRA = cebra.CEBRA
ContrastiveEncoder = cebra.ContrastiveEncoder


def test_reexport_available():
    assert CEBRA is not None and ContrastiveEncoder is not None


def test_fit_transform_shapes():
    X = jr.normal(jr.PRNGKey(0), (1500, 20))  # (time, parcels)-shaped
    m = CEBRA(out_dim=2, n_steps=50, key=jr.PRNGKey(0))
    hist = m.fit(X)
    assert len(hist) == 50
    assert m.transform(X).shape == (1500, 2)
