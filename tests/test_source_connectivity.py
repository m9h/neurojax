"""Directed connectivity on SOURCE estimates + a spatial-leakage diagnostic.

Valdés-Sosa's first methodological warning (and the HIGGS rationale): never run
ESI and *then* connectivity on naively-projected sources — spatial leakage
manufactures ghost directed connectivity. This wires neurojax's MVAR PDC/DTF
(analysis.mvar) onto source-space time courses (e.g. higgs_source_estimate), and
adds leakage_index (built on the resolution matrix M = W @ G) to flag when a
directed estimate is leakage-contaminated. The test demonstrates the ghost
connectivity that a minimum-norm projector injects vs leakage-free sources.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from neurojax.analysis.source_connectivity import (
    directed_source_connectivity,
    leakage_index,
)


def _forward_and_sources(seed=0, T=4000):
    rng = np.random.default_rng(seed)
    n_sensors, n_sources = 4, 6  # underdetermined => min-norm leaks (M != I)
    G = rng.standard_normal((n_sensors, n_sources))
    # independent AR(1) sources, NO cross-coupling; source 0 strong + distinct
    # dynamics so its leakage dominates the weak sources' naive estimates.
    a = np.array([0.7, 0.3, 0.3, 0.3, 0.3, 0.3])
    amp = np.array([3.0, 0.3, 0.3, 0.3, 0.3, 0.3])
    S = np.zeros((n_sources, T))
    e = rng.standard_normal((n_sources, T))
    for t in range(1, T):
        S[:, t] = a * S[:, t - 1] + amp * e[:, t]
    return jnp.asarray(G), jnp.asarray(S)


def test_directed_source_connectivity_shapes():
    G, S = _forward_and_sources()
    freqs = jnp.linspace(0.0, 50.0, 17)
    out = directed_source_connectivity(S, freqs, fs=100.0, order=1)
    assert out["pdc"].shape == (17, 6, 6)
    assert out["dtf"].shape == (17, 6, 6)


def test_minimum_norm_leakage_injects_ghost_directed_connectivity():
    G, S = _forward_and_sources()
    sensors = G @ S
    W = jnp.linalg.pinv(G)          # naive minimum-norm projector
    S_naive = W @ sensors           # leakage-mixed source estimates
    freqs = jnp.linspace(0.0, 50.0, 17)

    pdc_true = directed_source_connectivity(S, freqs, 100.0)["pdc"]
    pdc_naive = directed_source_connectivity(S_naive, freqs, 100.0)["pdc"]
    # ghost directed influence FROM strong source 0 INTO the weak sources (col 0)
    ghost_true = float(jnp.mean(pdc_true[:, 1:, 0]))
    ghost_naive = float(jnp.mean(pdc_naive[:, 1:, 0]))
    # leakage produces real ghost connectivity, many times the (near-zero) truth
    assert ghost_naive > 0.05
    assert ghost_naive > 4.0 * ghost_true

    # the resolution matrix M = W@G flags the leakage (non-zero off-diagonal)
    M = W @ G
    li = leakage_index(M)
    offdiag = li[~jnp.eye(6, dtype=bool)]
    assert float(jnp.max(offdiag)) > 0.1
