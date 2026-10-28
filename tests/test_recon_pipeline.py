"""End-to-end EMEG Recon orchestrator: sensor EEG -> sources -> directed connectivity.

Chains the validated pieces — an inverse operator (default MNE min-norm; pass a
dSPM/HIGGS operator for the leakage-aware path Valdés-Sosa recommends) and
analysis.source_connectivity (PDC/DTF + leakage diagnostic). The test plants a
known source-level directed coupling (0 -> 1), forwards it through a leadfield to
sensors, and checks the pipeline recovers the direction from the sensor data.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from neurojax.pipeline.recon import recon_directed_connectivity


def _simulate(seed=0, T=3000):
    rng = np.random.default_rng(seed)
    n_sensors, n_sources = 10, 4  # overdetermined => well-resolved, low leakage
    G = rng.standard_normal((n_sensors, n_sources))
    A = np.eye(n_sources) * 0.5
    A[1, 0] = 0.8  # source 0 drives source 1
    S = np.zeros((n_sources, T))
    e = rng.standard_normal((n_sources, T))
    for t in range(1, T):
        S[:, t] = A @ S[:, t - 1] + e[:, t]
    sensors = G @ S + 0.1 * rng.standard_normal((n_sensors, T))
    return jnp.asarray(G), jnp.asarray(sensors)


def test_recon_recovers_source_level_direction():
    G, Y = _simulate()
    freqs = jnp.linspace(0.0, 50.0, 17)
    out = recon_directed_connectivity(G, Y, freqs, fs=100.0, order=1)
    assert out["sources"].shape == (4, Y.shape[1])
    assert out["pdc"].shape == (17, 4, 4)
    assert "leakage" in out and out["leakage"].shape == (4, 4)
    # the recovered directed connectivity 0 -> 1 must dominate the reverse
    fwd = float(jnp.mean(out["pdc"][:, 1, 0]))
    rev = float(jnp.mean(out["pdc"][:, 0, 1]))
    assert fwd > rev + 0.1


def test_recon_accepts_a_plugged_inverse_operator():
    # passing an explicit inverse operator (e.g. a HIGGS/dSPM kernel) is honoured
    G, Y = _simulate()
    W = jnp.linalg.pinv(G)  # stand-in operator (n_sources, n_sensors)
    freqs = jnp.linspace(0.0, 50.0, 9)
    out = recon_directed_connectivity(G, Y, freqs, fs=100.0, inverse_operator=W)
    assert jnp.allclose(out["sources"], W @ Y)
