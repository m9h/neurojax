"""Phase-flow routing-mode extraction (Vinão-Carl).

A phase time-series on a mesh → per-time divergence (sources/sinks) and vorticity
(vortices) fields → PCA "routing modes" + their time activations → rerouting rates
(zero-crossings/s of the activations).  Built on the Hodge operators.
"""

import numpy as np
import jax.numpy as jnp

from neurojax.analysis.routing import (
    routing_fields,
    routing_modes,
    rerouting_rate,
)
from neurojax.geometry.hodge import divergence, curl


def flat_grid(k=20):
    xs = np.linspace(0.0, 1.0, k)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(k * k)], axis=1).astype(np.float64)
    F = []
    for i in range(k - 1):
        for j in range(k - 1):
            a, b = i * k + j, i * k + (j + 1)
            c, d = (i + 1) * k + j, (i + 1) * k + (j + 1)
            F += [[a, b, c], [b, d, c]]
    return jnp.asarray(V), jnp.asarray(np.array(F))


def test_routing_modes_recover_rank1():
    # field(t,x) = act(t) · pattern(x) -> top mode = pattern, activation = act
    rng = np.random.default_rng(0)
    T, n = 200, 60
    act = np.sin(2 * np.pi * 3 * np.arange(T) / T)
    pat = rng.standard_normal(n)
    field = np.outer(act, pat)
    modes, activ, var = routing_modes(jnp.asarray(field), n_modes=5)
    modes, activ, var = np.asarray(modes), np.asarray(activ), np.asarray(var)
    assert var[0] > 0.98                                          # rank-1 dominates
    assert abs(np.corrcoef(modes[0], pat)[0, 1]) > 0.99           # spatial pattern
    assert abs(np.corrcoef(activ[:, 0], act - act.mean())[0, 1]) > 0.99   # activation


def test_rerouting_rate_counts_sign_flips():
    # a pure sinusoid at f Hz crosses zero 2f times/s
    fs, f, T = 100.0, 5.0, 1000
    a = np.sin(2 * np.pi * f * np.arange(T) / fs)[:, None]
    rate = float(np.asarray(rerouting_rate(jnp.asarray(a), fs))[0])
    np.testing.assert_allclose(rate, 2 * f, atol=0.5)


def test_routing_fields_vortex():
    # a vortex phase φ = atan2(y, x) -> net positive vorticity, ~zero net divergence
    V, F = flat_grid(28)
    Vn = np.asarray(V)
    phase = np.arctan2(Vn[:, 1] - 0.5, Vn[:, 0] - 0.5)            # CCW winding
    div, vort = routing_fields(V, F, jnp.asarray(phase[None, :]))  # T=1
    div, vort = np.asarray(div)[0], np.asarray(vort)[0]
    interior = (Vn[:, 0] > 0.25) & (Vn[:, 0] < 0.75) & (Vn[:, 1] > 0.25) & (Vn[:, 1] < 0.75)
    assert vort[interior].sum() > 0                              # net CCW vortex
    assert abs(div[interior].sum()) < vort[interior].sum()      # mostly solenoidal
