"""Travelling-wave analysis in JAX.

Ports the Muller-lab ``wave-matlab`` + ``generalized-phase`` MATLAB methods
(Davis, Muller, Sejnowski, Miller and colleagues) to differentiable JAX, so
travelling/rotating cortical waves can be quantified on the same data we feed
the HMM/DyNeMo network-state models — and cross-tested against the structured
network cycles of Woolrich's TINDA (see docs/CYCLES_VS_TRAVELLING_WAVES.md).

Operators
---------
- ``generalized_phase``  — robust wideband analytic phase (Davis/Muller 2020),
  with the negative-instantaneous-frequency correction.
- ``phase_gradient``     — wrap-free spatial phase gradient via complex
  multiplication (Feldman 2011): gx = arg(V[x+1] · conj(V[x])).
- ``phase_gradient_directionality`` (PGD) — wave coherence, ->1 for a clean wave.
- ``wave_direction``     — propagation direction from the mean gradient.
- ``divergence`` / ``curl`` — source/sink (radial) and rotation detectors.
- ``singularity_location`` — phase-singularity (rotational-wave centre).

Fields are complex analytic signals on a regular 2-D grid ``(..., H, W)`` (the
Utah-array geometry).  The gradient/curl/divergence are discrete differential
operators; on an irregular cortical mesh, substitute the mesh operators from
``neurojax.geometry`` for the grid ``jnp.gradient`` calls.
"""

from __future__ import annotations

import math
from typing import Tuple

import jax
import jax.numpy as jnp

from neurojax.analysis.analytic import hilbert

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Generalized Phase (Davis/Muller 2020)
# ---------------------------------------------------------------------------


def _interp_negative_frequency(phase_1d: jnp.ndarray) -> jnp.ndarray:
    """Linearly interpolate an unwrapped 1-D phase across negative-instantaneous-
    frequency epochs (the GP correction).  Anchors are the running-maximum
    ("record-high") samples, which are monotonic in both index and value, so the
    interpolated phase is guaranteed non-decreasing (no spurious phase reversals).
    Runs eagerly (boolean masking), so call outside ``jit``."""
    T = phase_1d.shape[0]
    cummax = jax.lax.cummax(phase_1d, axis=0)
    anchor = phase_1d >= cummax  # record highs; sample 0 always anchors
    idx = jnp.arange(T)
    return jnp.interp(idx, idx[anchor], phase_1d[anchor])


def generalized_phase(
    x: jnp.ndarray, fs: float, neg_freq_correction: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Wideband analytic phase and amplitude.

    Parameters
    ----------
    x : (..., n_times) real signal, assumed already broadband-filtered
        (e.g. 5-40 Hz) — GP deliberately does not narrowband-filter.
    fs : sampling frequency (Hz).
    neg_freq_correction : interpolate phase across negative-instantaneous-
        frequency epochs (the Davis/Muller correction).

    Returns
    -------
    phase : (..., n_times) unwrapped instantaneous phase.
    amplitude : (..., n_times) instantaneous amplitude.
    """
    z = hilbert(x)
    amplitude = jnp.abs(z)
    phase = jnp.unwrap(jnp.angle(z), axis=-1)

    if neg_freq_correction:
        flat = phase.reshape(-1, phase.shape[-1])
        flat = jnp.stack([_interp_negative_frequency(row) for row in flat])
        phase = flat.reshape(phase.shape)

    return phase, amplitude


# ---------------------------------------------------------------------------
# Spatial phase gradient (wrap-free, via complex multiplication)
# ---------------------------------------------------------------------------


def phase_gradient(V: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Wrap-free spatial phase gradient of a complex analytic field.

    Parameters
    ----------
    V : (..., H, W) complex analytic field for one (or many) time frame(s).

    Returns
    -------
    gx, gy : (..., H, W) phase gradients along +x (columns) and +y (rows).
        ``g = arg(V_next · conj(V_here))`` avoids 2-pi wrap artefacts.
    """
    gx = jnp.angle(V[..., :, 1:] * jnp.conj(V[..., :, :-1]))
    gy = jnp.angle(V[..., 1:, :] * jnp.conj(V[..., :-1, :]))
    nd = V.ndim
    gx = jnp.pad(gx, [(0, 0)] * (nd - 1) + [(0, 1)], mode="edge")
    gy = jnp.pad(gy, [(0, 0)] * (nd - 2) + [(0, 1), (0, 0)], mode="edge")
    return gx, gy


def wave_direction(gx: jnp.ndarray, gy: jnp.ndarray) -> jnp.ndarray:
    """Propagation direction (radians) from the mean phase gradient."""
    return jnp.arctan2(jnp.mean(gy, axis=(-2, -1)), jnp.mean(gx, axis=(-2, -1)))


def phase_gradient_directionality(gx: jnp.ndarray, gy: jnp.ndarray) -> jnp.ndarray:
    """PGD = |mean gradient| / mean|gradient|, in [0, 1] (->1 = coherent wave)."""
    mx = jnp.mean(gx, axis=(-2, -1))
    my = jnp.mean(gy, axis=(-2, -1))
    num = jnp.sqrt(mx ** 2 + my ** 2)
    den = jnp.mean(jnp.sqrt(gx ** 2 + gy ** 2), axis=(-2, -1))
    return num / (den + 1e-12)


# ---------------------------------------------------------------------------
# Source / sink (divergence) and rotation (curl / singularities)
# ---------------------------------------------------------------------------


def divergence(gx: jnp.ndarray, gy: jnp.ndarray) -> jnp.ndarray:
    """Divergence of the gradient field — positive at expanding (source) waves."""
    return jnp.gradient(gx, axis=-1) + jnp.gradient(gy, axis=-2)


def curl(gx: jnp.ndarray, gy: jnp.ndarray) -> jnp.ndarray:
    """Curl (z-component) of the gradient field — nonzero at rotating waves /
    phase singularities."""
    return jnp.gradient(gy, axis=-1) - jnp.gradient(gx, axis=-2)


def singularity_location(gx: jnp.ndarray, gy: jnp.ndarray) -> Tuple[int, int]:
    """(row, col) of the strongest phase singularity (max |curl|), single frame."""
    c = jnp.abs(curl(gx, gy))
    iy, ix = jnp.unravel_index(jnp.argmax(c), c.shape)
    return int(iy), int(ix)
