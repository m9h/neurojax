"""Tests for JAX travelling-wave operators (analysis/waves.py).

Ports of the Muller-lab wave-matlab + generalized-phase methods: Generalized
Phase, phase-gradient via complex multiplication, PGD, divergence/curl, and
planar/rotational wave detection.  Validated against synthetic fields with known
propagation.
"""

import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neurojax.analysis.waves import (
    curl,
    divergence,
    generalized_phase,
    mesh_phase_gradient,
    mesh_phase_gradient_directionality,
    phase_gradient,
    phase_gradient_directionality,
    phase_singularity_charge,
    singularity_location,
    wave_direction,
)

TWO_PI = 2 * math.pi


# --- synthetic analytic fields --------------------------------------------

def planar_field(h, w, kx, ky):
    yy, xx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    return jnp.exp(1j * (kx * xx + ky * yy)).astype(jnp.complex64)


def rotating_field(h, w, cx, cy):
    yy, xx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    return jnp.exp(1j * jnp.arctan2(yy - cy, xx - cx)).astype(jnp.complex64)


def radial_field(h, w, cx, cy):
    yy, xx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    r = jnp.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return jnp.exp(1j * r).astype(jnp.complex64)


# --- Generalized Phase -----------------------------------------------------

class TestGeneralizedPhase:
    def test_tone_instantaneous_frequency(self):
        fs = 250.0
        t = jnp.arange(2500) / fs
        x = jnp.sin(TWO_PI * 10.0 * t)
        phase, amp = generalized_phase(x, fs)
        ifreq = jnp.diff(phase) * fs / TWO_PI
        # median instantaneous frequency ~ 10 Hz
        assert abs(float(jnp.median(ifreq)) - 10.0) < 1.0
        assert amp.shape == x.shape

    def test_negative_frequency_correction_monotonic(self):
        fs = 250.0
        t = jnp.arange(3000) / fs
        # wideband signal that produces transient phase reversals
        x = jnp.sin(TWO_PI * 8.0 * t) + 0.7 * jnp.sin(TWO_PI * 38.0 * t + 1.0)
        p_corr, _ = generalized_phase(x, fs, neg_freq_correction=True)
        p_raw, _ = generalized_phase(x, fs, neg_freq_correction=False)
        neg_corr = float(jnp.mean(jnp.diff(p_corr) < -1e-6))
        neg_raw = float(jnp.mean(jnp.diff(p_raw) < -1e-6))
        assert neg_corr <= neg_raw
        assert neg_corr < 0.01  # corrected phase essentially monotonic

    def test_multichannel_shape(self):
        fs = 250.0
        x = jr.normal(jr.PRNGKey(0), (8, 1000))
        phase, amp = generalized_phase(x, fs)
        assert phase.shape == (8, 1000)


# --- Phase gradient (complex multiplication) -------------------------------

class TestPhaseGradient:
    def test_planar_gradient_recovers_wavevector(self):
        V = planar_field(20, 20, 0.3, 0.1)
        gx, gy = phase_gradient(V)
        np.testing.assert_allclose(float(jnp.mean(gx)), 0.3, atol=0.02)
        np.testing.assert_allclose(float(jnp.mean(gy)), 0.1, atol=0.02)

    def test_wave_direction(self):
        V = planar_field(20, 20, 0.3, 0.1)
        gx, gy = phase_gradient(V)
        d = float(wave_direction(gx, gy))
        np.testing.assert_allclose(d, math.atan2(0.1, 0.3), atol=0.05)

    def test_pgd_high_for_planar(self):
        gx, gy = phase_gradient(planar_field(24, 24, 0.25, 0.15))
        assert float(phase_gradient_directionality(gx, gy)) > 0.95

    def test_pgd_low_for_random(self):
        V = jnp.exp(1j * jr.uniform(jr.PRNGKey(1), (24, 24)) * TWO_PI)
        gx, gy = phase_gradient(V)
        assert float(phase_gradient_directionality(gx, gy)) < 0.4


# --- Rotational / radial waves --------------------------------------------

class TestSingularities:
    def test_curl_detects_rotation(self):
        gx, gy = phase_gradient(rotating_field(21, 21, 10.0, 10.0))
        assert float(jnp.max(jnp.abs(curl(gx, gy)))) > 1.0

    def test_planar_has_negligible_curl(self):
        gx, gy = phase_gradient(planar_field(21, 21, 0.3, 0.1))
        assert float(jnp.max(jnp.abs(curl(gx, gy)))) < 0.5

    def test_singularity_location_at_center(self):
        gx, gy = phase_gradient(rotating_field(21, 21, 10.0, 10.0))
        iy, ix = singularity_location(gx, gy)
        assert abs(int(iy) - 10) <= 2 and abs(int(ix) - 10) <= 2

    def test_divergence_detects_radial_source(self):
        gx, gy = phase_gradient(radial_field(21, 21, 10.0, 10.0))
        d = divergence(gx, gy)
        # expanding wave -> positive divergence away from the center on average
        assert float(jnp.mean(d[5:16, 5:16])) > 0.1


# --- Mesh (cortical-surface) wave operators --------------------------------

def flat_mesh(n):
    """A flat n x n triangulated grid in the z=0 plane (x=col, y=row), CCW."""
    verts, faces = [], []
    for i in range(n):
        for j in range(n):
            verts.append([float(j), float(i), 0.0])
    idx = lambda i, j: i * n + j
    for i in range(n - 1):
        for j in range(n - 1):
            faces.append([idx(i, j), idx(i, j + 1), idx(i + 1, j + 1)])
            faces.append([idx(i, j), idx(i + 1, j + 1), idx(i + 1, j)])
    return jnp.asarray(verts), jnp.asarray(faces, dtype=jnp.int32)


class TestMeshWaves:
    def test_planar_field_no_singularities(self):
        v, f = flat_mesh(15)
        V = jnp.exp(1j * (0.3 * v[:, 0] + 0.1 * v[:, 1]))
        charge = phase_singularity_charge(V, f)
        assert float(jnp.max(jnp.abs(charge))) < 0.1

    def test_mesh_gradient_recovers_wavevector(self):
        v, f = flat_mesh(15)
        V = jnp.exp(1j * (0.3 * v[:, 0] + 0.1 * v[:, 1]))
        g = mesh_phase_gradient(V, v, f)  # (n_faces, 3)
        np.testing.assert_allclose(float(jnp.mean(g[:, 0])), 0.3, atol=0.02)
        np.testing.assert_allclose(float(jnp.mean(g[:, 1])), 0.1, atol=0.02)
        assert float(jnp.max(jnp.abs(g[:, 2]))) < 1e-3  # gradient is in-plane

    def test_rotating_field_one_singularity(self):
        v, f = flat_mesh(15)
        cx = cy = 7.5  # off-vertex centre -> exactly one enclosing triangle
        V = jnp.exp(1j * jnp.arctan2(v[:, 1] - cy, v[:, 0] - cx))
        charge = phase_singularity_charge(V, f)
        assert abs(float(jnp.sum(charge))) > 0.9     # net topological charge ~ +-1
        assert float(jnp.max(jnp.abs(charge))) > 0.9  # carried by one face

    def test_mesh_pgd_high_for_planar(self):
        v, f = flat_mesh(15)
        V = jnp.exp(1j * (0.3 * v[:, 0] + 0.1 * v[:, 1]))
        g = mesh_phase_gradient(V, v, f)
        assert float(mesh_phase_gradient_directionality(g, v, f)) > 0.95
