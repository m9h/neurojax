"""Tests for JAX-accelerated MEGA-PRESS pipeline.

Verifies numerical equivalence with the NumPy implementation and
confirms JAX-specific features (vmap, jit, grad) work correctly.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from neurojax.analysis.mega_press import (
    coil_combine_svd as np_coil_combine_svd,
    apply_correction as np_apply_correction,
    process_mega_press as np_process_mega_press,
)
from neurojax.analysis.mega_press_jax import (
    coil_combine_svd as jax_coil_combine_svd,
    apply_correction as jax_apply_correction,
    process_mega_press as jax_process_mega_press,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_singlet(ppm, amplitude, lw, n_pts, dwell, cf):
    """Create a single Lorentzian FID at a given ppm (NumPy)."""
    freq_hz = (ppm - 4.65) * (cf / 1e6)
    t = np.arange(n_pts) * dwell
    return amplitude * np.exp(2j * np.pi * freq_hz * t) * np.exp(-np.pi * lw * t)


def _make_mega_data(
    n_pts=2048, n_coils=4, n_dyn=16, dwell=2.5e-4, cf=123.25e6,
    gaba_conc=1.0, naa_conc=10.0, cr_conc=8.0, noise_level=0.01, seed=42,
):
    """Generate synthetic MEGA-PRESS data with known ground truth."""
    rng = np.random.default_rng(seed)
    naa = _make_singlet(2.01, naa_conc, 3.0, n_pts, dwell, cf)
    cr = _make_singlet(3.03, cr_conc, 4.0, n_pts, dwell, cf)
    gaba = _make_singlet(3.01, gaba_conc, 8.0, n_pts, dwell, cf)
    edit_on_signal = naa + cr + gaba
    edit_off_signal = naa + cr - gaba
    coil_weights = rng.standard_normal(n_coils) + 1j * rng.standard_normal(n_coils)
    coil_weights /= np.max(np.abs(coil_weights))
    t = np.arange(n_pts) * dwell
    data = np.zeros((n_pts, n_coils, 2, n_dyn), dtype=complex)
    for d in range(n_dyn):
        for c in range(n_coils):
            noise_on = noise_level * (rng.standard_normal(n_pts) + 1j * rng.standard_normal(n_pts))
            noise_off = noise_level * (rng.standard_normal(n_pts) + 1j * rng.standard_normal(n_pts))
            data[:, c, 0, d] = coil_weights[c] * edit_on_signal + noise_on
            data[:, c, 1, d] = coil_weights[c] * edit_off_signal + noise_off
    return data, {'dwell': dwell, 'cf': cf}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestJaxCoilCombine:
    def test_jax_coil_combine_matches_numpy(self):
        """JAX coil combine gives same result as NumPy (atol=1e-5)."""
        data_np, _ = _make_mega_data()
        data_jax = jnp.array(data_np)

        result_np = np_coil_combine_svd(data_np)
        result_jax = jax_coil_combine_svd(data_jax)

        np.testing.assert_allclose(
            np.array(result_jax), result_np, atol=1e-5,
            err_msg="JAX coil_combine_svd does not match NumPy",
        )


class TestJaxApplyCorrection:
    def test_jax_apply_correction_matches_numpy(self):
        """JAX correction matches NumPy."""
        rng = np.random.default_rng(123)
        fid_np = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
        freq_shift = 3.5
        phase_shift = 0.7
        dwell_time = 2.5e-4

        result_np = np_apply_correction(fid_np, freq_shift, phase_shift, dwell_time)
        result_jax = jax_apply_correction(
            jnp.array(fid_np),
            jnp.float64(freq_shift),
            jnp.float64(phase_shift),
            jnp.float64(dwell_time),
        )

        np.testing.assert_allclose(
            np.array(result_jax), result_np, atol=1e-5,
            err_msg="JAX apply_correction does not match NumPy",
        )


class TestJaxProcessMega:
    def test_jax_process_mega_matches_numpy(self):
        """Full pipeline JAX vs NumPy match on synthetic data."""
        data_np, truth = _make_mega_data(noise_level=0.001)
        dwell = truth['dwell']
        cf = truth['cf']

        result_np = np_process_mega_press(
            data_np, dwell, cf, align=False, reject=False,
        )
        result_jax = jax_process_mega_press(
            jnp.array(data_np), dwell, cf, align=False, reject=False,
        )

        np.testing.assert_allclose(
            np.array(result_jax.diff), result_np.diff, atol=1e-5,
            err_msg="JAX process diff does not match NumPy",
        )
        np.testing.assert_allclose(
            np.array(result_jax.edit_on), result_np.edit_on, atol=1e-5,
            err_msg="JAX process edit_on does not match NumPy",
        )
        np.testing.assert_allclose(
            np.array(result_jax.edit_off), result_np.edit_off, atol=1e-5,
            err_msg="JAX process edit_off does not match NumPy",
        )


class TestJaxVmap:
    def test_jax_vmap_subjects(self):
        """vmap over 4 synthetic subjects works."""
        subjects = []
        for seed in range(4):
            data_np, _ = _make_mega_data(seed=seed, n_dyn=8, noise_level=0.001)
            subjects.append(data_np)
        batch = jnp.stack(subjects, axis=0)  # (4, n_pts, n_coils, 2, n_dyn)

        dwell = 2.5e-4
        cf = 123.25e6

        # vmap over leading subject dimension
        vmapped = jax.vmap(
            lambda d: jax_process_mega_press(d, dwell, cf, align=False, reject=False).diff
        )
        diffs = vmapped(batch)

        assert diffs.shape == (4, 2048), f"Expected (4, 2048), got {diffs.shape}"
        # Each subject should produce a non-trivial diff
        for i in range(4):
            assert jnp.max(jnp.abs(diffs[i])) > 0


class TestJaxJit:
    def test_jax_jit_compiles(self):
        """jit compilation succeeds and produces correct output."""
        data_np, truth = _make_mega_data(noise_level=0.001)
        data_jax = jnp.array(data_np)
        dwell = truth['dwell']
        cf = truth['cf']

        @jax.jit
        def run(d):
            return jax_process_mega_press(d, dwell, cf, align=False, reject=False).diff

        diff_jit = run(data_jax)

        # Compare to non-jit
        diff_eager = jax_process_mega_press(
            data_jax, dwell, cf, align=False, reject=False,
        ).diff

        np.testing.assert_allclose(
            np.array(diff_jit), np.array(diff_eager), atol=1e-7,
            err_msg="JIT result differs from eager result",
        )


class TestJaxGrad:
    def test_jax_grad_through_correction(self):
        """Can differentiate through apply_correction."""
        rng = np.random.default_rng(99)
        fid = jnp.array(rng.standard_normal(512) + 1j * rng.standard_normal(512))
        dwell = 2.5e-4

        def loss_fn(freq_shift):
            corrected = jax_apply_correction(fid, freq_shift, 0.0, dwell)
            return jnp.sum(jnp.abs(corrected) ** 2).real

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(0.0)

        # The gradient should be finite and real-valued
        assert jnp.isfinite(g), f"Gradient is not finite: {g}"
        # For |exp(i*2*pi*f*t)|^2 = |fid|^2, gradient w.r.t. f should be ~0
        # because the magnitude doesn't depend on frequency shift
        assert abs(float(g)) < 1e-3, f"Expected near-zero gradient, got {g}"
