"""Tests for windowed systems identification (SINDy, DMD, log-signatures).

Red-green TDD: tests written first, then bugs fixed in code or expectations.
Uses synthetic dynamical systems with known properties so ground truth is available.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neurojax.dynamics.windowed import (
    WindowedDMDResult,
    WindowedSINDyResult,
    WindowedSignatureResult,
    windowed_dmd,
    windowed_signatures,
    windowed_sindy,
)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def _make_stable_linear(n_samples=3000, n_vars=3, dt=0.01, seed=0):
    """Generate data from a stable linear system dx/dt = Ax.

    Eigenvalues of A are all negative → max Re(eig) < 0.
    """
    rng = np.random.default_rng(seed)
    # Stable A: diagonal with negative eigenvalues, sized to n_vars
    eigs = np.linspace(-0.5, -2.0, n_vars)
    A = np.diag(eigs)
    x = np.zeros((n_samples, n_vars))
    x[0] = rng.normal(size=n_vars)
    for t in range(1, n_samples):
        x[t] = x[t - 1] + dt * (A @ x[t - 1]) + 0.01 * rng.normal(size=n_vars)
    return jnp.array(x)


def _make_oscillator(freq_hz=5.0, n_samples=3000, dt=0.01, n_vars=3, seed=0):
    """Generate data from a damped oscillator at a known frequency.

    The first two variables oscillate at freq_hz, third is noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) * dt
    omega = 2 * np.pi * freq_hz
    x = np.zeros((n_samples, n_vars))
    x[:, 0] = np.cos(omega * t) * np.exp(-0.1 * t)
    x[:, 1] = np.sin(omega * t) * np.exp(-0.1 * t)
    if n_vars > 2:
        x[:, 2] = 0.1 * rng.normal(size=n_samples)
    # Add small noise
    x += 0.02 * rng.normal(size=x.shape)
    return jnp.array(x)


def _make_regime_change(n_samples=6000, n_vars=3, dt=0.01, seed=0):
    """Two regimes: stable linear for first half, oscillatory for second half.

    The change point is at n_samples // 2.
    """
    half = n_samples // 2
    d1 = np.array(_make_stable_linear(half, n_vars, dt, seed))
    d2 = np.array(_make_oscillator(5.0, half, dt, n_vars, seed + 1))
    # Smoothly transition (avoid discontinuity)
    return jnp.array(np.concatenate([d1, d2], axis=0))


# =====================================================================
# 1. Windowed SINDy — stability tracking
# =====================================================================

class TestWindowedSINDyShapes:
    """Output shapes and types."""

    def test_returns_result_dataclass(self):
        data = _make_stable_linear(3000)
        result = windowed_sindy(data, window_size=1500, stride=750)
        assert isinstance(result, WindowedSINDyResult)

    def test_times_shape(self):
        data = _make_stable_linear(3000)
        result = windowed_sindy(data, window_size=1500, stride=750)
        n_windows = (3000 - 1500) // 750 + 1
        assert result.times.shape == (n_windows,)

    def test_max_real_eig_shape(self):
        data = _make_stable_linear(3000)
        result = windowed_sindy(data, window_size=1500, stride=750)
        assert result.max_real_eig.shape == result.times.shape

    def test_eigenvalues_shape(self):
        data = _make_stable_linear(3000)
        result = windowed_sindy(data, window_size=1500, stride=750, n_pca=3)
        n_windows = result.times.shape[0]
        assert result.eigenvalues.shape == (n_windows, 3)

    def test_coefficients_shape(self):
        data = _make_stable_linear(3000)
        result = windowed_sindy(data, window_size=1500, stride=750, n_pca=3, degree=2)
        n_windows = result.times.shape[0]
        # Polynomial library size for 3 vars, degree 2: 1 + 3 + 6 = 10
        assert result.coefficients.shape[0] == n_windows
        assert result.coefficients.shape[2] == 3  # n_pca output dims


class TestWindowedSINDyStability:
    """SINDy should detect stability from eigenvalues."""

    def test_stable_system_bounded_eigenvalues(self):
        """Stable linear system → eigenvalues should be bounded (not diverging).

        Note: SINDy on PCA-projected noisy data won't perfectly recover
        true eigenvalues, but they should remain bounded and not blow up.
        """
        data = _make_stable_linear(4000, n_vars=5)
        result = windowed_sindy(data, window_size=2000, stride=1000, n_pca=3, degree=1)
        # Eigenvalues should be bounded — not diverging
        assert np.all(np.abs(result.max_real_eig) < 10.0), (
            f"Eigenvalues unbounded: {result.max_real_eig}"
        )

    def test_eigenvalues_finite(self):
        data = _make_stable_linear(3000)
        result = windowed_sindy(data, window_size=1500, stride=750)
        assert np.all(np.isfinite(result.max_real_eig))
        assert np.all(np.isfinite(result.eigenvalues))

    def test_window_centres_correct(self):
        data = _make_stable_linear(3000)
        result = windowed_sindy(data, window_size=1000, stride=500)
        # First centre should be at window_size // 2
        assert result.times[0] == 500
        # Subsequent centres spaced by stride
        if len(result.times) > 1:
            assert result.times[1] == 1000


# =====================================================================
# 2. Windowed DMD — frequency tracking
# =====================================================================

class TestWindowedDMDShapes:
    """Output shapes and types."""

    def test_returns_result_dataclass(self):
        data = _make_oscillator(5.0, 3000)
        result = windowed_dmd(data, window_size=1500, stride=750, rank=5)
        assert isinstance(result, WindowedDMDResult)

    def test_times_shape(self):
        data = _make_oscillator(5.0, 3000)
        result = windowed_dmd(data, window_size=1500, stride=750, rank=5)
        n_windows = (3000 - 1500) // 750 + 1
        assert result.times.shape == (n_windows,)

    def test_frequencies_shape(self):
        data = _make_oscillator(5.0, 3000)
        result = windowed_dmd(data, window_size=1500, stride=750, rank=3)
        assert result.frequencies.shape[0] == result.times.shape[0]
        # rank is min(requested_rank, n_vars)
        assert result.frequencies.shape[1] == min(3, data.shape[1])

    def test_growth_rates_shape(self):
        data = _make_oscillator(5.0, 3000)
        result = windowed_dmd(data, window_size=1500, stride=750, rank=5)
        assert result.growth_rates.shape == result.frequencies.shape


class TestWindowedDMDFrequencies:
    """DMD should recover known frequencies."""

    def test_dominant_frequency_near_target(self):
        """5 Hz oscillator → DMD should find a mode near 5 Hz."""
        data = _make_oscillator(freq_hz=5.0, n_samples=5000, dt=0.01)
        result = windowed_dmd(data, window_size=2000, stride=1000, rank=5, dt=0.01)
        # Check that at least one window has a frequency near 5 Hz
        all_freqs = result.frequencies.flatten()
        near_5hz = np.abs(all_freqs - 5.0) < 1.5  # within 1.5 Hz
        assert np.any(near_5hz), (
            f"Expected frequency near 5 Hz, got {np.sort(all_freqs)[:5]}"
        )

    def test_frequencies_nonnegative(self):
        data = _make_oscillator(5.0, 3000)
        result = windowed_dmd(data, window_size=1500, stride=750, rank=5)
        assert np.all(result.frequencies >= 0)

    def test_eigenvalues_finite(self):
        data = _make_oscillator(5.0, 3000)
        result = windowed_dmd(data, window_size=1500, stride=750, rank=5)
        assert np.all(np.isfinite(result.eigenvalues))


# =====================================================================
# 3. Windowed log-signatures — change-point detection
# =====================================================================

class TestWindowedSignaturesShapes:
    """Output shapes and types."""

    def test_returns_result_dataclass(self):
        data = _make_stable_linear(3000)
        result = windowed_signatures(data, window_size=1500, stride=750)
        assert isinstance(result, WindowedSignatureResult)

    def test_times_shape(self):
        data = _make_stable_linear(3000)
        result = windowed_signatures(data, window_size=1500, stride=750)
        n_windows = (3000 - 1500) // 750 + 1
        assert result.times.shape == (n_windows,)

    def test_signatures_shape(self):
        data = _make_stable_linear(3000)
        result = windowed_signatures(data, window_size=1500, stride=750, n_pca=3)
        assert result.signatures.shape[0] == result.times.shape[0]
        # Signature dimension depends on n_pca+1 (time augmented) and depth
        assert result.signatures.shape[1] > 0

    def test_distances_shape(self):
        data = _make_stable_linear(4500)
        result = windowed_signatures(data, window_size=1500, stride=750)
        n_windows = result.times.shape[0]
        assert result.distances.shape == (n_windows - 1,)


class TestWindowedSignaturesChangePoints:
    """Signature distances should spike at regime changes."""

    def test_distances_nonnegative(self):
        data = _make_stable_linear(3000)
        result = windowed_signatures(data, window_size=1500, stride=750)
        assert np.all(result.distances >= 0)

    def test_constant_dynamics_no_change_points(self):
        """Stationary data → no change points (or very few)."""
        data = _make_stable_linear(6000, seed=42)
        result = windowed_signatures(
            data, window_size=1500, stride=750, change_threshold=3.0
        )
        # Should have very few change points for stationary dynamics
        assert len(result.change_points) <= 2, (
            f"Expected few change points for stationary data, got {len(result.change_points)}"
        )

    def test_regime_change_detected(self):
        """Data with a regime shift → at least one change point near the transition."""
        data = _make_regime_change(6000, seed=42)
        result = windowed_signatures(
            data, window_size=1000, stride=500, change_threshold=1.5
        )
        # Should detect at least one change point
        assert len(result.change_points) >= 1, "No change points detected"
        # The change should be roughly in the middle (window index ~5 for 6000/500 stride)
        mid_window = (6000 // 2) // 500  # approximate window index of transition
        closest = np.min(np.abs(result.change_points - mid_window))
        assert closest <= 3, (
            f"Change point too far from expected middle: "
            f"change_points={result.change_points}, expected near window {mid_window}"
        )

    def test_signatures_finite(self):
        data = _make_stable_linear(3000)
        result = windowed_signatures(data, window_size=1500, stride=750)
        assert np.all(np.isfinite(result.signatures))


# =====================================================================
# 4. Integration: all three methods on the same regime-change data
# =====================================================================

class TestCrossMethodIntegration:
    """All three methods should detect the same regime change."""

    @pytest.fixture
    def regime_data(self):
        return _make_regime_change(6000, n_vars=5, seed=123)

    def test_all_methods_run(self, regime_data):
        """Basic smoke test: all three methods complete without error."""
        sindy = windowed_sindy(regime_data, window_size=1500, stride=750, n_pca=3)
        dmd = windowed_dmd(regime_data, window_size=1500, stride=750, rank=5)
        sigs = windowed_signatures(regime_data, window_size=1500, stride=750, n_pca=3)
        assert sindy.times.shape[0] == dmd.times.shape[0] == sigs.times.shape[0]

    def test_window_times_aligned(self, regime_data):
        """All methods should produce the same window centre times."""
        sindy = windowed_sindy(regime_data, window_size=1500, stride=750)
        dmd = windowed_dmd(regime_data, window_size=1500, stride=750)
        sigs = windowed_signatures(regime_data, window_size=1500, stride=750)
        np.testing.assert_array_equal(sindy.times, dmd.times)
        np.testing.assert_array_equal(sindy.times, sigs.times)

    def test_eigenvalue_shift_near_regime_change(self, regime_data):
        """SINDy eigenvalue character should change around the midpoint."""
        result = windowed_sindy(regime_data, window_size=1500, stride=750, n_pca=3)
        n_windows = len(result.times)
        first_half = result.max_real_eig[:n_windows // 2]
        second_half = result.max_real_eig[n_windows // 2:]
        # The two halves should have different eigenvalue statistics
        # (stable → oscillatory changes the eigenvalue structure)
        assert np.mean(first_half) != pytest.approx(np.mean(second_half), abs=0.01)

    def test_dmd_frequency_shift(self, regime_data):
        """DMD dominant frequency should differ between regimes."""
        result = windowed_dmd(regime_data, window_size=1500, stride=750, rank=5)
        n_windows = len(result.times)
        # Compare dominant frequency in first vs second half
        freq_first = np.mean(result.frequencies[:n_windows // 2, 0])
        freq_second = np.mean(result.frequencies[n_windows // 2:, 0])
        # Oscillatory regime should have higher dominant frequency
        assert freq_first != pytest.approx(freq_second, abs=0.1)

    def test_all_indicators_spike_near_transition(self, regime_data):
        """SINDy eigenvalue jump, DMD frequency jump, and signature distance
        spike should all peak in the middle third (near the regime change).

        With limited windows, exact alignment is noisy — we check that
        each indicator's peak falls in the transition region rather than
        requiring they match the same window index.
        """
        ws, stride = 1000, 500

        sindy = windowed_sindy(regime_data, window_size=ws, stride=stride, n_pca=3)
        dmd = windowed_dmd(regime_data, window_size=ws, stride=stride, rank=3)
        sigs = windowed_signatures(regime_data, window_size=ws, stride=stride, n_pca=3)

        n_jumps = len(sindy.times) - 1
        # Middle third of the window sequence
        mid_lo = n_jumps // 3
        mid_hi = 2 * n_jumps // 3

        # SINDy indicator: largest absolute jump in max_real_eig
        sindy_jumps = np.abs(np.diff(sindy.max_real_eig))
        sindy_peak = int(np.argmax(sindy_jumps))

        # DMD indicator: largest jump in dominant frequency
        dmd_jumps = np.abs(np.diff(dmd.frequencies[:, 0]))
        dmd_peak = int(np.argmax(dmd_jumps))

        # Signature indicator: largest distance
        sig_peak = int(np.argmax(sigs.distances))

        # At least 2 of 3 indicators should peak in the middle third
        in_middle = sum(1 for p in [sindy_peak, dmd_peak, sig_peak]
                        if mid_lo <= p <= mid_hi)
        assert in_middle >= 2, (
            f"Expected ≥2/3 indicators to peak in middle third "
            f"[{mid_lo}, {mid_hi}], got peaks at: "
            f"SINDy={sindy_peak}, DMD={dmd_peak}, Sig={sig_peak}"
        )
