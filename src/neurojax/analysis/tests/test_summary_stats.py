"""Tests for HMM/DyNeMo summary statistics — TDD RED phase.

Summary stats are the standard way to characterize brain state dynamics:
fractional occupancy, lifetime, interval, switching rate, binarization.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neurojax.analysis.summary_stats import (
    binarize_alpha,
    fractional_occupancy,
    mean_interval,
    mean_lifetime,
    state_time_courses,
    switching_rate,
)


# ---------------------------------------------------------------------------
# Fixtures: known state sequences
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_states():
    """Hard state assignment: 3 states, 100 timepoints.
    State 0 for t=0-39, state 1 for t=40-69, state 2 for t=70-99.
    """
    s = jnp.zeros(100, dtype=int)
    s = s.at[40:70].set(1)
    s = s.at[70:100].set(2)
    return s


@pytest.fixture
def alternating_states():
    """Rapidly alternating between states 0 and 1, 200 timepoints.
    Switches every 10 steps.
    """
    s = jnp.zeros(200, dtype=int)
    for i in range(20):
        if i % 2 == 1:
            s = s.at[i * 10:(i + 1) * 10].set(1)
    return s


@pytest.fixture
def soft_alpha():
    """Soft mixing coefficients (3 modes, 100 timepoints).
    Mode 0 dominant first half, mode 1 dominant second half.
    """
    alpha = jnp.zeros((100, 3))
    alpha = alpha.at[:50, 0].set(0.8)
    alpha = alpha.at[:50, 1].set(0.15)
    alpha = alpha.at[:50, 2].set(0.05)
    alpha = alpha.at[50:, 0].set(0.1)
    alpha = alpha.at[50:, 1].set(0.8)
    alpha = alpha.at[50:, 2].set(0.1)
    return alpha


# ---------------------------------------------------------------------------
# state_time_courses — convert gamma/alpha to hard state sequence
# ---------------------------------------------------------------------------

class TestStateTimeCourses:
    def test_from_gamma(self):
        """Argmax of gamma gives hard assignments."""
        gamma = jnp.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
        states = state_time_courses(gamma)
        np.testing.assert_array_equal(states, [0, 1, 0])

    def test_shape(self):
        gamma = jr.normal(jr.PRNGKey(0), (50, 4))
        gamma = jax.nn.softmax(gamma, axis=1)
        states = state_time_courses(gamma)
        assert states.shape == (50,)

    def test_valid_range(self):
        gamma = jax.nn.softmax(jr.normal(jr.PRNGKey(0), (100, 5)), axis=1)
        states = state_time_courses(gamma)
        assert jnp.all(states >= 0)
        assert jnp.all(states < 5)


# ---------------------------------------------------------------------------
# fractional_occupancy
# ---------------------------------------------------------------------------

class TestFractionalOccupancy:
    def test_known_values(self, simple_states):
        fo = fractional_occupancy(simple_states, n_states=3)
        np.testing.assert_allclose(fo, [0.4, 0.3, 0.3], atol=1e-6)

    def test_sums_to_one(self, simple_states):
        fo = fractional_occupancy(simple_states, n_states=3)
        assert float(jnp.sum(fo)) == pytest.approx(1.0)

    def test_single_state(self):
        s = jnp.zeros(50, dtype=int)
        fo = fractional_occupancy(s, n_states=2)
        np.testing.assert_allclose(fo, [1.0, 0.0])

    def test_from_soft_alpha(self, soft_alpha):
        """Fractional occupancy from soft alpha = mean alpha per mode."""
        fo = fractional_occupancy(soft_alpha)
        assert fo.shape == (3,)
        assert float(jnp.sum(fo)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# mean_lifetime
# ---------------------------------------------------------------------------

class TestMeanLifetime:
    def test_known_values(self, simple_states):
        lt = mean_lifetime(simple_states, n_states=3)
        # State 0: one run of 40, state 1: one run of 30, state 2: one run of 30
        np.testing.assert_allclose(lt, [40.0, 30.0, 30.0])

    def test_alternating(self, alternating_states):
        lt = mean_lifetime(alternating_states, n_states=2)
        np.testing.assert_allclose(lt, [10.0, 10.0])

    def test_never_visited_state(self, simple_states):
        """State that never appears should have lifetime 0 or NaN."""
        lt = mean_lifetime(simple_states, n_states=4)
        # State 3 never appears
        assert lt[3] == 0.0 or jnp.isnan(lt[3])

    def test_in_seconds(self, simple_states):
        lt = mean_lifetime(simple_states, n_states=3, fs=100.0)
        # 40 samples at 100 Hz = 0.4s
        np.testing.assert_allclose(lt, [0.4, 0.3, 0.3], atol=1e-6)


# ---------------------------------------------------------------------------
# mean_interval
# ---------------------------------------------------------------------------

class TestMeanInterval:
    def test_alternating(self, alternating_states):
        mi = mean_interval(alternating_states, n_states=2)
        # Each state has gap of 10 samples between visits
        np.testing.assert_allclose(mi, [10.0, 10.0])

    def test_single_visit_no_interval(self, simple_states):
        """State visited only once has no interval (0 or NaN)."""
        mi = mean_interval(simple_states, n_states=3)
        # Each state visited once → no gap to measure
        for s in range(3):
            assert mi[s] == 0.0 or jnp.isnan(mi[s])


# ---------------------------------------------------------------------------
# switching_rate
# ---------------------------------------------------------------------------

class TestSwitchingRate:
    def test_no_switches(self):
        s = jnp.zeros(100, dtype=int)
        sr = switching_rate(s)
        assert float(sr) == 0.0

    def test_known_rate(self, alternating_states):
        sr = switching_rate(alternating_states)
        # 200 timepoints, switches at t=10,20,...,190 → 19 switches / 199 transitions
        expected = 19.0 / 199.0
        assert float(sr) == pytest.approx(expected, rel=0.01)

    def test_in_hz(self, alternating_states):
        sr = switching_rate(alternating_states, fs=100.0)
        # 19 switches in 2 seconds → 9.5 Hz
        expected = 19.0 / 199.0 * 100.0
        assert float(sr) == pytest.approx(expected, rel=0.01)

    def test_simple_states(self, simple_states):
        sr = switching_rate(simple_states)
        # 2 switches (0→1 at t=40, 1→2 at t=70)
        expected = 2.0 / 99.0
        assert float(sr) == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------------------
# binarize_alpha (GMM-based thresholding for DyNeMo)
# ---------------------------------------------------------------------------

class TestBinarizeAlpha:
    def test_output_shape(self, soft_alpha):
        binary = binarize_alpha(soft_alpha)
        assert binary.shape == soft_alpha.shape

    def test_binary_values(self, soft_alpha):
        """Binarized alpha should be 0 or 1."""
        binary = binarize_alpha(soft_alpha)
        unique = jnp.unique(binary)
        assert jnp.all((unique == 0) | (unique == 1))

    def test_dominant_mode_active(self, soft_alpha):
        """The clearly dominant mode should be active after binarization."""
        binary = binarize_alpha(soft_alpha)
        # Mode 0 dominant in first half
        assert float(jnp.mean(binary[:50, 0])) > 0.5
        # Mode 1 dominant in second half
        assert float(jnp.mean(binary[50:, 1])) > 0.5

    def test_threshold_method(self):
        """Simple threshold-based binarization as fallback."""
        alpha = jnp.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
        binary = binarize_alpha(alpha, method="threshold", threshold=0.5)
        np.testing.assert_array_equal(binary, [[1, 0], [0, 1], [1, 1]])
