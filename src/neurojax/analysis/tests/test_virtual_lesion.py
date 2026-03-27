"""Tests for virtual lesion framework — TDD RED phase.

Virtual lesions: selectively disconnect regions in a structural connectome,
re-simulate, and measure the functional consequence. Replicates Momi et al.
2023 methodology at scale.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neurojax.analysis.virtual_lesion import (
    apply_lesion,
    contribution_matrix,
    lesion_effect,
    local_network_transition,
    virtual_lesion_sweep,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_connectome():
    """6-region symmetric connectome with known structure."""
    W = jnp.array([
        [0.0, 0.8, 0.3, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.5, 0.2, 0.0, 0.0],
        [0.3, 0.5, 0.0, 0.7, 0.1, 0.0],
        [0.0, 0.2, 0.7, 0.0, 0.6, 0.3],
        [0.0, 0.0, 0.1, 0.6, 0.0, 0.9],
        [0.0, 0.0, 0.0, 0.3, 0.9, 0.0],
    ])
    return W


# ---------------------------------------------------------------------------
# apply_lesion: modify connectome
# ---------------------------------------------------------------------------

class TestApplyLesion:
    def test_single_region_zeros_row_and_column(self, toy_connectome):
        W_lesion = apply_lesion(toy_connectome, regions=[2])
        # Row 2 and column 2 should be zero
        np.testing.assert_allclose(W_lesion[2, :], 0.0)
        np.testing.assert_allclose(W_lesion[:, 2], 0.0)

    def test_other_connections_preserved(self, toy_connectome):
        W_lesion = apply_lesion(toy_connectome, regions=[2])
        # Connection 0-1 should be unchanged
        assert float(W_lesion[0, 1]) == pytest.approx(0.8)

    def test_multiple_regions(self, toy_connectome):
        W_lesion = apply_lesion(toy_connectome, regions=[1, 4])
        np.testing.assert_allclose(W_lesion[1, :], 0.0)
        np.testing.assert_allclose(W_lesion[:, 1], 0.0)
        np.testing.assert_allclose(W_lesion[4, :], 0.0)
        np.testing.assert_allclose(W_lesion[:, 4], 0.0)

    def test_shape_preserved(self, toy_connectome):
        W_lesion = apply_lesion(toy_connectome, regions=[0])
        assert W_lesion.shape == toy_connectome.shape

    def test_symmetry_preserved(self, toy_connectome):
        W_lesion = apply_lesion(toy_connectome, regions=[3])
        np.testing.assert_allclose(W_lesion, W_lesion.T)

    def test_partial_lesion(self, toy_connectome):
        """Partial lesion: reduce connections by a fraction rather than zeroing."""
        W_lesion = apply_lesion(toy_connectome, regions=[2], strength=0.5)
        # Row/column 2 should be halved, not zeroed
        expected = toy_connectome[2, :] * 0.5
        np.testing.assert_allclose(W_lesion[2, :], expected)

    def test_empty_regions_no_change(self, toy_connectome):
        W_lesion = apply_lesion(toy_connectome, regions=[])
        np.testing.assert_allclose(W_lesion, toy_connectome)


# ---------------------------------------------------------------------------
# lesion_effect: compare intact vs lesioned simulation output
# ---------------------------------------------------------------------------

class TestLesionEffect:
    def test_returns_dict(self, toy_connectome):
        """lesion_effect should return a dict with standard keys."""
        # Use a mock simulate function
        def simulate_fn(weights):
            key = jr.PRNGKey(0)
            T = 100
            n_regions = weights.shape[0]
            # Simple fake: activity proportional to degree
            deg = jnp.sum(weights, axis=1)
            return deg[:, None] * jnp.ones(T)[None, :]  # (n_regions, T)

        result = lesion_effect(
            toy_connectome, regions=[2], simulate_fn=simulate_fn
        )
        assert "intact" in result
        assert "lesioned" in result
        assert "difference" in result

    def test_difference_nonzero_for_connected_region(self, toy_connectome):
        def simulate_fn(weights):
            deg = jnp.sum(weights, axis=1)
            return deg[:, None] * jnp.ones(50)[None, :]

        result = lesion_effect(
            toy_connectome, regions=[2], simulate_fn=simulate_fn
        )
        # Region 2 is well-connected, so lesioning it should change things
        assert float(jnp.max(jnp.abs(result["difference"]))) > 0

    def test_difference_zero_for_isolated_region(self):
        # Region 5 only connects to itself (disconnected)
        W = jnp.eye(6) * 0  # no connections at all
        W = W.at[0, 1].set(1.0)
        W = W.at[1, 0].set(1.0)

        def simulate_fn(weights):
            deg = jnp.sum(weights, axis=1)
            return deg[:, None] * jnp.ones(50)[None, :]

        result = lesion_effect(W, regions=[5], simulate_fn=simulate_fn)
        # Lesioning region 5 (isolated) changes nothing
        np.testing.assert_allclose(result["difference"], 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# contribution_matrix: sweep all regions
# ---------------------------------------------------------------------------

class TestContributionMatrix:
    def test_shape(self, toy_connectome):
        def simulate_fn(weights):
            deg = jnp.sum(weights, axis=1)
            return deg[:, None] * jnp.ones(50)[None, :]

        C = contribution_matrix(toy_connectome, simulate_fn=simulate_fn)
        # (n_regions, n_timepoints)
        assert C.shape[0] == 6
        assert C.shape[1] == 50

    def test_hub_region_high_contribution(self, toy_connectome):
        """Highly connected regions should have larger lesion effects."""
        def simulate_fn(weights):
            deg = jnp.sum(weights, axis=1)
            return deg[:, None] * jnp.ones(50)[None, :]

        C = contribution_matrix(toy_connectome, simulate_fn=simulate_fn)
        # Region 3 has highest degree (0.2+0.7+0.6+0.3 = 1.8)
        mean_contribution = jnp.mean(C, axis=1)
        assert int(jnp.argmax(mean_contribution)) in [2, 3, 4]  # high-degree nodes


# ---------------------------------------------------------------------------
# virtual_lesion_sweep: multiple regions + targets
# ---------------------------------------------------------------------------

class TestVirtualLesionSweep:
    def test_returns_per_region_results(self, toy_connectome):
        def simulate_fn(weights):
            deg = jnp.sum(weights, axis=1)
            return deg[:, None] * jnp.ones(20)[None, :]

        results = virtual_lesion_sweep(
            toy_connectome, simulate_fn=simulate_fn
        )
        assert len(results) == 6  # one per region
        assert all("difference" in r for r in results)

    def test_subset_of_regions(self, toy_connectome):
        def simulate_fn(weights):
            return jnp.ones((6, 20))

        results = virtual_lesion_sweep(
            toy_connectome, simulate_fn=simulate_fn, regions=[0, 3, 5]
        )
        assert len(results) == 3


# ---------------------------------------------------------------------------
# local_network_transition: find when network effects dominate
# ---------------------------------------------------------------------------

class TestLocalNetworkTransition:
    def test_returns_time_index(self):
        """Given a contribution timecourse, find when network > local."""
        # Fake: target region dominates early, other regions dominate later
        C = jnp.zeros((6, 100))
        # Target region 2: high contribution at t=0-30
        C = C.at[2, :30].set(5.0)
        # Network regions: high contribution at t=50-100
        C = C.at[0, 50:].set(3.0)
        C = C.at[4, 50:].set(3.0)

        t_transition = local_network_transition(C, target_region=2)
        # Should be somewhere between 30 and 50
        assert 20 <= t_transition <= 60

    def test_all_local_returns_late(self):
        """If only the target region contributes, transition is at the end."""
        C = jnp.zeros((4, 50))
        C = C.at[1, :].set(10.0)
        t = local_network_transition(C, target_region=1)
        assert t >= 40  # near the end

    def test_all_network_returns_early(self):
        """If network dominates from the start, transition is early."""
        C = jnp.ones((4, 50)) * 5.0
        C = C.at[1, :].set(1.0)  # target is weakest
        t = local_network_transition(C, target_region=1)
        assert t <= 10
