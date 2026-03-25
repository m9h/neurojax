"""Layer 1 tests for fitness adapters.

Tests that each adapter conforms to the FitnessAdapter protocol and
produces valid FitnessResults independently, using the toy connectome.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neurojax.bench.fitness import FitnessAdapter, FitnessResult
from neurojax.bench.monitors.fc import fc


class TestVbjaxAdapter:
    """Layer 1: VbjaxFitnessAdapter runs independently."""

    @pytest.fixture
    def adapter(self, toy_connectome_4node):
        from neurojax.bench.adapters.vbjax_adapter import (
            VbjaxFitnessAdapter,
            VbjaxSimConfig,
        )

        weights, _ = toy_connectome_4node
        # Short simulation for testing — dt=0.1ms for JR stability
        config = VbjaxSimConfig(
            dt=0.1,
            duration=5_000.0,  # 5s
            bold_dt=500.0,  # subsample every 500ms
            warmup=1_000.0,
            seed=42,
        )
        # Create a synthetic empirical FC target
        key = jax.random.PRNGKey(99)
        target_bold = jax.random.normal(key, (4, 20))
        target_fc = fc(target_bold)

        return VbjaxFitnessAdapter(
            weights=weights,
            empirical_fc=target_fc,
            config=config,
        )

    def test_implements_protocol(self, adapter):
        """Adapter satisfies FitnessAdapter protocol."""
        assert isinstance(adapter, FitnessAdapter)

    def test_parameter_space_nonempty(self, adapter):
        """Has at least one parameter with valid bounds."""
        ps = adapter.parameter_space
        assert len(ps) > 0
        for name, (lo, hi) in ps.items():
            assert lo < hi, f"{name}: {lo} >= {hi}"

    def test_objectives_defined(self, adapter):
        """Has at least one objective."""
        objs = adapter.objectives
        assert len(objs) >= 1
        assert all(o.direction in ("maximize", "minimize") for o in objs)

    @staticmethod
    def _default_params():
        """JR default params + weak coupling — known to be stable."""
        import vbjax
        params = {k: getattr(vbjax.jr_default_theta, k)
                  for k in ("A", "B", "a", "b", "mu", "I")}
        params["K_gl"] = 0.01  # weak coupling for stability
        return params

    def test_evaluate_returns_fitness_result(self, adapter):
        """evaluate() returns a valid FitnessResult."""
        result = adapter.evaluate(self._default_params())
        assert isinstance(result, FitnessResult)
        assert -1.0 <= result.fc_correlation <= 1.0
        assert result.fcd_ks_distance >= 0.0
        assert result.wall_time > 0.0

    def test_evaluate_fc_is_finite(self, adapter):
        """Simulated FC matrix has no NaN/Inf."""
        result = adapter.evaluate(self._default_params())
        assert result.simulated_fc is not None
        assert np.all(np.isfinite(result.simulated_fc))

    def test_evaluate_bold_shape(self, adapter):
        """BOLD output has (n_regions, n_timepoints) shape."""
        result = adapter.evaluate(self._default_params())
        assert result.simulated_bold is not None
        assert result.simulated_bold.shape[0] == 4  # n_regions
        assert result.simulated_bold.shape[1] > 0  # n_timepoints

    def test_evaluate_batch(self, adapter):
        """Batch evaluation returns list of correct length."""
        params = self._default_params()
        results = adapter.evaluate_batch([params, params])
        assert len(results) == 2
        assert all(isinstance(r, FitnessResult) for r in results)

    def test_different_params_different_fc(self, adapter):
        """Different parameters produce different FC correlations."""
        params1 = self._default_params()
        params2 = self._default_params()
        params2["K_gl"] = 0.05  # Stronger coupling

        r1 = adapter.evaluate(params1)
        r2 = adapter.evaluate(params2)
        # They should differ (unless extremely unlikely coincidence)
        assert r1.fc_correlation != r2.fc_correlation


class TestNeurolibAdapter:
    """Layer 1: NeurolibFitnessAdapter basic protocol tests.

    These tests don't require neurolib to be installed — they test
    the adapter's interface and error handling.
    """

    def test_implements_protocol(self):
        from neurojax.bench.adapters.neurolib_adapter import NeurolibFitnessAdapter

        adapter = NeurolibFitnessAdapter(
            empirical_fc=np.eye(4),
        )
        assert isinstance(adapter, FitnessAdapter)

    def test_parameter_space(self):
        from neurojax.bench.adapters.neurolib_adapter import NeurolibFitnessAdapter

        adapter = NeurolibFitnessAdapter(empirical_fc=np.eye(4))
        ps = adapter.parameter_space
        assert len(ps) > 0
        for name, (lo, hi) in ps.items():
            assert lo < hi

    def test_objectives(self):
        from neurojax.bench.adapters.neurolib_adapter import NeurolibFitnessAdapter

        adapter = NeurolibFitnessAdapter(empirical_fc=np.eye(4))
        objs = adapter.objectives
        assert len(objs) >= 1

    def test_evaluate_handles_missing_neurolib(self):
        """Gracefully handles subprocess failure (e.g., neurolib not installed)."""
        from neurojax.bench.adapters.neurolib_adapter import NeurolibFitnessAdapter

        adapter = NeurolibFitnessAdapter(
            empirical_fc=np.eye(4),
            neurolib_python="/nonexistent/python",
        )
        result = adapter.evaluate({"K_gl": 2.0})
        assert isinstance(result, FitnessResult)
        # Should return error result, not crash
        assert result.fc_correlation == 0.0
