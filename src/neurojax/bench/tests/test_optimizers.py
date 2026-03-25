"""Layer 3 tests for optimizer wrappers and BenchmarkRunner.

Tests that each optimizer conforms to the Optimizer protocol,
runs without crashing on a toy problem, and produces valid results.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neurojax.bench.fitness import FitnessResult
from neurojax.bench.monitors.fc import fc
from neurojax.bench.optimizers.base import Optimizer, OptimizationResult


@pytest.fixture
def vbjax_adapter(toy_connectome_4node):
    """VbjaxFitnessAdapter on toy connectome for optimizer testing."""
    from neurojax.bench.adapters.vbjax_adapter import (
        VbjaxFitnessAdapter,
        VbjaxSimConfig,
    )

    weights, _ = toy_connectome_4node
    config = VbjaxSimConfig(
        dt=0.1,
        duration=5_000.0,
        bold_dt=500.0,
        warmup=1_000.0,
        seed=42,
    )
    key = jax.random.PRNGKey(99)
    target_bold = jax.random.normal(key, (4, 20))
    target_fc = fc(target_bold)

    return VbjaxFitnessAdapter(
        weights=weights,
        empirical_fc=target_fc,
        config=config,
    )


class TestCMAESWrapper:
    def test_implements_protocol(self):
        from neurojax.bench.optimizers.cmaes_wrapper import CMAESWrapper
        opt = CMAESWrapper()
        assert isinstance(opt, Optimizer)

    def test_optimize_returns_result(self, vbjax_adapter):
        from neurojax.bench.optimizers.cmaes_wrapper import CMAESWrapper
        opt = CMAESWrapper(sigma0=0.1, popsize=4)
        result = opt.optimize(vbjax_adapter, budget=8, seed=0)

        assert isinstance(result, OptimizationResult)
        assert result.optimizer_name == "CMA-ES"
        assert result.total_evaluations > 0
        assert result.total_evaluations <= 12  # popsize=4, ~2-3 gens
        assert result.wall_time > 0.0
        assert len(result.history) > 0
        assert -1.0 <= result.best_fitness.fc_correlation <= 1.0

    def test_convergence_tracked(self, vbjax_adapter):
        from neurojax.bench.optimizers.cmaes_wrapper import CMAESWrapper
        opt = CMAESWrapper(sigma0=0.1, popsize=4)
        result = opt.optimize(vbjax_adapter, budget=12, seed=0)

        # History should have gen, best_fc, n_evals
        for entry in result.history:
            assert "gen" in entry
            assert "best_fc" in entry
            assert "n_evals" in entry

    def test_respects_budget(self, vbjax_adapter):
        from neurojax.bench.optimizers.cmaes_wrapper import CMAESWrapper
        opt = CMAESWrapper(sigma0=0.1, popsize=4)
        result = opt.optimize(vbjax_adapter, budget=8, seed=0)
        # Should not exceed budget by more than one generation
        assert result.total_evaluations <= 8 + 4  # budget + popsize


class TestGradientOptimizer:
    def test_implements_protocol(self):
        from neurojax.bench.optimizers.gradient import GradientOptimizer
        opt = GradientOptimizer()
        assert isinstance(opt, Optimizer)

    def test_requires_vbjax_adapter(self):
        from neurojax.bench.adapters.neurolib_adapter import NeurolibFitnessAdapter
        from neurojax.bench.optimizers.gradient import GradientOptimizer
        opt = GradientOptimizer()
        adapter = NeurolibFitnessAdapter(empirical_fc=np.eye(4))
        with pytest.raises(TypeError, match="VbjaxFitnessAdapter"):
            opt.optimize(adapter, budget=5)

    def test_optimize_returns_result(self, vbjax_adapter):
        from neurojax.bench.optimizers.gradient import GradientOptimizer
        opt = GradientOptimizer(learning_rate=1e-2)
        result = opt.optimize(vbjax_adapter, budget=3, seed=0)

        assert isinstance(result, OptimizationResult)
        assert result.optimizer_name == "Adam-JAX"
        assert result.wall_time > 0.0

    def test_history_has_grad_norm(self, vbjax_adapter):
        from neurojax.bench.optimizers.gradient import GradientOptimizer
        opt = GradientOptimizer(learning_rate=1e-2)
        result = opt.optimize(vbjax_adapter, budget=3, seed=0)

        for entry in result.history:
            assert "grad_norm" in entry


class TestLLaMEAWrapper:
    def test_implements_protocol(self):
        from neurojax.bench.optimizers.llamea_wrapper import LLaMEAWrapper
        opt = LLaMEAWrapper()
        assert isinstance(opt, Optimizer)

    def test_fallback_random_search(self, vbjax_adapter):
        """Without llamea installed, falls back to random search."""
        from neurojax.bench.optimizers.llamea_wrapper import LLaMEAWrapper
        opt = LLaMEAWrapper()
        result = opt.optimize(vbjax_adapter, budget=5, seed=42)

        assert isinstance(result, OptimizationResult)
        assert result.total_evaluations == 5
        assert result.wall_time > 0.0
        # Check it's using fallback
        assert result.metadata.get("fallback") or result.optimizer_name == "LLaMEA-Claude"


class TestBenchmarkRunner:
    def test_runs_multiple_optimizers(self, vbjax_adapter):
        from neurojax.bench.optimizers.cmaes_wrapper import CMAESWrapper
        from neurojax.bench.optimizers.llamea_wrapper import LLaMEAWrapper
        from neurojax.bench.runner import BenchmarkRunner, BenchmarkConfig

        config = BenchmarkConfig(seeds=[0])
        runner = BenchmarkRunner(config)

        optimizers = [
            CMAESWrapper(sigma0=0.1, popsize=4),
            LLaMEAWrapper(),  # will fallback to random search
        ]
        comparison = runner.run(vbjax_adapter, optimizers, budget=8)

        assert len(comparison.results) == 2
        assert "CMA-ES" in comparison.results
        summary = comparison.summary()
        assert len(summary) == 2
        for opt_name, stats in summary.items():
            assert "fc_mean" in stats
            assert "n_runs" in stats
            assert stats["n_runs"] == 1  # single seed
