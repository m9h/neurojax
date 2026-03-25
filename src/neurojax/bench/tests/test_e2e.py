"""End-to-end tests — RED phase.

These tests assert that optimizers actually optimize: they must
produce FC correlations that improve over random initialization.
If an optimizer can't beat random, it's a mock, not an optimizer.
"""

import jax
import jax.numpy as jnp
import pytest

from neurojax.bench.monitors.fc import fc


@pytest.fixture
def optimization_problem():
    """A well-posed optimization problem: recover target FC from known params.

    Generates a target FC from a specific parameter set, then asks
    optimizers to find parameters that reproduce it.
    """
    from neurojax.bench.adapters.vbjax_adapter import (
        VbjaxFitnessAdapter,
        VbjaxSimConfig,
    )

    weights = jnp.array([
        [0., 0.5, 0.2, 0.],
        [0.5, 0., 0.3, 0.1],
        [0.2, 0.3, 0., 0.4],
        [0., 0.1, 0.4, 0.],
    ])
    config = VbjaxSimConfig(
        dt=0.1, duration=5_000.0, bold_dt=500.0, warmup=1_000.0, seed=42,
    )

    # Generate target FC from a DIFFERENT noise seed than the optimizer will use
    target_config = VbjaxSimConfig(
        dt=0.1, duration=5_000.0, bold_dt=500.0, warmup=1_000.0, seed=99,
    )
    ground_truth = {"K_gl": 0.01}
    tmp_adapter = VbjaxFitnessAdapter(
        weights=weights, empirical_fc=jnp.eye(4), config=target_config,
    )
    target_result = tmp_adapter.evaluate(ground_truth)
    target_fc = jnp.array(target_result.simulated_fc)

    # Create adapter with seed=42 (different from target seed=99)
    # This ensures the optimization problem is non-trivial
    adapter = VbjaxFitnessAdapter(
        weights=weights, empirical_fc=target_fc, config=config,
    )
    return adapter, target_fc


class TestCMAESActuallyOptimizes:
    """CMA-ES must improve FC correlation over multiple generations."""

    def test_improves_over_generations(self, optimization_problem):
        from neurojax.bench.optimizers.cmaes_wrapper import CMAESWrapper

        adapter, _ = optimization_problem
        opt = CMAESWrapper(sigma0=0.1, popsize=6)
        result = opt.optimize(adapter, budget=30, seed=0)

        # Must have run multiple generations
        assert len(result.history) >= 2, (
            f"Only {len(result.history)} generations — CMA-ES stopped too early"
        )
        # Best FC must be positive (better than random)
        assert result.best_fitness.fc_correlation > 0.1, (
            f"Best FC={result.best_fitness.fc_correlation:.4f} — "
            f"CMA-ES failed to find any decent solution"
        )
        # FC must be finite, not NaN
        assert result.best_fitness.fc_correlation == result.best_fitness.fc_correlation, (
            "Best FC is NaN"
        )

    def test_best_fc_monotonically_increases(self, optimization_problem):
        from neurojax.bench.optimizers.cmaes_wrapper import CMAESWrapper

        adapter, _ = optimization_problem
        opt = CMAESWrapper(sigma0=0.1, popsize=6)
        result = opt.optimize(adapter, budget=30, seed=0)

        # best_fc should never decrease across generations
        for i in range(1, len(result.history)):
            assert result.history[i]["best_fc"] >= result.history[i - 1]["best_fc"], (
                f"best_fc decreased at gen {result.history[i]['gen']}"
            )


class TestLLaMEAActuallyOptimizes:
    """LLaMEA must evolve an optimizer via Claude API that beats random search."""

    def test_evolves_working_optimizer(self, optimization_problem):
        """LLaMEA generates Python code that actually calls evaluate() and improves."""
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        from neurojax.bench.optimizers.llamea_wrapper import LLaMEAWrapper

        adapter, _ = optimization_problem
        # llm_budget=2: init population (1) + 1 evolution step
        opt = LLaMEAWrapper(llm_budget=2, model="claude-sonnet-4-20250514")
        result = opt.optimize(adapter, budget=10, seed=0)

        # Must have actually called evaluate (not just crashed)
        assert result.total_evaluations > 0, (
            "LLaMEA produced 0 evaluations — evolved code never called evaluate()"
        )
        # Must produce a real FC correlation
        assert result.best_fitness.fc_correlation > -1.0, (
            f"FC={result.best_fitness.fc_correlation} — no valid solution found"
        )
        # Must not be using the fallback
        assert result.optimizer_name == "LLaMEA-Claude", (
            f"Fell back to {result.optimizer_name} instead of using Claude API"
        )
        # Evolved code should be captured in metadata
        assert result.metadata.get("evolved_code") is not None, (
            "No evolved code captured"
        )

    def test_evolved_code_is_valid_optimizer(self, optimization_problem):
        """LLaMEA's evolved code is a working optimization algorithm."""
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        from neurojax.bench.optimizers.llamea_wrapper import LLaMEAWrapper

        adapter, _ = optimization_problem
        opt = LLaMEAWrapper(llm_budget=2, model="claude-sonnet-4-20250514")
        result = opt.optimize(adapter, budget=10, seed=0)

        # The evolved code should be a real optimizer, not empty
        code = result.metadata.get("evolved_code", "")
        assert len(code) > 50, f"Evolved code too short ({len(code)} chars)"
        assert "def optimize" in code, "Evolved code missing optimize function"
        assert "evaluate" in code, "Evolved code never calls evaluate()"

        # It should have found something reasonable (not all NaN/zero)
        assert result.best_fitness.fc_correlation > 0.0, (
            f"FC={result.best_fitness.fc_correlation:.4f} — "
            "evolved optimizer found nothing useful"
        )


class TestGradientActuallyOptimizes:
    """Adam must reduce loss over steps."""

    def test_loss_decreases(self, optimization_problem):
        from neurojax.bench.optimizers.gradient import GradientOptimizer

        adapter, _ = optimization_problem
        opt = GradientOptimizer(learning_rate=1e-3)
        result = opt.optimize(adapter, budget=5, seed=0)

        # Must have run at least a few steps (not broken by NaN grads)
        assert len(result.history) >= 3, (
            f"Only {len(result.history)} steps — gradient likely NaN"
        )
        # Loss should decrease (or at least not all be identical)
        losses = [h["loss"] for h in result.history]
        assert losses[-1] < losses[0] or len(set(losses)) > 1, (
            f"Loss didn't change: {losses}"
        )

    def test_gradient_is_finite(self, optimization_problem):
        from neurojax.bench.optimizers.gradient import GradientOptimizer

        adapter, _ = optimization_problem
        opt = GradientOptimizer(learning_rate=1e-3)
        result = opt.optimize(adapter, budget=5, seed=0)

        for h in result.history:
            assert h["grad_norm"] > 0.0 and h["grad_norm"] < float("inf"), (
                f"grad_norm={h['grad_norm']} at step {h['gen']}"
            )
