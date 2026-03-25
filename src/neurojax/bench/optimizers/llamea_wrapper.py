"""LLaMEA optimizer wrapper — LLM-guided evolutionary algorithm discovery.

Uses Claude API as the LLM backend via a custom Anthropic_LLM class
(candidate for upstream contribution to LLaMEA). LLaMEA operates at a
different level than CMA-ES or Adam: it evolves entire *optimization
algorithms* as Python code, not just parameter vectors.

This is the Level 2 comparison (algorithm discovery) from the plan.
For Level 1 (direct parameter fitting), LLaMEA generates parameter
update rules that the wrapper evaluates.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from neurojax.bench.fitness import FitnessAdapter, FitnessResult
from neurojax.bench.optimizers.base import OptimizationResult


def _make_anthropic_llm(model: str, api_key: Optional[str] = None):
    """Create a LLaMEA-compatible LLM using the Anthropic API.

    This is a candidate for upstream contribution to XAI-liacs/LLaMEA.
    """
    from llamea import LLM

    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic is required for Anthropic_LLM. "
            "Install with: uv add anthropic"
        )

    class Anthropic_LLM(LLM):
        """Anthropic Claude backend for LLaMEA."""

        def __init__(self, api_key, model="claude-sonnet-4-20250514", temperature=0.8):
            super().__init__(api_key, model, None)
            self.client = anthropic.Anthropic(api_key=api_key)
            self.temperature = temperature
            logging.getLogger("anthropic").setLevel(logging.ERROR)
            logging.getLogger("httpx").setLevel(logging.ERROR)

        def query(self, session, max_tokens=4096):
            # Convert LLaMEA session format to Anthropic messages format
            # LLaMEA uses [{"role": "user"/"assistant", "content": "..."}]
            # Extract system message if first message is system-like
            system = ""
            messages = []
            for msg in session:
                if msg["role"] == "system":
                    system = msg["content"]
                else:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=self.temperature,
                system=system if system else anthropic.NOT_GIVEN,
                messages=messages,
            )
            return response.content[0].text

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    return Anthropic_LLM(api_key=key, model=model)


# Template prompt describing the optimization problem to LLaMEA
_TASK_PROMPT = """You are designing an optimization algorithm to fit a neural mass model
to empirical brain connectivity data.

The algorithm should be a Python function called `optimize` that:
1. Takes a callable `evaluate(params: dict) -> float` that returns the
   FC correlation (higher is better) for a given parameter dict.
2. Takes a `parameter_space: dict[str, tuple[float, float]]` mapping
   parameter names to (lower_bound, upper_bound).
3. Takes a `budget: int` — maximum number of evaluate() calls allowed.
4. Returns a tuple of (best_params: dict, best_score: float).

The function should implement a metaheuristic optimization algorithm.
You can use numpy but no other external libraries.

```python
import numpy as np

def optimize(evaluate, parameter_space, budget):
    # Your optimization algorithm here
    # Must call evaluate() no more than `budget` times
    # Return (best_params_dict, best_score)
    pass
```
"""


class LLaMEAWrapper:
    """LLaMEA optimizer — evolves optimization algorithms via Claude API.

    Requires the `llamea` package and an Anthropic API key.
    Falls back to a simple random search if LLaMEA is not available.
    """

    name: str = "LLaMEA-Claude"

    def __init__(
        self,
        llm_budget: int = 10,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
        """
        Args:
            llm_budget: Number of LLM calls (algorithm generations).
            model: Claude model to use.
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env.
        """
        self.llm_budget = llm_budget
        self.model = model
        self.api_key = api_key

    def optimize(
        self,
        adapter: FitnessAdapter,
        budget: int,
        seed: int = 0,
    ) -> OptimizationResult:
        t0 = time.perf_counter()

        try:
            return self._optimize_with_llamea(adapter, budget, seed, t0)
        except ImportError:
            # Fallback: random search
            return self._random_search_fallback(adapter, budget, seed, t0)

    def _optimize_with_llamea(
        self,
        adapter: FitnessAdapter,
        budget: int,
        seed: int,
        t0: float,
    ) -> OptimizationResult:
        from llamea import LLaMEA, Solution

        ps = adapter.parameter_space
        history = []
        best_fc = -float("inf")
        best_params: dict[str, float] = {}
        best_result: Optional[FitnessResult] = None
        total_evals = 0

        def fitness_fn(solution: Solution, logger=None) -> Solution:
            nonlocal total_evals, best_fc, best_params, best_result

            try:
                # Execute the evolved optimizer code
                namespace = {"np": np}
                exec(solution.code, namespace)
                evolved_optimize = namespace.get("optimize")

                if evolved_optimize is None:
                    solution.set_scores(
                        fitness=0.0,
                        feedback="No 'optimize' function found in generated code.",
                    )
                    return solution

                # Create evaluate function for the evolved optimizer
                eval_count = 0

                def evaluate(params: dict) -> float:
                    nonlocal eval_count, total_evals, best_fc, best_params, best_result
                    if eval_count >= budget:
                        return 0.0
                    result = adapter.evaluate(params)
                    eval_count += 1
                    total_evals += 1
                    if result.fc_correlation > best_fc:
                        best_fc = result.fc_correlation
                        best_params = dict(params)
                        best_result = result
                    return result.fc_correlation

                # Run the evolved optimizer
                evolved_params, evolved_score = evolved_optimize(
                    evaluate, dict(ps), budget
                )

                solution.set_scores(
                    fitness=evolved_score,
                    feedback=f"FC correlation: {evolved_score:.4f} "
                    f"in {eval_count} evaluations",
                )
            except Exception as e:
                solution.set_scores(
                    fitness=0.0,
                    feedback=f"Execution error: {str(e)[:200]}",
                )

            return solution

        # Create Anthropic LLM backend
        llm = _make_anthropic_llm(model=self.model, api_key=self.api_key)

        optimizer = LLaMEA(
            f=fitness_fn,
            llm=llm,
            task_prompt=_TASK_PROMPT,
            budget=self.llm_budget,
            n_parents=1,
            n_offspring=1,
            log=False,
            max_workers=1,
            parallel_backend="sequential",
            eval_timeout=120,
        )
        best_solution = optimizer.run()

        if best_result is None:
            best_result = FitnessResult(fc_correlation=0.0, fcd_ks_distance=1.0)

        return OptimizationResult(
            best_params=best_params,
            best_fitness=best_result,
            history=history,
            total_evaluations=total_evals,
            wall_time=time.perf_counter() - t0,
            optimizer_name=self.name,
            metadata={
                "evolved_code": best_solution.code if best_solution else None,
                "llm_budget": self.llm_budget,
                "model": self.model,
            },
        )

    def _random_search_fallback(
        self,
        adapter: FitnessAdapter,
        budget: int,
        seed: int,
        t0: float,
    ) -> OptimizationResult:
        """Simple random search fallback when LLaMEA is not available."""
        rng = np.random.default_rng(seed)
        ps = adapter.parameter_space
        param_names = list(ps.keys())

        history = []
        best_fc = -float("inf")
        best_params: dict[str, float] = {}
        best_result: Optional[FitnessResult] = None

        for i in range(budget):
            params = {
                name: float(rng.uniform(lo, hi))
                for name, (lo, hi) in ps.items()
            }
            result = adapter.evaluate(params)

            if result.fc_correlation > best_fc:
                best_fc = result.fc_correlation
                best_params = params
                best_result = result

            history.append({
                "gen": i + 1,
                "best_fc": best_fc,
                "mean_fc": result.fc_correlation,
                "n_evals": i + 1,
            })

        if best_result is None:
            best_result = FitnessResult(fc_correlation=0.0, fcd_ks_distance=1.0)

        return OptimizationResult(
            best_params=best_params,
            best_fitness=best_result,
            history=history,
            total_evaluations=budget,
            wall_time=time.perf_counter() - t0,
            optimizer_name="RandomSearch-fallback",
            metadata={"fallback": True, "reason": "llamea not installed"},
        )
