"""Experiment B: JR whole-brain FC fitting — 3 optimizers × 3 seeds.

Compares LLaMEA-Claude, CMA-ES, and Adam-JAX on fitting Jansen-Rit
model parameters to reproduce a target functional connectivity matrix
from a small connectome.

Usage:
    python experiments/run_jr_fc.py                    # all optimizers
    python experiments/run_jr_fc.py --cmaes-only       # CMA-ES baseline
    python experiments/run_jr_fc.py --budget 50        # quick test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neurojax.bench.adapters.vbjax_adapter import VbjaxFitnessAdapter, VbjaxSimConfig
from neurojax.bench.monitors.fc import fc, matrix_correlation
from neurojax.bench.optimizers.base import OptimizationResult


def make_toy_connectome():
    """4-node symmetric connectome (same as test_e2e.py)."""
    return jnp.array([
        [0., 0.5, 0.2, 0.],
        [0.5, 0., 0.3, 0.1],
        [0.2, 0.3, 0., 0.4],
        [0., 0.1, 0.4, 0.],
    ])


def make_target_fc(weights):
    """Generate target FC from known good JR parameters."""
    config = VbjaxSimConfig(
        dt=0.1, duration=10_000.0, bold_dt=500.0, warmup=2000.0, seed=99,
    )
    adapter = VbjaxFitnessAdapter(weights=weights, empirical_fc=jnp.eye(4), config=config)
    # Known good parameters
    target_params = {"A": 3.25, "B": 22.0, "a": 0.1, "b": 0.05,
                     "mu": 0.22, "I": 0.3, "K_gl": 0.01}
    result = adapter.evaluate(target_params)
    if result.simulated_fc is not None:
        return jnp.array(result.simulated_fc)
    return jnp.eye(4)


def make_adapter(weights, target_fc):
    config = VbjaxSimConfig(
        dt=0.1, duration=10_000.0, bold_dt=500.0, warmup=2000.0,
        noise_sigma=0.1, seed=42,
    )
    return VbjaxFitnessAdapter(weights=weights, empirical_fc=target_fc, config=config)


def make_optimizers(which="all"):
    optimizers = []

    from neurojax.bench.optimizers.cmaes_wrapper import CMAESWrapper
    optimizers.append(CMAESWrapper(sigma0=0.3))

    if which in ("all", "gradient"):
        try:
            from neurojax.bench.optimizers.gradient import GradientOptimizer
            optimizers.append(GradientOptimizer(learning_rate=1e-3))
        except Exception:
            print("GradientOptimizer not available")

    if which in ("all", "llamea"):
        import os
        if os.environ.get("ANTHROPIC_API_KEY"):
            from neurojax.bench.optimizers.llamea_wrapper import LLaMEAWrapper
            optimizers.append(LLaMEAWrapper(llm_budget=10))
        else:
            print("ANTHROPIC_API_KEY not set — skipping LLaMEA")

    return optimizers


def serialize_result(result: OptimizationResult) -> dict:
    return {
        "optimizer_name": result.optimizer_name,
        "best_params": result.best_params,
        "best_fc": result.best_fitness.fc_correlation,
        "total_evaluations": result.total_evaluations,
        "wall_time": result.wall_time,
        "history": result.history,
    }


def main():
    parser = argparse.ArgumentParser(description="JR whole-brain FC fitting")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--cmaes-only", action="store_true")
    parser.add_argument("--output", type=str, default="experiments/results_jr_fc.json")
    args = parser.parse_args()

    print("═══ Experiment B: JR Whole-Brain FC Fitting ═══")
    print(f"Budget: {args.budget} evaluations per optimizer per seed")
    print(f"Seeds: {args.seeds}")

    weights = make_toy_connectome()
    print("\nGenerating target FC from known parameters...")
    target_fc = make_target_fc(weights)
    print(f"Target FC shape: {target_fc.shape}")

    adapter = make_adapter(weights, target_fc)
    which = "cmaes" if args.cmaes_only else "all"
    optimizers = make_optimizers(which)

    print(f"Optimizers: {[o.name for o in optimizers]}")
    print(f"Parameter space: {list(adapter.parameter_space.keys())}")
    print()

    all_results = {}
    for opt in optimizers:
        print(f"\n─── {opt.name} ───")
        opt_results = []
        for seed in args.seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            t0 = time.perf_counter()
            result = opt.optimize(adapter, budget=args.budget, seed=seed)
            elapsed = time.perf_counter() - t0
            print(f"fc={result.best_fitness.fc_correlation:.4f}  "
                  f"evals={result.total_evaluations}  "
                  f"time={elapsed:.1f}s")
            opt_results.append(result)

        fcs = [r.best_fitness.fc_correlation for r in opt_results]
        print(f"  Mean: {np.mean(fcs):.4f} ± {np.std(fcs):.4f}  "
              f"Max: {np.max(fcs):.4f}")

        all_results[opt.name] = [serialize_result(r) for r in opt_results]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
