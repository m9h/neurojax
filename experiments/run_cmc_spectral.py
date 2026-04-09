"""Experiment A: CMC spectral parameter tuning — 3 optimizers × 5 seeds.

Compares LLaMEA-Claude, CMA-ES, and gradient descent (Adam-JAX) on
tuning the canonical microcircuit's 9 connectivity parameters to produce
alpha-band oscillations.

Usage:
    python experiments/run_cmc_spectral.py                     # all optimizers
    python experiments/run_cmc_spectral.py --cmaes-only        # CMA-ES baseline only
    python experiments/run_cmc_spectral.py --seeds 0 1 2       # fewer seeds
    python experiments/run_cmc_spectral.py --budget 100        # fewer evaluations
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neurojax.bench.adapters.cmc_spectral_adapter import (
    CMCSpectralAdapter,
    CMCSpectralConfig,
)
from neurojax.bench.optimizers.base import OptimizationResult
from neurojax.bench.runner import BenchmarkConfig, BenchmarkRunner


def make_adapter():
    config = CMCSpectralConfig(
        dt=0.5,
        duration=8000.0,
        warmup=2000.0,
        noise_sigma=1e-3,
        seed=42,
        target_freq_lo=8.0,
        target_freq_hi=13.0,
        min_amplitude=0.1,
        nperseg=1024,
    )
    return CMCSpectralAdapter(config=config)


def make_optimizers(which="all"):
    optimizers = []

    # CMA-ES (always available)
    from neurojax.bench.optimizers.cmaes_wrapper import CMAESWrapper
    optimizers.append(CMAESWrapper(sigma0=0.3))

    if which in ("all", "gradient"):
        try:
            from neurojax.bench.optimizers.gradient import GradientOptimizer
            # GradientOptimizer requires VbjaxFitnessAdapter; CMCSpectralAdapter
            # has a loss() method but GradientOptimizer does an isinstance check.
            # Skip for now — gradient comparison uses cmc_tune_defaults.py instead.
            print("GradientOptimizer skipped (requires VbjaxFitnessAdapter type check)")
        except ImportError:
            print("GradientOptimizer not available (needs optax)")

    if which in ("all", "llamea"):
        import os
        if os.environ.get("GEMINI_API_KEY"):
            from neurojax.bench.optimizers.llamea_wrapper import LLaMEAWrapper
            optimizers.append(LLaMEAWrapper(llm_budget=10, backend="gemini"))
        elif os.environ.get("ANTHROPIC_API_KEY"):
            from neurojax.bench.optimizers.llamea_wrapper import LLaMEAWrapper
            optimizers.append(LLaMEAWrapper(llm_budget=10, backend="anthropic"))
        else:
            print("No LLM API key found — skipping LLaMEA (set GEMINI_API_KEY or ANTHROPIC_API_KEY)")

    return optimizers


def serialize_result(result: OptimizationResult) -> dict:
    """Convert OptimizationResult to JSON-safe dict."""
    return {
        "optimizer_name": result.optimizer_name,
        "best_params": result.best_params,
        "best_fc": result.best_fitness.fc_correlation,
        "best_fcd": result.best_fitness.fcd_ks_distance,
        "raw_objectives": result.best_fitness.raw_objectives,
        "total_evaluations": result.total_evaluations,
        "wall_time": result.wall_time,
        "history": result.history,
        "metadata": {
            k: v for k, v in result.metadata.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        },
    }


def main():
    parser = argparse.ArgumentParser(description="CMC spectral parameter tuning")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--cmaes-only", action="store_true")
    parser.add_argument("--output", type=str, default="experiments/results_cmc_spectral.json")
    args = parser.parse_args()

    print("═══ Experiment A: CMC Spectral Parameter Tuning ═══")
    print(f"Budget: {args.budget} evaluations per optimizer per seed")
    print(f"Seeds: {args.seeds}")
    print()

    adapter = make_adapter()
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
            print(f"fitness={result.best_fitness.fc_correlation:.4f}  "
                  f"evals={result.total_evaluations}  "
                  f"time={elapsed:.1f}s")
            opt_results.append(result)

        fcs = [r.best_fitness.fc_correlation for r in opt_results]
        print(f"  Mean: {np.mean(fcs):.4f} ± {np.std(fcs):.4f}  "
              f"Max: {np.max(fcs):.4f}")

        all_results[opt.name] = [serialize_result(r) for r in opt_results]

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print best parameters from best run
    print("\n═══ Best Parameters Found ═══")
    for opt_name, runs in all_results.items():
        best_run = max(runs, key=lambda r: r["best_fc"])
        print(f"\n{opt_name} (fc={best_run['best_fc']:.4f}):")
        for k, v in best_run["best_params"].items():
            print(f"  {k} = {v:.2f}")
        if "peak_freq" in best_run.get("metadata", {}):
            print(f"  → peak_freq = {best_run['metadata']['peak_freq']:.1f} Hz")


if __name__ == "__main__":
    main()
