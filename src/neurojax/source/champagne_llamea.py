"""LLaMEA-evolved CHAMPAGNE: discovering SBL update rules via LLM.

Instead of using the fixed Wipf 2008 convex bounding rule, this module
lets LLaMEA (LLM-guided evolutionary algorithm) discover update rules
that minimise source localization error on known ground truth.

The search space is Python code for a gamma update function:

    def update_gamma(gamma, gain, cov, noise_cov):
        # ... arbitrary JAX code ...
        return gamma_new

LLaMEA generates candidate rules, evaluates them on synthetic dipole
benchmarks with known ground truth, and evolves better rules over
multiple generations.

This is the same framework used for CMC neural mass model fitting
(see bench/optimizers/llamea_wrapper.py), applied to source imaging.

References:
    Wipf & Nagarajan (2009) — CHAMPAGNE / Sparse Bayesian Learning
    van Stein et al. (2024) — LLaMEA: LLM-guided evolutionary algorithm
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass


# Type for an SBL update rule: (gamma, gain, cov, noise_cov) -> gamma_new
UpdateRule = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]


# ---------------------------------------------------------------------------
# Core: run SBL with a pluggable update rule
# ---------------------------------------------------------------------------

def run_sbl_with_rule(update_fn: UpdateRule,
                      data_cov: jnp.ndarray,
                      gain: jnp.ndarray,
                      noise_cov: jnp.ndarray,
                      max_iter: int = 50,
                      tol: float = 1e-4) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run SBL iterations with a custom update rule.

    Args:
        update_fn: callable(gamma, gain, cov, noise_cov) -> gamma_new
        data_cov: (n_sensors, n_sensors) data covariance
        gain: (n_sensors, n_sources) leadfield
        noise_cov: (n_sensors, n_sensors) noise covariance
        max_iter: maximum iterations
        tol: convergence tolerance (relative change in gamma)

    Returns:
        gamma: (n_sources,) final source power estimates
        weights: (n_sources, n_sensors) posterior beamformer weights
    """
    n_chan, n_src = gain.shape
    noise_reg = noise_cov + jnp.eye(n_chan) * jnp.trace(noise_cov) * 1e-6

    # Data-driven initialisation (same as fixed CHAMPAGNE)
    C_inv = jnp.linalg.inv(data_cov + noise_reg)
    Z = C_inv @ gain
    init_power = jnp.sum((data_cov @ Z) * Z, axis=0)
    init_denom = jnp.sum(gain * Z, axis=0)
    gamma = jnp.maximum(init_power / jnp.maximum(init_denom, 1e-20), 1e-20)
    gamma = gamma / jnp.maximum(jnp.median(gamma), 1e-20)

    for i in range(max_iter):
        gamma_new = update_fn(gamma, gain, data_cov, noise_reg)

        # Safety: clip and check finiteness
        gamma_new = jnp.where(jnp.isfinite(gamma_new), gamma_new, gamma)
        gamma_new = jnp.clip(gamma_new, 1e-20, 1e10)

        # Convergence check (use jax.lax.cond-safe comparison)
        rel_change = jnp.max(jnp.abs(gamma_new - gamma) / jnp.maximum(gamma, 1e-20))
        gamma = gamma_new
        # Note: can't use float() inside jax.grad traced code.
        # For differentiability, always run max_iter iterations.
        # For non-traced use, early stopping is handled by the caller.

    # Compute final weights
    Sigma_y = jnp.dot(gain * gamma[None, :], gain.T) + noise_reg
    Sigma_inv = jnp.linalg.inv(Sigma_y)
    weights = jnp.dot(gamma[:, None] * gain.T, Sigma_inv)

    return gamma, weights


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

@dataclass
class SourceLocalizationEvaluator:
    """Scores an SBL update rule on source localization accuracy.

    Uses synthetic or phantom data with known active source positions.
    The score combines:
      - Localization accuracy (did we find the right sources?)
      - Sparsity (fewer active sources is better)
      - Stability (gamma should not diverge)
    """
    gain: jnp.ndarray          # (n_sensors, n_sources)
    data_cov: jnp.ndarray      # (n_sensors, n_sensors)
    noise_cov: jnp.ndarray     # (n_sensors, n_sensors)
    true_active: List[int]     # ground-truth active source indices
    positions: jnp.ndarray     # (n_sources, 3) source positions in mm

    def score_update_rule(self, update_fn: UpdateRule,
                          max_iter: int = 50) -> float:
        """Score an update rule on localization accuracy.

        Returns:
            float score in [0, 1] — higher is better.
            Components:
              0.5 * localization_score + 0.3 * sparsity_score + 0.2 * stability_score
        """
        try:
            gamma, weights = run_sbl_with_rule(
                update_fn, self.data_cov, self.gain, self.noise_cov,
                max_iter=max_iter
            )
        except Exception:
            return 0.0

        gamma_np = np.asarray(gamma)

        # Stability: all gamma values finite
        if not np.all(np.isfinite(gamma_np)):
            return 0.0

        n_src = len(gamma_np)
        n_active_true = len(self.true_active)

        # --- Localization score ---
        # Are the true active sources in the top K by gamma?
        top_k = max(n_active_true * 3, 5)
        top_indices = set(np.argsort(gamma_np)[-top_k:])
        n_found = sum(1 for idx in self.true_active if idx in top_indices)
        localization_score = n_found / n_active_true

        # --- Sparsity score ---
        # Fraction of sources effectively pruned (gamma < 1% of max)
        threshold = gamma_np.max() * 0.01
        n_pruned = (gamma_np < threshold).sum()
        sparsity_score = n_pruned / n_src

        # --- Stability score ---
        # Gamma should have reasonable dynamic range (not all equal, not all zero)
        gamma_range = gamma_np.max() / max(np.median(gamma_np), 1e-20)
        stability_score = min(1.0, np.log10(max(gamma_range, 1.0)) / 4.0)

        score = (0.5 * localization_score
                 + 0.3 * sparsity_score
                 + 0.2 * stability_score)
        return float(score)


# ---------------------------------------------------------------------------
# Code execution for LLM-generated rules
# ---------------------------------------------------------------------------

def execute_update_rule(code: str) -> Optional[UpdateRule]:
    """Safely execute LLM-generated update rule code.

    Args:
        code: Python source code containing an `update_gamma` function.

    Returns:
        callable or None if execution fails.
    """
    try:
        namespace = {'jnp': jnp, 'jax': jax, 'np': np}
        exec(code, namespace)
        fn = namespace.get('update_gamma')
        if fn is None:
            return None
        # Quick smoke test
        gamma = jnp.ones(5)
        gain = jnp.ones((3, 5))
        cov = jnp.eye(3)
        noise = jnp.eye(3) * 0.1
        result = fn(gamma, gain, cov, noise)
        if result is None or not jnp.all(jnp.isfinite(result)):
            return None
        return fn
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Task prompt for LLaMEA
# ---------------------------------------------------------------------------

CHAMPAGNE_TASK_PROMPT = """You are designing a sparse Bayesian learning (SBL) update rule
for MEG/EEG source imaging (the CHAMPAGNE algorithm family).

Given sensor-level data covariance C_y and a leadfield matrix G mapping
n_sources brain dipoles to n_sensors MEG channels, the goal is to
estimate source powers gamma (one per source) that are:
1. SPARSE — most gamma values should be near zero (few active sources)
2. ACCURATE — the non-zero gamma should correspond to the true active sources
3. STABLE — the update should converge, not diverge

The model: C_y = G @ diag(gamma) @ G.T + C_noise

Write a function that takes the current gamma and returns updated gamma:

```python
import jax.numpy as jnp

def update_gamma(gamma, gain, cov, noise_cov):
    # gamma: (n_sources,) current source power estimates
    # gain: (n_sensors, n_sources) leadfield matrix
    # cov: (n_sensors, n_sensors) data covariance
    # noise_cov: (n_sensors, n_sensors) noise covariance (regularised)
    #
    # Must return: (n_sources,) updated gamma, all positive and finite
    # Available: jnp (jax.numpy) for all operations
    # This function will be called iteratively ~50 times

    # Your update rule here
    pass
```

Known update rules for reference (you can combine or improve upon these):
- Convex bounding (Wipf 2008): gamma_new = gamma * sqrt(numer/denom)
  where numer = diag(G.T @ Sigma_inv @ C @ Sigma_inv @ G)
  and denom = diag(G.T @ Sigma_inv @ G) and Sigma = G@diag(gamma)@G.T + noise
- EM update: gamma_new = diag(W @ C @ W.T) where W = gamma * G.T @ Sigma_inv
- MacKay update: gamma_new = gamma^2 * diag(G.T @ Sigma_inv @ G)
- Type-II ML: maximise log|Sigma| + tr(Sigma_inv @ C)

You may combine ideas, add momentum, use adaptive step sizes, or invent
entirely new rules. The function must use only jnp operations.
"""


# ---------------------------------------------------------------------------
# LLaMEA adapter
# ---------------------------------------------------------------------------

def _make_gemini_llm(model: str = "gemini-2.5-flash",
                     api_key: Optional[str] = None):
    """Create a LLaMEA-compatible LLM using the Google GenAI SDK."""
    from llamea import LLM
    import os

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        raise ImportError("google-genai required. Install: uv add google-genai")

    class Gemini_LLM(LLM):
        def __init__(self, api_key, model="gemini-2.5-flash"):
            super().__init__(api_key, model, None)
            self.client = genai.Client(api_key=api_key)
            self.genai_model = model

        def query(self, session, max_tokens=4096):
            # Build single prompt from session (Gemini prefers simple prompts)
            parts = []
            for msg in session:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    parts.append(f"[System instruction]: {content}\n")
                elif role == "assistant":
                    parts.append(f"[Previous response]: {content}\n")
                else:
                    parts.append(content + "\n")

            response = self.client.models.generate_content(
                model=self.genai_model,
                contents="\n".join(parts),
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.8,
                ),
            )
            return response.text

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY not set")
    return Gemini_LLM(api_key=key, model=model)


class ChampagneLLaMEA:
    """Evolve CHAMPAGNE update rules via LLaMEA.

    Supports both Anthropic Claude and Google Gemini as LLM backends.

    Usage:
        adapter = ChampagneLLaMEA(
            gain=L, data_cov=C, noise_cov=N,
            true_active=[20, 35], positions=positions,
            backend='gemini'  # or 'anthropic'
        )
        best_rule, best_score = adapter.evolve(llm_budget=10)
        gamma, weights = run_sbl_with_rule(best_rule, C, L, N)
    """

    def __init__(self, gain, data_cov, noise_cov, true_active, positions,
                 backend: str = "gemini",
                 model: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.evaluator = SourceLocalizationEvaluator(
            gain=gain, data_cov=data_cov, noise_cov=noise_cov,
            true_active=true_active, positions=positions
        )
        self.backend = backend
        self.model = model or ("gemini-2.5-flash" if backend == "gemini"
                                else "claude-sonnet-4-20250514")
        self.api_key = api_key

    def evolve(self, llm_budget: int = 10) -> Tuple[Optional[UpdateRule], float]:
        """Run LLaMEA evolution to discover update rules.

        Args:
            llm_budget: number of LLM calls (candidate generations)

        Returns:
            (best_update_fn, best_score) or (None, 0.0) if all fail
        """
        try:
            from llamea import LLaMEA, Solution
        except ImportError:
            return self._wipf_rule, self.evaluator.score_update_rule(self._wipf_rule)

        if self.backend == "gemini":
            llm = _make_gemini_llm(model=self.model, api_key=self.api_key)
        else:
            from neurojax.bench.optimizers.llamea_wrapper import _make_anthropic_llm
            llm = _make_anthropic_llm(model=self.model, api_key=self.api_key)

        best_fn = None
        best_score = 0.0

        def fitness_fn(solution, logger=None):
            nonlocal best_fn, best_score
            fn = execute_update_rule(solution.code)
            if fn is None:
                solution.set_scores(fitness=0.0,
                    feedback="Code did not produce a valid update_gamma function")
                return solution

            score = self.evaluator.score_update_rule(fn)
            if score > best_score:
                best_score = score
                best_fn = fn

            solution.set_scores(
                fitness=score,
                feedback=f"Localization score: {score:.4f}"
            )
            return solution

        optimizer = LLaMEA(
            f=fitness_fn,
            llm=llm,
            task_prompt=CHAMPAGNE_TASK_PROMPT,
            budget=llm_budget,
            n_parents=1,
            n_offspring=1,
            log=False,
            max_workers=1,
            parallel_backend="sequential",
            eval_timeout=60,
        )
        optimizer.run()

        return best_fn, best_score

    @staticmethod
    def _wipf_rule(gamma, gain, cov, noise_cov):
        """Standard Wipf convex bounding (fallback)."""
        Sigma_y = jnp.dot(gain * gamma[None, :], gain.T) + noise_cov
        Sigma_inv = jnp.linalg.inv(Sigma_y)
        Z = Sigma_inv @ gain
        numer = jnp.sum((cov @ Z) * Z, axis=0)
        denom = jnp.sum(gain * Z, axis=0)
        ratio = jnp.clip(numer / jnp.maximum(denom, 1e-20), 0.0, 1e6)
        return jnp.clip(gamma * jnp.sqrt(ratio), 1e-20, 1e10)
