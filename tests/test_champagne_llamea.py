"""TDD tests for LLaMEA-evolved CHAMPAGNE update rules.

RED phase: define evaluation framework for evolving SBL update rules.
GREEN phase: implement ChampagneLLaMEA adapter.

The core idea: instead of using Wipf's fixed convex bounding rule,
let LLaMEA (LLM-guided evolutionary algorithm) discover update rules
that minimise source localization error on known ground truth.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp


def _make_synthetic_dipole(n_src=50, n_sen=32, n_times=30, active_idx=20, seed=42):
    """Synthetic forward problem with known ground truth."""
    rng = np.random.RandomState(seed)
    positions = rng.randn(n_src, 3).astype(np.float32) * 50
    sensor_pos = rng.randn(n_sen, 3).astype(np.float32) * 80
    diff = sensor_pos[:, None, :] - positions[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1.0)
    L = (1.0 / (dist ** 2) / np.max(1.0 / (dist ** 2)) * 0.01).astype(np.float32)

    J_true = np.zeros((n_src, n_times), dtype=np.float32)
    J_true[active_idx] = np.sin(2 * np.pi * 10 * np.linspace(0, 0.5, n_times)).astype(np.float32)

    Y = L @ J_true + rng.randn(n_sen, n_times).astype(np.float32) * 1e-5
    noise_cov = np.eye(n_sen, dtype=np.float32) * 1e-10
    data_cov = (Y @ Y.T / n_times).astype(np.float32)

    return {
        'Y': jnp.array(Y), 'L': jnp.array(L),
        'data_cov': jnp.array(data_cov),
        'noise_cov': jnp.array(noise_cov),
        'positions': jnp.array(positions),
        'active_idx': active_idx, 'n_src': n_src,
    }


# ===========================================================================
# Test the evaluation harness
# ===========================================================================

class TestSourceLocalizationEvaluator:
    """Evaluator that scores an SBL update rule on synthetic data."""

    def test_evaluator_returns_score(self):
        from neurojax.source.champagne_llamea import SourceLocalizationEvaluator
        p = _make_synthetic_dipole()
        evaluator = SourceLocalizationEvaluator(
            gain=p['L'], data_cov=p['data_cov'], noise_cov=p['noise_cov'],
            true_active=[p['active_idx']], positions=p['positions']
        )
        # Score the standard Wipf update rule
        score = evaluator.score_update_rule(wipf_convex_bounding)
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_evaluator_prefers_correct_localisation(self):
        """A rule that finds the true source should score higher."""
        from neurojax.source.champagne_llamea import SourceLocalizationEvaluator
        p = _make_synthetic_dipole()
        evaluator = SourceLocalizationEvaluator(
            gain=p['L'], data_cov=p['data_cov'], noise_cov=p['noise_cov'],
            true_active=[p['active_idx']], positions=p['positions']
        )
        # Good rule vs bad rule (identity = no update)
        score_wipf = evaluator.score_update_rule(wipf_convex_bounding)
        score_identity = evaluator.score_update_rule(lambda g, G, C, N: g)
        assert score_wipf > score_identity

    def test_evaluator_multi_dipole(self):
        """Should work with multiple active sources."""
        from neurojax.source.champagne_llamea import SourceLocalizationEvaluator
        p = _make_synthetic_dipole()
        # Add a second active source
        evaluator = SourceLocalizationEvaluator(
            gain=p['L'], data_cov=p['data_cov'], noise_cov=p['noise_cov'],
            true_active=[20, 35], positions=p['positions']
        )
        score = evaluator.score_update_rule(wipf_convex_bounding)
        assert np.isfinite(score)


# ===========================================================================
# Test the LLaMEA adapter
# ===========================================================================

class TestChampagneLLaMEA:
    """LLaMEA adapter for evolving CHAMPAGNE update rules."""

    def test_task_prompt_exists(self):
        from neurojax.source.champagne_llamea import CHAMPAGNE_TASK_PROMPT
        assert 'update_gamma' in CHAMPAGNE_TASK_PROMPT
        assert 'jax.numpy' in CHAMPAGNE_TASK_PROMPT or 'jnp' in CHAMPAGNE_TASK_PROMPT

    def test_execute_evolved_code(self):
        """Should safely execute LLM-generated update rule code."""
        from neurojax.source.champagne_llamea import execute_update_rule
        code = '''
import jax.numpy as jnp

def update_gamma(gamma, gain, cov, noise_cov):
    return gamma * 0.9  # simple decay
'''
        fn = execute_update_rule(code)
        assert fn is not None
        gamma = jnp.ones(10)
        result = fn(gamma, jnp.ones((5, 10)), jnp.eye(5), jnp.eye(5))
        np.testing.assert_allclose(result, 0.9, atol=1e-5)

    def test_execute_bad_code_returns_none(self):
        """Malformed code should return None, not crash."""
        from neurojax.source.champagne_llamea import execute_update_rule
        fn = execute_update_rule("this is not valid python {{{")
        assert fn is None

    def test_sbl_with_evolved_rule(self):
        """Run SBL iterations with a custom update rule."""
        from neurojax.source.champagne_llamea import run_sbl_with_rule
        p = _make_synthetic_dipole()
        gamma, weights = run_sbl_with_rule(
            update_fn=wipf_convex_bounding,
            data_cov=p['data_cov'], gain=p['L'], noise_cov=p['noise_cov'],
            max_iter=20
        )
        assert gamma.shape == (p['n_src'],)
        assert weights.shape == (p['n_src'], p['L'].shape[0])
        assert jnp.all(jnp.isfinite(gamma))

    def test_evolved_rule_differentiable(self):
        """Update rules should be differentiable for end-to-end training."""
        from neurojax.source.champagne_llamea import run_sbl_with_rule
        p = _make_synthetic_dipole()

        def loss(noise_scale):
            noise_cov = jnp.eye(p['L'].shape[0]) * noise_scale
            gamma, W = run_sbl_with_rule(
                wipf_convex_bounding, p['data_cov'], p['L'], noise_cov,
                max_iter=5
            )
            return jnp.sum(gamma)

        grad = jax.grad(loss)(1e-10)
        assert jnp.isfinite(grad)


# ===========================================================================
# Test benchmark suite
# ===========================================================================

class TestBenchmarkSuite:
    """Multi-condition benchmark for comparing update rules."""

    def test_benchmark_multiple_snr(self):
        """Score should degrade gracefully with noise."""
        from neurojax.source.champagne_llamea import SourceLocalizationEvaluator
        scores = []
        for snr_db in [20, 10, 0]:
            p = _make_synthetic_dipole()
            signal_power = float(jnp.mean(p['Y'] ** 2))
            noise_std = np.sqrt(signal_power / (10 ** (snr_db / 10)))
            Y_noisy = p['Y'] + noise_std * jax.random.normal(
                jax.random.PRNGKey(snr_db), p['Y'].shape)
            data_cov = Y_noisy @ Y_noisy.T / Y_noisy.shape[1]

            evaluator = SourceLocalizationEvaluator(
                gain=p['L'], data_cov=data_cov, noise_cov=p['noise_cov'],
                true_active=[p['active_idx']], positions=p['positions']
            )
            scores.append(evaluator.score_update_rule(wipf_convex_bounding))

        # Higher SNR should generally give better scores
        assert scores[0] >= scores[-1] - 0.5

    def test_benchmark_multiple_sources(self):
        """Should handle 1, 2, and 3 simultaneous sources."""
        from neurojax.source.champagne_llamea import SourceLocalizationEvaluator
        for n_active in [1, 2, 3]:
            p = _make_synthetic_dipole()
            active = list(range(20, 20 + n_active))
            evaluator = SourceLocalizationEvaluator(
                gain=p['L'], data_cov=p['data_cov'], noise_cov=p['noise_cov'],
                true_active=active, positions=p['positions']
            )
            score = evaluator.score_update_rule(wipf_convex_bounding)
            assert np.isfinite(score)


# ===========================================================================
# Reference update rule for testing
# ===========================================================================

def wipf_convex_bounding(gamma, gain, cov, noise_cov):
    """Standard Wipf 2008 convex bounding rule (reference)."""
    n_chan = gain.shape[0]
    noise_reg = noise_cov + jnp.eye(n_chan) * jnp.trace(noise_cov) * 1e-6
    Sigma_y = jnp.dot(gain * gamma[None, :], gain.T) + noise_reg
    Sigma_inv = jnp.linalg.inv(Sigma_y)
    Z = Sigma_inv @ gain
    numer = jnp.sum((cov @ Z) * Z, axis=0)
    denom = jnp.sum(gain * Z, axis=0)
    ratio = jnp.clip(numer / jnp.maximum(denom, 1e-20), 0.0, 1e6)
    return jnp.clip(gamma * jnp.sqrt(ratio), 1e-20, 1e10)
