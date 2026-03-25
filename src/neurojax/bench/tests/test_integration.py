"""Integration tests for the wired-up VbjaxFitnessAdapter.

Tests that BOLD hemodynamics, leadfield projection, regional parameters,
multi-modal loss, and vmap batch evaluation all work together through
the adapter.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import vbjax

from neurojax.bench.adapters.regional import RegionalParameterSpace
from neurojax.bench.adapters.vbjax_adapter import (
    LossWeights,
    VbjaxFitnessAdapter,
    VbjaxSimConfig,
)
from neurojax.bench.fitness import FitnessResult
from neurojax.bench.monitors.fc import fc


# ---------------------------------------------------------------------------
# Shared config: short simulation for fast tests
# ---------------------------------------------------------------------------

def _short_config(**overrides) -> VbjaxSimConfig:
    defaults = dict(
        dt=0.1,
        duration=5_000.0,  # 5s
        bold_dt=500.0,
        warmup=1_000.0,
        seed=42,
    )
    defaults.update(overrides)
    return VbjaxSimConfig(**defaults)


def _default_jr_params():
    params = {k: getattr(vbjax.jr_default_theta, k)
              for k in ("A", "B", "a", "b", "mu", "I")}
    params["K_gl"] = 0.01
    return params


@pytest.fixture
def weights_4():
    return jnp.array([
        [0.0, 0.5, 0.2, 0.0],
        [0.5, 0.0, 0.3, 0.1],
        [0.2, 0.3, 0.0, 0.4],
        [0.0, 0.1, 0.4, 0.0],
    ])


@pytest.fixture
def target_fc_4():
    key = jax.random.PRNGKey(99)
    return fc(jax.random.normal(key, (4, 20)))


@pytest.fixture
def target_bold_4():
    """Synthetic BOLD target with enough timepoints for FCD (window_size=30)."""
    key = jax.random.PRNGKey(77)
    return jax.random.normal(key, (4, 40))


# ---------------------------------------------------------------------------
# Balloon-Windkessel BOLD integration
# ---------------------------------------------------------------------------

class TestBOLDIntegration:
    """Balloon-Windkessel BOLD replaces naive subsampling in the adapter."""

    def test_hemodynamics_on_by_default(self, weights_4, target_fc_4):
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            config=_short_config(),
        )
        assert adapter._bold_monitor is not None

    def test_hemodynamics_off(self, weights_4, target_fc_4):
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            config=_short_config(use_hemodynamics=False),
        )
        assert adapter._bold_monitor is None

    def test_bold_output_finite(self, weights_4, target_fc_4):
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            config=_short_config(),
        )
        result = adapter.evaluate(_default_jr_params())
        assert isinstance(result, FitnessResult)
        assert np.all(np.isfinite(result.simulated_bold))
        assert result.simulated_bold.shape[0] == 4

    def test_bold_differs_from_subsampled(self, weights_4, target_fc_4):
        """Hemodynamic BOLD should differ from naive subsampling."""
        adapter_hemo = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            config=_short_config(use_hemodynamics=True),
        )
        adapter_sub = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            config=_short_config(use_hemodynamics=False),
        )
        params = _default_jr_params()
        r1 = adapter_hemo.evaluate(params)
        r2 = adapter_sub.evaluate(params)
        # BOLD values should differ (hemodynamic transform is nonlinear)
        assert not np.allclose(r1.simulated_bold, r2.simulated_bold)


# ---------------------------------------------------------------------------
# Leadfield / sensor-space integration
# ---------------------------------------------------------------------------

class TestLeadfieldIntegration:
    """Leadfield forward projection wired into the adapter."""

    def test_sensor_loss_in_evaluate(self, weights_4, target_fc_4):
        # 3 sensors observing 4 sources
        leadfield = jnp.array([
            [1.0, 0.5, 0.0, 0.0],
            [0.0, 0.5, 1.0, 0.3],
            [0.2, 0.0, 0.3, 1.0],
        ])
        target_sensor = jax.random.normal(jax.random.PRNGKey(55), (3, 8))

        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            empirical_sensor=target_sensor,
            leadfield=leadfield,
            config=_short_config(),
            loss_weights=LossWeights(fc=1.0, sensor=0.5),
        )

        assert adapter._forward is not None
        result = adapter.evaluate(_default_jr_params())
        assert "sensor_mse" in result.raw_objectives

    def test_sensor_loss_in_multimodal_loss(self, weights_4, target_fc_4):
        leadfield = jnp.eye(4)  # identity: sensors = sources
        target_sensor = jax.random.normal(jax.random.PRNGKey(55), (4, 8))

        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            empirical_sensor=target_sensor,
            leadfield=leadfield,
            config=_short_config(),
            loss_weights=LossWeights(fc=1.0, sensor=0.5),
        )

        x = adapter.default_param_array()
        loss_val = float(adapter.loss(x))
        fc_only = float(adapter.fc_loss(x))
        # Multi-modal loss includes sensor term, so should differ from FC-only
        assert loss_val != pytest.approx(fc_only, abs=1e-6)

    def test_avg_ref_eeg(self, weights_4, target_fc_4):
        leadfield = jnp.ones((3, 4))
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            leadfield=leadfield,
            leadfield_avg_ref=True,
            config=_short_config(),
        )
        assert adapter._forward.avg_ref is True


# ---------------------------------------------------------------------------
# Regional parameters integration
# ---------------------------------------------------------------------------

class TestRegionalIntegration:
    """Per-region parameters wired into the adapter."""

    def test_regional_param_space(self, weights_4, target_fc_4):
        regional = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0)},
            n_regions=4,
        )
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            regional_params=regional,
            config=_short_config(),
        )
        ps = adapter.parameter_space
        # 1 global (K_gl) + 4 regional (A_0..A_3)
        assert len(ps) == 5
        assert "K_gl" in ps
        assert "A_0" in ps
        assert "A_3" in ps

    def test_regional_n_params(self, weights_4, target_fc_4):
        regional = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03), "B": (10.0, 40.0)},
            regional_params={"A": (2.0, 5.0), "I": (0.0, 1.0)},
            n_regions=4,
        )
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            regional_params=regional,
            config=_short_config(),
        )
        assert adapter.n_params == 2 + 2 * 4  # 2 global + 8 regional

    def test_regional_default_array(self, weights_4, target_fc_4):
        regional = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0)},
            n_regions=4,
        )
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            regional_params=regional,
            config=_short_config(),
        )
        x = adapter.default_param_array()
        assert x.shape == (5,)
        # K_gl midpoint = 0.015, A midpoints = 3.5
        assert float(x[0]) == pytest.approx(0.015)
        assert float(x[1]) == pytest.approx(3.5)

    def test_regional_loss_differentiable(self, weights_4, target_fc_4):
        regional = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0)},
            n_regions=4,
        )
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            regional_params=regional,
            config=_short_config(),
        )
        x = adapter.default_param_array()
        grad = jax.grad(adapter.fc_loss)(x)
        assert grad.shape == (5,)
        assert jnp.all(jnp.isfinite(grad))

    def test_regional_bounds(self, weights_4, target_fc_4):
        regional = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0)},
            n_regions=4,
        )
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            regional_params=regional,
            config=_short_config(),
        )
        lo, hi = adapter.bounds_arrays()
        assert lo.shape == (5,)
        assert hi.shape == (5,)
        assert jnp.all(lo < hi)


# ---------------------------------------------------------------------------
# Multi-modal loss
# ---------------------------------------------------------------------------

class TestMultiModalLoss:
    """Multi-modal loss combining FC + FCD + sensor terms."""

    def test_fc_only_loss(self, weights_4, target_fc_4):
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            config=_short_config(),
            loss_weights=LossWeights(fc=1.0, fcd=0.0, sensor=0.0),
        )
        x = adapter.default_param_array()
        loss = float(adapter.loss(x))
        fc_loss = float(adapter.fc_loss(x))
        assert loss == pytest.approx(fc_loss, rel=1e-5)

    def test_fc_plus_fcd_loss(self, weights_4, target_fc_4, target_bold_4):
        # Longer simulation needed for FCD (window_size=30 needs >=30 BOLD samples)
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            empirical_bold=target_bold_4,
            config=_short_config(duration=20_000.0, bold_dt=500.0, use_hemodynamics=False),
            loss_weights=LossWeights(fc=1.0, fcd=0.5, sensor=0.0),
        )
        x = adapter.default_param_array()
        multi_loss = float(adapter.loss(x))
        fc_only = float(adapter.fc_loss(x))
        # FCD adds a non-negative term, so multi >= fc_only
        assert multi_loss >= fc_only - 0.01  # small tolerance

    def test_loss_is_differentiable(self, weights_4, target_fc_4, target_bold_4):
        # Longer simulation for FCD
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            empirical_bold=target_bold_4,
            config=_short_config(duration=20_000.0, bold_dt=500.0, use_hemodynamics=False),
            loss_weights=LossWeights(fc=1.0, fcd=0.5),
        )
        x = adapter.default_param_array()
        grad = jax.grad(adapter.loss)(x)
        assert jnp.all(jnp.isfinite(grad))

    def test_zero_weights_disable_terms(self, weights_4, target_fc_4):
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            config=_short_config(),
            loss_weights=LossWeights(fc=0.0, fcd=0.0, sensor=0.0),
        )
        x = adapter.default_param_array()
        loss = float(adapter.loss(x))
        assert loss == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# vmap batch evaluation
# ---------------------------------------------------------------------------

class TestVmapBatch:
    """Batch evaluation with vmap parallelism."""

    def test_batch_returns_correct_count(self, weights_4, target_fc_4):
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            config=_short_config(use_hemodynamics=False),
        )
        params = _default_jr_params()
        results = adapter.evaluate_batch([params, params, params])
        assert len(results) == 3
        assert all(isinstance(r, FitnessResult) for r in results)

    def test_batch_empty_returns_empty(self, weights_4, target_fc_4):
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            config=_short_config(),
        )
        assert adapter.evaluate_batch([]) == []

    def test_batch_results_finite(self, weights_4, target_fc_4):
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            config=_short_config(use_hemodynamics=False),
        )
        params = _default_jr_params()
        results = adapter.evaluate_batch([params])
        assert np.all(np.isfinite(results[0].simulated_fc))


# ---------------------------------------------------------------------------
# Convenience methods
# ---------------------------------------------------------------------------

class TestConvenienceMethods:
    """Tests for n_params, default_param_array, bounds_arrays."""

    def test_n_params_global(self, weights_4, target_fc_4):
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            config=_short_config(),
        )
        assert adapter.n_params == 7  # default JR bounds

    def test_default_param_array_shape(self, weights_4, target_fc_4):
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            config=_short_config(),
        )
        x = adapter.default_param_array()
        assert x.shape == (7,)

    def test_bounds_arrays_shape(self, weights_4, target_fc_4):
        adapter = VbjaxFitnessAdapter(
            weights=weights_4,
            empirical_fc=target_fc_4,
            config=_short_config(),
        )
        lo, hi = adapter.bounds_arrays()
        assert lo.shape == (7,)
        assert hi.shape == (7,)
        assert jnp.all(lo < hi)
