"""Tests for DyNeMo — pure JAX variational autoencoder for brain network modes."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neurojax.models.dynemo import (
    DyNeMo,
    DyNeMoConfig,
    DyNeMoModule,
    GRUStack,
    BiGRUStack,
    InferenceNetwork,
    ModelNetwork,
    ObservationModel,
    _compute_loss,
    _flat_to_tril,
    _inverse_softplus,
    _segment_data,
    _tril_to_flat,
)


# ---------------------------------------------------------------------------
# Small config for fast CPU tests
# ---------------------------------------------------------------------------

def _small_config(**overrides) -> DyNeMoConfig:
    defaults = dict(
        n_modes=3,
        n_channels=5,
        sequence_length=50,
        inference_n_units=8,
        inference_n_layers=1,
        model_n_units=8,
        model_n_layers=1,
        batch_size=4,
        learning_rate=1e-3,
        n_epochs=3,
        do_kl_annealing=True,
        kl_annealing_n_epochs=2,
    )
    defaults.update(overrides)
    return DyNeMoConfig(**defaults)


@pytest.fixture
def small_data():
    """Synthetic data: 2 sessions, 200 timepoints, 5 channels."""
    k1, k2 = jr.split(jr.PRNGKey(42))
    return [jr.normal(k1, (200, 5)), jr.normal(k2, (150, 5))]


# ---------------------------------------------------------------------------
# Cholesky helpers
# ---------------------------------------------------------------------------

class TestCholeskyHelpers:
    def test_flat_to_tril_shape(self):
        n = 4
        n_flat = n * (n + 1) // 2
        flat = jnp.ones(n_flat)
        L = _flat_to_tril(flat, n)
        assert L.shape == (n, n)

    def test_flat_to_tril_lower_triangular(self):
        flat = jnp.arange(6, dtype=float)
        L = _flat_to_tril(flat, 3)
        # Upper triangle (excluding diagonal) should be zero
        assert float(L[0, 1]) == 0.0
        assert float(L[0, 2]) == 0.0
        assert float(L[1, 2]) == 0.0

    def test_flat_to_tril_positive_diagonal(self):
        """Diagonal uses softplus → always positive."""
        flat = jnp.array([-5.0, 0.1, -3.0, 0.2, 0.3, -2.0])
        L = _flat_to_tril(flat, 3)
        assert jnp.all(jnp.diag(L) > 0)

    def test_inverse_softplus_roundtrip(self):
        x = jnp.array([0.5, 1.0, 2.0, 5.0])
        recovered = jax.nn.softplus(_inverse_softplus(x))
        np.testing.assert_allclose(recovered, x, atol=1e-4)


# ---------------------------------------------------------------------------
# GRU modules
# ---------------------------------------------------------------------------

class TestGRUStack:
    def test_output_shape(self):
        key = jr.PRNGKey(0)
        gru = GRUStack(input_size=5, hidden_size=8, n_layers=1, key=key)
        x = jr.normal(key, (20, 5))
        out = gru(x)
        assert out.shape == (20, 8)

    def test_multi_layer(self):
        key = jr.PRNGKey(1)
        gru = GRUStack(input_size=5, hidden_size=8, n_layers=3, key=key)
        x = jr.normal(key, (20, 5))
        out = gru(x)
        assert out.shape == (20, 8)

    def test_output_finite(self):
        key = jr.PRNGKey(2)
        gru = GRUStack(input_size=3, hidden_size=4, n_layers=1, key=key)
        x = jr.normal(key, (10, 3))
        out = gru(x)
        assert jnp.all(jnp.isfinite(out))


class TestBiGRUStack:
    def test_output_shape(self):
        key = jr.PRNGKey(0)
        bigru = BiGRUStack(input_size=5, hidden_size=8, n_layers=1, key=key)
        x = jr.normal(key, (20, 5))
        out = bigru(x)
        assert out.shape == (20, 16)  # 2 * hidden_size


# ---------------------------------------------------------------------------
# InferenceNetwork
# ---------------------------------------------------------------------------

class TestInferenceNetwork:
    def test_output_shapes(self):
        key = jr.PRNGKey(0)
        net = InferenceNetwork(n_channels=5, n_units=8, n_layers=1, n_modes=3, key=key)
        x = jr.normal(key, (30, 5))
        mu, sigma = net(x)
        assert mu.shape == (30, 3)
        assert sigma.shape == (30, 3)

    def test_sigma_positive(self):
        key = jr.PRNGKey(1)
        net = InferenceNetwork(n_channels=5, n_units=8, n_layers=1, n_modes=3, key=key)
        x = jr.normal(key, (30, 5))
        _, sigma = net(x)
        assert jnp.all(sigma > 0)


# ---------------------------------------------------------------------------
# ModelNetwork
# ---------------------------------------------------------------------------

class TestModelNetwork:
    def test_output_shapes(self):
        key = jr.PRNGKey(0)
        net = ModelNetwork(n_modes=3, n_units=8, n_layers=1, key=key)
        theta = jr.normal(key, (20, 3))
        mu, sigma = net(theta)
        assert mu.shape == (20, 3)
        assert sigma.shape == (20, 3)

    def test_sigma_positive(self):
        key = jr.PRNGKey(1)
        net = ModelNetwork(n_modes=3, n_units=8, n_layers=1, key=key)
        theta = jr.normal(key, (20, 3))
        _, sigma = net(theta)
        assert jnp.all(sigma > 0)


# ---------------------------------------------------------------------------
# ObservationModel
# ---------------------------------------------------------------------------

class TestObservationModel:
    def test_means_shape(self):
        key = jr.PRNGKey(0)
        obs = ObservationModel(n_modes=3, n_channels=5, diagonal=False,
                               epsilon=1e-6, key=key)
        assert obs.means.shape == (3, 5)

    def test_covariances_shape(self):
        key = jr.PRNGKey(0)
        obs = ObservationModel(n_modes=3, n_channels=5, diagonal=False,
                               epsilon=1e-6, key=key)
        covs = obs.get_covariances()
        assert covs.shape == (3, 5, 5)

    def test_covariances_positive_definite(self):
        key = jr.PRNGKey(0)
        obs = ObservationModel(n_modes=3, n_channels=5, diagonal=False,
                               epsilon=1e-6, key=key)
        covs = obs.get_covariances()
        for k in range(3):
            eigs = jnp.linalg.eigvalsh(covs[k])
            assert jnp.all(eigs > 0), f"Mode {k} not positive definite"

    def test_diagonal_covariances(self):
        key = jr.PRNGKey(0)
        obs = ObservationModel(n_modes=2, n_channels=4, diagonal=True,
                               epsilon=1e-6, key=key)
        covs = obs.get_covariances()
        assert covs.shape == (2, 4, 4)
        # Off-diagonal should be zero for diagonal mode
        for k in range(2):
            off_diag = covs[k] - jnp.diag(jnp.diag(covs[k]))
            assert jnp.allclose(off_diag, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# DyNeMoModule
# ---------------------------------------------------------------------------

class TestDyNeMoModule:
    def test_construction(self):
        config = _small_config()
        key = jr.PRNGKey(0)
        module = DyNeMoModule(config, key=key)
        assert module.config == config
        assert module.inference_net is not None
        assert module.model_net is not None
        assert module.obs_model is not None


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

class TestComputeLoss:
    def test_loss_is_scalar(self):
        config = _small_config()
        key = jr.PRNGKey(0)
        module = DyNeMoModule(config, key=key)
        data = jr.normal(jr.PRNGKey(1), (50, 5))
        loss, info = _compute_loss(module, data, jr.PRNGKey(2))
        assert loss.shape == ()

    def test_loss_is_finite(self):
        config = _small_config()
        key = jr.PRNGKey(0)
        module = DyNeMoModule(config, key=key)
        data = jr.normal(jr.PRNGKey(1), (50, 5))
        loss, info = _compute_loss(module, data, jr.PRNGKey(2))
        assert jnp.isfinite(loss), f"Loss is {loss}"

    def test_info_has_components(self):
        config = _small_config()
        module = DyNeMoModule(config, key=jr.PRNGKey(0))
        data = jr.normal(jr.PRNGKey(1), (50, 5))
        _, info = _compute_loss(module, data, jr.PRNGKey(2))
        assert "nll_loss" in info
        assert "kl_loss" in info
        assert "alpha" in info

    def test_alpha_sums_to_one(self):
        config = _small_config()
        module = DyNeMoModule(config, key=jr.PRNGKey(0))
        data = jr.normal(jr.PRNGKey(1), (50, 5))
        _, info = _compute_loss(module, data, jr.PRNGKey(2))
        alpha = info["alpha"]
        row_sums = alpha.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_alpha_nonnegative(self):
        config = _small_config()
        module = DyNeMoModule(config, key=jr.PRNGKey(0))
        data = jr.normal(jr.PRNGKey(1), (50, 5))
        _, info = _compute_loss(module, data, jr.PRNGKey(2))
        assert jnp.all(info["alpha"] >= 0)

    def test_alpha_shape(self):
        config = _small_config()
        module = DyNeMoModule(config, key=jr.PRNGKey(0))
        data = jr.normal(jr.PRNGKey(1), (50, 5))
        _, info = _compute_loss(module, data, jr.PRNGKey(2))
        assert info["alpha"].shape == (50, 3)

    def test_kl_weight_zero_eliminates_kl(self):
        config = _small_config()
        module = DyNeMoModule(config, key=jr.PRNGKey(0))
        data = jr.normal(jr.PRNGKey(1), (50, 5))
        loss_w0, info_w0 = _compute_loss(module, data, jr.PRNGKey(2), kl_weight=0.0)
        loss_w1, info_w1 = _compute_loss(module, data, jr.PRNGKey(2), kl_weight=1.0)
        # With kl_weight=0, loss should equal nll only
        assert float(loss_w0) == pytest.approx(float(info_w0["nll_loss"]), rel=1e-5)
        # With kl_weight=1, loss should be larger (KL >= 0)
        assert float(loss_w1) >= float(loss_w0) - 0.01


# ---------------------------------------------------------------------------
# Data segmentation
# ---------------------------------------------------------------------------

class TestSegmentData:
    def test_output_shape(self):
        data = [jr.normal(jr.PRNGKey(0), (200, 5))]
        segments = _segment_data(data, 50, key=jr.PRNGKey(1))
        assert segments.shape == (4, 50, 5)  # 200 // 50 = 4

    def test_short_sequence_padded(self):
        data = [jr.normal(jr.PRNGKey(0), (30, 5))]
        segments = _segment_data(data, 50, key=jr.PRNGKey(1))
        assert segments.shape == (1, 50, 5)

    def test_multiple_sessions(self):
        data = [
            jr.normal(jr.PRNGKey(0), (100, 5)),
            jr.normal(jr.PRNGKey(1), (200, 5)),
        ]
        segments = _segment_data(data, 50, key=jr.PRNGKey(2))
        assert segments.shape == (6, 50, 5)  # 2 + 4


# ---------------------------------------------------------------------------
# DyNeMo high-level API
# ---------------------------------------------------------------------------

class TestDyNeMo:
    def test_construction(self):
        model = DyNeMo(n_modes=4, n_channels=5)
        assert model.config.n_modes == 4
        assert model.config.n_channels == 5

    def test_construction_with_config(self):
        config = _small_config()
        model = DyNeMo(config)
        assert model.config.n_modes == 3

    def test_repr(self):
        model = DyNeMo(n_modes=8, n_channels=80)
        assert "n_modes=8" in repr(model)

    def test_fit_returns_history(self, small_data):
        config = _small_config(n_epochs=2)
        model = DyNeMo(config)
        history = model.fit(small_data, n_epochs=2)
        assert len(history) == 2
        assert "loss" in history[0]
        assert "nll_loss" in history[0]
        assert "kl_loss" in history[0]

    def test_fit_loss_finite(self, small_data):
        config = _small_config(n_epochs=2)
        model = DyNeMo(config)
        history = model.fit(small_data, n_epochs=2)
        for h in history:
            assert np.isfinite(h["loss"]), f"Loss is {h['loss']}"

    def test_infer_shapes(self, small_data):
        config = _small_config(n_epochs=2)
        model = DyNeMo(config)
        model.fit(small_data, n_epochs=2)
        alphas = model.infer(small_data)
        assert len(alphas) == 2
        assert alphas[0].shape == (200, 3)
        assert alphas[1].shape == (150, 3)

    def test_infer_sums_to_one(self, small_data):
        config = _small_config(n_epochs=2)
        model = DyNeMo(config)
        model.fit(small_data, n_epochs=2)
        alphas = model.infer(small_data)
        for a in alphas:
            row_sums = a.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_infer_nonnegative(self, small_data):
        config = _small_config(n_epochs=2)
        model = DyNeMo(config)
        model.fit(small_data, n_epochs=2)
        alphas = model.infer(small_data)
        for a in alphas:
            assert jnp.all(a >= 0)

    def test_get_alpha_alias(self, small_data):
        config = _small_config(n_epochs=1)
        model = DyNeMo(config)
        model.fit(small_data, n_epochs=1)
        a1 = model.infer(small_data)
        a2 = model.get_alpha(small_data)
        for x, y in zip(a1, a2):
            np.testing.assert_allclose(x, y)

    def test_get_means_shape(self, small_data):
        config = _small_config(n_epochs=1)
        model = DyNeMo(config)
        model.fit(small_data, n_epochs=1)
        means = model.get_means()
        assert means.shape == (3, 5)

    def test_get_covariances_shape(self, small_data):
        config = _small_config(n_epochs=1)
        model = DyNeMo(config)
        model.fit(small_data, n_epochs=1)
        covs = model.get_covariances()
        assert covs.shape == (3, 5, 5)

    def test_get_covariances_positive_definite(self, small_data):
        config = _small_config(n_epochs=1)
        model = DyNeMo(config)
        model.fit(small_data, n_epochs=1)
        covs = model.get_covariances()
        for k in range(3):
            eigs = jnp.linalg.eigvalsh(covs[k])
            assert jnp.all(eigs > 0)

    def test_infer_before_fit_raises(self):
        model = DyNeMo(n_modes=3, n_channels=5)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.infer([jr.normal(jr.PRNGKey(0), (50, 5))])

    def test_single_array_input(self, small_data):
        """Passing a single 2D array instead of a list should work."""
        config = _small_config(n_epochs=1)
        model = DyNeMo(config)
        model.fit(small_data[0], n_epochs=1)  # single array
        alphas = model.infer(small_data[0])
        assert len(alphas) == 1
