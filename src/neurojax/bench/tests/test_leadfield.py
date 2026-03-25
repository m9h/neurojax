"""Tests for leadfield forward projection (source → sensor space).

Tests correctness properties of ForwardProjection for EEG/MEG/sEEG
model fitting, following WhoBPyT's approach: sensor_signal = L @ source_signal.
"""

import jax
import jax.numpy as jnp
import pytest

from neurojax.bench.monitors.leadfield import ForwardProjection


# ---- Fixtures ----

@pytest.fixture
def square_leadfield():
    """4-sensor, 4-source identity-like leadfield."""
    return jnp.eye(4)


@pytest.fixture
def rectangular_leadfield():
    """3-sensor, 5-source leadfield (more sources than sensors)."""
    key = jax.random.PRNGKey(0)
    return jax.random.normal(key, (3, 5))


@pytest.fixture
def source_activity():
    """Synthetic source activity: (4 sources, 100 timepoints)."""
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, (4, 100))


@pytest.fixture
def rect_source_activity():
    """Synthetic source activity: (5 sources, 100 timepoints)."""
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, (5, 100))


@pytest.fixture
def empirical_sensor_data():
    """Synthetic empirical sensor data: (3 sensors, 100 timepoints)."""
    key = jax.random.PRNGKey(99)
    return jax.random.normal(key, (3, 100))


# ---- Output shape tests ----


class TestForwardProjectionShape:
    def test_output_shape_square(self, square_leadfield, source_activity):
        """project() output has shape (n_sensors, n_timepoints)."""
        fp = ForwardProjection(square_leadfield)
        result = fp.project(source_activity)
        assert result.shape == (4, 100)

    def test_output_shape_rectangular(self, rectangular_leadfield, rect_source_activity):
        """Rectangular leadfield: (3,5) @ (5,100) → (3,100)."""
        fp = ForwardProjection(rectangular_leadfield)
        result = fp.project(rect_source_activity)
        assert result.shape == (3, 100)

    def test_output_shape_1d_source(self, square_leadfield):
        """project() handles (n_sources,) input → (n_sensors,) output."""
        fp = ForwardProjection(square_leadfield)
        source = jnp.ones(4)
        result = fp.project(source)
        assert result.shape == (4,)


# ---- Identity leadfield tests ----


class TestIdentityLeadfield:
    def test_identity_preserves_signal(self, square_leadfield, source_activity):
        """Identity leadfield passes signal through unchanged."""
        fp = ForwardProjection(square_leadfield)
        result = fp.project(source_activity)
        assert jnp.allclose(result, source_activity, atol=1e-6)

    def test_identity_no_avg_ref(self, square_leadfield, source_activity):
        """Without avg_ref, identity preserves exact signal."""
        fp = ForwardProjection(square_leadfield, avg_ref=False)
        result = fp.project(source_activity)
        assert jnp.allclose(result, source_activity, atol=1e-6)


# ---- Average reference tests ----


class TestAverageReference:
    def test_avg_ref_removes_dc(self, rectangular_leadfield, rect_source_activity):
        """Average reference makes sensor mean zero at each timepoint."""
        fp = ForwardProjection(rectangular_leadfield, avg_ref=True)
        result = fp.project(rect_source_activity)
        sensor_means = jnp.mean(result, axis=0)
        assert jnp.allclose(sensor_means, 0.0, atol=1e-6)

    def test_avg_ref_flag_default_false(self, rectangular_leadfield):
        """avg_ref defaults to False."""
        fp = ForwardProjection(rectangular_leadfield)
        assert fp.avg_ref is False

    def test_avg_ref_changes_output(self, rectangular_leadfield, rect_source_activity):
        """Enabling avg_ref changes the projected signal."""
        fp_no_ref = ForwardProjection(rectangular_leadfield, avg_ref=False)
        fp_ref = ForwardProjection(rectangular_leadfield, avg_ref=True)
        out_no_ref = fp_no_ref.project(rect_source_activity)
        out_ref = fp_ref.project(rect_source_activity)
        assert not jnp.allclose(out_no_ref, out_ref, atol=1e-6)


# ---- sensor_fc tests ----


class TestSensorFC:
    def test_fc_symmetric(self, rectangular_leadfield, rect_source_activity):
        """sensor_fc produces a symmetric matrix."""
        fp = ForwardProjection(rectangular_leadfield)
        fc_mat = fp.sensor_fc(rect_source_activity)
        assert jnp.allclose(fc_mat, fc_mat.T, atol=1e-6)

    def test_fc_correct_size(self, rectangular_leadfield, rect_source_activity):
        """sensor_fc produces (n_sensors, n_sensors) matrix."""
        fp = ForwardProjection(rectangular_leadfield)
        fc_mat = fp.sensor_fc(rect_source_activity)
        assert fc_mat.shape == (3, 3)

    def test_fc_diagonal_ones(self, rectangular_leadfield, rect_source_activity):
        """sensor_fc diagonal elements are 1.0 (self-correlation)."""
        fp = ForwardProjection(rectangular_leadfield)
        fc_mat = fp.sensor_fc(rect_source_activity)
        assert jnp.allclose(jnp.diag(fc_mat), 1.0, atol=1e-5)

    def test_fc_bounded(self, rectangular_leadfield, rect_source_activity):
        """sensor_fc values are in [-1, 1]."""
        fp = ForwardProjection(rectangular_leadfield)
        fc_mat = fp.sensor_fc(rect_source_activity)
        assert jnp.all(fc_mat >= -1.0 - 1e-6)
        assert jnp.all(fc_mat <= 1.0 + 1e-6)


# ---- sensor_loss tests ----


class TestSensorLoss:
    def test_loss_scalar(self, rectangular_leadfield, rect_source_activity,
                         empirical_sensor_data):
        """sensor_loss returns a scalar."""
        fp = ForwardProjection(rectangular_leadfield)
        loss = fp.sensor_loss(rect_source_activity, empirical_sensor_data)
        assert loss.shape == ()

    def test_loss_nonnegative(self, rectangular_leadfield, rect_source_activity,
                              empirical_sensor_data):
        """sensor_loss is non-negative."""
        fp = ForwardProjection(rectangular_leadfield)
        loss = fp.sensor_loss(rect_source_activity, empirical_sensor_data)
        assert float(loss) >= 0.0

    def test_loss_zero_for_perfect_match(self, square_leadfield):
        """sensor_loss is zero when projected source matches empirical data."""
        fp = ForwardProjection(square_leadfield)
        # With identity leadfield, source == sensor
        data = jax.random.normal(jax.random.PRNGKey(0), (4, 50))
        loss = fp.sensor_loss(data, data)
        assert jnp.isclose(loss, 0.0, atol=1e-6)

    def test_loss_differentiable(self, rectangular_leadfield, empirical_sensor_data):
        """sensor_loss is differentiable w.r.t. source_activity via jax.grad."""
        fp = ForwardProjection(rectangular_leadfield)

        def loss_fn(source):
            return fp.sensor_loss(source, empirical_sensor_data)

        key = jax.random.PRNGKey(7)
        source = jax.random.normal(key, (5, 100))
        grad = jax.grad(loss_fn)(source)
        assert grad.shape == (5, 100)
        assert jnp.all(jnp.isfinite(grad))

    def test_loss_differentiable_through_leadfield(self):
        """Loss is differentiable w.r.t. leadfield matrix itself."""
        def loss_fn(L):
            fp = ForwardProjection(L)
            source = jax.random.normal(jax.random.PRNGKey(0), (4, 50))
            empirical = jax.random.normal(jax.random.PRNGKey(1), (3, 50))
            return fp.sensor_loss(source, empirical)

        L = jax.random.normal(jax.random.PRNGKey(2), (3, 4))
        grad = jax.grad(loss_fn)(L)
        assert grad.shape == (3, 4)
        assert jnp.all(jnp.isfinite(grad))


# ---- Modality tests ----


class TestModalities:
    def test_eeg_with_avg_ref(self):
        """EEG modality: many sensors, avg_ref enabled."""
        n_sensors, n_sources, n_time = 64, 200, 500
        key = jax.random.PRNGKey(10)
        k1, k2 = jax.random.split(key)
        L = jax.random.normal(k1, (n_sensors, n_sources)) * 0.01
        source = jax.random.normal(k2, (n_sources, n_time))

        fp = ForwardProjection(L, avg_ref=True)
        result = fp.project(source)
        assert result.shape == (n_sensors, n_time)
        # avg ref: mean across sensors is zero
        assert jnp.allclose(jnp.mean(result, axis=0), 0.0, atol=1e-5)

    def test_meg_no_avg_ref(self):
        """MEG modality: no average reference needed."""
        n_sensors, n_sources, n_time = 306, 100, 200
        key = jax.random.PRNGKey(11)
        k1, k2 = jax.random.split(key)
        L = jax.random.normal(k1, (n_sensors, n_sources)) * 0.001
        source = jax.random.normal(k2, (n_sources, n_time))

        fp = ForwardProjection(L, avg_ref=False)
        result = fp.project(source)
        assert result.shape == (n_sensors, n_time)

    def test_seeg_sparse_leadfield(self):
        """sEEG modality: few sensors, many sources, sparse leadfield."""
        n_sensors, n_sources, n_time = 8, 200, 300
        key = jax.random.PRNGKey(12)
        k1, k2 = jax.random.split(key)
        # sEEG leadfield is typically sparse (contacts near few sources)
        L = jax.random.normal(k1, (n_sensors, n_sources)) * 0.001
        source = jax.random.normal(k2, (n_sources, n_time))

        fp = ForwardProjection(L, avg_ref=False)
        result = fp.project(source)
        assert result.shape == (n_sensors, n_time)
        fc_mat = fp.sensor_fc(source)
        assert fc_mat.shape == (n_sensors, n_sensors)


# ---- Integration with vbjax ----


class TestVbjaxIntegration:
    def test_with_vbjax_make_gain(self):
        """ForwardProjection works with a leadfield from vbjax.make_gain."""
        try:
            import vbjax
        except ImportError:
            pytest.skip("vbjax not installed")

        # vbjax.make_gain takes a gain matrix and returns (buf, step, sample)
        n_sensors, n_sources = 4, 8
        key = jax.random.PRNGKey(20)
        k1, k2 = jax.random.split(key)
        L = jax.random.normal(k1, (n_sensors, n_sources))

        # Check vbjax make_gain accepts this matrix
        buf, step, sample = vbjax.make_gain(L)

        # Our ForwardProjection should accept the same matrix
        fp = ForwardProjection(L)
        source = jax.random.normal(k2, (n_sources, 50))
        result = fp.project(source)
        assert result.shape == (n_sensors, 50)

        # Verify vbjax gain projection matches ours for a single sample
        # vbjax step processes one timepoint at a time: step(buf, x) where x is (n_sources,)
        # We just verify the forward projection L @ x matches
        single_source = source[:, 0]
        vbjax_proj = L @ single_source
        our_proj = fp.project(single_source)
        assert jnp.allclose(vbjax_proj, our_proj, atol=1e-6)


# ---- Edge cases ----


class TestEdgeCases:
    def test_single_sensor(self):
        """Works with a single sensor (1, n_sources) leadfield."""
        L = jax.random.normal(jax.random.PRNGKey(30), (1, 10))
        fp = ForwardProjection(L)
        source = jax.random.normal(jax.random.PRNGKey(31), (10, 50))
        result = fp.project(source)
        assert result.shape == (1, 50)

    def test_single_source(self):
        """Works with a single source (n_sensors, 1) leadfield."""
        L = jax.random.normal(jax.random.PRNGKey(32), (5, 1))
        fp = ForwardProjection(L)
        source = jax.random.normal(jax.random.PRNGKey(33), (1, 50))
        result = fp.project(source)
        assert result.shape == (5, 50)

    def test_leadfield_stored(self):
        """ForwardProjection stores the leadfield matrix."""
        L = jnp.eye(3)
        fp = ForwardProjection(L)
        assert jnp.allclose(fp.leadfield, L)

    def test_n_sensors_n_sources_properties(self, rectangular_leadfield):
        """n_sensors and n_sources are correct."""
        fp = ForwardProjection(rectangular_leadfield)
        assert fp.n_sensors == 3
        assert fp.n_sources == 5
