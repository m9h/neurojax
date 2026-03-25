"""Tests for TMS stimulus protocol and TEP observation model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neurojax.bench.monitors.leadfield import ForwardProjection
from neurojax.bench.monitors.tep import (
    extract_tep,
    extract_tep_sensor,
    tep_combined_loss,
    tep_gfp_loss,
    tep_waveform_loss,
)
from neurojax.bench.stimuli.tms import (
    TMSProtocol,
    make_stimulus_train,
    tms_waveform,
)


# =====================================================================
# TMSProtocol and waveform generation
# =====================================================================


class TestTMSProtocol:
    def test_default_values(self):
        proto = TMSProtocol(t_onset=100.0, target_region=0)
        assert proto.amplitude == 1.0
        assert proto.pulse_width == 1.0
        assert proto.waveform == "monophasic"
        assert proto.tau == 0.3

    def test_custom_values(self):
        proto = TMSProtocol(
            t_onset=50.0, target_region=5, amplitude=3.0,
            pulse_width=2.0, waveform="biphasic"
        )
        assert proto.t_onset == 50.0
        assert proto.target_region == 5
        assert proto.amplitude == 3.0


class TestTMSWaveform:
    def test_monophasic_peak_at_onset(self):
        proto = TMSProtocol(t_onset=10.0, target_region=0, amplitude=2.0)
        t = jnp.array([9.9, 10.0, 10.1, 10.5, 11.0, 11.5])
        w = tms_waveform(t, proto)
        # Before onset: zero
        assert float(w[0]) == 0.0
        # At onset: peak amplitude
        assert float(w[1]) == pytest.approx(2.0)
        # Decays after onset
        assert float(w[2]) < float(w[1])

    def test_monophasic_decays(self):
        proto = TMSProtocol(
            t_onset=0.0, target_region=0, amplitude=1.0,
            pulse_width=5.0, tau=1.0
        )
        t = jnp.linspace(0, 4, 50)
        w = tms_waveform(t, proto)
        # Monotonically decreasing
        diffs = jnp.diff(w)
        assert jnp.all(diffs <= 0)

    def test_zero_outside_window(self):
        proto = TMSProtocol(
            t_onset=10.0, target_region=0, amplitude=5.0, pulse_width=2.0
        )
        t = jnp.array([0.0, 5.0, 9.9, 12.1, 20.0])
        w = tms_waveform(t, proto)
        assert float(w[0]) == 0.0
        assert float(w[1]) == 0.0
        assert float(w[2]) == 0.0
        assert float(w[3]) == 0.0
        assert float(w[4]) == 0.0

    def test_biphasic_crosses_zero(self):
        proto = TMSProtocol(
            t_onset=0.0, target_region=0, amplitude=1.0,
            pulse_width=2.0, waveform="biphasic"
        )
        t = jnp.linspace(0, 1.99, 100)
        w = tms_waveform(t, proto)
        # Should have both positive and negative values
        assert float(jnp.max(w)) > 0
        assert float(jnp.min(w)) < 0

    def test_square_constant_amplitude(self):
        proto = TMSProtocol(
            t_onset=5.0, target_region=0, amplitude=3.0,
            pulse_width=2.0, waveform="square"
        )
        t = jnp.array([5.0, 5.5, 6.0, 6.5, 6.99])
        w = tms_waveform(t, proto)
        expected = jnp.full(5, 3.0)
        assert jnp.allclose(w, expected)

    def test_waveform_is_differentiable(self):
        proto = TMSProtocol(t_onset=0.0, target_region=0, amplitude=1.0, pulse_width=5.0)
        def loss(amp):
            p = TMSProtocol(t_onset=0.0, target_region=0, amplitude=amp, pulse_width=5.0)
            t = jnp.linspace(0, 4, 20)
            w = tms_waveform(t, p)
            return jnp.sum(w ** 2)
        grad = jax.grad(loss)(1.0)
        assert jnp.isfinite(grad)
        assert float(grad) > 0


class TestMakeStimulusTrain:
    def test_shape(self):
        proto = TMSProtocol(t_onset=50.0, target_region=2, amplitude=1.0)
        stim = make_stimulus_train(proto, n_regions=10, dt=0.1, duration=200.0)
        assert stim.shape == (2000, 10)

    def test_only_target_region_active(self):
        proto = TMSProtocol(
            t_onset=10.0, target_region=3, amplitude=2.0, pulse_width=1.0
        )
        stim = make_stimulus_train(proto, n_regions=8, dt=0.1, duration=50.0)
        # Non-target regions should be zero everywhere
        for r in range(8):
            if r != 3:
                assert float(jnp.max(jnp.abs(stim[:, r]))) == 0.0
        # Target region should have non-zero values
        assert float(jnp.max(stim[:, 3])) > 0.0

    def test_stimulus_at_correct_time(self):
        proto = TMSProtocol(
            t_onset=10.0, target_region=0, amplitude=1.0, pulse_width=2.0
        )
        stim = make_stimulus_train(proto, n_regions=4, dt=0.1, duration=50.0)
        # At t=10ms (index 100): should be active
        assert float(stim[100, 0]) > 0.0
        # At t=5ms (index 50): should be zero
        assert float(stim[50, 0]) == 0.0
        # At t=13ms (index 130): should be zero (past pulse_width)
        assert float(stim[130, 0]) == 0.0

    def test_multiple_pulses(self):
        p1 = TMSProtocol(t_onset=10.0, target_region=0, amplitude=1.0)
        p2 = TMSProtocol(t_onset=30.0, target_region=2, amplitude=2.0)
        stim = make_stimulus_train([p1, p2], n_regions=4, dt=0.1, duration=50.0)
        # Both regions should be active at their respective times
        assert float(stim[100, 0]) > 0.0  # t=10, region 0
        assert float(stim[300, 2]) > 0.0  # t=30, region 2
        # Cross-check: no crosstalk
        assert float(stim[100, 2]) == 0.0
        assert float(stim[300, 0]) == 0.0

    def test_spatial_spread(self):
        spread = jnp.array([1.0, 0.5, 0.2, 0.0])
        proto = TMSProtocol(
            t_onset=5.0, target_region=0, amplitude=1.0,
            pulse_width=1.0, spatial_spread=spread
        )
        stim = make_stimulus_train(proto, n_regions=4, dt=0.1, duration=20.0)
        # At onset, region ratios should match spatial_spread
        idx = 50  # t=5ms
        r0 = float(stim[idx, 0])
        r1 = float(stim[idx, 1])
        r2 = float(stim[idx, 2])
        r3 = float(stim[idx, 3])
        assert r0 > 0
        assert r1 == pytest.approx(r0 * 0.5, rel=1e-5)
        assert r2 == pytest.approx(r0 * 0.2, rel=1e-5)
        assert r3 == 0.0

    def test_single_protocol_as_arg(self):
        """Single TMSProtocol (not wrapped in list) should work."""
        proto = TMSProtocol(t_onset=5.0, target_region=0)
        stim = make_stimulus_train(proto, n_regions=4, dt=0.1, duration=20.0)
        assert stim.shape == (200, 4)


# =====================================================================
# TEP observation model
# =====================================================================


class TestExtractTEP:
    def test_correct_window(self):
        ts = jnp.ones((4, 1000))  # 4 regions, 1000 timepoints
        tep = extract_tep(ts, t_onset=50.0, dt=0.1, t_pre=10.0, t_post=100.0)
        # Window: (50-10)/0.1 = 400 to (50+100)/0.1 = 1500 → clamped to 1000
        expected_start = 400
        expected_end = 1000
        assert tep.shape == (4, expected_end - expected_start)

    def test_small_window(self):
        ts = jnp.arange(500).reshape(1, 500).astype(float)
        tep = extract_tep(ts, t_onset=20.0, dt=0.1, t_pre=5.0, t_post=10.0)
        # Window: (20-5)/0.1=150 to (20+10)/0.1=300 → 150 samples
        assert tep.shape == (1, 150)
        assert float(tep[0, 0]) == 150.0  # first value at index 150

    def test_clamps_to_start(self):
        ts = jnp.ones((2, 100))
        # t_onset=0, t_pre=10 → start would be -100, clamped to 0
        tep = extract_tep(ts, t_onset=0.0, dt=0.1, t_pre=10.0, t_post=5.0)
        assert tep.shape[1] == 50  # 0 to (0+5)/0.1=50


class TestExtractTEPSensor:
    def test_sensor_projection(self):
        # 4 sources, 100 timepoints
        ts = jnp.ones((4, 100)) * 0.5
        leadfield = jnp.eye(4)[:3, :]  # 3 sensors from 4 sources
        forward = ForwardProjection(leadfield)
        tep = extract_tep_sensor(ts, forward, t_onset=5.0, dt=0.1, t_pre=2.0, t_post=3.0)
        assert tep.shape[0] == 3  # n_sensors
        assert tep.shape[1] == 50  # (5-2)/0.1 to (5+3)/0.1 = 30 to 80 = 50


class TestTEPLoss:
    def test_waveform_loss_zero_for_identical(self):
        tep = jax.random.normal(jax.random.PRNGKey(0), (3, 50))
        loss = tep_waveform_loss(tep, tep)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_waveform_loss_positive_for_different(self):
        key1, key2 = jax.random.split(jax.random.PRNGKey(0))
        t1 = jax.random.normal(key1, (3, 50))
        t2 = jax.random.normal(key2, (3, 50))
        loss = tep_waveform_loss(t1, t2)
        assert float(loss) > 0.0

    def test_gfp_loss_zero_for_identical(self):
        tep = jax.random.normal(jax.random.PRNGKey(0), (5, 40))
        loss = tep_gfp_loss(tep, tep)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_gfp_loss_positive_for_different(self):
        key1, key2 = jax.random.split(jax.random.PRNGKey(0))
        t1 = jax.random.normal(key1, (5, 40))
        t2 = jax.random.normal(key2, (5, 40)) * 3.0
        loss = tep_gfp_loss(t1, t2)
        assert float(loss) > 0.0

    def test_combined_loss_respects_weights(self):
        key1, key2 = jax.random.split(jax.random.PRNGKey(0))
        t1 = jax.random.normal(key1, (3, 30))
        t2 = jax.random.normal(key2, (3, 30))

        loss_both = float(tep_combined_loss(t1, t2, w_waveform=1.0, w_gfp=1.0))
        loss_wave_only = float(tep_combined_loss(t1, t2, w_waveform=1.0, w_gfp=0.0))
        loss_gfp_only = float(tep_combined_loss(t1, t2, w_waveform=0.0, w_gfp=1.0))

        assert loss_both > loss_wave_only
        assert loss_both > loss_gfp_only
        assert loss_both == pytest.approx(loss_wave_only + loss_gfp_only, rel=1e-5)

    def test_loss_is_differentiable(self):
        key1, key2 = jax.random.split(jax.random.PRNGKey(0))
        emp = jax.random.normal(key1, (3, 30))

        def loss_fn(sim):
            return tep_combined_loss(sim, emp)

        sim = jax.random.normal(key2, (3, 30))
        grad = jax.grad(loss_fn)(sim)
        assert grad.shape == (3, 30)
        assert jnp.all(jnp.isfinite(grad))

    def test_zero_weights_return_zero(self):
        key1, key2 = jax.random.split(jax.random.PRNGKey(0))
        t1 = jax.random.normal(key1, (3, 30))
        t2 = jax.random.normal(key2, (3, 30))
        loss = tep_combined_loss(t1, t2, w_waveform=0.0, w_gfp=0.0)
        assert float(loss) == pytest.approx(0.0)


# =====================================================================
# End-to-end: stimulus → extract TEP → loss
# =====================================================================


class TestTMSTEPPipeline:
    """Integration test: make stimulus → inject into fake neural → extract TEP → loss."""

    def test_pipeline_shapes(self):
        n_regions = 8
        dt = 0.1
        duration = 100.0  # 100ms

        # Create TMS pulse at t=20ms targeting region 3
        proto = TMSProtocol(t_onset=20.0, target_region=3, amplitude=2.0)
        stim = make_stimulus_train(proto, n_regions=n_regions, dt=dt, duration=duration)
        assert stim.shape == (1000, 8)

        # Fake "neural activity" = stimulus itself (transposed for regions-first)
        neural = stim.T  # (8, 1000)

        # Extract TEP around the pulse
        tep = extract_tep(neural, t_onset=20.0, dt=dt, t_pre=5.0, t_post=50.0)
        assert tep.shape[0] == 8
        assert tep.shape[1] == 550  # (20-5)/0.1 to (20+50)/0.1 = 150 to 700

    def test_pipeline_differentiable(self):
        """Gradient flows through stimulus → TEP → loss."""
        n_regions = 4

        proto = TMSProtocol(t_onset=10.0, target_region=0, amplitude=1.0, pulse_width=2.0)
        stim = make_stimulus_train(proto, n_regions=n_regions, dt=0.1, duration=50.0)

        target_tep = jax.random.normal(jax.random.PRNGKey(0), (4, 200))

        def pipeline_loss(scale):
            neural = stim.T * scale  # (4, 500), scaled
            tep = extract_tep(neural, t_onset=10.0, dt=0.1, t_pre=5.0, t_post=15.0)
            return tep_waveform_loss(tep, target_tep)

        grad = jax.grad(pipeline_loss)(1.0)
        assert jnp.isfinite(grad)
