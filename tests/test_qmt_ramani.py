"""Tests for JAX-ported Ramani QMT model with pulse shape characterization."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


class TestPulseShape:
    def test_gausshann_shape(self):
        from neurojax.qmri.qmt_ramani import gausshann_pulse
        t = jnp.linspace(0, 0.015, 100)
        pulse = gausshann_pulse(t, 0.015, bw=200.0)
        assert pulse.shape == (100,)
        assert jnp.all(jnp.isfinite(pulse))
        # Hanning zeros at endpoints, peak in middle
        assert float(pulse[0]) < float(pulse[50])
        assert float(pulse[-1]) < float(pulse[50])

    def test_pulse_amplitude(self):
        from neurojax.qmri.qmt_ramani import compute_pulse_amplitude, gausshann_pulse
        amp = compute_pulse_amplitude(332.0, 0.015, gausshann_pulse, bw=200.0)
        assert jnp.isfinite(amp)
        assert float(amp) > 0


class TestSfComputation:
    def test_sf_single_near_resonance(self):
        """Near-resonance MT pulse should produce measurable saturation."""
        from neurojax.qmri.qmt_ramani import compute_sf_single
        # 1000 Hz offset with 200Hz BW Gauss-Hanning: moderate saturation
        Sf = compute_sf_single(332.0, 1000.0, 12e-6, 0.015, n_steps=500)
        assert jnp.isfinite(Sf)
        assert float(Sf) < 0.95  # some saturation
        # Higher power (628°) gives more saturation
        Sf2 = compute_sf_single(628.0, 1000.0, 12e-6, 0.015, n_steps=500)
        assert float(Sf2) < float(Sf)  # more power → more saturation

    def test_sf_far_off_resonance(self):
        """Far off-resonance should have minimal saturation (Sf ≈ 1)."""
        from neurojax.qmri.qmt_ramani import compute_sf_single
        Sf = compute_sf_single(332.0, 50000.0, 12e-6, 0.015, n_steps=100)
        assert float(Sf) > 0.8  # minimal saturation far off-resonance

    def test_sf_table_shape(self):
        from neurojax.qmri.qmt_ramani import build_sf_table
        table = build_sf_table(
            angles_deg=[332.0, 628.0],
            offsets_hz=[1000.0, 5000.0, 50000.0],
            T2f_values=[10e-6, 12e-6, 15e-6],
            Trf=0.015, n_steps=50
        )
        assert table['values'].shape == (2, 3, 3)
        assert jnp.all(jnp.isfinite(table['values']))

    def test_sf_interpolation(self):
        from neurojax.qmri.qmt_ramani import build_sf_table, get_sf
        table = build_sf_table(
            angles_deg=[300.0, 400.0, 600.0],
            offsets_hz=[500.0, 2000.0, 10000.0, 50000.0],
            T2f_values=[8e-6, 12e-6, 18e-6],
            Trf=0.015, n_steps=50
        )
        # Interpolated value should be between grid points
        sf = get_sf(350.0, 5000.0, 11e-6, table)
        assert jnp.isfinite(sf)
        assert 0 < float(sf) < 1


class TestRamaniSignal:
    def test_signal_ratio_range(self):
        """MT signal ratio should be in [0, 1]."""
        from neurojax.qmri.qmt_ramani import build_sf_table, ramani_signal
        table = build_sf_table(
            angles_deg=[300.0, 400.0, 600.0, 700.0],
            offsets_hz=[500.0, 1000.0, 3000.0, 10000.0, 50000.0],
            T2f_values=[8e-6, 12e-6, 18e-6],
            Trf=0.015, n_steps=50
        )
        signal = ramani_signal(
            F=0.12, kf=3.0, R1f=1.0, T2f=12e-6, sf_table=table,
            angles_deg=[332.0, 332.0, 628.0, 628.0],
            offsets_hz=[1000.0, 50000.0, 1000.0, 50000.0],
            Trf=0.015, TR=0.055, readout_fa_deg=5.0
        )
        assert signal.shape == (4,)
        assert jnp.all(signal >= 0)
        assert jnp.all(signal <= 1)

    def test_higher_F_more_saturation(self):
        """Higher BPF → more MT effect → lower signal ratio."""
        from neurojax.qmri.qmt_ramani import build_sf_table, ramani_signal
        table = build_sf_table(
            angles_deg=[300.0, 700.0],
            offsets_hz=[500.0, 5000.0, 50000.0],
            T2f_values=[10e-6, 15e-6],
            Trf=0.015, n_steps=50
        )
        kw = dict(kf=3.0, R1f=1.0, T2f=12e-6, sf_table=table,
                  angles_deg=[628.0], offsets_hz=[2000.0],
                  Trf=0.015, TR=0.055, readout_fa_deg=5.0)
        sig_low = ramani_signal(F=0.05, **kw)
        sig_high = ramani_signal(F=0.20, **kw)
        assert float(sig_high[0]) < float(sig_low[0])


class TestQMTFit:
    def test_recovers_synthetic(self):
        """Recover F and kf from synthetic Ramani model data."""
        from neurojax.qmri.qmt_ramani import (
            build_sf_table, ramani_signal, qmt_fit_voxel
        )
        # Build table covering CUBRIC protocol range
        table = build_sf_table(
            angles_deg=[300.0, 400.0, 500.0, 600.0, 700.0],
            offsets_hz=[500.0, 1000.0, 3000.0, 10000.0, 30000.0, 50000.0],
            T2f_values=[8e-6, 10e-6, 12e-6, 15e-6, 20e-6],
            Trf=0.015, n_steps=100
        )
        # Ground truth
        F_true, kf_true, R1f = 0.12, 3.0, 1.1
        T2f_true = 12e-6
        angles = [332.0, 332.0, 628.0, 628.0, 628.0, 628.0, 333.0]
        offsets = [56360.0, 1000.0, 47180.0, 12060.0, 2750.0, 2770.0, 1000.0]

        data = ramani_signal(F_true, kf_true, R1f, T2f_true, table,
                              angles, offsets, 0.015, 0.055, 5.0)

        result = qmt_fit_voxel(data, table, angles, offsets,
                                0.015, 0.055, 5.0, R1f, n_iters=1000)

        assert abs(float(result['F']) - F_true) / F_true < 0.25, \
            f"F recovery: {float(result['F']):.3f} vs {F_true}"
        assert abs(float(result['kf']) - kf_true) / kf_true < 0.50, \
            f"kf recovery: {float(result['kf']):.2f} vs {kf_true}"

    def test_differentiable(self):
        """Loss gradients flow through Sf table interpolation."""
        from neurojax.qmri.qmt_ramani import build_sf_table, _qmt_loss
        table = build_sf_table(
            angles_deg=[300.0, 700.0],
            offsets_hz=[1000.0, 50000.0],
            T2f_values=[10e-6, 15e-6],
            Trf=0.015, n_steps=50
        )
        data = jnp.array([0.8, 0.95])
        params = jnp.array([0.10, jnp.log(3.0), 12e-6])
        grads = jax.grad(_qmt_loss)(
            params, data, table, [332.0], [1000.0],
            0.015, 0.055, 5.0, 1.0)
        assert jnp.all(jnp.isfinite(grads))
