"""Tests for qBOLD regional OEF from multi-echo GRE data."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


class TestQBOLDSignalModel:
    """Forward model: S(TE) with R2 + R2' separation."""

    def test_signal_shape(self):
        from neurojax.qmri.qbold import qbold_signal
        TEs = jnp.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035])
        signal = qbold_signal(S0=1000.0, R2=15.0, R2_prime=5.0, TEs=TEs)
        assert signal.shape == (7,)
        assert jnp.all(jnp.isfinite(signal))
        assert jnp.all(signal > 0)

    def test_r2prime_zero_matches_monoexp(self):
        """R2'=0 should give same result as mono-exponential T2* decay."""
        from neurojax.qmri.qbold import qbold_signal
        TEs = jnp.array([0.005, 0.010, 0.020, 0.035])
        S0, R2 = 500.0, 30.0
        signal = qbold_signal(S0, R2, R2_prime=0.0, TEs=TEs)
        expected = S0 * jnp.exp(-TEs * R2)
        np.testing.assert_allclose(signal, expected, rtol=1e-5)

    def test_r2prime_increases_decay(self):
        """Higher R2' = faster signal decay (more deoxy blood)."""
        from neurojax.qmri.qbold import qbold_signal
        TEs = jnp.array([0.005, 0.010, 0.020, 0.035])
        S0, R2 = 500.0, 20.0
        sig_low = qbold_signal(S0, R2, R2_prime=2.0, TEs=TEs)
        sig_high = qbold_signal(S0, R2, R2_prime=10.0, TEs=TEs)
        # Higher R2' → lower signal at late echoes
        assert float(sig_high[-1]) < float(sig_low[-1])

    def test_differentiable(self):
        from neurojax.qmri.qbold import qbold_signal
        TEs = jnp.array([0.005, 0.010, 0.020, 0.035])
        def loss(params):
            return jnp.sum(qbold_signal(params[0], params[1], params[2], TEs) ** 2)
        grads = jax.grad(loss)(jnp.array([500.0, 20.0, 5.0]))
        assert jnp.all(jnp.isfinite(grads))


class TestQBOLDFit:
    """Voxelwise qBOLD fitting: separate R2 from R2'."""

    def test_recovers_synthetic(self):
        """Recover known R2 and R2' from noiseless synthetic data."""
        from neurojax.qmri.qbold import qbold_signal, qbold_fit_voxel
        TEs = jnp.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035])
        S0_true, R2_true, R2p_true = 800.0, 18.0, 6.0
        data = qbold_signal(S0_true, R2_true, R2p_true, TEs)

        result = qbold_fit_voxel(data, TEs, n_iters=1000)
        assert abs(float(result['R2']) - R2_true) / R2_true < 0.15
        assert abs(float(result['R2_prime']) - R2p_true) / R2p_true < 0.20

    def test_recovers_with_noise(self):
        """Recovery with realistic SNR (~50)."""
        from neurojax.qmri.qbold import qbold_signal, qbold_fit_voxel
        TEs = jnp.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035])
        S0_true, R2_true, R2p_true = 800.0, 18.0, 6.0
        data = qbold_signal(S0_true, R2_true, R2p_true, TEs)
        noise = jax.random.normal(jax.random.PRNGKey(0), data.shape) * (S0_true / 50)
        data_noisy = data + noise

        result = qbold_fit_voxel(data_noisy, TEs, n_iters=500)
        assert abs(float(result['R2']) - R2_true) / R2_true < 0.20
        assert abs(float(result['R2_prime']) - R2p_true) / R2p_true < 0.30

    def test_vmap_batch(self):
        """Batch fitting via jax.vmap."""
        from neurojax.qmri.qbold import qbold_signal, qbold_fit_voxel
        TEs = jnp.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035])
        # 10 voxels with different R2'
        batch = jnp.stack([
            qbold_signal(800.0, 18.0, r2p, TEs)
            for r2p in jnp.linspace(2.0, 12.0, 10)
        ])
        fit_fn = jax.vmap(lambda d: qbold_fit_voxel(d, TEs, n_iters=300))
        results = fit_fn(batch)
        assert results['R2_prime'].shape == (10,)
        assert jnp.all(jnp.isfinite(results['R2_prime']))


class TestOEFConversion:
    """R2' → OEF → CMRO₂ pipeline."""

    def test_r2prime_to_oef_range(self):
        """OEF should be in [0, 1] for physiological R2'."""
        from neurojax.qmri.qbold import r2prime_to_oef
        R2_prime = jnp.array([2.0, 5.0, 10.0, 15.0])
        oef = r2prime_to_oef(R2_prime)
        assert jnp.all(oef >= 0)
        assert jnp.all(oef <= 1)

    def test_higher_r2prime_higher_oef(self):
        """More R2' = more deoxy blood = higher OEF."""
        from neurojax.qmri.qbold import r2prime_to_oef
        oef_low = r2prime_to_oef(jnp.array(3.0))
        oef_high = r2prime_to_oef(jnp.array(10.0))
        assert float(oef_high) > float(oef_low)

    def test_cmro2_computation(self):
        """CMRO₂ = CBF × OEF × CaO₂."""
        from neurojax.qmri.qbold import compute_regional_cmro2
        cbf = jnp.array(50.0)   # ml/100g/min (typical GM)
        oef = jnp.array(0.35)   # typical
        cmro2 = compute_regional_cmro2(cbf, oef)
        # Expected ~130-200 µmol/100g/min for GM
        assert float(cmro2) > 50
        assert float(cmro2) < 300
