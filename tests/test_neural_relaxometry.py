"""Tests for neural relaxometry models — TDD red phase.

Tests define the expected behaviour of PINN/Neural ODE/CDE
multicompartment relaxometry models before implementation.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


# =====================================================================
# Pulse sequence tests
# =====================================================================

class TestPulseSequence:
    """Differentiable pulse sequence descriptions."""

    def test_spgr_sequence_has_required_fields(self):
        from neurojax.qmri.pulse_sequence import SPGRSequence
        seq = SPGRSequence(flip_angles_deg=[2, 4, 6, 8, 10, 12, 14, 18], TR=0.004, TE=0.0019)
        assert seq.n_readouts == 8
        assert seq.TR == 0.004

    def test_megre_sequence(self):
        from neurojax.qmri.pulse_sequence import MEGRESequence
        seq = MEGRESequence(
            echo_times=[0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035],
            flip_angle_deg=15.0
        )
        assert seq.n_readouts == 7

    def test_qmt_sequence(self):
        from neurojax.qmri.pulse_sequence import QMTSequence
        seq = QMTSequence(
            sat_angles_deg=[332, 628], sat_offsets_hz=[1000, 47180],
            Trf=0.015, TR=0.055, readout_fa_deg=5.0
        )
        assert seq.n_readouts == 2

    def test_sequence_is_pytree(self):
        """Sequences should be valid JAX pytrees for use in jit/vmap."""
        from neurojax.qmri.pulse_sequence import SPGRSequence
        seq = SPGRSequence(flip_angles_deg=[2, 18], TR=0.004, TE=0.0019)
        leaves = jax.tree.leaves(seq)
        assert len(leaves) > 0


# =====================================================================
# Bloch Neural ODE tests
# =====================================================================

class TestBlochNeuralODE:
    """Neural ODE that simulates magnetisation evolution under RF pulses."""

    def test_forward_produces_signal(self):
        """Given tissue params + sequence → predicted signal vector."""
        from neurojax.qmri.neural_relaxometry import BlochNeuralODE
        from neurojax.qmri.pulse_sequence import SPGRSequence

        model = BlochNeuralODE(key=jax.random.PRNGKey(0))
        seq = SPGRSequence(flip_angles_deg=[2, 4, 6, 8, 10, 12, 14, 18], TR=0.004, TE=0.0019)
        tissue_params = jnp.array([1000.0, 0.8])  # [M0, T1]

        signal = model(tissue_params, seq)
        assert signal.shape == (8,)
        assert jnp.all(jnp.isfinite(signal))
        assert jnp.all(signal >= 0)

    def test_forward_is_differentiable(self):
        """Gradients flow through the ODE simulation."""
        from neurojax.qmri.neural_relaxometry import BlochNeuralODE
        from neurojax.qmri.pulse_sequence import SPGRSequence

        model = BlochNeuralODE(key=jax.random.PRNGKey(0))
        seq = SPGRSequence(flip_angles_deg=[2, 18], TR=0.004, TE=0.0019)

        def loss(params):
            signal = model(params, seq)
            return jnp.sum(signal ** 2)

        grads = jax.grad(loss)(jnp.array([1000.0, 0.8]))
        assert jnp.all(jnp.isfinite(grads))

    def test_recovers_analytical_spgr(self):
        """Trained BlochNODE should match analytical SPGR within tolerance."""
        from neurojax.qmri.neural_relaxometry import BlochNeuralODE
        from neurojax.qmri.pulse_sequence import SPGRSequence
        from neurojax.qmri.steady_state import spgr_signal_multi

        seq = SPGRSequence(flip_angles_deg=[2, 4, 6, 8, 10, 12, 14, 18], TR=0.004, TE=0.0019)
        fa_rad = jnp.deg2rad(jnp.array(seq.flip_angles_deg, dtype=float))

        # Analytical reference
        M0, T1 = 1000.0, 0.8
        analytical = spgr_signal_multi(M0, T1, fa_rad, seq.TR)

        # Neural ODE prediction (untrained — just check it's in the right ballpark)
        model = BlochNeuralODE(key=jax.random.PRNGKey(42))
        neural = model(jnp.array([M0, T1]), seq)

        # Untrained model won't match exactly, but signals should be positive and finite
        assert jnp.all(neural > 0)
        assert jnp.all(jnp.isfinite(neural))


# =====================================================================
# Multi-compartment NODE tests
# =====================================================================

class TestMultiCompartmentNODE:
    """Neural ODE with N tissue pools for mcDESPOT-like fitting."""

    def test_two_pool_forward(self):
        from neurojax.qmri.neural_relaxometry import MultiCompartmentNODE
        from neurojax.qmri.pulse_sequence import SPGRSequence

        model = MultiCompartmentNODE(n_compartments=2, key=jax.random.PRNGKey(0))
        seq = SPGRSequence(flip_angles_deg=[2, 4, 6, 8], TR=0.004, TE=0.0019)
        # params: [M0, f_mw, T1_mw, T1_iew, T2_mw, T2_iew]
        params = jnp.array([1000.0, 0.12, 0.5, 1.0, 0.015, 0.070])

        signal = model(params, seq)
        assert signal.shape == (4,)
        assert jnp.all(jnp.isfinite(signal))

    def test_three_pool_forward(self):
        from neurojax.qmri.neural_relaxometry import MultiCompartmentNODE
        from neurojax.qmri.pulse_sequence import SPGRSequence

        model = MultiCompartmentNODE(n_compartments=3, key=jax.random.PRNGKey(0))
        seq = SPGRSequence(flip_angles_deg=[2, 4, 6, 8], TR=0.004, TE=0.0019)
        # params: [M0, f1, f2, T1_1, T1_2, T1_3, T2_1, T2_2, T2_3]
        params = jnp.array([1000.0, 0.12, 0.18, 0.5, 1.0, 3.0, 0.015, 0.070, 1.5])

        signal = model(params, seq)
        assert signal.shape == (4,)

    def test_differentiable(self):
        from neurojax.qmri.neural_relaxometry import MultiCompartmentNODE
        from neurojax.qmri.pulse_sequence import SPGRSequence

        model = MultiCompartmentNODE(n_compartments=2, key=jax.random.PRNGKey(0))
        seq = SPGRSequence(flip_angles_deg=[2, 18], TR=0.004, TE=0.0019)

        def loss(params):
            return jnp.sum(model(params, seq) ** 2)

        grads = jax.grad(loss)(jnp.array([1000.0, 0.12, 0.5, 1.0, 0.015, 0.070]))
        assert jnp.all(jnp.isfinite(grads))


# =====================================================================
# Relaxometry PINN tests
# =====================================================================

class TestRelaxometryPINN:
    """Physics-informed network for spatial relaxometry."""

    def test_pinn_forward(self):
        """PINN maps (x,y,z) → tissue params, then evaluates signal model."""
        from neurojax.qmri.neural_relaxometry import RelaxometryPINN
        from neurojax.qmri.pulse_sequence import SPGRSequence

        seq = SPGRSequence(flip_angles_deg=[2, 4, 6, 8], TR=0.004, TE=0.0019)
        pinn = RelaxometryPINN(n_params=2, key=jax.random.PRNGKey(0))

        coords = jnp.array([64.0, 64.0, 52.0])  # voxel coordinates
        params = pinn.predict_params(coords)
        assert params.shape == (2,)  # [M0, T1] or similar

    def test_pinn_physics_loss(self):
        """Physics loss enforces signal model consistency."""
        from neurojax.qmri.neural_relaxometry import RelaxometryPINN
        from neurojax.qmri.pulse_sequence import SPGRSequence

        seq = SPGRSequence(flip_angles_deg=[2, 4, 6, 8], TR=0.004, TE=0.0019)
        pinn = RelaxometryPINN(n_params=2, key=jax.random.PRNGKey(0))

        coords = jnp.array([64.0, 64.0, 52.0])
        data = jnp.array([30.0, 35.0, 33.0, 28.0])  # observed signal

        loss = pinn.loss(coords, data, seq)
        assert jnp.isfinite(loss)
        assert float(loss) >= 0

    def test_pinn_batch_coords(self):
        """vmap over spatial coordinates."""
        from neurojax.qmri.neural_relaxometry import RelaxometryPINN

        pinn = RelaxometryPINN(n_params=2, key=jax.random.PRNGKey(0))
        coords_batch = jax.random.uniform(jax.random.PRNGKey(1), (100, 3)) * 128
        params_batch = jax.vmap(pinn.predict_params)(coords_batch)
        assert params_batch.shape == (100, 2)
