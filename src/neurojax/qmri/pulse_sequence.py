"""Differentiable pulse sequence descriptions.

Dataclasses describing MRI acquisition parameters that feed into
Neural ODE/CDE relaxometry models as control signals.
All are JAX pytree-compatible via equinox.
"""

import equinox as eqx
import jax.numpy as jnp
from typing import Sequence


class SPGRSequence(eqx.Module):
    """Spoiled gradient echo (VFA / DESPOT1) sequence."""
    flip_angles_deg: list
    TR: float
    TE: float

    @property
    def n_readouts(self) -> int:
        return len(self.flip_angles_deg)

    @property
    def flip_angles_rad(self) -> jnp.ndarray:
        return jnp.deg2rad(jnp.array(self.flip_angles_deg, dtype=float))


class bSSFPSequence(eqx.Module):
    """Balanced steady-state free precession (DESPOT2) sequence."""
    flip_angles_deg: list
    TR: float
    TE: float
    phase_cycles_deg: list = eqx.field(default_factory=lambda: [0.0, 180.0])

    @property
    def n_readouts(self) -> int:
        return len(self.flip_angles_deg) * len(self.phase_cycles_deg)

    @property
    def flip_angles_rad(self) -> jnp.ndarray:
        return jnp.deg2rad(jnp.array(self.flip_angles_deg, dtype=float))


class MEGRESequence(eqx.Module):
    """Multi-echo gradient recalled echo sequence."""
    echo_times: list
    flip_angle_deg: float = 15.0

    @property
    def n_readouts(self) -> int:
        return len(self.echo_times)

    @property
    def TEs(self) -> jnp.ndarray:
        return jnp.array(self.echo_times, dtype=float)


class QMTSequence(eqx.Module):
    """Quantitative magnetisation transfer sequence."""
    sat_angles_deg: list
    sat_offsets_hz: list
    Trf: float
    TR: float
    readout_fa_deg: float = 5.0

    @property
    def n_readouts(self) -> int:
        return len(self.sat_angles_deg)

    @property
    def sat_angles_rad(self) -> jnp.ndarray:
        return jnp.deg2rad(jnp.array(self.sat_angles_deg, dtype=float))


class IRSPGRSequence(eqx.Module):
    """Inversion recovery SPGR (DESPOT1-HIFI) sequence."""
    flip_angle_deg: float
    TR: float
    TI: float
    TE: float = 0.0019

    @property
    def n_readouts(self) -> int:
        return 1
