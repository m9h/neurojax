"""TMS (Transcranial Magnetic Stimulation) stimulus protocol.

Generates time-varying external input arrays for injection into neural
mass models (RWW, Jansen-Rit) during whole-brain simulation. The TMS
pulse is modeled as a brief current injection at one or more cortical
regions, following the WhoBPyT convention.

Waveform options:

- ``"monophasic"``: Exponential decay from peak amplitude.
  ``I(t) = A * exp(-(t - t_onset) / tau)``

- ``"biphasic"``: Positive then negative phase (models real TMS coil).
  ``I(t) = A * sin(2π(t - t_onset) / period)`` within the pulse window.

- ``"square"``: Constant amplitude for the pulse duration.

All functions are JAX-native and produce arrays compatible with
``jax.lax.scan`` integration loops.

References
----------
Griffiths JD et al. (2022). Whole-brain modelling: past, present, and
future. In *Computational Modelling of the Brain*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp


@dataclass
class TMSProtocol:
    """Specification of a single TMS pulse.

    Parameters
    ----------
    t_onset : float
        Stimulus onset time in ms.
    target_region : int
        Index of the primary stimulation target (0-based).
    amplitude : float
        Peak stimulus amplitude in nA. Typical range: 0.1–5.0.
    pulse_width : float
        Duration of the active pulse in ms. Default: 1.0.
    waveform : str
        Pulse shape: ``"monophasic"``, ``"biphasic"``, or ``"square"``.
    tau : float
        Decay time constant for monophasic waveform (ms). Default: 0.3.
    spatial_spread : jnp.ndarray or None
        Optional (n_regions,) array of spatial weights for multi-region
        targeting (e.g., Gaussian spread from coil focus). If None, only
        the target_region is stimulated.
    """

    t_onset: float
    target_region: int
    amplitude: float = 1.0
    pulse_width: float = 1.0
    waveform: Literal["monophasic", "biphasic", "square"] = "monophasic"
    tau: float = 0.3
    spatial_spread: jnp.ndarray | None = None


def tms_waveform(
    t: jnp.ndarray,
    protocol: TMSProtocol,
) -> jnp.ndarray:
    """Compute the temporal waveform of a TMS pulse.

    Parameters
    ----------
    t : jnp.ndarray
        Time points in ms, shape (n_timepoints,).
    protocol : TMSProtocol
        Stimulus specification.

    Returns
    -------
    jnp.ndarray
        Scalar waveform amplitude at each timepoint, shape (n_timepoints,).
        Zero outside the pulse window.
    """
    dt = t - protocol.t_onset
    active = (dt >= 0.0) & (dt < protocol.pulse_width)

    if protocol.waveform == "monophasic":
        # Exponential decay
        raw = protocol.amplitude * jnp.exp(-dt / protocol.tau)
    elif protocol.waveform == "biphasic":
        # Single sine cycle within pulse window
        raw = protocol.amplitude * jnp.sin(
            2.0 * jnp.pi * dt / protocol.pulse_width
        )
    elif protocol.waveform == "square":
        raw = jnp.full_like(t, protocol.amplitude)
    else:
        raise ValueError(f"Unknown waveform: {protocol.waveform}")

    return jnp.where(active, raw, 0.0)


def make_stimulus_train(
    protocols: list[TMSProtocol] | TMSProtocol,
    n_regions: int,
    dt: float,
    duration: float,
) -> jnp.ndarray:
    """Generate a full stimulus train for a simulation.

    Pre-computes the stimulus array for all timesteps and regions,
    suitable for passing into a ``jax.lax.scan`` loop alongside noise.

    Parameters
    ----------
    protocols : TMSProtocol or list of TMSProtocol
        One or more TMS pulse specifications. Multiple pulses can target
        different regions and/or different timepoints (e.g., paired-pulse
        TMS or multi-site stimulation).
    n_regions : int
        Number of brain regions in the model.
    dt : float
        Simulation timestep in ms.
    duration : float
        Total simulation duration in ms.

    Returns
    -------
    jnp.ndarray
        Stimulus array of shape (n_steps, n_regions). Each entry gives
        the external input current to inject at that timestep and region.

    Examples
    --------
    >>> proto = TMSProtocol(t_onset=100.0, target_region=5, amplitude=2.0)
    >>> stim = make_stimulus_train(proto, n_regions=80, dt=0.1, duration=1000.0)
    >>> stim.shape
    (10000, 80)
    """
    if isinstance(protocols, TMSProtocol):
        protocols = [protocols]

    n_steps = int(duration / dt)
    times = jnp.arange(n_steps) * dt  # (n_steps,)

    # Accumulate stimulus from all protocols
    stimulus = jnp.zeros((n_steps, n_regions))

    for proto in protocols:
        # Temporal waveform: (n_steps,)
        waveform = tms_waveform(times, proto)

        # Spatial pattern: (n_regions,)
        if proto.spatial_spread is not None:
            spatial = jnp.asarray(proto.spatial_spread)
        else:
            spatial = jnp.zeros(n_regions).at[proto.target_region].set(1.0)

        # Outer product: (n_steps, n_regions)
        stimulus = stimulus + waveform[:, None] * spatial[None, :]

    return stimulus
