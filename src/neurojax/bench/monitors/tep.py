"""TMS-Evoked Potential (TEP) observation model and loss functions.

Extracts the post-stimulus time window from a simulation, optionally
projects to sensor space via a leadfield, and computes differentiable
loss against empirical TEP data.

The TEP is the characteristic EEG/MEG waveform observed 15–300 ms after
a TMS pulse. Fitting simulated TEPs to empirical data constrains both
local excitability and network propagation parameters.

All operations are JAX-native and differentiable.
"""

from __future__ import annotations

import jax.numpy as jnp

from neurojax.bench.monitors.leadfield import ForwardProjection


def extract_tep(
    timeseries: jnp.ndarray,
    t_onset: float,
    dt: float,
    t_pre: float = 10.0,
    t_post: float = 300.0,
) -> jnp.ndarray:
    """Extract a TEP time window from a simulation.

    Parameters
    ----------
    timeseries : jnp.ndarray
        Neural activity of shape (n_regions, n_timepoints).
    t_onset : float
        TMS onset time in ms.
    dt : float
        Simulation timestep in ms.
    t_pre : float
        Time before onset to include (ms). Default: 10.
    t_post : float
        Time after onset to include (ms). Default: 300.

    Returns
    -------
    jnp.ndarray
        TEP segment of shape (n_regions, n_tep_samples).
    """
    idx_start = int((t_onset - t_pre) / dt)
    idx_end = int((t_onset + t_post) / dt)

    # Clamp to valid range
    idx_start = max(0, idx_start)
    idx_end = min(timeseries.shape[1], idx_end)

    return timeseries[:, idx_start:idx_end]


def extract_tep_sensor(
    timeseries: jnp.ndarray,
    forward: ForwardProjection,
    t_onset: float,
    dt: float,
    t_pre: float = 10.0,
    t_post: float = 300.0,
) -> jnp.ndarray:
    """Extract TEP in sensor space via leadfield projection.

    Parameters
    ----------
    timeseries : jnp.ndarray
        Source-level neural activity (n_regions, n_timepoints).
    forward : ForwardProjection
        Leadfield forward model (handles avg ref for EEG).
    t_onset : float
        TMS onset time in ms.
    dt : float
        Simulation timestep in ms.
    t_pre : float
        Time before onset to include (ms).
    t_post : float
        Time after onset to include (ms).

    Returns
    -------
    jnp.ndarray
        Sensor-space TEP of shape (n_sensors, n_tep_samples).
    """
    tep_source = extract_tep(timeseries, t_onset, dt, t_pre, t_post)
    return forward.project(tep_source)


def tep_waveform_loss(
    sim_tep: jnp.ndarray,
    empirical_tep: jnp.ndarray,
) -> jnp.ndarray:
    """MSE loss between simulated and empirical TEP waveforms.

    Parameters
    ----------
    sim_tep : jnp.ndarray
        Simulated TEP of shape (n_channels, n_samples).
    empirical_tep : jnp.ndarray
        Empirical TEP of same shape.

    Returns
    -------
    jnp.ndarray
        Scalar MSE loss.
    """
    return jnp.mean((sim_tep - empirical_tep) ** 2)


def tep_gfp_loss(
    sim_tep: jnp.ndarray,
    empirical_tep: jnp.ndarray,
) -> jnp.ndarray:
    """Loss comparing Global Field Power (GFP) of simulated vs empirical TEP.

    GFP is the spatial standard deviation across channels at each timepoint.
    It captures the overall amplitude envelope without being sensitive to
    exact spatial topography.

    Parameters
    ----------
    sim_tep : jnp.ndarray
        Simulated TEP of shape (n_channels, n_samples).
    empirical_tep : jnp.ndarray
        Empirical TEP of same shape.

    Returns
    -------
    jnp.ndarray
        Scalar MSE loss between GFP envelopes.
    """
    gfp_sim = jnp.std(sim_tep, axis=0)
    gfp_emp = jnp.std(empirical_tep, axis=0)
    return jnp.mean((gfp_sim - gfp_emp) ** 2)


def tep_combined_loss(
    sim_tep: jnp.ndarray,
    empirical_tep: jnp.ndarray,
    w_waveform: float = 1.0,
    w_gfp: float = 0.5,
) -> jnp.ndarray:
    """Combined TEP loss: waveform MSE + GFP envelope.

    Parameters
    ----------
    sim_tep : jnp.ndarray
        Simulated TEP of shape (n_channels, n_samples).
    empirical_tep : jnp.ndarray
        Empirical TEP of same shape.
    w_waveform : float
        Weight for waveform MSE term.
    w_gfp : float
        Weight for GFP envelope term.

    Returns
    -------
    jnp.ndarray
        Scalar combined loss.
    """
    loss = jnp.array(0.0)
    if w_waveform > 0:
        loss = loss + w_waveform * tep_waveform_loss(sim_tep, empirical_tep)
    if w_gfp > 0:
        loss = loss + w_gfp * tep_gfp_loss(sim_tep, empirical_tep)
    return loss
