"""Virtual lesion framework for whole-brain disconnection simulation.

Implements the virtual lesion methodology from Momi et al. (2023):
selectively disconnect brain regions in a structural connectome,
re-simulate whole-brain dynamics, and measure the functional consequence.

This enables decomposing TMS-evoked potentials (or any brain response)
into local reverberatory activity vs. recurrent network contributions.

Pipeline::

    Intact connectome + fitted model → simulate → intact_output
    For each region r:
        Lesioned connectome (zero row/col r) → simulate → lesioned_output
        Δ[r, t] = |intact - lesioned| at each timepoint
    → Contribution matrix (n_regions, n_timepoints)
    → Local/network transition time

References
----------
Momi D, Wang Z, Griffiths JD (2023). TMS-evoked responses are driven
by recurrent large-scale network dynamics. eLife 12:e83232.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import jax.numpy as jnp
import numpy as np


def apply_lesion(
    weights: jnp.ndarray,
    regions: list[int],
    strength: float = 0.0,
) -> jnp.ndarray:
    """Apply a virtual lesion to a structural connectome.

    Sets all connections to/from the specified regions to zero (complete
    disconnection) or scales them by ``strength`` (partial lesion).

    Parameters
    ----------
    weights : (N, N) structural connectivity matrix.
    regions : list of int — region indices to lesion.
    strength : float in [0, 1] — 0.0 = complete disconnection (default),
        0.5 = 50% reduction, 1.0 = no change.

    Returns
    -------
    W_lesion : (N, N) — modified connectivity matrix.
    """
    W = jnp.array(weights)
    for r in regions:
        W = W.at[r, :].multiply(strength)
        W = W.at[:, r].multiply(strength)
    return W


def lesion_effect(
    weights: jnp.ndarray,
    regions: list[int],
    simulate_fn: Callable,
    strength: float = 0.0,
) -> dict:
    """Compare intact vs. lesioned simulation output.

    Parameters
    ----------
    weights : (N, N) structural connectivity matrix.
    regions : list of int — regions to lesion.
    simulate_fn : callable(weights) → (n_regions, n_timepoints) array.
        Simulation function that takes a connectivity matrix and returns
        neural activity or BOLD timeseries.
    strength : float — lesion strength (0 = complete, 1 = none).

    Returns
    -------
    dict with:
        ``"intact"`` : (n_regions, T) — intact simulation output.
        ``"lesioned"`` : (n_regions, T) — lesioned simulation output.
        ``"difference"`` : (n_regions, T) — absolute difference.
        ``"regions"`` : list of int — lesioned regions.
    """
    intact = simulate_fn(weights)
    W_lesion = apply_lesion(weights, regions, strength)
    lesioned = simulate_fn(W_lesion)
    difference = jnp.abs(intact - lesioned)

    return {
        "intact": intact,
        "lesioned": lesioned,
        "difference": difference,
        "regions": regions,
    }


def contribution_matrix(
    weights: jnp.ndarray,
    simulate_fn: Callable,
    regions: Optional[list[int]] = None,
    strength: float = 0.0,
    metric: str = "mean_abs",
) -> jnp.ndarray:
    """Compute the contribution of each region to the overall dynamics.

    For each region, lesion it and measure how much the output changes.
    The result is a (n_regions, n_timepoints) matrix where each row
    shows the time-resolved impact of disconnecting that region.

    Parameters
    ----------
    weights : (N, N) structural connectivity matrix.
    simulate_fn : callable(weights) → (n_regions, n_timepoints).
    regions : list of int — which regions to test. Default: all.
    strength : float — lesion strength.
    metric : str — how to summarize the effect across output channels.
        ``"mean_abs"`` (default): mean absolute difference across channels.
        ``"max_abs"``: max absolute difference across channels.
        ``"gfp"``: global field power (spatial std of difference).

    Returns
    -------
    C : (n_tested_regions, n_timepoints) contribution matrix.
    """
    N = weights.shape[0]
    if regions is None:
        regions = list(range(N))

    intact = simulate_fn(weights)
    T = intact.shape[1]

    contributions = []
    for r in regions:
        W_lesion = apply_lesion(weights, [r], strength)
        lesioned = simulate_fn(W_lesion)
        diff = jnp.abs(intact - lesioned)

        if metric == "mean_abs":
            c = jnp.mean(diff, axis=0)  # (T,)
        elif metric == "max_abs":
            c = jnp.max(diff, axis=0)
        elif metric == "gfp":
            c = jnp.std(intact - lesioned, axis=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        contributions.append(c)

    return jnp.stack(contributions)  # (n_regions, T)


def virtual_lesion_sweep(
    weights: jnp.ndarray,
    simulate_fn: Callable,
    regions: Optional[list[int]] = None,
    strength: float = 0.0,
) -> list[dict]:
    """Run lesion_effect for multiple regions, returning full details.

    Parameters
    ----------
    weights : (N, N)
    simulate_fn : callable(weights) → (n_regions, n_timepoints)
    regions : list of int — default: all regions.
    strength : float

    Returns
    -------
    list of dict — one per region, each containing intact/lesioned/difference.
    """
    N = weights.shape[0]
    if regions is None:
        regions = list(range(N))

    results = []
    for r in regions:
        result = lesion_effect(weights, [r], simulate_fn, strength)
        results.append(result)

    return results


def local_network_transition(
    C: jnp.ndarray,
    target_region: int,
    smoothing_window: int = 5,
) -> int:
    """Find when network contributions exceed local contributions.

    Given a contribution matrix from ``contribution_matrix()``, identifies
    the timepoint where the sum of non-target region contributions first
    exceeds the target region's contribution. This marks the transition
    from local reverberatory activity to network-mediated feedback
    (Momi et al. 2023).

    Parameters
    ----------
    C : (n_regions, n_timepoints) — contribution matrix.
    target_region : int — index of the TMS target region in C.
    smoothing_window : int — moving average window for noise reduction.

    Returns
    -------
    t_transition : int — timepoint index where network > local.
    """
    n_regions, T = C.shape

    # Local contribution: target region
    local = C[target_region]

    # Network contribution: sum of all other regions
    mask = jnp.ones(n_regions).at[target_region].set(0.0)
    network = jnp.sum(C * mask[:, None], axis=0)

    # Smooth both signals
    if smoothing_window > 1:
        kernel = jnp.ones(smoothing_window) / smoothing_window
        local = jnp.convolve(local, kernel, mode="same")
        network = jnp.convolve(network, kernel, mode="same")

    # Find first crossing: network > local
    crossings = network > local
    # Find first True after some initial period (skip first few samples)
    crossing_indices = jnp.where(crossings, jnp.arange(T), T)
    t_transition = int(jnp.min(crossing_indices))

    return t_transition
