"""Summary statistics for HMM/DyNeMo brain state dynamics.

Computes standard descriptive statistics from state time courses:
fractional occupancy, mean lifetime, mean interval, switching rate.
Also provides alpha binarization for converting DyNeMo's soft mixing
coefficients into hard state assignments.

These are the metrics used to characterize dynamic brain states in
publications (e.g., "State 3 has a mean lifetime of 120 ms and is
occupied 15% of the time").

All functions accept either hard state sequences (int arrays from
HMM Viterbi decoding) or soft mixing coefficients (float arrays from
DyNeMo inference).
"""

from __future__ import annotations

from typing import Literal, Optional

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Hard state assignment from soft probabilities
# ---------------------------------------------------------------------------


def state_time_courses(gamma: jnp.ndarray) -> jnp.ndarray:
    """Convert soft state probabilities to hard assignments via argmax.

    Parameters
    ----------
    gamma : (T, K) array of state probabilities (HMM gamma or DyNeMo alpha).

    Returns
    -------
    states : (T,) int array of state assignments.
    """
    return jnp.argmax(gamma, axis=1)


# ---------------------------------------------------------------------------
# Fractional occupancy
# ---------------------------------------------------------------------------


def fractional_occupancy(
    states_or_alpha: jnp.ndarray,
    n_states: Optional[int] = None,
) -> jnp.ndarray:
    """Fraction of time spent in each state.

    Parameters
    ----------
    states_or_alpha : (T,) int hard states OR (T, K) soft alpha.
        If 2-D, returns mean alpha per mode (soft fractional occupancy).
    n_states : int, optional
        Number of states (required for hard state input, inferred for soft).

    Returns
    -------
    fo : (K,) array of fractional occupancies summing to ~1.
    """
    if states_or_alpha.ndim == 2:
        # Soft alpha: mean across time
        return jnp.mean(states_or_alpha, axis=0)

    # Hard states
    states = states_or_alpha
    T = states.shape[0]
    if n_states is None:
        n_states = int(jnp.max(states)) + 1
    counts = jnp.zeros(n_states)
    for k in range(n_states):
        counts = counts.at[k].set(jnp.sum(states == k))
    return counts / T


# ---------------------------------------------------------------------------
# Run-length encoding helper
# ---------------------------------------------------------------------------


def _run_lengths(states: np.ndarray, n_states: int) -> dict[int, list[int]]:
    """Compute run lengths for each state.

    Returns dict mapping state → list of consecutive run lengths.
    """
    states = np.asarray(states)
    runs: dict[int, list[int]] = {k: [] for k in range(n_states)}

    if len(states) == 0:
        return runs

    current_state = int(states[0])
    current_length = 1

    for t in range(1, len(states)):
        if int(states[t]) == current_state:
            current_length += 1
        else:
            runs[current_state].append(current_length)
            current_state = int(states[t])
            current_length = 1
    # Final run
    runs[current_state].append(current_length)

    return runs


def _gap_lengths(states: np.ndarray, n_states: int) -> dict[int, list[int]]:
    """Compute gap lengths (intervals between visits) for each state.

    Returns dict mapping state → list of gap durations between successive visits.
    """
    states = np.asarray(states)
    gaps: dict[int, list[int]] = {k: [] for k in range(n_states)}

    for k in range(n_states):
        # Find all run start indices for state k
        in_state = (states == k)
        if not np.any(in_state):
            continue

        # Find transitions into state k
        starts = []
        if in_state[0]:
            starts.append(0)
        for t in range(1, len(states)):
            if in_state[t] and not in_state[t - 1]:
                starts.append(t)

        # Gaps between end of one visit and start of next
        ends = []
        for t in range(len(states) - 1):
            if in_state[t] and not in_state[t + 1]:
                ends.append(t + 1)
        if in_state[-1]:
            ends.append(len(states))

        for i in range(len(ends) - 1):
            gap = starts[i + 1] - ends[i]
            if gap > 0:
                gaps[k].append(gap)

    return gaps


# ---------------------------------------------------------------------------
# Mean lifetime
# ---------------------------------------------------------------------------


def mean_lifetime(
    states: jnp.ndarray,
    n_states: Optional[int] = None,
    fs: float = 1.0,
) -> jnp.ndarray:
    """Mean duration of each state visit (lifetime).

    Parameters
    ----------
    states : (T,) int array of hard state assignments.
    n_states : int, optional
        Number of states. Inferred from max(states)+1 if not given.
    fs : float
        Sampling frequency. If > 1.0, returns lifetime in seconds.

    Returns
    -------
    lifetimes : (K,) array — mean lifetime per state (in samples, or seconds if fs given).
    """
    states_np = np.asarray(states)
    if n_states is None:
        n_states = int(np.max(states_np)) + 1

    runs = _run_lengths(states_np, n_states)

    result = np.zeros(n_states)
    for k in range(n_states):
        if runs[k]:
            result[k] = np.mean(runs[k])
        else:
            result[k] = 0.0

    return jnp.array(result) / fs


# ---------------------------------------------------------------------------
# Mean interval
# ---------------------------------------------------------------------------


def mean_interval(
    states: jnp.ndarray,
    n_states: Optional[int] = None,
    fs: float = 1.0,
) -> jnp.ndarray:
    """Mean gap between successive visits to each state.

    Parameters
    ----------
    states : (T,) int array.
    n_states : int, optional
    fs : float — sampling frequency.

    Returns
    -------
    intervals : (K,) — mean interval per state (samples or seconds).
    """
    states_np = np.asarray(states)
    if n_states is None:
        n_states = int(np.max(states_np)) + 1

    gaps = _gap_lengths(states_np, n_states)

    result = np.zeros(n_states)
    for k in range(n_states):
        if gaps[k]:
            result[k] = np.mean(gaps[k])
        else:
            result[k] = 0.0

    return jnp.array(result) / fs


# ---------------------------------------------------------------------------
# Switching rate
# ---------------------------------------------------------------------------


def switching_rate(
    states: jnp.ndarray,
    fs: float = 1.0,
) -> jnp.ndarray:
    """Average number of state transitions per unit time.

    Parameters
    ----------
    states : (T,) int array.
    fs : float — sampling frequency. If 1.0, returns switches per sample.

    Returns
    -------
    rate : scalar — switching rate (per sample or Hz if fs given).
    """
    states = jnp.asarray(states)
    n_switches = jnp.sum(states[1:] != states[:-1])
    n_transitions = states.shape[0] - 1
    return (n_switches / n_transitions) * fs


# ---------------------------------------------------------------------------
# Alpha binarization (DyNeMo soft → hard)
# ---------------------------------------------------------------------------


def binarize_alpha(
    alpha: jnp.ndarray,
    method: Literal["gmm", "threshold"] = "threshold",
    threshold: float = 0.5,
) -> jnp.ndarray:
    """Convert soft DyNeMo mixing coefficients to binary activations.

    Parameters
    ----------
    alpha : (T, K) soft mixing coefficients from DyNeMo.
    method : ``"gmm"`` or ``"threshold"``.
        ``"gmm"``: fit a 2-component Gaussian mixture per mode and
        assign each timepoint to the "active" component.
        ``"threshold"``: simple threshold on alpha values.
    threshold : float
        Threshold for the ``"threshold"`` method. Default: 0.5.

    Returns
    -------
    binary : (T, K) binary array (0 or 1).
    """
    if method == "threshold":
        return (alpha >= threshold).astype(jnp.float32)

    elif method == "gmm":
        # Per-mode GMM binarization
        T, K = alpha.shape
        binary = np.zeros((T, K), dtype=np.float32)

        for k in range(K):
            values = np.asarray(alpha[:, k])
            # Simple 2-component GMM via EM (lightweight, no sklearn needed)
            binary[:, k] = _gmm_binarize_1d(values)

        return jnp.array(binary)

    else:
        raise ValueError(f"Unknown method: {method}")


def _gmm_binarize_1d(values: np.ndarray, n_iter: int = 20) -> np.ndarray:
    """Fit 2-component 1-D GMM and return binary assignment (1 = high component).

    Simple EM implementation — no sklearn dependency.
    """
    values = values.flatten()
    N = len(values)

    # Initialize: split at median
    median = np.median(values)
    mu = np.array([np.mean(values[values <= median]),
                    np.mean(values[values > median])])
    sigma = np.array([np.std(values[values <= median]) + 1e-10,
                       np.std(values[values > median]) + 1e-10])
    pi = np.array([0.5, 0.5])

    for _ in range(n_iter):
        # E-step
        log_resp = np.zeros((N, 2))
        for c in range(2):
            log_resp[:, c] = (
                np.log(pi[c] + 1e-30)
                - 0.5 * np.log(2 * np.pi * sigma[c] ** 2)
                - 0.5 * ((values - mu[c]) / sigma[c]) ** 2
            )
        # Normalize
        log_max = log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp - log_max)
        resp /= resp.sum(axis=1, keepdims=True)

        # M-step
        Nk = resp.sum(axis=0) + 1e-10
        pi = Nk / N
        mu = (resp * values[:, None]).sum(axis=0) / Nk
        sigma = np.sqrt(
            (resp * (values[:, None] - mu[None, :]) ** 2).sum(axis=0) / Nk
        ) + 1e-10

    # Assign to "high" component (whichever has larger mean)
    high_idx = np.argmax(mu)
    return (resp[:, high_idx] > 0.5).astype(np.float32)
