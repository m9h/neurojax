"""Windowed systems identification for MEG dynamics comparison.

Applies SINDy, DMD/Koopman, and log-signature analysis to sliding windows
of prepared MEG data, producing per-window summaries that can be compared
across methods and against HMM/DyNeMo state timecourses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from jaxctrl import (
    KoopmanEstimator,
    SINDyOptimizer,
    polynomial_library,
)
from neurojax.analysis.rough import (
    augment_path,
    compute_log_signature,
    sliding_signature,
)


# ---------------------------------------------------------------------------
# Windowed SINDy — track Jacobian eigenvalues over time
# ---------------------------------------------------------------------------


@dataclass
class WindowedSINDyResult:
    """Result of windowed SINDy analysis.

    Attributes
    ----------
    times : (n_windows,) float — centre time of each window (in samples)
    max_real_eig : (n_windows,) — max Re(eigenvalue) per window
    eigenvalues : (n_windows, n_vars) complex — all eigenvalues per window
    coefficients : (n_windows, n_library, n_vars) — SINDy Xi per window
    """

    times: np.ndarray
    max_real_eig: np.ndarray
    eigenvalues: np.ndarray
    coefficients: np.ndarray


def windowed_sindy(
    data: jax.Array,
    window_size: int = 3000,
    stride: int = 1500,
    n_pca: int = 3,
    degree: int = 2,
    threshold: float = 0.01,
    dt: float = 0.01,
) -> WindowedSINDyResult:
    """Run SINDy on sliding windows and track Jacobian eigenvalues.

    For each window:
    1. PCA to ``n_pca`` dimensions (for tractable polynomial library)
    2. Compute finite-difference derivatives
    3. Fit SINDy with polynomial library
    4. Extract Jacobian at origin, compute eigenvalues

    A sign change in max Re(eigenvalue) indicates a bifurcation —
    compare these change-points against HMM state transitions.

    Parameters
    ----------
    data : (T, C) prepared data
    window_size : int — samples per window
    stride : int — step between windows
    n_pca : int — PCA dimensionality for SINDy (keep small)
    degree : int — polynomial library degree
    threshold : float — SINDy sparsification threshold
    dt : float — sampling interval (for derivative computation)
    """
    data_np = np.array(data)
    T, C = data_np.shape
    n_windows = (T - window_size) // stride + 1

    optimizer = SINDyOptimizer(threshold=threshold, max_iter=10)
    lib_fn = lambda X: polynomial_library(X, degree=degree)

    all_times = []
    all_max_eig = []
    all_eigs = []
    all_xi = []

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window = data_np[start:end]
        centre = start + window_size // 2
        all_times.append(centre)

        # PCA on the window
        window_c = window - window.mean(axis=0)
        U, S, Vh = np.linalg.svd(window_c, full_matrices=False)
        X_pca = (U[:, :n_pca] * S[:n_pca])  # (window_size, n_pca)

        # Finite-difference derivatives
        dX = np.gradient(X_pca, dt, axis=0)

        # SINDy fit
        X_jax = jnp.array(X_pca)
        dX_jax = jnp.array(dX)
        Xi = optimizer.fit(X_jax, dX_jax, lib_fn)
        Xi_np = np.array(Xi)

        # Jacobian at origin (linear block of polynomial coefficients)
        A = Xi_np[1: n_pca + 1, :].T  # (n_pca, n_pca)
        eigs = np.linalg.eigvals(A)

        all_max_eig.append(np.max(np.real(eigs)))
        all_eigs.append(eigs)
        all_xi.append(Xi_np)

    return WindowedSINDyResult(
        times=np.array(all_times),
        max_real_eig=np.array(all_max_eig),
        eigenvalues=np.array(all_eigs),
        coefficients=np.array(all_xi),
    )


# ---------------------------------------------------------------------------
# Windowed Koopman/DMD — track dominant frequencies over time
# ---------------------------------------------------------------------------


@dataclass
class WindowedDMDResult:
    """Result of windowed DMD analysis.

    Attributes
    ----------
    times : (n_windows,) float
    frequencies : (n_windows, rank) — dominant frequencies (Hz)
    growth_rates : (n_windows, rank) — growth/decay rates
    eigenvalues : (n_windows, rank) complex — discrete DMD eigenvalues
    """

    times: np.ndarray
    frequencies: np.ndarray
    growth_rates: np.ndarray
    eigenvalues: np.ndarray


def windowed_dmd(
    data: jax.Array,
    window_size: int = 3000,
    stride: int = 1500,
    rank: int = 10,
    dt: float = 0.01,
) -> WindowedDMDResult:
    """Run DMD/Koopman on sliding windows and extract frequencies.

    Compare dominant DMD frequencies against HMM state spectral content.

    Parameters
    ----------
    data : (T, C) prepared data
    window_size : int
    stride : int
    rank : int — SVD truncation rank for DMD
    dt : float — sampling interval (seconds)
    """
    data_np = np.array(data)
    T, C = data_np.shape
    n_windows = (T - window_size) // stride + 1

    estimator = KoopmanEstimator(rank=rank)

    all_times = []
    all_freqs = []
    all_growth = []
    all_eigs = []

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window = data_np[start:end]
        centre = start + window_size // 2
        all_times.append(centre)

        # DMD snapshot matrices (features x samples)
        X = jnp.array(window[:-1].T)  # (C, T-1)
        Y = jnp.array(window[1:].T)   # (C, T-1)

        _, eigs_discrete, _ = estimator.fit(X, Y)
        eigs_discrete = np.array(eigs_discrete)

        # Convert to continuous-time
        omega = np.log(eigs_discrete + 0j) / dt
        freqs = np.abs(np.imag(omega)) / (2 * np.pi)
        growth = np.real(omega)

        # Sort by frequency
        order = np.argsort(-np.abs(freqs))
        all_freqs.append(freqs[order])
        all_growth.append(growth[order])
        all_eigs.append(eigs_discrete[order])

    return WindowedDMDResult(
        times=np.array(all_times),
        frequencies=np.array(all_freqs),
        growth_rates=np.array(all_growth),
        eigenvalues=np.array(all_eigs),
    )


# ---------------------------------------------------------------------------
# Windowed log-signatures — track geometric features over time
# ---------------------------------------------------------------------------


@dataclass
class WindowedSignatureResult:
    """Result of windowed log-signature analysis.

    Attributes
    ----------
    times : (n_windows,) float
    signatures : (n_windows, sig_dim) — log-signature per window
    distances : (n_windows - 1,) — consecutive signature distances
    change_points : (n_changes,) int — indices where distance spikes
    """

    times: np.ndarray
    signatures: np.ndarray
    distances: np.ndarray
    change_points: np.ndarray


def windowed_signatures(
    data: jax.Array,
    window_size: int = 3000,
    stride: int = 1500,
    n_pca: int = 5,
    depth: int = 3,
    change_threshold: float = 2.0,
) -> WindowedSignatureResult:
    """Compute log-signatures over sliding windows and detect change points.

    Signature distance jumps indicate qualitative changes in dynamics —
    compare these against HMM/DyNeMo state transitions.

    Parameters
    ----------
    data : (T, C) prepared data
    window_size : int
    stride : int
    n_pca : int — reduce to this many dims before signature (controls sig_dim)
    depth : int — signature truncation depth
    change_threshold : float — std multiplier for change point detection
    """
    data_np = np.array(data)
    T, C = data_np.shape
    n_windows = (T - window_size) // stride + 1

    all_times = []
    all_sigs = []

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window = data_np[start:end]
        centre = start + window_size // 2
        all_times.append(centre)

        # PCA for manageable signature dimension
        window_c = window - window.mean(axis=0)
        U, S, Vh = np.linalg.svd(window_c, full_matrices=False)
        X_pca = U[:, :n_pca] * S[:n_pca]

        # Log-signature
        path = jnp.array(X_pca)
        path_aug = augment_path(path, add_time=True)
        logsig = compute_log_signature(path_aug, depth=depth)
        all_sigs.append(np.array(logsig))

    sigs = np.array(all_sigs)
    times = np.array(all_times)

    if len(sigs) < 2:
        return WindowedSignatureResult(
            times=times,
            signatures=sigs,
            distances=np.array([]),
            change_points=np.array([], dtype=int),
        )

    # Consecutive distances
    diffs = np.linalg.norm(np.diff(sigs, axis=0), axis=1)
    mean_d = diffs.mean()
    std_d = diffs.std()

    # Change points: where distance exceeds threshold
    change_mask = diffs > (mean_d + change_threshold * std_d)
    change_points = np.where(change_mask)[0]

    return WindowedSignatureResult(
        times=times,
        signatures=sigs,
        distances=diffs,
        change_points=change_points,
    )
