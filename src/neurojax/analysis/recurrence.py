"""Recurrence analysis: recurrence plots and RQA measures in JAX/numpy.

Inspired by pyunicorn's RecurrencePlot. Constructs recurrence matrices
from time series and computes Recurrence Quantification Analysis (RQA)
measures: determinism, laminarity, trapping time, entropy, etc.

All distance/threshold operations are JAX-native. Line-counting uses
numpy for variable-length structures.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def distance_matrix(x: jnp.ndarray, metric: str = "euclidean") -> jnp.ndarray:
    """Pairwise distance matrix for an embedded time series.

    Parameters
    ----------
    x : (T, D) — time-delay embedded points.
    metric : "euclidean", "manhattan", or "supremum".

    Returns
    -------
    D : (T, T) distance matrix.
    """
    diff = x[:, None, :] - x[None, :, :]  # (T, T, D)
    if metric == "euclidean":
        return jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
    elif metric == "manhattan":
        return jnp.sum(jnp.abs(diff), axis=-1)
    elif metric == "supremum":
        return jnp.max(jnp.abs(diff), axis=-1)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def recurrence_matrix(
    x: jnp.ndarray,
    threshold: float | None = None,
    recurrence_rate: float | None = None,
    metric: str = "euclidean",
) -> jnp.ndarray:
    """Binary recurrence matrix.

    Parameters
    ----------
    x : (T, D) embedded time series.
    threshold : float — fixed distance threshold.
    recurrence_rate : float in (0, 1) — target recurrence rate
        (threshold chosen to achieve this rate).
    metric : distance metric.

    Returns
    -------
    R : (T, T) binary matrix (1 = recurrent, 0 = not).
    """
    D = distance_matrix(x, metric)

    if recurrence_rate is not None:
        # Choose threshold to get the target rate
        threshold = float(jnp.percentile(D, recurrence_rate * 100))
    elif threshold is None:
        raise ValueError("Must specify either threshold or recurrence_rate")

    return (D <= threshold).astype(jnp.float32)


def recurrence_rate_measure(R: jnp.ndarray) -> float:
    """Fraction of recurrent points (excluding main diagonal)."""
    T = R.shape[0]
    mask = 1.0 - jnp.eye(T)
    return float(jnp.sum(R * mask) / jnp.sum(mask))


# ---------------------------------------------------------------------------
# Line structure extraction (numpy for variable-length)
# ---------------------------------------------------------------------------

def _diagonal_lines(R: np.ndarray, l_min: int = 2) -> list[int]:
    """Extract lengths of diagonal lines from the recurrence matrix."""
    T = R.shape[0]
    lines = []
    for k in range(-T + 1, T):
        if k == 0:
            continue  # skip main diagonal
        diag = np.diag(R, k)
        length = 0
        for val in diag:
            if val > 0.5:
                length += 1
            else:
                if length >= l_min:
                    lines.append(length)
                length = 0
        if length >= l_min:
            lines.append(length)
    return lines


def _vertical_lines(R: np.ndarray, v_min: int = 2) -> list[int]:
    """Extract lengths of vertical lines from the recurrence matrix."""
    T = R.shape[0]
    lines = []
    for col in range(T):
        length = 0
        for row in range(T):
            if R[row, col] > 0.5:
                length += 1
            else:
                if length >= v_min:
                    lines.append(length)
                length = 0
        if length >= v_min:
            lines.append(length)
    return lines


# ---------------------------------------------------------------------------
# RQA measures
# ---------------------------------------------------------------------------

def determinism(R: jnp.ndarray, l_min: int = 2) -> float:
    """Ratio of recurrence points forming diagonal structures to total."""
    R_np = np.asarray(R)
    lines = _diagonal_lines(R_np, l_min)
    if not lines:
        return 0.0
    diag_points = sum(lines)
    total = float(np.sum(R_np)) - R_np.shape[0]  # exclude main diagonal
    if total <= 0:
        return 0.0
    return diag_points / total


def laminarity(R: jnp.ndarray, v_min: int = 2) -> float:
    """Ratio of recurrence points in vertical structures to total."""
    R_np = np.asarray(R)
    lines = _vertical_lines(R_np, v_min)
    if not lines:
        return 0.0
    vert_points = sum(lines)
    total = float(np.sum(R_np))
    if total <= 0:
        return 0.0
    return vert_points / total


def average_diagonal_length(R: jnp.ndarray, l_min: int = 2) -> float:
    """Mean length of diagonal lines."""
    lines = _diagonal_lines(np.asarray(R), l_min)
    return float(np.mean(lines)) if lines else 0.0


def trapping_time(R: jnp.ndarray, v_min: int = 2) -> float:
    """Mean length of vertical lines (trapping time)."""
    lines = _vertical_lines(np.asarray(R), v_min)
    return float(np.mean(lines)) if lines else 0.0


def max_diagonal_length(R: jnp.ndarray) -> int:
    """Length of the longest diagonal line."""
    lines = _diagonal_lines(np.asarray(R), l_min=1)
    return max(lines) if lines else 0


def max_vertical_length(R: jnp.ndarray) -> int:
    """Length of the longest vertical line."""
    lines = _vertical_lines(np.asarray(R), v_min=1)
    return max(lines) if lines else 0


def diagonal_entropy(R: jnp.ndarray, l_min: int = 2) -> float:
    """Shannon entropy of the diagonal line length distribution."""
    lines = _diagonal_lines(np.asarray(R), l_min)
    if not lines:
        return 0.0
    lengths = np.array(lines)
    counts = np.bincount(lengths)
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


def rqa_summary(R: jnp.ndarray, l_min: int = 2, v_min: int = 2) -> dict:
    """Compute all RQA measures at once."""
    return {
        "recurrence_rate": recurrence_rate_measure(R),
        "determinism": determinism(R, l_min),
        "laminarity": laminarity(R, v_min),
        "average_diagonal_length": average_diagonal_length(R, l_min),
        "trapping_time": trapping_time(R, v_min),
        "max_diagonal_length": max_diagonal_length(R),
        "max_vertical_length": max_vertical_length(R),
        "diagonal_entropy": diagonal_entropy(R, l_min),
    }
