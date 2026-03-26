"""Surrogate data generation and significance testing.

Inspired by pyunicorn's Surrogates class. Generates surrogate time series
that preserve specific statistical properties of the original data, for
null hypothesis testing of coupling measures.

Surrogate types:
- Phase randomization: preserves power spectrum, destroys phase coupling
- AAFT: preserves amplitude distribution AND power spectrum
- Shuffle: destroys all temporal structure
- Block shuffle: preserves local structure within blocks
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


def phase_randomized_surrogate(
    x: jnp.ndarray, key: jax.Array
) -> jnp.ndarray:
    """Surrogate with randomized phases, preserving power spectrum.

    Parameters
    ----------
    x : (T,) or (T, C) time series.
    key : PRNG key.

    Returns
    -------
    surrogate : same shape as x, with same power spectrum but random phases.
    """
    if x.ndim == 1:
        return _phase_randomize_1d(x, key)
    else:
        keys = jr.split(key, x.shape[1])
        return jnp.stack(
            [_phase_randomize_1d(x[:, c], keys[c]) for c in range(x.shape[1])],
            axis=1,
        )


def _phase_randomize_1d(x: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Phase randomization for a single 1D signal."""
    T = x.shape[0]
    Xf = jnp.fft.rfft(x)
    amplitudes = jnp.abs(Xf)
    # Random phases (keep DC and Nyquist real)
    n_freqs = len(Xf)
    random_phases = jr.uniform(key, (n_freqs,), minval=0, maxval=2 * jnp.pi)
    random_phases = random_phases.at[0].set(0.0)
    if T % 2 == 0:
        random_phases = random_phases.at[-1].set(0.0)
    Xf_surr = amplitudes * jnp.exp(1j * random_phases)
    return jnp.fft.irfft(Xf_surr, n=T).real


def aaft_surrogate(x: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Amplitude Adjusted Fourier Transform surrogate.

    Preserves both amplitude distribution and (approximately) power spectrum.

    Parameters
    ----------
    x : (T,) or (T, C) time series.
    key : PRNG key.

    Returns
    -------
    surrogate : same shape, preserving amplitude distribution and spectrum.
    """
    if x.ndim == 1:
        return _aaft_1d(x, key)
    else:
        keys = jr.split(key, x.shape[1])
        return jnp.stack(
            [_aaft_1d(x[:, c], keys[c]) for c in range(x.shape[1])],
            axis=1,
        )


def _aaft_1d(x: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """AAFT for a single 1D signal."""
    x_np = np.asarray(x)
    T = len(x_np)

    # Step 1: Generate Gaussian with same rank order
    gaussian = np.sort(np.random.default_rng(int(key[0])).normal(size=T))
    rank_order = np.argsort(np.argsort(x_np))
    gaussian_ranked = gaussian[rank_order]

    # Step 2: Phase-randomize the ranked Gaussian
    key2 = jr.fold_in(key, 1)
    surr_gaussian = np.asarray(_phase_randomize_1d(jnp.array(gaussian_ranked), key2))

    # Step 3: Rank-order the original data to match the surrogate's rank
    surr_rank = np.argsort(np.argsort(surr_gaussian))
    x_sorted = np.sort(x_np)
    surrogate = x_sorted[surr_rank]

    return jnp.array(surrogate)


def shuffle_surrogate(x: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Random permutation surrogate (destroys all temporal structure).

    Parameters
    ----------
    x : (T,) or (T, C) time series.
    key : PRNG key.

    Returns
    -------
    surrogate : same shape, randomly permuted along time axis.
    """
    perm = jr.permutation(key, x.shape[0])
    return x[perm]


def block_shuffle_surrogate(
    x: jnp.ndarray, key: jax.Array, block_size: int = 50
) -> jnp.ndarray:
    """Block shuffle surrogate (preserves within-block structure).

    Parameters
    ----------
    x : (T,) or (T, C) time series.
    key : PRNG key.
    block_size : int — size of each block.

    Returns
    -------
    surrogate : same shape, blocks shuffled.
    """
    T = x.shape[0]
    n_blocks = T // block_size
    remainder = T % block_size

    # Split into blocks
    blocks = [x[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
    if remainder > 0:
        blocks.append(x[n_blocks * block_size:])

    # Shuffle block order
    perm = jr.permutation(key, len(blocks))
    perm_np = np.asarray(perm)
    shuffled = [blocks[int(i)] for i in perm_np]

    return jnp.concatenate(shuffled, axis=0)


# ---------------------------------------------------------------------------
# Significance testing
# ---------------------------------------------------------------------------

def surrogate_test(
    data: jnp.ndarray,
    statistic_fn: Callable,
    surrogate_fn: Callable,
    n_surrogates: int = 100,
    key: Optional[jax.Array] = None,
) -> dict:
    """Test whether a statistic is significant against surrogates.

    Parameters
    ----------
    data : input time series.
    statistic_fn : callable(data) → scalar statistic.
    surrogate_fn : callable(data, key) → surrogate data.
    n_surrogates : int — number of surrogates to generate.
    key : PRNG key.

    Returns
    -------
    dict with: observed, surrogate_distribution, p_value, significant.
    """
    if key is None:
        key = jr.PRNGKey(0)

    observed = float(statistic_fn(data))

    surr_stats = []
    for i in range(n_surrogates):
        k = jr.fold_in(key, i)
        surr = surrogate_fn(data, k)
        surr_stats.append(float(statistic_fn(surr)))

    surr_array = np.array(surr_stats)
    p_value = float(np.mean(surr_array >= observed))

    return {
        "observed": observed,
        "surrogate_distribution": surr_array,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }
