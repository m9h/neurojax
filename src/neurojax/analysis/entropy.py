"""GPU-accelerated entropy measures for MEG/EEG time series.

JAX-native implementations of sample entropy, approximate entropy, SVD entropy,
and spectral entropy — vectorized across channels via jax.vmap for GPU
acceleration on the DGX Spark.

The mne-features implementations loop over channels with sklearn KDTree
radius queries, taking ~30+ minutes for 303 channels × 59 epochs on CPU.
These JAX versions compute all channels simultaneously on GPU, targeting
~10-30 seconds for the same workload.

Reference implementations: mne-features (Schiratti et al. 2018)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial


def _embed(x: jnp.ndarray, d: int, tau: int = 1) -> jnp.ndarray:
    """Time-delay embedding of a 1D signal.

    Parameters
    ----------
    x : (n_times,) array
    d : int — embedding dimension
    tau : int — delay (samples)

    Returns
    -------
    embedded : (n_times - (d-1)*tau, d) array
    """
    n = x.shape[0] - (d - 1) * tau
    indices = jnp.arange(n)[:, None] + jnp.arange(d)[None, :] * tau
    return x[indices]


def _chebyshev_count(emb: jnp.ndarray, r: float) -> jnp.ndarray:
    """Count template matches within Chebyshev distance r.

    Parameters
    ----------
    emb : (m, d) array — embedded vectors
    r : float — tolerance radius

    Returns
    -------
    counts : (m,) array — number of matches per vector (including self)
    """
    # Pairwise Chebyshev (L-inf) distance: max absolute difference
    # (m, 1, d) - (1, m, d) → (m, m, d) → max over d → (m, m)
    diff = jnp.abs(emb[:, None, :] - emb[None, :, :])
    dist = jnp.max(diff, axis=-1)
    return jnp.sum(dist <= r, axis=-1).astype(jnp.float32)


def _samp_entropy_single(x: jnp.ndarray, emb: int = 2) -> jnp.ndarray:
    """Sample entropy for a single channel.

    Parameters
    ----------
    x : (n_times,) array
    emb : int — embedding dimension (default 2)

    Returns
    -------
    sampen : scalar
    """
    r = 0.2 * jnp.std(x, ddof=1)

    # Embedding at dim=emb
    emb_data1 = _embed(x, emb, tau=1)[:-1]  # exclude last for SampEn
    count1 = _chebyshev_count(emb_data1, r)
    # Subtract self-match
    phi0 = jnp.mean((count1 - 1) / (emb_data1.shape[0] - 1))

    # Embedding at dim=emb+1
    emb_data2 = _embed(x, emb + 1, tau=1)
    count2 = _chebyshev_count(emb_data2, r)
    phi1 = jnp.mean((count2 - 1) / (emb_data2.shape[0] - 1))

    return -jnp.log(jnp.clip(phi1 / jnp.clip(phi0, 1e-10), 1e-10))


def _app_entropy_single(x: jnp.ndarray, emb: int = 2) -> jnp.ndarray:
    """Approximate entropy for a single channel.

    Parameters
    ----------
    x : (n_times,) array
    emb : int — embedding dimension (default 2)

    Returns
    -------
    appen : scalar
    """
    r = 0.2 * jnp.std(x, ddof=1)

    emb_data1 = _embed(x, emb, tau=1)
    count1 = _chebyshev_count(emb_data1, r)
    phi0 = jnp.mean(jnp.log(count1 / emb_data1.shape[0]))

    emb_data2 = _embed(x, emb + 1, tau=1)
    count2 = _chebyshev_count(emb_data2, r)
    phi1 = jnp.mean(jnp.log(count2 / emb_data2.shape[0]))

    return phi0 - phi1


def _svd_entropy_single(x: jnp.ndarray, tau: int = 2,
                         emb: int = 10) -> jnp.ndarray:
    """SVD entropy for a single channel.

    Parameters
    ----------
    x : (n_times,) array
    tau : int — delay
    emb : int — embedding dimension

    Returns
    -------
    svd_ent : scalar
    """
    embedded = _embed(x, emb, tau)
    _, sv, _ = jnp.linalg.svd(embedded, full_matrices=False)
    sv_norm = sv / jnp.sum(sv)
    return -jnp.sum(sv_norm * jnp.log2(jnp.clip(sv_norm, 1e-10)))


def _spect_entropy_single(x: jnp.ndarray, fs: float = 250.0) -> jnp.ndarray:
    """Spectral entropy for a single channel.

    Parameters
    ----------
    x : (n_times,) array
    fs : float — sampling frequency

    Returns
    -------
    spect_ent : scalar
    """
    # Welch-like PSD via FFT
    psd = jnp.abs(jnp.fft.rfft(x)) ** 2
    psd = psd[1:]  # drop DC
    psd_norm = psd / jnp.sum(psd)
    return -jnp.sum(psd_norm * jnp.log2(jnp.clip(psd_norm, 1e-10)))


# -------------------------------------------------------------------------
# Vectorized (vmap) versions — all channels at once
# -------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(1,))
def sample_entropy(data: jnp.ndarray, emb: int = 2) -> jnp.ndarray:
    """Sample entropy for all channels (GPU-accelerated).

    Parameters
    ----------
    data : (n_channels, n_times) array
    emb : int — embedding dimension

    Returns
    -------
    sampen : (n_channels,) array
    """
    return jax.vmap(partial(_samp_entropy_single, emb=emb))(data)


@partial(jax.jit, static_argnums=(1,))
def approx_entropy(data: jnp.ndarray, emb: int = 2) -> jnp.ndarray:
    """Approximate entropy for all channels (GPU-accelerated).

    Parameters
    ----------
    data : (n_channels, n_times) array
    emb : int — embedding dimension

    Returns
    -------
    appen : (n_channels,) array
    """
    return jax.vmap(partial(_app_entropy_single, emb=emb))(data)


@partial(jax.jit, static_argnums=(1, 2))
def svd_entropy(data: jnp.ndarray, tau: int = 2,
                emb: int = 10) -> jnp.ndarray:
    """SVD entropy for all channels (GPU-accelerated).

    Parameters
    ----------
    data : (n_channels, n_times) array
    tau : int — delay
    emb : int — embedding dimension

    Returns
    -------
    svd_ent : (n_channels,) array
    """
    return jax.vmap(partial(_svd_entropy_single, tau=tau, emb=emb))(data)


@partial(jax.jit, static_argnums=(1,))
def spectral_entropy(data: jnp.ndarray, fs: float = 250.0) -> jnp.ndarray:
    """Spectral entropy for all channels (GPU-accelerated).

    Parameters
    ----------
    data : (n_channels, n_times) array
    fs : float — sampling frequency

    Returns
    -------
    spect_ent : (n_channels,) array
    """
    return jax.vmap(partial(_spect_entropy_single, fs=fs))(data)


def compute_all_entropies(data: jnp.ndarray, fs: float = 250.0,
                          emb: int = 2) -> dict[str, jnp.ndarray]:
    """Compute all four entropy measures for all channels.

    Parameters
    ----------
    data : (n_channels, n_times) array
    fs : float — sampling frequency
    emb : int — embedding dimension for sample/approx entropy

    Returns
    -------
    dict with keys: 'sample_entropy', 'approx_entropy', 'svd_entropy',
    'spectral_entropy' — each (n_channels,) array
    """
    return {
        "sample_entropy": sample_entropy(data, emb=emb),
        "approx_entropy": approx_entropy(data, emb=emb),
        "svd_entropy": svd_entropy(data),
        "spectral_entropy": spectral_entropy(data, fs=fs),
    }
