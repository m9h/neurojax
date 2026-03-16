"""Extension tokenizers leveraging existing neurojax modules.

These go beyond per-channel sample-level tokens:
  - SignatureTokenizer: rough path signatures over sliding windows
  - RiemannianTokenizer: SPD covariance manifold tokenization
  - TFRTokenizer: time-frequency patch tokenization via superlets
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jax import vmap
from functools import partial
from jaxtyping import Float, Int, Array

from neurojax.analysis.rough import sliding_signature
from neurojax.geometry.riemann import covariance_mean, map_tangent_space
from neurojax.analysis.superlet import superlet_transform
from neurojax.tokenizer._quantizer import VQQuantizer


class SignatureTokenizer(eqx.Module):
    """Tokenizer based on rough path signatures over sliding windows.

    Uses neurojax.analysis.rough.sliding_signature to compute path
    signatures, projects them, then quantizes via VQ codebook.

    Input: (T, C) multivariate time series
    Output: (n_windows,) token IDs representing window-level events
    """

    depth: int = eqx.field(static=True)
    window_size: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    sig_proj: eqx.nn.Linear
    codebook: VQQuantizer

    def __init__(
        self,
        sig_dim: int,
        embed_dim: int = 32,
        n_tokens: int = 64,
        depth: int = 3,
        window_size: int = 100,
        stride: int = 50,
        *,
        key: jax.Array,
    ):
        self.depth = depth
        self.window_size = window_size
        self.stride = stride
        k1, k2 = jax.random.split(key)
        self.sig_proj = eqx.nn.Linear(sig_dim, embed_dim, key=k1)
        self.codebook = VQQuantizer(embed_dim, n_tokens, key=k2)

    def __call__(
        self, data: Float[Array, "T C"], *, key: jax.Array | None = None
    ) -> tuple[Int[Array, "n_windows"], Float[Array, "n_windows embed"]]:
        """Tokenize a multivariate time series via path signatures.

        Args:
            data: Input time series of shape (T, C).
            key: Unused (API compatibility).

        Returns:
            Tuple of (token_ids, embeddings).
            token_ids: (n_windows,) integer token IDs.
            embeddings: (n_windows, embed_dim) projected signature vectors.
        """
        sigs = sliding_signature(data, self.depth, self.window_size, self.stride)
        z = vmap(self.sig_proj)(sigs)
        tokens = vmap(lambda h: self.codebook(h, jnp.array(0.0)))(z)
        token_ids = jnp.argmax(tokens, axis=-1)
        return token_ids, z


def _windowed_covariance(
    data: Float[Array, "C T"], window_size: int, stride: int
) -> Float[Array, "n_windows C C"]:
    """Compute covariance matrices over sliding windows.

    Args:
        data: Input of shape (C, T).
        window_size: Window length in samples.
        stride: Step size between windows.

    Returns:
        Covariance matrices of shape (n_windows, C, C).
    """
    C, T = data.shape
    n_windows = (T - window_size) // stride + 1
    starts = jnp.arange(n_windows) * stride

    def get_cov(start):
        window = jax.lax.dynamic_slice(data, (0, start), (C, window_size))
        window = window - jnp.mean(window, axis=1, keepdims=True)
        return window @ window.T / (window_size - 1)

    return vmap(get_cov)(starts)


class RiemannianTokenizer(eqx.Module):
    """Tokenizer based on Riemannian geometry of covariance matrices.

    Computes windowed covariance matrices, projects to tangent space
    at the Frechet mean, then quantizes the tangent vectors.

    Input: (C, T) multichannel time series
    Output: (n_windows,) token IDs representing covariance states
    """

    window_size: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    reference_mean: Float[Array, "C C"]
    tangent_proj: eqx.nn.Linear
    codebook: VQQuantizer

    def __init__(
        self,
        n_channels: int,
        embed_dim: int = 32,
        n_tokens: int = 64,
        window_size: int = 100,
        stride: int = 50,
        *,
        reference_mean: Float[Array, "C C"] | None = None,
        key: jax.Array,
    ):
        self.window_size = window_size
        self.stride = stride
        tangent_dim = n_channels * (n_channels + 1) // 2
        k1, k2 = jax.random.split(key)
        self.tangent_proj = eqx.nn.Linear(tangent_dim, embed_dim, key=k1)
        self.codebook = VQQuantizer(embed_dim, n_tokens, key=k2)
        if reference_mean is not None:
            self.reference_mean = reference_mean
        else:
            self.reference_mean = jnp.eye(n_channels)

    def fit_reference(self, data: Float[Array, "C T"]):
        """Compute and store Frechet mean from training data.

        Returns a new instance with updated reference_mean.
        """
        covs = _windowed_covariance(data, self.window_size, self.stride)
        mean = covariance_mean(covs)
        return eqx.tree_at(lambda m: m.reference_mean, self, mean)

    def __call__(
        self, data: Float[Array, "C T"], *, key: jax.Array | None = None
    ) -> tuple[Int[Array, "n_windows"], Float[Array, "n_windows embed"]]:
        """Tokenize via Riemannian covariance analysis.

        Args:
            data: Input of shape (C, T).
            key: Unused (API compatibility).

        Returns:
            Tuple of (token_ids, embeddings).
        """
        covs = _windowed_covariance(data, self.window_size, self.stride)
        tangent_vecs = map_tangent_space(covs, mean=self.reference_mean)
        z = vmap(self.tangent_proj)(tangent_vecs)
        tokens = vmap(lambda h: self.codebook(h, jnp.array(0.0)))(z)
        token_ids = jnp.argmax(tokens, axis=-1)
        return token_ids, z


def _extract_patches(
    tfr: Float[Array, "C F T"], patch_freq: int, patch_time: int
) -> Float[Array, "C n_patches patch_size"]:
    """Extract non-overlapping patches from a time-frequency representation.

    Args:
        tfr: TFR of shape (n_ch, n_freqs, n_times).
        patch_freq: Patch height (frequency bins).
        patch_time: Patch width (time bins).

    Returns:
        Patches of shape (n_ch, n_patches, patch_freq * patch_time).
    """
    C, F, T = tfr.shape
    nf = F // patch_freq
    nt = T // patch_time
    # Trim to exact patch grid
    tfr = tfr[:, : nf * patch_freq, : nt * patch_time]
    # Reshape to patches
    tfr = tfr.reshape(C, nf, patch_freq, nt, patch_time)
    tfr = jnp.transpose(tfr, (0, 1, 3, 2, 4))  # (C, nf, nt, pf, pt)
    tfr = tfr.reshape(C, nf * nt, patch_freq * patch_time)
    return tfr


class TFRTokenizer(eqx.Module):
    """Tokenizer based on time-frequency representation patches.

    Computes superlet TFR, extracts non-overlapping patches, projects
    each patch, then quantizes.

    Input: (n_ch, n_times) + frequency parameters
    Output: (n_ch, n_patches) token IDs
    """

    patch_freq: int = eqx.field(static=True)
    patch_time: int = eqx.field(static=True)
    patch_proj: eqx.nn.Linear
    codebook: VQQuantizer

    def __init__(
        self,
        patch_freq: int = 4,
        patch_time: int = 8,
        embed_dim: int = 32,
        n_tokens: int = 64,
        *,
        key: jax.Array,
    ):
        self.patch_freq = patch_freq
        self.patch_time = patch_time
        patch_size = patch_freq * patch_time
        k1, k2 = jax.random.split(key)
        self.patch_proj = eqx.nn.Linear(patch_size, embed_dim, key=k1)
        self.codebook = VQQuantizer(embed_dim, n_tokens, key=k2)

    def __call__(
        self,
        data: Float[Array, "C T"],
        sfreq: float,
        freqs: tuple,
        *,
        key: jax.Array | None = None,
    ) -> tuple[Int[Array, "C n_patches"], Float[Array, "C n_patches embed"]]:
        """Tokenize via time-frequency patch quantization.

        Args:
            data: Input of shape (n_ch, n_times).
            sfreq: Sampling frequency.
            freqs: Tuple of frequencies for superlet transform.
            key: Unused (API compatibility).

        Returns:
            Tuple of (token_ids, embeddings).
        """
        tfr = superlet_transform(data, sfreq, freqs)
        patches = _extract_patches(tfr, self.patch_freq, self.patch_time)
        # patches: (C, n_patches, patch_size)
        z = vmap(vmap(self.patch_proj))(patches)
        tokens = vmap(vmap(lambda h: self.codebook(h, jnp.array(0.0))))(z)
        token_ids = jnp.argmax(tokens, axis=-1)
        return token_ids, z
