"""Quantization backends for electrophysiology tokenization.

Three swappable quantizers:
  - TemperatureQuantizer: direct port of EphysTokenizer's softmax + STE approach
  - VQQuantizer: VQ-VAE with straight-through estimator
  - FSQQuantizer: Finite Scalar Quantization (codebook-free)
"""

import math
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


class TemperatureQuantizer(eqx.Module):
    """Temperature-annealed soft-to-hard quantizer (EphysTokenizer port).

    At temperature=1.0, outputs soft token weights (fully differentiable).
    At temperature=0.0, outputs hard one-hot via straight-through estimator.
    """

    linear: eqx.nn.Linear
    layer_norm: eqx.nn.LayerNorm
    n_tokens: int = eqx.field(static=True)

    def __init__(self, hidden_size: int, n_tokens: int, *, key: jax.Array):
        self.n_tokens = n_tokens
        self.linear = eqx.nn.Linear(hidden_size, n_tokens, key=key)
        self.layer_norm = eqx.nn.LayerNorm(n_tokens)

    def __call__(
        self, h: Float[Array, "hidden"], temperature: Float[Array, ""]
    ) -> Float[Array, "n_tokens"]:
        """Quantize a hidden state vector to token weights.

        Args:
            h: Hidden state of shape (hidden_size,).
            temperature: Scalar in [0, 1]. 1.0 = soft, 0.0 = hard (STE).

        Returns:
            Token weights of shape (n_tokens,).
        """
        logits = jax.nn.softplus(self.linear(h))
        logits = self.layer_norm(logits) / 0.1
        soft = jax.nn.softmax(logits)
        hard = jax.nn.one_hot(jnp.argmax(logits), self.n_tokens)
        hard_st = jax.lax.stop_gradient(hard - soft) + soft  # STE
        return temperature * soft + (1.0 - temperature) * hard_st


class VQQuantizer(eqx.Module):
    """Vector Quantization (VQ-VAE) with straight-through estimator.

    Maintains a learnable codebook. During forward pass, finds the nearest
    codebook entry and returns one-hot. Gradients flow through via STE.
    """

    codebook: Float[Array, "n_tokens hidden"]
    commitment_cost: float = eqx.field(static=True)
    n_tokens: int = eqx.field(static=True)

    def __init__(
        self, hidden_size: int, n_tokens: int, commitment_cost: float = 0.25, *, key: jax.Array
    ):
        self.n_tokens = n_tokens
        self.commitment_cost = commitment_cost
        self.codebook = jax.random.normal(key, (n_tokens, hidden_size)) * 0.1

    def __call__(
        self, h: Float[Array, "hidden"], temperature: Float[Array, ""]
    ) -> Float[Array, "n_tokens"]:
        """Quantize via nearest codebook entry.

        Args:
            h: Hidden state of shape (hidden_size,).
            temperature: Unused (API compatibility). VQ is always hard.

        Returns:
            One-hot token weights of shape (n_tokens,).
        """
        dists = jnp.sum((h[None, :] - self.codebook) ** 2, axis=-1)
        idx = jnp.argmin(dists)
        hard = jax.nn.one_hot(idx, self.n_tokens)
        # STE: gradients flow through as if identity
        return jax.lax.stop_gradient(hard - jnp.zeros_like(hard)) + jnp.zeros_like(hard)

    def commitment_loss(
        self, h: Float[Array, "hidden"]
    ) -> Float[Array, ""]:
        """Commitment loss for VQ-VAE training."""
        dists = jnp.sum((h[None, :] - self.codebook) ** 2, axis=-1)
        idx = jnp.argmin(dists)
        z_q = self.codebook[idx]
        # codebook loss + commitment loss
        codebook_loss = jnp.mean((jax.lax.stop_gradient(h) - z_q) ** 2)
        commit_loss = jnp.mean((h - jax.lax.stop_gradient(z_q)) ** 2)
        return codebook_loss + self.commitment_cost * commit_loss


def _round_ste(z: Float[Array, "D"], levels: tuple[int, ...]) -> Float[Array, "D"]:
    """Round each dimension to its quantization level with straight-through."""
    levels_arr = jnp.array(levels, dtype=jnp.float32)
    half_width = (levels_arr - 1) / 2
    # Scale from [-1, 1] to [0, levels-1], round, scale back
    z_scaled = (z + 1) / 2 * (levels_arr - 1)
    z_rounded = jnp.round(z_scaled)
    z_rounded = jnp.clip(z_rounded, 0, levels_arr - 1)
    # STE: forward uses rounded, backward uses continuous
    return z + jax.lax.stop_gradient(z_rounded / half_width - 1.0 - z)


def _multi_dim_to_flat(z_q: Float[Array, "D"], levels: tuple[int, ...]) -> int:
    """Convert multi-dimensional quantized indices to flat index."""
    levels_arr = jnp.array(levels, dtype=jnp.float32)
    half_width = (levels_arr - 1) / 2
    # Convert from [-1, 1] range to integer indices
    indices = jnp.round((z_q + 1) / 2 * (levels_arr - 1)).astype(jnp.int32)
    # Compute flat index via mixed-radix encoding
    flat = jnp.array(0, dtype=jnp.int32)
    for i in range(len(levels)):
        flat = flat * levels[i] + indices[i]
    return flat


class FSQQuantizer(eqx.Module):
    """Finite Scalar Quantization — codebook-free quantizer.

    Projects hidden state to a low-dimensional space, then rounds each
    dimension to a finite set of levels. Total codebook size = prod(levels).

    Reference: Mentzer et al., "Finite Scalar Quantization", ICLR 2024.
    """

    projection: eqx.nn.Linear
    levels: tuple[int, ...] = eqx.field(static=True)
    n_tokens: int = eqx.field(static=True)

    def __init__(self, hidden_size: int, levels: tuple[int, ...] = (8, 6, 5), *, key: jax.Array):
        self.levels = levels
        self.n_tokens = math.prod(levels)
        fsq_dim = len(levels)
        self.projection = eqx.nn.Linear(hidden_size, fsq_dim, key=key)

    def __call__(
        self, h: Float[Array, "hidden"], temperature: Float[Array, ""]
    ) -> Float[Array, "n_tokens"]:
        """Quantize via finite scalar quantization.

        Args:
            h: Hidden state of shape (hidden_size,).
            temperature: Unused (API compatibility).

        Returns:
            One-hot token weights of shape (n_tokens,).
        """
        z = jnp.tanh(self.projection(h))
        z_q = _round_ste(z, self.levels)
        flat_idx = _multi_dim_to_flat(z_q, self.levels)
        return jax.nn.one_hot(flat_idx, self.n_tokens)
