"""Evaluation metrics for tokenizer quality."""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array


@jax.jit
def pve(x: Float[Array, "..."], x_hat: Float[Array, "..."]) -> Float[Array, ""]:
    """Percentage of Variance Explained.

    PVE = 100 * (1 - var(residual) / var(x))

    Args:
        x: Original signal.
        x_hat: Reconstructed signal (same shape as x).

    Returns:
        Scalar PVE in [0, 100] (can be negative for bad reconstructions).
    """
    return 100.0 * (1.0 - jnp.var(x - x_hat) / jnp.maximum(jnp.var(x), 1e-10))


@jax.jit
def pve_per_channel(
    x: Float[Array, "B L C"], x_hat: Float[Array, "B L C"]
) -> Float[Array, "C"]:
    """PVE computed independently per channel.

    Args:
        x: Original signal of shape (B, L, C).
        x_hat: Reconstructed signal of shape (B, L, C).

    Returns:
        PVE per channel of shape (C,).
    """
    residual = x - x_hat
    var_res = jnp.var(residual, axis=(0, 1))
    var_x = jnp.maximum(jnp.var(x, axis=(0, 1)), 1e-10)
    return 100.0 * (1.0 - var_res / var_x)


def token_utilization(
    token_ids: Int[Array, "..."], n_tokens: int
) -> Float[Array, ""]:
    """Fraction of codebook entries used at least once.

    Args:
        token_ids: Flat or multi-dimensional token IDs.
        n_tokens: Total codebook size.

    Returns:
        Scalar utilization in [0, 1].
    """
    flat = token_ids.ravel()
    counts = jnp.zeros(n_tokens, dtype=jnp.int32)
    counts = counts.at[flat].add(1)
    used = jnp.sum(counts > 0)
    return used.astype(jnp.float32) / n_tokens
