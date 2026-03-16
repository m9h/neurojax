"""Non-learnable baseline tokenization methods.

Provides mu-law and quantile tokenization for comparison against
the learned EphysTokenizer.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array


@jax.jit
def mu_law_tokenize(
    x: Float[Array, "..."], n_tokens: int = 256, mu: float = 255.0
) -> Int[Array, "..."]:
    """Mu-law companding tokenization.

    Standard audio quantization baseline. Applies mu-law compression
    then uniform binning.

    Args:
        x: Input signal (any shape).
        n_tokens: Number of quantization bins.
        mu: Mu-law parameter (higher = more compression).

    Returns:
        Token IDs in [0, n_tokens-1], same shape as input.
    """
    x_norm = x / jnp.maximum(jnp.max(jnp.abs(x)), 1e-10)
    compressed = jnp.sign(x_norm) * jnp.log1p(mu * jnp.abs(x_norm)) / jnp.log1p(mu)
    # Map from [-1, 1] to [0, n_tokens-1]
    tokens = jnp.floor((compressed + 1.0) / 2.0 * n_tokens)
    return jnp.clip(tokens, 0, n_tokens - 1).astype(jnp.int32)


@jax.jit
def mu_law_detokenize(
    tokens: Int[Array, "..."], n_tokens: int = 256, mu: float = 255.0
) -> Float[Array, "..."]:
    """Inverse mu-law: reconstruct signal from tokens.

    Args:
        tokens: Token IDs in [0, n_tokens-1].
        n_tokens: Number of quantization bins.
        mu: Mu-law parameter (must match encoding).

    Returns:
        Reconstructed signal in [-1, 1].
    """
    compressed = (tokens.astype(jnp.float32) + 0.5) / n_tokens * 2.0 - 1.0
    return jnp.sign(compressed) * (jnp.power(1.0 + mu, jnp.abs(compressed)) - 1.0) / mu


def quantile_tokenize(
    x: Float[Array, "..."], n_tokens: int = 108
) -> tuple[Int[Array, "..."], Float[Array, "n_tokens_plus_1"]]:
    """Quantile-based tokenization with equal-frequency bins.

    Not JIT-compatible due to data-dependent quantile computation.

    Args:
        x: Input signal (any shape).
        n_tokens: Number of quantization bins.

    Returns:
        Tuple of (token_ids, boundaries).
        token_ids: Token IDs in [0, n_tokens-1], same shape as input.
        boundaries: Bin boundaries of shape (n_tokens+1,).
    """
    boundaries = jnp.quantile(x.ravel(), jnp.linspace(0, 1, n_tokens + 1))
    tokens = jnp.digitize(x, boundaries[1:-1])  # n_tokens bins between n_tokens+1 edges
    return tokens.astype(jnp.int32), boundaries
