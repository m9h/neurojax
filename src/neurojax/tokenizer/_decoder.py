"""Convolutional decoder for electrophysiology tokenization.

Ports EphysTokenizer's Conv1d decoder: each token has a learned waveform
kernel of length token_dim, and the output is the sum over token
reconstructions.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


class Conv1dDecoder(eqx.Module):
    """1D convolutional decoder mapping token weight sequences to waveforms.

    Each token contributes a learned waveform kernel. The reconstruction
    is the sum of all token contributions after convolution.
    """

    kernel: Float[Array, "n_tokens n_tokens token_dim"]
    bias: Float[Array, "n_tokens"]
    n_tokens: int = eqx.field(static=True)
    token_dim: int = eqx.field(static=True)

    def __init__(self, n_tokens: int, token_dim: int = 10, *, key: jax.Array):
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        k1, k2 = jax.random.split(key)
        # Xavier-like initialization
        scale = (2.0 / (n_tokens + n_tokens * token_dim)) ** 0.5
        self.kernel = jax.random.normal(k1, (n_tokens, n_tokens, token_dim)) * scale
        self.bias = jnp.zeros(n_tokens)

    def __call__(self, token_weights: Float[Array, "n_tokens L"]) -> Float[Array, "L"]:
        """Decode token weight sequences to a single-channel waveform.

        Args:
            token_weights: Token weights of shape (n_tokens, L).

        Returns:
            Reconstructed waveform of shape (L,).
        """
        # Manual "same" padding: total padding = token_dim - 1
        pad_total = self.token_dim - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        x = jnp.pad(token_weights, ((0, 0), (pad_left, pad_right)))
        # Add batch dim for conv: (1, n_tokens, L_padded)
        x = x[None, :, :]

        # conv_general_dilated: lhs=(N,C,W), rhs=(O,I,W) -> (N,O,W')
        y = jax.lax.conv_general_dilated(
            x,
            self.kernel,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NCW", "OIW", "NCW"),
        )
        # y: (1, n_tokens, L) -> sum over token dim -> (L,)
        return jnp.sum(y[0] + self.bias[:, None], axis=0)
