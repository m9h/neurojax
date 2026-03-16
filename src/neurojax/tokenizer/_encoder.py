"""RNN Encoder for electrophysiology tokenization.

Ports EphysTokenizer's stacked GRU from PyTorch to JAX/Equinox,
using eqx.nn.GRUCell inside jax.lax.scan for per-channel encoding.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


class RNNEncoder(eqx.Module):
    """Stacked GRU encoder that maps a univariate time series to hidden states.

    Designed to be vmapped over batch and channel dimensions.

    EphysTokenizer approach:
        (B,L,C) -> permute to (B,C,L) -> reshape to (B*C,L,1) -> nn.GRU

    JAX approach:
        (B,L,C) -> transpose to (B,C,L) -> vmap(vmap(encoder)) over B,C
        where encoder uses eqx.nn.GRUCell inside jax.lax.scan.
    """

    cells: list
    n_layers: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, hidden_size: int = 64, n_layers: int = 1, *, key: jax.Array):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        keys = jax.random.split(key, n_layers)
        cells = []
        for i in range(n_layers):
            input_size = 1 if i == 0 else hidden_size
            cells.append(eqx.nn.GRUCell(input_size, hidden_size, key=keys[i]))
        self.cells = cells

    def __call__(self, x: Float[Array, "L"]) -> Float[Array, "L hidden"]:
        """Encode a single-channel time series.

        Args:
            x: Univariate time series of shape (L,).

        Returns:
            Hidden states of shape (L, hidden_size).
        """
        # (L,) -> (L, 1) for first layer input
        h_seq = x[:, None]

        for cell in self.cells:
            h0 = jnp.zeros(self.hidden_size)

            def step(carry, x_t):
                h = cell(x_t, carry)
                return h, h

            _, h_seq = jax.lax.scan(step, h0, h_seq)

        return h_seq
