"""Composed EphysTokenizer: encoder + quantizer + decoder.

Ports the EphysTokenizer architecture (OHBA, Oxford) to JAX/Equinox,
composing RNNEncoder, a swappable quantizer, and Conv1dDecoder into
a single differentiable module.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jax import vmap
from jaxtyping import Float, Int, Array
from typing import NamedTuple

from neurojax.tokenizer._encoder import RNNEncoder
from neurojax.tokenizer._quantizer import TemperatureQuantizer, VQQuantizer, FSQQuantizer
from neurojax.tokenizer._decoder import Conv1dDecoder


class TokenizerOutput(NamedTuple):
    """Output of the EphysTokenizer forward pass."""

    token_ids: Int[Array, "..."]
    token_weights: Float[Array, "... n_tokens"]
    reconstruction: Float[Array, "..."]
    loss: Float[Array, ""]


class EphysTokenizer(eqx.Module):
    """JAX/Equinox port of EphysTokenizer for sample-level MEG/EEG tokenization.

    Architecture:
        1. RNNEncoder: per-channel GRU encoding (B,L,C) -> (B,C,L,hidden)
        2. Quantizer: hidden -> token weights (B,C,L,n_tokens)
        3. Conv1dDecoder: token weights -> reconstruction (B,L,C)

    Temperature is passed as an argument (not model state) to keep the
    model purely functional and avoid JIT recompilation.
    """

    encoder: RNNEncoder
    quantizer: TemperatureQuantizer | VQQuantizer | FSQQuantizer
    decoder: Conv1dDecoder
    n_tokens: int = eqx.field(static=True)

    def __init__(
        self,
        n_tokens: int = 64,
        hidden_size: int = 64,
        token_dim: int = 10,
        n_layers: int = 1,
        quantizer_type: str = "temperature",
        fsq_levels: tuple[int, ...] = (8, 6, 5),
        *,
        key: jax.Array,
    ):
        k1, k2, k3 = jax.random.split(key, 3)

        self.encoder = RNNEncoder(hidden_size=hidden_size, n_layers=n_layers, key=k1)

        if quantizer_type == "temperature":
            self.quantizer = TemperatureQuantizer(hidden_size, n_tokens, key=k2)
            self.n_tokens = n_tokens
        elif quantizer_type == "vq":
            self.quantizer = VQQuantizer(hidden_size, n_tokens, key=k2)
            self.n_tokens = n_tokens
        elif quantizer_type == "fsq":
            self.quantizer = FSQQuantizer(hidden_size, fsq_levels, key=k2)
            self.n_tokens = self.quantizer.n_tokens
        else:
            raise ValueError(f"Unknown quantizer_type: {quantizer_type}")

        self.decoder = Conv1dDecoder(self.n_tokens, token_dim, key=k3)

    @eqx.filter_jit
    def __call__(
        self,
        x: Float[Array, "B L C"],
        temperature: Float[Array, ""] = jnp.array(1.0),
    ) -> TokenizerOutput:
        """Forward pass: encode, quantize, decode.

        Args:
            x: Input data of shape (B, L, C).
            temperature: Annealing temperature. 1.0 = soft, 0.0 = hard.

        Returns:
            TokenizerOutput with token_ids, token_weights, reconstruction, loss.
        """
        B, L, C = x.shape

        # 1. Transpose: (B, L, C) -> (B, C, L)
        x_t = jnp.transpose(x, (0, 2, 1))

        # 2. Encode: vmap over B and C, encoder operates on (L,) -> (L, hidden)
        encode_channel = self.encoder  # (L,) -> (L, hidden)
        encode_batch_channels = vmap(vmap(encode_channel))  # (B, C, L) -> (B, C, L, hidden)
        hidden = encode_batch_channels(x_t)

        # 3. Quantize: vmap over B, C, L; quantizer operates on (hidden,) -> (n_tokens,)
        quantize_one = lambda h: self.quantizer(h, temperature)
        quantize_seq = vmap(quantize_one)  # (L, hidden) -> (L, n_tokens)
        quantize_channels = vmap(quantize_seq)  # (C, L, hidden) -> (C, L, n_tokens)
        quantize_batch = vmap(quantize_channels)  # (B, C, L, hidden) -> (B, C, L, n_tokens)
        token_weights = quantize_batch(hidden)

        # 4. Token IDs: argmax over token dim
        token_ids = jnp.argmax(token_weights, axis=-1)  # (B, C, L)

        # 5. Transpose for decoder: (B, C, L, n_tokens) -> (B, C, n_tokens, L)
        tw_t = jnp.transpose(token_weights, (0, 1, 3, 2))

        # 6. Decode: vmap over B and C, decoder operates on (n_tokens, L) -> (L,)
        decode_channel = self.decoder  # (n_tokens, L) -> (L,)
        decode_batch_channels = vmap(vmap(decode_channel))  # (B, C, n_tokens, L) -> (B, C, L)
        recon_t = decode_batch_channels(tw_t)

        # 7. Transpose back: (B, C, L) -> (B, L, C)
        reconstruction = jnp.transpose(recon_t, (0, 2, 1))

        # 8. MSE loss
        loss = jnp.mean((x - reconstruction) ** 2)

        return TokenizerOutput(
            token_ids=token_ids,
            token_weights=token_weights,
            reconstruction=reconstruction,
            loss=loss,
        )

    def tokenize(self, x: Float[Array, "B L C"]) -> Int[Array, "B C L"]:
        """Inference-only: return hard token IDs.

        Args:
            x: Input data of shape (B, L, C).

        Returns:
            Token IDs of shape (B, C, L) as integers.
        """
        out = self(x, temperature=jnp.array(0.0))
        return out.token_ids
