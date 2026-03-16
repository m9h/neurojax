"""Minimal transformer consumer for tokenization quality validation.

A small causal transformer that validates tokenization quality via
next-token prediction. Not a foundation model — a diagnostic tool.
"""

import math
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax import vmap
from jaxtyping import Float, Int, Array


class _TransformerBlock(eqx.Module):
    """Single causal self-attention block with residual + LayerNorm."""

    attn_proj: eqx.nn.Linear  # (embed,) -> (3*embed,) for Q,K,V
    attn_out: eqx.nn.Linear  # (embed,) -> (embed,)
    ff1: eqx.nn.Linear
    ff2: eqx.nn.Linear
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    embed_dim: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)

    def __init__(self, embed_dim: int, n_heads: int = 4, *, key: jax.Array):
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.attn_proj = eqx.nn.Linear(embed_dim, 3 * embed_dim, key=k1)
        self.attn_out = eqx.nn.Linear(embed_dim, embed_dim, key=k2)
        self.ff1 = eqx.nn.Linear(embed_dim, 4 * embed_dim, key=k3)
        self.ff2 = eqx.nn.Linear(4 * embed_dim, embed_dim, key=k4)
        self.ln1 = eqx.nn.LayerNorm(embed_dim)
        self.ln2 = eqx.nn.LayerNorm(embed_dim)

    def __call__(self, x: Float[Array, "T D"], *, key: jax.Array | None = None):
        T, D = x.shape
        head_dim = D // self.n_heads

        # Pre-norm self-attention
        h = vmap(self.ln1)(x)
        qkv = vmap(self.attn_proj)(h)  # (T, 3D)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each (T, D)

        # Reshape for multi-head: (T, n_heads, head_dim)
        q = q.reshape(T, self.n_heads, head_dim)
        k = k.reshape(T, self.n_heads, head_dim)
        v = v.reshape(T, self.n_heads, head_dim)

        # Attention scores: (n_heads, T, T)
        scale = math.sqrt(head_dim)
        scores = jnp.einsum("thd,shd->hts", q, k) / scale

        # Causal mask
        mask = jnp.tril(jnp.ones((T, T)))
        scores = jnp.where(mask[None], scores, -1e9)
        weights = jax.nn.softmax(scores, axis=-1)

        # Weighted sum: (n_heads, T, head_dim)
        attn_out = jnp.einsum("hts,shd->thd", weights, v)
        attn_out = attn_out.reshape(T, D)
        attn_out = vmap(self.attn_out)(attn_out)
        x = x + attn_out

        # Pre-norm feedforward
        h = vmap(self.ln2)(x)
        ff_out = vmap(self.ff2)(jax.nn.gelu(vmap(self.ff1)(h)))
        x = x + ff_out

        return x


class TokenConsumer(eqx.Module):
    """Minimal causal transformer for validating tokenization quality.

    Trains on next-token prediction over token sequences produced by
    a tokenizer. Lower perplexity = more learnable tokenization.
    """

    embedding: eqx.nn.Embedding
    blocks: list
    ln_final: eqx.nn.LayerNorm
    lm_head: eqx.nn.Linear
    max_seq_len: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        max_seq_len: int = 512,
        *,
        key: jax.Array,
    ):
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        keys = jax.random.split(key, n_layers + 2)
        self.embedding = eqx.nn.Embedding(vocab_size, embed_dim, key=keys[0])
        self.blocks = [
            _TransformerBlock(embed_dim, n_heads, key=keys[i + 1])
            for i in range(n_layers)
        ]
        self.ln_final = eqx.nn.LayerNorm(embed_dim)
        self.lm_head = eqx.nn.Linear(embed_dim, vocab_size, key=keys[-1])

    def __call__(
        self, token_ids: Int[Array, "T"], *, key: jax.Array | None = None
    ) -> Float[Array, "T vocab"]:
        """Forward pass: token IDs -> next-token logits.

        Args:
            token_ids: Input token IDs of shape (T,).
            key: PRNG key (unused in this minimal version).

        Returns:
            Logits of shape (T, vocab_size).
        """
        T = token_ids.shape[0]
        x = vmap(self.embedding)(token_ids)  # (T, embed_dim)

        # Sinusoidal positional encoding
        pos = jnp.arange(T)[:, None]
        dim = jnp.arange(0, self.embed_dim, 2)[None, :]
        angle = pos / (10000.0 ** (dim / self.embed_dim))
        pe = jnp.zeros((T, self.embed_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(angle))
        pe = pe.at[:, 1::2].set(jnp.cos(angle))
        x = x + pe

        for block in self.blocks:
            x = block(x, key=key)

        x = vmap(self.ln_final)(x)
        return vmap(self.lm_head)(x)

    def loss(
        self, token_ids: Int[Array, "T"], *, key: jax.Array | None = None
    ) -> Float[Array, ""]:
        """Cross-entropy loss for next-token prediction.

        Args:
            token_ids: Full sequence of shape (T,). Uses [:-1] as input
                and [1:] as targets.
            key: PRNG key.

        Returns:
            Scalar mean cross-entropy loss.
        """
        logits = self(token_ids[:-1], key=key)
        targets = token_ids[1:]
        return jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        )


def evaluate_tokenizer(tokenizer, consumer, data, *, key):
    """Evaluate tokenization quality using a trained consumer.

    Args:
        tokenizer: A trained EphysTokenizer.
        consumer: A trained TokenConsumer.
        data: Test data of shape (N, L, C).
        key: PRNG key.

    Returns:
        Dict with perplexity, token_utilization, reconstruction_pve.
    """
    from neurojax.tokenizer._metrics import pve, token_utilization

    out = tokenizer(data, temperature=jnp.array(0.01))

    # Perplexity from consumer
    # Flatten token_ids: (N, C, L) -> take first sample, first channel
    sample_tokens = out.token_ids[0, 0]  # (L,)
    ce_loss = consumer.loss(sample_tokens, key=key)
    perplexity = jnp.exp(ce_loss)

    # Token utilization
    util = token_utilization(out.token_ids, tokenizer.n_tokens)

    # Reconstruction PVE
    reconstruction_pve = pve(data, out.reconstruction)

    return {
        "perplexity": float(perplexity),
        "token_utilization": float(util),
        "reconstruction_pve": float(reconstruction_pve),
    }
