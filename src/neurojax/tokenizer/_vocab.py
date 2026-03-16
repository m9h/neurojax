"""Post-training vocabulary refactoring.

Sorts tokens by frequency so that token 0 = most common, token 1 = next, etc.
Permutes model weights accordingly using eqx.tree_at (functional update).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Int, Array


def refactor_vocabulary(model, token_weights: Float[Array, "... n_tokens"]):
    """Reorder token IDs by descending frequency without changing model output.

    After training, token IDs are arbitrary. This function sorts them so that
    token 0 is the most frequently assigned token, token 1 is next, etc.
    It permutes the quantizer and decoder weights accordingly.

    Args:
        model: An EphysTokenizer instance.
        token_weights: Token weight tensor from a forward pass (any leading dims).

    Returns:
        Tuple of (updated_model, label_map) where label_map[old_id] = new_id.
    """
    n_tokens = model.n_tokens

    # 1. Hard-assign tokens
    token_ids = jnp.argmax(token_weights.reshape(-1, n_tokens), axis=-1)

    # 2. Count frequencies
    counts = jnp.zeros(n_tokens, dtype=jnp.int32)
    counts = counts.at[token_ids].add(1)

    # 3. Sort by descending frequency -> permutation
    # perm[i] = old token id that becomes new token i
    perm = jnp.argsort(-counts)
    # Inverse: label_map[old_id] = new_id
    label_map = jnp.zeros(n_tokens, dtype=jnp.int32)
    label_map = label_map.at[perm].set(jnp.arange(n_tokens))

    # 4. Permute decoder kernel: (n_tokens, n_tokens, token_dim)
    new_kernel = model.decoder.kernel[perm][:, perm]
    new_bias = model.decoder.bias[perm]

    model = eqx.tree_at(lambda m: m.decoder.kernel, model, new_kernel)
    model = eqx.tree_at(lambda m: m.decoder.bias, model, new_bias)

    # 5. Permute quantizer weights (type-dependent)
    from neurojax.tokenizer._quantizer import TemperatureQuantizer

    if isinstance(model.quantizer, TemperatureQuantizer):
        # Permute the linear layer output (weight rows and bias)
        new_weight = model.quantizer.linear.weight[perm]
        new_linear_bias = model.quantizer.linear.bias[perm]
        model = eqx.tree_at(lambda m: m.quantizer.linear.weight, model, new_weight)
        model = eqx.tree_at(lambda m: m.quantizer.linear.bias, model, new_linear_bias)
        # LayerNorm params
        new_ln_weight = model.quantizer.layer_norm.weight[perm]
        new_ln_bias = model.quantizer.layer_norm.bias[perm]
        model = eqx.tree_at(lambda m: m.quantizer.layer_norm.weight, model, new_ln_weight)
        model = eqx.tree_at(lambda m: m.quantizer.layer_norm.bias, model, new_ln_bias)

    return model, label_map
