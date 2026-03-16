"""Training harness for EphysTokenizer.

Provides fit() with temperature annealing, optional VQ-VAE commitment loss,
checkpointing, and post-training vocabulary refactoring.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from pathlib import Path
from jaxtyping import Float, Array

from neurojax.tokenizer._vocab import refactor_vocabulary
from neurojax.tokenizer._quantizer import VQQuantizer


def _linear_anneal(epoch: int, start: float, end: float, anneal_epochs: int) -> float:
    """Linear annealing from start to end over anneal_epochs."""
    if epoch >= anneal_epochs:
        return end
    return start + (end - start) * epoch / anneal_epochs


def _batch_iterator(data: Float[Array, "N L C"], batch_size: int, key: jax.Array):
    """Yield shuffled batches from data."""
    n = data.shape[0]
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, n)
    data = data[perm]
    for i in range(0, n, batch_size):
        yield data[i : i + batch_size]


def fit(
    model,
    data: Float[Array, "N L C"],
    *,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    temp_start: float = 1.0,
    temp_end: float = 0.001,
    temp_epochs: int = 80,
    key: jax.Array,
    checkpoint_dir: str | None = None,
    refactor_vocab: bool = True,
):
    """Train an EphysTokenizer with temperature annealing.

    Args:
        model: An EphysTokenizer instance.
        data: Training data of shape (N, L, C) where N is number of segments.
        n_epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate for Adam optimizer.
        temp_start: Starting temperature (1.0 = fully soft).
        temp_end: Final temperature (0.001 ~ hard).
        temp_epochs: Number of epochs to anneal temperature.
        key: PRNG key for shuffling.
        checkpoint_dir: If provided, save model each epoch.
        refactor_vocab: If True, sort vocabulary by frequency after training.

    Returns:
        Tuple of (trained_model, history) where history is a dict with
        'loss' and 'temperature' per epoch.
    """
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    history = {"loss": [], "temperature": []}

    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    is_vq = isinstance(model.quantizer, VQQuantizer)

    for epoch in range(n_epochs):
        temp = jnp.array(_linear_anneal(epoch, temp_start, temp_end, temp_epochs))
        epoch_losses = []

        key, subkey = jax.random.split(key)
        for batch in _batch_iterator(data, batch_size, subkey):
            if batch.shape[0] == 0:
                continue

            @eqx.filter_value_and_grad
            def loss_fn(m):
                out = m(batch, temp)
                total_loss = out.loss
                if is_vq:
                    # Add VQ commitment loss averaged over all hidden states
                    # We recompute encoder output for commitment loss
                    x_t = jnp.transpose(batch, (0, 2, 1))
                    from jax import vmap

                    hidden = vmap(vmap(m.encoder))(x_t)
                    commit = vmap(
                        vmap(vmap(lambda h: m.quantizer.commitment_loss(h)))
                    )(hidden)
                    total_loss = total_loss + jnp.mean(commit)
                return total_loss

            loss, grads = loss_fn(model)
            updates, opt_state = opt.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, updates)
            epoch_losses.append(float(loss))

        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        history["loss"].append(avg_loss)
        history["temperature"].append(float(temp))

        if checkpoint_dir is not None:
            eqx.tree_serialise_leaves(
                str(Path(checkpoint_dir) / f"epoch_{epoch}.eqx"), model
            )

    # Post-training vocabulary refactoring
    if refactor_vocab:
        # Run forward pass to get token weights for frequency counting
        out = model(data[:min(batch_size, data.shape[0])], temperature=jnp.array(0.01))
        model, label_map = refactor_vocabulary(model, out.token_weights)

    return model, history
