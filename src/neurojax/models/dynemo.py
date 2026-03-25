"""Dynamic Network Modes (DyNeMo) — pure JAX, osl-dynamics-compatible.

A variational autoencoder that infers continuous mixing coefficients for
brain network modes.  Unlike an HMM (discrete, mutually exclusive states),
DyNeMo allows multiple modes to be active simultaneously via a softmax
mixing layer.

Architecture (following osl-dynamics)::

    Inference network:
        data -> Dropout -> BiGRU -> Dense(mu) + Dense(sigma, softplus)
        -> sample theta ~ N(mu, sigma)  (reparameterisation trick)
        -> alpha = softmax(theta / temperature)

    Observation model:
        m(t) = sum_k alpha_k(t) * mu_k          (mix means)
        C(t) = sum_k alpha_k(t) * Sigma_k        (mix covariances)
        x(t) ~ N(m(t), C(t))

    Temporal prior (model network):
        theta -> Dropout -> GRU -> Dense(mod_mu) + Dense(mod_sigma, softplus)
        KL[ q(theta_t | x) || p(theta_t | theta_{<t}) ]

    Loss = -ELBO = NLL + KL (with optional KL annealing)

Example
-------
>>> from neurojax.models.dynemo import DyNeMo, DyNeMoConfig
>>> model = DyNeMo(DyNeMoConfig(n_modes=8, n_channels=80))
>>> model.fit(prepared_data, n_epochs=30)
>>> alpha = model.infer(prepared_data)   # list of (T, K) mixing coefficients
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DyNeMoConfig:
    """Configuration for :class:`DyNeMo`.

    Parameters
    ----------
    n_modes : int
        Number of modes (soft states).
    n_channels : int
        Number of channels (observation dimensionality after TDE+PCA).
    sequence_length : int
        Length of subsequences used for training.
    inference_n_units : int
        Number of GRU hidden units in the inference network.
    inference_n_layers : int
        Number of GRU layers in the inference network.
    inference_dropout : float
        Dropout rate applied to data before the inference RNN.
    model_n_units : int
        Number of GRU hidden units in the model (prior) network.
    model_n_layers : int
        Number of GRU layers in the model network.
    model_dropout : float
        Dropout rate applied to theta before the model RNN.
    learn_means : bool
        Whether mode means are trainable.
    learn_covariances : bool
        Whether mode covariances are trainable.
    initial_means : optional array (K, C)
        Initial means for modes.
    initial_covariances : optional array (K, C, C)
        Initial covariances for modes.
    diagonal_covariances : bool
        If True, learn diagonal covariances only.
    covariances_epsilon : float
        Regularisation added to covariance diagonals.
    alpha_temperature : float
        Temperature for the softmax producing alpha.
    learn_alpha_temperature : bool
        Whether to learn the softmax temperature.
    theta_std_epsilon : float
        Small value added to inference sigma for numerical stability.
    do_kl_annealing : bool
        Whether to anneal the KL weight from 0 to 1 during training.
    kl_annealing_n_epochs : int
        Number of epochs over which to linearly anneal KL weight.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Learning rate for Adam.
    gradient_clip : float
        Global norm for gradient clipping.
    n_epochs : int
        Number of training epochs.
    """

    n_modes: int = 8
    n_channels: int = 80
    sequence_length: int = 200

    # Inference network
    inference_n_units: int = 64
    inference_n_layers: int = 1
    inference_dropout: float = 0.0

    # Model network (prior)
    model_n_units: int = 64
    model_n_layers: int = 1
    model_dropout: float = 0.0

    # Observation model
    learn_means: bool = True
    learn_covariances: bool = True
    initial_means: Optional[jnp.ndarray] = None
    initial_covariances: Optional[jnp.ndarray] = None
    diagonal_covariances: bool = False
    covariances_epsilon: float = 1e-6

    # Alpha / softmax
    alpha_temperature: float = 1.0
    learn_alpha_temperature: bool = False

    # KL / VAE
    theta_std_epsilon: float = 1e-6
    do_kl_annealing: bool = True
    kl_annealing_n_epochs: int = 10

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    gradient_clip: float = 0.5
    n_epochs: int = 30


# ---------------------------------------------------------------------------
# Equinox modules
# ---------------------------------------------------------------------------


class GRUStack(eqx.Module):
    """Stack of GRU layers (unidirectional)."""

    cells: list  # list of eqx.nn.GRUCell
    n_units: int = eqx.field(static=True)

    def __init__(self, input_size: int, hidden_size: int, n_layers: int, *, key):
        self.n_units = hidden_size
        keys = jr.split(key, n_layers)
        self.cells = []
        in_sz = input_size
        for i in range(n_layers):
            self.cells.append(eqx.nn.GRUCell(in_sz, hidden_size, key=keys[i]))
            in_sz = hidden_size

    def __call__(self, xs: jax.Array, key=None) -> jax.Array:
        """Forward pass over a sequence.

        Parameters
        ----------
        xs : (T, input_size)

        Returns
        -------
        outputs : (T, hidden_size)
        """
        T = xs.shape[0]

        for cell in self.cells:
            h = jnp.zeros(self.n_units)

            def step(h, x):
                h = cell(x, h)
                return h, h

            _, outputs = jax.lax.scan(step, h, xs)
            xs = outputs  # feed to next layer

        return xs


class BiGRUStack(eqx.Module):
    """Stack of bidirectional GRU layers."""

    fwd_cells: list
    bwd_cells: list
    n_units: int = eqx.field(static=True)

    def __init__(self, input_size: int, hidden_size: int, n_layers: int, *, key):
        self.n_units = hidden_size
        k1, k2 = jr.split(key)
        keys_f = jr.split(k1, n_layers)
        keys_b = jr.split(k2, n_layers)
        self.fwd_cells = []
        self.bwd_cells = []
        in_sz = input_size
        for i in range(n_layers):
            self.fwd_cells.append(eqx.nn.GRUCell(in_sz, hidden_size, key=keys_f[i]))
            self.bwd_cells.append(eqx.nn.GRUCell(in_sz, hidden_size, key=keys_b[i]))
            in_sz = hidden_size * 2  # concat of fwd + bwd

    def __call__(self, xs: jax.Array, key=None) -> jax.Array:
        """Forward pass.

        Parameters
        ----------
        xs : (T, input_size)

        Returns
        -------
        outputs : (T, 2 * hidden_size)
        """
        for fwd_cell, bwd_cell in zip(self.fwd_cells, self.bwd_cells):
            h_f = jnp.zeros(self.n_units)
            h_b = jnp.zeros(self.n_units)

            def step_fwd(h, x):
                h = fwd_cell(x, h)
                return h, h

            def step_bwd(h, x):
                h = bwd_cell(x, h)
                return h, h

            _, out_f = jax.lax.scan(step_fwd, h_f, xs)
            _, out_b = jax.lax.scan(step_bwd, h_b, xs[::-1])
            out_b = out_b[::-1]

            xs = jnp.concatenate([out_f, out_b], axis=-1)

        return xs


class InferenceNetwork(eqx.Module):
    """Inference RNN: data -> (mu_theta, sigma_theta).

    Uses a bidirectional GRU (matching osl-dynamics InferenceRNNLayer)
    followed by two Dense heads for the mean and (softplus) std of theta.
    """

    rnn: BiGRUStack
    mu_head: eqx.nn.Linear
    sigma_head: eqx.nn.Linear

    def __init__(self, n_channels: int, n_units: int, n_layers: int,
                 n_modes: int, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.rnn = BiGRUStack(n_channels, n_units, n_layers, key=k1)
        rnn_out_size = n_units * 2  # bidirectional
        self.mu_head = eqx.nn.Linear(rnn_out_size, n_modes, key=k2)
        self.sigma_head = eqx.nn.Linear(rnn_out_size, n_modes, key=k3)

    def __call__(self, x: jax.Array, key=None) -> Tuple[jax.Array, jax.Array]:
        """
        Parameters
        ----------
        x : (T, C) observation sequence

        Returns
        -------
        mu : (T, K)
        sigma : (T, K) — positive (softplus applied)
        """
        h = self.rnn(x)  # (T, 2*units)
        mu = jax.vmap(self.mu_head)(h)  # (T, K)
        sigma = jax.nn.softplus(jax.vmap(self.sigma_head)(h))  # (T, K)
        return mu, sigma


class ModelNetwork(eqx.Module):
    """Temporal prior RNN: theta -> (mod_mu, mod_sigma).

    Unidirectional GRU (the prior sees only past theta).
    """

    rnn: GRUStack
    mu_head: eqx.nn.Linear
    sigma_head: eqx.nn.Linear

    def __init__(self, n_modes: int, n_units: int, n_layers: int, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.rnn = GRUStack(n_modes, n_units, n_layers, key=k1)
        self.mu_head = eqx.nn.Linear(n_units, n_modes, key=k2)
        self.sigma_head = eqx.nn.Linear(n_units, n_modes, key=k3)

    def __call__(self, theta: jax.Array, key=None) -> Tuple[jax.Array, jax.Array]:
        """
        Parameters
        ----------
        theta : (T, K) sampled logits

        Returns
        -------
        mod_mu : (T, K)
        mod_sigma : (T, K) — positive
        """
        h = self.rnn(theta)  # (T, units)
        mod_mu = jax.vmap(self.mu_head)(h)  # (T, K)
        mod_sigma = jax.nn.softplus(jax.vmap(self.sigma_head)(h))  # (T, K)
        return mod_mu, mod_sigma


class ObservationModel(eqx.Module):
    """Learnable mode means and covariances.

    Covariances are parameterised via their Cholesky factor (lower-triangular
    with positive diagonal via softplus) so that they remain positive-definite.
    """

    means: jax.Array  # (K, C)
    _chol_flat: jax.Array  # (K, C*(C+1)//2) for full, or (K, C) for diagonal
    n_modes: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    diagonal: bool = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(self, n_modes: int, n_channels: int, diagonal: bool,
                 epsilon: float, initial_means=None, initial_covs=None,
                 *, key):
        self.n_modes = n_modes
        self.n_channels = n_channels
        self.diagonal = diagonal
        self.epsilon = epsilon

        k1, k2 = jr.split(key)

        # Means
        if initial_means is not None:
            self.means = jnp.array(initial_means, dtype=jnp.float32)
        else:
            self.means = jr.normal(k1, (n_modes, n_channels)) * 0.01

        # Covariance parameterisation
        if diagonal:
            if initial_covs is not None:
                # initial_covs could be (K, C) or (K, C, C) — take diagonal
                ic = jnp.array(initial_covs, dtype=jnp.float32)
                if ic.ndim == 3:
                    ic = jnp.diagonal(ic, axis1=-2, axis2=-1)
                # Store as log(diag) so exp gives positive values
                self._chol_flat = jnp.log(jnp.clip(ic, 1e-6, None))
            else:
                self._chol_flat = jnp.zeros((n_modes, n_channels))
        else:
            n_tril = n_channels * (n_channels + 1) // 2
            if initial_covs is not None:
                ic = jnp.array(initial_covs, dtype=jnp.float32)
                # Cholesky decomposition of initial covariances
                L = jnp.linalg.cholesky(ic + epsilon * jnp.eye(n_channels))
                # Pack lower triangle into flat vector
                flat = jax.vmap(lambda l: _tril_to_flat(l, n_channels))(L)
                self._chol_flat = flat
            else:
                # Initialise as identity (flat representation)
                I = jnp.eye(n_channels)
                flat_I = _tril_to_flat(I, n_channels)
                self._chol_flat = jnp.tile(flat_I[None, :], (n_modes, 1))

    def get_covariances(self) -> jax.Array:
        """Return (K, C, C) covariance matrices."""
        if self.diagonal:
            diag = jnp.exp(self._chol_flat) + self.epsilon  # (K, C)
            return jax.vmap(jnp.diag)(diag)
        else:
            def _to_cov(flat):
                L = _flat_to_tril(flat, self.n_channels)
                return L @ L.T + self.epsilon * jnp.eye(self.n_channels)
            return jax.vmap(_to_cov)(self._chol_flat)

    def get_cholesky(self) -> jax.Array:
        """Return (K, C, C) lower Cholesky factors."""
        if self.diagonal:
            diag = jnp.sqrt(jnp.exp(self._chol_flat) + self.epsilon)
            return jax.vmap(jnp.diag)(diag)
        else:
            def _to_L(flat):
                L = _flat_to_tril(flat, self.n_channels)
                # Add epsilon to diagonal for stability
                return L + self.epsilon * jnp.eye(self.n_channels)
            return jax.vmap(_to_L)(self._chol_flat)


class DyNeMoModule(eqx.Module):
    """Full DyNeMo model as an equinox Module (holds all trainable params)."""

    inference_net: InferenceNetwork
    model_net: ModelNetwork
    obs_model: ObservationModel
    log_temperature: jax.Array  # scalar, log(alpha_temperature)
    config: DyNeMoConfig = eqx.field(static=True)

    def __init__(self, config: DyNeMoConfig, *, key):
        k1, k2, k3 = jr.split(key, 3)

        self.config = config
        self.inference_net = InferenceNetwork(
            config.n_channels, config.inference_n_units,
            config.inference_n_layers, config.n_modes, key=k1,
        )
        self.model_net = ModelNetwork(
            config.n_modes, config.model_n_units,
            config.model_n_layers, key=k2,
        )
        self.obs_model = ObservationModel(
            config.n_modes, config.n_channels, config.diagonal_covariances,
            config.covariances_epsilon, config.initial_means,
            config.initial_covariances, key=k3,
        )
        self.log_temperature = jnp.log(jnp.float32(config.alpha_temperature))


# ---------------------------------------------------------------------------
# Helper functions for Cholesky parameterisation
# ---------------------------------------------------------------------------


def _tril_indices(n: int):
    """Return (row_idx, col_idx) for lower triangle of n x n matrix."""
    rows, cols = jnp.tril_indices(n)
    return rows, cols


def _tril_to_flat(L: jax.Array, n: int) -> jax.Array:
    """Pack lower-triangular matrix into flat vector, softplus-encoding diagonal."""
    rows, cols = _tril_indices(n)
    flat = L[rows, cols]
    # Encode the diagonal elements via inverse-softplus so that softplus
    # recovers the original positive diagonal
    diag_mask = (rows == cols)
    diag_vals = jnp.where(
        diag_mask,
        _inverse_softplus(jnp.clip(flat, 1e-6, None)),
        flat,
    )
    return diag_vals


def _flat_to_tril(flat: jax.Array, n: int) -> jax.Array:
    """Unpack flat vector into lower-triangular matrix with positive diagonal."""
    rows, cols = _tril_indices(n)
    diag_mask = (rows == cols)
    # Apply softplus to diagonal entries to ensure positivity
    vals = jnp.where(diag_mask, jax.nn.softplus(flat), flat)
    L = jnp.zeros((n, n))
    L = L.at[rows, cols].set(vals)
    return L


def _inverse_softplus(x):
    """Inverse of softplus: log(exp(x) - 1)."""
    return jnp.log(jnp.exp(x) - 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


def _compute_loss(
    module: DyNeMoModule,
    data: jax.Array,
    key: jax.Array,
    kl_weight: float = 1.0,
    training: bool = True,
) -> Tuple[jax.Array, dict]:
    """Compute -ELBO loss for a single sequence.

    Parameters
    ----------
    module : DyNeMoModule
    data : (T, C) observation sequence
    key : PRNG key
    kl_weight : float
        Weight on KL term (for annealing).
    training : bool
        If True, sample theta; if False, use mean.

    Returns
    -------
    loss : scalar
    info : dict with nll_loss, kl_loss, alpha
    """
    config = module.config
    T, C = data.shape
    K = config.n_modes
    eps = config.theta_std_epsilon

    # --- Inference network ---
    inf_mu, inf_sigma = module.inference_net(data)  # (T, K), (T, K)
    inf_sigma = inf_sigma + eps

    # Sample theta (reparameterisation trick)
    if training:
        theta = inf_mu + inf_sigma * jr.normal(key, inf_mu.shape)
    else:
        theta = inf_mu

    # --- Alpha via softmax ---
    temperature = jnp.exp(module.log_temperature)
    alpha = jax.nn.softmax(theta / temperature, axis=-1)  # (T, K)

    # --- Observation model ---
    means = module.obs_model.means  # (K, C)
    L_modes = module.obs_model.get_cholesky()  # (K, C, C)

    # Mix means: m(t) = sum_k alpha_k(t) * mu_k
    m = jnp.einsum("tk,kc->tc", alpha, means)  # (T, C)

    # Mix covariance Cholesky factors:
    # We mix the full covariances, then take Cholesky for the log-prob
    covs = module.obs_model.get_covariances()  # (K, C, C)
    C_mixed = jnp.einsum("tk,kij->tij", alpha, covs)  # (T, C, C)

    # Add epsilon to diagonal for numerical stability
    C_mixed = C_mixed + config.covariances_epsilon * jnp.eye(C)[None, :, :]

    # Log-likelihood: x(t) ~ N(m(t), C_mixed(t))
    L_mixed = jnp.linalg.cholesky(C_mixed)  # (T, C, C)
    # log p(x|m,C) = -0.5 * [C*log(2pi) + log|C| + (x-m)^T C^{-1} (x-m)]
    diff = data - m  # (T, C)
    # Solve L * v = diff  =>  v = L^{-1} diff
    # Then maha = ||v||^2
    v = jax.scipy.linalg.solve_triangular(L_mixed, diff[:, :, None], lower=True)
    maha = jnp.sum(v[:, :, 0] ** 2, axis=-1)  # (T,)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L_mixed, axis1=-2, axis2=-1)),
                              axis=-1)  # (T,)
    nll = 0.5 * (C * jnp.log(2 * jnp.pi) + log_det + maha)  # (T,)
    nll_loss = jnp.mean(nll)

    # --- KL divergence ---
    # Model network predicts p(theta_t | theta_{<t})
    mod_mu, mod_sigma = module.model_net(theta)  # (T, K), (T, K)
    mod_sigma = mod_sigma + eps

    # The model RNN output at time t predicts theta_{t+1}
    # So we compare: q(theta_{t+1}) vs p(theta_{t+1} | theta_{<=t})
    # Clip: model uses [0:-1], inference uses [1:]
    q_mu = inf_mu[1:]  # (T-1, K)
    q_sigma = inf_sigma[1:]
    p_mu = mod_mu[:-1]  # (T-1, K)
    p_sigma = mod_sigma[:-1]

    # KL[N(q_mu, q_sigma^2) || N(p_mu, p_sigma^2)] per dimension
    kl = (
        jnp.log(p_sigma / q_sigma)
        + (q_sigma ** 2 + (q_mu - p_mu) ** 2) / (2 * p_sigma ** 2)
        - 0.5
    )
    # Sum over modes, mean over time
    kl_loss = jnp.mean(jnp.sum(kl, axis=-1))

    loss = nll_loss + kl_weight * kl_loss

    info = {
        "nll_loss": nll_loss,
        "kl_loss": kl_loss,
        "loss": loss,
        "alpha": alpha,
    }
    return loss, info


# Batched loss: vmap over a batch of sequences
@partial(jax.jit, static_argnums=(4,))
def _batch_loss(
    module: DyNeMoModule,
    batch: jax.Array,
    key: jax.Array,
    kl_weight: float,
    training: bool = True,
) -> Tuple[jax.Array, dict]:
    """Mean loss over a batch of sequences.

    Parameters
    ----------
    module : DyNeMoModule
    batch : (B, T, C)
    key : PRNG key
    kl_weight : float

    Returns
    -------
    loss : scalar
    info : dict
    """
    B = batch.shape[0]
    keys = jr.split(key, B)

    def single(data_seq, k):
        return _compute_loss(module, data_seq, k, kl_weight, training)

    losses, infos = jax.vmap(single)(batch, keys)
    loss = jnp.mean(losses)
    info = jax.tree.map(lambda x: jnp.mean(x, axis=0) if x.ndim > 0 else x, infos)
    # Keep alpha from first sequence for diagnostics
    info["alpha"] = infos["alpha"][0]
    return loss, info


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


@eqx.filter_jit
def _train_step(
    module: DyNeMoModule,
    opt_state,
    optimizer,
    batch: jax.Array,
    key: jax.Array,
    kl_weight: float,
):
    """Single gradient descent step.

    Returns
    -------
    module : updated module
    opt_state : updated optimizer state
    loss : scalar
    info : dict
    """
    # Determine which parameters are trainable
    config = module.config

    def loss_fn(m):
        return _batch_loss(m, batch, key, kl_weight, True)

    (loss, info), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(module)

    # Zero out gradients for non-trainable parameters
    if not config.learn_means:
        grads = eqx.tree_at(
            lambda m: m.obs_model.means, grads,
            jnp.zeros_like(module.obs_model.means),
        )
    if not config.learn_covariances:
        grads = eqx.tree_at(
            lambda m: m.obs_model._chol_flat, grads,
            jnp.zeros_like(module.obs_model._chol_flat),
        )
    if not config.learn_alpha_temperature:
        grads = eqx.tree_at(
            lambda m: m.log_temperature, grads,
            jnp.zeros_like(module.log_temperature),
        )

    updates, opt_state = optimizer.update(grads, opt_state, module)
    module = eqx.apply_updates(module, updates)

    return module, opt_state, loss, info


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def _segment_data(
    data: List[jax.Array], sequence_length: int, *, key: jax.Array
) -> jax.Array:
    """Cut data into overlapping segments of fixed length.

    Parameters
    ----------
    data : list of (T_i, C) arrays
    sequence_length : int

    Returns
    -------
    segments : (N, sequence_length, C) array
    """
    segments = []
    for x in data:
        T = x.shape[0]
        if T < sequence_length:
            # Pad short sequences
            pad = jnp.zeros((sequence_length - T, x.shape[1]))
            segments.append(jnp.concatenate([x, pad], axis=0)[None])
        else:
            # Non-overlapping segments (with random offset)
            n_segs = T // sequence_length
            trim = n_segs * sequence_length
            x_trim = x[:trim]
            segs = x_trim.reshape(n_segs, sequence_length, -1)
            segments.append(segs)

    return jnp.concatenate(segments, axis=0)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------


class DyNeMo:
    """DyNeMo model with an osl-dynamics-compatible API.

    Parameters
    ----------
    config : DyNeMoConfig
        Model configuration.
    key : jax random key, optional
        Seed for parameter initialisation.

    Example
    -------
    >>> config = DyNeMoConfig(n_modes=8, n_channels=80, n_epochs=30)
    >>> model = DyNeMo(config)
    >>> history = model.fit(prepared_data)
    >>> alpha = model.infer(prepared_data)
    """

    def __init__(
        self,
        config: Optional[DyNeMoConfig] = None,
        *,
        n_modes: int = 8,
        n_channels: int = 80,
        key: Optional[jax.Array] = None,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = DyNeMoConfig(n_modes=n_modes, n_channels=n_channels)

        if key is None:
            key = jr.PRNGKey(0)

        self._key = key
        self._module: Optional[DyNeMoModule] = None
        self.history: List[dict] = []

    def _build_module(self, key: jax.Array) -> DyNeMoModule:
        """Create the equinox module with current config."""
        return DyNeMoModule(self.config, key=key)

    def _init_from_data(
        self, data: List[jax.Array], key: jax.Array
    ) -> DyNeMoModule:
        """Initialise observation model parameters from data statistics."""
        k1, k2 = jr.split(key)
        module = self._build_module(k1)

        # Compute global statistics for initialisation
        X = jnp.concatenate(data, axis=0)
        global_mean = jnp.mean(X, axis=0)
        C = X.shape[1]
        K = self.config.n_modes

        if self.config.initial_means is None:
            # Initialise means around the global mean with small perturbations
            offsets = jr.normal(k2, (K, C)) * 0.1
            new_means = global_mean[None, :] + offsets
            module = eqx.tree_at(lambda m: m.obs_model.means, module, new_means)

        if self.config.initial_covariances is None:
            # Use global covariance (regularised) for each mode
            n_samples = min(X.shape[0], 50000)  # cap to avoid OOM
            X_sub = X[:n_samples]
            global_cov = jnp.cov(X_sub.T) + 1e-4 * jnp.eye(C)

            if self.config.diagonal_covariances:
                diag = jnp.diag(global_cov)
                new_flat = jnp.log(jnp.clip(diag, 1e-6, None))
                new_flat = jnp.tile(new_flat[None, :], (K, 1))
            else:
                L = jnp.linalg.cholesky(global_cov)
                flat = _tril_to_flat(L, C)
                new_flat = jnp.tile(flat[None, :], (K, 1))

            module = eqx.tree_at(
                lambda m: m.obs_model._chol_flat, module, new_flat
            )

        return module

    def fit(
        self,
        data: List[jax.Array],
        n_epochs: Optional[int] = None,
        key: Optional[jax.Array] = None,
    ) -> List[dict]:
        """Fit the model to data.

        Parameters
        ----------
        data : list of (T_i, C) arrays
            Prepared data (one array per session/run).
        n_epochs : int, optional
            Override config.n_epochs.
        key : jax random key, optional

        Returns
        -------
        history : list of dict with keys 'loss', 'nll_loss', 'kl_loss'
        """
        if isinstance(data, jnp.ndarray) and data.ndim == 2:
            data = [data]

        n_epochs = n_epochs or self.config.n_epochs
        if key is None:
            key = self._key

        k_init, k_train = jr.split(key)

        # Initialise module
        if self._module is None:
            self._module = self._init_from_data(data, k_init)
            logger.info(
                "DyNeMo initialised: n_modes=%d, n_channels=%d",
                self.config.n_modes, self.config.n_channels,
            )

        # Segment data into fixed-length sequences
        segments = _segment_data(data, self.config.sequence_length, key=k_init)
        N = segments.shape[0]
        B = min(self.config.batch_size, N)
        logger.info(
            "Training data: %d segments of length %d, batch_size=%d",
            N, self.config.sequence_length, B,
        )

        # Optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.gradient_clip),
            optax.adam(self.config.learning_rate),
        )
        opt_state = optimizer.init(eqx.filter(self._module, eqx.is_array))

        # Training loop
        self.history = []
        for epoch in range(n_epochs):
            k_epoch, k_train = jr.split(k_train)

            # KL annealing
            if self.config.do_kl_annealing and self.config.kl_annealing_n_epochs > 0:
                kl_weight = min(1.0, epoch / self.config.kl_annealing_n_epochs)
            else:
                kl_weight = 1.0

            # Shuffle and iterate batches
            perm = jr.permutation(k_epoch, N)
            epoch_losses = []
            epoch_nll = []
            epoch_kl = []
            n_batches = max(1, N // B)

            for b in range(n_batches):
                k_batch, k_train = jr.split(k_train)
                idx = perm[b * B: (b + 1) * B]
                batch = segments[idx]

                self._module, opt_state, loss, info = _train_step(
                    self._module, opt_state, optimizer,
                    batch, k_batch, kl_weight,
                )
                epoch_losses.append(float(loss))
                epoch_nll.append(float(info["nll_loss"]))
                epoch_kl.append(float(info["kl_loss"]))

            epoch_info = {
                "loss": sum(epoch_losses) / len(epoch_losses),
                "nll_loss": sum(epoch_nll) / len(epoch_nll),
                "kl_loss": sum(epoch_kl) / len(epoch_kl),
                "kl_weight": kl_weight,
            }
            self.history.append(epoch_info)
            logger.info(
                "Epoch %d/%d  loss=%.4f  nll=%.4f  kl=%.4f  kl_w=%.3f",
                epoch + 1, n_epochs,
                epoch_info["loss"], epoch_info["nll_loss"],
                epoch_info["kl_loss"], epoch_info["kl_weight"],
            )

        return self.history

    def infer(self, data: List[jax.Array]) -> List[jax.Array]:
        """Infer mode mixing coefficients (alpha) for data.

        Parameters
        ----------
        data : list of (T_i, C) arrays

        Returns
        -------
        alphas : list of (T_i, K) arrays — mode mixing coefficients
        """
        if self._module is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if isinstance(data, jnp.ndarray) and data.ndim == 2:
            data = [data]

        module = self._module
        config = module.config

        alphas = []
        for x in data:
            # Run inference network (no sampling — use mean)
            inf_mu, _ = module.inference_net(x)
            temperature = jnp.exp(module.log_temperature)
            alpha = jax.nn.softmax(inf_mu / temperature, axis=-1)
            alphas.append(alpha)

        return alphas

    def get_alpha(self, data: List[jax.Array]) -> List[jax.Array]:
        """Alias for ``infer()``."""
        return self.infer(data)

    def get_means(self) -> jax.Array:
        """Return learned mode means (K, C)."""
        if self._module is None:
            raise RuntimeError("Model not fitted.")
        return self._module.obs_model.means

    def get_covariances(self) -> jax.Array:
        """Return learned mode covariances (K, C, C)."""
        if self._module is None:
            raise RuntimeError("Model not fitted.")
        return self._module.obs_model.get_covariances()

    def get_means_covariances(self) -> Tuple[jax.Array, jax.Array]:
        """Return (means, covariances) tuple."""
        return self.get_means(), self.get_covariances()

    def __repr__(self) -> str:
        return (
            f"DyNeMo(n_modes={self.config.n_modes}, "
            f"n_channels={self.config.n_channels})"
        )
