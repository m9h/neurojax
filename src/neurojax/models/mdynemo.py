"""Multi-dynamic Network Modes (M-DyNeMo) — pure JAX, osl-dynamics-compatible.

M-DyNeMo (Huang, Gohil & Woolrich, *Human Brain Mapping* 2025) extends DyNeMo by
giving **power** and **functional connectivity** their own, independent mode time
courses.  Where DyNeMo has one ``alpha`` mixing a single set of covariances,
M-DyNeMo has two::

    alpha (n_modes)       inference+prior RNNs ->  mix means + stds  -> m_t, D_t
    gamma (n_corr_modes)  inference+prior RNNs ->  mix correlations  -> C_t
    Sigma_t = D_t @ C_t @ D_t,        x_t ~ N(m_t, Sigma_t)

D_t is the diagonal matrix of mixed standard deviations and C_t the mixed
correlation matrix, so amplitude (power) and connectivity (correlation) dynamics
are decoupled.

This module reuses DyNeMo's inference/prior RNNs and Cholesky helpers; only the
observation model and the dual-time-course loss are new.

Example
-------
>>> from neurojax.models.mdynemo import MDyNeMo, MDyNeMoConfig
>>> model = MDyNeMo(MDyNeMoConfig(n_modes=6, n_corr_modes=6, n_channels=80))
>>> model.fit(prepared_data, n_epochs=30)
>>> alpha, gamma = model.infer(prepared_data)[0]   # power / FC time courses
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from neurojax.models.dynemo import (
    InferenceNetwork,
    ModelNetwork,
    _flat_to_tril,
    _inverse_softplus,
    _segment_data,
    _tril_to_flat,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MDyNeMoConfig:
    """Configuration for :class:`MDyNeMo`.

    ``n_corr_modes`` defaults to ``n_modes`` when left as None.  ``learn_means``,
    ``learn_stds`` and ``learn_corrs`` toggle the three observation components
    independently (M-DyNeMo's whole point is decoupling power from FC).
    """

    n_modes: int = 6
    n_channels: int = 80
    n_corr_modes: Optional[int] = None
    sequence_length: int = 200

    # Inference network (shared architecture, separate weights per time course)
    inference_n_units: int = 64
    inference_n_layers: int = 1
    inference_dropout: float = 0.0

    # Model (prior) network
    model_n_units: int = 64
    model_n_layers: int = 1
    model_dropout: float = 0.0

    # Observation model
    learn_means: bool = True
    learn_stds: bool = True
    learn_corrs: bool = True
    initial_means: Optional[jnp.ndarray] = None
    initial_stds: Optional[jnp.ndarray] = None
    initial_corrs: Optional[jnp.ndarray] = None
    stds_epsilon: float = 1e-6
    corrs_epsilon: float = 1e-6

    # Softmax temperatures
    alpha_temperature: float = 1.0
    learn_alpha_temperature: bool = False
    theta_std_epsilon: float = 1e-6

    # Training
    do_kl_annealing: bool = True
    kl_annealing_n_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.01
    gradient_clip: float = 5.0
    n_epochs: int = 30

    def __post_init__(self):
        if self.n_corr_modes is None:
            self.n_corr_modes = self.n_modes


# ---------------------------------------------------------------------------
# Observation model: means, stds (power) + correlation matrices (FC)
# ---------------------------------------------------------------------------


class MDyNeMoObservation(eqx.Module):
    """Mode means + standard deviations (power) and correlation matrices (FC).

    Correlation matrices are parameterised by a Cholesky factor (positive
    diagonal via softplus); the resulting PSD matrix is normalised to unit
    diagonal, guaranteeing a valid correlation matrix.
    """

    means: jax.Array            # (Km, C)
    _stds_raw: jax.Array        # (Km, C)  -> softplus -> stds
    _corr_flat: jax.Array       # (Kc, C*(C+1)//2)
    n_modes: int = eqx.field(static=True)
    n_corr_modes: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    stds_epsilon: float = eqx.field(static=True)
    corrs_epsilon: float = eqx.field(static=True)

    def __init__(self, config: MDyNeMoConfig, *, key):
        Km, Kc, C = config.n_modes, config.n_corr_modes, config.n_channels
        self.n_modes, self.n_corr_modes, self.n_channels = Km, Kc, C
        self.stds_epsilon = config.stds_epsilon
        self.corrs_epsilon = config.corrs_epsilon
        k1, k2 = jr.split(key)

        if config.initial_means is not None:
            self.means = jnp.asarray(config.initial_means, dtype=jnp.float32)
        else:
            self.means = jr.normal(k1, (Km, C)) * 0.01

        if config.initial_stds is not None:
            s = jnp.asarray(config.initial_stds, dtype=jnp.float32)
            if s.ndim == 3:  # (Km, C, C) -> take diagonal
                s = jnp.sqrt(jnp.diagonal(s, axis1=-2, axis2=-1))
            self._stds_raw = _inverse_softplus(jnp.clip(s, 1e-4, None))
        else:
            # softplus(0) ~ 0.69 -> reasonable starting std
            self._stds_raw = jnp.zeros((Km, C))

        # Correlation Cholesky flats: initialise to identity correlation.
        flat_I = _tril_to_flat(jnp.eye(C), C)
        self._corr_flat = jnp.tile(flat_I[None, :], (Kc, 1))

    def get_means(self) -> jax.Array:
        return self.means

    def get_stds(self) -> jax.Array:
        """(Km, C) positive standard deviations."""
        return jax.nn.softplus(self._stds_raw) + self.stds_epsilon

    def _corr_from_flat(self, flat: jax.Array) -> jax.Array:
        L = _flat_to_tril(flat, self.n_channels)
        M = L @ L.T + self.corrs_epsilon * jnp.eye(self.n_channels)
        d = jnp.sqrt(jnp.diagonal(M))
        return M / (d[:, None] * d[None, :])

    def get_corrs(self) -> jax.Array:
        """(Kc, C, C) correlation matrices (unit diagonal, PSD)."""
        return jax.vmap(self._corr_from_flat)(self._corr_flat)


class MDyNeMoModule(eqx.Module):
    """Full M-DyNeMo as an equinox module (all trainable params)."""

    inf_alpha: InferenceNetwork
    inf_gamma: InferenceNetwork
    mod_alpha: ModelNetwork
    mod_gamma: ModelNetwork
    obs: MDyNeMoObservation
    log_temp_alpha: jax.Array
    log_temp_gamma: jax.Array
    config: MDyNeMoConfig = eqx.field(static=True)

    def __init__(self, config: MDyNeMoConfig, *, key):
        ka, kg, ma, mg, ko = jr.split(key, 5)
        self.config = config
        iu, il = config.inference_n_units, config.inference_n_layers
        mu, ml = config.model_n_units, config.model_n_layers
        self.inf_alpha = InferenceNetwork(config.n_channels, iu, il, config.n_modes, key=ka)
        self.inf_gamma = InferenceNetwork(config.n_channels, iu, il, config.n_corr_modes, key=kg)
        self.mod_alpha = ModelNetwork(config.n_modes, mu, ml, key=ma)
        self.mod_gamma = ModelNetwork(config.n_corr_modes, mu, ml, key=mg)
        self.obs = MDyNeMoObservation(config, key=ko)
        self.log_temp_alpha = jnp.log(jnp.float32(config.alpha_temperature))
        self.log_temp_gamma = jnp.log(jnp.float32(config.alpha_temperature))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _diag_gaussian_kl(q_mu, q_sigma, p_mu, p_sigma):
    """Mean-over-time, sum-over-modes KL[N(q)||N(p)] for diagonal Gaussians."""
    kl = (
        jnp.log(p_sigma / q_sigma)
        + (q_sigma ** 2 + (q_mu - p_mu) ** 2) / (2 * p_sigma ** 2)
        - 0.5
    )
    return jnp.mean(jnp.sum(kl, axis=-1))


def _time_course(inf_net, mod_net, data, log_temp, eps, key, training):
    """Return (mixing time course, KL) for one inference/prior path."""
    inf_mu, inf_sigma = inf_net(data)
    inf_sigma = inf_sigma + eps
    if training:
        theta = inf_mu + inf_sigma * jr.normal(key, inf_mu.shape)
    else:
        theta = inf_mu
    mix = jax.nn.softmax(theta / jnp.exp(log_temp), axis=-1)
    mod_mu, mod_sigma = mod_net(theta)
    mod_sigma = mod_sigma + eps
    kl = _diag_gaussian_kl(inf_mu[1:], inf_sigma[1:], mod_mu[:-1], mod_sigma[:-1])
    return mix, kl


def _build_sigma(alpha, gamma, means, stds, corrs, eps):
    """m_t = alpha.means; Sigma_t = D_t C_t D_t with D_t = diag(alpha.stds)."""
    m = alpha @ means                               # (T, C)
    std_t = alpha @ stds                            # (T, C)  (>0)
    C_t = jnp.einsum("tk,kij->tij", gamma, corrs)   # (T, C, C)
    sigma = std_t[:, :, None] * C_t * std_t[:, None, :]
    C = means.shape[1]
    sigma = sigma + eps * jnp.eye(C)[None]
    return m, sigma


def _compute_loss(module: MDyNeMoModule, data, key, kl_weight=1.0, training=True):
    cfg = module.config
    T, C = data.shape
    eps = cfg.theta_std_epsilon
    ka, kg = jr.split(key)

    alpha, kl_a = _time_course(
        module.inf_alpha, module.mod_alpha, data, module.log_temp_alpha, eps, ka, training
    )
    gamma, kl_g = _time_course(
        module.inf_gamma, module.mod_gamma, data, module.log_temp_gamma, eps, kg, training
    )

    m, sigma = _build_sigma(
        alpha, gamma,
        module.obs.get_means(), module.obs.get_stds(), module.obs.get_corrs(),
        cfg.corrs_epsilon,
    )

    L = jnp.linalg.cholesky(sigma)                  # (T, C, C)
    diff = data - m
    v = jax.scipy.linalg.solve_triangular(L, diff[:, :, None], lower=True)
    maha = jnp.sum(v[:, :, 0] ** 2, axis=-1)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
    nll_loss = jnp.mean(0.5 * (C * jnp.log(2 * jnp.pi) + log_det + maha))

    kl_loss = kl_a + kl_g
    loss = nll_loss + kl_weight * kl_loss
    info = {"nll_loss": nll_loss, "kl_loss": kl_loss, "loss": loss}
    return loss, info


@partial(jax.jit, static_argnums=(4,))
def _batch_loss(module, batch, key, kl_weight, training=True):
    keys = jr.split(key, batch.shape[0])
    losses, infos = jax.vmap(lambda d, k: _compute_loss(module, d, k, kl_weight, training))(
        batch, keys
    )
    info = jax.tree.map(lambda x: jnp.mean(x), infos)
    return jnp.mean(losses), info


@eqx.filter_jit
def _train_step(module, opt_state, optimizer, batch, key, kl_weight):
    cfg = module.config

    def loss_fn(m):
        return _batch_loss(m, batch, key, kl_weight, True)

    (loss, info), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(module)

    def zero(grads, where):
        leaf = where(module)
        return eqx.tree_at(where, grads, jnp.zeros_like(leaf))

    if not cfg.learn_means:
        grads = zero(grads, lambda m: m.obs.means)
    if not cfg.learn_stds:
        grads = zero(grads, lambda m: m.obs._stds_raw)
    if not cfg.learn_corrs:
        grads = zero(grads, lambda m: m.obs._corr_flat)
    if not cfg.learn_alpha_temperature:
        grads = zero(grads, lambda m: m.log_temp_alpha)
        grads = zero(grads, lambda m: m.log_temp_gamma)

    updates, opt_state = optimizer.update(grads, opt_state, module)
    module = eqx.apply_updates(module, updates)
    return module, opt_state, loss, info


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------


class MDyNeMo:
    """M-DyNeMo with an osl-dynamics-flavoured API."""

    def __init__(
        self,
        config: Optional[MDyNeMoConfig] = None,
        *,
        n_modes: int = 6,
        n_corr_modes: Optional[int] = None,
        n_channels: int = 80,
        key: Optional[jax.Array] = None,
    ):
        self.config = config or MDyNeMoConfig(
            n_modes=n_modes, n_corr_modes=n_corr_modes, n_channels=n_channels
        )
        self._key = key if key is not None else jr.PRNGKey(0)
        self._module: Optional[MDyNeMoModule] = self._build_module(self._key)
        self.history: List[dict] = []

    def _build_module(self, key):
        return MDyNeMoModule(self.config, key=key)

    def _init_from_data(self, data, key):
        module = self._build_module(key)
        X = jnp.concatenate(data, axis=0)
        Km = self.config.n_modes
        if self.config.initial_means is None:
            offsets = jr.normal(key, (Km, X.shape[1])) * 0.1
            module = eqx.tree_at(
                lambda m: m.obs.means, module, jnp.mean(X, axis=0)[None] + offsets
            )
        if self.config.initial_stds is None:
            gstd = jnp.std(X, axis=0).clip(1e-4)
            raw = _inverse_softplus(gstd)
            module = eqx.tree_at(
                lambda m: m.obs._stds_raw, module, jnp.tile(raw[None], (Km, 1))
            )
        return module

    def fit(self, data, n_epochs=None, key=None):
        if isinstance(data, jnp.ndarray) and data.ndim == 2:
            data = [data]
        n_epochs = n_epochs or self.config.n_epochs
        key = key or self._key
        k_init, k_train = jr.split(key)

        self._module = self._init_from_data(data, k_init)
        segments = _segment_data(data, self.config.sequence_length, key=k_init)
        N = segments.shape[0]
        B = min(self.config.batch_size, N)

        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.gradient_clip),
            optax.adam(self.config.learning_rate),
        )
        opt_state = optimizer.init(eqx.filter(self._module, eqx.is_array))

        self.history = []
        for epoch in range(n_epochs):
            k_epoch, k_train = jr.split(k_train)
            if self.config.do_kl_annealing and self.config.kl_annealing_n_epochs > 0:
                kl_weight = min(1.0, epoch / self.config.kl_annealing_n_epochs)
            else:
                kl_weight = 1.0

            perm = jr.permutation(k_epoch, N)
            losses, nlls, kls = [], [], []
            for b in range(max(1, N // B)):
                k_batch, k_train = jr.split(k_train)
                batch = segments[perm[b * B: (b + 1) * B]]
                self._module, opt_state, loss, info = _train_step(
                    self._module, opt_state, optimizer, batch, k_batch, kl_weight
                )
                losses.append(float(loss))
                nlls.append(float(info["nll_loss"]))
                kls.append(float(info["kl_loss"]))

            self.history.append(
                {
                    "loss": sum(losses) / len(losses),
                    "nll_loss": sum(nlls) / len(nlls),
                    "kl_loss": sum(kls) / len(kls),
                    "kl_weight": kl_weight,
                }
            )
            logger.info(
                "Epoch %d/%d loss=%.4f nll=%.4f kl=%.4f",
                epoch + 1, n_epochs, self.history[-1]["loss"],
                self.history[-1]["nll_loss"], self.history[-1]["kl_loss"],
            )
        return self.history

    def _alpha_gamma(self, x):
        mod = self._module
        a_mu, _ = mod.inf_alpha(x)
        g_mu, _ = mod.inf_gamma(x)
        alpha = jax.nn.softmax(a_mu / jnp.exp(mod.log_temp_alpha), axis=-1)
        gamma = jax.nn.softmax(g_mu / jnp.exp(mod.log_temp_gamma), axis=-1)
        return alpha, gamma

    def infer(self, data):
        """Return list of (alpha, gamma) — power and FC mode time courses."""
        if isinstance(data, jnp.ndarray) and data.ndim == 2:
            data = [data]
        return [self._alpha_gamma(x) for x in data]

    def get_mode_time_courses(self, data):
        """Return (list of alpha, list of gamma)."""
        pairs = self.infer(data)
        return [a for a, _ in pairs], [g for _, g in pairs]

    def get_means(self):
        return self._module.obs.get_means()

    def get_stds(self):
        return self._module.obs.get_stds()

    def get_corrs(self):
        return self._module.obs.get_corrs()

    def get_covariances(self, data_seq):
        """Reconstruct the time-varying covariance Sigma_t = D_t C_t D_t."""
        alpha, gamma = self._alpha_gamma(data_seq)
        _, sigma = _build_sigma(
            alpha, gamma, self.get_means(), self.get_stds(), self.get_corrs(),
            self.config.corrs_epsilon,
        )
        return sigma

    def __repr__(self):
        return (
            f"MDyNeMo(n_modes={self.config.n_modes}, "
            f"n_corr_modes={self.config.n_corr_modes}, "
            f"n_channels={self.config.n_channels})"
        )
