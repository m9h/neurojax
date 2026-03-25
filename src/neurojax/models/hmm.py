"""Gaussian Hidden Markov Model — pure JAX, osl-dynamics-compatible.

Provides a Baum-Welch EM implementation for fitting HMMs with multivariate
Gaussian emissions to parcellated MEG/EEG data.  Designed as a drop-in
replacement for ``osl_dynamics.models.hmm`` with the same data preparation
pipeline (TDE → PCA → HMM).

Example
-------
>>> from neurojax.models.hmm import GaussianHMM
>>> from neurojax.data import Data
>>> data = Data("recon_dir/")
>>> data.prepare({"tde_pca": {"n_embeddings": 15, "n_pca_components": 80}})
>>> model = GaussianHMM(n_states=8, n_channels=80)
>>> model.fit(data.prepared_data, n_epochs=20)
>>> alpha = model.infer(data.prepared_data)  # state probabilities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (mirrors osl-dynamics Config pattern)
# ---------------------------------------------------------------------------


@dataclass
class HMMConfig:
    """Configuration for :class:`GaussianHMM`.

    Parameters
    ----------
    n_states : int
        Number of hidden states.
    n_channels : int
        Dimensionality of the observation space (after TDE+PCA).
    learn_means : bool
        Whether to update state means during EM.
    learn_covariances : bool
        Whether to update state covariances during EM.
    learn_trans_prob : bool
        Whether to update the transition matrix during EM.
    covariance_type : ``"full"`` or ``"diag"``
        Parameterisation of the emission covariance.
    stay_prob : float or None
        If set, initialises the transition matrix with this self-transition
        probability and uniform off-diagonal mass.
    """

    n_states: int = 8
    n_channels: int = 80
    learn_means: bool = True
    learn_covariances: bool = True
    learn_trans_prob: bool = True
    covariance_type: str = "full"  # "full" or "diag"
    stay_prob: Optional[float] = None


# ---------------------------------------------------------------------------
# Core log-space algorithms
# ---------------------------------------------------------------------------


def _log_mvn_pdf(
    x: jax.Array,
    means: jax.Array,
    precisions: jax.Array,
    log_dets: jax.Array,
) -> jax.Array:
    """Log-density of multivariate normal for all states at one time step.

    Parameters
    ----------
    x : (C,)
    means : (S, C)
    precisions : (S, C, C)  — inverse covariance matrices
    log_dets : (S,)  — log determinant of each covariance

    Returns
    -------
    (S,)  log p(x | state)
    """
    C = x.shape[0]
    diff = x[None, :] - means  # (S, C)
    # Mahalanobis: diff^T Precision diff  per state
    maha = jnp.einsum("si,sij,sj->s", diff, precisions, diff)
    return -0.5 * (C * jnp.log(2 * jnp.pi) + log_dets + maha)


def _log_emission_matrix(
    data: jax.Array,
    means: jax.Array,
    precisions: jax.Array,
    log_dets: jax.Array,
) -> jax.Array:
    """Compute log B[t, s] = log p(x_t | z_t=s) for all t, s.

    Parameters
    ----------
    data : (T, C)
    means : (S, C)
    precisions : (S, C, C)
    log_dets : (S,)

    Returns
    -------
    (T, S)
    """
    return jax.vmap(lambda x: _log_mvn_pdf(x, means, precisions, log_dets))(
        data
    )


def forward(
    log_B: jax.Array,
    log_trans: jax.Array,
    log_pi: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Forward algorithm in log-space.

    Parameters
    ----------
    log_B : (T, S)  log emission probabilities
    log_trans : (S, S)  log transition matrix  [from, to]
    log_pi : (S,)  log initial state probs

    Returns
    -------
    log_alpha : (T, S)  forward log-probabilities
    log_ll : scalar  total log-likelihood
    """
    T, S = log_B.shape

    def scan_fn(log_alpha_prev, log_b_t):
        # log_alpha_t[j] = log_b_t[j] + logsumexp_i(log_alpha_prev[i] + log_trans[i,j])
        log_alpha_t = log_b_t + logsumexp(
            log_alpha_prev[:, None] + log_trans, axis=0
        )
        return log_alpha_t, log_alpha_t

    log_alpha_0 = log_pi + log_B[0]
    _, log_alpha_rest = jax.lax.scan(scan_fn, log_alpha_0, log_B[1:])
    log_alpha = jnp.concatenate([log_alpha_0[None], log_alpha_rest], axis=0)

    log_ll = logsumexp(log_alpha[-1])
    return log_alpha, log_ll


def backward(
    log_B: jax.Array,
    log_trans: jax.Array,
) -> jax.Array:
    """Backward algorithm in log-space.

    Parameters
    ----------
    log_B : (T, S)
    log_trans : (S, S)

    Returns
    -------
    log_beta : (T, S)
    """
    T, S = log_B.shape

    def scan_fn(log_beta_next, log_b_next):
        # log_beta_t[i] = logsumexp_j(log_trans[i,j] + log_b_next[j] + log_beta_next[j])
        log_beta_t = logsumexp(
            log_trans + log_b_next[None, :] + log_beta_next[None, :], axis=1
        )
        return log_beta_t, log_beta_t

    log_beta_T = jnp.zeros(S)
    _, log_beta_rev = jax.lax.scan(
        scan_fn, log_beta_T, log_B[1:][::-1]
    )
    log_beta = jnp.concatenate([log_beta_rev[::-1], log_beta_T[None]], axis=0)
    return log_beta


def e_step(
    log_B: jax.Array,
    log_trans: jax.Array,
    log_pi: jax.Array,
) -> Tuple[jax.Array, jax.Array, float]:
    """E-step: compute state posteriors gamma and pairwise xi.

    Returns
    -------
    gamma : (T, S)  p(z_t=s | x_{1:T})
    xi : (T-1, S, S)  p(z_t=i, z_{t+1}=j | x_{1:T})
    log_ll : scalar
    """
    log_alpha, log_ll = forward(log_B, log_trans, log_pi)
    log_beta = backward(log_B, log_trans)

    # gamma
    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = jnp.exp(log_gamma)

    # xi[t, i, j] = alpha[t,i] * trans[i,j] * B[t+1,j] * beta[t+1,j] / p(x)
    log_xi = (
        log_alpha[:-1, :, None]
        + log_trans[None, :, :]
        + log_B[1:, None, :]
        + log_beta[1:, None, :]
    )
    log_xi = log_xi - logsumexp(
        log_xi.reshape(log_xi.shape[0], -1), axis=1, keepdims=True
    ).reshape(-1, 1, 1)
    xi = jnp.exp(log_xi)

    return gamma, xi, log_ll


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class GaussianHMM:
    """Gaussian HMM with Baum-Welch EM, pure JAX.

    Mirrors the osl-dynamics ``Model`` API where it matters:
    ``fit(data)``, state probabilities via ``infer(data)``, and
    Viterbi decoding via ``decode(data)``.

    Parameters
    ----------
    config : HMMConfig or None
        Full configuration.  Alternatively, pass ``n_states`` and
        ``n_channels`` directly.
    n_states : int
        Number of hidden states (ignored if *config* is given).
    n_channels : int
        Observation dimensionality (ignored if *config* is given).
    """

    def __init__(
        self,
        config: Optional[HMMConfig] = None,
        *,
        n_states: int = 8,
        n_channels: int = 80,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = HMMConfig(n_states=n_states, n_channels=n_channels)

        S = self.config.n_states
        C = self.config.n_channels

        # Parameters — will be initialised in fit() or init_params()
        self.means: Optional[jax.Array] = None  # (S, C)
        self.covariances: Optional[jax.Array] = None  # (S, C, C)
        self.trans_prob: Optional[jax.Array] = None  # (S, S)
        self.init_prob: Optional[jax.Array] = None  # (S,)

        # Cached inverse / log-det for speed
        self._precisions: Optional[jax.Array] = None
        self._log_dets: Optional[jax.Array] = None

        # Standardisation stats (set during fit if standardize=True)
        self._data_mean: Optional[jax.Array] = None  # (C,)
        self._data_std: Optional[jax.Array] = None  # (C,)
        self._standardize: bool = False

        # Training history
        self.history: List[float] = []

    # -- initialisation ------------------------------------------------------

    def init_params(self, data: List[jax.Array], key: jax.Array | None = None) -> None:
        """Initialise parameters from data statistics + random perturbation."""
        if key is None:
            key = jr.PRNGKey(0)

        S = self.config.n_states
        C = self.config.n_channels

        # Concatenate all sessions for global stats
        X = jnp.concatenate(data, axis=0)  # (N, C)
        global_mean = jnp.mean(X, axis=0)
        global_cov = jnp.cov(X.T) + 1e-6 * jnp.eye(C)

        # Means: global mean + small random offset per state
        k1, k2, key = jr.split(key, 3)
        offsets = jr.normal(k1, (S, C)) * jnp.sqrt(jnp.diag(global_cov))[None, :] * 0.2
        self.means = global_mean[None, :] + offsets

        # Covariances: global covariance for each state (will diversify via EM)
        self.covariances = jnp.broadcast_to(global_cov, (S, C, C)).copy()

        # Transition matrix
        if self.config.stay_prob is not None:
            sp = self.config.stay_prob
            off = (1.0 - sp) / (S - 1)
            self.trans_prob = jnp.full((S, S), off)
            self.trans_prob = self.trans_prob.at[jnp.diag_indices(S)].set(sp)
        else:
            # Strong bias toward staying (matches osl-dynamics default)
            sp = 0.9
            off = (1.0 - sp) / (S - 1)
            self.trans_prob = jnp.full((S, S), off)
            self.trans_prob = self.trans_prob.at[jnp.diag_indices(S)].set(sp)

        self.init_prob = jnp.ones(S) / S

        self._update_cache()
        logger.info("Parameters initialised (n_states=%d, n_channels=%d)", S, C)

    def _update_cache(self) -> None:
        """Recompute precision matrices and log-determinants from covariances."""
        # Regularise covariances
        C = self.config.n_channels
        reg = 1e-6 * jnp.eye(C)
        covs = self.covariances + reg[None, :, :]

        self._precisions = jnp.linalg.inv(covs)
        # log det via Cholesky for stability
        L = jnp.linalg.cholesky(covs)
        self._log_dets = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)

    # -- inference -----------------------------------------------------------

    def _get_log_params(self):
        """Return log transition matrix and log initial probs."""
        log_trans = jnp.log(self.trans_prob + 1e-30)
        log_pi = jnp.log(self.init_prob + 1e-30)
        return log_trans, log_pi

    def infer(self, data: List[jax.Array] | jax.Array) -> List[jax.Array]:
        """Compute state probability time courses (gamma) for each session.

        Parameters
        ----------
        data : list of (T_i, C) arrays, or single (T, C) array

        Returns
        -------
        list of (T_i, S) arrays — posterior state probabilities
        """
        if isinstance(data, jax.Array) and data.ndim == 2:
            data = [data]

        # Apply the same standardisation used during fit
        if self._standardize and self._data_mean is not None:
            data = [(x - self._data_mean) / self._data_std for x in data]

        log_trans, log_pi = self._get_log_params()
        gammas = []
        for x in data:
            log_B = _log_emission_matrix(x, self.means, self._precisions, self._log_dets)
            gamma, _, _ = e_step(log_B, log_trans, log_pi)
            gammas.append(gamma)
        return gammas

    def decode(self, data: List[jax.Array] | jax.Array) -> List[jax.Array]:
        """Viterbi decoding — most likely state sequence.

        Returns
        -------
        list of (T_i,) int arrays — hard state assignments
        """
        if isinstance(data, jax.Array) and data.ndim == 2:
            data = [data]

        # Apply the same standardisation used during fit
        if self._standardize and self._data_mean is not None:
            data = [(x - self._data_mean) / self._data_std for x in data]

        log_trans, log_pi = self._get_log_params()
        sequences = []
        for x in data:
            log_B = _log_emission_matrix(x, self.means, self._precisions, self._log_dets)
            states = _viterbi(log_B, log_trans, log_pi)
            sequences.append(states)
        return sequences

    # -- EM fitting ----------------------------------------------------------

    def fit(
        self,
        data: List[jax.Array],
        n_epochs: int = 20,
        n_init: int = 1,
        standardize: bool = True,
        key: jax.Array | None = None,
    ) -> List[float]:
        """Fit the model to data using Baum-Welch EM.

        Parameters
        ----------
        data : list of (T_i, C) arrays
            Prepared data (one array per session/subject).
        n_epochs : int
            Number of EM iterations.
        n_init : int
            Number of random initialisations.  A short warm-up (max 5 epochs)
            is run for each; the initialisation with the best log-likelihood
            is kept and training continues for the remaining epochs.
        standardize : bool
            If True, z-score the concatenated data before EM (matching
            osl-dynamics behaviour) and store the mean/std for use in
            ``infer()``.
        key : jax random key, optional

        Returns
        -------
        list of float — log-likelihood history per epoch
        """
        if key is None:
            key = jr.PRNGKey(0)

        self._standardize = standardize

        # --- Standardise data if requested ---
        if standardize:
            X_all = jnp.concatenate(data, axis=0)
            self._data_mean = jnp.mean(X_all, axis=0)
            self._data_std = jnp.std(X_all, axis=0).clip(1e-10)
            data = [(x - self._data_mean) / self._data_std for x in data]

        S = self.config.n_states
        C = self.config.n_channels

        # --- Multiple random initialisations with warm-up ---
        warmup_epochs = min(5, n_epochs)
        if n_init > 1:
            best_ll = -jnp.inf
            best_state = None
            for i in range(n_init):
                k_init = jr.fold_in(key, i)
                # Reset parameters for this init
                self.means = None
                self.history = []
                self.init_params(data, key=k_init)

                # Run warm-up epochs
                self._run_em(data, warmup_epochs)
                final_ll = self.history[-1]
                logger.info(
                    "Init %d/%d  LL after %d warmup epochs: %.2f",
                    i + 1, n_init, warmup_epochs, final_ll,
                )
                if final_ll > best_ll:
                    best_ll = final_ll
                    best_state = {
                        "means": self.means,
                        "covariances": self.covariances,
                        "trans_prob": self.trans_prob,
                        "init_prob": self.init_prob,
                        "_precisions": self._precisions,
                        "_log_dets": self._log_dets,
                        "history": list(self.history),
                    }

            # Restore best init
            self.means = best_state["means"]
            self.covariances = best_state["covariances"]
            self.trans_prob = best_state["trans_prob"]
            self.init_prob = best_state["init_prob"]
            self._precisions = best_state["_precisions"]
            self._log_dets = best_state["_log_dets"]
            self.history = best_state["history"]

            # Continue training for remaining epochs
            remaining = n_epochs - warmup_epochs
            if remaining > 0:
                self._run_em(data, remaining)
        else:
            # Single init path
            if self.means is None:
                self.init_params(data, key=key)
            self._run_em(data, n_epochs)

        return self.history

    def _run_em(self, data: List[jax.Array], n_epochs: int) -> None:
        """Run *n_epochs* EM iterations on (already-standardised) data."""
        S = self.config.n_states
        C = self.config.n_channels

        for epoch in range(n_epochs):
            # ---- E-step (over all sessions) ----
            log_trans, log_pi = self._get_log_params()

            total_gamma = []
            total_xi = []
            total_ll = 0.0

            for x in data:
                log_B = _log_emission_matrix(
                    x, self.means, self._precisions, self._log_dets
                )
                gamma, xi, ll = e_step(log_B, log_trans, log_pi)
                total_gamma.append((x, gamma))
                total_xi.append(xi)
                total_ll += ll

            self.history.append(float(total_ll))
            logger.info("Epoch %d  LL=%.2f", len(self.history), total_ll)

            # ---- M-step ----
            # Accumulate sufficient statistics
            gamma_sum = jnp.zeros(S)
            weighted_x = jnp.zeros((S, C))
            xi_sum = jnp.zeros((S, S))

            for x, gamma in total_gamma:
                gamma_sum += gamma.sum(axis=0)
                weighted_x += jnp.einsum("ts,tc->sc", gamma, x)

            for xi in total_xi:
                xi_sum += xi.sum(axis=0)

            # Update means
            if self.config.learn_means:
                self.means = weighted_x / gamma_sum[:, None].clip(1e-10)

            # Update covariances
            if self.config.learn_covariances:
                new_covs = jnp.zeros((S, C, C))
                for x, gamma in total_gamma:
                    diff = x[:, None, :] - self.means[None, :, :]  # (T, S, C)
                    new_covs += jnp.einsum("ts,tsi,tsj->sij", gamma, diff, diff)
                self.covariances = new_covs / gamma_sum[:, None, None].clip(1e-10)

            # Update transition matrix
            if self.config.learn_trans_prob:
                row_sums = xi_sum.sum(axis=1, keepdims=True).clip(1e-10)
                self.trans_prob = xi_sum / row_sums

            # Update initial probs (from first time step of each session)
            gamma_init = jnp.stack(
                [gamma[0] for _, gamma in total_gamma]
            ).mean(axis=0)
            self.init_prob = gamma_init / gamma_init.sum()

            self._update_cache()

    # -- properties ----------------------------------------------------------

    @property
    def state_means(self) -> jax.Array:
        """(S, C) state mean vectors."""
        return self.means

    @property
    def state_covariances(self) -> jax.Array:
        """(S, C, C) state covariance matrices."""
        return self.covariances

    @property
    def transition_matrix(self) -> jax.Array:
        """(S, S) transition probability matrix."""
        return self.trans_prob

    def __repr__(self) -> str:
        return (
            f"GaussianHMM(n_states={self.config.n_states}, "
            f"n_channels={self.config.n_channels})"
        )


# ---------------------------------------------------------------------------
# Viterbi decoding
# ---------------------------------------------------------------------------


def _viterbi(
    log_B: jax.Array,
    log_trans: jax.Array,
    log_pi: jax.Array,
) -> jax.Array:
    """Viterbi algorithm in log-space.

    Returns
    -------
    states : (T,) int array — most likely state sequence
    """
    T, S = log_B.shape

    def scan_fn(carry, log_b_t):
        log_delta_prev = carry
        # log_delta_t[j] = log_b_t[j] + max_i(log_delta_prev[i] + log_trans[i,j])
        scores = log_delta_prev[:, None] + log_trans  # (S, S)
        psi_t = jnp.argmax(scores, axis=0)  # (S,)
        log_delta_t = log_b_t + jnp.max(scores, axis=0)
        return log_delta_t, (log_delta_t, psi_t)

    log_delta_0 = log_pi + log_B[0]
    _, (log_deltas, psis) = jax.lax.scan(scan_fn, log_delta_0, log_B[1:])

    # Backtrack
    states_rev = [jnp.argmax(log_deltas[-1])]
    for t in range(T - 2, 0, -1):
        states_rev.append(psis[t - 1][states_rev[-1]])
    # First state
    states_rev.append(jnp.argmax(log_delta_0) if T == 1 else psis[0][states_rev[-1]])

    # Handle T=1 edge case
    if T == 1:
        return jnp.array([jnp.argmax(log_delta_0)])

    states = jnp.array(states_rev[::-1])
    return states
