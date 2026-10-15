"""Likelihoods for an estimated power spectrum.

Two forms.  The Gaussian likelihood on log power with a free log-precision
hyperparameter is what DCM and BSD use; it is robust and lets the data set
the residual scale.  The Whittle likelihood is the exact one for a Welch
estimate that averages ``n_segments`` independent periodograms, each
``S_hat = S * chi^2_2 / 2``, so that ``n S_hat / S ~ Gamma(n, 1)``; it has
no free noise parameter and is the right choice when ``n_segments`` is
known and small.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np


def gaussian_log_likelihood(log_psd_model, log_psd_obs, log_precision):
    """``sum_f ln N(ln S_obs | ln S_model, exp(-log_precision))``."""
    r = jnp.asarray(log_psd_obs, dtype=float) - log_psd_model
    lam = jnp.exp(log_precision)
    n = r.shape[0]
    return 0.5 * n * (log_precision - jnp.log(2.0 * jnp.pi)) - 0.5 * lam * jnp.sum(r ** 2)


def whittle_log_likelihood(log_psd_model, log_psd_obs, n_segments):
    """Log density of a Welch estimate averaging ``n_segments`` periodograms.

    ``S_hat ~ S Gamma(n, 1/n)``, so in terms of ``ln S_hat``:
    ``ln p = sum_f [n ln n - ln Gamma(n) + n (ln S_hat - ln S) - n S_hat / S]``
    (the Jacobian of the log transform is included).
    """
    n = float(n_segments)
    ls_obs = jnp.asarray(log_psd_obs, dtype=float)
    ratio = jnp.exp(ls_obs - log_psd_model)
    const = n * np.log(n) - math.lgamma(n)
    return jnp.sum(const + n * (ls_obs - log_psd_model) - n * ratio)
