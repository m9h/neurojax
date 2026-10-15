"""The canonical-microcircuit cross-spectral model as a competitor in Bayesian spectral decomposition.

A single-source CMC with one LFP electrode (``neurojax.dcm.cmc_csd``) seen
through the :class:`~neurojax.bsd.models.SpectralModel` interface: the
unconstrained parameter vector is SPM's own log-scaling parameterisation,
the Gaussian priors are SPM's prior variances, and ``log_psd`` is the log of
the predicted auto-spectrum.  This is DCM for CSD under the same free
energy as the specparam, Liley and ξ-α models, which is what lets the four
be compared on one spectrum.

Free parameters (16): intrinsic gains ``G`` (4), synaptic time constants
``T`` (4), sigmoid slope ``S``, intrinsic delay ``D``, neuronal innovations
``a`` (amplitude, exponent), non-specific channel noise ``b`` (2),
specific channel noise ``c`` (2), electrode gain ``L`` (log).  The DCT
innovation coefficients ``d`` are held at zero (prior mean) by default to
keep the comparison fair at 16 parameters; set ``free_d=True`` for SPM's
full 20.

The mechanistic content: the CMC has four populations and therefore more
than one resonance, which is the thing the LEMON and propofol comparisons
showed the two-process and damping-family models lack.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from neurojax.bsd.models import SpectralModel
from neurojax.dcm.cmc_csd import cmc_csd, cmc_priors


class CMCSpectrum(SpectralModel):
    """Single-source canonical microcircuit auto-spectrum with SPM's priors."""

    free_d: bool = eqx.field(static=True)

    def __init__(self, free_d=False, centre=None):
        self.free_d = bool(free_d)
        names = ([f"G{i}" for i in range(4)] + [f"T{i}" for i in range(4)] + ["S", "D"]
                 + ["a_amp", "a_exp", "b_amp", "b_exp", "c_amp", "c_exp", "log_L"])
        sd = [np.sqrt(1 / 32)] * 4 + [np.sqrt(1 / 32)] * 4 + [np.sqrt(1 / 64), np.sqrt(1 / 64)] + [np.sqrt(1 / 128)] * 6 + [np.sqrt(64.0)]
        mean = [0.0] * 17
        if self.free_d:
            names += [f"d{i}" for i in range(4)]; mean += [0.0] * 4; sd += [np.sqrt(1 / 128)] * 4
        self.names = tuple(names)
        self.prior_mean = jnp.asarray(mean, dtype=float)
        self.prior_sd = jnp.asarray(sd, dtype=float)

    def params(self, theta):
        pE, _ = cmc_priors([np.zeros((1, 1))] * 3, (), np.ones((1, 1)))
        P = dict(pE)
        P["G"] = theta[0:4].reshape(1, 4)
        P["T"] = theta[4:8].reshape(1, 4)
        P["S"] = theta[8]
        P["D"] = theta[9].reshape(1, 1)
        P["a"] = theta[10:12].reshape(2, 1)
        P["b"] = theta[12:14].reshape(2, 1)
        P["c"] = theta[14:16].reshape(2, 1)
        P["L"] = jnp.exp(theta[16]).reshape(1, 1)
        P["d"] = theta[17:21].reshape(4, 1) if self.free_d else jnp.zeros((4, 1))
        return P

    def log_psd(self, freqs, theta):
        G = cmc_csd(self.params(theta), jnp.asarray(freqs, dtype=float), np.zeros((1, 0)))[0]
        return jnp.log(jnp.maximum(jnp.real(G[:, 0, 0]), 1e-300))

    def natural(self, theta):
        d = {n: float(v) for n, v in zip(self.names, theta)}
        d["L"] = float(jnp.exp(theta[16]))
        return d
