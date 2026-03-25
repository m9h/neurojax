"""Reduced Wong-Wang (RWW) neural mass model (Deco et al. 2013, J Neurosci).

A 2-state mean-field model of excitatory (S_E) and inhibitory (S_I) synaptic
gating variables, derived from spiking network dynamics.  This is the primary
model used by WhoBPyT (Griffiths Lab) for whole-brain fMRI fitting.

Equations (reduced form)::

    dS_E/dt = -S_E/tau_E + (1 - S_E) * gamma_E * r_E + sigma * noise_E
    dS_I/dt = -S_I/tau_I + gamma_I * r_I + sigma * noise_I

    r_E = H_E(I_E) = (a_E * I_E - b_E) / (1 - exp(-d_E * (a_E * I_E - b_E)))
    r_I = H_I(I_I) = (a_I * I_I - b_I) / (1 - exp(-d_I * (a_I * I_I - b_I)))

    I_E = W_E*I_0 + w_plus*J_NMDA*S_E + G*J_NMDA*coupling - J_I*S_I + I_ext
    I_I = W_I*I_0 + J_NMDA*S_E - S_I + I_ext_I

The drift function ``rww_dfun`` is compatible with ``vbjax.make_sde``: it has
the signature ``(state, coupling, theta) -> dstate`` where state has shape
``(2, n_nodes)`` with ``state[0] = S_E`` and ``state[1] = S_I``.

References
----------
Deco G, Ponce-Alvarez A, Mantini D, Romani GL, Hagmann P, Corbetta M (2013).
Resting-state functional connectivity emerges from structurally and dynamically
shaped slow linear fluctuations. J Neurosci 33(27):11239-11252.

Wong K-F, Wang X-J (2006). A recurrent network mechanism of time integration
in perceptual decisions. J Neurosci 26(4):1314-1328.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class RWWTheta(NamedTuple):
    """Parameters for the Reduced Wong-Wang model (Deco et al. 2013).

    Attributes
    ----------
    a_E, b_E, d_E : float
        Excitatory transfer function parameters (nC^-1, Hz, s).
    a_I, b_I, d_I : float
        Inhibitory transfer function parameters.
    tau_E, tau_I : float
        Excitatory (NMDA) and inhibitory (GABA) time constants (s).
    gamma_E, gamma_I : float
        Kinetic conversion factors (/ms -> /s).
    W_E, W_I : float
        Local excitatory and inhibitory weights.
    w_plus : float
        Recurrent excitatory weight (local E->E).
    J_NMDA : float
        NMDA synaptic coupling strength (nA).
    J_I : float
        Inhibitory synaptic coupling strength (nA).
    I_0 : float
        Overall effective external input (nA).
    I_ext : float
        External excitatory input current (nA).
    I_ext_I : float
        External inhibitory input current (nA).
    G : float
        Global coupling scaling.
    sigma : float
        Noise amplitude.
    """

    a_E: float = 310.0
    b_E: float = 125.0
    d_E: float = 0.16
    a_I: float = 615.0
    b_I: float = 177.0
    d_I: float = 0.087
    tau_E: float = 0.1
    tau_I: float = 0.01
    gamma_E: float = 0.641
    gamma_I: float = 1.0
    W_E: float = 1.0
    W_I: float = 0.7
    w_plus: float = 1.4
    J_NMDA: float = 0.15
    J_I: float = 1.0
    I_0: float = 0.382
    I_ext: float = 0.0
    I_ext_I: float = 0.0
    G: float = 2.0
    sigma: float = 0.01


#: Default parameter set from Deco et al. (2013).
rww_default_theta = RWWTheta()


def rww_transfer_E(I_E: jnp.ndarray, theta: RWWTheta) -> jnp.ndarray:
    """Excitatory transfer function H_E (input-output, Hz).

    H_E(I) = (a_E * I - b_E) / (1 - exp(-d_E * (a_E * I - b_E)))

    Uses the identity ``x / (1 - exp(-c*x)) = x * sigmoid(c*x) / (1 - sigmoid(c*x))``
    rewritten via softplus to avoid 0/0 at ``x=0``.

    Parameters
    ----------
    I_E : array
        Total excitatory input current (nA).
    theta : RWWTheta
        Model parameters.

    Returns
    -------
    r_E : array
        Excitatory firing rate (Hz).
    """
    x = theta.a_E * I_E - theta.b_E
    # x / (1 - exp(-d*x))  is the Abbott form.
    # Numerically stable: use the identity x/(1 - e^{-cx}) = x + x/(e^{cx}-1)
    # but the simplest safe form uses jnp.where to handle x~0.
    cx = theta.d_E * x
    # For large |cx|, one branch dominates.  Near 0 use Taylor: x/(1-e^{-cx}) ~ 1/c.
    safe = jnp.where(
        jnp.abs(cx) > 1e-6,
        x / (1.0 - jnp.exp(-cx)),
        # Taylor first order: 1/d_E + x/2 + d_E*x^2/12 ...
        1.0 / theta.d_E + x / 2.0,
    )
    return safe


def rww_transfer_I(I_I: jnp.ndarray, theta: RWWTheta) -> jnp.ndarray:
    """Inhibitory transfer function H_I (input-output, Hz).

    H_I(I) = (a_I * I - b_I) / (1 - exp(-d_I * (a_I * I - b_I)))

    Parameters
    ----------
    I_I : array
        Total inhibitory input current (nA).
    theta : RWWTheta
        Model parameters.

    Returns
    -------
    r_I : array
        Inhibitory firing rate (Hz).
    """
    x = theta.a_I * I_I - theta.b_I
    cx = theta.d_I * x
    safe = jnp.where(
        jnp.abs(cx) > 1e-6,
        x / (1.0 - jnp.exp(-cx)),
        1.0 / theta.d_I + x / 2.0,
    )
    return safe


def rww_dfun(
    state: jnp.ndarray,
    coupling: jnp.ndarray,
    theta: RWWTheta,
) -> jnp.ndarray:
    """Drift function for the Reduced Wong-Wang model.

    Compatible with ``vbjax.make_sde``: when wrapped in a closure that
    computes coupling from state, the signature becomes ``drift(state, p)``.

    Parameters
    ----------
    state : array, shape (2, n_nodes)
        ``state[0]`` = S_E (excitatory gating), ``state[1]`` = S_I (inhibitory gating).
    coupling : array, shape (n_nodes,)
        Pre-computed long-range coupling input to each node
        (typically ``weights @ S_E`` scaled by ``G * J_NMDA``).
    theta : RWWTheta
        Model parameters.

    Returns
    -------
    dstate : array, shape (2, n_nodes)
        Time derivatives ``[dS_E/dt, dS_I/dt]`` (deterministic part only;
        noise is handled by the SDE integrator).
    """
    S_E = state[0]
    S_I = state[1]

    # Total input currents
    I_E = (
        theta.W_E * theta.I_0
        + theta.w_plus * theta.J_NMDA * S_E
        + theta.G * theta.J_NMDA * coupling
        - theta.J_I * S_I
        + theta.I_ext
    )
    I_I = (
        theta.W_I * theta.I_0
        + theta.J_NMDA * S_E
        - S_I
        + theta.I_ext_I
    )

    # Firing rates via transfer functions
    r_E = rww_transfer_E(I_E, theta)
    r_I = rww_transfer_I(I_I, theta)

    # Synaptic gating dynamics
    dS_E = -S_E / theta.tau_E + (1.0 - S_E) * theta.gamma_E * r_E
    dS_I = -S_I / theta.tau_I + theta.gamma_I * r_I

    return jnp.array([dS_E, dS_I])
