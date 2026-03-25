"""Balloon-Windkessel BOLD hemodynamic monitor.

Wraps vbjax's Balloon-Windkessel forward model to convert neural mass model
output (e.g. synaptic activity) into simulated BOLD fMRI signal.

The Balloon-Windkessel model (Friston et al. 2000, 2003) transforms a
neuronal input signal through a cascade of hemodynamic state variables:

    s  — vasodilatory signal
    f  — blood inflow
    v  — blood volume
    q  — deoxyhemoglobin content

The BOLD signal is a nonlinear function of v and q.

All operations are JAX-native and differentiable for gradient-based fitting.

References
----------
Friston KJ, Mechelli A, Turner R, Price CJ (2000). Nonlinear responses in
fMRI: the Balloon model, Volterra kernels, and other hemodynamics. NeuroImage
12(4):466-477.

Stephan KE, Weiskopf N, Drysdale PM, Robinson PA, Friston KJ (2007).
Comparing hemodynamic models with DCM. NeuroImage 38(3):387-401.
"""

from __future__ import annotations

import jax.numpy as jnp
import vbjax


class BalloonWindkessel:
    """Balloon-Windkessel BOLD hemodynamic model via vbjax.

    Converts neural activity timeseries to BOLD signal using the
    Balloon-Windkessel hemodynamic model. Replaces naive subsampling
    with a physiologically grounded forward model.

    Parameters
    ----------
    n_regions : int
        Number of brain regions.
    neural_dt : float
        Timestep of the neural simulation in seconds.
    bold_dt : float
        Desired BOLD sampling interval in seconds (e.g., TR = 2.0).
    bold_theta : vbjax.BOLDTheta, optional
        Hemodynamic parameters. Default: vbjax.bold_default_theta.

    Examples
    --------
    >>> bw = BalloonWindkessel(n_regions=80, neural_dt=0.0001, bold_dt=2.0)
    >>> neural = jnp.ones((80, 600000))  # 80 regions, 60s at 0.1ms
    >>> bold = bw.transform(neural)      # (80, 30) — 30 TRs at TR=2s
    """

    def __init__(
        self,
        n_regions: int,
        neural_dt: float,
        bold_dt: float = 2.0,
        bold_theta: vbjax.BOLDTheta | None = None,
    ):
        self.n_regions = n_regions
        self.neural_dt = neural_dt
        self.bold_dt = bold_dt
        self.bold_theta = bold_theta or vbjax.bold_default_theta

        # Build the BOLD integrator from vbjax
        self._sfvq0, self._step, self._sample = vbjax.make_bold(
            (n_regions,), neural_dt, self.bold_theta
        )

        # How many neural timesteps per BOLD sample
        self._subsample_factor = max(1, int(bold_dt / neural_dt))

    def transform(self, neural_activity: jnp.ndarray) -> jnp.ndarray:
        """Convert neural activity to BOLD signal.

        Parameters
        ----------
        neural_activity : jnp.ndarray
            Neural timeseries of shape (n_regions, n_timepoints).
            This is typically the excitatory synaptic variable (S_E for RWW)
            or pyramidal PSP (y1-y2 for Jansen-Rit).

        Returns
        -------
        jnp.ndarray
            BOLD signal of shape (n_regions, n_bold_timepoints).
        """
        n_regions, n_time = neural_activity.shape

        # Integrate hemodynamic state at neural resolution using scan
        def scan_fn(sfvq, x_t):
            sfvq_next = self._step(sfvq, x_t)
            return sfvq_next, sfvq_next

        _, all_states = jax.lax.scan(scan_fn, self._sfvq0, neural_activity.T)
        # all_states shape: (n_time, 4, n_regions)

        # Subsample to BOLD resolution
        bold_states = all_states[self._subsample_factor - 1 :: self._subsample_factor]

        # Apply observation function to get BOLD signal
        # vbjax sample returns (state, bold_value)
        bold_samples = jax.vmap(lambda st: self._sample(st)[1])(bold_states)
        # bold_samples shape: (n_bold, n_regions)

        return bold_samples.T  # (n_regions, n_bold)

    def transform_online(self, neural_activity: jnp.ndarray) -> jnp.ndarray:
        """Same as transform but returns only the final BOLD sample.

        Useful inside a simulation loop where BOLD is sampled periodically.

        Parameters
        ----------
        neural_activity : jnp.ndarray
            Neural timeseries of shape (n_regions, n_timepoints) for one
            BOLD sampling period.

        Returns
        -------
        jnp.ndarray
            Single BOLD sample of shape (n_regions,).
        """
        def scan_fn(sfvq, x_t):
            sfvq_next = self._step(sfvq, x_t)
            return sfvq_next, None

        final_state, _ = jax.lax.scan(scan_fn, self._sfvq0, neural_activity.T)
        _, bold_value = self._sample(final_state)
        return bold_value


# Need jax for lax.scan
import jax
