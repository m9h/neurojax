"""Leadfield forward projection: source-space → sensor-space.

Projects source-level neural mass model output through a leadfield (gain)
matrix to produce sensor-space EEG/MEG/sEEG signals for model fitting.
Follows WhoBPyT convention: sensor_signal = L @ source_signal.

Supports:
- EEG (with optional average reference)
- MEG (no average reference needed)
- sEEG (sparse leadfield, few contacts)

All operations are JAX-native and differentiable for gradient-based fitting.
"""

import jax
import jax.numpy as jnp

from neurojax.bench.monitors.fc import fc


class ForwardProjection:
    """Forward projection from source space to sensor space via leadfield.

    Parameters
    ----------
    leadfield : jnp.ndarray
        Leadfield (gain) matrix of shape (n_sensors, n_sources).
        Maps source-level activity to sensor-level measurements.
    avg_ref : bool, optional
        If True, apply average reference (subtract sensor mean at each
        timepoint). Standard for EEG but not MEG. Default: False.

    Examples
    --------
    >>> L = jnp.eye(4)
    >>> fp = ForwardProjection(L)
    >>> source = jnp.ones((4, 100))
    >>> sensor = fp.project(source)  # (4, 100)
    """

    def __init__(self, leadfield: jnp.ndarray, avg_ref: bool = False):
        self.leadfield = leadfield
        self.avg_ref = avg_ref

    @property
    def n_sensors(self) -> int:
        """Number of sensors (rows of leadfield)."""
        return self.leadfield.shape[0]

    @property
    def n_sources(self) -> int:
        """Number of sources (columns of leadfield)."""
        return self.leadfield.shape[1]

    def project(self, source_activity: jnp.ndarray) -> jnp.ndarray:
        """Project source activity to sensor space.

        Parameters
        ----------
        source_activity : jnp.ndarray
            Source-level activity. Shape (n_sources, n_timepoints) for
            timeseries or (n_sources,) for a single timepoint.

        Returns
        -------
        jnp.ndarray
            Sensor-level signal. Shape (n_sensors, n_timepoints) or
            (n_sensors,) matching input dimensionality.
        """
        sensor = self.leadfield @ source_activity
        if self.avg_ref:
            sensor = sensor - jnp.mean(sensor, axis=0, keepdims=True)
        return sensor

    def sensor_fc(self, source_activity: jnp.ndarray) -> jnp.ndarray:
        """Compute functional connectivity in sensor space.

        Projects source activity to sensor space, then computes the
        Pearson correlation matrix across sensors.

        Parameters
        ----------
        source_activity : jnp.ndarray
            Source-level activity of shape (n_sources, n_timepoints).

        Returns
        -------
        jnp.ndarray
            Sensor-space FC matrix of shape (n_sensors, n_sensors).
        """
        sensor = self.project(source_activity)
        return fc(sensor)

    def sensor_loss(
        self,
        source_activity: jnp.ndarray,
        empirical_sensor_data: jnp.ndarray,
    ) -> jnp.ndarray:
        """Differentiable MSE loss between projected and empirical sensor data.

        Computes the mean squared error between the forward-projected
        source activity and empirical sensor recordings.

        Parameters
        ----------
        source_activity : jnp.ndarray
            Source-level activity of shape (n_sources, n_timepoints).
        empirical_sensor_data : jnp.ndarray
            Empirical sensor recordings of shape (n_sensors, n_timepoints).

        Returns
        -------
        jnp.ndarray
            Scalar MSE loss value.
        """
        projected = self.project(source_activity)
        return jnp.mean((projected - empirical_sensor_data) ** 2)
