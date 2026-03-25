"""VbjaxFitnessAdapter — wraps vbjax JR simulation for whole-brain benchmarking.

Runs Jansen-Rit neural mass model on a connectome, generates BOLD signal
via Balloon-Windkessel hemodynamics, optionally projects to sensor space
via a leadfield, computes FC/FCD/sensor-space metrics, and returns
FitnessResult. Supports JAX autodiff through the entire pipeline via
a differentiable multi-modal loss.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import vbjax

from neurojax.bench.adapters.regional import RegionalParameterSpace
from neurojax.bench.fitness import FitnessAdapter, FitnessResult, ObjectiveSpec
from neurojax.bench.monitors.bold import BalloonWindkessel
from neurojax.bench.monitors.fc import fc, matrix_correlation
from neurojax.bench.monitors.fcd import fcd_ks_distance
from neurojax.bench.monitors.leadfield import ForwardProjection
from neurojax.bench.monitors.tep import extract_tep, extract_tep_sensor, tep_combined_loss
from neurojax.bench.stimuli.tms import TMSProtocol, make_stimulus_train


# JR parameters we expose for optimization (subset of JRTheta fields)
_JR_PARAM_BOUNDS = {
    "A": (2.0, 5.0),
    "B": (10.0, 40.0),
    "a": (0.05, 0.2),
    "b": (0.02, 0.1),
    "mu": (0.0, 0.5),
    "I": (0.0, 1.0),
    "K_gl": (0.0, 0.03),  # global coupling strength (stable range)
}


@dataclass
class LossWeights:
    """Weights for each term in the multi-modal loss.

    Set a weight to 0.0 to disable that loss term.
    """

    fc: float = 1.0
    fcd: float = 0.0
    sensor: float = 0.0
    tep: float = 0.0


@dataclass
class VbjaxSimConfig:
    """Configuration for a vbjax JR network simulation."""

    dt: float = 0.1  # ms
    duration: float = 60_000.0  # ms (60s)
    bold_dt: float = 2000.0  # ms (TR = 2s, 0.5 Hz)
    noise_sigma: float = 0.1
    warmup: float = 5_000.0  # ms to discard
    seed: int = 42
    use_hemodynamics: bool = True  # Use Balloon-Windkessel instead of subsampling


class VbjaxFitnessAdapter:
    """FitnessAdapter wrapping vbjax Jansen-Rit network simulation.

    Given a connectome (weights, delays) and empirical targets (FC, BOLD,
    sensor data), evaluates how well a parameter set reproduces the targets.

    Supports:
    - Balloon-Windkessel BOLD hemodynamics (replaces naive subsampling)
    - Per-region heterogeneous parameters via RegionalParameterSpace
    - Sensor-space loss via leadfield forward projection
    - Multi-modal loss (FC + FCD + sensor-space) with configurable weights
    - vmap-based batch evaluation for GPU parallelism

    Parameters
    ----------
    weights : jnp.ndarray
        Structural connectivity matrix (n_regions, n_regions).
    empirical_fc : jnp.ndarray
        Target FC matrix (n_regions, n_regions).
    empirical_bold : jnp.ndarray, optional
        Target BOLD timeseries (n_regions, n_timepoints) for FCD.
    empirical_sensor : jnp.ndarray, optional
        Target sensor-space data (n_sensors, n_timepoints) for sensor loss.
    config : VbjaxSimConfig, optional
        Simulation configuration.
    param_bounds : dict, optional
        Global parameter bounds. Ignored if regional_params is provided.
    regional_params : RegionalParameterSpace, optional
        Per-region parameter specification. Overrides param_bounds.
    leadfield : jnp.ndarray, optional
        Leadfield matrix (n_sensors, n_sources) for sensor-space projection.
    leadfield_avg_ref : bool
        Apply average reference in leadfield projection (EEG=True, MEG=False).
    loss_weights : LossWeights, optional
        Weights for multi-modal loss terms.
    tms_protocols : TMSProtocol or list of TMSProtocol, optional
        TMS stimulus specification(s). When provided, stimulus current is
        injected into the JR model's ``mu`` parameter at each timestep.
    empirical_tep : jnp.ndarray, optional
        Empirical TMS-evoked potential (n_sensors_or_regions, n_tep_samples)
        for TEP loss. Requires tms_protocols and loss_weights.tep > 0.
    """

    def __init__(
        self,
        weights: jnp.ndarray,
        empirical_fc: jnp.ndarray,
        empirical_bold: Optional[jnp.ndarray] = None,
        empirical_sensor: Optional[jnp.ndarray] = None,
        config: Optional[VbjaxSimConfig] = None,
        param_bounds: Optional[dict[str, tuple[float, float]]] = None,
        regional_params: Optional[RegionalParameterSpace] = None,
        leadfield: Optional[jnp.ndarray] = None,
        leadfield_avg_ref: bool = False,
        loss_weights: Optional[LossWeights] = None,
        tms_protocols: Optional[list[TMSProtocol] | TMSProtocol] = None,
        empirical_tep: Optional[jnp.ndarray] = None,
    ):
        self.weights = jnp.asarray(weights)
        self.n_regions = weights.shape[0]
        self.empirical_fc = jnp.asarray(empirical_fc)
        self.empirical_bold = (
            jnp.asarray(empirical_bold) if empirical_bold is not None else None
        )
        self.empirical_sensor = (
            jnp.asarray(empirical_sensor) if empirical_sensor is not None else None
        )
        self.config = config or VbjaxSimConfig()
        self.loss_weights = loss_weights or LossWeights()

        # Parameter space: regional or global
        self._regional = regional_params
        if regional_params is not None:
            self._param_bounds = None  # use regional_params instead
        else:
            self._param_bounds = param_bounds or _JR_PARAM_BOUNDS

        # Leadfield forward projection
        self._forward = None
        if leadfield is not None:
            self._forward = ForwardProjection(
                jnp.asarray(leadfield), avg_ref=leadfield_avg_ref
            )

        # Balloon-Windkessel BOLD monitor
        self._bold_monitor = None
        if self.config.use_hemodynamics:
            # vbjax BOLD expects time in seconds; our dt is in ms
            neural_dt_sec = self.config.dt / 1000.0
            bold_dt_sec = self.config.bold_dt / 1000.0
            self._bold_monitor = BalloonWindkessel(
                n_regions=self.n_regions,
                neural_dt=neural_dt_sec,
                bold_dt=bold_dt_sec,
            )

        # TMS stimulus
        self._stimulus = None
        self.empirical_tep = (
            jnp.asarray(empirical_tep) if empirical_tep is not None else None
        )
        if tms_protocols is not None:
            n_steps = int(self.config.duration / self.config.dt)
            self._stimulus = make_stimulus_train(
                tms_protocols, self.n_regions, self.config.dt, self.config.duration
            )
            # Store onset for TEP extraction
            if isinstance(tms_protocols, TMSProtocol):
                self._tms_onset = tms_protocols.t_onset
            else:
                self._tms_onset = tms_protocols[0].t_onset

        # Precompute coupling function
        self._cfun = vbjax.make_linear_cfun(self.weights)

        # Pre-build SDE integrator for gradient path
        def _grad_drift(y, p):
            theta, k_gl = p
            psp = y[1] - y[2]
            c = self._cfun(psp) * k_gl
            return vbjax.jr_dfun(y, c, theta)

        def _grad_diffusion(y, p):
            return self.config.noise_sigma

        self._grad_step, self._grad_loop = vbjax.make_sde(
            self.config.dt, _grad_drift, _grad_diffusion
        )

        # Build stimulus-aware integrator (custom scan loop)
        if self._stimulus is not None:
            def _stim_drift(y, p, stim):
                """Drift with external stimulus added to JR mu parameter."""
                theta, k_gl = p
                psp = y[1] - y[2]
                c = self._cfun(psp) * k_gl
                # Inject stimulus into the coupling term (adds to excitatory input)
                c_with_stim = c + stim
                return vbjax.jr_dfun(y, c_with_stim, theta)

            sqrt_dt = jnp.sqrt(self.config.dt)

            def _stim_step(y, z_t, stim_t, p):
                """Single Heun step with stimulus injection."""
                noise = sqrt_dt * z_t * self.config.noise_sigma
                d1 = _stim_drift(y, p, stim_t)
                y_tilde = y + self.config.dt * d1 + noise
                d2 = _stim_drift(y_tilde, p, stim_t)
                return y + 0.5 * self.config.dt * (d1 + d2) + noise

            @jax.jit
            def _stim_loop(y0, zs, stimulus, p):
                def op(y, inputs):
                    z_t, stim_t = inputs
                    y_next = _stim_step(y, z_t, stim_t, p)
                    return y_next, y_next
                _, states = jax.lax.scan(op, y0, (zs, stimulus))
                return states

            self._stim_loop = _stim_loop

    # ------------------------------------------------------------------
    # Parameter space
    # ------------------------------------------------------------------

    @property
    def parameter_space(self) -> dict[str, tuple[float, float]]:
        """Parameter names mapped to (lower_bound, upper_bound).

        When using regional parameters, returns the flat-array bounds
        as a dict keyed by the expanded parameter names (e.g.,
        "K_gl" for global, "A_0", "A_1", ... for per-region A).
        """
        if self._regional is not None:
            lower, upper = self._regional.bounds
            names = self._regional.param_names
            return {name: (float(lo), float(hi))
                    for name, lo, hi in zip(names, lower, upper)}
        return dict(self._param_bounds)

    @property
    def objectives(self) -> list[ObjectiveSpec]:
        objs = [ObjectiveSpec("fc_correlation", "maximize")]
        if self.loss_weights.fcd > 0:
            objs.append(ObjectiveSpec("fcd_ks_distance", "minimize"))
        if self.loss_weights.sensor > 0 and self._forward is not None:
            objs.append(ObjectiveSpec("sensor_mse", "minimize"))
        return objs

    # ------------------------------------------------------------------
    # Theta construction
    # ------------------------------------------------------------------

    def _make_theta(self, params: dict[str, float]) -> vbjax.JRTheta:
        """Create JRTheta from param dict, using defaults for unspecified."""
        defaults = vbjax.jr_default_theta._asdict()
        jr_params = {k: v for k, v in params.items() if k in defaults}
        defaults.update(jr_params)
        return vbjax.JRTheta(**defaults)

    def _params_from_array(self, param_array: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Convert a flat parameter array to a dict.

        Handles both global-only and regional parameter spaces.
        """
        if self._regional is not None:
            # Use regional parameter space to unpack
            # Return as JAX arrays for differentiability
            result = {}
            idx = 0
            for name in self._regional._global_names:
                result[name] = param_array[idx]
                idx += 1
            for name in self._regional._regional_names:
                result[name] = param_array[idx:idx + self._regional.n_regions]
                idx += self._regional.n_regions
            return result
        else:
            param_names = list(self._param_bounds.keys())
            return {name: param_array[i] for i, name in enumerate(param_names)}

    # ------------------------------------------------------------------
    # Simulation (non-differentiable path)
    # ------------------------------------------------------------------

    def _simulate(
        self, params: dict[str, float], key: jax.Array
    ) -> jnp.ndarray:
        """Run JR network simulation, return BOLD timeseries.

        Returns:
            (n_regions, n_timepoints) BOLD array.
        """
        theta = self._make_theta(params)
        n_steps = int(self.config.duration / self.config.dt)
        warmup_steps = int(self.config.warmup / self.config.dt)

        y0 = jnp.zeros((6, self.n_regions))
        noise = (
            jax.random.normal(key, (n_steps, 6, self.n_regions))
            * self.config.noise_sigma
        )

        k_gl = params.get("K_gl", 0.01)

        def drift(y, p):
            psp = y[1] - y[2]
            c = self._cfun(psp) * k_gl
            return vbjax.jr_dfun(y, c, p)

        def diffusion(y, p):
            return self.config.noise_sigma

        _, loop = vbjax.make_sde(self.config.dt, drift, diffusion)
        states = loop(y0, noise, theta)  # (n_steps, 6, n_regions)

        # Extract pyramidal PSP as neural activity
        neural_activity = states[:, 1, :] - states[:, 2, :]  # (n_steps, n_regions)
        neural_activity = neural_activity[warmup_steps:]

        # Convert to BOLD
        bold = self._neural_to_bold(neural_activity)
        return bold

    # ------------------------------------------------------------------
    # Simulation (differentiable JAX path)
    # ------------------------------------------------------------------

    def _simulate_jax(
        self, params: dict[str, jnp.ndarray], key: jax.Array
    ) -> jnp.ndarray:
        """JAX-traceable simulation using pre-built integrator.

        Parameters are passed as (theta, k_gl) pytree through the `p`
        argument of the scan loop — no closure over traced values.

        When TMS stimulus is configured, uses the stimulus-aware loop
        that injects external current at each timestep.
        """
        theta = self._make_theta(
            {k: v for k, v in params.items() if k in vbjax.jr_default_theta._asdict()}
        )
        k_gl = params.get("K_gl", jnp.array(0.01))
        combined_params = (theta, k_gl)

        n_steps = int(self.config.duration / self.config.dt)
        warmup_steps = int(self.config.warmup / self.config.dt)

        y0 = jnp.zeros((6, self.n_regions))
        noise = jax.random.normal(key, (n_steps, 6, self.n_regions))

        if self._stimulus is not None:
            states = self._stim_loop(y0, noise, self._stimulus, combined_params)
        else:
            # Scale noise for the non-stimulus path (vbjax loop expects pre-scaled)
            states = self._grad_loop(y0, noise * self.config.noise_sigma, combined_params)

        neural_activity = states[:, 1, :] - states[:, 2, :]
        neural_activity = neural_activity[warmup_steps:]

        bold = self._neural_to_bold(neural_activity)
        return bold

    def _simulate_jax_neural(
        self, params: dict[str, jnp.ndarray], key: jax.Array
    ) -> jnp.ndarray:
        """Like _simulate_jax but returns neural activity instead of BOLD.

        Used for TEP extraction where we need the raw neural timeseries
        at simulation resolution, not BOLD-subsampled.

        Returns
        -------
        jnp.ndarray
            Neural activity of shape (n_regions, n_timepoints) at
            simulation dt resolution (post-warmup).
        """
        theta = self._make_theta(
            {k: v for k, v in params.items() if k in vbjax.jr_default_theta._asdict()}
        )
        k_gl = params.get("K_gl", jnp.array(0.01))
        combined_params = (theta, k_gl)

        n_steps = int(self.config.duration / self.config.dt)
        warmup_steps = int(self.config.warmup / self.config.dt)

        y0 = jnp.zeros((6, self.n_regions))
        noise = jax.random.normal(key, (n_steps, 6, self.n_regions))

        if self._stimulus is not None:
            states = self._stim_loop(y0, noise, self._stimulus, combined_params)
        else:
            states = self._grad_loop(y0, noise * self.config.noise_sigma, combined_params)

        neural_activity = states[:, 1, :] - states[:, 2, :]
        neural_activity = neural_activity[warmup_steps:]
        return neural_activity.T  # (n_regions, n_timepoints)

    # ------------------------------------------------------------------
    # Neural → BOLD conversion (shared by both paths)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_neural(neural_activity: jnp.ndarray) -> jnp.ndarray:
        """Normalize neural activity for hemodynamic input.

        JR pyramidal PSP can range from -40 to +15. The Balloon-Windkessel
        model expects input near [0, 1]. We apply a gentle sigmoid
        (differentiable) to squash the signal, following the WhoBPyT
        convention of passing neural activity through a transfer function
        before hemodynamics.

        Parameters
        ----------
        neural_activity : jnp.ndarray, shape (n_time, n_regions)

        Returns
        -------
        jnp.ndarray, same shape, values in (0, 1).
        """
        return jax.nn.sigmoid(neural_activity * 0.1)

    def _neural_to_bold(self, neural_activity: jnp.ndarray) -> jnp.ndarray:
        """Convert neural activity (n_time, n_regions) to BOLD (n_regions, n_bold).

        Uses Balloon-Windkessel if enabled, otherwise subsamples.
        """
        if self._bold_monitor is not None:
            normalized = self._normalize_neural(neural_activity)
            bold = self._bold_monitor.transform(normalized.T)  # (n_regions, n_bold)
        else:
            bold_subsample = int(self.config.bold_dt / self.config.dt)
            bold = neural_activity[::bold_subsample].T  # (n_regions, n_bold)
        return bold

    # ------------------------------------------------------------------
    # Evaluation (non-differentiable, returns FitnessResult)
    # ------------------------------------------------------------------

    def evaluate(self, params: dict[str, float]) -> FitnessResult:
        """Run simulation and compute fitness against empirical targets."""
        t0 = time.perf_counter()

        key = jax.random.PRNGKey(self.config.seed)
        bold = self._simulate(params, key)

        sim_fc = fc(bold)
        r_fc = float(matrix_correlation(sim_fc, self.empirical_fc))

        raw_objectives = {"fc_correlation": r_fc}

        # FCD (only if empirical BOLD available)
        d_fcd = 0.0
        if self.empirical_bold is not None:
            d_fcd = float(
                fcd_ks_distance(bold, self.empirical_bold, window_size=30, step_size=5)
            )
            raw_objectives["fcd_ks_distance"] = d_fcd

        # Sensor-space loss
        sensor_mse = 0.0
        if self._forward is not None and self.empirical_sensor is not None:
            # Use neural activity (pre-BOLD) for sensor projection
            # Re-simulate to get neural activity — or project BOLD as approximation
            sensor_mse = float(self._forward.sensor_loss(bold, self.empirical_sensor))
            raw_objectives["sensor_mse"] = sensor_mse

        wall_time = time.perf_counter() - t0

        return FitnessResult(
            fc_correlation=r_fc,
            fcd_ks_distance=d_fcd,
            raw_objectives=raw_objectives,
            simulated_fc=np.asarray(sim_fc),
            simulated_bold=np.asarray(bold),
            wall_time=wall_time,
        )

    # ------------------------------------------------------------------
    # Batch evaluation with vmap
    # ------------------------------------------------------------------

    def evaluate_batch(
        self, params_batch: list[dict[str, float]]
    ) -> list[FitnessResult]:
        """Batch evaluation. Uses vmap when possible for GPU parallelism."""
        if not params_batch:
            return []

        # Try vmap path: convert dicts to stacked arrays
        try:
            return self._evaluate_batch_vmap(params_batch)
        except Exception:
            # Fallback to sequential
            return [self.evaluate(p) for p in params_batch]

    def _evaluate_batch_vmap(
        self, params_batch: list[dict[str, float]]
    ) -> list[FitnessResult]:
        """vmap-accelerated batch evaluation."""
        param_names = list(
            (self._param_bounds or {k: (0, 1) for k in params_batch[0]}).keys()
        )
        batch_size = len(params_batch)

        # Stack parameters into (batch_size, n_params) array
        param_array = jnp.array(
            [[p.get(name, 0.0) for name in param_names] for p in params_batch]
        )

        # Generate per-sample PRNG keys
        keys = jax.random.split(jax.random.PRNGKey(self.config.seed), batch_size)

        # vmap over the simulation + FC computation
        def _single_eval(param_vec, key):
            params = {name: param_vec[i] for i, name in enumerate(param_names)}
            bold = self._simulate_jax(params, key)
            sim_fc = fc(bold)
            r = matrix_correlation(sim_fc, self.empirical_fc)
            return r, sim_fc, bold

        batched_eval = jax.vmap(_single_eval)
        r_fcs, sim_fcs, bolds = batched_eval(param_array, keys)

        # Convert to list of FitnessResult
        results = []
        for i in range(batch_size):
            results.append(FitnessResult(
                fc_correlation=float(r_fcs[i]),
                fcd_ks_distance=0.0,
                simulated_fc=np.asarray(sim_fcs[i]),
                simulated_bold=np.asarray(bolds[i]),
            ))
        return results

    # ------------------------------------------------------------------
    # Differentiable loss (for gradient-based optimization)
    # ------------------------------------------------------------------

    def loss(self, param_array: jnp.ndarray) -> jnp.ndarray:
        """Multi-modal differentiable loss for gradient-based optimization.

        Combines FC correlation, FCD KS distance, and sensor-space MSE
        with configurable weights from self.loss_weights.

        Parameters
        ----------
        param_array : jnp.ndarray
            1D array of parameter values. Layout matches parameter_space
            ordering (global params for flat, or regional layout).

        Returns
        -------
        jnp.ndarray
            Scalar loss value (minimize this). Fully JAX-traceable.
        """
        params = self._params_from_array(param_array)
        key = jax.random.PRNGKey(self.config.seed)
        bold = self._simulate_jax(params, key)

        total_loss = jnp.array(0.0)

        # FC loss: negative correlation (maximize correlation = minimize negative)
        if self.loss_weights.fc > 0:
            sim_fc = fc(bold)
            fc_loss = -matrix_correlation(sim_fc, self.empirical_fc)
            total_loss = total_loss + self.loss_weights.fc * fc_loss

        # FCD loss: KS distance (minimize)
        if self.loss_weights.fcd > 0 and self.empirical_bold is not None:
            fcd_loss = fcd_ks_distance(
                bold, self.empirical_bold, window_size=30, step_size=5
            )
            total_loss = total_loss + self.loss_weights.fcd * fcd_loss

        # Sensor-space loss: MSE after leadfield projection (minimize)
        if (
            self.loss_weights.sensor > 0
            and self._forward is not None
            and self.empirical_sensor is not None
        ):
            sensor_loss = self._forward.sensor_loss(bold, self.empirical_sensor)
            total_loss = total_loss + self.loss_weights.sensor * sensor_loss

        # TEP loss: waveform + GFP matching for TMS-evoked potentials
        if (
            self.loss_weights.tep > 0
            and self._stimulus is not None
            and self.empirical_tep is not None
        ):
            neural = self._simulate_jax_neural(params, key)
            # Project to sensor space if leadfield available
            if self._forward is not None:
                sim_tep = extract_tep_sensor(
                    neural, self._forward,
                    t_onset=self._tms_onset, dt=self.config.dt,
                )
            else:
                sim_tep = extract_tep(
                    neural,
                    t_onset=self._tms_onset, dt=self.config.dt,
                )
            # Match shapes: truncate to smaller of sim/empirical
            n_samples = min(sim_tep.shape[1], self.empirical_tep.shape[1])
            sim_tep = sim_tep[:, :n_samples]
            emp_tep = self.empirical_tep[:, :n_samples]
            tep_loss_val = tep_combined_loss(sim_tep, emp_tep)
            total_loss = total_loss + self.loss_weights.tep * tep_loss_val

        return total_loss

    def fc_loss(self, param_array: jnp.ndarray) -> jnp.ndarray:
        """Backward-compatible: FC-only differentiable loss.

        Equivalent to loss() with loss_weights=(fc=1, fcd=0, sensor=0).
        """
        params = self._params_from_array(param_array)
        key = jax.random.PRNGKey(self.config.seed)
        bold = self._simulate_jax(params, key)
        sim_fc = fc(bold)
        return -matrix_correlation(sim_fc, self.empirical_fc)

    # ------------------------------------------------------------------
    # Convenience: number of parameters
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """Total number of scalar parameters in the optimization."""
        if self._regional is not None:
            return self._regional.n_params
        return len(self._param_bounds)

    def default_param_array(self) -> jnp.ndarray:
        """Initial parameter array at midpoint of bounds."""
        if self._regional is not None:
            return jnp.array(self._regional.default_flat_array())
        ps = self._param_bounds
        return jnp.array([(lo + hi) / 2 for lo, hi in ps.values()])

    def bounds_arrays(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Lower and upper bound arrays for the optimizer."""
        if self._regional is not None:
            lo, hi = self._regional.bounds
            return jnp.array(lo), jnp.array(hi)
        ps = self._param_bounds
        lo = jnp.array([lo for lo, _ in ps.values()])
        hi = jnp.array([hi for _, hi in ps.values()])
        return lo, hi
