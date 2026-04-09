"""CMCSpectralAdapter — tunes CMC connectivity parameters for physiological spectra.

Optimizes the 8 intrinsic connectivity weights + external input of the
canonical microcircuit to produce oscillatory dynamics with:
- Peak frequency in the alpha band (8-13 Hz)
- Sufficient oscillation amplitude
- Stable bounded dynamics

This adapter targets single-node spectral properties, not whole-brain FC.
Use it to find good default parameters before whole-brain fitting.

Compatible with LLaMEA, CMA-ES, and gradient optimizers via the
FitnessAdapter protocol.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import vbjax

from neurojax.bench.fitness import FitnessAdapter, FitnessResult, ObjectiveSpec


# CMC connectivity parameters to optimize + their search bounds
_CMC_SPECTRAL_BOUNDS = {
    "g_ss_sp": (10.0, 500.0),
    "g_sp_ii": (5.0, 200.0),
    "g_sp_dp": (10.0, 500.0),
    "g_dp_ii": (5.0, 200.0),
    "g_dp_sp": (10.0, 500.0),
    "g_ii_ss": (5.0, 200.0),
    "g_ii_sp": (5.0, 200.0),
    "g_ii_dp": (5.0, 200.0),
    "I": (50.0, 500.0),
}


@dataclass
class CMCSpectralConfig:
    """Configuration for CMC spectral optimization."""

    dt: float = 0.5          # integration timestep (ms)
    duration: float = 8000.0  # simulation duration (ms)
    warmup: float = 2000.0   # transient to discard (ms)
    noise_sigma: float = 1e-3
    seed: int = 42

    # Spectral targets
    target_freq_lo: float = 8.0    # alpha band lower (Hz)
    target_freq_hi: float = 13.0   # alpha band upper (Hz)
    min_amplitude: float = 0.1     # minimum oscillation std
    nperseg: int = 1024            # Welch PSD segment length


class CMCSpectralAdapter:
    """Fitness adapter that scores CMC parameters on spectral properties.

    Evaluates how well a set of CMC connectivity parameters produces
    physiologically plausible oscillatory dynamics (alpha-range, sufficient
    amplitude, stable).

    Fitness = alpha_power_fraction * amplitude_score * stability_bonus

    Parameters
    ----------
    config : CMCSpectralConfig
        Simulation and spectral target configuration.
    param_bounds : dict, optional
        Override default parameter bounds.
    """

    def __init__(
        self,
        config: CMCSpectralConfig | None = None,
        param_bounds: dict[str, tuple[float, float]] | None = None,
    ):
        self.config = config or CMCSpectralConfig()
        self._bounds = param_bounds or _CMC_SPECTRAL_BOUNDS
        self._n_steps = int(self.config.duration / self.config.dt)
        self._warmup_steps = int(self.config.warmup / self.config.dt)
        self._fs = 1000.0 / self.config.dt  # sampling freq in Hz

        # Pre-generate noise (fixed across evaluations for fair comparison)
        key = jax.random.PRNGKey(self.config.seed)
        self._zs = jax.random.normal(key, (self._n_steps, 8))

        # Pre-compile the simulation loop
        self._compile_loop()

    def _compile_loop(self):
        """JIT-compile the SDE loop once."""
        dt = self.config.dt
        sigma = self.config.noise_sigma

        def _dfun(y, p):
            return vbjax.cmc_dfun(y, 0.0, p)

        _, self._loop = vbjax.make_sde(dt=dt, dfun=_dfun, gfun=sigma)

    @property
    def parameter_space(self) -> dict[str, tuple[float, float]]:
        return dict(self._bounds)

    @property
    def objectives(self) -> list[ObjectiveSpec]:
        return [
            ObjectiveSpec("fc_correlation", "maximize", 1.0),
            ObjectiveSpec("alpha_power_frac", "maximize", 1.0),
            ObjectiveSpec("amplitude", "maximize", 0.5),
        ]

    def _params_to_theta(self, params: dict[str, float]) -> vbjax.CMCTheta:
        """Map optimizer params dict to CMCTheta, keeping PSP/sigmoid fixed."""
        base = vbjax.cmc_default_theta
        return base._replace(**{
            k: params[k] for k in params if k in base._fields
        })

    def _compute_spectral_fitness(
        self, sp: np.ndarray, all_states: np.ndarray
    ) -> dict[str, float]:
        """Score a simulation based on spectral properties."""
        from scipy.signal import welch

        metrics = {}

        # Check stability
        if not np.all(np.isfinite(sp)):
            return {"fc_correlation": -1.0, "alpha_power_frac": 0.0,
                    "amplitude": 0.0, "peak_freq": 0.0, "stable": 0.0}

        amplitude = float(np.std(sp))
        metrics["amplitude"] = amplitude
        metrics["stable"] = 1.0

        if amplitude < 1e-8:
            return {"fc_correlation": -1.0, "alpha_power_frac": 0.0,
                    "amplitude": 0.0, "peak_freq": 0.0, "stable": 1.0}

        # PSD via Welch
        f, pxx = welch(sp, fs=self._fs, nperseg=min(self.config.nperseg, len(sp)))

        # Alpha band power fraction
        alpha_mask = (f >= self.config.target_freq_lo) & (f <= self.config.target_freq_hi)
        total_power = np.sum(pxx[1:])  # exclude DC
        alpha_power = np.sum(pxx[alpha_mask]) if alpha_mask.any() else 0.0
        alpha_frac = alpha_power / max(total_power, 1e-20)
        metrics["alpha_power_frac"] = float(alpha_frac)

        # Peak frequency
        peak_idx = np.argmax(pxx[1:]) + 1
        metrics["peak_freq"] = float(f[peak_idx])

        # Amplitude score: sigmoid mapping, half-max at min_amplitude
        amp_score = float(np.tanh(amplitude / max(self.config.min_amplitude, 1e-10)))
        metrics["amplitude_score"] = amp_score

        # Check all populations are bounded (no blowup)
        max_state = float(np.max(np.abs(all_states)))
        bounded = 1.0 if max_state < 1e6 else 0.0
        metrics["bounded"] = bounded

        # Composite fitness: alpha fraction * amplitude * stability
        # Map to fc_correlation range [-1, 1] for protocol compatibility
        fitness = alpha_frac * amp_score * bounded
        # Scale: 0.5 alpha fraction with good amplitude → ~0.5 fitness
        metrics["fc_correlation"] = float(2.0 * fitness - 1.0)

        return metrics

    def evaluate(self, params: dict[str, float]) -> FitnessResult:
        """Run CMC simulation and score spectral properties."""
        t0 = time.perf_counter()

        theta = self._params_to_theta(params)
        y0 = jnp.zeros(8)

        try:
            ys = self._loop(y0, self._zs, theta)
            ys_np = np.array(ys)
        except Exception:
            return FitnessResult(
                fc_correlation=-1.0,
                fcd_ks_distance=1.0,
                wall_time=time.perf_counter() - t0,
                metadata={"error": "simulation_failed"},
            )

        # Extract superficial pyramidal after warmup
        sp = ys_np[self._warmup_steps:, 1]
        all_states = ys_np[self._warmup_steps:, :4]

        metrics = self._compute_spectral_fitness(sp, all_states)
        wall_time = time.perf_counter() - t0

        return FitnessResult(
            fc_correlation=metrics.get("fc_correlation", -1.0),
            fcd_ks_distance=1.0 - max(metrics.get("alpha_power_frac", 0.0), 0.0),
            raw_objectives=metrics,
            wall_time=wall_time,
            metadata={"theta": theta._asdict(), "peak_freq": metrics.get("peak_freq", 0)},
        )

    def evaluate_batch(
        self, params_batch: list[dict[str, float]]
    ) -> list[FitnessResult]:
        return [self.evaluate(p) for p in params_batch]

    # ── Gradient optimizer support ───────────────────────────────────

    def default_param_array(self) -> jnp.ndarray:
        """Midpoint of parameter bounds as a flat JAX array."""
        return jnp.array([
            (lo + hi) / 2 for lo, hi in self._bounds.values()
        ])

    def bounds_arrays(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Lower and upper bound arrays for clipping."""
        lo = jnp.array([b[0] for b in self._bounds.values()])
        hi = jnp.array([b[1] for b in self._bounds.values()])
        return lo, hi

    def _params_from_array(self, param_array: jnp.ndarray) -> dict[str, float]:
        """Convert flat parameter array back to named dict."""
        return {k: float(v) for k, v in zip(self._bounds.keys(), param_array)}

    def loss(self, param_array: jnp.ndarray) -> jnp.ndarray:
        """JAX-differentiable spectral loss for gradient-based optimization.

        Uses jnp.fft.rfft instead of scipy.signal.welch so the full
        pipeline is traceable by jax.grad.

        Lower is better (negative alpha power fraction + amplitude penalty).
        """
        param_names = list(self._bounds.keys())
        params_dict = {k: param_array[i] for i, k in enumerate(param_names)}
        theta = self._params_to_theta(params_dict)
        y0 = jnp.zeros(8)

        # Simulate (uses pre-compiled loop)
        ys = self._loop(y0, self._zs, theta)
        sp = ys[self._warmup_steps:, 1]

        # FFT-based power spectrum (JAX-traceable)
        n = sp.shape[0]
        fft_vals = jnp.fft.rfft(sp)
        psd = jnp.abs(fft_vals) ** 2 / n
        freqs = jnp.fft.rfftfreq(n, d=self.config.dt / 1000.0)

        # Alpha band power fraction
        alpha_mask = (freqs >= self.config.target_freq_lo) & (freqs <= self.config.target_freq_hi)
        alpha_power = jnp.sum(psd * alpha_mask)
        total_power = jnp.sum(psd[1:]) + 1e-12  # exclude DC
        alpha_frac = alpha_power / total_power

        # Amplitude term (encourage oscillation)
        amplitude = jnp.std(sp)
        amp_score = jnp.tanh(amplitude / self.config.min_amplitude)

        # Stability penalty (soft, via max state magnitude)
        max_state = jnp.max(jnp.abs(ys[self._warmup_steps:, :4]))
        stability = jnp.where(max_state < 1e5, 1.0, 0.01)

        # Minimize negative composite score
        return -(alpha_frac * amp_score * stability)
