"""Parametric models of ``ln S(f)`` with Gaussian priors on unconstrained parameters.

Every model exposes the same small interface so that inversion and
comparison code never sees the model's internals:

``names``            parameter names in the unconstrained vector
``prior()``          ``(mean, sd)`` of the Gaussian prior, unconstrained space
``log_psd(f, theta)`` natural-log power at frequencies ``f`` (Hz)
``natural(theta)``   dict of parameters on their natural scale

Positive quantities are carried as logarithms, so a Gaussian prior on the
vector is log-normal on the quantity; centre frequencies are carried
linearly in Hz.  Frequencies are in Hz throughout; angular frequency is
formed inside the model.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

TWO_PI = 2.0 * np.pi


class SpectralModel(eqx.Module):
    """Base class; subclasses set ``names`` and implement ``log_psd``."""

    names: tuple[str, ...] = eqx.field(static=True)
    prior_mean: jnp.ndarray
    prior_sd: jnp.ndarray

    @property
    def n_params(self) -> int:
        return len(self.names)

    def prior(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.prior_mean, self.prior_sd

    def log_psd(self, freqs, theta):  # pragma: no cover - abstract
        raise NotImplementedError

    def natural(self, theta) -> dict:
        return {n: float(v) for n, v in zip(self.names, theta)}

    def index(self, name: str) -> int:
        return self.names.index(name)


# ---------------------------------------------------------------------------
# specparam: aperiodic + Gaussian peaks in log power
# ---------------------------------------------------------------------------

class SpecParam(SpectralModel):
    """``ln S = offset - ln(knee + f^chi) + sum_k pw_k exp(-(f - cf_k)^2 / 2 bw_k^2)``.

    ``knee=False`` gives the "fixed" aperiodic ``offset - chi ln f``.  Peaks
    are Gaussians in log power as in specparam; their prior centre
    frequencies are supplied so that multi-peak fits are not label-symmetric.
    Peak centres are mapped onto ``centre_bounds`` by a logistic so a peak
    cannot wander below the fitted band to mimic aperiodic curvature.
    The peak height ``pw`` is carried *linearly* (not as a log) so that zero
    means "no peak": that is what lets Bayesian model reduction switch a
    peak off by shrinking its prior to zero, as DCM does with coupling
    parameters.  A negative height is a dip and is admissible.
    """

    knee: bool = eqx.field(static=True)
    n_peaks: int = eqx.field(static=True)
    centre_bounds: tuple = eqx.field(static=True)

    def __init__(self, peak_centres=(10.0,), knee=False, exponent_prior=(0.0, 0.5),
                 centre_sd=2.0, power_prior=(0.5, 1.0), width_prior=(np.log(2.0), 0.5),
                 centre_bounds=(3.0, 40.0)):
        self.knee = bool(knee)
        self.n_peaks = len(peak_centres)
        self.centre_bounds = (float(centre_bounds[0]), float(centre_bounds[1]))
        names = ["offset", "log_exponent"] + (["log_knee"] if knee else [])
        mean = [0.0, exponent_prior[0]] + ([np.log(1.0)] if knee else [])
        sd = [3.0, exponent_prior[1]] + ([1.0] if knee else [])
        lo, hi = self.centre_bounds
        for k, cf in enumerate(peak_centres):
            names += [f"cf{k}", f"pw{k}", f"log_bw{k}"]
            # The centre is carried through a logistic map onto (lo, hi); the prior
            # SD in Hz is converted to the unconstrained scale at the prior centre.
            u = float(np.log((cf - lo) / (hi - cf)))
            dcf_du = (cf - lo) * (hi - cf) / (hi - lo)
            mean += [u, power_prior[0], width_prior[0]]
            sd += [centre_sd / dcf_du, power_prior[1], width_prior[1]]
        self.names = tuple(names)
        self.prior_mean = jnp.asarray(mean, dtype=float)
        self.prior_sd = jnp.asarray(sd, dtype=float)

    def _centre(self, u):
        lo, hi = self.centre_bounds
        return lo + (hi - lo) * jax.nn.sigmoid(u)

    def log_psd(self, freqs, theta):
        f = jnp.asarray(freqs, dtype=float)
        offset, chi = theta[0], jnp.exp(theta[1])
        i = 2
        if self.knee:
            knee = jnp.exp(theta[2]); i = 3
            out = offset - jnp.log(knee + f ** chi)
        else:
            out = offset - chi * jnp.log(f)
        for k in range(self.n_peaks):
            cf, pw, bw = self._centre(theta[i]), theta[i + 1], jnp.exp(theta[i + 2])
            out = out + pw * jnp.exp(-((f - cf) ** 2) / (2.0 * bw ** 2))
            i += 3
        return out

    def natural(self, theta):
        d = {"offset": float(theta[0]), "exponent": float(jnp.exp(theta[1]))}
        i = 2
        if self.knee:
            d["knee"] = float(jnp.exp(theta[2])); i = 3
        for k in range(self.n_peaks):
            d[f"cf{k}"] = float(self._centre(theta[i])); d[f"pw{k}"] = float(theta[i + 1])
            d[f"bw{k}"] = float(jnp.exp(theta[i + 2])); i += 3
        return d


# ---------------------------------------------------------------------------
# Liley: stochastically driven damped oscillators
# ---------------------------------------------------------------------------

def _oscillator_psd(omega, omega0, gamma, peak_power):
    """Power of ``x'' + 2 gamma x' + omega0^2 x = sigma dW`` with the drive expressed as the power at resonance.

    ``S = sigma^2 / ((omega0^2 - omega^2)^2 + 4 gamma^2 omega^2)`` and
    ``S(omega0) = sigma^2 / (4 gamma^2 omega0^2)``, so ``sigma^2 = peak_power * 4 gamma^2 omega0^2``.
    Parameterising by ``peak_power`` keeps the prior on a human scale (power
    units of the spectrum) instead of ``omega0^4`` units.
    """
    sigma2 = peak_power * 4.0 * gamma ** 2 * omega0 ** 2
    return sigma2 / ((omega0 ** 2 - omega ** 2) ** 2 + 4.0 * gamma ** 2 * omega ** 2)


class DampedOscillators(SpectralModel):
    """Sum of ``n_osc`` second-order stochastic oscillators plus a white floor.

    Parameters per oscillator: natural frequency (Hz, log), damping rate
    ``gamma`` (1/s, log) and power at resonance (log).  With one oscillator and
    the floor this is the ARMA(2,·) picture of Liley & Muthukumaraswamy
    (2019): alpha blocking is an increase of ``gamma``.
    """

    n_osc: int = eqx.field(static=True)

    def __init__(self, centres=(10.0,), damping_prior=(np.log(TWO_PI * 1.0), 1.0),
                 centre_sd=0.3, drive_prior=(0.0, 3.0), floor_prior=(-3.0, 3.0)):
        self.n_osc = len(centres)
        names, mean, sd = [], [], []
        for k, c in enumerate(centres):
            names += [f"log_f{k}", f"log_gamma{k}", f"log_peak{k}"]
            mean += [np.log(float(c)), damping_prior[0], drive_prior[0]]
            sd += [centre_sd, damping_prior[1], drive_prior[1]]
        names.append("log_floor"); mean.append(floor_prior[0]); sd.append(floor_prior[1])
        self.names = tuple(names)
        self.prior_mean = jnp.asarray(mean, dtype=float)
        self.prior_sd = jnp.asarray(sd, dtype=float)

    def log_psd(self, freqs, theta):
        omega = TWO_PI * jnp.asarray(freqs, dtype=float)
        total = jnp.exp(theta[-1]) * jnp.ones_like(omega)
        for k in range(self.n_osc):
            f0, gamma, drive = (jnp.exp(theta[3 * k]), jnp.exp(theta[3 * k + 1]), jnp.exp(theta[3 * k + 2]))
            total = total + _oscillator_psd(omega, TWO_PI * f0, gamma, drive)
        return jnp.log(total)

    def natural(self, theta):
        d = {}
        for k in range(self.n_osc):
            d[f"f{k}"] = float(jnp.exp(theta[3 * k])); d[f"gamma{k}"] = float(jnp.exp(theta[3 * k + 1]))
            d[f"peak{k}"] = float(jnp.exp(theta[3 * k + 2]))
        d["floor"] = float(jnp.exp(theta[-1]))
        return d


class DampingDistribution(SpectralModel):
    """One oscillator family with a log-normal distribution of damping rates.

    The Liley hypothesis for the aperiodic background: a population of
    alpha-band relaxation oscillators whose damping rates are spread, so the
    superposition has a 1/f-like tail without a separate aperiodic process.
    The mixture is integrated by Gauss-Hermite quadrature over ``ln gamma``.
    Parameters: centre frequency (log Hz), mean and spread of ``ln gamma``,
    power at resonance of the median-damped member (log), white floor (log).
    """

    n_quad: int = eqx.field(static=True)
    nodes: jnp.ndarray
    weights: jnp.ndarray

    def __init__(self, centre=10.0, n_quad=24, centre_sd=0.3,
                 damping_mean_prior=(np.log(TWO_PI * 1.0), 1.0), spread_prior=(np.log(0.7), 0.7),
                 drive_prior=(0.0, 3.0), floor_prior=(-3.0, 3.0)):
        self.n_quad = int(n_quad)
        x, w = np.polynomial.hermite.hermgauss(self.n_quad)
        self.nodes = jnp.asarray(x, dtype=float)
        self.weights = jnp.asarray(w / np.sqrt(np.pi), dtype=float)
        self.names = ("log_f0", "mu_log_gamma", "log_spread", "log_peak", "log_floor")
        self.prior_mean = jnp.asarray([np.log(float(centre)), damping_mean_prior[0], spread_prior[0],
                                       drive_prior[0], floor_prior[0]], dtype=float)
        self.prior_sd = jnp.asarray([centre_sd, damping_mean_prior[1], spread_prior[1],
                                     drive_prior[1], floor_prior[1]], dtype=float)

    def log_psd(self, freqs, theta):
        omega = TWO_PI * jnp.asarray(freqs, dtype=float)
        omega0 = TWO_PI * jnp.exp(theta[0])
        mu, s, drive, floor = theta[1], jnp.exp(theta[2]), jnp.exp(theta[3]), jnp.exp(theta[4])
        log_gammas = mu + jnp.sqrt(2.0) * s * self.nodes
        gammas = jnp.exp(log_gammas)
        comps = jax.vmap(lambda g: _oscillator_psd(omega, omega0, g, drive))(gammas)
        return jnp.log(floor + jnp.sum(self.weights[:, None] * comps, axis=0))

    def natural(self, theta):
        return {"f0": float(jnp.exp(theta[0])), "gamma_median": float(jnp.exp(theta[1])),
                "spread": float(jnp.exp(theta[2])), "peak": float(jnp.exp(theta[3])),
                "floor": float(jnp.exp(theta[4]))}


# ---------------------------------------------------------------------------
# Valdes-Sosa: xi (aperiodic Matérn) + alpha (frequency-shifted Matérn)
# ---------------------------------------------------------------------------

def _matern_psd(omega, peak_power, lam, nu):
    """Matérn spectral density ``A (lam^2 + omega^2)^-(nu + 1/2)`` with ``A`` set by the power at ``omega = 0``.

    ``nu = 1/2`` is the Lorentzian of an OU process; larger ``nu`` is smoother.
    ``peak_power = A lam^-(2 nu + 1)`` so the prior is in spectrum units.
    """
    amplitude = peak_power * lam ** (2.0 * nu + 1.0)
    return amplitude * (lam ** 2 + omega ** 2) ** (-(nu + 0.5))


class XiAlpha(SpectralModel):
    """``S = S_xi + S_alpha``: a zero-frequency Matérn process and a shifted one.

    ``S_xi(omega) = A_xi (lam_xi^2 + omega^2)^-(nu_xi + 1/2)`` is the aperiodic
    ξ process; ``S_alpha`` is the same form centred at ``±omega_alpha`` (a
    damped oscillator's Matérn generalisation).  Both are rational spectra,
    i.e. Hida-Matérn processes with linear state-space representations, which
    is the ξ-α NET construction.  Parameters (all log except as noted):
    zero-lag powers, length rates ``lam`` (rad/s), smoothness ``nu``, and the
    alpha centre frequency (log Hz).
    """

    def __init__(self, centre=10.0, centre_sd=0.3, nu_prior=(np.log(1.0), 0.5),
                 xi_rate_prior=(np.log(TWO_PI * 1.0), 1.0), alpha_rate_prior=(np.log(TWO_PI * 1.0), 1.0),
                 amplitude_prior=(0.0, 4.0), floor_prior=(-3.0, 3.0)):
        self.names = ("log_P_xi", "log_lam_xi", "log_nu_xi",
                      "log_P_alpha", "log_lam_alpha", "log_nu_alpha", "log_f_alpha", "log_floor")
        self.prior_mean = jnp.asarray([amplitude_prior[0], xi_rate_prior[0], nu_prior[0],
                                       amplitude_prior[0], alpha_rate_prior[0], nu_prior[0],
                                       np.log(float(centre)), floor_prior[0]], dtype=float)
        self.prior_sd = jnp.asarray([amplitude_prior[1], xi_rate_prior[1], nu_prior[1],
                                     amplitude_prior[1], alpha_rate_prior[1], nu_prior[1],
                                     centre_sd, floor_prior[1]], dtype=float)

    def log_psd(self, freqs, theta):
        omega = TWO_PI * jnp.asarray(freqs, dtype=float)
        a_xi, lam_xi, nu_xi = jnp.exp(theta[0]), jnp.exp(theta[1]), jnp.exp(theta[2])
        a_al, lam_al, nu_al = jnp.exp(theta[3]), jnp.exp(theta[4]), jnp.exp(theta[5])
        omega_al, floor = TWO_PI * jnp.exp(theta[6]), jnp.exp(theta[7])
        s_xi = _matern_psd(omega, a_xi, lam_xi, nu_xi)
        s_al = _matern_psd(omega - omega_al, a_al, lam_al, nu_al) + _matern_psd(omega + omega_al, a_al, lam_al, nu_al)
        return jnp.log(floor + s_xi + s_al)

    def natural(self, theta):
        return {"P_xi": float(jnp.exp(theta[0])), "lam_xi": float(jnp.exp(theta[1])), "nu_xi": float(jnp.exp(theta[2])),
                "P_alpha": float(jnp.exp(theta[3])), "lam_alpha": float(jnp.exp(theta[4])),
                "nu_alpha": float(jnp.exp(theta[5])), "f_alpha": float(jnp.exp(theta[6])),
                "floor": float(jnp.exp(theta[7]))}


class DampedMixture(SpectralModel):
    """Liley's full picture: a damping-distributed background family plus one resonant oscillator.

    Muthukumaraswamy & Liley (2018) explain the 1/f background by many
    relaxation processes with a spread of rates and the alpha peak by a
    resonant oscillator; :class:`DampingDistribution` keeps only one family
    and so has to choose between the two jobs.  This model has both: a
    background family with a wide damping spread (its centre frequency and
    peak power free) and a single :func:`_oscillator_psd` resonance, plus a
    white floor.  Eight parameters, the same count as :class:`XiAlpha`, so
    the two mechanistic hypotheses are compared at equal complexity.
    """

    n_quad: int = eqx.field(static=True)
    nodes: jnp.ndarray
    weights: jnp.ndarray

    def __init__(self, centre=10.0, n_quad=24, centre_sd=0.3, background_centre=3.0,
                 bg_damping_prior=(np.log(TWO_PI * 2.0), 1.0), bg_spread_prior=(np.log(1.0), 0.7),
                 alpha_damping_prior=(np.log(TWO_PI * 1.0), 1.0), peak_prior=(0.0, 3.0), floor_prior=(-3.0, 3.0)):
        self.n_quad = int(n_quad)
        x, w = np.polynomial.hermite.hermgauss(self.n_quad)
        self.nodes = jnp.asarray(x, dtype=float)
        self.weights = jnp.asarray(w / np.sqrt(np.pi), dtype=float)
        self.names = ("log_f_bg", "mu_log_gamma_bg", "log_spread_bg", "log_peak_bg",
                      "log_f_alpha", "log_gamma_alpha", "log_peak_alpha", "log_floor")
        self.prior_mean = jnp.asarray([np.log(float(background_centre)), bg_damping_prior[0], bg_spread_prior[0], peak_prior[0],
                                       np.log(float(centre)), alpha_damping_prior[0], peak_prior[0], floor_prior[0]], dtype=float)
        self.prior_sd = jnp.asarray([0.7, bg_damping_prior[1], bg_spread_prior[1], peak_prior[1],
                                     centre_sd, alpha_damping_prior[1], peak_prior[1], floor_prior[1]], dtype=float)

    def log_psd(self, freqs, theta):
        omega = TWO_PI * jnp.asarray(freqs, dtype=float)
        w_bg = TWO_PI * jnp.exp(theta[0])
        mu, s, p_bg = theta[1], jnp.exp(theta[2]), jnp.exp(theta[3])
        gammas = jnp.exp(mu + jnp.sqrt(2.0) * s * self.nodes)
        bg = jnp.sum(self.weights[:, None] * jax.vmap(lambda g: _oscillator_psd(omega, w_bg, g, p_bg))(gammas), axis=0)
        alpha = _oscillator_psd(omega, TWO_PI * jnp.exp(theta[4]), jnp.exp(theta[5]), jnp.exp(theta[6]))
        return jnp.log(jnp.exp(theta[7]) + bg + alpha)

    def natural(self, theta):
        return {"f_bg": float(jnp.exp(theta[0])), "gamma_bg_median": float(jnp.exp(theta[1])),
                "spread_bg": float(jnp.exp(theta[2])), "peak_bg": float(jnp.exp(theta[3])),
                "f_alpha": float(jnp.exp(theta[4])), "gamma_alpha": float(jnp.exp(theta[5])),
                "peak_alpha": float(jnp.exp(theta[6])), "floor": float(jnp.exp(theta[7]))}


class DampedMixture2(DampedMixture):
    """Liley mixture with a second resonance (beta): background family + alpha + beta oscillators + floor, 11 parameters."""

    def __init__(self, centre=10.0, beta_centre=20.0, n_quad=24, centre_sd=0.3, background_centre=3.0,
                 bg_damping_prior=(np.log(TWO_PI * 2.0), 1.0), bg_spread_prior=(np.log(1.0), 0.7),
                 alpha_damping_prior=(np.log(TWO_PI * 1.0), 1.0), peak_prior=(0.0, 3.0), floor_prior=(-3.0, 3.0)):
        super().__init__(centre, n_quad, centre_sd, background_centre, bg_damping_prior, bg_spread_prior,
                         alpha_damping_prior, peak_prior, floor_prior)
        # the beta centre is parametrised as a ratio above alpha, f_beta = f_alpha * (1 + exp(theta)),
        # so the two resonances can never swap labels (the first batch swapped 13 of 60 LEMON rows)
        self.names = self.names + ("log_beta_ratio", "log_gamma_beta", "log_peak_beta")
        ratio0 = np.log(float(beta_centre) / float(centre) - 1.0)
        self.prior_mean = jnp.concatenate([self.prior_mean, jnp.asarray([ratio0, alpha_damping_prior[0], peak_prior[0] - 1.0])])
        self.prior_sd = jnp.concatenate([self.prior_sd, jnp.asarray([2.0 * centre_sd, alpha_damping_prior[1], peak_prior[1]])])

    def f_beta(self, theta):
        return jnp.exp(theta[4]) * (1.0 + jnp.exp(theta[8]))

    def log_psd(self, freqs, theta):
        base = jnp.exp(super().log_psd(freqs, theta[:8]))
        omega = TWO_PI * jnp.asarray(freqs, dtype=float)
        beta = _oscillator_psd(omega, TWO_PI * self.f_beta(theta), jnp.exp(theta[9]), jnp.exp(theta[10]))
        return jnp.log(base + beta)

    def natural(self, theta):
        d = super().natural(theta[:8])
        d.update({"f_beta": float(self.f_beta(theta)), "gamma_beta": float(jnp.exp(theta[9])), "peak_beta": float(jnp.exp(theta[10]))})
        return d
