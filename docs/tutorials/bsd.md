# Bayesian spectral decomposition: competing models of a power spectrum

`neurojax.bsd` fits parametric models of `ln S(f)` by variational Laplace and
compares them by free energy. It exists because three different generative
stories are told about the same resting EEG/MEG spectrum, and only an evidence
can adjudicate between them:

| model | hypothesis | parameters |
|---|---|---|
| `SpecParam` | aperiodic `offset − χ ln f` (or knee) plus Gaussian peaks in log power (Donoghue et al. 2020) | offset, exponent, [knee], per peak centre / height / width |
| `DampedOscillators` | K stochastically driven second-order oscillators plus a white floor | per oscillator frequency / damping / peak power, floor |
| `DampingDistribution` | one oscillator family with a log-normal spread of damping rates; the 1/f tail comes from the spread, not from a separate process (Liley) | centre, median damping, spread, peak power, floor |
| `XiAlpha` | a zero-frequency Matérn process ξ plus a frequency-shifted Matérn process α, two separate Hida-Matérn processes (Valdes-Sosa; ξ-α NET) | per process power / rate / smoothness, alpha centre, floor |
| `DampedMixture` | Liley's full picture: a damping-distributed background family plus one resonant alpha oscillator and a floor | background centre / damping / spread / power, alpha centre / damping / power, floor |
| `DampedMixture2` | `DampedMixture` plus a beta resonance whose centre is a ratio above alpha (no label swapping) | + beta ratio / damping / power |
| `CMCSpectrum` | canonical-microcircuit transfer function (SPM DCM-CSD priors), single source | 16 log-scaling parameters; inversion not yet reliable, see `docs/bsd_results.md` |

All positive quantities are carried as logs, so the Gaussian priors are
log-normal; peak heights in `SpecParam` are linear so that zero means "no peak"
and Bayesian model reduction can switch a peak off.

```python
import jax; jax.config.update("jax_enable_x64", True)   # evidence needs float64
import jax.numpy as jnp
from neurojax.bsd import (SpecParam, DampingDistribution, XiAlpha, variational_laplace,
                          psd_from_timeseries, posterior_model_probabilities,
                          random_effects_bms, bayesian_model_reduction)

freqs, log_psd, n_seg = psd_from_timeseries(x, fs=250.0, segment_s=2.0, f_lo=1.0, f_hi=45.0)

results = [variational_laplace(m, freqs, log_psd, restarts=2)
           for m in (SpecParam(), DampingDistribution(), XiAlpha())]
F = [r.free_energy for r in results]
print(posterior_model_probabilities(F), results[0].natural())
```

`variational_laplace` returns a `VLResult` with the posterior mean and
covariance (including the log-precision hyperparameter for the Gaussian
likelihood), the free energy, its accuracy/complexity decomposition, the fitted
spectrum and the residual. Use `likelihood="whittle", n_segments=n_seg` for the
exact Welch-estimate likelihood with no free noise parameter.

## Model evaluation

- `posterior_model_probabilities(F)` and `log_bayes_factors(F)` for one spectrum.
- `fixed_effects_bms(F)` sums free energies over subjects.
- `random_effects_bms(F)` is the Dirichlet variational scheme of Stephan et al.
  (2009) with exceedance probabilities, the Bayesian omnibus risk and
  protected exceedance probabilities of Rigoux et al. (2014).
- `bayesian_model_reduction(result, reduced_prior_mean, reduced_prior_cov)`
  gives the analytic change in evidence for a reduced prior (Friston et al.
  2016). It is accurate when the reduced model lies inside the full posterior's
  quadratic region; when it removes a strong component it only bounds the
  penalty, which the tests check against direct inversion.
- `bayesian_model_average(results)` averages parameters across models by
  posterior probability, matching by name.

## What the tests establish

With Welch estimation noise from 32 segments over 1–45 Hz, each model recovers
its own generator, and in a 3 × 3 × 3 confusion (generators × models ×
realisations) the generating model wins by fixed effects with probability above
0.9 and never loses a single realisation by more than 3 nats. The specparam
phenomenology and the ξ-α two-process model are the closest pair; the
damping-distribution model is the most distinct.

## Relation to SPM

This is the construction of SPM's BSD toolbox (Medrano et al. 2025, included in
SPM 26) with the model set widened from "aperiodic plus peaks" to the two
mechanistic alternatives, and written so that `jax.grad` flows through the
free energy. The variational Laplace here uses BFGS on the log joint and the
Hessian at the mode; SPM's `spm_nlsi_GN` uses Gauss-Newton with a different
step, so free energies will not agree to the nat with SPM, as SPM 26's own
notes say of its change from SPM 25.
