# Spectral DCM in JAX

`neurojax.inverse.csd_dcm` fits a generative model of the M/EEG
cross-spectral density (CSD) whose parameters are synaptic: the
"DCM for CSD" of Moran et al. (2009) and Friston et al. (2012), as in
`spm_dcm_csd`, written in JAX so the Jacobians used by variational Laplace
come from forward-mode autodiff and `jax.grad` flows from sensors to every
synaptic parameter.

## Generative model

1. **Neural mass.** A network of Jansen-Rit columns (6 states per region:
   pyramidal, excitatory and inhibitory interneuron potentials and their
   derivatives).  Region *j* drives the excitatory interneurons of region
   *i* with gain `A_forward[i, j]`.  The default mean afferent input
   (`p0 = 500/s`) puts each column on the upper stable branch of the
   Jansen-Rit bifurcation diagram, where the linearised column has a
   ~10 Hz inhibition-governed resonance (the 120-320/s range of Jansen &
   Rit 1995 is the limit cycle, which a linear CSD model cannot use).
2. **Linearisation.** `jr_fixed_point` solves for the fixed point by
   Newton iteration; `jr_linearise` returns `(A, B, C)` by autodiff of the
   vector field there.
3. **Transfer function.** `transfer_function` gives `H(ω) = C (iωI − A)⁻¹ B`.
4. **Spectra.** Source CSD `H(ω) diag(a_u f^-b_u) H(ω)ᴴ`, mapped by the
   leadfield to sensors, plus channel-specific and common `f^-b_n` noise.

All parameters are log-scalings of their defaults with Gaussian priors
(see the `SpectralDCM` docstring; the spectral nuisance priors are tight,
1/128, as in SPM).

## Pharmacological handles

| parameter | synaptic meaning | drug |
|---|---|---|
| `tau_i`, `Hi` | inhibitory (GABA-A) decay time and gain | propofol, benzodiazepines |
| `tau_e`, `He` | excitatory kinetics and gain | ketamine (NMDA), AMPA modulators |
| `C` | intrinsic connectivity | — |
| `A_forward` | between-region gains | psychedelics (5-HT2A gain), anaesthetics |

## Example

```python
import jax.numpy as jnp
from neurojax.inverse import JansenRitSpec, SpectralDCM, model_posterior

spec = JansenRitSpec(n_regions=2, A_forward=jnp.array([[0.0, 0.0], [100.0, 0.0]]))
dcm = SpectralDCM(spec, leadfield=L, freqs=jnp.arange(1.0, 45.5, 0.5))

placebo = dcm.invert(csd_placebo)      # csd: (n_freq, n_sensors, n_sensors) complex
drug = dcm.invert(csd_drug)
print(drug.summary()["tau_i[0]"])      # (posterior mean, sd) of the log-scaling

# Is the drug effect on the inhibitory time constant needed?  Reduced model:
reduced = dcm.with_fixed("tau_i").invert(csd_drug)
print(model_posterior([drug.F, reduced.F]))
```

`invert` runs variational Laplace from the prior mean and from three
draws of the prior over the synaptic parameters (`n_starts=4`), each with
a coarse-to-fine smoothing schedule along frequency, and keeps the start
with the highest free energy (`result.F_starts` lists all).  The free
energy of these models is multimodal — a lower inhibitory gain with a
shorter inhibitory time constant can mimic a longer one — so a single
Gauss-Newton run from the prior is not enough.

Data are fitted as log power (diagonal) and complex coherency
(off-diagonal); the data CSD is rescaled so that its mean power matches the
prior prediction (`result.scale`).  Noise-free synthetic data give an
unbounded noise precision and should be avoided.

## Free dipole moments

With a fixed-orientation leadfield the multi-source fits on the Chennu
propofol data misfit by about one log unit at every frequency alike (one
spatial mode under-, two others over-predicted): a spatial, not a spectral,
misfit.  Passing a `(n_sensors, n_regions, 3)` leadfield frees each
source's dipole moment, as the ECD moments of `spm_L_priors` do:

```python
dcm = SpectralDCM(CMCSpec(n_regions=4, ...), leadfield=L3, orientation=normals,
                  freqs=freqs, moment_var=1.0)
m = dcm.moments(result.Ep)            # (n_regions, 3) posterior-mean moments
```

The moment is `orientation + L[i, :]` with `L` additive (so it can rotate
and flip; prior mean 0, variance `moment_var` per component).  SPM uses a
zero prior mean with variance 64; here the given orientation is the prior
mean because a zero moment is a stationary point of the log-power
features.  Those features have a barrier wherever a sensor projection
`L·m` changes sign, so from the prior orientation Gauss-Newton stops in a
local optimum of the wrong sign pattern (F −914 against 1625 at the truth
in the unit test); `invert` therefore draws the starting moments from the
prior along with the neural-mass parameters.

## Simulating a fitted model

```python
x = result.simulate(fs=250.0, duration_s=60.0, key=jax.random.PRNGKey(0))             # linear-Gaussian, fitted CSD
x = result.simulate(fs=250.0, duration_s=60.0, key=key, nonlinear=True, amplitude=3.0)  # the network itself
```

The linear simulation is a Gaussian process with exactly the predicted
sensor cross-spectrum (Fourier synthesis; `tests/test_csd_dcm.py::TestSimulate`
checks it against a multitaper estimate).  The nonlinear one integrates
the network's vector field (Heun at 2 kHz) from its fixed point, driven by
innovations with the fitted 1/f spectrum, and decimates to `fs`.  A
cross-spectral fit does not identify the absolute amplitude of the states
(the data are rescaled on inversion), so `amplitude` sets how far the
dynamics reach into the sigmoid: as `amplitude → 0` the simulation has the
predicted CSD, larger values add the model's nonlinearity at the same
spectrum.  `scripts/consciousness_bench/dcm_simulate_measures.py` uses
this to ask whether the complexity-measure contrasts of the bench are
reproduced by the fitted linear model, by its nonlinearity at some
amplitude, or by neither.

## What is not there yet

- An NMDA/AMPA split (needed for the ketamine model of Muthukumaraswamy
  et al. 2015).
- Conduction delays between regions.
- SPM's CSD-side features and hyperpriors as an option (see below).

## Parity with SPM (checked 2026-10-25 with SPM-Python 25.1.2, MATLAB Runtime R2025b)

- **Neural mass.** `CMCSpec` at the prior gives the same Jacobian spectrum
  and the same transfer function as `spm_fx_cmc` / `spm_gx_erp` (to 1e-6;
  `tests/data/spm_cmc_prior_linearisation.npz`, exported on an x86 machine
  because the MATLAB Runtime is x86-only).  SPM's exogenous input enters
  with gain 512, now `CMCSpec.u_gain`.  SPM orders states voltage/current
  per population; neurojax voltages then currents.
- **Spectra (`spm_csd_mtf_gu`).** Neuronal innovations
  `Gu = exp(a₁) f^(−exp(a₂))` times a DCT modulation `exp(X d)` (4
  coefficients); common channel noise `Gn = exp(b₁ − 2) f^(−exp(b₂))`;
  specific channel noise `Gs = exp(c₁ − 2) f^(−exp(c₂))`; all priors
  N(0, 1/128); an optional frequency warp `exp(f₁ + f₂ k/n)` (prior 1/64).
  neurojax: `a_u` ≙ a₁ (prior 1/16), `b_u` ≙ a₂, `a_c` ≙ b₁, `a_n` ≙ c₁,
  `b_n` ≙ b₂ = c₂ (one shared channel-noise exponent); no DCT modulation
  and no warp; channel-noise defaults are `noise_fraction` of the prior
  neural power instead of e⁻².
- **Features and hyperpriors (`spm_dcm_csd`, `spm_fs_csd`).** SPM fits the
  complex CSD, the cross-covariance (every 8th lag) and the MAR(8)
  coefficients, with the data scaled so the largest cross-covariance is 8,
  `Nmodes = 8` spatial modes and a tight noise hyperprior (hE = 8,
  hC = 1/128).  neurojax fits log power and complex coherency with a weak
  hyperprior; the multi-start / smoothing continuation in `invert` has no
  SPM counterpart.  Full CSD-level parity therefore needs a feature-mode
  switch; the linearisation is where the pharmacology lives and is exact.
