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

## What is not there yet

- Parity against `spm_dcm_csd` (needs the SPM-Python oracle container).
- A canonical-microcircuit model and an NMDA/AMPA split (needed for the
  ketamine model of Muthukumaraswamy et al. 2015).
- Conduction delays between regions.
- Hierarchical (PEB-style) priors across subjects, studies and drugs.
