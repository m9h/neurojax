## Unreleased

### Added
- `neurojax.bsd`: Bayesian spectral decomposition. Four parametric models of
  `ln S(f)` (`SpecParam`, `DampedOscillators`, `DampingDistribution` for the
  Liley damping-distribution hypothesis, `XiAlpha` for the Valdes-Sosa
  two-process hypothesis), Gaussian-log and Whittle likelihoods, variational
  Laplace inversion with free energy and accuracy/complexity, and model
  evaluation tools: Bayes factors, fixed- and random-effects BMS with protected
  exceedance probabilities, Bayesian model reduction and model averaging.
  Tutorial in `docs/tutorials/bsd.md`; 16 tests in `tests/test_bsd.py`.
- `neurojax.bsd` mechanistic mixtures: `DampedMixture` (Liley background family +
  resonant alpha) and `DampedMixture2` (adds a beta resonance parametrised as a
  ratio above alpha so the labels cannot swap); `CMCSpectrum` (canonical
  microcircuit transfer-function spectrum, SPM priors; inversion not yet
  reliable); `variational_laplace(prefit=False)`.
- `neurojax.bsd.multiway`: spectral tensors, HOSVD, Marchenko-Pastur HOSVD
  denoising, non-negative PARAFAC (HALS), `tucker_energy_denoise` (trilinear
  basis at a fixed explained-variance fraction).
- `neurojax.dcm.cmc_csd`: JAX port of SPM's DCM for cross-spectral densities
  (canonical microcircuit, delays, transfer functions, neuronal and channel
  noise), verified against SPM 25.01 to 3e-10 relative error with a stored
  oracle (`tests/data/dcm/`, `scripts/spm_oracle/`).
- `scripts/bsd_batch.py` (dataset adapters: LEMON, Chennu propofol, ds005620,
  ds004541, HBN rest/task, NIMH; `--reduce`, `--refit`), `scripts/bsd_summarise.py`
  (FFX/RFX selection, parameter tables, within-subject contrasts); results in
  `docs/bsd_results.md`, DCM verification in `docs/dcm_cmc_verification.md`.

# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-01-28

### Added
- Initial release of NeuroJAX (OSL-JAX).
- JAX-accelerated GLM permutation testing.
- Documentation structure (`docs/`).
- RPM packaging specification.
