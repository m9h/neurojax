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

# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-01-28

### Added
- Initial release of NeuroJAX (OSL-JAX).
- JAX-accelerated GLM permutation testing.
- Documentation structure (`docs/`).
- RPM packaging specification.
