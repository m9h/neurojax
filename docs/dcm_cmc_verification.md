# DCM for cross-spectral density in JAX: port and verification against SPM 26

Written 2026-10-15. `neurojax.dcm.cmc_csd` is a line-by-line port of the
canonical-microcircuit CSD model in SPM's DCM for M/EEG: `spm_fx_cmc`
(flow), `spm_dcm_delay` (delay operator, Robbins-Monro Taylor scheme),
`spm_dcm_mtf` (transfer functions by eigen-solution with SPM's conditioning of
slow modes), `spm_csd_mtf_gu` (innovations with DCT-structured spectra,
specific and non-specific channel noise), `spm_gen_Q` (between-trial effects
on every extrinsic matrix, the modulatory matrix and the intrinsic gain),
`spm_lx_erp` / `spm_erp_L` for LFP electrodes, and the priors of
`spm_cmc_priors`, `spm_L_priors` and `spm_ssr_priors` as they stand in
SPM 26. The whole prediction is differentiable: the delay loops are
fixed-length masked scans and the eigen-decomposition has eigenvector
derivatives enabled.

## What SPM 26's suite tests, and what we reproduce

SPM 26 ships `tests/test_regress_spm_dcm_meeg.m` (107 parameterised tests;
the release notes say 117) over nine neural-mass models. Seven test kinds:

| SPM test | what it checks | ours (`tests/test_dcm_cmc.py`) |
|---|---|---|
| 1 | steady-state Jacobian stable (max real eigenvalue < 1e-9) | single and four-node CMC at prior mean, with and without the delay operator |
| 2 | model initialises at its steady state | `f(0, 0) = 0` for single and four-node |
| 3 | three integrators agree | not ported (CSD needs no integrator) |
| 4 | fits an ERP generated from its own priors | not ported (ERP branch) |
| 5 | fits five fixed ERP datasets against stored references | not ported (ERP branch; needs the test-data payload) |
| 6 | prior CSD finite and non-zero for single and four-node LFP networks | reproduced, plus Hermitian symmetry and positive auto-spectra |
| 7 | fits CSD to `LFP_BG.mat` against stored F/R2/Ep | pending: the payload is not public and the live oracle needs the MATLAB Runtime |

Added beyond the suite: the delay operator is checked to machine precision
against a plain transcription of SPM's loop algorithm (single and four-node,
non-trivial delays), `gen_Q` is checked against its definition, and the CSD
is checked to be differentiable in the intrinsic gains.

## The oracle problem

SPM's DCM-CSD example (`dLFP_white_noise_r24_anaes`, two-channel rodent LFP)
ships fitted DCMs with data cross-spectra, priors, posteriors, free energy and
predictions. They are not a valid reference for the current code: diffing
`spm_fx_cmc` between SPM8 and SPM 26 shows different intrinsic rates
(`G(3)` 4 vs 8), different delays (`[1 16]` vs `[1 8]` ms), a different input
convention (`C*u` vs `u*512` endogenous), and the SPM8 noise spectra carry a
factor 4 where the current ones carry `exp(-2)`. Our port reproduces the
stored predictions to 20–140%, which is the size of those changes. SPM 26's
own notes make the same point about SPM 25 versus 26 for the conductance
models.

The valid oracle is live SPM. `tests/test_dcm_cmc_oracle.py` runs
`spm_csd_mtf` through `spm-python` (SPM 25.01 compiled against the MATLAB
Runtime) for the single and four-node architectures of test 6, at the prior
mean and at a perturbed parameter set, and requires agreement to 1e-4 relative.
It is skipped unless `NEUROJAX_SPM_PYTHON` names an interpreter with the
runtime installed. Note that spm-python is SPM 25, not 26; the CSD forward
code did not change between them (the 26 changes are to `spm_nlsi_GN` and the
conductance models' fixed points), so it is a valid forward oracle.

## What this adds to the spectral-model set

The CMC CSD is the mechanistic model with a second resonance that the
LEMON and propofol comparisons showed the Liley mixture and ξ-α lack
(`docs/bsd_results.md`). Fitting it under the same free energy as the other
models is the next step; the port's parameters are already in SPM's units so
the priors can be used as-is.

## Oracle result (2026-10-15)

Live SPM 25.01 (spm-python 25.1.2 under MATLAB Runtime R2025b) was run for the
single- and four-node LFP architectures of test 6, at the prior mean and at a
perturbed parameter set (`G + 0.2`, `T - 0.1`, `D + 0.3`, `a + 0.5`), and its
cross-spectra saved to `tests/data/dcm/cmc_csd_spm25_oracle.mat`
(`scripts/spm_oracle/`). The JAX port reproduces every case to a maximum
relative error of 2.7e-10 over all frequencies, channels and trials, and its
priors equal SPM's exactly (`tests/test_dcm_cmc_oracle.py`, 8 tests, run
without SPM against the shipped file). That is the forward model of test 6
and the forward half of test 7 verified against the current equations; what
remains of test 7 is the inversion and the free energy, which need the
`LFP_BG` payload.

Two things cost most of a day and are worth recording. The compiled SPM
endpoint cannot return cells of complex N-D arrays, so the oracle runs as
evaluated MATLAB code and saves a MAT file. And on macOS the runtime must be
started with `-mlstartup -nojvm,-nodisplay`: without it the process exits
silently at the first compiled call, with or without a system JDK (a JDK
alone only removes macOS's "install a JDK" dialog); spm-python's launcher
documents the option but does not pass it, so `scripts/spm_oracle/
mwpython_launch.py` calls `mwpython` directly with the runtime's library
paths and the base interpreter that ships `libpython3.12.dylib`.

## What this adds to the spectral-model set

The CMC CSD is the mechanistic model with a second resonance that the
LEMON and propofol comparisons showed the Liley mixture and ξ-α lack
(`docs/bsd_results.md`). Fitting it under the same free energy as the other
models is the next step; the port's parameters are already in SPM's units so
the priors can be used as-is, and the forward model is now known to be
SPM's to ten significant figures.
