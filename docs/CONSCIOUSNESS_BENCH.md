# Consciousness complexity benchmark

Reproduces, in NeuroJAX, the McGill/Milan/Liège comparison of complexity and
criticality measures on resting EEG under propofol, xenon and ketamine
(Maschke, O'Byrne, ... Sarasso, Gosseries, Blain-Moraes et al., *Commun. Biol.*
7:946, 2024; code `BIAPT/Criticality_PCI_Anesthesia`, `jnobyrne/edgeofpy`) and
the complexity-entropy causal plane follow-up (Newman, Maschke, Mashour,
Blain-Moraes, medRxiv 2025.12.16.25342405), and extends the same measure set
to psychedelic EEG.

What the reference comparison lacks, and this benchmark adds, is the
Mediano/Rosas/Barrett/Bor spectral-vs-phasic decomposition of every measure's
between-condition change: how much of each "complexity" effect is reproduced
by phase-randomised surrogates with the same spectrum.

## Code

| Piece | Where |
|---|---|
| Measure registry (30 named estimators, parameters, references) | `src/neurojax/analysis/nltsa.py` (`MEASURES`) |
| Batched GPU versions of the O(T²) measures (recurrence profile, Lyapunov, correlation dimension, 0-1 chaos) | `src/neurojax/analysis/nltsa_batched.py` (`BATCHED`, `score_batched`); used automatically by `bench.consciousness_bench.score_epochs` when the default JAX device is an accelerator |
| LZ variants + spectral/phasic decomposition | `src/neurojax/analysis/complexity.py` |
| Harness: epoching, scoring, paired contrasts, decomposition | `src/neurojax/bench/consciousness_bench.py` |
| Spectral baselines: specparam/FOOOF, band powers, SEF95, alpha peak | `src/neurojax/analysis/spectral_baselines.py` (`SPECTRAL_MEASURES`) |
| Reference-library parity (antropy, NeuroKit2, nolds) | `src/neurojax/analysis/tests/test_nltsa_oracles.py` |
| Dataset loaders (skip if missing) | `src/neurojax/io/consciousness_bench.py` |
| Downloads | `scripts/consciousness_bench/download_datasets.sh` |
| Driver (Slurm) | `scripts/consciousness_bench/run_bench.py` (`--measures all|nltsa|spectral`, `--tag`), `run_bench.sbatch` (cpu partition, 20 workers) or `gpu.sbatch <script.py> …` (GB10 + 4 CPU workers; `gpu_check.py` validates parity and timing) |
| Spectral-explained analysis | `scripts/consciousness_bench/spectral_explained.py` → `bench.consciousness_bench.spectral_explained` |
| Tests | `src/neurojax/analysis/tests/test_nltsa.py`, `test_nltsa_batched.py` (batched = scalar numpy), `src/neurojax/bench/tests/test_consciousness_bench.py`, `src/neurojax/io/tests/test_consciousness_bench_io.py` |

## Measures

| name | family | estimator / parameters | reference |
|---|---|---|---|
| lz76_mean | lempel-ziv | LZ76, binarised at mean, ÷ T/log₂T | Maschke 2024 |
| lz76_envelope | lempel-ziv | LZ76, Hilbert envelope > median | Schartner 2017 |
| lz78_median | lempel-ziv | LZ78 dictionary size, median | Lempel & Ziv 1978 |
| permutation_entropy | ordinal | m=3, τ=1, ÷ log m! | Bandt & Pompe 2002 |
| statistical_complexity | ordinal | Jensen-Shannon C = Q_J·H, m=3 | Rosso 2007; Newman 2025 |
| sample_entropy | entropy | m=2, r=0.2 SD | Richman & Moorman 2000 |
| mse_sum | entropy | sample entropy summed over scales 1–20, r=0.15 SD of original | Costa 2002; Maschke 2024 |
| spectral_entropy | entropy | Shannon entropy of normalised PSD | Inouye 1991 |
| svd_entropy | entropy | τ=2, d=10 embedding singular spectrum | Roberts 1999 |
| katz_fd | fractal | | Katz 1988; Maschke 2024 |
| higuchi_fd | fractal | k_max=10 | Higuchi 1988 |
| petrosian_fd | fractal | | Petrosian 1995 |
| dfa | scaling | linear detrend, windows 4…T/4 | Peng 1994 |
| dfa_alpha_envelope | scaling | DFA of 8–13 Hz envelope, 1–20 s | Linkenkaer-Hansen 2001; Maschke 2024 |
| aperiodic_exponent | spectral | log-log Welch fit 1–40 Hz (no peak model) | Donoghue 2020 |
| hjorth_mobility, hjorth_complexity | spectral | | Hjorth 1970 |
| chaos_01 | chaos | modified 0-1 test, median K over 100 c | Gottwald & Melbourne 2009; Maschke 2024 |
| lyapunov | chaos | Rosenstein, m=2, τ=1, Theiler fs/10, 20 steps | Rosenstein 1993; Maschke 2024 |
| correlation_dimension | chaos | Grassberger-Procaccia, m=5 | GP 1983 |
| rqa_determinism, rqa_laminarity, rqa_trapping_time, rqa_mean_diag_length, rqa_diag_entropy, rqa_mean_recurrence_time | recurrence | m=3, τ=1, Euclidean, RR=5 %, l_min=v_min=2; vectorised line counting, pyunicorn conventions (verified against pyunicorn 1.0) | Marwan 2007; Ngamga 2007 |
| rn_transitivity, rn_global_clustering | recurrence network | recurrence matrix as adjacency, self-loops removed | Donges, Marwan, Kurths 2011/2012 |
| pair_correlation | criticality (multivariate) | N·Var|R(t)|, alpha band | Yoon 2015; Maschke 2024 |
| branching_ratio | criticality (multivariate) | |z|≥2 SD events, 4 ms bins | Beggs & Plenz 2003; Maschke 2024 |

### Spectral baselines (`SPECTRAL_MEASURES`)

| name | estimator |
|---|---|
| specparam_exponent, specparam_offset, specparam_r2 | specparam 2.0rc7, fixed aperiodic mode, 1–40 Hz, Welch 2 s segments, max 6 peaks |
| specparam_alpha_cf, specparam_alpha_power | strongest fitted peak in 7–13 Hz (power over the aperiodic fit; 0 / NaN when none) |
| relpow_delta … relpow_gamma | Welch relative power, bands 1–4, 4–8, 8–13, 13–30, 30–45 Hz, fraction of 1–45 Hz |
| log_total_power | log10 Welch power 1–45 Hz |
| spectral_edge_95 | SEF95 below 45 Hz |
| alpha_peak_frequency | raw PSD argmax in 7–13 Hz |

`spectral_explained(scores, spectral_scores, a, b)` regresses each NLTSA
measure (subject × condition values) on z-scored specparam exponent, offset,
alpha power and relative delta/theta/beta power and reports `r2`, the paired
Cohen's d of the raw measure and of the residual. A measure with high `r2` and
`d_residual ≈ 0` is a re-expression of the FOOOF/band-power parameters.

### Reference-library parity

`test_nltsa_oracles.py` checks the neurojax estimators against antropy 0.2.3
(permutation, sample, spectral and SVD entropy, Higuchi/Petrosian/Katz FD,
LZ76, DFA), NeuroKit2 0.2.13 (Hjorth) and nolds (Rosenstein Lyapunov,
correlation dimension), with the tolerance and the algorithmic reason for any
difference stated in the test. Katz FD follows the antropy/NeuroKit
amplitude-only convention (scale-free); DFA differs from antropy only in the
window set.

### Peers, not add-ons (roadmap)

Recurrence methods, SINDy and DMD are to be treated as full peers of the
scalar-measure set in the pharmaco-EEG and EEG-fMRI modelling, not as extra
columns:

- **Recurrence** (pyunicorn as oracle): the RQA/recurrence-network family is
  in the registry; still to add are *twin surrogates* (Thiel 2006) as a
  recurrence-preserving null to set beside the Prichard-Theiler spectral
  null in the decomposition, cross-/joint-recurrence between channel pairs
  (multivariate measures), and for EEG-fMRI the joint recurrence of the EEG
  state with the BOLD state (pyunicorn `JointRecurrencePlot`,
  `InterSystemRecurrenceNetwork`).
- **SINDy** (weak-form + ensemble, PySINDy): fit a low-dimensional
  generative model per subject × condition on reduced state (PCA/DMD modes,
  band envelopes + phase, or connectome-harmonic coefficients), compare
  conditions on coefficient inclusion probabilities and linearisation
  eigenvalues, and simulate from the fitted model to compute every scalar
  measure — the generative counterpart of the surrogate decomposition.
  Unfiltered data + weak-form SINDy makes drift a state variable instead of
  something to filter.
- **AI Institute in Dynamic Systems (Brunton/Kutz) methods to add**: SHRED
  (shallow recurrent decoder; SINDy-SHRED for sparse latent dynamics from a
  few sensors), HAVOK (Hankel alternative view of Koopman, delay-embedded
  DMD + forcing), SINDy autoencoders (Champion 2019), PyKoopman, PySensors
  (optimal sensor/channel selection).
- **DMD / Koopman**: `bench.sensitivity.dmd_lowmode_removal` is the
  low-frequency treatment; the same Hankel-DMD spectrum per epoch (mode
  frequencies, growth rates, mode amplitudes) is the modal description of
  the state and belongs in the measure table beside the specparam fit.

### Drug time course (roadmap)

- **PK/PD**: effect-site concentration (Sheiner k_e0 hysteresis) from
  plasma or infusion schedule, sigmoid Emax per measure; compare measures by
  PD consistency (EC50/hysteresis across subjects). Anchors: propofol
  slow-wave saturation (Ní Mhuircheartaigh 2013, Warnaby 2017); DMT plasma +
  real-time intensity (Timmermann 2019/2023); 5-HT2A occupancy (Madsen 2019).
- **State models (OHBA)**: TDE-HMM / DyNeMo (osl-dynamics; ports in
  `neurojax.models`) fitted on pooled placebo + drug, then state occupancy,
  lifetimes, switching rate and TINDA cycle strength as PD outputs; burst
  detection (Quinn 2019); multitaper state spectra + sign flipping already in
  `neurojax.analysis`.
- **Change-point / regime models**: Bayesian online change-point, kernel CPD
  (ruptures); regime-switching and (DCC-)GARCH from econometrics for envelope
  and dynamic-connectivity volatility; critical slowing down (Scheffer /
  Dakos; Maturana 2020 in epilepsy) as early-warning statistics along the
  time course; state-space oscillator models (Beck & Purdon 2022).
- **Foundation models as detectors**: EEG FMs (LaBraM, EEGPT, BIOT, BENDR)
  and generic TS FMs (Chronos, TimesFM, Moirai) give embedding trajectories
  and change scores over the time course; treat as detectors to be explained,
  not explanations.

### Mechanistic models (started 2026-10-25)

The spectral results above say the data-side object that matters is the
full cross-spectrum plus a small non-Gaussian residue, so the next layer is
a generative model of the spectrum whose parameters are pharmacological:
`neurojax.inverse.csd_dcm` is a JAX spectral DCM (Jansen-Rit network
linearised at its fixed point → transfer function → 1/f innovations →
leadfield → sensor CSD; variational Laplace with free energy;
`docs/tutorials/spectral_dcm.md`).  Handles: `tau_i`/`Hi` (GABA-A,
propofol), `He`/`tau_e` (glutamatergic kinetics, ketamine), `A_forward`
(between-region gains).  Plan: SPM-Python `spm_dcm_csd` parity → ketamine
MEG (NMDA) and Chennu propofol (GABA-A) as first fits → hierarchical
(study × drug) priors on the receptor-level parameters → regional receptor
maps → simulate from fitted models to test whether they reproduce the
phasic MSE / Lyapunov / 0-1 chaos residues.

**First fit, Chennu propofol (job 1866, `dcm_propofol.py`; 20 subjects × 4
levels, one region on the leading spatial mode, 80 inversions, 21 s each):**

| contrast vs baseline | tau_i (GABA-A decay) | Hi (inhibitory gain) | He | C | a_u |
|---|---|---|---|---|---|
| mild | −0.09 (d −0.33, p 0.29) | +0.03 | +0.02 | +0.04 | +0.02 |
| moderate | **−0.40 (d −0.88, p 0.002)** | **+0.20 (d 0.82, p 0.001)** | +0.09 (d 0.81) | **+0.17 (d 1.07, p 1e-4)** | +0.11 (d 0.79) |
| recovery | +0.08 (d 0.39, p 0.08) | −0.04 (d −0.34, p 0.02) | −0.02 | −0.02 | −0.02 |

(log-scalings; paired Cohen's d and Wilcoxon p over 20 subjects.)  Moderate
sedation is read by the one-column model as *faster, stronger* inhibition
with more intrinsic gain — the opposite of the GABA-A expectation (propofol
prolongs the IPSC) — because a single Jansen-Rit column has one resonance
and the only way it can produce the propofol ~15-20 Hz beta is a shorter
inhibitory time constant.  The fits are also poor (RMSE 0.6-0.7 in log
power; the predicted alpha peak is 0.5 log units above its neighbours
where the data show 1.8), and the free energy is multimodal on real data
too (the best of the 4 starts was not the prior start in 8/80 fits; median
F spread across starts 22 nats).  Conclusion: the inverter works, the
model is too small.  Next: a two-population (or canonical-microcircuit)
column with separate alpha and beta generators, several sources with a
real leadfield, and then the `spm_dcm_csd` parity check (SPM-Python needs
the x86-only MATLAB Runtime, so on the aarch64 Spark it has to run under
emulation).  Outputs: `propofol-chennu2016/dcm/{posteriors,fits,spectra,contrasts}.csv`.

**Canonical microcircuit on the same data (job 1867, `--model cmc`, 19
parameters, 27 s per fit, `dcm_cmc/`):** the fits improve (RMSE in log
power 0.40-0.54 vs 0.6-0.7, the 1/f and beta range now followed) but the
alpha peak is still under-predicted by ~1 log unit at baseline (the
default CMC modes at 7 and 17 Hz are heavily damped, max Re λ = −34/s, and
the 1/16 priors on `G` do not let the fit approach the instability needed
for a sharp alpha), and the free energy is lower than Jansen-Rit's for 66
of 80 fits (mean −22 nats: more parameters, little extra accuracy).  The
moderate-sedation contrast moves to the glutamatergic / deep-layer side:
spiny-stellate time constant `T_ss` +0.78 (d 0.81, p 0.006),
superficial-pyramidal self-inhibition `G_sp→sp` +0.35 (d 0.78, p 0.002),
deep-pyramidal ↔ interneuron gains `G_dp→ii` / `G_ii→dp` −0.27 (d −0.68,
p 0.001), innovation amplitude −0.09 (d −0.61); the inhibitory time
constant `T_ii` is +0.12 (d 0.18, n.s.) — the right sign for GABA-A
prolongation this time, but not where the model puts the effect.  Gains
that enter the linearised loop only as a product (`G_dp→ii` · `G_ii→dp`,
`G_ii→ss` · `G_ss→ii`) come out with identical posteriors: the
linearisation makes them non-identifiable, as in SPM.  Both one-column
fits therefore say the same thing — on a single spatial mode the model
explains sedation by reshaping the resonances, not by a clean GABA-A
parameter — and the next steps stand: a multi-source model with a real
leadfield (alpha and beta have different topographies under propofol:
occipital alpha loss, frontal alpha/beta gain), looser or
hierarchically-estimated priors on the gains, and the SPM parity check
(running on the x86 machine).

**Multi-source CMC with a real leadfield (job 1868, `dcm_propofol_sources.py`,
`dcm_sources/`):** four sources (bilateral occipital and frontal, fsaverage
ico-5 vertices nearest the MNI targets, cortical-normal orientation) through
the fsaverage BEM to the 91-channel EGI montage, reduced to 6 spatial modes
(87 % of baseline variance) as in `spm_dcm_csd`; occipital → frontal
forward and frontal → occipital backward CMC connections; 64 parameters,
2 starts, 72 s per fit.  The GABA-A effect now has a location: the right
frontal inhibitory time constant `T_ii` rises under moderate sedation
(+0.49, d 0.69, p 0.015) with the right-frontal ii self-inhibition falling
(−0.58, d −1.1, p < 0.001), while occipital `T_ii` does not move (d 0.00,
−0.07) and left frontal only weakly (+0.09) — consistent with propofol's
frontal alpha/beta and occipital alpha loss being different mechanisms.
But the fits are worse than the single-mode ones (RMSE in log power 0.74 /
0.97 / 1.28 at the 10/50/90 % quantiles) and 37 of 320 extrinsic-gain
posteriors sit more than 3 prior SD from the prior, so the model is still
not explaining these spectra well: fixed dipole orientations, only two
starts, and no lateral connections are the likely causes.  Treat the
frontal `T_ii` result as a lead to confirm with the SPM parity check and a
better-converged fit, not as a finding.

**Second level (`neurojax.inverse.peb`, `dcm_peb.py`, job 1869 refit of
the one-source CMC with full posterior covariances):** parametric
empirical Bayes with subject indicator columns plus mild / moderate /
recovery columns, Bayesian model reduction for each effect.  Effects with
> 3 nats of evidence: mild and moderate raise the superficial-pyramidal
time constant `T_sp` (+0.74 ± 0.20, 7 nats) and `T_ss` (+0.5-0.6 ± 0.16,
4-6 nats) and lower `G_ss→sp` (−0.45 ± 0.16); recovery raises `T_ss`
(+0.72 ± 0.15, 11 nats) and `T_ii` (+0.70 ± 0.19, 6 nats).  268 of 1520
first-level information eigenvalues are non-positive (parameters the
one-source fit did not learn about), so the hierarchy is working on a
weak first level: with recovery looking as different from baseline as
moderate sedation, the one-source CMC posteriors are not yet
pharmacologically interpretable, and the PEB result is a demonstration of
the machinery (the per-study / per-drug hyperprior layer of the planned
mega-analysis) rather than a finding.  The first-level fits also changed
between runs when the CMC input gain was aligned to SPM (512), i.e. the
posteriors are sensitive to the innovation scale.

**Free dipole moments (job 1870, 4 starts, 80 parameters, 170-430 s per
fit; *these fits ran in float32* — the joblib workers did not inherit the
x64 setting — and are kept as `dcm_sources_free_f32/`; the float64 refit is
job 1873 → `dcm_sources_free/`, results to be updated):** the fixed-orientation misfit was spatial
(mode 0 under-predicted by 1-1.8 log units, modes 2 and 4 over-predicted by
1-2.7, at every frequency alike), so each source's moment is now a free
parameter with the cortical normal as prior mean (as `spm_L_priors`' ECD
moments; the starts draw moments from the prior because the log-power
features have a barrier wherever a sensor projection changes sign).  Log
power RMSE drops from ~1.0 to 0.36 at all levels and F rises by ~2000 nats
per fit.  Moderate − baseline (paired d, Wilcoxon): frontal inhibitory time
constants rise (right `T_ii` +0.71, d 0.98, p 0.001; left +0.32, d 0.51,
p 0.03 at occipital-left), frontal interneuron self-inhibition `G_ii→ii`
falls (left −0.72, d −1.28; right −0.65, d −1.0; both p < 0.001), frontal
superficial-pyramidal self-inhibition `G_sp→sp` rises (+0.6, d 1.1, p <
0.001) and left-frontal `T_ss` rises (+0.38, d 1.14).  Mild sedation shows
the same frontal `G_sp→sp` change at half the size (d 0.81); recovery only
extrinsic gains (d 0.4-0.6).  So with a leadfield the model can fit, the
propofol effect is frontal, inhibitory and in the GABA-A direction (slower
`T_ii`, weaker ii→ii) — but two things forbid calling it a finding yet: (i)
86 of 320 extrinsic-gain posteriors sit > 3 prior SD from the prior and the
free energy still spreads 150 nats (median) across the 4 starts; (ii)
**52 of the 80 posterior-mean linearisations are unstable** (max Re λ > 0,
up to +97/s, always on the ~10 Hz mode): the inverter sharpens the alpha
peak by pushing the alpha mode across the Hopf point, where the
transfer-function CSD is still defined but no longer describes a stationary
linear process.  A stability-constrained inversion (barrier on max Re λ, or
SPM's practice of accepting it) is the next model step.  Outputs:
`dcm_sources_free/{posteriors,fits,spectra,contrasts}.csv`,
`posterior_<subject>_<cond>.npz`, `leadfield_<subject>.npz` (`L3`, `nn`, `U`).

**Float64 refit (job 1873, `dcm_sources_free/`):** same fit quality (log
power RMSE 0.33-0.36, F −450 to −750, median F spread across starts 145
nats) and the same picture — moderate − baseline: frontal `G_ii→ii` falls
(right −0.63, d −1.01, p < 0.001; left −0.59, d −0.72), `G_sp→sp` rises in
all four sources (d 0.76-1.04), left-frontal `T_ii` +0.56 (d 0.77, p 0.006;
the right-frontal `T_ii` of the float32 run is no longer significant),
frontal `T_ss` +0.36-0.41 (d 0.65-0.71), `G_dp↔ii` −0.48 (d −0.63);
mild: `G_sp→sp` again (d 0.62-0.68); recovery: right `G_ii→ii` −0.28 (d
−0.54), `G_sp→sp` +0.34 (d 0.67).  89 of 320 extrinsic-gain posteriors are
> 3 prior SD from the prior and **48 of 80 linearisations are unstable**
(moderate: median max Re λ +15.5/s), so precision was not the cause of
either.

**PEB on these posteriors (`dcm_peb.py`, shared between-subject
log-precision, BMR on the sedation columns):** two more fixes were needed
first — a per-parameter log-precision is intractable at 80 parameters
(hours in the Hessian) and, more importantly, the reduction identity blew
the between-subject variance up (sd 72, then 6·10⁵ on synthetic data) in
*barely informed* first-level directions: there the posterior mean differs
from the prior mean only by optimisation noise, so the implied data
location Λ⁻¹h is enormous.  A direction now counts as informed only if the
first level gained ≥ 10 % of the prior precision in it.  With that, the
effects > 3 nats are: every sedation level, *including recovery*, raises
`T_ss` and `T_sp` in all four sources by +0.6-1.0 (sd 0.18, 4-17 nats), and
mild / moderate raise `G_sp→sp` (+0.5-0.67, 3.5-5.5 nats); nothing on the
inhibitory parameters survives (the random-effects sd settles at 0.97 in
prior units, i.e. the subject-to-subject scatter of these fits is large).
An effect shared identically by mild, moderate and recovery is not
pharmacology; the likely culprit is that the spatial modes `U` are the
baseline covariance's eigenvectors, so the baseline CSD is "aligned" with
the modes and the other three levels are not — the next run should take
the modes from the pooled covariance of all four levels.

**Simulating the fitted models against the measures (job 1872,
`dcm_simulate_measures.py`, `dcm_sources_free/sim_vs_observed.csv`):** each
posterior-mean model was simulated at the 91 sensors (60 s, 250 Hz), scored
with 15 bench measures on the bench's 10 channels × 6 epochs and
contrasted as the real data were.  The linear-Gaussian simulation (the
fitted cross-spectrum, `amplitude 0`) reproduces the spectral contrasts —
relative beta d 1.06 / 1.03 / 0.42 (mild / moderate / recovery) against
observed 1.13 / 1.41 / 0.42, Hjorth mobility 0.76 vs 0.86, aperiodic
exponent −0.46 vs −0.61, LZ76 0.53 vs 0.79, recovery's alpha 0.52 vs 0.49 —
but not the phasic residues: multiscale entropy gives d −0.23 under
moderate sedation where the data give −0.90 (the bench's 60 %-phasic
measure), the Lyapunov exponent 0.02 vs −0.45, pair correlation 0.04 vs
−0.52 (mild).  The nonlinear integration of the same posteriors (innovation
`amplitude` 0.1-3; the CSD fit does not identify it) moves MSE part of the
way (−0.37 to −0.53) but breaks the spectral contrasts even at amplitude
0.1 (relative beta d 0.28) and inverts the Lyapunov sign at amplitude 3 —
because 52 of the 80 fitted networks are unstable (above) and settle on a
limit cycle, not because of a small nonlinear correction.  Conclusion so
far: the spectral DCM accounts for the spectral family of measures by
construction and for none of the phasic residues; testing whether the
model's nonlinearity can produce them needs the stability-constrained
fits first.  The multitaper used throughout (`analysis.multitaper`) was
found to be low by a factor `nfft` and ~250× slower than necessary (dense
DPSS eigenproblem per call); both fixed 2026-10-25 — absolute spectral
levels in earlier tables are off by that constant, nothing relative is.

**Simulation comparison on the float64 posteriors (job 1874,
amplitudes 0 / 0.1 / 0.3 / 1):** the linear simulation again carries the
spectral family (relative beta d 1.14 / 1.20 vs 1.13 / 1.41 for mild /
moderate, Hjorth mobility 0.82 vs 0.86, aperiodic exponent −0.53 vs −0.61,
Higuchi 0.45 vs 0.75, LZ76 0.54 vs 0.79) and now also most of the
multiscale-entropy drop under moderate sedation (−0.63 vs −0.90; the
float32 fits gave −0.23), but not the Lyapunov exponent (0.01 vs −0.45)
nor pair correlation (0.24 vs −0.17), and it produces a large spurious
0-1-chaos drop (−1.07 vs 0.02).  The nonlinear integrations still break
the spectral contrasts at every amplitude (relative beta d 0.4) because
48 of the 80 fits are unstable; the stability-constrained, pooled-modes
refit (job 1876, `dcm_sources_stable/`) is the one to redo this with.

Not ported from Maschke 2024: avalanche exponents / DCC (needs `powerlaw`
fits and long recordings), Fano factor, avalanche repertoire, the
integrated-covariance width Δ, and specparam-based slope (the log-log fit is
used; `neurojax.analysis.specparam_analysis` needs `specparam` installed).

Every value in `nltsa.py` is a plain function with explicit defaults so a
result table can name the estimator and parameters; the table above is the
report.

## GPU path (`nltsa_batched.py`, `gpu.sbatch`)

Six measure families are re-expressed as jitted functions over a (B, T)
stack of channel-epochs and dispatched to the default JAX device when it is
an accelerator: the recurrence profile (8 RQA / recurrence-network
measures from one N×N matrix), Rosenstein Lyapunov, correlation
dimension, 0-1 chaos, sample entropy and multiscale entropy. They were
~90 % of the per-epoch CPU cost. The rest (Lempel-Ziv parsing, which is
sequential; ordinal, fractal, Hjorth, DFA, specparam) stay in numpy on
joblib workers pinned to the CPU backend, running concurrently with the
device. Parity with the scalar numpy versions (which are the ones checked
against pyunicorn / nolds / antropy) is float32-exact: on the propofol
benchmark every batched measure agrees with the 20-core CPU run to
< 3 × 10⁻⁴ of a between-subject SD and every Cohen's d to four decimals.

GB10 per channel-epoch (10 s at 250 Hz, chunk 64), against one CPU core:

| measure | GPU | numpy, 1 core |
|---|---|---|
| recurrence profile (8 measures) | 17 ms | 500 ms |
| correlation dimension | 6.5 ms | 126 ms |
| 0-1 chaos | 0.3 ms | 134 ms |
| Lyapunov | 0.1 ms | 78 ms |
| multiscale entropy | 2 ms | 30 ms |

Same propofol job (80 recordings × 6 epochs × 10 channels, 43 measures):

| | wall | cores |
|---|---|---|
| `run_bench.sbatch` (cpu partition) | 824 s | 20 |
| `gpu.sbatch` (GB10 + joblib) | 195 s | 8 |

(195 s still paid a recompile per call, fixed afterwards; `gpu_profile_epochs.py`
on 120 channel-epochs: device part 3.1 s, 8-core joblib part 2.6 s, overlapped
3.6 s, against 34.3 s for all 30 measures on 8 CPU cores — 9.5× at equal cores.)

Two things that made the GPU path slower than numpy before they were
fixed, kept here so they are not re-introduced: (1) the run-length
histogram for RQA lines used a scatter-add; on the GPU every non-line cell
is an atomic add into bin 0 and the recurrence profile took 300 s per
item — it is now a sort + searchsorted; (2) `lax.map` outside `jit`
re-traces and recompiles on every call when handed a fresh `partial`, so
the compiled chunked maps are cached by (measure, parameters, chunk, T).
Matmul precision is forced to "highest" for the adjacency products (TF32
default on the GB10; see the HIGGS note in the memory file).

## Datasets

Root: `/data/raw/consciousness-bench/` (override with `CB_ROOT`).

| dataset | contrast | N | EEG | access |
|---|---|---|---|---|
| `propofol-chennu2016` | baseline → mild → moderate sedation → recovery | 20 | 91-ch, 10 s epochs, EEGLAB | open, CC BY 2.0 UK, Cambridge repository 1810/252736 |
| `dmt-pallavicini2021` | eyes-closed rest vs inhaled DMT (+ eyes-open) | 35 | BDF (2 XDF) | open, CC BY 4.0, Zenodo 3992359 |
| `psilocybin-microdose-cavanna2022` | placebo vs 0.5 g dried psilocybe, eyes closed (EO also available) | 34 (31 BDF, 3 XDF skipped) | 24-ch BDF, 500 Hz, ~3 min | open, CC BY 4.0, Zenodo 5745892 (`Data.rar`, RAR5: extract with bsdtar) |
| `anesthesia-sarasso2015` | wake vs propofol / xenon / ketamine (the Maschke 2024 data) | 5 + 5 + 5 (+4 N3 sleep) | 60-ch Nexstim TMS-compatible | **restricted**: Zenodo 806176, motivated access request to Massimini/Laureys, 99.8 GB |
| LSD / psilocybin / ketamine MEG (Schartner 2017) | placebo vs drug | 15 / 14 / 19 | CTF 275 MEG | on request from Imperial/Cardiff; not public |

The first three are fetched by `download_datasets.sh`. The Sarasso set must
be requested by the user and unpacked under `$CB_ROOT/anesthesia-sarasso2015/`.

## Outputs

`run_bench.py` writes, per dataset, `scores.csv` (subject × condition ×
measure), `contrast_<a>_vs_<b>.csv` (paired Cohen's d, Wilcoxon p, AUC) and
`decomposition_<a>_vs_<b>.csv` (spectral / phasic parts, `spectral_fraction`,
`phasic_z`) to `/data/datasets/consciousness-bench/<dataset>/`; the spectral
baselines go to `*_spectral.csv` (`--measures spectral --tag spectral`) and
`spectral_explained.py` joins the two into `spectral_explained_<a>_vs_<b>.csv`.

Run settings used for the first pass (CPU, 20 cores): 10 evenly spaced
channels, 6 × 10 s epochs per recording, 10 Prichard-Theiler surrogates.

## First-pass results (2026-10-22; 10 channels, 6 × 10 s epochs, paired Cohen's d)

Full tables: `/data/datasets/consciousness-bench/<dataset>/`.

### Spectral tools beat the NLTSA measures on every contrast

| dataset / contrast | n | best spectral feature (d) | best NLTSA feature (d) |
|---|---|---|---|
| propofol mild − baseline | 20 | relpow_theta −1.49, relpow_beta +1.13 | hjorth_mobility +0.86, lz78 +0.83 |
| propofol moderate − baseline | 20 | relpow_beta +1.41, relpow_theta −1.12, relpow_alpha −1.03 | katz_fd +0.89, mse_sum −0.90 |
| propofol recovery − baseline | 20 | specparam_alpha_cf −0.87, relpow_delta −0.87 | dfa −0.91 |
| DMT − eyes closed | 34 | specparam_alpha_power −2.30, relpow_alpha −1.70 | dfa +1.22, mse_sum −0.94 |
| eyes open − eyes closed (DMT study) | 34 | specparam_alpha_power −1.88, specparam_exponent +0.94 | dfa +1.02 |
| DMT − eyes open | 34 | specparam_exponent −0.67 | hjorth_complexity +0.44 |
| psilocybin microdose − placebo | 31 | specparam_offset −0.52, specparam_exponent −0.47 | aperiodic_exponent −0.61 (itself spectral); all others |d| < 0.2 |

### Most NLTSA measures are re-expressions of six spectral parameters

R² of each measure on z-scored specparam exponent, offset, alpha peak power
and relative delta/theta/beta power (all subject × condition rows):

| measure | propofol | DMT | psilocybin |
|---|---|---|---|
| dfa | 0.97 | 0.68 | 0.88 |
| hjorth_mobility | 0.95 | – | – |
| lz76_mean / lz78 | 0.94 / 0.94 | 0.60 / – | 0.57 / 0.51 |
| permutation_entropy | 0.90 | – | – |
| higuchi_fd | 0.88 | 0.40 | 0.51 |
| mse_sum | 0.43 | 0.82 | 0.82 |
| pair_correlation | 0.23 | 0.68 | 0.64 |
| chaos_01 / branching_ratio / dfa_alpha_envelope | 0.13 / 0.11 / 0.05 | – | – |

After regressing the spectral parameters out, every propofol NLTSA contrast
falls below |d| = 0.4 except pair_correlation at mild sedation (raw −0.52,
residual −0.51): the only non-spectral signal in that dataset. The measures
that are *not* spectral (0-1 chaos, branching ratio, alpha-envelope DFA)
carry no contrast at these doses.

### Measure overlap (within-subject change correlations, 30 NLTSA measures; `overlap/`)

| | propofol (3 contrasts) | DMT / psilocybin | eyes-open control |
|---|---|---|---|
| mean |r| of changes | 0.57–0.61 | 0.40–0.46 | 0.50 |
| participation ratio | 1.9–2.1 | 3.0–3.8 | 3.2 |
| first-PC share | 67–72 % | 46–53 % | 53 % |

Propofol compresses the set onto one axis (one cluster of 23–25 measures:
LZ, entropies, fractal dimensions, Hjorth mobility, correlation dimension,
Lyapunov, the whole RQA / recurrence-network family; DFA + Hjorth
complexity as a second); psychedelics split it into an amplitude-based
block and an ordinal block (permutation entropy, statistical complexity,
Petrosian, Higuchi, ± Lyapunov, ± aperiodic exponent), and on DMT − eyes-open
the diagonal-line RQA measures and spectral entropy form a third block.
Mantel r between contrasts: 0.91 within class, 0.83 anaesthesia-vs-
psychedelic; mean matrices r = 0.88 (p < 0.001, 5000 permutations),
RV = 0.90. Largest coupling changes (propofol → psychedelic): Higuchi–RQA
diagonal entropy −0.94 → +0.17, Higuchi–spectral entropy +0.69 → −0.32,
Petrosian–RQA diagonal entropy −0.93 → +0.03, permutation entropy–spectral
entropy +0.58 → −0.33, permutation entropy–SVD entropy +0.83 → +0.15: under
propofol the ordinal / fractal measures and the recurrence line structure
move as one; under DMT they decouple. Partialling the six spectral
parameters lowers the mean |r| only from 0.61 to 0.48–0.52 (propofol): the
shared variance is the common dependence on the full spectral shape, not on
six summary parameters.

Recurrence family on propofol (moderate − baseline): laminarity d = −0.50,
network clustering −0.48, transitivity −0.43, all spectral (fraction
0.8–1.0); none adds a phasic effect. On psilocybin: null.

### Spectral/phasic decomposition, propofol (Prichard-Theiler multivariate null, 10 surrogates)

- Mild − baseline: every NLTSA measure is spectral (fraction 0.78–1.06,
  |phasic z| < 2.6 except Higuchi, whose phasic part is 0.005). The pair
  correlation effect that survived the univariate spectral regression is
  reproduced once cross-spectra are preserved.
- Moderate − baseline: multiscale entropy is 60 % phasic (total −1.27,
  spectral −0.50, phasic −0.76, z = −13); alpha-envelope DFA rises
  (+0.017) where the spectrum predicts a fall (−0.043), so its change is
  entirely non-Gaussian; LZ variants are 82–85 % spectral with a detectable
  but few-percent phasic residue.
- Recovery − baseline: all spectral (largest |phasic z| 3.2 on the aperiodic
  exponent with phasic part 0.012).

### Spectral/phasic decomposition, DMT (Prichard-Theiler multivariate null, 10 surrogates, 30 measures)

- DMT − eyes-closed: LZ76/78, DFA, permutation entropy, statistical
  complexity, Petrosian, Higuchi, correlation dimension, Hjorth complexity
  and the recurrence family are spectral (fraction 0.8–1.5; the LZ
  "over-prediction" of 1.3–1.5 means the surrogates lose slightly more LZ
  than the drug does). Multiscale entropy is 27 % phasic (total −6.2,
  phasic −1.7, z = −12), the Lyapunov exponent 71 % phasic (−0.80, phasic
  −0.57, z = −9), and the 0-1 chaos test is the only measure whose change is
  *opposite* to its spectral prediction (total +0.015, spectral −0.083,
  phasic +0.098, z = 17). Pair correlation is 60 % phasic (z = 11) but tiny
  (+0.047). Spectral tools remain the strongest contrast (specparam alpha
  power d = −2.30, relative alpha −1.70, relative delta +1.38; best NLTSA:
  DFA +1.22, MSE −0.94, permutation entropy +0.82).
- DMT − eyes-open (the drug contrast): the aperiodic exponent (d = −0.67)
  is the strongest single feature after the specparam fit error; it is
  55 % spectral and its phasic 45 % is the within-recording nonstationarity
  of the drug time course, which the surrogates destroy. MSE, SVD entropy
  and the Lyapunov exponent again move against or well past their spectral
  prediction (phasic z = 24, 20, 6); the 0-1 chaos test is again phasic
  (z = 18). Everything in the ordinal / fractal / recurrence families is
  spectral (fraction 0.8–1.4). Spectral regression on six parameters
  explains R² 0.98 of the aperiodic exponent, 0.82 of MSE, 0.73 of SVD
  entropy, 0.6–0.7 of DFA / LZ / PCF / Hjorth complexity, but only
  0.1–0.4 of the recurrence, chaos and fractal measures; residual d after
  that regression stays at 0.4–0.7 for LZ78 (−0.67), RN clustering
  (+0.66), DFA (+0.53), LZ76 (−0.53), spectral entropy (−0.47) — the part
  of those measures that six spectral numbers do not carry but the full
  spectrum (phase surrogates) does.

### DMT time course (n=34; `scripts/consciousness_bench/dmt_timecourse.py`, outputs in `dmt-pallavicini2021/timecourse/`)

t = 0 is exhalation (recording starts at exhalation per Pallavicini 2021);
10 s windows, 5 s step, z vs each subject's own eyes-closed windows.

- Every feature is already at or near its extreme in the first window:
  inhaled DMT acts within seconds, so no absorption phase is resolved and
  Bateman fits succeed in only 35–59 % of subjects (ordinal measures best:
  Higuchi/Petrosian/permutation entropy 56–59 %, alpha peak power 50 %).
  Time-to-peak medians 35–95 s (specparam offset 35 s, Lyapunov 49 s, alpha
  power 54 s, ordinal measures 70–90 s).
- Shape: alpha power −3.7 z at onset, −4.5 z at 45 s, recovering only to
  −2.8 z by 8 min; LZ −3 z at onset recovering to −1.3 z by 150 s then flat;
  delta power +5 z → +2.8 z; DFA +3.7 → +1.5 z. The aperiodic exponent is
  the one feature with a biphasic course: +1.5 z (steeper) at onset, baseline
  by 100 s, −1 z (flatter) after 5 min.
- TDE-HMM (15 lags, PCA 30, 6 states, pooled EC + DMT): one state is a
  transient peaking at 45 s (+6 z) and gone by 2 min; a second is elevated
  for the first 2 min then halves; a third rises monotonically from 2 min on;
  one state is suppressed throughout (−2.5 z). Switching rate barely moves.
  So the HMM resolves an early (0–2 min) and a late (>2 min) DMT regime that
  the scalar measures only show as a monotone recovery.
- **No feature returns to baseline by the end of the recording** (all |z| > 1
  at 465 s) although subjects reported return to baseline at 6 ± 1.4 min.
  Without a post-drug eyes-closed recording this plateau cannot be separated
  from a session/state difference (posture, sweating, residual arousal) —
  the drift problem again, on the slowest time scale.

### Windowing and low-frequency sensitivity, propofol baseline → moderate (`sensitivity_baseline_vs_moderate.csv/.png`)

`scripts/consciousness_bench/run_sensitivity.py`: 8 preprocessings
(demean, linear detrend, HPF 0.1/0.5/1/2 Hz, EMD with IMFs below 0.5 Hz
dropped, Hankel-DMD with modes below 0.5 Hz removed) × epoch lengths
2/5/10/20 s, 10 channels, 60 s per recording, 30 measures, n = 20.
Reference setting is HPF 0.5 Hz, 10 s. `rho_ref` is the Spearman
correlation of per-subject changes against the reference (does the
setting re-rank subjects?), `mantel` the Mantel r of the 30 × 30
overlap matrix against the reference (does it change the pattern?).

| setting (10 s) | mean \|d\| | max \|d − d_ref\| | mean rho_ref | Mantel vs ref |
|---|---|---|---|---|
| demean / detrend / EMD 0.5 / DMD 0.5 | 0.34 | 0.05–0.06 | 0.98 | 0.99 |
| HPF 0.1 | 0.35 | 0.10 | 0.97 | 0.99 |
| HPF 0.5 (reference) | 0.35 | 0 | 1.00 | 1.00 |
| HPF 1.0 | 0.40 | 0.38 | 0.92 | 0.99 |
| HPF 2.0 | 0.43 | 0.74 | 0.85 | 0.98 |

- **The Chennu data are not a test of the sub-0.5 Hz alternatives.** The
  authors released them already band-passed 0.5–45 Hz, so demean, detrend,
  EMD and DMD have nothing below 0.5 Hz to remove and reproduce the 0.5 Hz
  reference to within d ± 0.06. The informative run is the raw-BDF DMT
  study (next section).
- **Raising the HPF cutoff is not neutral.** 1 Hz moves DFA, Hjorth
  complexity, Lyapunov, spectral entropy and recurrence-network clustering
  by 0.3–0.7 in d; 2 Hz turns DFA from a null (−0.07) into −0.81, doubles
  Hjorth complexity (−0.50 → −1.22) and flips the Lyapunov sign
  (−0.41 → +0.13). The top-5 ranking is stable (MSE, LZ78, LZ76, Katz,
  Hjorth mobility) for every setting up to 0.5 Hz and changes to Hjorth
  complexity / spectral entropy / clustering at 1–2 Hz. These measures are
  reading the delta band, which 1–2 Hz filters remove: a cutoff choice
  silently selects which spectral band the "complexity" summarises.
- **Epoch length matters more than preprocessing for the scaling
  measures.** With the reference HPF, 2 s epochs make DFA (−0.83) and the
  DFA alpha envelope (−0.83) look like strong effects that vanish at 10 s
  (−0.07, +0.17): the scaling range on 2 s is too short and the fit picks
  up the filter edge. Branching ratio and 0–1 chaos change sign across
  epoch lengths (±0.4). MSE sum weakens from −1.22 (2 s) to −0.62 (20 s)
  as the coarse scales get enough samples. LZ, Katz, Hjorth, permutation
  entropy and the RQA profile are stable within 0.1 across 2–20 s.
- **Subject ranking is the more sensitive readout.** Even where group d is
  unchanged, branching ratio (rho_ref 0.12–0.64), chaos_01 (0.74–0.98) and
  MSE (0.63–0.97) re-rank subjects between settings; the overlap structure
  itself is invariant (Mantel ≥ 0.96 everywhere).

### Windowing and low-frequency sensitivity, DMT eyes-open → DMT (`dmt-pallavicini2021/sensitivity_EO_vs_DMT.csv/.png`)

Same sweep on the raw BDF recordings (only the 45 Hz low-pass and
average reference before the alternatives are applied), n = 34, 10
channels, 60 s per recording. This is the run where demean, detrend,
EMD and DMD actually have sub-0.5 Hz content to act on. GPU job: 3.1 h
for 8 × 4 settings × 30 measures (the propofol sweep took ~11 h on 20
CPU cores with n = 20).

| setting (10 s) | mean \|d\| | #p<0.05 | max \|d − d_ref\| | mean rho_ref | Mantel vs ref |
|---|---|---|---|---|---|
| demean | 0.26 | 6 | 0.46 | 0.63 | 0.90 |
| detrend linear | 0.48 | 22 | 0.87 | 0.64 | 0.89 |
| HPF 0.1 | 0.43 | 19 | 0.78 | 0.78 | 0.94 |
| HPF 0.5 (reference) | 0.19 | 3 | 0 | 1.00 | 1.00 |
| HPF 1.0 | 0.22 | 8 | 0.34 | 0.89 | 0.96 |
| HPF 2.0 | 0.25 | 13 | 0.51 | 0.80 | 0.90 |
| EMD 0.5 | 0.28 | 12 | 0.95 | 0.70 | 0.80 |
| DMD 0.5 | 0.45 | 21 | 0.99 | 0.77 | 0.93 |

- **Below 0.5 Hz the "effect" is drift.** With the 0.5 Hz reference the
  drug contrast is small: aperiodic exponent (−0.70) and the symbolic
  measures (permutation entropy +0.38, Petrosian +0.36, statistical
  complexity −0.32) and nothing else past |d| 0.3. Every setting that
  leaves sub-0.5 Hz content in (demean, detrend, HPF 0.1, DMD 0.5)
  roughly doubles the mean |d| and makes spectral entropy (−0.57 to
  −1.10), SVD entropy, MSE sum, LZ76/78, RQA diagonal entropy and
  recurrence-network clustering/transitivity look like large effects
  (|d| 0.6–1.1) that the 0.5 Hz filter removes entirely. DMT sessions
  have more sweating, movement and breathing drift than eyes-open rest,
  and these measures read it as "lower complexity / more determinism".
  The strongest published DMT complexity findings are at risk from exactly
  this unless the cutoff is reported.
- **HPF 1–2 Hz again flips signs, in the same measures as propofol.** DFA
  goes +0.04 → −0.40, Hjorth complexity +0.21 → −0.28, LZ76/78 and
  spectral entropy change sign, SVD entropy grows to +0.52. The
  aperiodic exponent is the one measure whose d grows monotonically with
  the cutoff (−0.66 → −0.83) because the 1/f fit range shrinks.
- **The symbolic/ordinal measures are cutoff-invariant.** Permutation
  entropy, Petrosian, Higuchi, statistical complexity, pair correlation
  and the DFA alpha envelope have rho_ref ≥ 0.98 and d within 0.05 under
  every HPF, demean, detrend and DMD. They are also the only measures that
  survive the 0.5 Hz reference, so the robust DMT signature is the
  ordinal-pattern one (consistent with Pallavicini's own LZ/PE result).
- **EMD is not an alternative to a high-pass.** It is the only setting
  that breaks the invariant measures too (permutation entropy +0.38 →
  +0.05, Petrosian 0.36 → 0.00, pair correlation 0.30 → 0.03, rho_ref
  0.5–0.9) and it flips the aperiodic exponent (−0.70 → +0.25) and
  chaos_01 (−0.12 → −0.69). Dropping whole IMFs by mean Hilbert
  frequency removes broadband mode-mixed components, not a band, so it
  alters fine-scale structure per subject in a way the other settings
  do not (Mantel 0.80, the lowest of the sweep).
- **DMD low-mode removal ≈ linear detrend here.** DMD 0.5 and
  detrend_linear agree within 0.25 in d for 27/30 measures
  (the exceptions are chaos_01, Lyapunov and RQA mean diagonal length), both well
  away from HPF 0.5: with 15 s windows and rank ≤ 60 the retained slow
  modes do not capture the 0.1–0.5 Hz content that the Butterworth
  removes. It is not a substitute for the filter at this cutoff.
- **Epoch length matters less than on propofol.** Under the reference HPF
  every measure stays within 0.35 across 2–20 s except chaos_01 (range
  0.44); recurrence time and trapping time are strongest on 2 s epochs
  (0.48/0.37 vs 0.19/0.15 at 10 s), and chaos_01, branching ratio and DFA
  again change sign across lengths.

### What filtering and windowing the published literature used (`consciousness_bench/literature_windowing.csv`)

Methods extracted from 108 analyses: the 46 anaesthesia EEG/TMS-EEG
studies of Sarasso et al. 2021 (Neurosci Conscious, Table 1) plus
Chennu 2016, Toker 2022, Maschke 2024 and Newman 2026; and 43 human
EEG/MEG complexity studies under acute psilocybin, LSD, DMT,
5-MeO-DMT, DMT+harmine or psychedelic-dose ketamine. Each row carries a
verbatim Methods quote, access status (93 full text, 10 abstract only)
and `NR` where a value was not reported. `hpf_class`, `epoch_class` and
`signal_complexity` (single-signal complexity, entropy, criticality or
aperiodic measure, as opposed to connectivity/spectral) were assigned by
hand. Yan 2020 (acupoint stimulation, not a drug) is flagged `exclude`.
Girn et al. 2026 (Nat Med) is an fMRI functional-connectivity
mega-analysis with whole-scan correlations and only a 128 s DCT
high-pass, so it is not in the table. No pooled or meta-analysis of
complexity measures under anaesthesia exists; Sarasso 2021 lists
studies without methods and Vakitbilir 2026 (BMC Biomed Eng, 94
BIS/entropy studies) reports heterogeneity qualitatively.

Signal-complexity rows only, high-pass × epoch length:

| literature | high-pass | <2 s | 2–3 s | 4–8 s | 10 s | >10 s | evoked | NR |
|---|---|---|---|---|---|---|---|---|
| anaesthesia (24) | none, demean/detrend | 1 | | | 3 | 1 | | |
| | <0.5 Hz | | | | | | 2 | |
| | 0.5 Hz | | 1 | 1 | 3 | | | 1 |
| | 1 Hz | | | | 1 | 1 | 1 | |
| | NR | | | | | | 4 | 4 |
| psychedelic (48) | none, demean/detrend | 1 | 2 | | 3 | | | |
| | <0.5 Hz | | 1 | | 1 | | | 1 |
| | 0.5 Hz | 2 | 2 | 3 | 1 | | | 1 |
| | 1 Hz | | 17 | 2 | | 2 | 2 | 1 |
| | ≥2 Hz | | 1 | | | | | |
| | band-limited / NR | | | | | 1 | | 4 |

- **The two literatures use different defaults.** Psychedelic EEG/MEG
  complexity is dominated by one recipe: 1 Hz high-pass and 2–3 s epochs
  (17 of 48 rows; Schartner 2017, Timmermann 2019/2023, Pallavicini 2021,
  Cavanna 2022 and most Imperial-lineage reanalyses). The anaesthesia
  complexity work since 2016 converges on 0.5 Hz and 10 s epochs
  (Maschke 2024, Newman 2026, both on Chennu-style preprocessing;
  connectivity papers such as Chennu 2016 and Afshani 2019 use the
  same), which is this benchmark's
  reference setting; earlier work is mostly unfiltered or not reported.
- **Several high-profile analyses keep all slow content.** Toker 2022
  (anaesthesia ECoG and LSD MEG), Mediano 2024, Li 2019 and Schartner
  2015's main analysis apply no high-pass, only demean/detrend, on 1–10 s
  windows. On the DMT data this is the setting that doubled effect sizes
  for MSE, LZ and RQA measures.
- **The cutoff decides the sign of DMT LZ on these data.** Our sweep at
  the psychedelic literature's recipe (1 Hz, 2 s) gives DMT − eyes-open
  d = +0.33 for envelope LZ76 (the Pallavicini/Schartner binarisation),
  +0.16 for LZ76 and +0.10 for LZ78. At the anaesthesia recipe (0.5 Hz,
  10 s) the same contrast is +0.03, −0.05 and −0.14, and with demean
  only it is −0.17, −0.25 and −0.13. Permutation entropy is +0.33 to
  +0.41 under all three. The "increased complexity under psychedelics"
  LZ result is therefore not comparable with the anaesthesia "decreased
  complexity" result without matching the filter.
- **Propofol LZ is robust to the same switch; DFA and MSE are not.** At
  1 Hz / 2 s the propofol moderate − baseline LZ effects stay within 0.1
  of the reference, but DFA goes −0.07 → −0.91, MSE −0.88 → −1.12 and
  chaos_01 +0.10 → −0.47.
- **Only one paper tests its own cutoff.** Schartner 2015 repeated the
  propofol LZ/ACE/SCE contrasts after a 1 Hz high-pass (subjects with
  d > 0.8 fell from 7/7 to 5–6/7). Ort 2023 is the only one to justify
  its epoch length (10 s, because LZ depends on it). Binarisation also
  differs across the psychedelic papers (Hilbert amplitude at the mean
  or median, raw signal at the mean or median) and LZ76 vs LZ78.
- **Shared data.** Counts are analyses, not datasets; `dataset_id`
  groups reanalyses of the same recordings (108 rows → 77 datasets;
  72 complexity rows → 47 datasets, 13 anaesthesia and 36 psychedelic,
  with Farnes 2020 and Li 2019 in both). The Massimini/Laureys
  Milan–Liège propofol/xenon/ketamine TMS-EEG and hd-EEG recordings
  (Zenodo 806176) account for 8 rows: Casali 2013, Sarasso 2015,
  Casarotto 2016, Colombo 2019 ×2, Comolatti 2019, Ruiz de Miras 2019
  and Maschke 2024. The Carhart-Harris 2016 LSD MEG accounts for 6
  (Schartner 2017, Toker 2022 ×2, Mediano 2024, Rajpal 2022,
  Muthukumaraswamy & Liley 2018), the Imperial DMT EEG for 5 and the
  Murphy 2011 propofol EEG for 3 (Barrett 2012, Schartner 2015 ×2).

### Data availability for an EEG/MEG mega-analysis (`consciousness_bench/data_availability.csv`)

Target design: an EEG/MEG analogue of Girn et al. 2026 (Nat Med), which
pooled 11 raw rs-fMRI psychedelic datasets, ran one fMRIPrep pipeline at
each site, reported four denoising pipelines and fitted a Bayesian
hierarchical model with per-study and per-drug hyperpriors. The
literature table collapses to 63 distinct recordings (reanalyses merged
via `dataset_id` and notes); 11 further public datasets were found by
searching OpenNeuro, Zenodo, Dataverse, Dryad, DANDI and PhysioNet.
Metadata only; nothing downloaded.

| status (63 from the literature) | n |
|---|---|
| open | 11 (3 of them derived values only) |
| restricted (request / DUA) | 4 |
| on request from authors | 14 |
| no statement found | 32 |
| not available | 2 |

Open raw or preprocessed data usable now (verified via the repository
APIs where marked):

| dataset | drug | n | design | where |
|---|---|---|---|---|
| Pallavicini 2021 (in hand) | inhaled DMT | 34 | EO/EC baseline, no placebo | Zenodo 3992359 |
| Cavanna 2022 (in hand) | psilocybin 0.5 g microdose | ~30 | placebo crossover | Zenodo 5745892 |
| Chennu 2016 (in hand) | propofol | 20 | baseline/mild/moderate/recovery | Cambridge repository |
| Carhart-Harris 2016 LSD MEG ✓ | LSD 75 µg IV | 15 pairs | placebo crossover | Harvard Dataverse doi:10.7910/DVN/9Q1SKM, CC0 |
| Muthukumaraswamy 2015 ketamine MEG ✓ | ketamine | 18 pairs | placebo crossover | same Dataverse deposit |
| PsiConnect, Stoliker 2026 ✓ | psilocybin 19 mg | 62–65 | baseline vs drug session, BIDS | OpenNeuro ds006110, CC0 |
| Bajwa, Wisconsin ✓ | propofol sedation | 21 | awake / sedated, 65 ch, BIDS | OpenNeuro ds005620, CC0 |
| Niedernhuber 2026 | DMT + harmine | 25 | placebo + harmine-only | Zenodo 18850897 (preprocessed) |
| Farnes 2020 | ketamine | 10 | no placebo | Dryad (preprocessed) |
| Sutherland 2026 | psilocybin + midazolam | 8 | open label | Zenodo 17080375 (post-ICA) |

Restricted: Milan–Liège propofol/xenon/ketamine (Zenodo 806176, not yet
requested), Oxford ultraslow propofol (Zenodo 1168447), Stanford propofol
(Eagleman 2019), Imperial psilocybin dry-EEG (Godfrey/Lyons). Highest
value on request: Imperial DMT EEG-fMRI (Timmermann 2023, n = 20,
placebo), Imperial DMT EEG (Timmermann 2019, n = 13), psilocybin MEG
(Muthukumaraswamy 2013, n = 15), Prague psilocybin 256-ch time course
(Nikolic, n = 20), Zurich psilocybin rest + TMS-EEG (Ort 2023, n = 22),
UCL 5-MeO-DMT (Blackburne 2025, n = 29), Michigan ketamine and
sevoflurane. No statement but worth asking: Murray 2024 (LSD low dose,
n = 21) and Lewis-Healey 2026 (inhaled DMT, n = 19). OpenNeuro has no
LSD, DMT or ayahuasca EEG/MEG.

With what is open today a first pass covers DMT, LSD, psilocybin
(full dose and microdose), ketamine, DMT + harmine and propofol at
two sites: about 6 psychedelic and 2 anaesthesia datasets, versus
Girn's 11.

### Caveats

- The DMT recordings start at exhalation and run until the self-reported
  return (6 ± 1.4 min), with no event markers; they contain the whole
  time course from peak to comedown. The DMT − eyes-closed contrast is
  mostly the loss of the eyes-closed alpha state. DMT − eyes-open isolates
  the drug, but is also nonstationary (see the time course above).
- Psilocybin 0.5 g microdose is a null for every complexity measure; only the
  aperiodic offset/exponent move.
- The Maschke/Sarasso propofol-xenon-ketamine data (Zenodo 806176) are
  restricted and not yet requested.
