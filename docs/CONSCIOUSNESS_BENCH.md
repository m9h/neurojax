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

Not ported from Maschke 2024: avalanche exponents / DCC (needs `powerlaw`
fits and long recordings), Fano factor, avalanche repertoire, the
integrated-covariance width Δ, and specparam-based slope (the log-log fit is
used; `neurojax.analysis.specparam_analysis` needs `specparam` installed).

Every value in `nltsa.py` is a plain function with explicit defaults so a
result table can name the estimator and parameters; the table above is the
report.

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
| mean |r| of changes | 0.57–0.61 | 0.39–0.46 | 0.48 |
| participation ratio | 1.9–2.1 | 3.0–3.8 | 3.1 |
| first-PC share | 67–72 % | 44–53 % | 53 % |

Propofol compresses the set onto one axis (one cluster of ~20 measures:
LZ, entropies, fractal dimensions, Hjorth mobility, correlation dimension,
Lyapunov, RQA; DFA + Hjorth complexity as a second); psychedelics split it
into an amplitude-based block and an ordinal block (permutation entropy,
statistical complexity, Petrosian, Higuchi, Lyapunov). Mantel r between
contrasts: 0.90 within class, 0.81 anaesthesia-vs-psychedelic; mean
matrices r = 0.86 (p < 0.001, 5000 permutations), RV = 0.89. Largest
coupling changes: Higuchi–spectral entropy +0.69 → −0.32, permutation
entropy–spectral entropy +0.58 → −0.33, permutation entropy–SVD entropy
+0.83 → +0.15. Partialling the six spectral parameters lowers the mean |r|
only from 0.61 to 0.48–0.52 (propofol): the shared variance is the common
dependence on the full spectral shape, not on six summary parameters.

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

### Spectral/phasic decomposition (DMT study, first run with per-channel surrogates; rerun with the multivariate null in progress)

LZ, DFA, permutation entropy and the fractal dimensions are fully reproduced
by phase-randomised surrogates (spectral fraction 0.8–1.5). The 0-1 chaos
test and the Lyapunov exponent are the only univariate measures whose change
is mostly phasic; on the eyes-open contrast multiscale entropy, SVD entropy
and the Lyapunov exponent move *against* their spectral prediction.

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
  study (job re-queued after a crash; see below).
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
