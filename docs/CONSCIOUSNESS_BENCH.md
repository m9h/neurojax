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
| Measure registry (23 named estimators, parameters, references) | `src/neurojax/analysis/nltsa.py` (`MEASURES`) |
| LZ variants + spectral/phasic decomposition | `src/neurojax/analysis/complexity.py` |
| Harness: epoching, scoring, paired contrasts, decomposition | `src/neurojax/bench/consciousness_bench.py` |
| Spectral baselines: specparam/FOOOF, band powers, SEF95, alpha peak | `src/neurojax/analysis/spectral_baselines.py` (`SPECTRAL_MEASURES`) |
| Reference-library parity (antropy, NeuroKit2, nolds) | `src/neurojax/analysis/tests/test_nltsa_oracles.py` |
| Dataset loaders (skip if missing) | `src/neurojax/io/consciousness_bench.py` |
| Downloads | `scripts/consciousness_bench/download_datasets.sh` |
| Driver (Slurm) | `scripts/consciousness_bench/run_bench.py` (`--measures all|nltsa|spectral`, `--tag`), `run_bench.sbatch` |
| Spectral-explained analysis | `scripts/consciousness_bench/spectral_explained.py` → `bench.consciousness_bench.spectral_explained` |
| Tests | `src/neurojax/analysis/tests/test_nltsa.py`, `src/neurojax/bench/tests/test_consciousness_bench.py`, `src/neurojax/io/tests/test_consciousness_bench_io.py` |

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
- **DMD / Koopman**: `bench.sensitivity.dmd_lowmode_removal` is the
  low-frequency treatment; the same Hankel-DMD spectrum per epoch (mode
  frequencies, growth rates, mode amplitudes) is the modal description of
  the state and belongs in the measure table beside the specparam fit.

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

### Caveats

- The DMT recordings (~7 min) are longer than the rest recordings (~5 min)
  and the Zenodo record has no event timing, so they include pre-inhalation
  and comedown; the DMT − eyes-closed contrast is mostly the loss of the
  eyes-closed alpha state. DMT − eyes-open isolates the drug.
- Psilocybin 0.5 g microdose is a null for every complexity measure; only the
  aperiodic offset/exponent move.
- The Maschke/Sarasso propofol-xenon-ketamine data (Zenodo 806176) are
  restricted and not yet requested.
