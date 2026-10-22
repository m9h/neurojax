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
| Dataset loaders (skip if missing) | `src/neurojax/io/consciousness_bench.py` |
| Downloads | `scripts/consciousness_bench/download_datasets.sh` |
| Driver (Slurm) | `scripts/consciousness_bench/run_bench.py`, `run_bench.sbatch` |
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
| rqa_determinism | recurrence | m=3, RR=5 %, l_min=2 | Marwan 2007 |
| pair_correlation | criticality (multivariate) | N·Var|R(t)|, alpha band | Yoon 2015; Maschke 2024 |
| branching_ratio | criticality (multivariate) | |z|≥2 SD events, 4 ms bins | Beggs & Plenz 2003; Maschke 2024 |

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
`phasic_z`) to `/data/datasets/consciousness-bench/<dataset>/`.
