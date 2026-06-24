# Real-data validation — Wakeman-Henson MEG (2026-06-24)

Tested the JAX dynamics models on **real MEG that Woolrich's group used**: the
Wakeman-Henson face-recognition dataset (OpenNeuro `ds000117`) — the
"elekta_task" data in the osl-dynamics toolbox paper (Gohil et al. 2024).

## Setup
- Subject sub-05, **all 6 runs** (~49 min, 102 magnetometers @ 1100 Hz).
- Sensor-level (a la Gohil/Woolrich 2025 "Canonical HMM", no source recon).
- Preparation via **osl-dynamics** `Data.prepare`: band-pass 1-45 Hz, resample
  250 Hz, TDE (15 embeddings) + PCA (40 comps) + standardize -> (741666, 40).
- The *same* prepared array feeds both the osl-dynamics oracle and the JAX models.
- Scripts: `scripts/real_data/wh_prep_oracle.py` (osl env) and
  `wh_jax_compare.py` (JAX/GPU env).

## Results — multi-run (headline)
All three JAX models trained on real MEG **on the GB10 GPU**. A 6-state
covariance-only TDE-HMM was fit by both osl-dynamics (oracle) and the JAX
Baum-Welch implementation on identical data, then states were matched by
**network-map (covariance) similarity** — the standard osl-dynamics comparison.

| Metric (JAX HMM vs osl-dynamics oracle) | Value |
|------------------------------------------|-------|
| State network-map correlation (matched), mean | **0.859** |
| per state | 0.992, 0.989, 0.988, 0.982, 0.727, 0.477 |
| Per-timepoint segmentation agreement (cov-matched) | **0.951** |
| Fractional occupancy — JAX | 0.005, 0.027, 0.16, 0.161, 0.323, 0.324 |
| Fractional occupancy — oracle | 0.006, 0.023, 0.16, 0.16, 0.325, 0.325 |

Four of six state networks correlate **> 0.98**; the two weak ones (0.48, 0.73)
are precisely the low-occupancy transient states (0.5%, 2.7%) where the
covariance estimate is noisy. Occupancy profiles are near-identical.

- **DyNeMo**: ELBO 701 -> 43 (trains).
- **M-DyNeMo**: ELBO 54 -> 40; power/FC time-course corr = 0.25 — the defining
  decoupling holds on real MEG.

## Single-run (initial, for contrast)
One 8-minute run gave a degenerate dominant-state HMM (one state at 0.78
occupancy) and only **0.567** segmentation agreement (ceiling 0.725). More data
removes the degeneracy — hence the multi-run result above.

## Interpretation
On a stable, well-powered fit the JAX HMM **reproduces osl-dynamics' brain
network states on real Woolrich MEG**: 95% temporal agreement, near-identical
occupancy, and >0.98 network-map correlation for every well-occupied state.
Residual differences sit in the lowest-occupancy transients and reflect the
Baum-Welch (closed-form) vs osl-dynamics SGD estimator difference documented in
the synthetic oracle parity tests.
