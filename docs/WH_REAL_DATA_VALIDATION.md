# Real-data validation — Wakeman-Henson MEG (2026-06-24)

Tested the JAX dynamics models on **real MEG that Woolrich's group used**: the
Wakeman-Henson face-recognition dataset (OpenNeuro `ds000117`) — the
"elekta_task" data in the osl-dynamics toolbox paper (Gohil et al. 2024).

## Setup
- Subject sub-05, run-01 (491 s, 102 magnetometers @ 1100 Hz).
- Sensor-level (à la Gohil/Woolrich 2025 "Canonical HMM", no source recon).
- Preparation via **osl-dynamics** `Data.prepare`: band-pass 1–45 Hz, resample
  250 Hz, TDE (15 embeddings) + PCA (40 comps) + standardize → (122 736, 40).
- The *same* prepared array feeds both the osl-dynamics oracle and the JAX models.
- Scripts: `scripts/real_data/wh_prep_oracle.py` (osl env) and
  `wh_jax_compare.py` (JAX/GPU env).

## Results
All three JAX models trained on real MEG **on the GB10 GPU**:

| Model | Outcome on real Wakeman-Henson MEG |
|-------|-----------------------------------|
| **GaussianHMM** | fit; vs osl-dynamics oracle — see below |
| **DyNeMo** | ELBO 60.7 → 50.9 (trains); collapses toward 1 dominant mode |
| **M-DyNeMo** | ELBO 56.6 → 45.0; **power/FC time-course corr = −0.10** — the defining decoupling holds on real MEG |

**HMM vs osl-dynamics oracle (identical prepared data):**
- State-segmentation agreement **0.567**, against a **ceiling of 0.725** (the two
  fits converged to different occupancy profiles, so 1.0 is unreachable).
- Dominant-state timepoint overlap **0.649**; both find one dominant background
  state (JAX 0.51, oracle 0.79) plus transients.

## Interpretation
The JAX models run and train on real Woolrich MEG on the GPU — the core result.
The moderate HMM agreement reflects (1) the **Baum-Welch (closed-form) vs
osl-dynamics SGD** estimator difference already documented on synthetic data and
in the oracle parity tests, and (2) a single 8-minute run, which yields a
degenerate dominant-state regime where the two algorithms partition the
background differently. Both nonetheless localise the same dominant state.

## Cleaner comparison (next step)
For a stronger parity number: concatenate multiple runs/subjects (less
degenerate), and compare **state network maps** (covariance matrices matched by
similarity) — the standard osl-dynamics HMM comparison — rather than only
per-timepoint segmentation, which is sensitive to label timing.
