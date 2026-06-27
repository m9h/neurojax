# Three-leg cross-test on WAND resting MEG (prototype, 2026-06-26)

First run of the state-cycle / waves / governing-dynamics cross-test on the WAND
cohort — the project's own 169-subject resting-state MEG dataset.

## Setup
- 10 WAND subjects, resting CTF MEG (1200 Hz), 6 min each.
- fsaverage **template** coregistration (only sub-08033 has individual FreeSurfer)
  -> LCMV beamformer -> **Desikan-68** parcellation -> TDE-PCA -> **K=12 HMM**.
- 900k samples of source-parcellated resting MEG; HMM occupancy spread across all
  12 states. Scripts: `scripts/real_data/wand_source_prep.py`, `wand_legc.py`,
  `wh_k12_cycle.py`.

## Results
**Leg A — TINDA structured cycle (the discrete view).**
Cycle strength **S = +0.052**, vs block-shuffle null +0.0095 ± 0.0014 →
**z = 30.8, a significant directional cycle** on real WAND resting MEG.

**Leg C — governing dynamics (the continuous view).**
- **CEBRA**: the 12-state trajectory embeds as a **full ring** (circle coverage
  **1.00**) — the cyclic *topology* is recovered independently of TINDA.
- **DMD, SINDy, DYSCO, and an assumption-free phase-slope**: all give a net
  rotation frequency of **~0 Hz** on the ring.

## Interpretation — the cycle is STOCHASTIC, not a limit cycle
The ring is real (TINDA + CEBRA agree the state space is cyclic) but there is **no
deterministic net rotation** around it. The system performs a **stochastically-
driven biased walk on a ring**, not a clean rotating attractor. This is exactly
van Es et al.'s framing: *individual state transitions are stochastic; only the
aggregate ordering is cyclical*. TINDA detects that as a small statistical bias
(S = 0.05) — significant only because n ≈ 9×10⁵ — and the drift-only
governing-equation methods correctly report ~0 because the directional signal is
in the noise statistics, not a deterministic flow.

This discriminates a **stochastic cyclic process** from a **deterministic limit
cycle** — a stronger, more biologically-correct outcome than a forced frequency
match, and it cross-validates: a weak TINDA bias ⇔ ~0 deterministic rotation.

**Leg C extension — band-resolved irreversibility (time-frequency, 2026-06-27).**
The smoothed/single-timescale view collapses irreversibility to ~0. Going to a
joint time-frequency front-end (EEGLAB-style complex Morlet → amplitude envelope →
Donoho-shrunk whitening) and measuring the time-reversal asymmetry ‖L−Lᵀ‖ of the
lagged envelope covariance (INSIDEOUT/entropy-production proxy; Deco 2022,
Tewarie 2023) at a 120 ms network lag — vs a time-shuffle null — recovers it, and
shows it is **frequency-resolved**, not flat:

| band  | Hz    | irrev | null  | ratio |
|-------|-------|-------|-------|-------|
| delta | 2–4   | 0.402 | 0.175 | 2.29  |
| theta | 4–8   | 0.451 | 0.167 | 2.70  |
| alpha | 8–13  | 0.419 | 0.176 | 2.38  |
| beta  | 13–30 | 0.309 | 0.172 | 1.80  |
| gamma | 30–45 | 0.215 | 0.178 | 1.21  |

Every band sits above its shuffle null (irreversibility *is* present), but the
asymmetry is **low-frequency dominant** — peaks at theta and declines
monotonically to near-null at gamma. The network-timescale lead-lag structure
lives in the slow envelopes (delta–alpha); gamma envelopes are near-noise at a
120 ms lag. A single-timescale analysis averages this 2.7→1.2 gradient down to one
near-null number — which is why the whole-band entropy-production came out flat.
(Note: an earlier pass used an inverted null — dividing by ‖L+Lᵀ‖ — and mistakenly
read a beta/gamma peak; the table above, with the raw ‖L−Lᵀ‖ asymmetry and the
shrunk whitening, is the corrected result.) Script:
`scripts/real_data/wand_timefreq_irrev.py`.

**Leg C extension — band-resolved Langevin (model-based, 2026-06-27).**
The model-free ‖L−Lᵀ‖ asymmetry shows irreversibility is present and
frequency-resolved, but it conflates a genuine solenoidal (broken-detailed-balance)
*drift* with a plain amplitude/diffusion effect. Fitting a linear Langevin
`dz = A z dt + √(2D) dW` per band (Gavish-Donoho optimal-rank PCA embedding → drift
via the 1st Kramers–Moyal moment, diffusion via the 2nd) separates them: the
gradient drift `A_rev = −D Σ⁻¹`, the solenoidal drift `A_sol = A + D Σ⁻¹`, and the
entropy-production rate `Ṡ = tr(A_sol Σ A_solᵀ D⁻¹) ≥ 0` (invariant under any
invertible linear change of coordinates, so the only reported choice is the rank):

| band  | Hz    | rank | EPR   | null  | EPR/null | f_sol/Hz | sol/tot |
|-------|-------|-----:|------:|------:|---------:|---------:|--------:|
| delta | 2–4   | 18   | 1.850 | 0.048 | 38.5     | 0.044    | 0.64    |
| theta | 4–8   | 18   | 0.909 | 0.042 | 21.6     | 0.061    | 0.26    |
| alpha | 8–13  | 19   | 0.636 | 0.046 | 13.8     | 0.074    | 0.13    |
| beta  | 13–30 | 18   | 0.220 | 0.049 | 4.5      | 0.089    | 0.05    |
| gamma | 30–45 | 17   | 0.110 | 0.029 | 3.8      | 0.115    | 0.02    |

The verdict is **a genuine solenoidal drift, not a diffusion artifact**: entropy
production sits **3.8–38× above the shuffle null** in every band, and Ṡ by
construction isolates the irreversible part of the drift (with D⁻¹ weighting), so
the low-frequency-dominant asymmetry *is* a real probability current. Three things
the fit adds over the model-free proxy: (1) `sol/tot` = ‖A_sol‖/‖A‖ is scale-free
and collapses **0.64 → 0.02** delta→gamma — slow-band drift is 64% rotational,
gamma drift is 98% pure gradient relaxation; the dynamics change *character* with
frequency. (2) `f_sol` = |Im λ(A_sol)|/2π puts an actual frequency on the cycle
DMD/SINDy/DYSCO all returned as ~0: **0.04–0.12 Hz**, an *infraslow* rotation
(8–23 s period) of the band envelopes — buried in the noise statistics, invisible
to a drift-only deterministic fit but recovered from the solenoidal eigenvalue.
(3) the peak shifts theta→delta versus the model-free metric (the D⁻¹-weighted
drift-based estimator pushes it one band lower), but both agree on the headline.
This is exactly the SMNI-style "cyclic bias in the noise-induced term" predicted
below. Script: `scripts/real_data/wand_band_langevin.py`.

## Connection to SMNI (and the right next tool)
The cyclic structure lives in the **diffusion**, not the **drift**. DMD/SINDy/
DYSCO model the drift ż = f(z), so they see ~0 — as they should. Capturing the
cycle needs the **drift + diffusion** description, i.e. a **Langevin / Fokker–
Planck** estimator (Boninsegna; Frishman–Ronceray), which is the data-driven
member of Ingber's SMNI family. That is the indicated Leg-C extension for
jaxctrl: estimate μ(z) and D(z) from the trajectory's conditional moments, and
test whether the cyclic bias appears in the noise-induced (diffusion) term — the
quantity SMNI derives mechanistically from columnar firing statistics.

## What worked / scope
- Pipeline runs end-to-end on real WAND CTF resting MEG (template coreg, Desikan,
  K=12 HMM, TINDA, CEBRA, SINDy, DMD, DYSCO, Langevin/Fokker–Planck, and the
  time-frequency irreversibility front-end) on the GB10.
- The Langevin/Fokker–Planck estimator anticipated above is now implemented in
  jaxctrl (`fit_linear_langevin`, gradient/solenoidal split, entropy production)
  and re-exported by `neurojax.dynamics`; the two band-resolved tables above are
  its model-free (‖L−Lᵀ‖) and model-based (Ṡ, f_sol, sol/tot) views, which agree
  that the irreversibility is a real, low-frequency-dominant solenoidal current.
- Prototype scale: 10 subjects, 6 min, template coreg. Next: individual
  FreeSurfer source recon (as more subjects are reconstructed), Leg B (mesh waves)
  on the WAND surfaces, and a Donoho–Tanner identifiability check on the sparse
  fits (does the (rank, sample) regime sit below the phase-transition boundary
  where the recovered drift is identifiable?).
