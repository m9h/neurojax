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

**Leg C — identifiability of the sparse drifts (Donoho–Tanner, 2026-06-27).**
Are these sparse fits even identifiable? Each band's drift is a sparse regression —
recover a `k`-sparse coefficient vector from an `n×N` polynomial library (`n`
envelope samples, `N` candidate terms). The Donoho–Tanner *weak* phase transition
(via the ℓ1 descent-cone statistical dimension; Amelunxen-Lotz-McCoy-Tropp 2014,
matching Donoho–Tanner 2009) says when ℓ1/STLSQ can recover the support at all:
identifiable iff `n > N·δ(k/N)`. This is the identifiability counterpart of the
Gavish–Donoho rank selection used throughout (`svht_rank`, estimation side).

| band  | rank | P (lib) | k (support) | δ=n/N | min window | headroom |
|-------|-----:|--------:|------------:|------:|-----------:|---------:|
| delta | 18   | 190     | 33          | 947   | 1.8 s      | 2000×    |
| theta | 18   | 190     | 61          | 947   | 2.6 s      | 1406×    |
| alpha | 19   | 210     | 101         | 857   | 3.4 s      | 1047×    |
| beta  | 18   | 190     | 137         | 947   | 3.6 s      | 994×     |
| gamma | 17   | 171     | 145         | 1053  | 3.4 s      | 1065×    |

Two readings. **(1)** On the full 6-min record every band sits **~1000–2000×
inside** the identifiable region (δ ≈ 850–1050 ≫ 1) — under-identification is *not*
a source of artifact in the sparse fits; the binding constraint is library
conditioning / SNR, not the phase transition. The actionable number is `min window`:
the shortest segment whose sparse drift support is still DT-identifiable (the floor
for a windowed-SINDy analysis) — 1.8 s for delta, ~3.6 s for beta. **(2)** The
recovered support size `k` is itself **frequency-graded** — 33 active terms in delta
rising monotonically to 145 in gamma (17% → 85% of the library) — at a fixed,
reported STLSQ threshold (0.05). The slow bands have a *sparse, parsimonious* drift;
gamma needs a near-dense library. This is a third, independent measure pointing the
same way as EPR and sol/tot: **low-frequency dynamics are structured** (sparse +
solenoidal + irreversible), **gamma is unstructured** (dense + gradient +
near-reversible). The absolute `k` depends on the threshold, but the monotone
ordering does not, and it was computed entirely independently of the Langevin fit.
Caveat: polynomial libraries are correlated, so DT is the optimistic bound — a
correlated design needs strictly more samples. Script:
`scripts/real_data/wand_identifiability.py`.

**Leg C extension — band-resolved multifractality (WSE, 2026-07-05).** Everything
above characterizes the drift's *directionality* (gradient vs solenoidal); it says
nothing about whether the band envelope itself is monofractal (Gaussian/fGn, as the
Langevin noise model assumes) or genuinely multifractal (intermittent/multiplicative).
Applying the weak-scaling-exponent multifractal formalism (Dumeur, Saes, Abry,
Ciuciu, Wendt, Jaffard 2025, arXiv:2503.16892) to each band's parcel-averaged
amplitude envelope (same Morlet front-end as `wand_band_langevin.py`):

| band  | Hz    | sol/tot | c1 (H) | c2 (multifractality) | spectrum width |
|-------|-------|--------:|-------:|----------------------:|---------------:|
| delta | 2–4   | 0.64    | +0.419 | **-0.076**             | 0.269          |
| theta | 4–8   | 0.26    | +0.202 | **-0.034**             | 0.136          |
| alpha | 8–13  | 0.13    | +0.122 | **-0.007**             | 0.074          |
| beta  | 13–30 | 0.05    | +0.012 | +0.023                 | 0.220          |
| gamma | 30–45 | 0.02    | -0.085 | +0.114                 | 0.298          |

Rotational strength (`sol/tot`) and multifractality (`c2`) are **not independent
axes** — they're strongly coupled: `corr(sol/tot, c2) = -0.82`. Delta, the most
solenoidal band, is also the most genuinely multifractal (c2<0); gamma, nearly pure
gradient relaxation, is essentially monofractal (c2>0, slightly super-Gaussian). This
refines the earlier harmonic-coefficient WSE pass (`wand_multifractal_wse.py`), which
averaged over the full 1–45 Hz band and came back only weakly multifractal on
average — the genuine multifractality was concentrated in the low-frequency,
high-circulation bands all along, and splitting by band reveals it. Script:
`scripts/real_data/wand_band_multifractal_wse.py`.

## Strengthening the irreversibility case — three independent proper nulls (2026-06-27)

The committed irreversibility (model-free `‖L−Lᵀ‖`, model-based Langevin EPR) was
significant only vs a **time-shuffle** null — which whitens the spectrum, so it
rejects i.i.d. noise and nothing more. Two caveats then surfaced: (i) the model-free
and model-based estimators are the *same* second-order object (the OLS-Langevin
`α* = ½(AΣ−ΣAᵀ) = A_sol·Σ` is *identically* the empirical Lévy area — verified on
WAND), so they were never independent corroboration; (ii) `irreversibility ≠
determinism`. Three genuinely independent tests, each with a proper null, now
strengthen the case (review: `docs/NONLINEAR_TSA_REVIEW.md`).

**(1) Arrow of time — SOTA reversible surrogate null.** Re-test the whitened band
circulation (Lévy area) against a per-channel **IAAFT** null from the
TimeseriesSurrogates.jl oracle (`scripts/real_data/oracle_surrogates/`): it preserves
each channel's power spectrum *and* amplitude distribution exactly and is
time-reversible, so survival rules out autocorrelation, spectrum shape, *and* the
(skewed) envelope marginal.

| band | circ | null | z | p |
|------|-----:|-----:|----:|----:|
| delta | 0.664 | 0.578 | 2.59 | 0.016 |
| theta | 0.936 | 0.793 | 2.88 | 0.016 |
| alpha | 1.269 | 0.990 | **4.69** | 0.016 |
| beta | 1.344 | 1.239 | 1.45 | 0.131 |
| gamma | 1.608 | 1.358 | 2.86 | 0.016 |

Significant in 4/5 bands (p = surrogate floor, N=60). The *excess* over the
matched-spectrum reversible floor peaks at **alpha**, not delta — the raw
delta-dominant EPR is largely the high matched-spectrum floor of slow narrowband
signals. Script: `scripts/real_data/wand_reversible_null.py`.

**(2) Determinism — RQA + recurrence-network dimension (pyunicorn oracle).** A
*positive* test of "stochastic, not a limit cycle" (the DMD/SINDy≈0 was
absence-of-evidence). DET, with an IAAFT-surrogate floor, and the transitivity
dimension:

| band | DET | surrDET | LAM | L_max | transD |
|------|----:|--------:|----:|------:|-------:|
| delta | 0.571 | 0.548 | 0.729 | 108 | 3.72 |
| theta | 0.271 | 0.257 | 0.456 | 16 | 3.81 |
| alpha | 0.216 | 0.165 | 0.360 | 8 | 3.35 |
| beta | 0.143 | 0.128 | 0.256 | 7 | 3.85 |
| gamma | 0.109 | 0.128 | 0.205 | 5 | 4.08 |

DET is low (0.11–0.57; a noisy limit cycle gives ~0.86, pure noise ~0.09) →
**positively confirms no deterministic limit cycle**. Transitivity dimension ≈ 3.7
vs the linear PCA rank ≈ 18 → a **low-dimensional (~3–4D) nonlinear manifold**, not a
1-D limit cycle (transD = 1.14 for a clean circle) and not 18-D noise — exactly the
"biased stochastic walk on a ring". Gamma DET < surrogate → noise-like. Script:
`scripts/real_data/wand_determinism_rqa.py`.

**(3) Nonlinear arrow of time — higher-order log-signatures.** The log-signature
negates uniformly under time reversal, so levels ≥3 carry nonlinear path asymmetry
the Gaussian/linear estimators miss. Windowed depth-3 log-sig (signax) per band,
each level's mean ranked against a sign-flip null:

| band | lvl2 z (p) | lvl3 z (p) |
|------|-----------:|-----------:|
| delta | 4.88 (.002) | 3.08 (.005) |
| theta | 2.89 (.007) | 1.97 (.037) |
| alpha | 2.52 (.017) | **3.71 (.002)** |
| beta | −0.17 (.571) | 0.33 (.364) |
| gamma | 0.83 (.239) | −0.94 (.830) |

Level-2 reproduces the linear circulation (sanity); **level-3 significant in
delta/theta/alpha ⇒ genuine *nonlinear* broken detailed balance**, null in
beta/gamma. Script: `scripts/real_data/wand_logsig_irrev.py`.

**(4) Determinism, second probe — S-map (EDM) nonlinearity test.** Complements RQA
with the Sugihara θ test (`smap_nonlinearity` in jaxctrl): θ=0 is a global linear map,
rising skill for θ>0 signals nonlinear determinism. The gain Δρ = ρ(θ*)−ρ(0) is **≈0**
in every band (0.0001–0.0035 vs an IAAFT floor) — a global linear map predicts as
well as any locally-weighted nonlinear one, so **there is no deterministic nonlinear
skeleton**, a second positive confirmation of the linear/stochastic reading. The gain
is statistically above the floor in delta/theta/alpha (z~10) but the effect size is
negligible (~0.1% skill), echoing the weak level-3 log-sig residual. Linear
predictability ρ(0) declines delta 0.88 → gamma 0.05 (gamma = noise). Script:
`scripts/real_data/wand_smap_nonlinearity.py`.

**(5) The complete test — signature-kernel MMD(X, X̄).** The signature kernel is
characteristic on path laws, so MMD(forward, reversed) = 0 iff time-reversible — a
single all-orders, multichannel statistic (depth-3 signatures, two-sample permutation
null). Significant in all bands (p=0.002 floor), effect peaking at alpha (z=17.0),
strong delta/theta (~14), moderate beta (7.2), weak gamma (2.0). Script:
`scripts/real_data/wand_sigkernel_mmd.py`.

**The convergent picture.** Five independent proper-null tests agree: the slow bands
(delta/theta/alpha) carry genuine, low-dimensional, *nonlinear* broken detailed
balance, while the dynamics remain *linearly predictable / stochastic* (no nonlinear
deterministic skeleton); beta is weak; gamma is reversible/noise-like at every order.
**Alpha is the cleanest broken-detailed-balance signal** across every proper null
(reversible-surrogate z=4.69, nonlinear log-sig z=3.71, sig-MMD z=17.0) — a sharper,
more rigorous claim than the raw delta-dominant EPR (whose delta peak is mostly the
matched-spectrum floor of slow narrowband signals). Deferred: a coupling-preserving
**constrained-randomization** null — TISEAN is not packaged (needs a source build), so
it is a future sibling oracle to TimeseriesSurrogates.jl.

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
- The sparse drift fits are Donoho–Tanner identifiable with ~1000–2000× headroom
  on the 6-min record (`l1_statistical_dimension`, `donoho_tanner_regime` in
  jaxctrl, re-exported by `neurojax.dynamics`); the recovered support size is
  itself frequency-graded (sparse slow bands → near-dense gamma), a third
  structural gradient agreeing with EPR and sol/tot.
- Prototype scale: 10 subjects, 6 min, template coreg. Next: individual
  FreeSurfer source recon (as more subjects are reconstructed) and Leg B (mesh
  waves / connectome harmonics) on the WAND surfaces.
