# Full dynamics-technique comparison on WAND resting MEG (2026-06-28)

Every dynamics method we built, run on the same WAND resting-state MEG (10 subjects,
Desikan-68 source parcels), asking one question: **what kind of dynamical object is
the resting cycle?**  The striking result is that ~a dozen independent techniques —
discrete-state, continuous-flow, governing-equation, recurrence, information-theoretic,
rough-path, and physical-flow — **converge on the same answer**.

## The convergent answer
The WAND resting cycle is a **stochastic, low-dimensional, broken-detailed-balance
(solenoidal) cycle — not a deterministic limit cycle**: a noise-driven biased walk on
a ring, with a genuine probability current, rotating among a few low-order connectome
harmonics at an infraslow ~0.1 Hz, and realised in physical space as cortical
vortices expressing those same harmonics.

## The comparison

| technique | what it measures | WAND finding | verdict on the cycle |
|---|---|---|---|
| **HMM + TINDA** (Leg A) | discrete state-transition cycle | S=+0.052, z=30.8 vs block-shuffle | a significant directional cycle exists |
| **CEBRA** | latent topology | ring coverage 1.00 | the state space is cyclic (a ring) |
| **DMD / SINDy / DYSCO** | deterministic drift rotation | ≈ 0 net rotation | **not** a deterministic limit cycle |
| **S-map θ** (EDM) | nonlinear determinism | Δρ ≈ 0 (vs IAAFT) | linear/stochastic, no deterministic skeleton |
| **RQA** (pyunicorn) | determinism + dimension | DET 0.11–0.57 (noise-end); transD ≈ 3.7 vs PCA-rank 18 | low-dim **stochastic** manifold, not a limit cycle |
| **Langevin / Fokker–Planck** | drift gradient⊕solenoidal split, EPR | EPR 0.11–1.85; f_sol 0.04–0.12 Hz; A_sol≠0 | genuine **solenoidal circulation** (broken detailed balance) |
| **Lévy area / log-sig** (rough path) | path circulation, time-asymmetry | α*=A_sol·Σ (identity); level-3 nonlinear irreversibility (alpha cleanest) | the circulation is real + weakly **nonlinear** |
| **Reversible IAAFT null** (TimeseriesSurrogates.jl) | vs spectrum+marginal-matched reversible | circulation > null 4/5 bands (z 2.6–4.7) | irreversibility survives a proper null |
| **Signature-kernel MMD** | all-orders time-reversibility | significant all bands (alpha z=17) | statistically time-irreversible |
| **Connectome harmonics** (Leg B) | spatial eigenbasis of the cycle | circulation among low-order harmonics; EPR≈0.10, z 4.4–5.7; basis-invariant (geometric≈structural≈surface-LBO) | the cycle = rotation among a few **low-order structural modes** |
| **Transfer entropy** (multivariate Gaussian, conditional) | directed information flow among harmonics | 9 significant edges (p<0.01 vs 100 circular-shift surrogates); directed cycles at lengths 2–7 among H1–H15 | independent **information-theoretic** confirmation of the directed loop |
| **WSE multifractal** (weak scaling exponent, `pymultifracs`) | per-harmonic regularity + multifractality (c1, c2 log-cumulants) | all 15 harmonics c1<0 (H_min-like, negative); c2 small, mean +0.008, only 3/15 slightly negative | **weakly multifractal / near-monofractal** — consistent with the existing Gaussian/fGn Langevin noise model, no strong hidden multiplicative structure |
| **Phase-flow routing** (Vinão-Carl) | physical-space Hodge: vortices/sources | vortices express harmonics H[1,2,7]; net rotation bias +0.10; rerouting ~4/s | physical **vortices** = the spatial realization of the circulation |
| **Donoho–Tanner** | sparse-fit identifiability | ~1000–2000× inside the identifiable region; support frequency-graded | the fits are well-posed; complexity ↑ with frequency |

## Why they agree — one Helmholtz/Hodge split in four spaces
The non-equilibrium (solenoidal) signature is the common thread, and it is literally
the **same gradient⊕solenoidal decomposition** viewed in four coordinate systems:

| space | gradient (relaxation) | **solenoidal (the cycle)** | tool |
|---|---|---|---|
| discrete states | dwell / self-transitions | **TINDA directional cycle** | `transition_flux` |
| **state space** | `A_rev = −DΣ⁻¹` | **`A_sol` / EPR / Lévy area** | `jaxctrl._circulation` |
| **physical space** | sources/sinks `∇·F` | **vortices `∇×F`** | `geometry.hodge` |
| spatial basis | high-order harmonic decay | **rotation among low-order harmonics** | `spatial.connectome_harmonics` |

The state-space circulation `A_sol·Σ` (Tomita–Tomita = expected Lévy area), the
physical-space vortices, and the low-order harmonic rotation are the **same object**,
empirically: the phase-flow vortices project onto the same harmonics (H1/H2/H7) the
Langevin circulation rotates among, at the same ~0.1 Hz, both significant against
proper reversible nulls. Multivariate transfer entropy among the harmonics adds a
fourth, purely information-theoretic view of the same object: the directed loops it
finds (H14→H2, H14→H7, H7→H14, H11→H8, H7→H11, ...) are the TE signature of a
solenoidal (non-gradient) coupling structure, not a feed-forward cascade.

The WSE multifractal analysis (Dumeur et al. 2025) checks a different axis entirely:
not *is there a directed cycle*, but *is each harmonic's own dynamics simple
(Gaussian/monofractal, as the Langevin model assumes) or intermittent/multifractal*.
All 15 harmonics come back weakly multifractal at most (c2 ≈ 0) — no evidence the
Langevin/fGn-style noise model is missing real multiplicative structure. Notably, all
15 also have negative c1 (H_min-like exponent), which is exactly the condition the
paper shows breaks the standard wavelet p-leader formalism outright (without an ad hoc
large fractional-integration order) — a concrete case where the older multifractal
tools would have been the wrong tool for this data.

## The stack (all differentiable-JAX-native)
`HMM/DyNeMo (states) → phase-flow Hodge (physical routing, mesh + point-cloud
backends) → Langevin/circulation (state space) → connectome harmonics (structural
basis) → SMNI path-integral (stochastic field)` — every layer a Helmholtz/Hodge split,
every layer `jax.grad`/`vmap`-able. Scripts: `wand_*` in `scripts/real_data/`; tools
in `jaxctrl` (`_circulation`, `_langevin`, `_denoise`, `_sysid`) and `neurojax`
(`geometry.hodge[_pointcloud]`, `analysis.routing`, `spatial.harmonics`,
`analysis.timefreq`).

## Cohort-scale extension: individual-anatomy source connectivity (n=27, 2026-07-06)
A separate, complementary pipeline (`scripts/run_wand_recon.py` +
`scripts/run_wand_cohort.py`) replaces the template (fsaverage) coregistration
above with each subject's **own** FreeSurfer surfaces, a single-sphere MEG
conductor, and head-digitization coregistration — real individual anatomy, not
harmonic/template space. Now runs on the full 27/27 FreeSurfer-complete +
resting-MEG WAND cohort: the one prior holdout (`sub-14445`) failed oct6 source-
space setup on an MNE-side icosahedral-decimation edge case (FreeSurfer's own
recon-all reported no error) — `run_wand_recon.py` now falls back to oct5/oct4
when oct6 fails, at the cost of a coarser source space for that one subject
(33/68 usable parcels vs. the typical 63-68). Aggregated as mean directed
PDC/DTF over aparc parcels common to all 27 subjects (23/68 — down from 41/68 at
n=26, since sub-14445's coarse oct4 recon shrinks the common-parcel
intersection).

Top leakage-clean alpha-band directed edges (group mean, n=27):

| driver | receiver | PDC | leakage |
|---|---|---|---|
| precentral-rh | paracentral-rh | 0.141 | 0.26 |
| posteriorcingulate-rh | paracentral-rh | 0.120 | 0.13 |
| precuneus-rh | paracentral-rh | 0.118 | 0.33 |
| postcentral-rh | precentral-rh | 0.117 | 0.64 |
| precentral-rh | postcentral-rh | 0.114 | 0.64 |

Sensorimotor (precentral/paracentral/postcentral) and orbitofrontal↔cingulate
directed coupling dominate — the same qualitative pattern as the n=26 pass
despite the smaller common-parcel set, a real-anatomy, larger-N (27 vs 10)
sanity check that directed structure exists and is consistent across subjects.

**Reconciliation with the harmonic circulation (2026-07-06) — partial, basis-dependent, not a
clean confirmation.** Computed against the prior n=26/41-common-parcel cohort snapshot (before
`sub-14445` was added at n=27/23-common-parcels above); not yet rerun on the updated file.
Expressed the group alpha-band directed-PDC matrix (41 common parcels) as a
bilinear form in the same harmonic coordinates as the circulation analysis (`Phi41.T @ P @ Phi41`,
`Phi41` = the 68-node harmonic eigenvectors restricted to the 41 available rows by parcel name),
took the antisymmetric part as the PDC analogue of the Langevin circulation's `A_sol`, and tested
two things against parcel-identity permutation nulls (2000 permutations):
`scripts/real_data/wand_reconcile_pdc_harmonics.py`.

| basis | rotational-structure COM (of 20) | vs null | flagged-harmonic loading enrichment | vs null |
|---|---|---|---|---|
| geometric | 9.34 (ref: ~9.3) | z=−1.21, p=0.118 (n.s.) | **0.70×** (top-PDC parcels *less* loaded on H1,H2,H7) | z=−2.65, p=0.9995 (n.s., wrong direction) |
| structural | 11.23 (ref: ~10.6) | z=0.03, p=0.444 (n.s.) | **1.75×** (top-PDC parcels *more* loaded on H13,H8,H15) | z=1.81, **p=0.059** (marginal) |

Two things line up loosely — the geometric-basis centre-of-mass matches the template-coreg
reference almost exactly (9.34 vs 9.3), and H1/H7 recur among the top rotational pairs found here
(H1↔H7, H1↔H6, H6↔H7, H7↔H13) even though the specific partner harmonics (H6, H13) differ from the
reference's (H12, H8, H2) — but neither permutation test reaches significance in the geometric
basis, and the loading-enrichment check there is significant in the *wrong* direction (the
anatomically PDC-active regions are *less* loaded on H1/H2/H7 than an average parcel, not more).
The structural basis gives the one suggestively positive number (1.75× enrichment, p=0.059) but its
rotational-structure test is null. **Verdict: this reconciliation attempt does not yet cleanly tie
the cohort's real-anatomy directed structure to the harmonic circulation** — the two pipelines
differ in more ways than just anatomy (alpha-band-restricted PDC vs the circulation's broadband
Langevin fit; group-averaged directed connectivity vs a per-subject dynamical fit; largely
non-overlapping subject sets, template n=10 vs individual-anatomy n=26), any of which could dilute
a real correspondence rather than its absence being the true state of affairs. Sharper next
attempt: broadband (not alpha-only) PDC, and/or fit the harmonic circulation itself on the same
n=26 individual-anatomy subjects so both legs use the same subjects and coregistration.

## Caveats / scope
- 10 subjects, template (fsaverage) coregistration, Desikan-68. Cohort scale-up needs
  per-subject FS/TRACULA (FastSurfer seg path; bedpostx is the GPU bottleneck).
- WSE multifractal analysis ran on the same subject-concatenated `harmonic_coeffs.npy`
  as the TE analysis (~10 subjects x 6 min @ 25 Hz). The coarsest scale examined
  (2^12 samples ~ 164s) approaches a sizeable fraction of one subject's segment, and
  internal subject-concatenation boundaries are not NaN-masked the way the series
  edges are — a per-subject rerun would rule out coarse-scale boundary leakage, though
  the result is consistent across all 15 harmonics.
- Phase-flow uses alpha-band phase (narrowband, as Vinão-Carl do) — distinct from the
  band-free harmonic dynamics.
- Reference oracles studied (licenses audited): MARBLE (MIT, ported), HADES/CHAP (no
  license, read-only), lapy/BrainEigenmodes (open).
- Transfer entropy: the JAX-native Gaussian estimator (`jaxctrl._information`,
  `scripts/real_data/wand_jax_te.py`) ran to completion; the IDTxl oracle comparison
  run did not produce output and has not been re-run, so the TE result above is not
  yet cross-checked against the reference (nonlinear/KSG) estimator.
