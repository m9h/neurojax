# Connectome harmonics as the unifying spatial basis (2026-06-26)

Selen Atasoy's connectome harmonics give the *anatomical spatial eigenbasis* that
ties together the three legs (state-cycle / waves / governing-dynamics) and the
Fokker–Planck (Friston/SMNI) framing — and WAND's DWI lets us build them per
subject.

## The papers
- **Atasoy, Donnelly, Pearson (2016)** "Human brain networks function in
  connectome-specific harmonic waves." *Nat Commun* 7:10340. — **structural**
  connectome harmonics: eigenvectors of the Laplacian of the connectome graph
  (local gray-matter mesh + long-range white-matter tracts from DWI), the brain's
  standing-wave modes ordered by spatial frequency.
- **Atasoy, Glomb, Deco, Hagmann, Pearson, Kringelbach (2021)** "Functional
  harmonics reveal multi-dimensional basis functions underlying cortical
  organization." *Cell Reports* 36(8):109554 (DOI 10.1016/j.celrep.2021.109554) —
  the functional-connectivity sibling (harmonics of the dense FC graph).
- Debate: geometric (surface) vs connectome eigenmodes — **Pang et al. (2023)**
  *Nature* 618:566 "Geometric constraints on human brain function." WAND's DWI lets
  us do the *structural-connectome* version properly and compare.

## The key insight: harmonics = eigenbasis of the dissipative FP operator
The connectome Laplacian **L** is the structural **diffusion operator** = the
dissipative **Γ** of the Fokker–Planck flow `f = (Γ + Q)∇log p` (see
`docs/LANGEVIN_FOKKER_PLANCK_NEUROIMAGING.md`). In the harmonic basis:
- the **gradient / dissipative** flow is **diagonal** — each harmonic relaxes at
  its eigenvalue rate (high spatial frequencies decay fastest);
- the **solenoidal / rotational** flow (our stochastic cycle) is the **off-diagonal
  rotation among harmonics**.

So connectome harmonics are the **natural coordinates that separate gradient from
solenoidal flow** — an anatomically-grounded latent where the Langevin/Helmholtz
decomposition and entropy-production estimate become interpretable (each axis is a
known structural mode; the cycle is rotation among specific harmonics).

## Mapping to the legs
- **Waves (B):** a travelling/rotating wave = a pair of connectome harmonics in
  quadrature (sin/cos, phase-offset) = a DMD rotational mode pair. Harmonics give
  the principled modal basis for the mesh phase-gradient/curl operators.
- **Governing dynamics (C):** project activity onto harmonics → harmonic-coefficient
  time series = principled, interpretable latent for SINDy/DYSCO/Langevin.
- **Cycle (A) + Langevin:** the TINDA cycle / NESS solenoidal current becomes
  circulation among low-order harmonics; entropy production computed in the basis
  where the dissipative part is diagonal.

## WAND plan (multimodal: DWI structure + MEG dynamics, 169 subjects)
WAND derivatives already have `fsl-bedpostx`, `fsl-dtifit`, `eddy_qc`, `fsl-dwi`
(tractography-ready). Per subject:
1. **Build the structural connectome** from tractography (+ local mesh adjacency)
   → graph Laplacian → **connectome harmonics** (eigendecomposition).
2. **Project the resting MEG** (source-localized; Desikan-68 now, vertex-level
   ideally) onto that subject's harmonics → harmonic-coefficient time series.
3. **Run the dynamics framework in the harmonic basis:** harmonic power spectrum
   (state signature), SINDy/DYSCO/Langevin on the coefficients, and the
   INSIDEOUT / Stochastic-Force-Inference solenoidal-current + entropy-production
   estimate.
4. **Central test:** is the resting cycle a **solenoidal rotation among a small set
   of low-order connectome harmonics**? (deterministic drift rotation ≈ 0, but a
   nonzero probability current circulating in harmonic space).

## NeuroJAX implementation
- `io/connectome.py` (loader exists) + `geometry/` (cotangent Laplacian, mesh ops)
  → a `connectome_harmonics` routine (graph Laplacian eigendecomposition;
  differentiable in JAX via `jnp.linalg.eigh`).
- Combine with the planned `jaxctrl._langevin` (Kramers–Moyal / SFI + Helmholtz
  decomposition + entropy production) applied to the harmonic-coefficient latent.
- Novelty: connectome harmonics + non-equilibrium/solenoidal analysis + MEG
  dynamics on a 169-subject multimodal cohort — not previously combined.

## No EEG bands — the harmonic spectrum is the frequency axis (2026-06-27)
Atasoy's framework uses **no conventional EEG bands**: it decomposes activity onto
the connectome harmonics, ordered by *spatial* frequency (eigenvalue), and reports
harmonic **power** and **energy** (frequency-weighted, the harmonic's intrinsic
oscillation from the wave-equation dispersion). LSD raises high-frequency-harmonic
power; loss of consciousness collapses to low-frequency harmonics. So we **drop the
delta/theta/…/gamma bins** and run the WAND dynamics **broadband in the harmonic
basis** — the eigenvalue spectrum is the structure-derived frequency axis.

**HADES (Harmonic Decomposition of Spacetime; Vohryzek, Atasoy, Kringelbach, Deco
et al. 2023–24, on DMT)** is the temporal-dynamics-in-harmonic-basis framework:
harmonic modes in space expressed over time (fractional occupancy, lifetime, latent
space, a "spacetime hierarchy") — the harmonic-basis analogue of our Leg-A
occupancy/TINDA. Two openings it leaves that we fill: (i) **HADES is fMRI-only and
explicitly flags MEG/EEG as the next step** — the WAND harmonic leg *is* that MEG
extension; (ii) **HADES has no non-equilibrium analysis** — our solenoidal
circulation-among-harmonics / EPR (α* = A_sol·Σ in the harmonic basis) is the piece
it lacks. HADES uses *functional* harmonics (FC-graph eigenmodes; Atasoy 2021); we
use *structural/geometric* — worth computing the functional version too (from the
MEG's own FC, no DWI) for a direct, cheap comparison.

## First result — geometric eigenmodes, broadband (`wand_connectome_harmonics.py`)
Desikan-68 region graph weighted by centroid proximity (Gaussian σ=76 mm) →
connectome harmonics → project broadband source MEG onto the 20 low-order modes →
broadband analytic envelope → linear Langevin. The eigenvalue spectrum (λ ≈ 15→37)
is the frequency axis; resting power concentrates in low-order harmonics (H1,H2,H7).
**Broadband EPR = 0.098, vs a reversible IAAFT null 0.067±0.007 (z=4.42, p=0.024)**
→ genuine harmonic-basis broken detailed balance. The irreversible-circulation
matrix α* = A_sol·Σ **rotates among low-order harmonics** (top pairs H2↔H7, H1↔H12,
H1↔H8; centre-of-mass ≈ harmonic 9 of 20) at **f_sol = 0.099 Hz** — the same
infraslow ~10 s cycle the band analysis found at 0.04–0.12 Hz. The central test
passes: the resting cycle is a solenoidal rotation among a small set of low-order
connectome harmonics.

## Structural connectome (proper Atasoy basis) — robustness confirmed
`wand_structural_connectome.py` built sub-08033's Desikan-68 SC from TRACULA's
BBR-registered `aparc+aseg` (diffusion space) + bedpostX → `probtrackx2_gpu
--network` (261 s on the GB10), ROIs in exact `parcels68` order. Re-running the same
broadband harmonic analysis on the structural basis (`WAND_GRAPH=structural`,
log1p streamline counts):

| | geometric | structural |
|---|---|---|
| broadband EPR | 0.098 | 0.101 |
| reversible IAAFT null | 0.067±0.007 | 0.068±0.007 |
| z | 4.42 | 4.65 |
| f_sol | 0.099 Hz | 0.101 Hz |
| circulation centre-of-mass | 9.3 / 20 | 10.6 / 20 |

**The non-equilibrium signature is basis-invariant**: both bases give a significant
(z≈4.5), infraslow (f_sol≈0.10 Hz) solenoidal cycle that is rotation among low-order
connectome harmonics. The *specific* harmonics differ (geometric H1/H2/H7 carry
power & circulation; structural H13/H8/H15 — different graphs, different eigenvectors)
but the aggregate result is robust, and the structural (proper Atasoy) basis sharpens
it slightly. So the resting cycle = a low-dimensional solenoidal current among
low-order connectome harmonics, independent of whether the harmonics come from
cortical geometry or white-matter structure.

Next: the **functional-harmonic** version (HADES's own basis, from the MEG's FC — no
DWI), for the three-way comparison; map the dominant structural harmonics to anatomy;
non-standardised parcels; scale to more subjects as their DWI is reconstructed.
