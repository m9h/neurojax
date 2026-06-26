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
