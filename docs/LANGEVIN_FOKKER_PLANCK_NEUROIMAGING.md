# Langevin / Fokker–Planck in neuroimaging — and our stochastic MEG ring

Compiled 2026-06-26. Research dive connecting the drift-based dynamics methods
(DMD, SINDy, CEBRA, DYSCO) to the stochastic-dynamics literature (Friston's
Fokker–Planck framing; non-equilibrium / entropy-production work), to interpret
the WAND finding (`docs/WAND_THREE_LEG_CROSSTEST.md`): a resting-MEG network cycle
that is a **ring** (CEBRA), **directionally significant** (TINDA), yet shows **~0
deterministic rotation** (DMD/SINDy/DYSCO).

## Bottom line
That pattern is the signature of a **non-equilibrium steady-state (NESS)
solenoidal cycle**: a net circulation of *probability* around a loop in state
space (a divergence-free probability current **J ≠ 0**), **not** a deterministic
limit cycle. DMD/SINDy/DYSCO recover only the conditional-mean **drift** ż=f(z)
and discard the noise, so a cycle carried by the **diffusion-coupled solenoidal**
flow is invisible to them *by construction*. CEBRA gives the manifold geometry
(ring); TINDA gives the flux asymmetry; the drift methods give ~0 mean rotation —
all three are consistent with one stochastic NESS cycle.

## 1. Friston's gradient + solenoidal Fokker–Planck decomposition
For an Itô–Langevin SDE `dx = f(x)dt + ς dω` with diffusion `Γ = ½ςςᵀ` and
stationary density `p`, the drift splits (Helmholtz / Ao–Kwon):

> **f(x) = (Γ + Q)·∇log p(x) = −(Γ + Q)·∇ℑ(x)**,  ℑ = −log p (surprisal)

- **Γ** symmetric PSD → **dissipative / gradient / curl-free** flow that descends
  surprisal (point-attractor-like), balancing noise to keep p stationary.
- **Q** antisymmetric → **solenoidal / divergence-free / rotational** flow.

Because Q is antisymmetric, `Q∇log p` is **orthogonal to ∇log p** — it transports
probability **around** the iso-density contours, contributing **zero** to ∂p/∂t
yet sustaining a persistent circulating current. **Q = 0 ⇔ detailed balance,
time-reversible, zero current (equilibrium); Q ≠ 0 ⇔ broken detailed balance,
positive entropy production, genuine NESS.** A system can have **zero deterministic
rotation in its mean** yet a strong solenoidal current — *that is our ring.*

Key refs: Friston & Ao 2012 (*Comput Math Methods Med*, DOI 10.1155/2012/937860);
Friston 2019 "A free energy principle for a particular physics" (arXiv:1906.10184);
**Da Costa, Friston, Heins, Pavliotis 2021** "Bayesian mechanics for stationary
processes" (*Proc R Soc A* 477:20210518; JAX code github.com/conorheins/bayesian-mechanics-sdes);
Friston et al. 2023 "FEP made simpler" (*Phys Reports* 1024); "Path integrals,
particular kinds, strange things" (*Phys Life Rev* 47); Da Costa & Pavliotis 2023
"Entropy production of stationary diffusions" (arXiv:2212.05125).

## 2. Langevin/FP already in EMEG & fMRI
- **Stochastic DCM / generalised filtering (Friston lab):** state-noise SDEs in
  generalised coordinates inverted from imaging — Li, Daunizeau, Stephan, Penny,
  Friston 2011 "Generalised filtering and stochastic DCM for fMRI" (*NeuroImage*
  58:442); DEM (Friston, Trujillo-Barreto, Daunizeau 2008); Generalised Filtering
  (Friston et al. 2010); csd-DCM for resting/MEG (Friston et al. 2012, 2014). SPM12
  (`spm_DEM`, `spm_LAP`, `spm_dcm_fmri_csd`). Caveat: DEM uses smooth analytic noise.
- **Noisy neural mass/field models (generative backbone):** Wilson–Cowan,
  Jansen–Rit (stochastic: Ableidinger et al. 2017); FP mean-field of spiking nets
  (Brunel 2000); whole-brain noisy-Hopf / dynamic mean field (Deco, Jirsa,
  Robinson, Breakspear, Friston 2008; Deco et al. 2013, 2017); stochastic neural
  fields (Bressloff). Code: **The Virtual Brain**; **vbjax** (JAX whole-brain SDEs
  — already in NeuroJAX's stack); Diffrax for differentiable SDEs.

## 3. Mapping our methods onto the FP picture
The Koopman *generator* of an SDE = backward Kolmogorov operator = drift term
(b·∇) + diffusion term (½tr(ςςᵀ∇²)). Drift-based methods miss the second term.

| Method | gradient drift | solenoidal drift | diffusion D(z) | gives |
|---|---|---|---|---|
| **DMD/Koopman** | linear | linear (cc eigenpairs) | no | linear drift operator + modes |
| **SINDy** | yes (nonlinear) | yes (incl. limit cycles) | no | deterministic ż=f(z) |
| **DYSCO** | linear/instantaneous | partial | no | latent + drift law |
| **CEBRA** | — | — | — | **manifold geometry only** (not a dynamics model) |
| **Stochastic SINDy / Langevin regression** | yes | yes | **yes** | full SDE (drift+D) |
| **Koopman generator / gEDMD** | yes | yes | **yes** | drift+diffusion generator |

So for our ring: CEBRA = "it's a ring", TINDA = "transitions are asymmetric",
DMD/SINDy/DYSCO = "mean rotation ≈ 0" → the cycle is the **diffusion-coupled
solenoidal probability current**, which none of the drift methods represent.
Stochastic-SINDy refs: Boninsegna, Nüske, Clementi 2018; **Callaham, Loiseau,
Rigas, Brunton 2021** "Langevin regression" (*Proc R Soc A* 477:20210092;
github.com/dynamicslab/langevin-regression); Klus et al. 2020 (gEDMD).

## 4. Non-equilibrium / broken-detailed-balance work = direct measurement of our cycle
These operationalize "is there a solenoidal current?" as "is the series
time-irreversible?":
- **Lynn, Cornblath, Papadopoulos, Bertolero, Bassett 2021** "Broken detailed
  balance and entropy production in the human brain" (*PNAS* 118:e2109889118; code
  github.com/ChrisWLynn/Broken_detailed_balance) — discrete-state pairwise flux
  F_ij ∝ (P_ij − P_ji), flux maps, entropy-production rate.
- **Deco, Sanz Perl, … Kringelbach 2022** INSIDEOUT (*Commun Biol* 5:572; code
  github.com/decolab/insideout) — irreversibility = forward vs time-reversed
  lagged-correlation distance.
- **Tewarie, Hindriks, Lai, Sotiropoulos, Kringelbach, Deco 2023** "Non-reversibility
  outperforms FC … in MEG" (*NeuroImage* 276:120186) — **the source-space MEG
  template** (lagged amplitude-envelope asymmetry; strongest in gamma).
- **Nartallo-Kaluarachchi, Kringelbach, Deco, Lambiotte, Goriely 2026**
  "Nonequilibrium physics of brain dynamics" (*Phys Reports* 1152; arXiv:2504.12188)
  — the definitive review. Foundations: Seifert 2012; Schnakenberg 1976.

**Caveats:** (1) a stationary **Gaussian-linear** process is time-reversible
(Weiss 1975) → irreversibility needs non-Gaussianity *or* **asymmetric lagged
cross-covariance** = exactly what TINDA detects. (2) Coarse-graining/partial
observation biases entropy production **down** (lower bounds). (3) **MEG > BOLD
for this:** the diffusion is fast (~ms) in-band noise; dominant irreversibility is
at 20–40+ Hz, above the BOLD passband. (4) Validate symbolic/permutation estimators
against phase-randomized + AAFT surrogates.

## 5. Concrete test: is our ring a NESS solenoidal cycle?
Run on the CEBRA/HMM latent z(t) (or a low-D source-MEG embedding) at native rate.

**Track 1 — model-free irreversibility (fast, matches MEG lit):**
1. INSIDEOUT lagged-cross-covariance asymmetry on z(t) (Tewarie's lagged
   amplitude-envelope variant for source MEG) → surrogate-significant asymmetry ⇒
   broken detailed balance ⇒ Q ≠ 0.
2. Coarse-grain to the HMM states, apply Lynn's pipeline: pairwise flux F_ij,
   **flux map** — *does it circulate around the ring in the TINDA direction?* +
   entropy-production rate with bootstrap CIs.
3. Time-reversal surrogates (phase-randomized, AAFT) to exclude Gaussian-linear /
   symbolization artifacts.

**Track 2 — explicit drift+diffusion + solenoidal extraction:**
4. Estimate μ(z), D(z) by Kramers–Moyal conditional moments (`kramersmoyal`),
   **Stochastic Force Inference** (Frishman & Ronceray 2020, *PRX* 10:021009;
   github.com/ronceray/StochasticForceInference), or Langevin regression.
5. Form the current **J(z) = μ(z)p(z) − ∇·(D(z)p(z))**, decompose μ into the
   gradient part (D∇log p_ss) and the **divergence-free solenoidal remainder**;
   report entropy production **Ṡ = ⟨νᵀD⁻¹ν⟩** (ν = J/p_ss). **Distinguishing
   prediction vs a limit cycle: deterministic (drift) rotation ≈ 0 while the
   solenoidal current J ≠ 0 and loops around the ring.**

## 6. The NeuroJAX/jaxctrl niche
**No JAX-native Kramers–Moyal / Stochastic-Force-Inference / entropy-production
package exists** (mid-2026; references are numpy/scipy, Lynn's is MATLAB). The
estimators (kernel-weighted conditional moments, least-squares drift/diffusion,
loop integrals of J, Ṡ = ⟨νᵀD⁻¹ν⟩) **vectorize cleanly in `jax.numpy` +
`vmap`/`scan`, and `jax.grad` flows through the whole drift→diffusion→entropy-
production estimate** — differentiable, GPU-native, an open niche. This is the
data-driven member of both **Ingber's SMNI** and **Friston's FP**, and the correct
Leg-C tool to capture *this* cycle. Template: bayesian-mechanics-sdes (JAX);
forward SDE via Diffrax. Proposed home: `jaxctrl._langevin` (drift+diffusion +
Helmholtz/solenoidal decomposition + entropy production), re-exported by
`neurojax.dynamics`, applied to the WAND/HD-EMEG latent.

*(Citations cross-checked from search; full-text fetch was blocked, so a few
page/article-number fields should be verified before manuscript use.)*
