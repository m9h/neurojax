# Three legs of neural dynamics: state-cycles, travelling-waves, and governing equations

Compiled 2026-06-25. Extends `docs/CYCLES_VS_TRAVELLING_WAVES.md` with a third
leg — data-driven latent dynamics / governing equations (SINDy, CEBRA, DYSCO) —
and the thesis that all three describe **one low-dimensional cyclic attractor**.

---

## The three legs

| | **A. Generative state-space** | **B. Field / wave dynamics** | **C. Governing dynamics** |
|---|---|---|---|
| Method | HMM / DyNeMo / M-DyNeMo + **TINDA** | Generalized Phase, mesh phase-gradient, singularities | SINDy / Koopman + **CEBRA → DynCL → DYSCO** |
| Object | discrete states / mode mixtures; their **cyclic ordering** | spatial phase gradients; rotating/planar waves | latent manifold + flow ż = f(z) |
| Output | state time-courses, transition matrix, cycle strength S | wave direction/speed, curl/singularities, PGD | latent trajectory + (symbolic) governing equation |
| NeuroJAX | ✅ `models/` HMM,DyNeMo,M-DyNeMo; `wh_*` TINDA | ✅ `analysis/waves.py` (grid+mesh) | ⚙️ `dynamics/` SINDy,Koopman,windowed; CEBRA/DYSCO TODO |

**Unifying thesis.** TINDA's "structured cycle" (12 HMM states on a ring,
300–1000 ms, S=0.065/z=56.6 on our WH data) is the *discrete shadow* of a
**continuous closed orbit (limit cycle / ring attractor) on a low-dimensional
neural manifold**. The three legs are three views of that one object:
- **A (HMM+TINDA)** discretizes the orbit into states and recovers its *ordering*.
- **B (waves/jPCA)** recovers its *rotational geometry* — a travelling wave is a
  limit-cycle trajectory in phase space; "travelling waves explain rotational
  dynamics" (Sci. Rep. 2024) via skew-symmetric flow.
- **C (CEBRA/DYSCO/SINDy)** recovers the *same cycle as a continuous latent
  manifold with an explicit governing equation* ż = f(z), with the cycle phase as
  a state variable and cycle speed as its angular frequency.

---

## Leg C deep dive — SINDy → CEBRA → DynCL → DYSCO

The key finding from the literature is a clean lineage that turns contrastive
representation learning into **system identification**, culminating in DYSCO:

1. **SINDy** — Brunton, Proctor & Kutz, *PNAS* 2016. Sparse regression on a
   candidate library: ẋ = Θ(x)Ξ → parsimonious symbolic ODEs. Needs *clean state
   measurements*. Variants for the noisy/low-data neural regime: weak/integral
   WSINDy, SINDy-PI (rational, noise-robust), **E-SINDy** (ensemble, uncertainty),
   SINDyG (graph/oscillator networks). `dynamicslab/pysindy` (Python); a
   **JAX-native SINDy** exists in NGC-Learn.
2. **SINDy-autoencoder** — Champion et al., *PNAS* 2019. First to learn
   *coordinates + equations jointly* (autoencoder + SINDy loss). The conceptual
   parent of DYSCO.
3. **CEBRA** — Schneider, Lee & **Mathis**, *Nature* 2023. Contrastive (InfoNCE)
   latent embeddings of neural population activity, jointly shaped by behaviour or
   time; formally non-linear ICA with identifiability. Recovers topological
   manifolds (e.g. ring/torus for spatial coding). `AdaptiveMotorControlLab/CEBRA`
   (PyTorch; **no JAX port**).
4. **DynCL** — González Laiz, Schmidt & Schneider, **ICLR 2025** (arXiv:2410.14673).
   *Proves contrastive SSL performs non-linear system identification in latent
   space* — the bridge from "CEBRA finds the manifold" to "contrastive learning
   finds the flow on the manifold." `dynamical-inference/dyncl`.
5. **DYSCO** — Muratore & **Mathis**, **arXiv:2606.13260** (June 2026; this is the
   paper at your link). Multi-view temporal contrastive learning + a **JEPA
   encoder/predictor** (the predictor *is* the latent dynamics operator) +
   a **structured functional basis** (SINDy-like) → recovers latent trajectories
   AND the **governing flow field**, with **symbolic** recovery up to an affine
   gauge. Identifiability extended to **noisy, nonlinear, Poisson** observations;
   validated on chaotic / oscillatory / **metastable** regimes; Lorenz symbolic
   recovery. *No public code yet* — watch `AdaptiveMotorControlLab` /
   `dynamical-inference`.

DYSCO = CEBRA's contrastive identifiability + SINDy's symbolic governing
equations, robust to the exact noise (Poisson spike counts, nonlinear mixing)
that defeats vanilla SINDy on neural data.

---

## Why this is a green field for NeuroJAX

- **CEBRA** is applied almost entirely to spikes/calcium — **no flagship
  CEBRA-on-MEG/EEG-source paper**. SINDy on whole-brain is mostly via
  neural-mass/Kuramoto/Stuart-Landau models, not source time-series. DYSCO is
  brand-new with no neuro-imaging application. So applying Leg C to **WAND /
  Wakeman-Henson MEG source dynamics** — and checking whether the recovered limit
  cycle matches the HMM+TINDA structured cycle — is genuinely novel.
- **NeuroJAX already has the half that's hardest to find in JAX:** `dynamics/`
  ships JAX SINDy (`SINDyOptimizer`, `polynomial_library`, `fourier_library`),
  Koopman/DMD, and **windowed** dynamics whose change-points are *explicitly meant
  to be compared against HMM/DyNeMo states* (per the package docstring). That is
  Leg C ↔ Leg A already wired conceptually.
- The Kidger stack covers the rest: **Equinox** for a CEBRA-style InfoNCE encoder
  and a DYSCO JEPA encoder/predictor; **Diffrax** to integrate the learned latent
  flow ż = f(z); `jax.grad` to keep it all differentiable end-to-end on the GB10.

---

## Architecture (where the code lives)

The *methods* are general differentiable system identification, so they live in
**jaxctrl** (next to SINDy/Koopman in `_sysid`); **neurojax re-exports and applies**
them — mirroring how `neurojax.dynamics.sindy` re-exports `jaxctrl.SINDyOptimizer`.
- `jaxctrl/_contrastive.py` — `CEBRA`, `ContrastiveEncoder`, `info_nce` (DONE; 4 tests).
- `jaxctrl/_dysco.py` — `DYSCO`, `LatentFlow` (DONE; 3 tests). Contrastive encoder
  + SINDy-parameterized latent flow (reuses `_sysid.polynomial_library`) trained
  JEPA-style → latent trajectory + governing equation ż = Θ(z)·Ξ. Validated: from
  a nonlinearly-observed 2-D rotation it recovers a latent flow with imaginary
  eigenvalues (~±iω) — i.e. it identifies the **limit cycle**, up to the affine gauge.
- `neurojax.dynamics.{CEBRA,DYSCO}` — thin re-exports (`dynamics/cebra.py`,
  `dynamics/dysco.py`); jaxctrl made optional so neurojax still imports in a lean
  GPU env. `from neurojax.dynamics import CEBRA, DYSCO, SINDyOptimizer, KoopmanEstimator`.

## Concrete plan

1. **JAX CEBRA encoder** — DONE in `jaxctrl/_contrastive.py`: a time-contrastive
   InfoNCE encoder (Equinox); L2-normalized 2-D output puts a cyclic attractor on
   a ring. Validated on synthetic ring data — temporal-neighbourhood preservation
   is clean; *pure time-contrastive* gives only partial *global* ring recovery
   (behaviour-conditioned CEBRA or DYSCO would tighten it). NEXT: embed WH/WAND MEG
   source-parcel time-series → does the structured cycle appear as a ring/loop?
2. **JAX SINDy/DYSCO on the latent** — fit ż = f(z) on the CEBRA latent (SINDy,
   already in `dynamics/`), then a DYSCO-style JEPA predictor (Equinox+Diffrax) for
   the governing flow with affine-gauge identifiability. Is the attractor a
   **limit cycle**? Does its angular frequency match the TINDA cycle rate?
3. **Three-way cross-test** (the payoff) on the same MEG source data:
   - Leg A: HMM+TINDA cycle order + strength S (done: S=0.065, z=56.6).
   - Leg B: mesh wave direction / rotational singularities (done: organized > null).
   - Leg C: CEBRA/DYSCO latent limit-cycle phase + governing ż = f(z).
   Test whether the TINDA cycle order, the wave rotation, and the latent-orbit
   phase are **the same cyclic coordinate** — i.e. one rotating attractor seen
   three ways. Nulls throughout (block-shuffle for TINDA; spatial-shuffle for
   waves; surrogate dynamics for SINDy/DYSCO).

---

## Sources
- DYSCO — Muratore & Mathis, arXiv:2606.13260 (2026): https://arxiv.org/abs/2606.13260
- CEBRA — Schneider, Lee & Mathis, Nature 617:360 (2023): https://www.nature.com/articles/s41586-023-06031-6 · code https://github.com/AdaptiveMotorControlLab/CEBRA
- DynCL — González Laiz et al., ICLR 2025, arXiv:2410.14673: https://arxiv.org/abs/2410.14673 · code https://github.com/dynamical-inference/dyncl
- xCEBRA — arXiv:2502.12977 (AISTATS 2025)
- SINDy — Brunton, Proctor & Kutz, PNAS 113:3932 (2016): https://www.pnas.org/doi/10.1073/pnas.1517384113 · pysindy https://github.com/dynamicslab/pysindy · JAX SINDy (NGC-Learn) https://ngc-learn.readthedocs.io/en/latest/museum/sindy.html
- SINDy-autoencoder — Champion et al., PNAS 116:22445 (2019), arXiv:1904.02107 · E-SINDy arXiv:2111.10992 · SINDyG arXiv:2409.04463
- Travelling waves ↔ rotational dynamics — Sci. Rep. 2024: https://www.nature.com/articles/s41598-024-53907-2
- Structured cycles / TINDA — van Es et al., Nat. Neurosci. 28:2118 (2025): https://www.nature.com/articles/s41593-025-02052-8
- Mathis lab preprints (newest-than-DYSCO check): http://www.mackenziemathislab.org/preprints
