# Structured network cycles (Woolrich) vs travelling waves (Miller) — comparison & a NeuroJAX bridge

Compiled 2026-06-24. Pairs the osl-dynamics "structured cycles" result with Earl
Miller / Lyle Muller's travelling-wave program, and lays out concrete techniques
+ code to compare them on the same MEG source data we already process.

---

## 1. The two phenomena

### A. Structured cycles — van Es, Higgins, Gohil, Quinn, Vidaurre, Woolrich (2025)
*"Large-scale cortical functional networks are organized in structured cycles"*,
**Nature Neuroscience** 28(10):2118–2128. DOI 10.1038/s41593-025-02052-8;
open access PMC12497652; bioRxiv 2023.07.25.550338.

- **Claim:** individual state→state transitions are stochastic, but the
  *aggregate ordering* of whole-brain network activations is a robust **cycle**;
  one full traversal takes **300–1000 ms**. States with similar function/spectra
  cluster at the same cycle phase; the cycle splits into **4 metastates**, with
  **DMN and DAN at opposite phases** (α/β sensorimotor on one side, δ/θ
  frontotemporal on the other).
- **Substrate:** **TDE-HMM**, K=12 spectrally-defined whole-brain states on
  source-parcellated MEG (5 datasets: MEG-UK, Cam-CAN, HCP-MEG, Replay,
  Wakeman-Henson). States = power + coherence networks.
- **Novel method — TINDA** (Temporal Interval Network Density Analysis): for each
  reference state, take inter-visit intervals, split each in half, and measure
  every other state's fractional-occupancy **asymmetry** (first vs second half)
  → a directed graph → states laid on a unit circle, optimized for **cycle
  strength S** (+1 = perfectly clockwise, 0 = stochastic). S and cycle rate are
  heritable and track cognition/awareness.
- **Granularity:** *temporal* ordering of **discrete, whole-brain** states. NOT
  framed as a spatial wavefront. Each "state" is a global topography.
- **Code:** `OHBA-analysis/Tinda` (MATLAB) and `osl_dynamics.analysis.tinda`
  (Python) — ships in osl-dynamics.

### B. Travelling waves — Davis, Muller, Sejnowski, Reynolds, Miller (2020+)
Key papers: Muller et al. *Nat Rev Neurosci* 2018 (review); Davis*/Muller* et al.
*Nature* 2020 (spontaneous waves gate perception, introduces Generalized Phase);
Bhattacharya, Brincat, Lundqvist, Miller *PLoS Comput Biol* 2022 (rotating waves
in PFC during working memory); Muller et al. *Science* 2026 (brain-wide rotating
waves as a spatiotemporal clock); human EEG/iEEG follow-ups 2025–2026.

- **Claim:** neural oscillations have smooth **spatial phase gradients** — phase
  advances across the cortical sheet, so activity literally *travels* (planar,
  radial, or **rotating** around a phase singularity). Spontaneous, present in
  awake behavior; the wave phase before a stimulus gates perception; PFC waves
  (mostly rotational) are modulated by working memory.
- **Substrate:** invasive **LFP/Utah-array** (96 ch, 4×4 mm, ~400 µm spacing) in
  monkey/marmoset — a regular 2-D grid in real cortical coordinates. Speeds
  ~0.1–0.6 m/s; wideband 5–40 Hz (via Generalized Phase, not narrowband).
- **Granularity:** *spatially-continuous phase propagation*, typically within one
  area, at fine (sub-mm) scale.
- **Code:** `mullerlab/wave-matlab`, `mullerlab/generalized-phase` (MATLAB).

---

## 2. The bridge — are these the same thing at different granularity?

| | Woolrich cycles (TINDA) | Miller waves |
|---|---|---|
| Object | discrete whole-brain network states | continuous phase field on cortex |
| Structure | temporal cycle through a state repertoire | spatial gradient that propagates |
| Scale | macroscopic, whole-brain, ~100 ms dwell | mesoscopic, single-area, ms |
| Modality | source-space MEG | Utah-array LFP (also iEEG/EEG/wide-field) |
| Direction read-out | cycle strength S, FO asymmetry | phase-gradient direction, curl |
| Code | osl-dynamics TINDA (Py/MATLAB) | mullerlab wave-matlab (MATLAB) |

The unifying debate is **"waves vs sequentially-activated discrete modules"**:
Orsher, Shein-Idelson et al., *eLife* 12:e92254 (2024) show that sequential
activation of adjacent discrete modules **appears as a smooth travelling wave**
under coarse spatiotemporal sampling. So the structured cycle (discrete state
ordering, MEG) and a rotating wave (continuous phase, LFP) may be **two readouts
of one spatiotemporal flow** at different granularities. Roberts et al.
"Metastable brain waves" (*Nat Commun* 2019) already showed spiral/source/sink
wave patterns *visited in sequence* — a wave-native analogue of cycling states.

**The open question a NeuroJAX analysis can pose:** does TINDA's directional flow
(DMN → … → DAN → … → DMN) correspond to a **macroscopic spatial propagation
direction** across the cortical mesh? Neither paper answers it.

---

## 3. Miller/Muller techniques — deep dive (the reusable methods)

Priority methods, all vectorized phase arithmetic → differentiable, GPU-friendly,
straightforward to port to JAX:

1. **Generalized Phase (GP)** — `mullerlab/generalized-phase`. Robust analytic
   phase of *wideband* (5–40 Hz) signals (avoids narrowband waveform distortion).
   Pipeline: loose broadband filter → Hilbert/analytic signal → take phase of the
   dominant fluctuation; **correct two failure modes**: (i) low-frequency
   intrusions offsetting the complex-plane origin (re-center); (ii) high-frequency
   intrusions causing spurious **negative instantaneous frequency** — detect
   sign-reversals of dφ/dt and **interpolate phase across those epochs**. The
   negative-frequency correction is the line-level-fidelity spot for a port.
   Python map: `scipy.signal.hilbert` / `mne` analytic signal + a neg-frequency
   detector/interpolator.
2. **Phase gradient via complex multiplication** —
   `wave-matlab/analysis/phase_gradient_complex_multiplication.m`. For the 2-D
   analytic field V, ∂x phase = `arg(V[x,y]·conj(V[x+1,y]))` (and y likewise);
   wrap-free. Returns gradient magnitude/direction. The **wave source** is the
   point maximizing the **divergence** of this field.
3. **Circular-linear regression (planar waves)** — fit φ(x,y) = k·(x,y)+φ0; the
   gradient **k** gives direction (angle) and speed (frequency/|k|); circular-
   linear ρ vs a shuffle threshold = detection. `phase_correlation_distance.m`.
4. **Phase singularity / curl (rotational waves)** — rotational waves spin around
   a singularity; detect via **curl of the phase-gradient field** / closed-contour
   phase integral = ±2π (topological charge, sign = rotation direction).
   `phase_correlation_rotation.m`.
5. **Quality metrics** — **PGD** (phase-gradient directionality = |mean gradient|
   / mean|gradient|, →1 = coherent wave), speed distribution, divergence (source/
   sink) and curl (rotation) maps; significance via spatial/temporal shuffles.

Repos (MATLAB; deps CircStat + export_fig): `mullerlab/wave-matlab`,
`mullerlab/generalized-phase`, `mullerlab/davis2021ncomms`,
`mullerlab/benignoEAwavecomp`. **No official Python/JAX port exists** — that gap
is the deliverable.

**MEG caveat (must be the null):** apparent MEG "waves" can be artifacts of (a)
two discrete dipoles beating (PMC7615062) or (b) source leakage, or (c) the
discrete-module-sampling effect (eLife 92254). Any wave claim on MEG must reject
these nulls. The model-based forward-projection approach (PMC12037073) is the
rigorous alternative to inverse-localizing waves.

---

## 4. Proposed NeuroJAX work — compare cycles and waves on the same MEG source data

We already produce source-parcellated MEG and fit HMM/DyNeMo/M-DyNeMo (see
`docs/WH_REAL_DATA_VALIDATION.md`). Concrete plan:

1. **TINDA in-graph** — apply `osl_dynamics.analysis.tinda` (or a JAX reimpl) to
   our JAX-HMM state time courses → cycle strength S, the directed cycle, the 4
   metastates. (We already match osl-dynamics networks at r≈0.86–0.90 on real WH
   MEG, so the states feeding TINDA are validated.)
2. **Port the wave operators to JAX** — Generalized Phase (+ neg-frequency
   correction), phase gradient, divergence, **curl** on the **cortical mesh**.
   NeuroJAX's `geometry/` discrete-differential-geometry already provides mesh
   grad/div/curl operators — apply them to the source-space analytic-signal field
   instead of a regular grid. All differentiable → runs on the GB10 GPU.
3. **Per-frame wave classification** — PGD + circular-linear ρ + curl on the
   source map → label each MEG time frame planar / radial / rotational / none.
4. **Cross-test (the novel bit)** — do HMM/DyNeMo state-transition times align
   with wave-pattern switches? Does the TINDA cycle order match the rotation
   order of a persistent phase singularity? Compare the **discrete** propagation
   direction (sequenceness/TDLM on state time courses) with the **continuous**
   wave direction (phase gradient). Agreement ⇒ the structured cycle is the
   network-state shadow of a macroscopic travelling/rotating wave.
5. **Nulls** — two-dipole beating, source leakage, discrete-module sampling.

**Most reusable single deliverable:** a JAX `analysis/waves.py` implementing GP +
phase-gradient-complex-multiplication + curl-singularity + PGD on a cortical mesh,
validated against `mullerlab/wave-matlab` outputs, then run alongside TINDA on the
same source data. Line-level fidelity to check when porting: GP's negative-
frequency interpolation and the ρ-thresholds in `phase_correlation_*`.

## 5. Implemented so far (2026-06-25)
- **`src/neurojax/analysis/waves.py`** — JAX port of the Muller-lab wave methods:
  `generalized_phase` (wideband analytic phase + negative-instantaneous-frequency
  correction via running-max anchors), `phase_gradient` (wrap-free, complex
  multiplication), `phase_gradient_directionality` (PGD), `wave_direction`,
  `divergence`, `curl`, `singularity_location`. **11 TDD tests** (`tests/test_waves.py`)
  on synthetic planar / rotating / radial fields — recovers wave vector & direction,
  PGD≈1 for coherent vs <0.4 random, curl flags rotation, divergence flags radial
  sources, singularity located at the rotation centre.
- **TINDA reproduced** on our real WH sensor multi-run HMM states
  (`scripts/real_data/wh_tinda_cycle.py`): both the osl-dynamics oracle and the JAX
  HMM give **cycle strength S ≈ +0.006** (near-zero) — at this scale (1 subject,
  K=6) there is no robust cycle, consistent with the paper needing K=12 and
  hundreds of subjects (Cam-CAN n=612). Both implementations agreeing on S≈0 is
  itself a parity check.

- **Mesh operators** (DONE) — `mesh_phase_gradient` (linear-FEM gradient with
  wrap-free edge phase differences), `phase_singularity_charge` (per-face winding
  number = mesh curl/rotation detector), `mesh_phase_gradient_directionality`
  (area-weighted PGD). 4 more TDD tests (15 total) on a flat triangulated grid.
- **Run on real MEG** (`scripts/real_data/wh_mesh_waves{_prep,}.py`): WH sub-05
  source-localized to the fsaverage LH cortical mesh (10242 verts, 20480 faces),
  alpha band (8-12 Hz), 60 s; mesh wave operators applied per frame on the GB10
  GPU (6000 frames). Result: **PGD median 0.055** (no hemisphere-wide planar
  wave), but **~294 phase singularities/frame** (every frame). The high
  singularity count is the signature of a complex/turbulent phase field — at this
  stage likely dominated by phase noise (each noisy patch spawns +-1 winding
  pairs). Interpreting it requires (i) spatial phase smoothing / amplitude
  thresholding and (ii) the source-leakage + two-dipole nulls. So: the pipeline
  runs end-to-end on real MEG, but no travelling-wave *claim* — null-testing next.

**Next:** (a) amplitude-thresholded / spatially-smoothed singularity detection +
the two-dipole/leakage nulls before any wave claim; (b) a K=12 multi-subject HMM
to obtain a real TINDA cycle (S >> 0); then cross-test wave direction/rotation
against the TINDA cycle order on the same source mesh.

## Sources
- Cycles: https://www.nature.com/articles/s41593-025-02052-8 · https://pmc.ncbi.nlm.nih.gov/articles/PMC12497652/ · code https://github.com/OHBA-analysis/Tinda · https://osl-dynamics.readthedocs.io/ (`analysis/tinda`)
- Waves: Muller NRN 2018 https://www.nature.com/articles/nrn.2018.20 · Davis/Muller Nature 2020 https://www.nature.com/articles/s41586-020-2802-y · Bhattacharya/Miller PLoS CB 2022 https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009827 · Muller Science 2026 https://www.science.org/doi/10.1126/science.adx1369 · code https://github.com/mullerlab/wave-matlab · https://github.com/mullerlab/generalized-phase
- Bridge/nulls: eLife 92254 https://elifesciences.org/articles/92254 · Metastable brain waves https://www.nature.com/articles/s41467-019-08999-0 · MEG two-source artifact https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7615062/
