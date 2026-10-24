# OSL / Woolrich Group Updates — Research Note

**Compiled:** 2026-06-22
**Question:** Latest papers from Mark Woolrich (Oxford / OHBA Analysis Group) developing OSL; have there been updates for OSL?

---

## TL;DR

Yes — substantial updates. The big structural change is that the old monolithic **OSL**
has **split into two separately-published packages** (`osl-ephys` for
preprocessing/source-recon, `osl-dynamics` for dynamics modelling), and the Woolrich/OHBA
group put out several new methods papers in 2025.

---

## Software updates

### 1. OSL split → `osl-ephys` + `osl-dynamics`
The classic monolithic OSL (preprocessing, coregistration, source reconstruction) is now
**`osl-ephys`**, published as its own paper in 2025:

- van Es, Gohil, Quinn & Woolrich, *"osl-ephys: a Python toolbox for the analysis of
  electrophysiology data"*, **Frontiers in Neuroscience** 19:1522675 (2025).
  - Builds on MNE-Python.
  - Batch parallel processing, config API with log-keeping, HTML QC reports.
  - New **volumetric** coregistration / source-recon / parcellation pipeline that avoids
    surface-based processing.
  - Apache License, on PyPI + GitHub. Developed by the OHBA Methods Group, Oxford.
  - arXiv preprint: 2410.22051

### 2. `osl-dynamics` releases
Latest is the **3.2.x** line (HMM, DyNeMo, M-DyNeMo, spectral estimation). Recent changelog:
- Improved covariance-from-cross-spectra calculation; option to use the median in spectral
  estimation.
- mat7.3 support; Path-object inputs.
- New / corrected parcellations (AAL116, Schaefer, low-density AAL-24).
- **NOTE — version date is ambiguous:** PyPI metadata vs. the GitHub releases page disagree
  on whether the 3.2.x releases are dated 2024 or April 2026. Confirm directly against the
  container with `pip index versions osl-dynamics` if the exact version matters.

### 3. OSL Workshop 2025
Fresh teaching notebooks: `OHBA-analysis/osl-workshop-2025-dynamics` on GitHub.

---

## New methods papers from the group (2025)

- **Canonical HMM Networks** — Gohil, Huang, Higgins, van Es, Quinn, Vidaurre & Woolrich,
  *"Canonical Hidden Markov Model Networks for Studying M/EEG"*, **bioRxiv** (Oct 2025).
  A pre-trained / transferable HMM applied to an Alzheimer's MEG dataset, eyes-open/closed
  EEG, and **sensor-level MEG without source reconstruction**. Released as an open resource
  (`OHBA-analysis/Canonical-HMM-Networks`).

- **M-DyNeMo** — Huang, Gohil & Woolrich, *"Evidence for Transient, Uncoupled Power and
  Functional Connectivity Dynamics"*, **Human Brain Mapping** 46:e70179 (2025).
  New generative model separating power vs. functional-connectivity network dynamics.

- **Gaussian-Linear HMM** — Vidaurre et al., a Python package (**Imaging Neuroscience**, 2025)
  generalizing the HMM for task + rest; companion **Nature Protocols** paper,
  *"A comprehensive framework for statistical testing of brain dynamics"* (2025).

- **Structured cycles** — *"Large-scale cortical functional networks are organized in
  structured cycles"*, **Nature Neuroscience** (2025). Network state transitions form robust
  cyclical patterns at 300–1000 ms.

---

## Adjacent / related reference (added per request)

- Muratore & Mathis, *"Extracting Governing Equations from Latent Dynamics via Multi-View
  Contrastive Learning"* (DYSCO), **arXiv 2606.13260** (submitted 2026-06-11).
  - Multi-view temporal contrastive learning to jointly recover latent trajectories AND
    governing dynamics from noisy high-dimensional observations; symbolic equation recovery
    via structured functional bases; identifiability guarantees under nonlinear observations.
  - Demonstrated on chaotic / oscillatory / metastable dynamics with Gaussian and Poisson
    noise.
  - **Not from the Woolrich/OHBA group** and no direct OSL connection — but relevant to
    NeuroJAX's `analysis/SINDy` and neural-dynamics work (system identification, symbolic
    dynamics, Poisson-noise neural recordings).

---

## 2026 status & container update (added 2026-06-22)

### Latest versions (verified against PyPI / FSL conda)
| Package | Latest | In container before | After this update |
|---------|--------|---------------------|-------------------|
| osl-dynamics | **3.2.2** | 3.0.0 | 3.2.2 |
| osl-ephys | **2.4.0** | (absent) | 2.4.0 added |
| FSL | **6.0.7.22** | (absent) | 6.0.7.22 added |
| legacy `osl` | 1.1.0 (deprecated) | never pinned | n/a |

- FSL latest is **6.0.7.22** (6.0.7.17 was 24 Feb 2025; .18–.22 followed). The 6.0.7.22
  conda manifest ships **`linux-aarch64`** alongside `linux-64`, `macos-64`, `macos-M1` —
  so FSL runs natively on the DGX Spark GB10 (aarch64).
- **FSL Course 2026:** Bordeaux, France, **22–26 June 2026** (InterContinental Bordeaux) —
  the only FSL course in 2026. Covers FEAT/MELODIC, FLIRT/FNIRT, BET/FAST, SIENA, FDT/TBSS.

### Container recipe changes — `containers/neurocontainer-osl-dynamics/build.yaml`
The recipe only ever installed osl-dynamics (there was **no** legacy `osl` to repin). Updated to:
- Bump `osl-dynamics` 3.0.0 → **3.2.2**; container `version` 3.0.0 → 3.2.2.
- Add `osl-ephys==2.4.0` (pulls `fslpy`; needs FSL binaries at runtime).
- Install **FSL 6.0.7.22** via `fslinstaller.py -d /opt/fsl -V 6.0.7.22 --no_self_update`
  (auto-detects linux-64 vs linux-aarch64). Set `FSLDIR`, `FSLOUTPUTTYPE`, PATH; deploy
  `bet`/`flirt`/`fast` wrappers. Added `python3` + `tcsh` apt deps for the installer/FSL scripts.
- Add **`aarch64`** to `architectures` for the GB10.

### RESOLVED — TensorFlow on aarch64 is CPU-only (verified 2026-06-23)
- `tensorflow==2.17.1` **does** ship `manylinux2014_aarch64` wheels, so it installs on the
  GB10 — but Google builds the ARM wheels **without CUDA** (CPU-only). The `[and-cuda]` extra
  only pulls `nvidia-*-cu12` packages that are **x86_64-only**, so it's a no-op on ARM.
- NVIDIA's GPU TF for ARM/SBSA (NGC `nvcr.io/nvidia/tensorflow`, or the `nvidia-tensorflow`
  wheel on pypi.ngc.nvidia.com) exists, but **NVIDIA ended TF container releases after 25.02**
  (TF 2.17-based) — frozen, and predates the GB10, so Blackwell support there is unverified.
- **Decision (user, 2026-06-23):** accept **CPU TensorFlow on aarch64**. osl-dynamics
  HMM/DyNeMo train on the 20 Grace cores; FSL + osl-ephys preprocessing (the WAND bottleneck)
  run natively. Heavy GPU dynamics training, if needed, goes to an x86 box. The recipe's CUDA
  base image is retained so the **x86_64** target of the same recipe keeps GPU TF.
- **FSL image size:** adds several GB; build time grows. Acceptable for a combined oracle.

## Keras 3 port scoping — can osl-dynamics run GPU on the GB10? (2026-06-23)

Motivation: GPU TF doesn't exist on aarch64/GB10. The only route to GPU osl-dynamics on
the Spark is Keras 3 with a **JAX** backend inside `nvcr.io/nvidia/jax:26.05-py3` (there is
**no NGC TF 26.05** — NVIDIA ended TF containers at 25.02). Scoped against upstream commit
`289f3ea`:

| Metric | Value |
|--------|-------|
| Package | 77 files, ~46.7k LOC |
| Port surface (inference/+models/+data/) | ~17.3k LOC |
| Custom `keras.layers.Layer` subclasses | 43 (all use `tf.*` → need `keras.ops`) |
| Model families | 9 (hmm, hmm_poi, dynemo, mdynemo, hive, dive, dyneste, sc_dynemo, obs_mod) |
| `tensorflow_probability` | 7 files, 38 call sites, 11 distribution types |
| Keras 3 readiness | zero (`from tensorflow.keras import …`) |

Three hard blockers for the JAX backend:
1. **TFP is TF-locked** — swap to `tfp.substrates.jax` or distrax; validate all 11 dists.
2. **HMM Baum-Welch** — custom `@tf.function _baum_welch` using `tensor_scatter_nd_update` /
   `boolean_mask`; needs a from-scratch `jax.lax.scan` reimplementation.
3. **Stateless training** — Keras 3 JAX backend needs stateless `train_step`; the VI ELBO
   (`add_loss`) and HMM E-step/M-step dual-update loops must be rewritten. `tf.data` input
   (26 refs) can stay (Keras 3 JAX accepts it, CPU-side).

Effort: TF-backend migration ~1–3 wk (no GPU win); JAX-backend port (all 9 models) ~2–4+ mo
with numerical-parity validation, plus a maintenance fork vs OHBA's TF upstream.

**Recommendation (not yet decided):** don't port. Reimplement only the needed models natively
— **dynamax** (JAX) for the HMM, **Equinox + Diffrax** for DyNeMo/M-DyNeMo — and keep
osl-dynamics (CPU/TF) as the numerical **validation oracle** (`neurojax/oracle-osl`). Gives
GB10 GPU via JAX, fits NeuroJAX's stack and oracle-container pattern, avoids forking 47k LOC.

## Path C execution — status (verified on the GB10, 2026-06-23)

Decision: **Path C** (reimplement needed models in JAX, keep osl-dynamics as CPU/TF oracle).
Much of it already existed in-repo; verified and extended on the actual DGX Spark (GB10,
aarch64, driver 580.159.03, CUDA 13.0):

- **JAX dynamics models already implemented:** `src/neurojax/models/hmm.py` (`GaussianHMM`,
  full forward/backward/Baum-Welch EM/Viterbi) and `dynemo.py` (Equinox BiGRU inference net
  + Gaussian observation model + ELBO). Deps (equinox/diffrax/optax) already in pyproject.
- **Runs on the GB10 GPU via plain pip — no NGC image needed.** JAX ships CUDA-12 *and*
  CUDA-13 plugin wheels for linux aarch64 (`jax-cuda13-plugin`, `jax-cuda13-pjrt`,
  `jaxlib` → `manylinux_2_27_aarch64`). `pip install "jax[cuda13]"` in a plain Python venv
  (no system CUDA toolkit) gives `jax.devices() -> [CudaDevice(id=0)]`, `backend=gpu`; the
  CUDA 13 runtime + cuDNN are bundled in the wheels, only the host driver is needed.
- **Verified results:** full model suite **66 passed, 2 skipped** on `backend=gpu, cuda:0`
  under both CUDA 12.9 and native CUDA 13.
- **Oracle built (red-green TDD, 2026-06-23).** osl-dynamics 3.2.2 + TF 2.17.1 installed on
  the GB10 (aarch64 CPU); `containers/scripts/{generate_synthetic,run_hmm_baseline}.py` ran to
  dump baseline `.npy` into `tests/data/oracle_osl/hmm/` (1.2 MB fixture). Exact working pins
  captured in `containers/oracle-osl-requirements.txt` and `oracle-osl.Dockerfile` rebuilt on a
  slim CPU base (the old loose ranges break: scikit-image pulls numpy≥2 vs TF/numba; osl-dynamics
  eagerly imports PyYAML + mat73, absent from its wheel metadata).
- **Tests, all green:** `test_hmm_oracle_parity.py` (3 recovery + 2 osl-dynamics parity) and
  `test_oracle_hmm_baseline.py` (7 oracle-validity). Full model suite **75 passed, 0 skipped**.
- **Finding — osl-dynamics shrinks mean parameters.** On the same standardized data both fits
  segment the states well (oracle gamma vs true states 98.8%; neurojax decode 94.8%; mutual
  segmentation agreement asserted), but their *mean parameters* diverge: Baum-Welch's closed-form
  M-step recovers the true means (~0.02 mean-abs error) while osl-dynamics' SGD-learned means
  shrink toward zero (~0.28 error, barely moving from 15→200 epochs). So parity is asserted on
  **state inference**, not mean-parameter equality — and the JAX reimpl is *more* accurate on means.
- **New container** `containers/Dockerfile.dynamics`: slim `python:3.12-slim` + `jax[cuda13]`
  + equinox/optax + Keras 3 (JAX backend), source via PYTHONPATH. Avoids the full pyproject
  (gmsh / jax-fem have **no aarch64 wheels** — they block `uv sync` on the Spark).

- **M-DyNeMo reimplemented** (`src/neurojax/models/mdynemo.py`, TDD). The 2025 power/FC-separating
  model: independent α (means+stds → D_t) and γ (correlations → C_t) mode time courses,
  Σ_t = D_t C_t D_t. Reuses DyNeMo's RNNs + Cholesky helpers; 10 tests green incl. power/FC
  decoupling. Full model suite now **85 passed** GPU-accelerated on the GB10.
- **Oracle Docker verified end-to-end:** `neurojax/oracle-osl` builds and `docker run` produces
  artifacts that pass all 7 oracle-validity tests.

Latest GPU Keras on the gx10 = Keras 3.14.1 on JAX backend (runs on the same wheels). HMM, DyNeMo
and M-DyNeMo now all run GPU on the Spark, validated against the CPU osl-dynamics oracle.

## Relevance to NeuroJAX

- **`neurojax/oracle-osl` container:** the package boundary has moved. Preprocessing /
  source-recon now lives in `osl-ephys`, not the legacy monolithic `osl`. If the container
  still pins legacy `osl`, repin to `osl-ephys` + `osl-dynamics`.
- **`models/` (HMM, DyNeMo):** M-DyNeMo and the canonical / sensor-level HMM are directly
  comparable to the in-repo DyNeMo/HMM implementations.
- **`analysis/SINDy`:** DYSCO (arXiv 2606.13260) is an adjacent system-identification method
  worth comparing against.

---

## Sources

- osl-ephys (Frontiers in Neuroscience 2025): https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1522675/full
- osl-ephys arXiv preprint: https://arxiv.org/abs/2410.22051
- Canonical HMM Networks (bioRxiv 2025): https://www.biorxiv.org/content/10.1101/2025.10.21.683692v1
- Canonical HMM GitHub: https://github.com/OHBA-analysis/Canonical-HMM-Networks
- M-DyNeMo / Uncoupled Power & FC Dynamics (HBM 2025): https://onlinelibrary.wiley.com/doi/10.1002/hbm.70179
- Statistical testing of brain dynamics (Nature Protocols 2025): https://www.nature.com/articles/s41596-025-01300-2
- Structured cycles (Nature Neuroscience 2025): https://www.nature.com/articles/s41593-025-02052-8
- osl-dynamics releases: https://github.com/OHBA-analysis/osl-dynamics/releases
- osl-dynamics PyPI: https://pypi.org/project/osl-dynamics/
- OSL Workshop 2025: https://github.com/OHBA-analysis/osl-workshop-2025-dynamics
- OHBA Analysis Group: https://ohba-analysis.github.io/
- DYSCO (arXiv 2026): https://arxiv.org/abs/2606.13260
