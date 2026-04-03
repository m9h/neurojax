<p align="center">
  <h1 align="center">NeuroJAX</h1>
  <p align="center">
    <em>Differentiable, GPU-accelerated multi-modal neuroimaging</em>
  </p>
</p>

<p align="center">
  <a href="https://github.com/m9h/neurojax/actions/workflows/ci.yml"><img src="https://github.com/m9h/neurojax/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/google/jax"><img src="https://img.shields.io/badge/JAX-Accelerated-9cf?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJMMiAyMmgyMEwxMiAyeiIgZmlsbD0id2hpdGUiLz48L3N2Zz4=" alt="JAX"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://docs.kidger.site/equinox/"><img src="https://img.shields.io/badge/Equinox-Neural_Networks-blueviolet" alt="Equinox"></a>
  <a href="https://github.com/m9h/neurojax"><img src="https://img.shields.io/badge/Source_Imaging-15_methods-green" alt="15 inverse solvers"></a>
  <a href="https://openmeeg.github.io/"><img src="https://img.shields.io/badge/OpenMEEG-BEM-blue" alt="OpenMEEG"></a>
</p>

---

End-to-end gradient descent from sensor measurements to biophysical parameters
--- preprocessing, source imaging, head modeling, and neural dynamics in a
single computational graph.

## Architecture

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                     Multi-Modal Data                             │
  │   MEG (CTF 275ch)  ·  EEG  ·  qMRI  ·  MRS  ·  TMS-EMG       │
  └───────────┬──────────────────────────────────────┬───────────────┘
              │                                      │
              ▼                                      ▼
  ┌───────────────────────┐              ┌───────────────────────────┐
  │    Preprocessing       │              │   Quantitative MRI        │
  │  ASR · ICA · Filtering │              │  DESPOT · QMT · qBOLD    │
  │  Artifact Rejection    │              │  Neural Relaxometry       │
  └───────────┬───────────┘              │  (PINN / Neural ODE)     │
              │                           └─────────────┬─────────────┘
              ▼                                         │
  ┌───────────────────────┐              ┌──────────────▼──────────────┐
  │  Head Modeling         │              │  Tissue Properties          │
  │  CHARM 60-tissue       │◄─────────────│  σ(T1, BPF) conductivity   │
  │  BEM / FEM / OpenMEEG  │              │  Anisotropy from DTI       │
  │  Leadfield L           │              └───────────────────────────┘
  └───────────┬───────────┘
              │  Y = L·J + n
              ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │                   Source Imaging (15 methods)                      │
  │  MNE · dSPM · sLORETA · eLORETA · LCMV · SAM · DICS             │
  │  CHAMPAGNE · VARETA · LAURA · HIGGS · ADMM · PI-GNN             │
  └───────────┬───────────────────────────────────────┬───────────────┘
              │                                       │
              ▼                                       ▼
  ┌───────────────────────┐              ┌───────────────────────────┐
  │  Brain Dynamics        │              │  Connectivity & Stats     │
  │  HMM · DyNeMo · SINDy │              │  Spectral · Graph · GLM  │
  │  Koopman · specparam   │              │  Permutation testing     │
  └────────────────────────┘              └───────────────────────────┘
```

**Everything above is differentiable** --- `jax.grad` flows from the loss
function back through source imaging, the leadfield, FEM assembly,
conductivity mapping, and into qMRI tissue parameters.

## Capabilities

| Module | What it does | Key methods |
|--------|-------------|-------------|
| **Source imaging** | Reconstruct neural current density from M/EEG | MNE, dSPM, sLORETA, eLORETA, LCMV, SAM, DICS, CHAMPAGNE, VARETA, LAURA, HIGGS, ADMM, PI-GNN |
| **Head modeling** | Forward model from anatomy to sensors | 3-layer BEM (MNE + OpenMEEG), CHARM 60-tissue, differentiable FEM, qMRI conductivity |
| **Quantitative MRI** | Tissue property mapping | DESPOT1/mcDESPOT, QMT BPF, multi-echo T2*, qBOLD OEF, B1 correction, PINN/NODE relaxometry |
| **Spectroscopy** | MRS quantification | Phase correction, HLSVD, Tucker/PARAFAC tensor decomposition, coil combination |
| **Preprocessing** | Artifact removal | ASR, ICA, bandpass/notch filtering, interpolation, resampling |
| **Dynamics** | Brain state inference | HMM, DyNeMo, SINDy, Koopman operators, specparam |
| **Statistics** | Inference and testing | Permutation GLM, cluster correction, Riemannian geometry |
| **Graph neural nets** | Cortical mesh learning | PI-GNN with multimodal vertex features (normals, curvature, myelin) |

## The Stack

Built on the [Kidger stack](https://docs.kidger.site/) for scientific ML:

| Layer | Libraries |
|-------|-----------|
| **Neural networks** | [Equinox](https://docs.kidger.site/equinox/) |
| **Dynamics** | [Diffrax](https://docs.kidger.site/diffrax/) (neural ODEs/SDEs) |
| **Optimisation** | [Optax](https://optax.readthedocs.io/) |
| **Linear algebra** | [Lineax](https://docs.kidger.site/lineax/) |
| **Graphs** | [jraph](https://github.com/google-deepmind/jraph) |
| **Forward models** | [OpenMEEG](https://openmeeg.github.io/), [JAX-FEM](https://github.com/deepmodeling/jax-fem) + PETSc |
| **Data I/O** | [MNE-Python](https://mne.tools/), nibabel, meshio |

## Quick start

```bash
# Install with uv (recommended)
uv sync

# Run test suite (120+ tests)
uv run pytest tests/ -v
```

### Source imaging with PI-GNN

```python
from neurojax.source import SourceGNN, mesh_to_graph, estimate_tikhonov_reg

graph = mesh_to_graph(vertices, faces)
reg = estimate_tikhonov_reg(leadfield)  # always explicit, always reported

model = SourceGNN(n_features=6, n_times=300, hidden_dim=64,
                  tikhonov_reg=reg, key=jax.random.PRNGKey(0))
model, losses = train_source_gnn(model, Y, L, graph, features, normals, laplacian)
J = model(Y, L, graph, features)
```

### Differentiable head modeling

```python
from neurojax.geometry.fem_forward import assemble_stiffness, sigma_from_qmri

sigma = sigma_from_qmri(t1_values, bpf_values, labels, params)  # differentiable
K = assemble_stiffness(vertices, elements, sigma)
grad_params = jax.grad(loss)(params)  # end-to-end through FEM
```

## Inverse solvers

15 source imaging methods spanning five families:

| Family | Methods | Key innovation |
|--------|---------|----------------|
| L2 minimum norm | MNE, dSPM, sLORETA, eLORETA | Noise-normalised distributed estimates |
| Biophysical prior | LAURA, VARETA | 1/d^3 decay, adaptive resolution |
| Beamformers | LCMV, SAM, DICS | Adaptive spatial filters |
| Bayesian sparse | CHAMPAGNE, HIGGS | Type-II ML, joint source + connectivity |
| Deep learning | PI-GNN | Physics-informed graph convolution on cortical mesh |

## Head modeling pipeline

```
T1w image
  ──► CHARM 60-tissue segmentation (SAMSEG + extended atlas)
  ──► Tissue surface extraction (marching cubes)
  ──► Conductivity assignment (literature + qMRI-informed)
  ──► BEM (MNE / OpenMEEG) or FEM (JAX-FEM + PETSc)
  ──► Leadfield matrix L
  ──► Source imaging (any of 15 inverse methods)
```

## Research lineage

NeuroJAX synthesises decades of computational neuroscience research:

> **CAUCHY/SimBio** (1993+, Wolters, Buchner, Anwander; Münster/Jena)
> --- foundational FEM for bioelectric forward modeling. NeuroFEM kernel
> underlies BESA MRI and CURRY.
>
> **OpenMEEG** (2000s, Clerc, Papadopoulo, Gramfort; INRIA Rennes
> Athena/Odyssee) --- symmetric BEM and subtraction method for dipole
> singularity.
>
> **Wendling neural mass models** (2002+, Wendling, Bartolomei,
> Bellanger; LTSI/INSERM Rennes) --- cortical column dynamics for
> EEG simulation.
>
> **Fijee** (2014, programmed by Yann Cobigo, now at UCSF; from
> Wendling's group) --- C++/FEniCS implementation coupling FEM forward
> with neural mass dynamics and dense-array EEG source connectivity
> (Hassan et al. 2014, PLOS ONE).
>
> **VARETA / HIGGS** (Valdes-Sosa et al., Cuban Neuroscience Centre)
> --- adaptive-resolution inverse and joint source-connectivity via
> hidden Gaussian graphical spectral models.
>
> **LAURA** (Grave de Peralta Menendez et al.) --- biophysical 1/d^3
> spatial prior for source localisation.
>
> **OSL / osl-dynamics** (Oxford, OHBA) --- HMM and DyNeMo brain state
> analysis.
>
> **SCI Head Model** (Warner et al. 2019, Utah) --- high-resolution
> tetrahedral head model for EEG, EIT, and transcranial ultrasound.
>
> **Kidger stack** (Patrick Kidger) --- Equinox, Diffrax, Lineax,
> Optimistix for scientific ML in JAX.
>
> **WAND** (Welsh Advanced Neuroimaging Database, CUBRIC) --- 170-subject
> multi-modal dataset (7T MRI, 300 mT/m gradients, CTF MEG, TMS-EMG).

## Tutorials

| Tutorial | Topic |
|----------|-------|
| [MEG pipeline](docs/tutorials/meg_pipeline.md) | End-to-end from raw CTF data to source-level HMM |
| [MRS MEGA-PRESS](docs/tutorials/mrs_mega_press.md) | Spectroscopy: loading, preprocessing, quantification |
| [Beamforming](docs/tutorials/beamforming.md) | LCMV theory and implementation |
| [ASR preprocessing](docs/tutorials/preprocessing_asr.md) | Artifact subspace reconstruction |
| [GLM inference](docs/tutorials/glm_inference.md) | Permutation testing and cluster correction |
| [ICA source separation](docs/tutorials/ica_source_separation.md) | FastICA and probabilistic ICA |
| [Source imaging](docs/tutorials/source_imaging.md) | LAURA, VARETA, PI-GNN comparison |
| [Head modeling](docs/tutorials/head_modeling.md) | BEM, FEM, CHARM, qMRI conductivity |

## Examples

59 runnable scripts in [`examples/`](examples/) covering MEG/EEG
workflows, source imaging, spectral analysis, sleep staging, neural
dynamics, WAND dataset processing, and validation against established
tools.

## Testing

```bash
uv run pytest tests/ -v                    # full suite (120+ tests)
uv run pytest tests/test_source_gnn.py -v  # PI-GNN (26 tests)
uv run pytest tests/test_laura.py -v       # LAURA (9 tests)
uv run pytest tests/test_qmri.py -v        # qMRI (22 tests)
```

## Citation

If you use NeuroJAX, please cite the WAND dataset and relevant method
papers (see `paper/references.bib`).

For computational modeling methodology:

> Hess AJ, Iglesias S, Koechli L, Marino S, Mueller-Schrader M, Rigoux L,
> Mathys C, Harrison OK, Heinzle J, Fraessle S, Stephan KE (2025).
> Bayesian Workflow for Generative Modeling in Computational Psychiatry.
> *Computational Psychiatry* 9(1):76--99. doi:10.5334/cpsy.116

## License

MIT
