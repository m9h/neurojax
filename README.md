[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# NeuroJAX

Differentiable, GPU-accelerated toolkit for multi-modal neuroimaging analysis.
End-to-end gradient descent from sensor measurements to biophysical parameters
--- preprocessing, source imaging, head modeling, and neural dynamics in a
single computational graph.

## Capabilities

| Module | What it does | Key methods |
|--------|-------------|-------------|
| **Source imaging** | Reconstruct neural current density from M/EEG | MNE, dSPM, sLORETA, eLORETA, LCMV, SAM, DICS, CHAMPAGNE, VARETA, LAURA, HIGGS, ADMM, PI-GNN |
| **Head modeling** | Forward model from anatomy to sensors | 3-layer BEM (MNE + OpenMEEG), CHARM 60-tissue segmentation, differentiable FEM, qMRI-derived conductivity |
| **Quantitative MRI** | Tissue property mapping | DESPOT1/mcDESPOT T1, QMT bound pool fraction, multi-echo T2*, qBOLD OEF, B1 correction, neural relaxometry (PINN/NODE) |
| **Spectroscopy** | MRS quantification | Phase correction, HLSVD water removal, Tucker/PARAFAC tensor decomposition, coil combination |
| **Preprocessing** | Artifact removal and filtering | ASR, ICA, bandpass/notch filtering, interpolation, resampling |
| **Dynamics** | Brain state inference | HMM, DyNeMo, SINDy, Koopman operators, specparam |
| **Statistics** | Inference and testing | Permutation GLM, cluster correction, Riemannian geometry |
| **Graph neural networks** | Cortical mesh learning | PI-GNN source imaging with multimodal vertex features |

## The Stack

Built on the [Kidger stack](https://docs.kidger.site/) for scientific machine learning:

- **JAX** + **Equinox** --- neural network modules and pytrees
- **Diffrax** --- neural ODEs/SDEs for hemodynamic and neural mass models
- **Optax** --- gradient-based optimisation
- **Lineax** --- linear solvers (GLM, beamforming)
- **jraph** --- graph neural networks on cortical meshes
- **OpenMEEG** --- symmetric BEM forward solutions
- **JAX-FEM** + **PETSc** --- differentiable finite element method
- **MNE-Python** --- data I/O, sensor layouts, BEM surfaces

## Quick start

```bash
uv sync          # install all dependencies
uv run pytest    # run test suite (120+ tests)
```

### Source imaging with PI-GNN

```python
from neurojax.source import SourceGNN, mesh_to_graph, estimate_tikhonov_reg

# Build graph from cortical mesh
graph = mesh_to_graph(vertices, faces)

# Estimate regularisation from data (transparent, reported)
reg = estimate_tikhonov_reg(leadfield)

# Train physics-informed GNN
model = SourceGNN(n_features=6, n_times=300, hidden_dim=64,
                  tikhonov_reg=reg, key=jax.random.PRNGKey(0))
model, losses = train_source_gnn(model, Y, L, graph, features,
                                  normals, laplacian)
J = model(Y, L, graph, features)  # source estimates
```

### Differentiable head modeling

```python
from neurojax.geometry.charm import run_charm_segmentation, assign_conductivities
from neurojax.geometry.fem_forward import assemble_stiffness, solve_forward

# CHARM 60-tissue segmentation
seg_path = run_charm_segmentation(t1_path, output_dir)

# qMRI-informed conductivity (differentiable)
sigma = sigma_from_qmri(t1_values, bpf_values, tissue_labels, params)

# FEM forward solve --- gradients flow through everything
K = assemble_stiffness(vertices, elements, sigma)
phi = solve_forward(vertices, elements, sigma, source_rhs)
grad_sigma = jax.grad(loss)(sigma)  # end-to-end
```

### Quantitative MRI fitting

```python
from neurojax.qmri.despot import despot1_fit
from neurojax.qmri.neural_relaxometry import BlochNeuralODE

# Classical DESPOT1
T1_map, M0_map = despot1_fit(spgr_data, flip_angles, TR)

# Neural ODE relaxometry (learns corrections to Bloch equations)
model = BlochNeuralODE(hidden_dim=32, key=jax.random.PRNGKey(0))
```

## Research context

NeuroJAX draws on several research lineages:

- **OSL / osl-dynamics** (Oxford, OHBA) --- HMM and DyNeMo brain state
  analysis. NeuroJAX provides an osl-dynamics-compatible data pipeline
  with GPU-accelerated preprocessing and source reconstruction.

- **VARETA / HIGGS / BC-VARETA** (Cuban Neuroscience Centre,
  Valdes-Sosa et al.) --- adaptive-resolution inverse solutions and
  joint source-connectivity estimation via hidden Gaussian graphical
  spectral models. Implemented in JAX with automatic differentiation.

- **LAURA** (Grave de Peralta Menendez et al.) --- biophysical 1/d^3
  spatial prior for source localisation, distinguishing it from
  Laplacian-based LORETA.

- **Fijee** (github.com/m9h/Fijee-Project) --- the predecessor C++/FEniCS
  project for FEM-based EEG forward modeling with anisotropic
  conductivity, dipole subtraction, and complete electrode model for
  tDCS/EIT. NeuroJAX ports this to differentiable JAX.

- **SCI Head Model** (Utah, Warner et al. 2019) --- high-resolution
  tetrahedral head model used across EEG, EIT, and transcranial
  ultrasound simulation.

- **Kidger stack** (Patrick Kidger) --- Equinox, Diffrax, Lineax,
  Optimistix for scientific machine learning in JAX.

- **WAND** (Welsh Advanced Neuroimaging Database, CUBRIC) --- 170-subject
  multi-modal dataset (7T MRI, 300 mT/m gradients, CTF MEG, TMS-EMG)
  providing the validation data for all pipelines.

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
  --> CHARM 60-tissue segmentation (SAMSEG + extended atlas)
  --> Tissue surface extraction (marching cubes)
  --> Conductivity assignment (literature + qMRI-informed)
  --> BEM (MNE / OpenMEEG) or FEM (JAX-FEM + PETSc)
  --> Leadfield matrix L
  --> Source imaging (any of 15 inverse methods)
```

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

## 59 examples

See [`examples/`](examples/) for runnable scripts covering MEG/EEG
workflows, source imaging, spectral analysis, sleep staging, neural
dynamics, WAND dataset processing, and validation against established
tools.

## Testing

```bash
uv run pytest tests/ -v              # unit tests
uv run pytest tests/test_source_gnn.py  # PI-GNN (26 tests)
uv run pytest tests/test_laura.py       # LAURA (9 tests)
uv run pytest tests/test_qmri.py        # qMRI (22 tests)
```

## Citation

If you use NeuroJAX in your research, please cite the WAND dataset
and the relevant method papers (see `paper/references.bib`).

## License

MIT
