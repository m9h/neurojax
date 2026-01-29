[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NeuroJAX** (OSL-JAX) is a differentiable, GPU-accelerated reimplementation of the OSL (Oxford Centre for Human Brain Activity) analysis stack.

## Vision
To unify **Preprocessing** (osl-ephys) and **Modelling** (osl-dynamics) into a single computational graph, enabling end-to-end gradient descent from sensor error to biophysical parameters.

## The Stack
* **Core:** `JAX`, `Equinox`
* **Solvers:** `Lineax` (GLM/Beamforming), `Optimistix`
* **Dynamics:** `Diffrax` (Neural ODEs / DCM)
* **Inverse:** `Scico` (Sparse/Iterative solvers)

## Roadmap
1.  **GLM:** Mass-univariate permutation testing on GPU (`src/neurojax/glm.py`).
2.  **Inverse:** Differentiable Beamformers and CHAMPAGNE algorithm.
3.  **Biophysics:** Differentiable implementations of Wong-Wang and Canonical Microcircuit models (replacing TVB/DCM).
4.  **Foundation:** Mamba-based sequence modelling for whole-brain dynamics.

### Installation
We recommended using [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies and project
uv sync
```

### GLM Permutation Testing
```python
from neurojax.glm import GeneralLinearModel, run_permutation_test
# See examples/demo_glm.py for full usage
```
