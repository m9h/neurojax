[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NeuroJAX** is a differentiable, GPU-accelerated cli tool for EMEG processing and non linear dynamical systems analysis

## Vision
To unify **Preprocessing** and **Modelling** into a single computational graph, enabling end-to-end gradient descent from sensor error to biophysical parameters.

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
