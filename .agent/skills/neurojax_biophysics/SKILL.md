---
name: neurojax_biophysics
description: Guidelines for implementing biophysical neural mass models in NeuroJAX.
---

# NeuroJAX (OSL-JAX) Biophysics

## Goal
To implement differentiable biophysical models (Neural Masses) that can be fitted to data using `diffrax` and `optimistix`.

## Physics Kernels
All models should inherit from a common base and solve ODEs/SDEs.

### Wong-Wang (Reduced)
- **Use Case**: Whole-brain functional connectivity fitting.
- **Complexity**: Low (2 variables).
- **Implementation**: See `vbjax` for reference equations. Wraps in `equinox.Module`.

### Canonical Microcircuit (CMC)
- **Use Case**: Layer-specific inference (Laminar Dynamics).
- **Complexity**: High (4 populations: SS, SP, II, DP).
- **Origin**: SPM Dynamic Causal Modelling (DCM).
- **Implementation**: Needs `diffrax` ODE solver.

## Implementation Pattern
```python
class AbstractNeuralMass(eqx.Module):
    def vector_field(self, t, y, args):
        raise NotImplementedError

class WongWang(AbstractNeuralMass):
    coupling: float
    def vector_field(self, t, y, args):
        # dx/dt = ...
        return dS
```
