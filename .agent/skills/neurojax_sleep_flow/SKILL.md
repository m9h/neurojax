---
name: neurojax_sleep_flow
description: Model Wake-NREM transitions as a continuous Neural ODE flow.
---

# Skill: Continuous Sleep Flow Modeling

## Context
Sleep stages (N1, N2, N3) are artificial discretizations of a continuous biological process. Your goal is to model the "Sleep Manifold" using Neural ODEs on the OpenNeuro/Bitbrain sleep datasets.

## Strategy
Instead of classification (State $\in \{0,1,2\}$), we learn a velocity field $\dot{x} = f_\theta(x, t)$ that describes the flow of brain dynamics on a manifold defined by band powers.

## Pipeline steps

### 1. Preprocessing & Feature Extraction
- **Input**: BIDS-formatted EEG (e.g., OpenNeuro).
- **Resample**: 100 Hz.
- **Features**: Compute relative band powers in log-space ($\log P_\delta, \log P_\theta, \dots$) for every 30s epoch (or finer 5s resolution for continuity).
- **Manifold**: Treat this 5D vector as the state $x(t)$.

### 2. Neural ODE Training
- **Model**: Define a `diffrax.NeuralODE` using `equinox`.
  ```python
  class VectorField(eqx.Module):
      mlp: eqx.nn.MLP
      def __call__(self, t, y, args):
          return self.mlp(y)
  ```
- **Training**:
  - Use `diffrax.diffeqsolve` to predict $x(t+\Delta t)$ from $x(t)$.
  - Loss: MSE between predicted band power trajectory and actual future trajectory.
  - *Advanced*: Use "Multiple Shooting" for long sequences to stabilize training.

### 3. Analysis: Flow Visualization
- **Streamplot**: Visualize the learned vector field $f_\theta(x)$ in the $(\delta, \alpha)$ plane.
- **Attractors**: Identify fixed points ($\dot{x} \approx 0$).
  - Expected: A "Wake" attractor (High $\alpha$, Low $\delta$) and a "Deep Sleep" attractor (High $\delta$, Low $\alpha$).
- **Success**: The vector field shows a clear "river" or trajectory connecting Wake to Sleep, with N1/N2 as transient states along the flow.

## Critical Instructions
- **Normalization**: Log-band powers must be z-scored per subject before training.
- **Stiffness**: Sleep dynamics can be stiff (fast spindles vs slow waves). Use `diffrax.Tsit5` or `diffrax.Kvaerno5` with adaptive stepping.
