---
name: neurojax_anesthesia_dynamics
description: Detect Loss of Consciousness (LOC) using SINDy-based bifurcation analysis on EEG.
---

# Skill: Anesthesia Dynamics & Bifurcation Analysis

## Context
You are an expert in nonlinear dynamics and EEG processing. Your goal is to detect the transition from wakefulness to unconsciousness (LOC) in the PhysioNet GABAergic Anesthesia dataset using `neurojax`.

## Strategy
We model the brain state as a dynamical system $\dot{z} = f(z; \mu)$. As the drug concentration ($\mu$) increases, the system undergoes a **Hopf Bifurcation** (stable limit cycle $\to$ stable fixed point). We track the eigenvalues of the Jacobian $J = \partial f / \partial z$.

## Pipeline steps

### 1. Data Ingestion
- Use `neurojax.io.load_physionet` (if available) or standard `mne` loader for EDF files.
- **Filter**: Bandpass 0.1-40 Hz to align with slow-wave dynamics.
- **Reference**: Use average reference or Laplacian if HD-EEG.

### 2. Dimensionality Reduction
- **Spectrogram**: Compute power in standard bands ($\delta, \theta, \alpha, \beta, \gamma$).
- **Embedding**: Use `sklearn.decomposition.PCA` or `neurojax.preprocessing.pca` to reduce the high-dim spectrogram/channel space to a latent trajectory $z(t) \in \mathbb{R}^3$.

### 3. Continuous Dynamics (The Core)
- **Windowing**: Iterate over sliding windows (e.g., 30s window, 5s step).
- **SINDy Fit**: For each window $W_t$, use `neurojax.dynamics.SINDyOptimizer`:
  ```python
  from neurojax.dynamics import SINDyOptimizer, polynomial_library
  opt = SINDyOptimizer(threshold=0.01)
  Xi = opt.fit(z, dz, polynomial_library)
  ```
- **Stability Analysis**:
  - Construct the Jacobian $J$ from coefficients `Xi`.
  - Compute eigenvalues $\lambda = \text{eig}(J)$.
  - **Metric**: Track $\max(\text{Re}(\lambda))$. A crossing from + (unstable/oscillatory) to - (stable) usually indicates LOC (suppression of alpha/beta).

### 4. Verification Check
- **Plot**: Time vs $\max(\text{Re}(\lambda))$.
- **Overlay**: Expert annotations of LOC boundaries.
- **Success**: The metric shifts significantly at the annotated LOC time.

## Critical Instructions
- **Do not interpolate**: Use the raw irregular data if possible, but SINDy usually needs a grid. If needed, resample to uniform `dt`.
- **Noise Sensitivity**: Derivatives `dz` are noisy. Use `diffrax` or a robust differentiator (Savitzky-Golay) before passing to SINDy.
