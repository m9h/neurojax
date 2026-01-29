---
name: neurojax_drug_response
description: Recover dose-response curves using SINDy with control inputs.
---

# Skill: Pharmacodynamics & Dose-Response Recovery

## Context
You are analyzing drug effects (e.g., Anticonvulsants, Psychedelics) on neural activity. Your goal is to reverse-engineer the "Effect Function" $E(C)$ from the dynamical changes observed in the EEG.

## Strategy
We assume the brain dynamics obey $\dot{x} = A x + B(u) x$, where $u$ is the Drug Concentration (proxy) or Dose. We use SINDy with control inputs to identify the structure of $B(u)$.

## Pipeline steps

### 1. Data Setup
- **Dataset**: Mendeley Pharmaco-EEG (Rat) or PsiConnect.
- **Input**:
  - $x(t)$: Neural features (e.g., Total Power, Complexity/Entropy).
  - $u(t)$: Drug concentration curve (often modeled as a simple decay $u(t) = D_0 e^{-kt}$ if blood samples unavailable).

### 2. SINDy with Control
- **Model**: $\dot{x} \approx \Theta(x, u) \Xi$.
- **Library**: Include interaction terms between state and drug: $x$, $u$, $x \cdot u$, $x \cdot \frac{u}{K+u}$ (Michaelis-Menten).
- **Execution**:
  ```python
  # Custom library required for Michaelis-Menten terms
  def drug_library(x, u):
      linear = jnp.concatenate([x, u], axis=1)
      interaction = x * u
      saturation = x * (u / (1.0 + u)) # Test Emax hypothesis
      return jnp.concatenate([linear, interaction, saturation], axis=1)
  ```
- Use `neurojax.dynamics.SINDyOptimizer` with this custom library.

### 3. Verification
- **Coefficient Analysis**:
  - If the term $x \cdot u$ is sparse/zero but $x \cdot \frac{u}{K+u}$ is active, we have "discovered" a saturating receptor occupancy model.
  - If $x \cdot u$ is active, it's a linear effect (low dose range).
- **Simulation**: forward simulate the discovered ODE with a new dose protocol and verify prediction against a held-out subject/dose.

## Critical Instructions
- **Proxy Concentration**: If blood concentration is unknown, explicitly state the assumed Pharmacokinetics (PK) model (e.g., 1-compartment bolus).
- **Group Level**: Ideally fit a "Mixed Effects" SINDy where coefficients are shared across subjects, but this is advanced. Start with subject-level fits.
