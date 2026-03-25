---
name: neurojax_phantom_ctf
description: Validate BEM Forward Model Accuracy using the MNE CTF Phantom dataset.
---

# CTF Phantom Validation

This agent validates the accuracy of the `neurojax.geometry.bem_jinns` (PINN Forward Model) using the **CTF Phantom** dataset.

## Objectives
1.  **Data Ingestion**: Load the CTF Phantom data (known spherical geometry).
2.  **BEM Forward Solve**:
    - Set up the `BemSolver` (PINN) to model the conducting sphere.
    - Compute the potential field at the sensor locations for the known active dipoles.
3.  **Analytical Comparison**:
    - Compute the **Analytical Solution** (Sarvas formula for sphere) for the same dipoles.
4.  **Error Quantification**:
    - **Relative Difference Measure (RDM)**: $\sqrt{\sum (y_{est} - y_{ref})^2 / \sum y_{ref}^2}$.
    - **Magnitude Error (MAG)**.

## Instructions
1.  Create/Run `examples/demo_phantom_ctf.py`.
2.  Define the spherical geometry in the PINN.
3.  Compare PINN predictions vs Analytical sphere model for the specific sensor layout of the CTF system.
4.  Report accuracy metrics.
