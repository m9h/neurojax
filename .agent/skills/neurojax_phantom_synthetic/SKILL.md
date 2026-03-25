---
name: neurojax_phantom_synthetic
description: Quantify Inverse Solver Leakage and Resolution using a Synthetic Phantom.
---

# Synthetic Phantom Validation

This agent validates the resolution properties of the `neurojax` inverse solvers (Native ADMM/Scico) by computing the **Resolution Matrix** and quantifying leakage artifacts.

## Objectives
1.  **Generate Synthetic Source Space**: Create a random or grid-based source space and corresponding random Leadfield $L$.
2.  **Compute Resolution Matrix**: Calculate $R = G \cdot L$ for the Native ADMM solver.
3.  **Quantify Leakage**:
    - **Peak Localization Error (PLE)**: Distance between the peak of the PSF (row of R) and the true source.
    - **Spatial Dispersion (SD)**: Spread of the PSF around the peak.
    - **Crosstalk**: Magnitude of off-diagonal elements in the CTF (column of R).
4.  **Visualize**: Plot PSF and CTF for representative sources.

## Instructions
1.  Create/Run `examples/demo_phantom_synthetic.py`.
2.  The script must:
    - Simulate a setup with $N_{sensors} \approx 64$ and $N_{sources} \approx 256$.
    - Run `inverse_scico.solve_inverse_admm` to get the inverse operator $G$ (or approximate it via linearity if using L1). 
    - *Note*: For non-linear solvers (L1), $R$ is strictly state-dependent. Use small perturbations or a linear approximation (Weighted MNE) for stable Resolution Matrix analysis, OR empirically compute PSF by inverting unit impulses $\delta_i$.
    - Save plots to `results_phantom_synthetic/`.
3.  Report statistical metrics (Mean/Max PLE).
