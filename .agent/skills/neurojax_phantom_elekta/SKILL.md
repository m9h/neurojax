---
name: neurojax_phantom_elekta
description: Validate Inverse Solver Accuracy using the MNE Elekta Phantom dataset.
---

# Elekta Phantom Validation

This agent validates the localization accuracy of `neurojax.source.inverse_scico` using the physical **Elekta Phantom** dataset (available via MNE-Python).

## Objectives
1.  **Data Ingestion**: Download/Load the `phantom_elekta` dataset using `mne.datasets.phantom_4dbti` (or similar standard set).
    - *Note*: The standard MNE sample dataset includes phantom data. Check `mne.datasets.sample` or `mne.datasets.hf_sef` if specific phantom sets are unavailable. Use `mne.dipole.get_phantom_dipoles` to get ground truth.
2.  **Preprocessing**:
    - MaxFilter (if raw).
    - Epoching (if continuous).
3.  **Inverse Solve**:
    - Run `neurojax.source.inverse_scico.solve_inverse_admm` (Native ADMM) on the evoked data for each dipole.
4.  **Error Quantification**:
    - Compute **Euclidean Distance** between the estimated peak source location and the known ground truth dipole location.
    - Check if error is within acceptable bounds (< 5mm for good SNR).

## Instructions
1.  Create/Run `examples/demo_phantom_elekta.py`.
2.  Use MNE utilities to fetch the phantom data and ground truth coordinates.
3.  Convert MNE Forward/Data structures to JAX arrays.
4.  Solve and compute error statistics.
5.  Generate a report `results_phantom_elekta/report.md`.
