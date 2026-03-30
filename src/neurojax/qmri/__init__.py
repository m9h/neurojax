"""neurojax.qmri — Differentiable quantitative MRI in JAX.

GPU-accelerated, end-to-end differentiable implementations of qMRI
relaxometry, MRS spectroscopy, and neural physics models.

Modules:
    steady_state        — SPGR, bSSFP, IR-SPGR, multi-echo, QMT, MP2RAGE
    despot              — DESPOT1/2/mcDESPOT fitting (NLLS + Bayesian)
    multiecho           — Multi-echo T2/T2* fitting
    fitting             — Generic VoxelwiseFitter (vmap + optax + Laplace)
    roi                 — FreeSurfer ROI extraction + multi-tool comparison
    io                  — NIfTI/MGZ ↔ JAX I/O
    b1                  — B1+ field correction
    pulse_sequence      — Equinox pulse sequence dataclasses
    neural_relaxometry  — BlochNeuralODE, MultiCompartmentNODE, RelaxometryPINN
    mrs                 — MRS preprocessing (phase, freq align, HLSVD, coil combine)
    mrs_tensor          — Multiway analysis (Tucker, PARAFAC, MCR-ALS)
"""

from neurojax.qmri.steady_state import spgr_signal, bssfp_signal, ir_spgr_signal
from neurojax.qmri.despot import despot1_fit, despot1hifi_fit
from neurojax.qmri.multiecho import monoexp_t2star_fit
from neurojax.qmri.io import load_nifti, save_nifti
from neurojax.qmri.roi import extract_roi_stats, extract_tissue_stats, compare_tools
from neurojax.qmri.b1 import correct_fa_for_b1, correct_t1_for_b1
