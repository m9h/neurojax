"""neurojax.qmri — Differentiable quantitative MRI signal models in JAX.

GPU-accelerated, end-to-end differentiable implementations of qMRI
relaxometry models. All models are pure JAX functions compatible with
jax.grad, jax.vmap, jax.jit.

Modules:
    steady_state  — SPGR, bSSFP, IR-SPGR signal equations
    despot        — DESPOT1/2/mcDESPOT fitting (NLLS + Bayesian)
    qmt           — Two-pool magnetization transfer (Ramani/Sled-Pike)
    multiecho     — Multi-echo T2/T2* fitting
    mp2rage       — MP2RAGE T1 estimation
    bloch         — Differentiable Bloch equation simulation (NODE)
    fitting       — Voxelwise fitting engine (vmap + optax)
"""

from neurojax.qmri.steady_state import spgr_signal, bssfp_signal, ir_spgr_signal
from neurojax.qmri.despot import despot1_fit, despot1hifi_fit
from neurojax.qmri.multiecho import monoexp_t2star_fit
