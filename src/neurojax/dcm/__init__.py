"""Dynamic causal modelling for cross-spectral density, ported from SPM to JAX.

:mod:`neurojax.dcm.cmc_csd` is the canonical-microcircuit CSD model of
SPM's DCM for M/EEG (``spm_fx_cmc``, ``spm_dcm_delay``, ``spm_dcm_mtf``,
``spm_csd_mtf``, ``spm_csd_mtf_gu``, ``spm_gen_Q``, ``spm_lx_erp``), written
so that ``jax.grad`` flows through the predicted cross-spectra.  It is
verified against the predictions stored in SPM's own fitted DCM for the
two-channel rodent LFP example (``docs/dcm_cmc_verification.md``).
"""

from neurojax.dcm.cmc_csd import (
    CMC_DEFAULTS,
    cmc_csd,
    cmc_priors,
    cmc_priors_from_oracle,
    four_node_test_architecture,
    delayed_jacobian,
    fx_cmc,
    gen_Q,
    innovations,
    transfer_function,
)

__all__ = ["CMC_DEFAULTS", "fx_cmc", "delayed_jacobian", "transfer_function", "innovations",
           "gen_Q", "cmc_csd", "cmc_priors", "cmc_priors_from_oracle", "four_node_test_architecture"]
