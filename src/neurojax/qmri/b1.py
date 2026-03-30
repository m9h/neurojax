"""B1+ field correction for quantitative MRI maps."""

import jax
import jax.numpy as jnp
from neurojax.qmri.steady_state import spgr_signal_multi
from neurojax.qmri.despot import despot1hifi_fit


def correct_fa_for_b1(nominal_fa_rad: jnp.ndarray,
                       b1_ratio: jnp.ndarray) -> jnp.ndarray:
    """Scale nominal flip angles by B1+ ratio map.

    actual_FA = nominal_FA * B1_ratio
    B1_ratio = 1.0 means perfect transmit field.
    """
    return nominal_fa_rad * b1_ratio


def correct_t1_for_b1(t1_map: jnp.ndarray,
                       b1_ratio: jnp.ndarray,
                       nominal_fa_deg: float = 10.0,
                       TR: float = 0.004) -> jnp.ndarray:
    """Post-hoc T1 correction given a B1 ratio map.

    Corrects the bias introduced by fitting DESPOT1 with nominal flip
    angles when the actual flip angles differ due to B1 inhomogeneity.

    Uses the analytical SPGR signal model to compute what T1 would have
    been estimated as under the true B1, then inverts.
    """
    fa_nom = jnp.deg2rad(nominal_fa_deg)
    fa_actual = fa_nom * b1_ratio

    E1_apparent = jnp.exp(-TR / t1_map)
    # The apparent E1 was fitted assuming nominal FA
    # True E1 satisfies: S(true_FA, true_T1) = S(nom_FA, apparent_T1)
    # For small corrections: T1_true ≈ T1_apparent * (sin(fa_actual)/sin(fa_nom)) * ...
    # When B1 < 1, actual FA < nominal FA, DESPOT1 underestimates T1.
    # Correction: T1_true ≈ T1_apparent * (tan(fa_nom) / tan(fa_actual))
    correction = jnp.tan(fa_nom) / jnp.tan(fa_actual)
    t1_corrected = t1_map * jnp.clip(correction, 0.5, 2.0)

    return t1_corrected


def estimate_b1_map(spgr_data, ir_data, spgr_fa_rad, ir_fa_rad,
                     TR_spgr, TR_ir, TI):
    """Estimate B1+ ratio map from SPGR + IR-SPGR data (DESPOT1-HIFI).

    Returns B1 ratio map (1.0 = nominal).
    """
    fit_fn = jax.vmap(
        lambda s, i: despot1hifi_fit(s, i, spgr_fa_rad, ir_fa_rad,
                                      TR_spgr, TR_ir, TI)
    )
    # This requires flattened voxel arrays as input
    results = fit_fn(spgr_data, ir_data)
    return results["B1"]
