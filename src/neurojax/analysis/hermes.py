"""HERMES 4-condition spectral editing for simultaneous GABA and GSH.

HERMES (Hadamard Encoding and Reconstruction of MEGA-Edited Spectroscopy)
uses 4 editing conditions (A, B, C, D) with a Hadamard encoding scheme:

    Condition A: GABA_ON,  GSH_ON
    Condition B: GABA_ON,  GSH_OFF
    Condition C: GABA_OFF, GSH_ON
    Condition D: GABA_OFF, GSH_OFF

Difference spectra:
    GABA = (A + B) - (C + D)
    GSH  = (A + C) - (B + D)

References:
    Chan et al. (2016) HERMES: Hadamard encoding and reconstruction of
    MEGA-edited spectroscopy. MRM 76:11-19.

    Saleh et al. (2016) Multi-step spectral editing of in vivo 1H-MRS.
    NeuroImage 134:360-367.
"""

import numpy as np
from typing import NamedTuple


class HermesResult(NamedTuple):
    """Result of HERMES processing."""
    gaba_diff: np.ndarray     # GABA difference spectrum: (A+B) - (C+D)
    gsh_diff: np.ndarray      # GSH difference spectrum: (A+C) - (B+D)
    conditions: np.ndarray    # Averaged conditions, shape (n_spec, 4)
    n_averages: int           # Number of averages per condition
    dwell_time: float         # Dwell time in seconds
    bandwidth: float          # Spectral bandwidth in Hz


def process_hermes(
    data: np.ndarray,
    dwell_time: float,
    centre_freq: float = 123.0e6,
    align: bool = False,
) -> HermesResult:
    """Process HERMES 4-condition data to GABA and GSH difference spectra.

    Parameters
    ----------
    data : ndarray, shape (n_spec, 4, n_dyn)
        HERMES data with 4 editing conditions (A, B, C, D).
    dwell_time : float
        Dwell time in seconds.
    centre_freq : float
        Spectrometer frequency in Hz.
    align : bool
        Whether to perform frequency/phase alignment (not yet implemented).

    Returns
    -------
    HermesResult
    """
    if data.ndim != 3:
        raise ValueError(f"Expected 3D data (n_spec, 4, n_dyn), got {data.ndim}D")

    n_spec, n_cond, n_dyn = data.shape
    if n_cond != 4:
        raise ValueError(f"Expected 4 conditions, got {n_cond}")

    bw = 1.0 / dwell_time

    # Average each condition across dynamics
    cond_a = data[:, 0, :].mean(axis=1)  # (n_spec,)
    cond_b = data[:, 1, :].mean(axis=1)
    cond_c = data[:, 2, :].mean(axis=1)
    cond_d = data[:, 3, :].mean(axis=1)

    # Hadamard reconstruction
    gaba_diff = (cond_a + cond_b) - (cond_c + cond_d)
    gsh_diff = (cond_a + cond_c) - (cond_b + cond_d)

    # Store averaged conditions
    conditions = np.column_stack([cond_a, cond_b, cond_c, cond_d])

    return HermesResult(
        gaba_diff=gaba_diff,
        gsh_diff=gsh_diff,
        conditions=conditions,
        n_averages=n_dyn,
        dwell_time=dwell_time,
        bandwidth=bw,
    )
