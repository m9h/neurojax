"""Native Siemens TWIX reader -- no spec2nii dependency.

Reads Siemens TWIX .dat files using the mapVBVD Python package and returns
a standardized MRSData dataclass with FID data and scan metadata.

The mapVBVD package is the Python port of the widely-used MATLAB mapVBVD
by Philipp Ehses (https://github.com/pehses/mapVBVD).

References:
    Mikkelsen et al. (2017) Big GABA: Edited MR spectroscopy at 24
    research sites. NeuroImage 159:32-45.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# MRSData container
# ---------------------------------------------------------------------------

@dataclass
class MRSData:
    """Standardized container for MRS data loaded from any vendor format.

    Attributes
    ----------
    data : np.ndarray
        Complex FID array. Shape is (n_spec, n_coils, n_dyn) for standard
        acquisitions, or (n_spec, n_coils, n_edit, n_dyn) for edited
        sequences (MEGA-PRESS, HERMES).
    dwell_time : float
        Dwell time in seconds.
    centre_freq : float
        Spectrometer centre frequency in Hz.
    te : float
        Echo time in milliseconds.
    tr : float
        Repetition time in milliseconds.
    field_strength : float
        Static magnetic field strength in Tesla.
    n_coils : int
        Number of receive coils.
    n_averages : int
        Number of averages (dynamic repetitions).
    dim_info : dict
        Mapping of dimension names to axis indices, e.g.
        {'spec': 0, 'coil': 1, 'dyn': 2}.
    water_ref : np.ndarray or None
        Water reference FID, if available.
    """
    data: np.ndarray
    dwell_time: float
    centre_freq: float
    te: float = 0.0
    tr: float = 0.0
    field_strength: float = 0.0
    n_coils: int = 1
    n_averages: int = 1
    dim_info: dict = field(default_factory=dict)
    water_ref: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# mapVBVD import helper
# ---------------------------------------------------------------------------

def _import_mapvbvd():
    """Import mapVBVD, adding FSL site-packages if needed."""
    fsl_site = '/home/mhough/fsl/lib/python3.12/site-packages'
    if fsl_site not in sys.path:
        sys.path.insert(0, fsl_site)
    from mapvbvd import mapVBVD
    return mapVBVD


# ---------------------------------------------------------------------------
# TWIX reader
# ---------------------------------------------------------------------------

def read_twix(
    filepath: str | Path,
    load_water_ref: bool = False,
) -> MRSData:
    """Read a Siemens TWIX .dat file and return an MRSData object.

    Parameters
    ----------
    filepath : str or Path
        Path to the .dat file.
    load_water_ref : bool
        If True, attempt to load a water reference scan from the file.

    Returns
    -------
    MRSData
        Standardized MRS data container.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"TWIX file not found: {filepath}")

    mapVBVD = _import_mapvbvd()

    # mapVBVD returns a list for multi-raid files; take the last entry
    # (the actual measurement, not noise/adjustment scans).
    twix = mapVBVD(str(filepath))
    if isinstance(twix, list):
        twix = twix[-1]

    # ------------------------------------------------------------------
    # Extract FID data from the 'image' MDH
    # ------------------------------------------------------------------
    img = twix['image']

    # Some mapVBVD versions require explicit flags before reading
    try:
        img.flagRemoveOS = False
        img.flagDoAverage = False
    except (AttributeError, TypeError):
        pass

    # Read the raw data -- returns ndarray with vendor-specific ordering.
    # mapVBVD Python typically returns shape:
    #   (n_col, n_cha, n_lin, n_par, n_slc, n_ave, n_phs, n_eco, n_rep, ...)
    # We need to squeeze and reorder to our standard layout.
    raw = np.array(img[''])  # empty-string index triggers full read
    raw = np.squeeze(raw)

    # ------------------------------------------------------------------
    # Parse header metadata
    # ------------------------------------------------------------------
    hdr = twix['hdr']

    # Dwell time
    dwell_time = _extract_dwell_time(hdr, raw)

    # Centre frequency
    centre_freq = _extract_centre_freq(hdr)

    # TE, TR (in ms)
    te = _extract_te(hdr)
    tr = _extract_tr(hdr)

    # Field strength (Tesla)
    field_strength = _extract_field_strength(hdr, centre_freq)

    # ------------------------------------------------------------------
    # Reshape to standard layout
    # ------------------------------------------------------------------
    fid, n_coils, n_averages, dim_info = _reshape_fid(raw, hdr)

    # ------------------------------------------------------------------
    # Water reference
    # ------------------------------------------------------------------
    water_ref = None
    if load_water_ref:
        water_ref = _load_water_ref(twix)

    return MRSData(
        data=fid,
        dwell_time=dwell_time,
        centre_freq=centre_freq,
        te=te,
        tr=tr,
        field_strength=field_strength,
        n_coils=n_coils,
        n_averages=n_averages,
        dim_info=dim_info,
        water_ref=water_ref,
    )


# ---------------------------------------------------------------------------
# Header extraction helpers
# ---------------------------------------------------------------------------

def _get_nested(d, *keys, default=None):
    """Safely traverse nested dicts/objects."""
    obj = d
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k, None)
        else:
            obj = getattr(obj, k, None)
        if obj is None:
            return default
    return obj


def _try_meas_yaps(hdr, *keys, default=None):
    """Look up a value in MeasYaps using tuple-key format from mapVBVD."""
    yaps = hdr.get('MeasYaps', None)
    if yaps is None:
        return default
    val = yaps.get(keys, None)
    if val is None:
        return default
    if isinstance(val, str):
        # mapVBVD sometimes returns space-separated array strings
        val = val.strip().split()[0]
    return val


def _extract_dwell_time(hdr, raw):
    """Extract dwell time in seconds from TWIX header."""
    # Try MeasYaps tuple-key path (VB/VD/VE)
    dwell_ns = _try_meas_yaps(hdr, 'sRXSPEC', 'alDwellTime', '0')
    if dwell_ns is not None:
        return float(dwell_ns) * 1e-9  # nanoseconds -> seconds

    # Try Meas dict path
    meas = hdr.get('Meas', {})
    if isinstance(meas, dict):
        dwell_ns = meas.get('alDwellTime', None)
        if dwell_ns is not None:
            if isinstance(dwell_ns, str):
                dwell_ns = dwell_ns.strip().split()[0]
            return float(dwell_ns) * 1e-9

    # Fallback: typical spectroscopy dwell time
    return 2.5e-4


def _extract_centre_freq(hdr):
    """Extract spectrometer centre frequency in Hz."""
    # MeasYaps tuple-key path
    freq = _try_meas_yaps(hdr, 'sTXSPEC', 'asNucleusInfo', '0', 'lFrequency')
    if freq is not None:
        return float(freq)

    # Flat search in MeasYaps
    myaps = hdr.get('MeasYaps', {})
    if isinstance(myaps, dict):
        for k, v in myaps.items():
            if 'lFrequency' in str(k):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass

    # Fallback: 3T proton
    return 123.25e6


def _extract_te(hdr):
    """Extract echo time in milliseconds."""
    # MeasYaps tuple-key path
    te = _try_meas_yaps(hdr, 'alTE', '0')
    if te is not None:
        te_val = float(te)
        if te_val > 1000:
            return te_val / 1000.0  # us -> ms
        return te_val

    return 0.0


def _extract_tr(hdr):
    """Extract repetition time in milliseconds."""
    tr = _try_meas_yaps(hdr, 'alTR', '0')
    if tr is not None:
        tr_val = float(tr)
        if tr_val > 100000:
            return tr_val / 1000.0  # us -> ms
        return tr_val

    return 0.0


def _extract_field_strength(hdr, centre_freq):
    """Extract static field strength in Tesla."""
    b0 = _try_meas_yaps(hdr, 'sProtConsistencyInfo', 'flNominalB0')
    if b0 is not None:
        return float(b0)

    # Derive from centre frequency (gyromagnetic ratio of 1H = 42.576 MHz/T)
    GAMMA_H1_MHZ = 42.576
    b0 = (centre_freq / 1e6) / GAMMA_H1_MHZ
    return float(round(b0, 2))


# ---------------------------------------------------------------------------
# FID reshaping
# ---------------------------------------------------------------------------

def _reshape_fid(raw, hdr):
    """Reshape squeezed mapVBVD output to standard MRS layout.

    Standard layout:
        3D: (n_spec, n_coils, n_dyn)
        4D: (n_spec, n_coils, n_edit, n_dyn)

    mapVBVD Python typically returns the data with the column (spectral)
    dimension first and channel dimension second.  The remaining dimensions
    depend on the sequence (averages, edit conditions, repetitions, etc.).

    Returns
    -------
    fid : ndarray
    n_coils : int
    n_averages : int
    dim_info : dict
    """
    if raw.ndim < 2:
        # Single-coil, single-average edge case
        fid = raw[:, np.newaxis, np.newaxis]
        return fid, 1, 1, {'spec': 0, 'coil': 1, 'dyn': 2}

    n_spec = raw.shape[0]

    # Determine n_coils -- second dimension for multi-coil data
    # For single-coil data the second dim may already be dynamics.
    # Heuristic: coil count is typically <= 64; dynamics can be large.
    n_coils_candidate = raw.shape[1]

    if raw.ndim == 2:
        # (n_spec, n_coils) -- single average, single edit
        fid = raw[:, :, np.newaxis]
        return fid, n_coils_candidate, 1, {'spec': 0, 'coil': 1, 'dyn': 2}

    if raw.ndim == 3:
        # (n_spec, n_coils, n_dyn)
        n_dyn = raw.shape[2]
        return raw, n_coils_candidate, n_dyn, {'spec': 0, 'coil': 1, 'dyn': 2}

    if raw.ndim == 4:
        # Could be (n_spec, n_coils, n_edit, n_dyn)
        # or (n_spec, n_coils, n_ave, n_rep)
        n_dim2 = raw.shape[2]
        n_dim3 = raw.shape[3]

        if n_dim2 in (2, 4):
            # Likely edit dimension (MEGA=2, HERMES=4)
            return raw, n_coils_candidate, n_dim3, {
                'spec': 0, 'coil': 1, 'edit': 2, 'dyn': 3,
            }
        else:
            # Treat as (n_spec, n_coils, n_ave, n_rep) -> merge last two
            merged = raw.reshape(n_spec, n_coils_candidate, -1)
            return merged, n_coils_candidate, merged.shape[2], {
                'spec': 0, 'coil': 1, 'dyn': 2,
            }

    # ndim >= 5: flatten extra dims into dynamics
    fid = raw.reshape(n_spec, n_coils_candidate, -1)
    return fid, n_coils_candidate, fid.shape[2], {'spec': 0, 'coil': 1, 'dyn': 2}


def _load_water_ref(twix):
    """Try to load water reference scan from TWIX object."""
    # Water reference can be in 'noise', 'phasecor', or a dedicated
    # scan in the multi-raid file.  In mapVBVD it often appears as
    # twix['refscan'] or twix['image'] with a different MDH flag.
    for key in ['refscan', 'ref', 'phasecor', 'noise']:
        ref = twix.get(key, None) if isinstance(twix, dict) else getattr(twix, key, None)
        if ref is not None:
            try:
                ref_data = np.array(ref[''])
                ref_data = np.squeeze(ref_data)
                if ref_data.size > 0:
                    return ref_data
            except (KeyError, TypeError, IndexError):
                continue
    return None
