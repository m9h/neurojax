#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = ["nibabel>=5.0", "numpy>=1.24"]
# ///
"""Convert WAND MRS TWIX to NIfTI-MRS, bypassing spec2nii PatientName bug.

Reads TWIX via pymapVBVD, constructs NIfTI-MRS manually with proper
headers. Handles both sLASER (ses-04) and MEGA-PRESS (ses-05).
"""
import sys
sys.path.insert(0, '/Users/mhough/fsl/lib/python3.12/site-packages')

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from mapvbvd import mapVBVD


def twix_to_niftimrs(dat_file, out_path, is_mega=False):
    """Convert TWIX .dat to NIfTI-MRS .nii.gz."""
    twix = mapVBVD(dat_file, quiet=True)
    if isinstance(twix, list):
        twix = twix[-1]

    img = twix['image']
    img.squeeze = True
    img.flagRemoveOS = True
    data = np.array(img[''])
    hdr = twix['hdr']

    # Extract parameters
    dwell_ns = hdr['MeasYaps'][('sRXSPEC', 'alDwellTime', '0')]
    dwell = dwell_ns * 1e-9
    te = hdr['MeasYaps'][('alTE', '0')] * 1e-6  # us → s
    tr = hdr['MeasYaps'][('alTR', '0')] * 1e-6
    cf = hdr['MeasYaps'][('sTXSPEC', 'asNucleusInfo', '0', 'lFrequency')]

    # Coil combine with phase alignment
    if data.ndim == 4 and is_mega:
        n_spec, n_chan, n_edit, n_avg = data.shape
        weights = np.abs(data[0, :, 0, :]).mean(axis=1)
        weights /= weights.max() + 1e-10
        phase_ref = np.angle(data[0, :, 0, 0])
        phase_corr = np.exp(-1j * phase_ref)

        combined = np.zeros((n_spec, n_edit, n_avg), dtype=complex)
        for e in range(n_edit):
            combined[:, e, :] = np.sum(
                data[:, :, e, :] * weights[np.newaxis, :, np.newaxis] * phase_corr[np.newaxis, :, np.newaxis],
                axis=1
            )
        # Average each edit condition
        edit_on = combined[:, 0, :].mean(axis=1)
        edit_off = combined[:, 1, :].mean(axis=1)
        diff = edit_on - edit_off
        fid_avg = edit_off  # Use OFF for standard fitting
        fid_diff = diff     # Use DIFF for GABA
        extra = {'edit_on': edit_on, 'edit_off': edit_off, 'diff': diff}
    elif data.ndim == 3:
        n_spec, n_chan, n_avg = data.shape
        weights = np.abs(data[0, :, :]).mean(axis=1)
        weights /= weights.max() + 1e-10
        phase_ref = np.angle(data[0, :, 0])
        phase_corr = np.exp(-1j * phase_ref)
        combined = np.sum(data * weights[np.newaxis, :, np.newaxis] * phase_corr[np.newaxis, :, np.newaxis], axis=1)
        fid_avg = combined.mean(axis=1)
        extra = {}
    elif data.ndim == 2:
        n_spec, n_chan = data.shape
        n_avg = 1
        weights = np.abs(data[0, :])
        weights /= weights.max() + 1e-10
        phase_ref = np.angle(data[0, :])
        phase_corr = np.exp(-1j * phase_ref)
        fid_avg = np.sum(data * weights[np.newaxis, :] * phase_corr[np.newaxis, :], axis=1)
        extra = {}
    else:
        # Squeeze all extra dims, average everything except spectral
        d = data.squeeze()
        if d.ndim == 1:
            fid_avg = d
        else:
            # Average all non-spectral dims
            fid_avg = d.reshape(d.shape[0], -1).mean(axis=1)
        extra = {}

    # Create NIfTI-MRS: shape (1, 1, 1, n_spectral) as complex pairs
    # Store as float32 real/imag interleaved
    n_spec = len(fid_avg)
    nifti_data = np.zeros((1, 1, 1, n_spec, 2), dtype=np.float32)
    nifti_data[0, 0, 0, :, 0] = np.real(fid_avg).astype(np.float32)
    nifti_data[0, 0, 0, :, 1] = np.imag(fid_avg).astype(np.float32)

    affine = np.eye(4)
    # Set pixdim[4] = dwell time for spectral dimension
    img_nii = nib.Nifti1Image(nifti_data, affine)
    img_nii.header['pixdim'][4] = dwell
    nib.save(img_nii, str(out_path))

    # Also save as simple complex numpy (for direct fsl_mrs if NIfTI-MRS doesn't work)
    np.save(str(out_path).replace('.nii.gz', '_fid.npy'), fid_avg)

    # Save metadata
    meta = {
        'dwell_time_s': float(dwell),
        'te_s': float(te),
        'tr_s': float(tr),
        'centre_frequency_hz': float(cf),
        'n_spectral': int(n_spec),
        'n_channels': int(n_chan) if 'n_chan' in dir() else 1,
        'n_averages': int(n_avg) if 'n_avg' in dir() else 1,
    }
    with open(str(out_path).replace('.nii.gz', '_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Save extras for MEGA
    for name, arr in extra.items():
        np.save(str(out_path).replace('.nii.gz', f'_{name}.npy'), arr)

    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default='sub-08033')
    parser.add_argument('--wand-root', default='/Users/mhough/dev/wand')
    args = parser.parse_args()

    wand = Path(args.wand_root)
    sub = args.subject
    out_base = wand / 'derivatives' / 'fsl-mrs' / sub

    vois = ['anteriorcingulate', 'occipital', 'rightauditory', 'smleft']

    for ses, acq, is_mega in [('ses-04', 'slaser', False), ('ses-05', 'mega', True)]:
        print(f'\n=== {ses} ({acq}) ===')

        for voi in vois:
            out_dir = out_base / ses / voi
            out_dir.mkdir(parents=True, exist_ok=True)

            # Metabolite data
            svs_dat = wand / sub / ses / 'mrs' / f'{sub}_{ses}_acq-{acq}_voi-{voi}_svs.dat'
            if svs_dat.exists():
                print(f'  {voi} SVS...', end=' ')
                meta = twix_to_niftimrs(str(svs_dat), out_dir / 'svs.nii.gz', is_mega=is_mega)
                print(f'TE={meta["te_s"]*1000:.0f}ms, TR={meta["tr_s"]*1000:.0f}ms')

            # Water reference
            ref_dat = wand / sub / ses / 'mrs' / f'{sub}_{ses}_acq-{acq}_voi-{voi}_ref.dat'
            if ref_dat.exists():
                print(f'  {voi} REF...', end=' ')
                meta_ref = twix_to_niftimrs(str(ref_dat), out_dir / 'ref.nii.gz', is_mega=False)
                print(f'done')

    print('\nAll conversions complete')


if __name__ == '__main__':
    main()
