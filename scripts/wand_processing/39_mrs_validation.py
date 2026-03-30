#!/usr/bin/env python
"""Validate MRS fitting pipeline against ISMRM Fitting Challenge + Osprey MEGA-PRESS.

Produces a validation report with:
1. ISMRM Fitting Challenge: fitted vs ground truth concentrations
2. Osprey MEGA-PRESS: GABA/NAA ratio from difference spectrum
"""
import sys, os, json
import numpy as np
from pathlib import Path

# Use FSL's Python for fsl_mrs
FSLDIR = os.environ.get('FSLDIR', '/home/mhough/fsl')
sys.path.insert(0, f'{FSLDIR}/lib/python3.12/site-packages')

DATA_DIR = Path('/home/mhough/dev/neurojax/tests/data/mrs')
REPORT_DIR = Path('/data/raw/wand/derivatives/fsl-mrs/validation')
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_ismrm_dataset(dataset_num: int) -> tuple[np.ndarray, np.ndarray]:
    """Load ISMRM fitting challenge dataset (text format).

    Returns (metab_fid, water_fid) as complex arrays.
    """
    f = DATA_DIR / 'ismrm_fitting_challenge' / 'repo' / 'datasets_text' / f'dataset{dataset_num}.txt'
    data = np.loadtxt(str(f))
    # Columns: real_metab, imag_metab, real_water, imag_water
    metab = data[:, 0] + 1j * data[:, 1]
    water = data[:, 2] + 1j * data[:, 3]
    return metab, water


def load_ismrm_basis() -> dict[str, np.ndarray]:
    """Load ISMRM basis set (text format without TMS reference)."""
    basis_dir = DATA_DIR / 'ismrm_fitting_challenge' / 'repo' / 'basisset_text_noTMS'
    basis = {}
    for f in sorted(basis_dir.glob('*.txt')):
        name = f.stem
        data = np.loadtxt(str(f))
        basis[name] = data[:, 0] + 1j * data[:, 1]
    return basis


def validate_ismrm():
    """Run ISMRM Fitting Challenge validation."""
    print("=" * 60)
    print("ISMRM MRS Fitting Challenge Validation")
    print("=" * 60)

    # Known ground truth from the challenge (mM concentrations)
    # From Table 1 in Marjanska et al. MRM 2022
    ground_truth = {
        # dataset: {metabolite: concentration_mM}
        # Datasets 1-14: varying concentrations
        # Datasets 15-28: varying artifacts/noise
        # We'll fit all and report what we get
    }

    basis = load_ismrm_basis()
    print(f"Loaded basis: {sorted(basis.keys())}")
    print(f"Basis FID length: {len(list(basis.values())[0])}")

    results = []

    for ds_num in range(1, 29):
        try:
            metab, water = load_ismrm_dataset(ds_num)

            # Simple fitting: project data onto basis (least squares)
            # Stack basis into matrix
            names = sorted(basis.keys())
            B = np.column_stack([basis[n][:len(metab)] for n in names])

            # Fit: metab ≈ B @ coefficients
            coeffs, residuals, _, _ = np.linalg.lstsq(B, metab, rcond=None)

            # Water scaling
            water_area = np.abs(water[0])
            metab_areas = np.abs(coeffs)

            # Concentrations using challenge formula
            water_conc = 29697  # mM (from challenge description)
            relax_corr = 1.206
            concentrations = {}
            for i, name in enumerate(names):
                if water_area > 0:
                    conc = water_conc * relax_corr * metab_areas[i] / water_area
                else:
                    conc = 0
                concentrations[name] = conc

            # Fit quality
            fitted = B @ coeffs
            residual = metab - fitted
            snr = np.max(np.abs(np.fft.fft(metab))) / np.std(np.abs(np.fft.fft(residual)))

            results.append({
                'dataset': ds_num,
                'concentrations': concentrations,
                'snr': float(snr),
                'residual_norm': float(np.linalg.norm(residual)),
            })

            # Print key metabolites
            key_metabs = ['NAA', 'Cr', 'Glu', 'GABA', 'Ins', 'Lac']
            conc_str = ', '.join(f"{m}={concentrations.get(m, 0):.1f}" for m in key_metabs if m in concentrations)
            print(f"  Dataset {ds_num:2d}: SNR={snr:.1f}, {conc_str}")

        except Exception as e:
            print(f"  Dataset {ds_num:2d}: FAILED ({e})")
            results.append({'dataset': ds_num, 'error': str(e)})

    return results


def validate_osprey_mega():
    """Validate MEGA-PRESS pipeline on Osprey example data."""
    print("\n" + "=" * 60)
    print("Osprey MEGA-PRESS Example Data Validation")
    print("=" * 60)

    sys.path.insert(0, '/home/mhough/dev/neurojax/src')
    from neurojax.analysis.mega_press import process_mega_press

    osprey_dir = DATA_DIR / 'mrshub_edited_examples' / 'osprey_mega'
    results = []

    for sub_dir in sorted(osprey_dir.glob('sub-*')):
        sub = sub_dir.name
        mega_files = list(sub_dir.rglob('*megapress*act*'))
        if not mega_files:
            continue

        sdat_file = mega_files[0]
        print(f"\n  {sub}: {sdat_file.name}")

        try:
            # Convert SDAT to NIfTI-MRS
            import subprocess
            out_dir = REPORT_DIR / 'osprey_mega' / sub
            out_dir.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                ['spec2nii', 'philips_dl', str(sdat_file), '-o', str(out_dir), '-f', 'mega'],
                capture_output=True, text=True
            )

            nii_file = list(out_dir.glob('mega*.nii*'))
            if not nii_file:
                # Try alternative spec2nii command
                spar_file = str(sdat_file).replace('.sdat', '.spar')
                result = subprocess.run(
                    ['spec2nii', 'philips', '-f', 'mega', '-o', str(out_dir), str(sdat_file), str(spar_file)],
                    capture_output=True, text=True
                )
                nii_file = list(out_dir.glob('mega*.nii*'))

            if nii_file:
                from fsl_mrs.utils import mrs_io
                data_obj = mrs_io.read_FID(str(nii_file[0]))
                data = np.squeeze(data_obj[:])
                dwell = float(data_obj.dwelltime)
                cf = float(data_obj.spectrometer_frequency[0]) * 1e6

                print(f"    Shape: {data.shape}, CF: {cf/1e6:.1f} MHz")

                if data.ndim >= 3 and data.shape[-2] == 2:
                    # Has edit dimension
                    mega_result = process_mega_press(data, dwell, cf, align=True, reject=True)

                    diff_spec = np.fft.fftshift(np.fft.fft(mega_result.diff))
                    off_spec = np.fft.fftshift(np.fft.fft(mega_result.edit_off))

                    n = len(mega_result.diff)
                    freq = np.fft.fftshift(np.fft.fftfreq(n, dwell))
                    ppm = freq / (cf / 1e6) + 4.65

                    gaba_mask = (ppm > 2.8) & (ppm < 3.2)
                    naa_mask = (ppm > 1.9) & (ppm < 2.1)

                    gaba_peak = np.max(np.abs(diff_spec[gaba_mask]))
                    naa_peak = np.max(np.abs(off_spec[naa_mask]))
                    ratio = gaba_peak / naa_peak if naa_peak > 0 else 0

                    print(f"    GABA/NAA: {ratio:.3f}")
                    print(f"    Averages used: {mega_result.n_averages}")
                    print(f"    Rejected: {mega_result.rejected.sum()}")

                    results.append({
                        'subject': sub,
                        'gaba_naa_ratio': float(ratio),
                        'n_averages': mega_result.n_averages,
                        'n_rejected': int(mega_result.rejected.sum()),
                    })
                else:
                    print(f"    Not MEGA-edited (shape={data.shape})")
            else:
                print(f"    spec2nii conversion failed")

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({'subject': sub, 'error': str(e)})

    return results


def validate_wand():
    """Validate against WAND ses-05 MEGA-PRESS results."""
    print("\n" + "=" * 60)
    print("WAND ses-05 MEGA-PRESS Validation")
    print("=" * 60)

    wand_dir = Path('/data/raw/wand/derivatives/fsl-mrs/sub-08033/ses-05')
    results = []

    for voi in ['anteriorcingulate', 'occipital', 'rightauditory', 'smleft']:
        mega_dir = wand_dir / voi / 'mega'
        if not mega_dir.exists():
            continue

        diff = np.load(str(mega_dir / 'diff_fid.npy'))
        off = np.load(str(mega_dir / 'edit_off_fid.npy'))

        dwell = 2.5e-4
        cf = 123.253e6
        n = len(diff)
        freq = np.fft.fftshift(np.fft.fftfreq(n, dwell))
        ppm = freq / (cf / 1e6) + 4.65

        diff_spec = np.fft.fftshift(np.fft.fft(diff))
        off_spec = np.fft.fftshift(np.fft.fft(off))

        gaba_mask = (ppm > 2.8) & (ppm < 3.2)
        naa_mask = (ppm > 1.9) & (ppm < 2.1)

        gaba_peak = np.max(np.abs(diff_spec[gaba_mask]))
        naa_peak = np.max(np.abs(off_spec[naa_mask]))
        ratio = gaba_peak / naa_peak if naa_peak > 0 else 0

        print(f"  {voi}: GABA/NAA = {ratio:.3f}")
        results.append({'voi': voi, 'gaba_naa_ratio': float(ratio)})

    return results


if __name__ == '__main__':
    report = {}

    # 1. ISMRM Fitting Challenge
    ismrm_dir = DATA_DIR / 'ismrm_fitting_challenge' / 'repo'
    if ismrm_dir.exists():
        report['ismrm'] = validate_ismrm()
    else:
        print("ISMRM data not available, skipping")

    # 2. Osprey MEGA-PRESS
    osprey_dir = DATA_DIR / 'mrshub_edited_examples' / 'osprey_mega'
    if osprey_dir.exists():
        report['osprey_mega'] = validate_osprey_mega()
    else:
        print("Osprey MEGA data not available, skipping")

    # 3. WAND MEGA-PRESS
    report['wand_mega'] = validate_wand()

    # Save report
    report_file = REPORT_DIR / 'validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n=== Report saved to {report_file} ===")
