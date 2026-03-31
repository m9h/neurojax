"""Integration tests for MRS quantification on real Big GABA and WAND data.

These tests require external data and are skipped when the data is not
available. They validate the full pipeline from raw TWIX / NIfTI-MRS
through to GABA quantification and QC reporting.

Data paths:
    Big GABA: /data/datasets/big_gaba/S5/S5_MP/S01/ ... S12/
    WAND:     /data/raw/wand/derivatives/fsl-mrs/sub-08033/ses-05/
"""

import sys
import numpy as np
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Data availability guards
# ---------------------------------------------------------------------------

BIG_GABA_ROOT = Path('/data/datasets/big_gaba/S5/S5_MP')
BIG_GABA_AVAILABLE = BIG_GABA_ROOT.exists() and (BIG_GABA_ROOT / 'S01').exists()

WAND_SVS_PATH = Path(
    '/data/raw/wand/derivatives/fsl-mrs/sub-08033/ses-05/'
    'anteriorcingulate/niftimrs/svs.nii.gz'
)
WAND_AVAILABLE = WAND_SVS_PATH.exists()


# ---------------------------------------------------------------------------
# Helpers: TWIX loading
# ---------------------------------------------------------------------------

def _load_twix_mega_press(dat_path: str):
    """Load Siemens TWIX MEGA-PRESS .dat file and return
    (data, dwell_time, centre_freq).

    data shape: (n_spec, 2, n_dyn) — already coil-combined.
    """
    sys.path.insert(0, '/home/mhough/fsl/lib/python3.12/site-packages')
    from mapvbvd import mapVBVD

    twix = mapVBVD(dat_path, quiet=True)
    if isinstance(twix, list):
        twix = twix[-1]  # Take the last measurement (scan data)

    # Extract image data
    twix.image.squeeze = True
    twix.image.flagRemoveOS = False
    raw = np.array(twix.image[''])  # shape varies by sequence

    # Get acquisition parameters from header
    hdr = twix.hdr
    dwell_time = float(hdr['MeasYaps'][('sRXSPEC', 'alDwellTime', '0')]) * 1e-9  # ns -> s
    centre_freq = float(hdr['Dicom']['lFrequency'])  # Hz

    # Handle various raw shapes from mapVBVD
    # Big GABA Siemens: (n_spec, n_coils, n_edit, n_dyn) = (4096, 4, 2, 160)
    if raw.ndim == 4:
        # Already (n_spec, n_coils, n_edit, n_dyn) — pass directly to process_mega_press
        data = raw
    elif raw.ndim == 3:
        if raw.shape[1] <= 64:
            # (n_spec, n_coils, n_total) — need to split ON/OFF
            from neurojax.analysis.mega_press import coil_combine_svd
            combined = coil_combine_svd(raw)  # (n_spec, n_total)
            n_dyn = combined.shape[1] // 2
            edit_on = combined[:, 0::2]
            edit_off = combined[:, 1::2]
            data = np.stack([edit_on, edit_off], axis=1)
        else:
            # (n_spec, n_total, ??) — reshape
            data = raw
    elif raw.ndim == 2:
        n_spec, n_total = raw.shape
        n_dyn = n_total // 2
        edit_on = raw[:, 0::2]
        edit_off = raw[:, 1::2]
        data = np.stack([edit_on, edit_off], axis=1)
    else:
        raise ValueError(f"Unexpected TWIX data shape: {raw.shape}")

    return data, dwell_time, centre_freq


def _load_twix_water(dat_path: str):
    """Load water reference TWIX .dat file and return averaged FID."""
    sys.path.insert(0, '/home/mhough/fsl/lib/python3.12/site-packages')
    from mapvbvd import mapVBVD

    twix = mapVBVD(dat_path, quiet=True)
    if isinstance(twix, list):
        twix = twix[-1]

    twix.image.squeeze = True
    twix.image.flagRemoveOS = False
    raw = np.array(twix.image[''])

    if raw.ndim == 3:
        n_coils, n_spec, n_avg = raw.shape
        from neurojax.analysis.mega_press import coil_combine_svd
        raw_reordered = raw.transpose(1, 0, 2)  # (n_spec, n_coils, n_avg)
        combined = coil_combine_svd(raw_reordered)  # (n_spec, n_avg)
        water_fid = combined.mean(axis=1)
    elif raw.ndim == 2:
        water_fid = raw.mean(axis=1)
    else:
        water_fid = raw.ravel()

    return water_fid


# ---------------------------------------------------------------------------
# Test 1: Big GABA single subject
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not BIG_GABA_AVAILABLE,
    reason="Big GABA data not available at /data/datasets/big_gaba/S5/S5_MP/"
)
class TestBigGABASingleSubject:
    """Process one Big GABA S5 subject end-to-end."""

    def test_big_gaba_single_subject(self):
        from neurojax.analysis.mrs_quantify import quantify_mega_press

        subj_dir = BIG_GABA_ROOT / 'S01'
        dat_files = sorted(subj_dir.glob('*GABA_68.dat'))
        assert len(dat_files) > 0, f"No GABA .dat files found in {subj_dir}"

        dat_path = str(dat_files[0])
        data, dwell_time, centre_freq = _load_twix_mega_press(dat_path)

        # Try to load water reference
        water_files = sorted(subj_dir.glob('*GABA_68_H2O.dat'))
        water_ref = None
        if water_files:
            water_ref = _load_twix_water(str(water_files[0]))

        result = quantify_mega_press(
            data, dwell_time=dwell_time, centre_freq=centre_freq,
            water_ref=water_ref,
            tissue_fracs={'gm': 0.6, 'wm': 0.4, 'csf': 0.0},
            te=0.068, tr=2.0,
        )

        # GABA/NAA ratio should be physiologically reasonable
        assert 'gaba_naa_ratio' in result
        gaba_naa = result['gaba_naa_ratio']
        assert 0.01 < gaba_naa < 1.0, (
            f"GABA/NAA ratio {gaba_naa:.4f} outside expected range [0.01, 1.0]"
        )

        # SNR should be reasonable
        assert result['snr'] > 2.0, f"SNR too low: {result['snr']:.1f}"


# ---------------------------------------------------------------------------
# Test 2: Big GABA batch consistency
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not BIG_GABA_AVAILABLE,
    reason="Big GABA data not available at /data/datasets/big_gaba/S5/S5_MP/"
)
class TestBigGABABatchConsistency:
    """Process 3 Big GABA subjects, check inter-subject CV."""

    def test_big_gaba_batch_consistency(self):
        from neurojax.analysis.mrs_quantify import quantify_mega_press

        subjects = ['S01', 'S02', 'S03']
        gaba_naa_ratios = []

        for subj in subjects:
            subj_dir = BIG_GABA_ROOT / subj
            dat_files = sorted(subj_dir.glob('*GABA_68.dat'))
            if not dat_files:
                pytest.skip(f"No GABA .dat files for {subj}")

            data, dwell_time, centre_freq = _load_twix_mega_press(
                str(dat_files[0])
            )

            result = quantify_mega_press(
                data, dwell_time=dwell_time, centre_freq=centre_freq,
            )

            gaba_naa_ratios.append(result['gaba_naa_ratio'])

        gaba_naa_ratios = np.array(gaba_naa_ratios)
        mean_ratio = np.mean(gaba_naa_ratios)
        std_ratio = np.std(gaba_naa_ratios)
        cv = (std_ratio / mean_ratio) * 100.0 if mean_ratio > 0 else 999.0

        assert cv < 50.0, (
            f"Coefficient of variation of GABA/NAA across {len(subjects)} "
            f"subjects is {cv:.1f}%, expected < 50%"
        )
        assert all(r > 0 for r in gaba_naa_ratios), (
            "All GABA/NAA ratios should be positive"
        )


# ---------------------------------------------------------------------------
# Test 3: Big GABA QC report
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not BIG_GABA_AVAILABLE,
    reason="Big GABA data not available at /data/datasets/big_gaba/S5/S5_MP/"
)
class TestBigGABAQCReport:
    """Generate a QC report for one Big GABA subject."""

    def test_big_gaba_qc_report(self, tmp_path):
        from neurojax.analysis.mrs_quantify import quantify_mega_press
        from neurojax.analysis.mrs_qc import generate_qc_report

        subj_dir = BIG_GABA_ROOT / 'S01'
        dat_files = sorted(subj_dir.glob('*GABA_68.dat'))
        assert len(dat_files) > 0

        data, dwell_time, centre_freq = _load_twix_mega_press(
            str(dat_files[0])
        )

        result = quantify_mega_press(
            data, dwell_time=dwell_time, centre_freq=centre_freq,
        )

        # Build the result dict expected by generate_qc_report
        qc_input = {
            'diff': result['diff_fid'],
            'edit_on': result['edit_on_fid'],
            'edit_off': result['edit_off_fid'],
            'sum_spec': result.get('sum_spec_fid', result['diff_fid']),
            'freq_shifts': result['freq_shifts'],
            'phase_shifts': result['phase_shifts'],
            'rejected': result['rejected'],
            'n_averages': result['n_averages'],
            'dwell_time': dwell_time,
            'bandwidth': 1.0 / dwell_time,
            'centre_freq': centre_freq,
        }

        fitting_results = {
            'GABA': {
                'concentration_mM': result.get('gaba_conc_mM', 'N/A'),
                'crlb_percent': result['crlb_percent'],
            },
        }

        html = generate_qc_report(qc_input, fitting_results=fitting_results,
                                   title="Big GABA S01 QC Report")

        # Verify it is valid HTML
        assert html.startswith('<!DOCTYPE html>')
        assert '</html>' in html
        assert 'GABA' in html

        # Write to file to verify it is a complete document
        out_path = tmp_path / 'qc_report.html'
        out_path.write_text(html)
        assert out_path.stat().st_size > 1000, (
            f"QC report is too small ({out_path.stat().st_size} bytes)"
        )


# ---------------------------------------------------------------------------
# Test 4: WAND MEGA-PRESS quantification
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not WAND_AVAILABLE,
    reason=f"WAND data not available at {WAND_SVS_PATH}"
)
class TestWANDMegaQuantification:
    """Process WAND ses-05 ACC NIfTI-MRS data end-to-end."""

    def test_wand_mega_quantification(self):
        from neurojax.analysis.mrs_quantify import quantify_mega_press

        # Load NIfTI-MRS using fsl_mrs
        sys.path.insert(0, '/home/mhough/fsl/lib/python3.12/site-packages')
        from fsl_mrs.utils import mrs_io

        mrs_data = mrs_io.read_FID(str(WAND_SVS_PATH))

        # Extract data array and parameters
        # fsl_mrs stores data as (n_spec, n_coils, n_dyn) or similar
        raw = np.squeeze(mrs_data[:])

        # Get acquisition parameters
        dwell_time = mrs_data.dwelltime
        centre_freq = mrs_data.spectrometer_frequency[0] * 1e6  # MHz -> Hz

        # NIfTI-MRS MEGA: (n_spec, n_coils, n_edit, n_dyn) from squeeze
        data = raw  # process_mega_press handles all shapes

        result = quantify_mega_press(
            data, dwell_time=dwell_time, centre_freq=centre_freq,
        )

        # GABA/NAA ratio should be in a reasonable range
        assert 'gaba_naa_ratio' in result
        gaba_naa = result['gaba_naa_ratio']
        assert gaba_naa > 0, f"GABA/NAA ratio should be positive, got {gaba_naa}"
        assert result['snr'] > 1.0, f"SNR too low: {result['snr']:.1f}"
