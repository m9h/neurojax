#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "nibabel>=5.0",
#     "numpy>=1.24",
#     "scipy>=1.10",
# ]
# ///
"""WAND ses-03 perfusion quantification: CBF, OEF, CMRO₂.

Complete cerebral oxygen metabolism pipeline:
  1. Inversion Recovery (Look-Locker) → blood T1 calibration
  2. pCASL + M0 + blood T1 → CBF map (via FSL oxford_asl/BASIL)
  3. TRUST → venous T2 → SvO₂ → OEF (Lu et al. 2008)
  4. CBF × OEF × CaO₂ → CMRO₂ (Fick's principle)

Data:
  pCASL:  64×64×22×110 vols, bolus=1.8s, PLD=2.0s, TR=4.6s
  TRUST:  64×64×1×24 vols, TI=1.02s, 4 eTEs × 3 repeats × 2 (ctrl/label)
  InvRec: 128×128×1×960, Look-Locker for blood T1
  M0:     64×64×22×3 (AP), equilibrium magnetization
  Angio:  384×512×60 (phase-contrast vascular anatomy)

References:
  Alsop DC et al. (2015) ASL White Paper, MRM
  Lu H et al. (2008) TRUST MRI, MRM
  Germuska M et al. (2019) Dual-calibrated fMRI, NeuroImage
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.optimize import curve_fit


def fit_blood_t1(ir_file, perf_out):
    """Fit blood T1 from inversion recovery Look-Locker data.

    WAND InvRec: 128×128×1×960, TR=150ms, Look-Locker continuous sampling.
    FrameTimesStart from 0 to 145.63s — multi-inversion-time T1 recovery.
    """
    print("\n=== Step 1: Blood T1 from Inversion Recovery ===")

    img = nib.load(str(ir_file))
    data = img.get_fdata().astype(np.float64)
    print(f"InvRec data: {data.shape}")

    # Get TI values from JSON sidecar
    meta_file = str(ir_file).replace('.nii.gz', '.json')
    with open(meta_file) as f:
        meta = json.load(f)

    frame_times = np.array(meta.get('FrameTimesStart', []))
    n_TIs = len(frame_times)

    if n_TIs != data.shape[-1]:
        print(f"  WARNING: FrameTimesStart ({n_TIs}) != data volumes ({data.shape[-1]})")
        # Fallback: generate TIs from TR
        TR = meta.get('RepetitionTime', 0.15)
        frame_times = np.arange(data.shape[-1]) * TR

    print(f"TIs: {frame_times[0]:.3f} to {frame_times[-1]:.3f}s ({n_TIs} points)")

    # Squeeze to 2D + time
    if data.ndim == 4:
        data = data[:, :, 0, :]  # (128, 128, 960)

    # ROI: center of sagittal sinus (high signal, blood compartment)
    # Use the peak signal region in the late TIs (when blood is fully recovered)
    late_signal = np.mean(data[:, :, -50:], axis=2)
    threshold = np.percentile(late_signal[late_signal > 0], 90)
    roi_mask = late_signal > threshold
    n_roi = roi_mask.sum()
    print(f"Blood ROI: {n_roi} voxels (>90th percentile late signal)")

    # Extract mean ROI signal
    roi_signal = np.mean(data[roi_mask, :], axis=0)

    # Look-Locker T1 fitting
    # Signal model: S(t) = S0 * |1 - (1-cos(α)) * exp(-t/T1*)|
    # where T1* = 1 / (1/T1 - ln(cos(α))/TR) is the apparent T1
    # For small flip angles: T1 ≈ T1* * (S0/S_inf - 1)

    # Simple approach: fit S(t) = A * (1 - B * exp(-t/T1star))
    def ll_model(t, A, B, T1star):
        return A * (1 - B * np.exp(-t / T1star))

    # Initial guess
    A0 = roi_signal[-1]
    B0 = 1.5
    T1star0 = 1.2

    try:
        popt, pcov = curve_fit(
            ll_model, frame_times, roi_signal,
            p0=[A0, B0, T1star0],
            bounds=([0, 0, 0.1], [roi_signal.max() * 3, 3.0, 5.0]),
            maxfev=10000
        )
        A, B, T1star = popt
        T1star_std = np.sqrt(pcov[2, 2])

        # Correct T1* to true T1
        # T1 = T1* * B  (Deichmann & Haase 1992 correction)
        T1_blood = T1star * B
        print(f"T1* (apparent): {T1star*1000:.0f} ms")
        print(f"B factor: {B:.3f}")
        print(f"Blood T1: {T1_blood*1000:.0f} ms")
        print(f"Expected at 3T: 1600-1700 ms")

        if T1_blood < 1.0 or T1_blood > 3.0:
            print(f"  WARNING: T1_blood={T1_blood*1000:.0f}ms outside expected range")
            print(f"  Using default T1_blood = 1650 ms")
            T1_blood = 1.65

    except Exception as e:
        print(f"Look-Locker fitting failed: {e}")
        T1_blood = 1.65
        print(f"Using default T1_blood = {T1_blood*1000:.0f} ms")

    # Save
    np.save(str(perf_out / "T1_blood.npy"), T1_blood)
    with open(perf_out / "T1_blood.txt", "w") as f:
        f.write(f"{T1_blood:.4f}")
    print(f"Saved: {perf_out / 'T1_blood.txt'}")

    return T1_blood


def run_oxford_asl(pcasl_file, m0_file, anat_dir, perf_out, t1_blood,
                    fmap_dir=None, subject=None, wand_root=None):
    """Run FSL oxford_asl for CBF quantification.

    pCASL: 64×64×22×110 volumes, bolus=1.8s, PLD=2.0s, TR=4.6s
    Interleaved control-label pairs → 55 pairs.
    """
    print("\n=== Step 2: pCASL → CBF (oxford_asl) ===")

    cbf_out = perf_out / "oxford_asl"

    # Check if already completed
    cbf_native = cbf_out / "native_space" / "perfusion_calib.nii.gz"
    if cbf_native.exists():
        print(f"  oxford_asl already complete: {cbf_native}")
        return cbf_out

    # Prepare fieldmap args
    fmap_args = []
    if fmap_dir and subject and wand_root:
        fmap_phase = Path(wand_root) / subject / "ses-03" / "fmap" / f"{subject}_ses-03_phasediff.nii.gz"
        fmap_mag = Path(wand_root) / subject / "ses-03" / "fmap" / f"{subject}_ses-03_magnitude1.nii.gz"
        if fmap_phase.exists() and fmap_mag.exists():
            # Need to prepare fieldmap first
            mag_brain = fmap_dir / "mag_brain"
            fieldmap = fmap_dir / "fieldmap_rads"

            if not Path(f"{mag_brain}.nii.gz").exists():
                print("  Preparing fieldmap...")
                subprocess.run(["bet", str(fmap_mag), str(mag_brain), "-f", "0.5"],
                               capture_output=True)

                # Get delta TE
                meta1 = json.load(open(str(fmap_mag).replace('.nii.gz', '.json')))
                meta2_path = str(fmap_mag).replace('magnitude1', 'magnitude2').replace('.nii.gz', '.json')
                if Path(meta2_path).exists():
                    meta2 = json.load(open(meta2_path))
                    delta_te = (meta2.get('EchoTime', 0.00738) - meta1.get('EchoTime', 0.00492)) * 1000
                else:
                    delta_te = 2.46  # default

                subprocess.run([
                    "fsl_prepare_fieldmap", "SIEMENS",
                    str(fmap_phase), str(mag_brain),
                    str(fieldmap), str(delta_te)
                ], capture_output=True)

            if Path(f"{fieldmap}.nii.gz").exists():
                fmap_args = [
                    f"--fmap={fieldmap}",
                    f"--fmapmag={fmap_mag}",
                    f"--fmapmagbrain={mag_brain}",
                    "--echospacing=0.000265",
                    "--pedir=y-",
                ]

    # Build oxford_asl command
    cmd = [
        "oxford_asl",
        f"-i", str(pcasl_file),
        f"-o", str(cbf_out),
        "--casl",
        "--bolus=1.8",
        "--pld=2.0",
        "--iaf=tc",               # tag-control interleaved
        f"-c", str(m0_file),      # calibration (M0) image
        "--cmethod=voxel",        # voxel-wise calibration
        f"--t1b={t1_blood}",      # subject-specific blood T1
        "--mc",                   # motion correction
        "--pvcorr",               # partial volume correction
    ]

    # Add structural if available
    anat_t1 = anat_dir / "T1_biascorr.nii.gz" if anat_dir else None
    if anat_t1 and anat_t1.exists():
        cmd.extend([
            f"-s", str(anat_dir / "T1_biascorr"),
            f"--fslanat={anat_dir}",
        ])

    cmd.extend(fmap_args)

    print(f"  Command: {' '.join(cmd[:8])}...")
    print(f"  pCASL: {pcasl_file}")
    print(f"  M0: {m0_file}")
    print(f"  T1_blood: {t1_blood:.3f}s")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"  oxford_asl FAILED:")
        print(f"  {result.stderr[-500:]}")
        return None

    print(f"  oxford_asl complete: {cbf_out}")

    # Report CBF values
    if cbf_native.exists():
        cbf_img = nib.load(str(cbf_native))
        cbf_data = cbf_img.get_fdata()
        brain = cbf_data[cbf_data > 0]
        if len(brain) > 0:
            print(f"  CBF range: [{brain.min():.1f}, {np.percentile(brain, 99):.1f}] ml/100g/min")
            print(f"  Median CBF: {np.median(brain):.1f} ml/100g/min")
            print(f"  Expected GM: ~50-80, WM: ~20-30 ml/100g/min")

    return cbf_out


def fit_trust(trust_file, perf_out):
    """Fit TRUST data for venous T2 → SvO₂ → OEF.

    TRUST: 64×64×1×24 volumes, TI=1.02s (venous blood labeling inversion)
    Protocol: 4 effective TEs × 3 repeats × 2 (control/label) = 24 volumes

    The effective TE (eTE) is the T2 preparation time.
    Standard TRUST eTEs at 3T: 0, 40, 80, 160 ms
    """
    print("\n=== Step 3: TRUST → SvO₂ → OEF ===")

    img = nib.load(str(trust_file))
    data = img.get_fdata().astype(np.float64)
    print(f"TRUST data: {data.shape}")

    meta_file = str(trust_file).replace('.nii.gz', '.json')
    with open(meta_file) as f:
        meta = json.load(f)

    # Squeeze to 2D + volumes
    if data.ndim == 4:
        data = data[:, :, 0, :]  # (64, 64, 24)

    n_vols = data.shape[-1]
    print(f"Volumes: {n_vols}")

    # TRUST protocol: interleaved control-label pairs at different eTEs
    # 24 = 4 eTEs × 3 repeats × 2 (control/label)
    # Ordering: [ctrl_eTE1_rep1, label_eTE1_rep1, ctrl_eTE1_rep2, ...]
    n_pairs = n_vols // 2
    n_eTEs = 4
    n_repeats = n_pairs // n_eTEs

    # Standard CUBRIC TRUST eTEs (ms → s)
    eTEs = np.array([0, 40, 80, 160]) * 1e-3

    print(f"eTEs: {eTEs * 1000} ms")
    print(f"Repeats per eTE: {n_repeats}")

    # Separate control and label
    control = data[:, :, 0::2]   # even volumes
    label = data[:, :, 1::2]     # odd volumes
    diff = np.abs(control - label)  # (64, 64, 12)

    # Average repeats per eTE
    diff_avg = np.zeros((*diff.shape[:2], n_eTEs))
    for i in range(n_eTEs):
        indices = list(range(i, n_pairs, n_eTEs))
        diff_avg[:, :, i] = np.mean(diff[:, :, indices], axis=2)

    # Identify sagittal sinus ROI (maximum difference signal at eTE=0)
    eTE0_signal = diff_avg[:, :, 0]
    threshold = np.percentile(eTE0_signal[eTE0_signal > 0], 92)
    sinus_roi = eTE0_signal > threshold
    n_roi = sinus_roi.sum()
    print(f"Sagittal sinus ROI: {n_roi} voxels")

    if n_roi < 3:
        print("  WARNING: Insufficient sinus voxels. Using top 20 voxels.")
        flat = eTE0_signal.flatten()
        top_idx = np.argsort(flat)[-20:]
        sinus_roi = np.zeros_like(eTE0_signal, dtype=bool)
        sinus_roi.flat[top_idx] = True
        n_roi = sinus_roi.sum()

    # Extract ROI signal at each eTE
    roi_signal = np.array([np.mean(diff_avg[sinus_roi, i]) for i in range(n_eTEs)])
    print(f"ROI signal by eTE: {roi_signal.round(2)}")

    # Fit mono-exponential T2 decay: S(eTE) = S0 * exp(-eTE / T2)
    def t2_model(eTE, S0, T2):
        return S0 * np.exp(-eTE / T2)

    try:
        popt, pcov = curve_fit(
            t2_model, eTEs[:len(roi_signal)], roi_signal,
            p0=[roi_signal[0], 0.060],
            bounds=([0, 0.005], [roi_signal[0] * 3, 0.300]),
            maxfev=10000
        )
        T2_blood = popt[1]
        T2_std = np.sqrt(pcov[1, 1])
        print(f"Venous blood T2: {T2_blood*1000:.1f} ± {T2_std*1000:.1f} ms")

    except Exception as e:
        print(f"T2 fitting failed: {e}")
        T2_blood = 0.055  # fallback
        print(f"Using default T2_blood = {T2_blood*1000:.1f} ms")

    # T2 → SvO₂ calibration (Lu & Ge 2008, 3T bovine blood)
    # 1/T2 = A + B*(1-Y)² + C*(1-Y)⁴
    # or simplified: 1/T2 = A + B*(1-Y) + C*(1-Y)²
    # Coefficients for Hct=0.42 at 3T (Lu et al. 2012):
    A = 4.50    # 1/s, T2 of fully oxygenated blood
    B = 47.1    # 1/s
    C = 55.5    # 1/s
    # These are approximate — exact values depend on Hct and field strength

    R2 = 1.0 / T2_blood

    # Solve: C*(1-Y)² + B*(1-Y) + (A - R2) = 0
    a_coeff = C
    b_coeff = B
    c_coeff = A - R2

    discriminant = b_coeff**2 - 4 * a_coeff * c_coeff

    results = {
        "T2_blood_s": float(T2_blood),
        "T2_blood_ms": float(T2_blood * 1000),
        "R2_blood": float(R2),
        "eTEs_ms": [float(x * 1000) for x in eTEs],
        "roi_signal": [float(x) for x in roi_signal],
        "n_roi_voxels": int(n_roi),
    }

    if discriminant >= 0:
        one_minus_Y = (-b_coeff + np.sqrt(discriminant)) / (2 * a_coeff)
        SvO2 = 1.0 - one_minus_Y
        SaO2 = 0.98  # assumed arterial saturation
        OEF = (SaO2 - SvO2) / SaO2

        print(f"\nSvO₂: {SvO2*100:.1f}%")
        print(f"OEF:  {OEF*100:.1f}%")
        print(f"Expected: SvO₂ ~60-68%, OEF ~32-40%")

        if OEF < 0.15 or OEF > 0.60:
            print(f"  WARNING: OEF={OEF*100:.1f}% outside typical range")

        results.update({
            "SvO2": float(SvO2),
            "OEF": float(OEF),
            "SaO2": float(SaO2),
            "Hct_assumed": 0.42,
        })
    else:
        print("T2→SvO₂ calibration failed (negative discriminant)")
        print("  This can happen if T2 is too long (fully oxygenated) or too short")
        # Use literature default
        OEF = 0.35
        SvO2 = 0.98 * (1 - OEF)
        results.update({
            "SvO2": float(SvO2),
            "OEF": float(OEF),
            "SaO2": 0.98,
            "note": "default OEF used (calibration failed)",
        })

    # Save
    with open(perf_out / "trust_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {perf_out / 'trust_results.json'}")

    return results


def compute_cmro2(cbf_out, trust_results, perf_out):
    """Compute CMRO₂ from CBF and OEF.

    CMRO₂ = CBF × OEF × CaO₂
    where CaO₂ = Hb × 1.34 × SaO₂ ≈ 8.97 µmol O₂/ml blood
    (for Hb=15 g/dl, SaO₂=0.98)

    Units: CBF in ml/100g/min, CaO₂ in µmol O₂/ml
    → CMRO₂ in µmol O₂/100g/min
    """
    print("\n=== Step 4: CMRO₂ = CBF × OEF × CaO₂ ===")

    # Load CBF map
    cbf_file = cbf_out / "native_space" / "perfusion_calib.nii.gz"
    if not cbf_file.exists():
        # Try standard space
        cbf_file = cbf_out / "native_space" / "perfusion.nii.gz"
    if not cbf_file.exists():
        print(f"  CBF map not found at {cbf_out}/native_space/")
        return None

    cbf_img = nib.load(str(cbf_file))
    cbf = cbf_img.get_fdata().astype(np.float64)

    OEF = trust_results.get("OEF", 0.35)
    SaO2 = trust_results.get("SaO2", 0.98)

    # CaO₂ = Hb × 1.34 × SaO₂ (ml O₂ / ml blood)
    # Typical Hb = 15 g/dl = 0.15 g/ml
    # 1.34 ml O₂ per gram Hb at full saturation (Hüfner's constant)
    Hb = 15.0  # g/dl
    CaO2_ml = Hb * 1.34 * SaO2 / 100  # ml O₂ / ml blood = 0.197
    # Convert to µmol: 1 mol O₂ = 22400 ml, so 1 µmol = 0.0224 ml
    CaO2_umol = CaO2_ml / 0.0224  # µmol O₂ / ml blood

    print(f"OEF: {OEF*100:.1f}%")
    print(f"CaO₂: {CaO2_ml:.3f} ml O₂/ml blood = {CaO2_umol:.1f} µmol/ml")

    # CMRO₂ (µmol O₂ / 100g / min) = CBF (ml/100g/min) × OEF × CaO₂ (µmol/ml)
    cmro2 = cbf * OEF * CaO2_umol

    # Clip to physiological range
    cmro2 = np.clip(cmro2, 0, 500)

    brain = cmro2[cbf > 5]  # mask by reasonable CBF
    if len(brain) > 0:
        print(f"CMRO₂ range: [{brain.min():.1f}, {np.percentile(brain, 99):.1f}] µmol/100g/min")
        print(f"Median CMRO₂: {np.median(brain):.1f} µmol/100g/min")
        print(f"Expected GM: ~130-200, WM: ~60-80 µmol/100g/min")

    # Save
    nib.save(nib.Nifti1Image(cmro2, cbf_img.affine, cbf_img.header),
             str(perf_out / "CMRO2_map.nii.gz"))
    print(f"Saved: {perf_out / 'CMRO2_map.nii.gz'}")

    # Also save CBF × OEF (= oxygen delivery used) in ml/100g/min for comparison
    cbf_oef = cbf * OEF
    nib.save(nib.Nifti1Image(cbf_oef, cbf_img.affine),
             str(perf_out / "CBF_times_OEF.nii.gz"))

    # Summary JSON
    summary = {
        "OEF": float(OEF),
        "CaO2_ml_per_ml": float(CaO2_ml),
        "CaO2_umol_per_ml": float(CaO2_umol),
        "Hb_g_per_dl": float(Hb),
        "CMRO2_median": float(np.median(brain)) if len(brain) > 0 else None,
        "CMRO2_mean_GM": None,  # requires tissue mask
        "CMRO2_mean_WM": None,
        "CBF_median": float(np.median(cbf[cbf > 5])) if np.any(cbf > 5) else None,
    }

    with open(perf_out / "perfusion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return cmro2


def main():
    parser = argparse.ArgumentParser(
        description="WAND ses-03 perfusion quantification"
    )
    parser.add_argument("subject", help="Subject ID")
    parser.add_argument("--wand-root", default="/Users/mhough/dev/wand")
    parser.add_argument("--skip-oxford-asl", action="store_true",
                        help="Skip oxford_asl (if already run)")
    args = parser.parse_args()

    subject = args.subject
    wand = Path(args.wand_root)
    perf_dir = wand / subject / "ses-03" / "perf"
    anat_dir = (wand / "derivatives" / "fsl-anat" / subject /
                "ses-03" / "anat" / f"{subject}_ses-03_T1w.anat")
    perf_out = wand / "derivatives" / "perfusion" / subject
    perf_out.mkdir(parents=True, exist_ok=True)
    fmap_dir = perf_out / "fieldmap"
    fmap_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"WAND Perfusion Quantification: {subject}")
    print("=" * 60)

    # Step 1: Blood T1
    ir_file = perf_dir / f"{subject}_ses-03_acq-InvRec_cbf.nii.gz"
    if ir_file.exists():
        t1_blood = fit_blood_t1(ir_file, perf_out)
    else:
        print("InvRec data not found, using default T1_blood = 1650 ms")
        t1_blood = 1.65

    # Step 2: oxford_asl → CBF
    pcasl_file = perf_dir / f"{subject}_ses-03_acq-PCASL_cbf.nii.gz"
    m0_file = perf_dir / f"{subject}_ses-03_dir-AP_m0scan.nii.gz"

    cbf_out = None
    if not args.skip_oxford_asl and pcasl_file.exists() and m0_file.exists():
        cbf_out = run_oxford_asl(
            pcasl_file, m0_file, anat_dir, perf_out, t1_blood,
            fmap_dir=fmap_dir, subject=subject, wand_root=str(wand)
        )
    elif args.skip_oxford_asl:
        cbf_out = perf_out / "oxford_asl"
        if not (cbf_out / "native_space").exists():
            print("  WARNING: --skip-oxford-asl but no output found")
            cbf_out = None

    # Step 3: TRUST → OEF
    trust_file = perf_dir / f"{subject}_ses-03_acq-TRUST_cbf.nii.gz"
    trust_results = None
    if trust_file.exists():
        trust_results = fit_trust(trust_file, perf_out)

    # Step 4: CMRO₂
    if cbf_out and trust_results:
        compute_cmro2(cbf_out, trust_results, perf_out)
    else:
        if not cbf_out:
            print("\n  Cannot compute CMRO₂: CBF not available")
        if not trust_results:
            print("\n  Cannot compute CMRO₂: TRUST results not available")

    # Summary
    print("\n" + "=" * 60)
    print(f"Outputs: {perf_out}/")
    print(f"  T1_blood:        T1_blood.txt")
    print(f"  CBF:             oxford_asl/native_space/perfusion_calib.nii.gz")
    print(f"  TRUST:           trust_results.json")
    print(f"  CMRO₂:           CMRO2_map.nii.gz")
    print(f"  Summary:         perfusion_summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
