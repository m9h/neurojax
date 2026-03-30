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


def find_sss_roi(target_img, signal_2d, anat_dir=None, perf_out=None,
                 label="sss"):
    """Find superior sagittal sinus ROI in a single-slice perfusion image.

    Three-tier strategy:
      Tier 1: Structural-guided — project superior midline brain surface
              from FSL-anat T1w onto the target slice via affine composition
      Tier 2: Signal + spatial constraints — midline + posterior + top-N
      Tier 3: Return None (caller uses literature defaults)

    Args:
        target_img: nibabel image of the target single-slice acquisition
        signal_2d: 2D array (same spatial dims as target) with signal metric
                   (e.g. eTE0 difference for TRUST, late-TI signal for InvRec)
        anat_dir: Path to FSL-anat directory (contains T1_biascorr_brain*)
        perf_out: Path to save diagnostic ROI mask
        label: prefix for saved files

    Returns:
        roi_mask: 2D boolean array, or None if all tiers fail
        method: str describing which tier succeeded
    """
    target_affine = target_img.affine
    shape_2d = signal_2d.shape

    # --- Tier 1: Structural-guided ---
    if anat_dir is not None:
        brain_path = None
        mask_path = None
        # Handle both Path and string, and the nested .anat directory
        anat_dir = Path(anat_dir)
        for candidate in [anat_dir, anat_dir.parent]:
            bp = candidate / "T1_biascorr_brain.nii.gz"
            mp = candidate / "T1_biascorr_brain_mask.nii.gz"
            if bp.exists() and mp.exists():
                brain_path, mask_path = bp, mp
                break

        if brain_path is not None:
            print(f"  Tier 1: structural-guided ROI from {brain_path.parent.name}")
            struct_img = nib.load(str(brain_path))
            struct_mask = nib.load(str(mask_path)).get_fdata() > 0
            struct_affine = struct_img.affine

            # Mapping: struct voxel → scanner coord → target voxel
            struct_to_target = np.linalg.inv(target_affine) @ struct_affine

            # Find midline superior brain surface voxels in structural space
            # Compute scanner x-coordinate for each structural i-index
            struct_shape = struct_mask.shape
            i_indices = np.arange(struct_shape[0])
            # Scanner x for each i (at j=0, k=0): x = affine[0,0]*i + affine[0,3]
            scanner_x = struct_affine[0, 0] * i_indices + struct_affine[0, 3]

            candidates = []
            for x_tol in [8.0, 12.0]:  # Try tight first, widen if needed
                candidates = []
                midline_i = np.where(np.abs(scanner_x) < x_tol)[0]

                for i in midline_i:
                    for j in range(struct_shape[1]):
                        # Find topmost brain voxel in this column
                        col = struct_mask[i, j, :]
                        if not col.any():
                            continue
                        k_top = np.max(np.where(col)[0])
                        # Take top 2 voxels (SSS sits on brain surface)
                        for k in range(k_top, max(k_top - 2, -1), -1):
                            if struct_mask[i, j, k]:
                                candidates.append([i, j, k])

                if not candidates:
                    continue

                candidates = np.array(candidates)

                # Project to target voxel space
                ones = np.ones((len(candidates), 1))
                struct_coords_h = np.hstack([candidates, ones])
                target_coords = (struct_to_target @ struct_coords_h.T).T[:, :3]

                # Keep only candidates that fall within the target slice
                # Target is single-slice: k should be near 0
                slice_k = target_coords[:, 2]
                in_slice = np.abs(slice_k - 0.0) < 1.0  # within 1 voxel of slice

                # Also within image bounds
                in_bounds = (
                    (target_coords[:, 0] >= 0) &
                    (target_coords[:, 0] < shape_2d[0]) &
                    (target_coords[:, 1] >= 0) &
                    (target_coords[:, 1] < shape_2d[1])
                )
                valid = in_slice & in_bounds
                if valid.sum() < 3:
                    continue

                # Round to nearest target voxel
                ti = np.round(target_coords[valid, 0]).astype(int)
                tj = np.round(target_coords[valid, 1]).astype(int)

                # Get unique voxels
                unique_voxels = set(zip(ti, tj))
                ti = np.array([v[0] for v in unique_voxels])
                tj = np.array([v[1] for v in unique_voxels])

                # Intersect with signal: keep above median of candidates
                sig_vals = signal_2d[ti, tj]
                sig_thresh = np.median(sig_vals)
                keep = sig_vals > sig_thresh
                ti, tj = ti[keep], tj[keep]

                if len(ti) >= 3:
                    roi_mask = np.zeros(shape_2d, dtype=bool)
                    roi_mask[ti, tj] = True
                    print(f"    {roi_mask.sum()} voxels (|x|<{x_tol}mm, "
                          f"signal>{sig_thresh:.1f})")
                    if perf_out:
                        _save_roi_mask(roi_mask, target_img, perf_out, label)
                    return roi_mask, f"tier1_structural_xtol{x_tol}"

            print("    Tier 1 failed (no candidates in target slice)")

    # --- Tier 2: Signal + spatial constraints ---
    print("  Tier 2: signal + spatial constraints")
    # Compute scanner coordinates for each target voxel
    ii, jj = np.meshgrid(np.arange(shape_2d[0]), np.arange(shape_2d[1]),
                         indexing='ij')
    # Scanner x for each voxel (k=0 for single slice)
    scanner_x_map = (target_affine[0, 0] * ii + target_affine[0, 1] * jj +
                     target_affine[0, 3])
    scanner_y_map = (target_affine[1, 0] * ii + target_affine[1, 1] * jj +
                     target_affine[1, 3])

    # Midline constraint: |x| < 7mm
    midline_mask = np.abs(scanner_x_map) < 7.0

    # Posterior constraint: y < 25th percentile of brain y-coordinates
    brain_voxels = signal_2d > np.percentile(signal_2d[signal_2d > 0], 10)
    if brain_voxels.any():
        y_thresh = np.percentile(scanner_y_map[brain_voxels], 25)
        posterior_mask = scanner_y_map < y_thresh
    else:
        posterior_mask = np.ones(shape_2d, dtype=bool)

    spatial_mask = midline_mask & posterior_mask & (signal_2d > 0)
    n_spatial = spatial_mask.sum()
    print(f"    Spatial candidates: {n_spatial} (midline & posterior)")

    if n_spatial >= 3:
        # Take top N by signal
        for n_top in [5, 10, 15]:
            sig_in_mask = signal_2d[spatial_mask]
            if len(sig_in_mask) < n_top:
                n_top = len(sig_in_mask)
            thresh = np.sort(sig_in_mask)[-n_top]
            roi_mask = spatial_mask & (signal_2d >= thresh)
            if roi_mask.sum() >= 3:
                print(f"    {roi_mask.sum()} voxels (top-{n_top} by signal)")
                if perf_out:
                    _save_roi_mask(roi_mask, target_img, perf_out, label)
                return roi_mask, f"tier2_spatial_top{n_top}"

    # --- Tier 3: failure ---
    print("  Tier 3: ROI selection failed, using literature defaults")
    return None, "tier3_default"


def _save_roi_mask(roi_mask, target_img, perf_out, label):
    """Save ROI mask as NIfTI for QC."""
    perf_out = Path(perf_out)
    # Construct a 3D volume (add slice dimension back)
    mask_3d = roi_mask[:, :, np.newaxis].astype(np.uint8)
    nib.save(nib.Nifti1Image(mask_3d, target_img.affine),
             str(perf_out / f"{label}_roi_mask.nii.gz"))


def fit_blood_t1(ir_file, perf_out, anat_dir=None):
    """Fit blood T1 from inversion recovery data.

    WAND InvRec: 128x128x1x960 — 16 discrete IR blocks of 60 readouts each.
    Block interval ~150ms, inter-block gap ~265ms.
    Fit block-averaged IR recovery: S(TI) = |M0 * (1 - C * exp(-TI/T1))|.
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
        TR = meta.get('RepetitionTime', 0.15)
        frame_times = np.arange(data.shape[-1]) * TR

    # Squeeze to 2D + time
    if data.ndim == 4:
        data = data[:, :, 0, :]  # (128, 128, 960)

    # --- Detect multi-block structure ---
    diffs = np.diff(frame_times)
    median_dt = np.median(diffs)
    gap_indices = np.where(diffs > 1.5 * median_dt)[0]

    if len(gap_indices) > 0:
        block_starts = np.concatenate([[0], gap_indices + 1])
        block_ends = np.concatenate([gap_indices + 1, [len(frame_times)]])
        n_blocks = len(block_starts)
        fpb = block_ends[0] - block_starts[0]  # frames per block
        print(f"Detected {n_blocks} IR blocks, {fpb} frames each, "
              f"dt={median_dt*1000:.0f}ms, gap={diffs[gap_indices[0]]*1000:.0f}ms")
    else:
        # Single continuous acquisition — treat as 1 block
        n_blocks = 1
        fpb = len(frame_times)
        block_starts = np.array([0])
        block_ends = np.array([fpb])
        print(f"Single IR block, {fpb} frames")

    # --- ROI selection ---
    late_signal = np.mean(data[:, :, -50:], axis=2)
    roi_mask, roi_method = find_sss_roi(
        img, late_signal, anat_dir=anat_dir, perf_out=perf_out,
        label="invrec_sss"
    )

    if roi_mask is None:
        print("Using default T1_blood = 1650 ms (ROI selection failed)")
        T1_blood = 1.65
        _save_t1_blood(T1_blood, perf_out)
        return T1_blood

    n_roi = roi_mask.sum()
    print(f"Blood ROI: {n_roi} voxels ({roi_method})")

    # --- Block-averaged inversion recovery ---
    # Compute per-block relative TI times and average signal across blocks
    block_tis = frame_times[block_starts[0]:block_ends[0]] - frame_times[block_starts[0]]
    roi_blocks = np.zeros((n_blocks, fpb))

    for b in range(n_blocks):
        s, e = block_starts[b], block_ends[b]
        if e - s != fpb:
            continue  # skip partial blocks
        roi_blocks[b] = np.mean(data[roi_mask, s:e], axis=0)

    # Average across blocks for a clean recovery curve
    roi_mean = np.mean(roi_blocks, axis=0)

    # --- Fit IR model: S(TI) = |M0 * (1 - C * exp(-TI/T1))| ---
    def ir_model(ti, M0, C, T1):
        return np.abs(M0 * (1 - C * np.exp(-ti / T1)))

    M0_guess = roi_mean[-1]
    try:
        popt, pcov = curve_fit(
            ir_model, block_tis, roi_mean,
            p0=[M0_guess, 2.0, 1.65],
            bounds=([0, 1.0, 0.5], [M0_guess * 5, 2.5, 3.0]),
            maxfev=20000
        )
        M0, C, T1_blood = popt
        T1_std = np.sqrt(pcov[2, 2]) if pcov[2, 2] > 0 else 0

        print(f"IR fit: M0={M0:.1f}, C={C:.3f}, T1={T1_blood*1000:.0f} +/- {T1_std*1000:.0f} ms")
        print(f"Expected at 3T: 1600-1700 ms")

        # Cross-validate with null-point method
        # Null point: S = 0 → TI_null = T1 * ln(C)
        null_idx = np.argmin(roi_mean[:fpb // 2])  # null in first half
        TI_null = block_tis[null_idx]
        if TI_null > 0 and C > 1.0:
            T1_null = TI_null / np.log(C)
            print(f"Null-point T1: {T1_null*1000:.0f} ms (TI_null={TI_null*1000:.0f}ms)")
            discrepancy = abs(T1_null - T1_blood) / T1_blood
            if discrepancy > 0.15:
                print(f"  WARNING: null-point vs fit discrepancy {discrepancy*100:.0f}%")

        # Per-block T1 for uncertainty estimate
        block_t1s = []
        for b in range(n_blocks):
            try:
                bp, _ = curve_fit(
                    ir_model, block_tis, roi_blocks[b],
                    p0=[M0, C, T1_blood],
                    bounds=([0, 1.0, 0.5], [M0 * 5, 2.5, 3.0]),
                    maxfev=5000
                )
                block_t1s.append(bp[2])
            except Exception:
                pass
        if block_t1s:
            block_t1s = np.array(block_t1s)
            iqr = np.percentile(block_t1s, 75) - np.percentile(block_t1s, 25)
            print(f"Per-block T1: median={np.median(block_t1s)*1000:.0f}ms, "
                  f"IQR={iqr*1000:.0f}ms (n={len(block_t1s)})")

        if T1_blood < 1.0 or T1_blood > 2.5:
            print(f"  WARNING: T1_blood={T1_blood*1000:.0f}ms outside expected range")
            print(f"  Using default T1_blood = 1650 ms")
            T1_blood = 1.65

    except Exception as e:
        print(f"IR fitting failed: {e}")
        T1_blood = 1.65
        print(f"Using default T1_blood = {T1_blood*1000:.0f} ms")

    _save_t1_blood(T1_blood, perf_out)
    return T1_blood


def _save_t1_blood(T1_blood, perf_out):
    """Save blood T1 value."""
    np.save(str(perf_out / "T1_blood.npy"), T1_blood)
    with open(perf_out / "T1_blood.txt", "w") as f:
        f.write(f"{T1_blood:.4f}")
    print(f"Saved: {perf_out / 'T1_blood.txt'}")


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


def fit_trust(trust_file, perf_out, anat_dir=None):
    """Fit TRUST data for venous T2 → SvO₂ → OEF.

    TRUST: 64x64x1x24 volumes, TI=1.02s (venous blood labeling inversion)
    Protocol: 4 effective TEs x 3 repeats x 2 (control/label) = 24 volumes
    Standard CUBRIC TRUST eTEs at 3T: 0, 40, 80, 160 ms

    Uses structural-guided SSS ROI and per-voxel T2 quality control.
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
    n_pairs = n_vols // 2
    n_eTEs = 4
    n_repeats = n_pairs // n_eTEs

    # Standard CUBRIC TRUST eTEs (ms → s)
    eTEs = np.array([0, 40, 80, 160]) * 1e-3
    print(f"eTEs: {eTEs * 1000} ms, repeats: {n_repeats}")

    # Separate control and label, compute difference
    control = data[:, :, 0::2]
    label = data[:, :, 1::2]
    diff = np.abs(control - label)

    # Average repeats per eTE
    diff_avg = np.zeros((*diff.shape[:2], n_eTEs))
    for i in range(n_eTEs):
        indices = list(range(i, n_pairs, n_eTEs))
        diff_avg[:, :, i] = np.mean(diff[:, :, indices], axis=2)

    # --- ROI selection ---
    eTE0_signal = diff_avg[:, :, 0]
    sinus_roi, roi_method = find_sss_roi(
        img, eTE0_signal, anat_dir=anat_dir, perf_out=perf_out,
        label="trust_sss"
    )

    if sinus_roi is None:
        print("  ROI selection failed — using default OEF=0.35")
        results = _trust_defaults(eTEs)
        with open(perf_out / "trust_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return results

    n_roi = sinus_roi.sum()
    print(f"Sagittal sinus ROI: {n_roi} voxels ({roi_method})")

    # --- Per-voxel T2 fitting with quality control ---
    def t2_model(eTE, S0, T2):
        return S0 * np.exp(-eTE / T2)

    voxel_coords = np.where(sinus_roi)
    voxel_t2s = []
    voxel_s0s = []
    voxel_status = []

    for idx in range(n_roi):
        i, j = voxel_coords[0][idx], voxel_coords[1][idx]
        sig = diff_avg[i, j, :]

        if sig[0] < 5:
            voxel_status.append("rejected_low_S0")
            continue

        try:
            popt, pcov = curve_fit(
                t2_model, eTEs, sig,
                p0=[sig[0], 0.050],
                bounds=([0, 0.005], [sig[0] * 5, 0.300]),
                maxfev=5000
            )
            t2_v, s0_v = popt[1], popt[0]

            # Compute R-squared
            predicted = t2_model(eTEs, *popt)
            ss_res = np.sum((sig - predicted) ** 2)
            ss_tot = np.sum((sig - sig.mean()) ** 2)
            r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if t2_v > 0.100:
                voxel_status.append(f"rejected_T2_high_{t2_v*1000:.0f}ms")
            elif t2_v < 0.015:
                voxel_status.append(f"rejected_T2_low_{t2_v*1000:.0f}ms")
            elif r_sq < 0.95:
                voxel_status.append(f"rejected_Rsq_{r_sq:.3f}")
            else:
                voxel_t2s.append(t2_v)
                voxel_s0s.append(s0_v)
                voxel_status.append(f"accepted_T2={t2_v*1000:.1f}ms")
        except Exception:
            voxel_status.append("rejected_fit_failed")

    n_accepted = len(voxel_t2s)
    print(f"  Voxel-wise T2: {n_accepted}/{n_roi} accepted")
    for s in voxel_status:
        print(f"    {s}")

    if n_accepted >= 1:
        # Weighted median T2 (weighted by S0 — more blood = higher weight)
        voxel_t2s = np.array(voxel_t2s)
        voxel_s0s = np.array(voxel_s0s)
        sort_idx = np.argsort(voxel_t2s)
        cumw = np.cumsum(voxel_s0s[sort_idx])
        median_idx = np.searchsorted(cumw, cumw[-1] / 2)
        T2_blood = voxel_t2s[sort_idx[median_idx]]
        print(f"  Weighted-median T2: {T2_blood*1000:.1f} ms")
        if n_accepted > 1:
            print(f"  T2 range: [{voxel_t2s.min()*1000:.1f}, {voxel_t2s.max()*1000:.1f}] ms")
    else:
        # Fallback: ROI-mean fit
        print("  All voxels rejected — falling back to ROI-mean fit")
        roi_signal = np.array([np.mean(diff_avg[sinus_roi, i]) for i in range(n_eTEs)])
        try:
            popt, _ = curve_fit(
                t2_model, eTEs, roi_signal,
                p0=[roi_signal[0], 0.050],
                bounds=([0, 0.005], [roi_signal[0] * 3, 0.300]),
                maxfev=10000
            )
            T2_blood = popt[1]
            print(f"  ROI-mean T2: {T2_blood*1000:.1f} ms")
        except Exception as e:
            print(f"  ROI-mean fit also failed: {e}")
            T2_blood = 0.055
            print(f"  Using default T2_blood = {T2_blood*1000:.1f} ms")

    # --- T2 → SvO₂ calibration (Lu et al. 2012, Hct=0.42, 3T) ---
    # 1/T2 = A + B*(1-Y) + C*(1-Y)²
    A, B, C_cal = 4.50, 47.1, 55.5  # s⁻¹

    R2 = 1.0 / T2_blood
    discriminant = B**2 - 4 * C_cal * (A - R2)

    # Build ROI-level signal for output
    roi_signal = np.array([np.mean(diff_avg[sinus_roi, i]) for i in range(n_eTEs)])

    results = {
        "T2_blood_s": float(T2_blood),
        "T2_blood_ms": float(T2_blood * 1000),
        "R2_blood": float(R2),
        "eTEs_ms": [float(x * 1000) for x in eTEs],
        "roi_signal": [float(x) for x in roi_signal],
        "n_roi_voxels": int(n_roi),
        "n_accepted_voxels": n_accepted,
        "roi_method": roi_method,
        "voxel_t2_ms": [float(t * 1000) for t in voxel_t2s] if n_accepted > 0 else [],
        "voxel_status": voxel_status,
    }

    if discriminant >= 0:
        one_minus_Y = (-B + np.sqrt(discriminant)) / (2 * C_cal)
        SvO2 = 1.0 - one_minus_Y
        SaO2 = 0.98
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
        results.update(_trust_defaults(eTEs))

    with open(perf_out / "trust_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {perf_out / 'trust_results.json'}")

    return results


def _trust_defaults(eTEs):
    """Return default TRUST results when fitting fails."""
    OEF = 0.35
    SvO2 = 0.98 * (1 - OEF)
    return {
        "SvO2": float(SvO2),
        "OEF": float(OEF),
        "SaO2": 0.98,
        "Hct_assumed": 0.42,
        "note": "default OEF used (ROI or fitting failed)",
    }


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
                "ses-03" / f"{subject}_ses-03_T1w.anat")
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
        t1_blood = fit_blood_t1(ir_file, perf_out, anat_dir=anat_dir)
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
        trust_results = fit_trust(trust_file, perf_out, anat_dir=anat_dir)

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
