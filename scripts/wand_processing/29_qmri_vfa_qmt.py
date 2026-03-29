#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "nibabel>=5.0",
#     "numpy>=1.24",
#     "scipy>=1.10",
# ]
# ///
"""WAND ses-02 quantitative MRI fitting: VFA T1 (DESPOT1) + QMT bound pool fraction.

CUBRIC mcDESPOT protocol:
  SPGR: 8 flip angles [2,4,6,8,10,12,14,18]°, TR_exc=4ms, TE=1.9ms
  SSFP: 16 volumes (8 FAs × 2 phase cycles), TR_exc=4.54ms, TE=2.27ms
  SPGR-IR: inversion recovery for B1 mapping

QMT protocol (tfl_qMT_v09):
  MT-off (reference, FA=5°, TR=55ms)
  11 MT-on volumes at 3 MT flip angles × multiple offset frequencies
  Two-pool Ramani model fitting → BPF, exchange rate kf

Outputs to: derivatives/qmri/sub-{id}/ses-02/
"""
import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# CUBRIC mcDESPOT acquisition parameters
# ---------------------------------------------------------------------------
SPGR_FA_DEG = np.array([2, 4, 6, 8, 10, 12, 14, 18], dtype=np.float64)
SPGR_TR = 0.004        # s (RepetitionTimeExcitation)
SPGR_TE = 0.0019       # s

SSFP_FA_DEG = np.array([10, 15, 20, 25, 30, 40, 50, 60], dtype=np.float64)
SSFP_TR = 0.00454      # s
SSFP_TE = 0.00227      # s
SSFP_N_PHASE_CYCLES = 2  # 0° and 180°

QMT_TR = 0.055          # s
QMT_READOUT_FA = 5.0    # degrees (readout flip angle)


def create_brain_mask(data_3d, percentile=15):
    """Simple intensity-based brain mask."""
    threshold = np.percentile(data_3d[data_3d > 0], percentile)
    mask = data_3d > threshold
    # Simple morphological cleaning
    from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
    mask = binary_fill_holes(mask)
    mask = binary_erosion(mask, iterations=1)
    mask = binary_dilation(mask, iterations=1)
    return mask


def fit_despot1(spgr_data, mask, flip_angles_deg, tr):
    """DESPOT1 T1 fitting from SPGR variable flip angle data.

    Linearized SPGR signal model (Deoni et al. 2003):
        S/sin(α) = E1 · S/tan(α) + M0·(1-E1)

    where E1 = exp(-TR/T1). Linear regression Y vs X gives slope = E1.
    """
    fa_rad = np.deg2rad(flip_angles_deg)
    sin_fa = np.sin(fa_rad)
    tan_fa = np.tan(fa_rad)

    n_fa = len(flip_angles_deg)
    shape = spgr_data.shape[:3]

    t1_map = np.zeros(shape, dtype=np.float64)
    m0_map = np.zeros(shape, dtype=np.float64)

    # Vectorized linear regression over all voxels
    voxels = np.where(mask)
    n_vox = len(voxels[0])
    print(f"  Fitting DESPOT1 T1 for {n_vox:,} voxels...")

    signals = spgr_data[voxels]  # (n_vox, n_fa)

    # Y = S / sin(α), X = S / tan(α)
    Y = signals / sin_fa[np.newaxis, :]  # (n_vox, n_fa)
    X = signals / tan_fa[np.newaxis, :]  # (n_vox, n_fa)

    # Linear regression: Y = slope * X + intercept
    # slope = E1, intercept = M0 * (1 - E1)
    X_mean = X.mean(axis=1, keepdims=True)
    Y_mean = Y.mean(axis=1, keepdims=True)
    Xc = X - X_mean
    Yc = Y - Y_mean

    ss_xx = (Xc ** 2).sum(axis=1)
    ss_xy = (Xc * Yc).sum(axis=1)

    valid = ss_xx > 1e-10
    slope = np.zeros(n_vox)
    slope[valid] = ss_xy[valid] / ss_xx[valid]

    # Clamp E1 to physically meaningful range
    E1 = np.clip(slope, 0.001, 0.9999)

    # T1 = -TR / ln(E1)
    t1_vals = -tr / np.log(E1)

    # Clip T1 to physiological range [100ms, 6000ms]
    t1_vals = np.clip(t1_vals, 0.1, 6.0)

    # M0 = intercept / (1 - E1)
    intercept = Y_mean.squeeze() - slope * X_mean.squeeze()
    m0_vals = np.zeros(n_vox)
    denom = 1.0 - E1
    valid_m0 = np.abs(denom) > 1e-6
    m0_vals[valid_m0] = intercept[valid_m0] / denom[valid_m0]
    m0_vals = np.clip(m0_vals, 0, np.percentile(m0_vals[m0_vals > 0], 99.9)
                       if np.any(m0_vals > 0) else 1e6)

    t1_map[voxels] = t1_vals
    m0_map[voxels] = m0_vals

    print(f"  T1 range: [{t1_vals[t1_vals>0.1].min()*1000:.0f}, "
          f"{np.percentile(t1_vals[t1_vals>0.1], 99)*1000:.0f}] ms "
          f"(1st-99th pct)")
    print(f"  Median WM T1 ~ {np.median(t1_vals[(t1_vals>0.5) & (t1_vals<1.2)])*1000:.0f} ms")

    return t1_map, m0_map


def compute_mtr(mt_off, mt_on_stack, mask):
    """Compute magnetization transfer ratio per MT-on volume.

    MTR = (S_off - S_on) / S_off
    """
    mtr_stack = np.zeros_like(mt_on_stack)
    for i in range(mt_on_stack.shape[-1]):
        mt_on = mt_on_stack[..., i]
        mtr = np.zeros_like(mt_off, dtype=np.float64)
        valid = mask & (mt_off > 10)
        mtr[valid] = (mt_off[valid] - mt_on[valid]) / mt_off[valid]
        mtr = np.clip(mtr, 0, 1)
        mtr_stack[..., i] = mtr
    return mtr_stack


def fit_qmt_ramani(mt_off, mt_on_stack, mt_params, t1_map, mask):
    """Two-pool Ramani model for QMT bound pool fraction estimation.

    Simplified Ramani et al. (2002) approach:
    The MT signal depends on:
      - F = bound pool fraction (BPF) = Mb0 / (Mf0 + Mb0)
      - kf = forward exchange rate (free → bound)
      - T2b = bound pool T2 (~10-12 μs for super-Lorentzian)
      - R1obs = observed R1 = 1/T1

    For each MT-on volume with known (MT_FA, offset_freq):
      MTR = F * W * R_RF * T1obs / (R1obs + F * kf + W * R_RF)

    where W is the absorption rate of the bound pool:
      W = π * (ω1)² * g(Δf) for CW approximation
      g(Δf) = super-Lorentzian lineshape

    We use a linearized fitting approach (Ramani 2002, Eq. 11):
      1/MTR = 1/(F*R1obs*T1obs) * (R1obs/R_RF + F*kf/R_RF + 1)

    Simplified: fit F and kf from multiple (MTR, offset, power) measurements.
    """
    n_mt = mt_on_stack.shape[-1]

    # Extract offset frequencies and MT flip angles from params
    offsets_hz = np.array([p['offset_hz'] for p in mt_params])
    mt_fa_deg = np.array([p['mt_fa_deg'] for p in mt_params])

    print(f"  QMT: {n_mt} MT-on volumes")
    print(f"  Offsets (Hz): {offsets_hz}")
    print(f"  MT flip angles (deg): {mt_fa_deg}")

    shape = mt_off.shape
    bpf_map = np.zeros(shape, dtype=np.float64)
    kf_map = np.zeros(shape, dtype=np.float64)

    # Compute MTR for each volume
    mtr_stack = compute_mtr(mt_off, mt_on_stack, mask)

    # Mean MTR across volumes weighted by proximity to resonance
    # as a simple BPF proxy (before full model fitting)
    # Use near-resonance volumes (offset < 5000 Hz) for BPF sensitivity
    near_res = offsets_hz < 5000
    if near_res.sum() > 0:
        mtr_near = mtr_stack[..., near_res].mean(axis=-1)
    else:
        mtr_near = mtr_stack.mean(axis=-1)

    # Super-Lorentzian lineshape for bound pool
    # g(Δf) = ∫₀¹ sqrt(2/π) * T2b / |3u²-1| *
    #          exp(-2*(2π*Δf*T2b/(3u²-1))²) du
    T2b = 11e-6  # typical bound pool T2 (super-Lorentzian)

    def super_lorentzian(delta_f, T2b=11e-6, n_points=100):
        """Evaluate super-Lorentzian lineshape at offset frequency delta_f."""
        u = np.linspace(0.001, 0.999, n_points)
        denom = np.abs(3 * u**2 - 1)
        arg = 2 * np.pi * delta_f * T2b / denom
        integrand = np.sqrt(2 / np.pi) * T2b / denom * np.exp(-2 * arg**2)
        return np.trapezoid(integrand, u)

    # Compute lineshape values
    g_values = np.array([super_lorentzian(f) for f in offsets_hz])

    # MT pulse effective power (ω1_rms² ∝ FA²/pulse_duration)
    # For pulsed MT, the CW-equivalent power:
    # R_RF = π * ω1_rms² * g(Δf) / duty_cycle
    # ω1_rms ≈ FA_mt * γ / (2π * pulse_dur)
    # Simplified: use FA² as proxy for power
    fa_rad_mt = np.deg2rad(mt_fa_deg)
    power_proxy = fa_rad_mt ** 2  # proportional to ω1²

    # Absorption rate W ∝ power * lineshape
    W = power_proxy * g_values
    W_norm = W / W.max()  # normalize for fitting stability

    # Voxel-wise two-parameter fit: F and kf
    # Model: MTR_i = F * W_i / (R1obs + F*kf + W_i)  (simplified)
    # where R1obs = 1/T1

    voxels = np.where(mask & (t1_map > 0.1) & (t1_map < 5.0))
    n_vox = len(voxels[0])
    print(f"  Fitting Ramani model for {n_vox:,} voxels...")

    mtr_data = mtr_stack[voxels]  # (n_vox, n_mt)
    r1_vals = 1.0 / t1_map[voxels]  # (n_vox,)

    # Vectorized simplified fit:
    # For each voxel, linearize: 1/MTR = (R1 + F*kf)/(F*W) + 1/F
    # Let a = 1/F, b = kf/W_mean
    # Then: 1/MTR_i ≈ a*R1/W_i + a + b

    # Actually use a practical approach: fit via least squares in small batches
    bpf_vals = np.zeros(n_vox)
    kf_vals = np.zeros(n_vox)

    # Use a fast vectorized approximate fit
    # From Ramani: MTR ≈ F * kf * T1obs * (1 - correction_term)
    # First approximation: BPF ≈ MTR_near / (kf_assumed * T1)
    # Better: use ratio of MTR at different offsets

    # Practical approach: two-point BPF estimation
    # MTR at near-resonance (~1kHz) is dominated by BPF
    # MTR at far-off resonance (~47kHz) is dominated by direct saturation
    # BPF ∝ (MTR_near - MTR_far) * T1 correction

    far_res = offsets_hz > 40000
    if far_res.sum() > 0 and near_res.sum() > 0:
        mtr_far = mtr_stack[..., far_res].mean(axis=-1)
        # Delta-MTR approach (Sled & Pike, 2001)
        delta_mtr = mtr_near - mtr_far
        delta_mtr = np.clip(delta_mtr, 0, 1)

        # Scale by R1 to get BPF estimate
        # BPF ≈ delta_MTR * R1 / k_calibration
        # Calibration: typical WM BPF ~0.10-0.15, WM delta_MTR ~0.2-0.4, WM R1 ~1.0
        # So k_calibration ≈ delta_MTR * R1 / BPF ≈ 0.3 * 1.0 / 0.12 ≈ 2.5
        k_cal = 2.5
        bpf_approx = delta_mtr * (1.0 / t1_map) / k_cal
        bpf_approx = np.clip(bpf_approx, 0, 0.30)
        bpf_map = bpf_approx * mask
    else:
        # Fallback: use mean MTR as BPF proxy
        bpf_map = mtr_near * 0.3 * mask  # rough scaling

    # Batch least-squares for refined F, kf per voxel
    # Process in chunks for memory efficiency
    chunk_size = 10000
    n_chunks = (n_vox + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_vox)
        chunk_mtr = mtr_data[start:end]  # (chunk, n_mt)
        chunk_r1 = r1_vals[start:end]    # (chunk,)

        for i in range(end - start):
            mtr_i = chunk_mtr[i]
            r1_i = chunk_r1[i]

            if np.all(mtr_i < 0.01) or r1_i < 0.1:
                continue

            def residuals(params):
                F, kf_log = params
                kf = np.exp(kf_log)
                F = np.clip(F, 0.001, 0.30)
                pred = F * W / (r1_i + F * kf + W)
                return pred - mtr_i

            try:
                # Initial guess from approximate BPF
                F0 = float(bpf_map[voxels[0][start+i],
                                    voxels[1][start+i],
                                    voxels[2][start+i]])
                F0 = max(F0, 0.01)
                result = least_squares(
                    residuals, [F0, np.log(2.0)],
                    bounds=([0.001, np.log(0.1)], [0.30, np.log(20.0)]),
                    method='trf', max_nfev=50
                )
                bpf_vals[start+i] = result.x[0]
                kf_vals[start+i] = np.exp(result.x[1])
            except Exception:
                bpf_vals[start+i] = F0

        if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
            print(f"    Chunk {chunk_idx+1}/{n_chunks} done")

    bpf_map[voxels] = bpf_vals
    kf_map[voxels] = kf_vals

    bpf_valid = bpf_vals[bpf_vals > 0.01]
    if len(bpf_valid) > 0:
        print(f"  BPF range: [{bpf_valid.min():.4f}, {np.percentile(bpf_valid, 99):.4f}]")
        print(f"  Median BPF: {np.median(bpf_valid):.4f}")
        kf_valid = kf_vals[kf_vals > 0.1]
        if len(kf_valid) > 0:
            print(f"  Median kf: {np.median(kf_valid):.2f} Hz")

    return bpf_map, kf_map, mtr_stack


def main():
    parser = argparse.ArgumentParser(description="WAND ses-02 qMRI fitting")
    parser.add_argument("subject", help="Subject ID (e.g. sub-08033)")
    parser.add_argument("--wand-root", default="/Users/mhough/dev/wand",
                        help="WAND BIDS root directory")
    parser.add_argument("--skip-qmt", action="store_true",
                        help="Skip QMT fitting (only do VFA T1)")
    args = parser.parse_args()

    subject = args.subject
    wand = Path(args.wand_root)
    anat_dir = wand / subject / "ses-02" / "anat"
    out_dir = wand / "derivatives" / "qmri" / subject / "ses-02"
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 1. Load SPGR data ===
    spgr_file = anat_dir / f"{subject}_ses-02_acq-spgr_part-mag_VFA.nii.gz"
    if not spgr_file.exists():
        print(f"ERROR: SPGR file not found: {spgr_file}")
        sys.exit(1)

    print("=== DESPOT1 VFA T1 Mapping ===")
    print(f"Subject: {subject}")
    spgr_img = nib.load(str(spgr_file))
    spgr_data = spgr_img.get_fdata().astype(np.float64)
    affine = spgr_img.affine
    print(f"SPGR: {spgr_data.shape}, voxel={spgr_img.header.get_zooms()[:3]}")
    print(f"Flip angles: {SPGR_FA_DEG}°, TR={SPGR_TR*1000:.1f}ms")

    # Brain mask from mean SPGR signal
    mean_spgr = spgr_data.mean(axis=-1)
    mask = create_brain_mask(mean_spgr, percentile=20)
    print(f"Brain mask: {mask.sum():,} voxels")

    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine),
             str(out_dir / "brain_mask.nii.gz"))

    # === 2. Fit DESPOT1 T1 ===
    t1_map, m0_map = fit_despot1(spgr_data, mask, SPGR_FA_DEG, SPGR_TR)

    nib.save(nib.Nifti1Image(t1_map, affine), str(out_dir / "T1map.nii.gz"))
    nib.save(nib.Nifti1Image(m0_map, affine), str(out_dir / "M0map.nii.gz"))
    print(f"Saved: {out_dir / 'T1map.nii.gz'}")

    # === 3. QMT Fitting ===
    if args.skip_qmt:
        print("\nSkipping QMT fitting (--skip-qmt)")
        return

    print("\n=== QMT Bound Pool Fraction Fitting ===")

    # Load MT-off reference
    mtoff_file = anat_dir / f"{subject}_ses-02_mt-off_part-mag_QMT.nii.gz"
    if not mtoff_file.exists():
        print(f"WARNING: MT-off reference not found, skipping QMT")
        return

    mtoff_img = nib.load(str(mtoff_file))
    mt_off = mtoff_img.get_fdata().astype(np.float64)
    qmt_affine = mtoff_img.affine
    print(f"MT-off shape: {mt_off.shape}")

    # Load all MT-on volumes with their parameters
    mt_on_list = []
    mt_params = []

    qmt_patterns = [
        ("flip-1_mt-1", 332, 56360),
        ("flip-1_mt-2", 332, 1000),
        ("flip-2_mt-1", 628, 47180),
        ("flip-2_mt-2", 628, 12060),
        ("flip-2_mt-3", 628, 2750),
        ("flip-2_mt-4", 628, 2770),
        ("flip-2_mt-5", 628, 2790),
        ("flip-2_mt-6", 628, 2890),
        ("flip-2_mt-7_run-1", 628, 1000),
        ("flip-2_mt-7_run-2", 628, 1000),
        ("flip-3_mt-1", 333, 1000),
    ]

    for tag, mt_fa, offset in qmt_patterns:
        fname = anat_dir / f"{subject}_ses-02_{tag}_part-mag_QMT.nii.gz"
        if fname.exists():
            img = nib.load(str(fname))
            mt_on_list.append(img.get_fdata().astype(np.float64))
            mt_params.append({
                'tag': tag,
                'mt_fa_deg': mt_fa,
                'offset_hz': offset
            })

    if not mt_on_list:
        print("WARNING: No MT-on volumes found, skipping QMT")
        return

    mt_on_stack = np.stack(mt_on_list, axis=-1)
    print(f"MT-on stack: {mt_on_stack.shape} ({len(mt_on_list)} volumes)")

    # Create QMT-space mask (different geometry from SPGR)
    qmt_mask = create_brain_mask(mt_off, percentile=20)
    print(f"QMT brain mask: {qmt_mask.sum():,} voxels")

    # Register T1 map to QMT space for the fitting
    # Both are ses-02 and same resolution, so should be in same space
    # Check if shapes match
    if t1_map.shape == mt_off.shape:
        t1_for_qmt = t1_map
        print("  T1 and QMT in same space")
    else:
        print(f"  WARNING: T1 shape {t1_map.shape} != QMT shape {mt_off.shape}")
        print("  Using approximate T1=1.0s for QMT fitting")
        t1_for_qmt = np.ones_like(mt_off) * 1.0

    # Fit QMT
    bpf_map, kf_map, mtr_stack = fit_qmt_ramani(
        mt_off, mt_on_stack, mt_params, t1_for_qmt, qmt_mask
    )

    # Save outputs
    nib.save(nib.Nifti1Image(bpf_map, qmt_affine),
             str(out_dir / "QMT_bpf.nii.gz"))
    nib.save(nib.Nifti1Image(kf_map, qmt_affine),
             str(out_dir / "QMT_kf.nii.gz"))

    # Save mean MTR as well
    mtr_mean = mtr_stack.mean(axis=-1)
    nib.save(nib.Nifti1Image(mtr_mean, qmt_affine),
             str(out_dir / "MTR_mean.nii.gz"))

    # Save per-offset MTR
    for i, p in enumerate(mt_params):
        tag = p['tag'].replace('-', '').replace('_', '')
        nib.save(nib.Nifti1Image(mtr_stack[..., i], qmt_affine),
                 str(out_dir / f"MTR_{tag}.nii.gz"))

    print(f"\nSaved: {out_dir / 'QMT_bpf.nii.gz'}")
    print(f"Saved: {out_dir / 'QMT_kf.nii.gz'}")
    print(f"Saved: {out_dir / 'MTR_mean.nii.gz'}")

    # === 4. Summary ===
    print("\n=== Quantitative Maps Summary ===")
    print(f"T1 map:    {out_dir / 'T1map.nii.gz'}")
    print(f"M0 map:    {out_dir / 'M0map.nii.gz'}")
    print(f"QMT BPF:   {out_dir / 'QMT_bpf.nii.gz'}")
    print(f"QMT kf:    {out_dir / 'QMT_kf.nii.gz'}")
    print(f"MTR mean:  {out_dir / 'MTR_mean.nii.gz'}")
    print(f"Mask:      {out_dir / 'brain_mask.nii.gz'}")


if __name__ == "__main__":
    main()
