#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = ["nibabel>=5.0", "numpy>=1.24", "jax>=0.4", "equinox>=0.11", "optax>=0.2"]
# ///
"""Validate B1 correction approaches on WAND sub-08033 ses-02 VFA data.

Compares three T1 mapping strategies:
  1. Linearized DESPOT1 (no B1 correction) — baseline
  2. DESPOT1 NLLS via JAX (no B1 correction) — gradient-based
  3. DESPOT1-HIFI (joint T1 + B1 from SPGR + IR-SPGR) — B1-corrected

The SPGR-IR acquisition provides the inversion-recovery reference needed
for DESPOT1-HIFI to disentangle T1 from B1+ inhomogeneity.

Usage:
  python examples/validate_b1_wand.py [--n-voxels 1000] [--wand-root /data/raw/wand]
"""
import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np

# JAX (CPU fallback is fine for validation)
import jax
import jax.numpy as jnp

from neurojax.qmri.steady_state import spgr_signal_multi, ir_spgr_signal
from neurojax.qmri.despot import despot1_fit_voxel, despot1hifi_fit
from neurojax.qmri.b1 import correct_t1_for_b1


def main():
    parser = argparse.ArgumentParser(description="B1 validation on WAND VFA")
    parser.add_argument("--wand-root", default="/data/raw/wand")
    parser.add_argument("--subject", default="sub-08033")
    parser.add_argument("--n-voxels", type=int, default=500,
                        help="Number of random brain voxels to fit (CPU speed)")
    parser.add_argument("--ti", type=float, default=0.45,
                        help="Assumed IR inversion time (s) — CUBRIC default ~450ms")
    args = parser.parse_args()

    wand = Path(args.wand_root)
    subject = args.subject
    anat = wand / subject / "ses-02" / "anat"

    # === Load data ===
    print("=== Loading WAND VFA data ===")
    spgr_file = anat / f"{subject}_ses-02_acq-spgr_part-mag_VFA.nii.gz"
    ir_file = anat / f"{subject}_ses-02_acq-spgrIR_part-mag_VFA.nii.gz"

    spgr_img = nib.load(str(spgr_file))
    ir_img = nib.load(str(ir_file))

    spgr_data = spgr_img.get_fdata().astype(np.float64)
    ir_data = ir_img.get_fdata().astype(np.float64)

    print(f"SPGR: {spgr_data.shape} (8 flip angles)")
    print(f"IR-SPGR: {ir_data.shape} (single TI={args.ti*1000:.0f}ms)")

    # CUBRIC mcDESPOT parameters
    fa_deg = np.array([2, 4, 6, 8, 10, 12, 14, 18], dtype=np.float64)
    fa_rad = jnp.deg2rad(jnp.array(fa_deg))
    TR = 0.004  # 4ms
    TI = args.ti  # assumed — not in JSON sidecar (custom sequence)

    # IR flip angle (readout FA for IR-SPGR, from JSON: FlipAngle=5)
    ir_fa_deg = 5.0
    ir_fa_rad = jnp.deg2rad(ir_fa_deg)
    TR_ir = TR  # same excitation TR

    # Brain mask from mean SPGR
    mean_spgr = spgr_data.mean(axis=-1)
    threshold = np.percentile(mean_spgr[mean_spgr > 0], 20)
    mask = mean_spgr > threshold

    # Handle orientation mismatch (SPGR is PSR, IR is RAS)
    from nibabel.orientations import io_orientation, ornt_transform, apply_orientation
    spgr_ornt = io_orientation(spgr_img.affine)
    ir_ornt = io_orientation(ir_img.affine)

    if spgr_data.shape[:3] != ir_data.shape:
        transform = ornt_transform(ir_ornt, spgr_ornt)
        ir_reoriented = apply_orientation(ir_data, transform)
        print(f"Reoriented IR {ir_data.shape} → {ir_reoriented.shape}")
        ir_data = ir_reoriented

    if spgr_data.shape[:3] != ir_data.shape:
        print(f"WARNING: shape mismatch SPGR {spgr_data.shape[:3]} vs IR {ir_data.shape}")
        print("Skipping HIFI — shapes don't match")
        ir_data = None

    # Sample random brain voxels
    brain_idx = np.where(mask)
    n_brain = len(brain_idx[0])
    n_sample = min(args.n_voxels, n_brain)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_brain, n_sample, replace=False)

    si = brain_idx[0][sample_idx]
    sj = brain_idx[1][sample_idx]
    sk = brain_idx[2][sample_idx]

    print(f"\nFitting {n_sample} voxels from {n_brain:,} brain voxels (CPU)")

    # === 1. Linearized DESPOT1 (numpy, baseline) ===
    print("\n--- 1. Linearized DESPOT1 (no B1) ---")
    t0 = time.time()
    t1_linear = np.zeros(n_sample)
    sin_fa = np.sin(np.deg2rad(fa_deg))
    tan_fa = np.tan(np.deg2rad(fa_deg))

    for idx in range(n_sample):
        sig = spgr_data[si[idx], sj[idx], sk[idx], :]
        Y = sig / sin_fa
        X = sig / tan_fa
        Xc, Yc = X - X.mean(), Y - Y.mean()
        ss_xx = (Xc ** 2).sum()
        slope = (Xc * Yc).sum() / (ss_xx + 1e-10)
        E1 = np.clip(slope, 0.001, 0.9999)
        t1_linear[idx] = np.clip(-TR / np.log(E1), 0.1, 6.0)

    dt_linear = time.time() - t0
    print(f"  Time: {dt_linear:.2f}s")
    print(f"  T1 median: {np.median(t1_linear)*1000:.0f} ms")
    print(f"  T1 IQR: [{np.percentile(t1_linear, 25)*1000:.0f}, "
          f"{np.percentile(t1_linear, 75)*1000:.0f}] ms")

    # === 2. DESPOT1 NLLS (JAX, no B1) ===
    print("\n--- 2. DESPOT1 NLLS (JAX, no B1) ---")
    t0 = time.time()
    spgr_jax = jnp.array(spgr_data[si, sj, sk, :])  # (n_sample, 8)

    # vmap across all sampled voxels
    fit_fn = jax.vmap(lambda v: despot1_fit_voxel(v, fa_rad, TR, n_iters=200, lr=5e-3))
    results_nlls = fit_fn(spgr_jax)

    t1_nlls = np.array(results_nlls.T1)
    m0_nlls = np.array(results_nlls.M0)
    dt_nlls = time.time() - t0
    print(f"  Time: {dt_nlls:.2f}s (includes JIT compilation)")
    print(f"  T1 median: {np.median(t1_nlls)*1000:.0f} ms")
    print(f"  T1 IQR: [{np.percentile(t1_nlls, 25)*1000:.0f}, "
          f"{np.percentile(t1_nlls, 75)*1000:.0f}] ms")
    print(f"  RMSE median: {np.median(np.array(results_nlls.rmse)):.2f}")

    # === 3. DESPOT1-HIFI (JAX, joint T1 + B1) ===
    if ir_data is not None:
        print(f"\n--- 3. DESPOT1-HIFI (joint T1 + B1, TI={TI*1000:.0f}ms) ---")
        t0 = time.time()
        ir_jax = jnp.array(ir_data[si, sj, sk])  # (n_sample,) single volume

        fit_hifi = jax.vmap(
            lambda s, ir: despot1hifi_fit(
                s, ir, fa_rad, ir_fa_rad, TR, TR_ir, TI, n_iters=300)
        )
        results_hifi = fit_hifi(spgr_jax, ir_jax)

        t1_hifi = np.array(results_hifi["T1"])
        b1_hifi = np.array(results_hifi["B1"])
        dt_hifi = time.time() - t0

        # Filter physiological range
        valid = (t1_hifi > 0.1) & (t1_hifi < 5.0) & (b1_hifi > 0.5) & (b1_hifi < 1.5)

        print(f"  Time: {dt_hifi:.2f}s")
        print(f"  T1 median: {np.median(t1_hifi[valid])*1000:.0f} ms")
        print(f"  T1 IQR: [{np.percentile(t1_hifi[valid], 25)*1000:.0f}, "
              f"{np.percentile(t1_hifi[valid], 75)*1000:.0f}] ms")
        print(f"  B1 ratio median: {np.median(b1_hifi[valid]):.3f}")
        print(f"  B1 ratio IQR: [{np.percentile(b1_hifi[valid], 25):.3f}, "
              f"{np.percentile(b1_hifi[valid], 75):.3f}]")
        print(f"  Valid voxels: {valid.sum()}/{n_sample}")

        # T1 bias: compare HIFI vs linear
        if valid.sum() > 10:
            bias = t1_hifi[valid] - t1_linear[valid]
            print(f"\n  T1 bias (HIFI - linear): {np.median(bias)*1000:.0f} ms median")
            print(f"  B1 < 1 voxels: {(b1_hifi[valid] < 0.95).sum()} "
                  f"({(b1_hifi[valid] < 0.95).mean()*100:.0f}%)")
            print(f"  B1 > 1 voxels: {(b1_hifi[valid] > 1.05).sum()} "
                  f"({(b1_hifi[valid] > 1.05).mean()*100:.0f}%)")

    # === Summary ===
    print("\n=== Summary ===")
    print(f"{'Method':25s} {'T1 median':>10s} {'T1 IQR':>15s} {'Time':>8s}")
    print(f"{'Linear DESPOT1':25s} {np.median(t1_linear)*1000:>8.0f} ms "
          f"[{np.percentile(t1_linear,25)*1000:.0f}-{np.percentile(t1_linear,75)*1000:.0f}] "
          f"{dt_linear:>6.2f}s")
    print(f"{'NLLS DESPOT1 (JAX)':25s} {np.median(t1_nlls)*1000:>8.0f} ms "
          f"[{np.percentile(t1_nlls,25)*1000:.0f}-{np.percentile(t1_nlls,75)*1000:.0f}] "
          f"{dt_nlls:>6.2f}s")
    if ir_data is not None and valid.sum() > 10:
        print(f"{'HIFI (T1+B1, JAX)':25s} {np.median(t1_hifi[valid])*1000:>8.0f} ms "
              f"[{np.percentile(t1_hifi[valid],25)*1000:.0f}-"
              f"{np.percentile(t1_hifi[valid],75)*1000:.0f}] "
              f"{dt_hifi:>6.2f}s")
        print(f"\n  B1 field: median={np.median(b1_hifi[valid]):.3f}, "
              f"range=[{np.percentile(b1_hifi[valid],5):.3f}, "
              f"{np.percentile(b1_hifi[valid],95):.3f}]")


if __name__ == "__main__":
    main()
