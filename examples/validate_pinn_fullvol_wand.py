#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = ["nibabel>=5.0", "numpy>=1.24", "jax>=0.4", "equinox>=0.11", "optax>=0.2", "scipy>=1.10"]
# ///
"""Full-volume RelaxometryPINN T1+B1 mapping on WAND sub-08033.

Trains a RelaxometryPINN that maps (x,y,z) -> [M0, T1, B1] using
all masked brain voxels (subsampled for CPU feasibility).

SPGR VFA (8 flip angles) + IR-SPGR provide the HIFI constraint.
The PINN's shared MLP enforces spatial smoothness on the B1 field
by construction — no explicit spatial regularisation needed.

After training:
  - Saves T1_pinn.nii.gz and B1_pinn.nii.gz to derivatives
  - Reports T1/B1 median, IQR, and B1 neighbour-diff smoothness
  - Compares with voxelwise DESPOT1-HIFI on 500 random voxels

Usage (CPU, subsampled):
  .venv/bin/python examples/validate_pinn_fullvol_wand.py --n-voxels 8000 --n-epochs 2000

Usage (GPU container, full volume):
  python examples/validate_pinn_fullvol_wand.py --n-voxels 0 --n-epochs 3000
"""
import argparse
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from neurojax.qmri.neural_relaxometry import RelaxometryPINN
from neurojax.qmri.pulse_sequence import SPGRSequence
from neurojax.qmri.despot import despot1hifi_fit


# =====================================================================
# Data loading
# =====================================================================

def load_wand_data(wand_root: str, subject: str):
    """Load SPGR, IR-SPGR, and brain mask for a WAND subject.

    Returns:
        spgr_data: (X, Y, Z, 8) float64
        ir_data: (X, Y, Z) float64 — reoriented to match SPGR
        mask: (X, Y, Z) bool
        spgr_img: nibabel image (for affine/header)
    """
    wand = Path(wand_root)
    anat = wand / subject / "ses-02" / "anat"
    deriv = wand / "derivatives" / "qmri" / subject / "ses-02"

    spgr_img = nib.load(str(anat / f"{subject}_ses-02_acq-spgr_part-mag_VFA.nii.gz"))
    ir_img = nib.load(str(anat / f"{subject}_ses-02_acq-spgrIR_part-mag_VFA.nii.gz"))

    spgr_data = spgr_img.get_fdata().astype(np.float64)
    ir_data = ir_img.get_fdata().astype(np.float64)

    # Reorient IR to match SPGR spatial axes
    from nibabel.orientations import io_orientation, ornt_transform, apply_orientation
    if spgr_data.shape[:3] != ir_data.shape[:3]:
        transform = ornt_transform(
            io_orientation(ir_img.affine),
            io_orientation(spgr_img.affine),
        )
        ir_data = apply_orientation(ir_data, transform)
        print(f"  Reoriented IR: {ir_img.shape} -> {ir_data.shape}")

    # Brain mask: use provided mask or create from SPGR intensity
    mask_path = deriv / "brain_mask.nii.gz"
    if mask_path.exists():
        mask = nib.load(str(mask_path)).get_fdata().astype(bool)
        print(f"  Loaded brain mask: {mask.sum()} voxels")
    else:
        print(f"  WARNING: {mask_path} not found, generating from SPGR intensity")
        mean_spgr = spgr_data.mean(axis=-1)
        threshold = np.percentile(mean_spgr[mean_spgr > 0], 20)
        mask = mean_spgr > threshold

    # Sanity checks
    assert spgr_data.shape[:3] == ir_data.shape[:3], (
        f"Shape mismatch after reorientation: SPGR {spgr_data.shape[:3]} vs IR {ir_data.shape[:3]}"
    )
    assert spgr_data.shape[:3] == mask.shape, (
        f"Mask shape mismatch: {mask.shape} vs SPGR {spgr_data.shape[:3]}"
    )

    return spgr_data, ir_data, mask, spgr_img


# =====================================================================
# Signal normalisation
# =====================================================================

def normalise_signals(spgr_vox, ir_vox):
    """Normalise SPGR and IR signals to [0, ~1] range.

    Uses the 95th percentile of SPGR signals as the scale factor.
    This prevents M0 from absorbing arbitrary scanner units.

    Returns:
        spgr_norm, ir_norm, scale_factor
    """
    scale = np.percentile(spgr_vox, 95)
    if scale < 1e-6:
        scale = 1.0
    return spgr_vox / scale, ir_vox / scale, scale


# =====================================================================
# Training
# =====================================================================

def build_model(coord_scale, key):
    """Create a RelaxometryPINN with the WAND volume geometry."""
    return RelaxometryPINN(
        n_params=3,
        hidden_size=128,
        depth=5,
        coord_scale=np.array(coord_scale, dtype=np.float64),
        key=key,
    )


def train_pinn(model, coords_jax, spgr_jax, ir_jax, sequence,
               ir_fa_rad, TR_ir, TI, n_epochs, batch_size, lr):
    """Train RelaxometryPINN with mini-batch SGD.

    Returns:
        model: trained model
        loss_history: list of epoch-average losses
    """
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    n_vox = coords_jax.shape[0]
    n_batches = max(1, n_vox // batch_size)

    @eqx.filter_value_and_grad
    def batch_loss(model, coords_batch, spgr_batch, ir_batch):
        def single_loss(coord, spgr, ir):
            return model.loss(
                coord, spgr, sequence,
                ir_data=ir, ir_fa_rad=ir_fa_rad,
                TR_ir=TR_ir, TI=TI,
                lambda_smooth=0.001,
            )
        losses = jax.vmap(single_loss)(coords_batch, spgr_batch, ir_batch)
        return jnp.mean(losses)

    @eqx.filter_jit
    def train_step(model, opt_state, coords_batch, spgr_batch, ir_batch):
        loss, grads = batch_loss(model, coords_batch, spgr_batch, ir_batch)
        updates, opt_state_new = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state_new, loss

    loss_history = []
    key = jax.random.PRNGKey(1)
    t0 = time.time()

    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n_vox)

        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            model, opt_state, loss = train_step(
                model, opt_state,
                coords_jax[idx], spgr_jax[idx], ir_jax[idx],
            )
            epoch_loss += float(loss)

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if (epoch + 1) % 200 == 0 or epoch == 0:
            elapsed = time.time() - t0
            # Quick sample of parameter estimates
            params_sample = jax.vmap(model.predict_params)(coords_jax[:10])
            t1_s = np.array(params_sample[:, 1])
            b1_s = np.array(params_sample[:, 2])
            print(
                f"  Epoch {epoch+1:5d}/{n_epochs}: "
                f"loss={avg_loss:.4f}, "
                f"T1={np.median(t1_s)*1000:.0f}ms, "
                f"B1={np.median(b1_s):.3f}, "
                f"elapsed={elapsed:.0f}s"
            )

    total_time = time.time() - t0
    print(f"  Training complete: {total_time:.1f}s ({total_time/n_epochs:.3f} s/epoch)")
    return model, loss_history


# =====================================================================
# Prediction + saving
# =====================================================================

def predict_all(model, coords_jax, chunk_size=4096):
    """Predict tissue parameters for all voxels, in chunks to save memory.

    Returns:
        params: (n_vox, 3) array [M0, T1, B1]
    """
    n_vox = coords_jax.shape[0]
    all_params = []
    for start in range(0, n_vox, chunk_size):
        end = min(start + chunk_size, n_vox)
        chunk = jax.vmap(model.predict_params)(coords_jax[start:end])
        all_params.append(np.array(chunk))
    return np.concatenate(all_params, axis=0)


def save_nifti_map(data_3d, ref_img, out_path, description=""):
    """Save a 3D parameter map as NIfTI, using the reference image geometry."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data_3d.astype(np.float32), ref_img.affine, ref_img.header)
    img.header.set_data_dtype(np.float32)
    if description:
        img.header["descrip"] = description.encode()[:80]
    nib.save(img, str(out_path))
    print(f"  Saved: {out_path}")


def build_volume_maps(params, brain_idx, vol_shape, scale_factor):
    """Reconstruct 3D volumes from flat parameter arrays.

    Returns:
        T1_map: (X, Y, Z) in milliseconds
        B1_map: (X, Y, Z) as ratio
        M0_map: (X, Y, Z) rescaled to original scanner units
    """
    T1_map = np.zeros(vol_shape, dtype=np.float32)
    B1_map = np.zeros(vol_shape, dtype=np.float32)
    M0_map = np.zeros(vol_shape, dtype=np.float32)

    T1_map[brain_idx] = params[:, 1] * 1000.0  # s -> ms
    B1_map[brain_idx] = params[:, 2]
    M0_map[brain_idx] = params[:, 0] * scale_factor  # undo normalisation

    return T1_map, B1_map, M0_map


# =====================================================================
# Evaluation metrics
# =====================================================================

def report_pinn_stats(params, brain_idx, vol_shape):
    """Report T1 and B1 statistics from PINN predictions."""
    t1_vals = params[:, 1] * 1000.0  # ms
    b1_vals = params[:, 2]
    m0_vals = params[:, 0]

    # Filter physiologically plausible values
    valid = (t1_vals > 100) & (t1_vals < 5000) & (b1_vals > 0.5) & (b1_vals < 1.5)
    n_valid = valid.sum()

    print(f"\n=== PINN Results ({n_valid}/{len(t1_vals)} valid voxels) ===")
    print(f"  T1 median: {np.median(t1_vals[valid]):.0f} ms")
    print(f"  T1 IQR:    [{np.percentile(t1_vals[valid], 25):.0f}, "
          f"{np.percentile(t1_vals[valid], 75):.0f}] ms")
    print(f"  B1 median: {np.median(b1_vals[valid]):.4f}")
    print(f"  B1 IQR:    [{np.percentile(b1_vals[valid], 25):.4f}, "
          f"{np.percentile(b1_vals[valid], 75):.4f}]")
    print(f"  M0 median: {np.median(m0_vals[valid]):.1f} (normalised)")

    return valid


def compute_b1_smoothness(b1_vals, coords, valid_mask):
    """Compute B1 field smoothness as std of nearest-neighbour differences.

    Lower values indicate smoother fields.
    """
    from scipy.spatial import KDTree

    coords_valid = coords[valid_mask]
    b1_valid = b1_vals[valid_mask]

    if len(coords_valid) < 10:
        print("  Too few valid voxels for smoothness computation")
        return np.nan

    tree = KDTree(coords_valid)
    _, nn_idx = tree.query(coords_valid, k=2)
    nn_idx = nn_idx[:, 1]  # exclude self

    b1_nn_diff = b1_valid - b1_valid[nn_idx]
    smoothness = np.std(b1_nn_diff)
    return smoothness


def compare_with_hifi(model, coords_jax, spgr_jax, ir_jax,
                      sequence, ir_fa_rad, TR_ir, TI, n_compare=500):
    """Run voxelwise DESPOT1-HIFI on a random subset and compare with PINN.

    Returns a dict of comparison statistics.
    """
    n_vox = coords_jax.shape[0]
    rng = np.random.default_rng(42)
    n_compare = min(n_compare, n_vox)
    compare_idx = rng.choice(n_vox, n_compare, replace=False)

    spgr_sub = spgr_jax[compare_idx]
    ir_sub = ir_jax[compare_idx]
    coords_sub = coords_jax[compare_idx]
    fa_rad = sequence.flip_angles_rad

    # --- Voxelwise HIFI ---
    print(f"\n--- Running voxelwise DESPOT1-HIFI on {n_compare} voxels ---")
    t0 = time.time()
    fit_hifi = jax.vmap(
        lambda s, ir: despot1hifi_fit(
            s, ir, fa_rad, ir_fa_rad, 0.004, TR_ir, TI, n_iters=300
        )
    )
    hifi_results = fit_hifi(spgr_sub, ir_sub)
    t1_hifi = np.array(hifi_results["T1"])  # seconds
    b1_hifi = np.array(hifi_results["B1"])
    dt_hifi = time.time() - t0
    print(f"  HIFI time: {dt_hifi:.1f}s")

    # --- PINN predictions on same voxels ---
    pinn_params = np.array(jax.vmap(model.predict_params)(coords_sub))
    t1_pinn = pinn_params[:, 1]   # seconds
    b1_pinn = pinn_params[:, 2]

    # --- Filter valid in both ---
    valid_hifi = (t1_hifi > 0.1) & (t1_hifi < 5.0) & (b1_hifi > 0.5) & (b1_hifi < 1.5)
    valid_pinn = (t1_pinn > 0.1) & (t1_pinn < 5.0) & (b1_pinn > 0.5) & (b1_pinn < 1.5)
    both_valid = valid_hifi & valid_pinn
    n_both = both_valid.sum()

    if n_both < 10:
        print(f"  WARNING: only {n_both} voxels valid in both methods, skipping comparison")
        return {}

    print(f"\n=== PINN vs HIFI Comparison ({n_both} voxels) ===")
    header = f"{'Metric':22s} {'HIFI (voxelwise)':>18s} {'PINN (spatial)':>18s}"
    print(header)
    print("-" * len(header))

    t1h, t1p = t1_hifi[both_valid] * 1000, t1_pinn[both_valid] * 1000
    b1h, b1p = b1_hifi[both_valid], b1_pinn[both_valid]

    print(f"{'T1 median (ms)':22s} {np.median(t1h):>18.0f} {np.median(t1p):>18.0f}")
    print(f"{'T1 IQR (ms)':22s} "
          f"{'['+str(int(np.percentile(t1h,25)))+'-'+str(int(np.percentile(t1h,75)))+']':>18s} "
          f"{'['+str(int(np.percentile(t1p,25)))+'-'+str(int(np.percentile(t1p,75)))+']':>18s}")
    print(f"{'B1 median':22s} {np.median(b1h):>18.4f} {np.median(b1p):>18.4f}")
    print(f"{'B1 IQR':22s} "
          f"{'['+f'{np.percentile(b1h,25):.3f}'+'-'+f'{np.percentile(b1h,75):.3f}'+']':>18s} "
          f"{'['+f'{np.percentile(b1p,25):.3f}'+'-'+f'{np.percentile(b1p,75):.3f}'+']':>18s}")
    print(f"{'Valid voxels':22s} {valid_hifi.sum():>18d} {valid_pinn.sum():>18d}")

    # Correlation
    r_t1 = np.corrcoef(t1h, t1p)[0, 1]
    r_b1 = np.corrcoef(b1h, b1p)[0, 1]
    print(f"\n{'T1 Pearson r':22s} {r_t1:>18.4f}")
    print(f"{'B1 Pearson r':22s} {r_b1:>18.4f}")

    # B1 smoothness comparison (nearest-neighbour diff std)
    coords_np = np.array(coords_sub[both_valid])
    b1_smooth_hifi = compute_b1_smoothness(b1h, coords_np, np.ones(n_both, dtype=bool))
    b1_smooth_pinn = compute_b1_smoothness(b1p, coords_np, np.ones(n_both, dtype=bool))
    print(f"\n{'B1 nn-diff std':22s} {b1_smooth_hifi:>18.5f} {b1_smooth_pinn:>18.5f}")
    print(f"  (lower = smoother B1 field)")

    # T1 MAE
    t1_mae = np.mean(np.abs(t1h - t1p))
    print(f"\n{'T1 MAE (ms)':22s} {t1_mae:>18.1f}")

    return {
        "r_t1": r_t1, "r_b1": r_b1,
        "t1_mae_ms": t1_mae,
        "b1_smooth_hifi": b1_smooth_hifi,
        "b1_smooth_pinn": b1_smooth_pinn,
        "n_compared": n_both,
    }


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full-volume RelaxometryPINN T1+B1 mapping on WAND",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wand-root", default="/data/raw/wand",
                        help="Root of WAND dataset")
    parser.add_argument("--subject", default="sub-08033",
                        help="BIDS subject ID")
    parser.add_argument("--n-voxels", type=int, default=8000,
                        help="Number of brain voxels to use (0=all)")
    parser.add_argument("--n-epochs", type=int, default=2000,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Adam learning rate")
    parser.add_argument("--ti", type=float, default=0.45,
                        help="Inversion time for IR-SPGR (seconds)")
    parser.add_argument("--n-compare", type=int, default=500,
                        help="Number of voxels for HIFI comparison")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: derivatives/qmri/<subject>/ses-02)")
    parser.add_argument("--seed", type=int, default=0,
                        help="PRNG seed for model initialisation")
    args = parser.parse_args()

    print("=" * 70)
    print("RelaxometryPINN Full-Volume T1+B1 Mapping")
    print("=" * 70)
    print(f"  JAX backend:  {jax.default_backend()}")
    print(f"  JAX devices:  {jax.devices()}")
    print(f"  Subject:      {args.subject}")
    print(f"  Voxels:       {'all' if args.n_voxels == 0 else args.n_voxels}")
    print(f"  Epochs:       {args.n_epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"  TI:           {args.ti}s")
    print()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("--- 1. Loading data ---")
    spgr_data, ir_data, mask, spgr_img = load_wand_data(args.wand_root, args.subject)
    vol_shape = spgr_data.shape[:3]
    print(f"  Volume shape: {vol_shape}")
    print(f"  SPGR shape:   {spgr_data.shape}")
    print(f"  IR shape:     {ir_data.shape}")
    print(f"  Brain voxels: {mask.sum()}")

    # ------------------------------------------------------------------
    # 2. Extract brain voxels + subsample
    # ------------------------------------------------------------------
    print("\n--- 2. Preparing voxel data ---")
    brain_idx = np.where(mask)
    n_brain = len(brain_idx[0])

    # Subsample if requested
    rng = np.random.default_rng(args.seed)
    if args.n_voxels > 0 and args.n_voxels < n_brain:
        sample = rng.choice(n_brain, args.n_voxels, replace=False)
        sample.sort()  # keep spatial ordering for cache locality
    else:
        sample = np.arange(n_brain)
        args.n_voxels = n_brain

    n_use = len(sample)
    sub_idx = (brain_idx[0][sample], brain_idx[1][sample], brain_idx[2][sample])

    coords = np.stack([sub_idx[0], sub_idx[1], sub_idx[2]], axis=1).astype(np.float64)
    spgr_vox = spgr_data[sub_idx[0], sub_idx[1], sub_idx[2], :]
    ir_vox = ir_data[sub_idx[0], sub_idx[1], sub_idx[2]]

    print(f"  Using {n_use} / {n_brain} brain voxels")
    print(f"  SPGR range:   [{spgr_vox.min():.1f}, {spgr_vox.max():.1f}]")
    print(f"  IR range:     [{ir_vox.min():.1f}, {ir_vox.max():.1f}]")

    # Normalise signals
    spgr_norm, ir_norm, scale_factor = normalise_signals(spgr_vox, ir_vox)
    print(f"  Scale factor: {scale_factor:.1f}")

    # Convert to JAX arrays
    coords_jax = jnp.array(coords)
    spgr_jax = jnp.array(spgr_norm)
    ir_jax = jnp.array(ir_norm)

    # ------------------------------------------------------------------
    # 3. Build sequence + model
    # ------------------------------------------------------------------
    print("\n--- 3. Building model ---")
    fa_deg = [2, 4, 6, 8, 10, 12, 14, 18]
    sequence = SPGRSequence(flip_angles_deg=fa_deg, TR=0.004, TE=0.0019)
    ir_fa_rad = jnp.deg2rad(5.0)
    TR_ir = 0.004
    TI = args.ti

    key = jax.random.PRNGKey(args.seed)
    model = build_model(coord_scale=vol_shape, key=key)
    n_params_total = sum(
        x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))
    )
    print(f"  PINN:          n_params=3, hidden=128, depth=5")
    print(f"  Total weights: {n_params_total:,}")
    print(f"  coord_scale:   {list(vol_shape)}")
    print(f"  Protocol:      FA={fa_deg}, TR=4ms, IR FA=5, TI={TI}s")

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    print(f"\n--- 4. Training ({args.n_epochs} epochs, batch={args.batch_size}) ---")
    model, loss_history = train_pinn(
        model, coords_jax, spgr_jax, ir_jax, sequence,
        ir_fa_rad=ir_fa_rad, TR_ir=TR_ir, TI=TI,
        n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr,
    )

    # ------------------------------------------------------------------
    # 5. Predict T1 + B1 maps
    # ------------------------------------------------------------------
    print("\n--- 5. Predicting parameter maps ---")
    params = predict_all(model, coords_jax)

    # Report statistics
    valid = report_pinn_stats(params, sub_idx, vol_shape)

    # B1 smoothness on all valid PINN voxels
    b1_vals = params[:, 2]
    b1_smooth = compute_b1_smoothness(b1_vals, coords, valid)
    print(f"  B1 nn-diff std (smoothness): {b1_smooth:.5f}")

    # ------------------------------------------------------------------
    # 6. Save NIfTI maps
    # ------------------------------------------------------------------
    print("\n--- 6. Saving parameter maps ---")
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(args.wand_root) / "derivatives" / "qmri" / args.subject / "ses-02"
    )

    T1_map, B1_map, M0_map = build_volume_maps(params, sub_idx, vol_shape, scale_factor)
    save_nifti_map(T1_map, spgr_img, out_dir / "T1_pinn.nii.gz",
                   description=f"PINN T1 (ms), {n_use} voxels, {args.n_epochs} epochs")
    save_nifti_map(B1_map, spgr_img, out_dir / "B1_pinn.nii.gz",
                   description=f"PINN B1 ratio, {n_use} voxels, {args.n_epochs} epochs")
    save_nifti_map(M0_map, spgr_img, out_dir / "M0_pinn.nii.gz",
                   description=f"PINN M0 (a.u.), {n_use} voxels, {args.n_epochs} epochs")

    # ------------------------------------------------------------------
    # 7. Compare with voxelwise HIFI
    # ------------------------------------------------------------------
    print(f"\n--- 7. Comparison with voxelwise HIFI ---")
    comparison = compare_with_hifi(
        model, coords_jax, spgr_jax, ir_jax,
        sequence, ir_fa_rad, TR_ir, TI,
        n_compare=args.n_compare,
    )

    # ------------------------------------------------------------------
    # 8. Loss convergence summary
    # ------------------------------------------------------------------
    print(f"\n--- 8. Convergence summary ---")
    print(f"  Initial loss:  {loss_history[0]:.6f}")
    print(f"  Final loss:    {loss_history[-1]:.6f}")
    print(f"  Reduction:     {loss_history[0] / max(loss_history[-1], 1e-12):.1f}x")

    # Check for convergence (last 10% of training)
    tail = loss_history[int(0.9 * len(loss_history)):]
    if len(tail) > 1:
        tail_change = abs(tail[-1] - tail[0]) / max(abs(tail[0]), 1e-12)
        print(f"  Last 10% change: {tail_change:.4f} "
              f"({'converged' if tail_change < 0.05 else 'still improving'})")

    print("\n" + "=" * 70)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
