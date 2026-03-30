#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = ["nibabel>=5.0", "numpy>=1.24", "jax>=0.4", "equinox>=0.11", "optax>=0.2"]
# ///
"""Validate RelaxometryPINN with spatial B1 learning on WAND VFA data.

Compares:
  1. Voxelwise DESPOT1-HIFI (T1+B1 per voxel, no spatial coherence)
  2. RelaxometryPINN (T1+B1 as smooth spatial fields via shared MLP)

Both use SPGR VFA (8 FA) + IR-SPGR data from WAND ses-02.
The PINN learns spatially coherent B1 by construction — a single network
maps (x,y,z) → (M0, T1, B1), so B1 is inherently smooth.

Usage:
  python examples/validate_pinn_b1_wand.py [--n-voxels 500] [--n-epochs 500]
"""
import argparse
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


def main():
    parser = argparse.ArgumentParser(description="PINN B1 validation on WAND")
    parser.add_argument("--wand-root", default="/data/raw/wand")
    parser.add_argument("--subject", default="sub-08033")
    parser.add_argument("--n-voxels", type=int, default=500)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ti", type=float, default=0.45)
    args = parser.parse_args()

    wand = Path(args.wand_root)
    subject = args.subject
    anat = wand / subject / "ses-02" / "anat"

    # === Load data ===
    print("=== Loading WAND VFA data ===")
    spgr_img = nib.load(str(anat / f"{subject}_ses-02_acq-spgr_part-mag_VFA.nii.gz"))
    ir_img = nib.load(str(anat / f"{subject}_ses-02_acq-spgrIR_part-mag_VFA.nii.gz"))

    spgr_data = spgr_img.get_fdata().astype(np.float64)
    ir_data = ir_img.get_fdata().astype(np.float64)

    # Reorient IR to match SPGR
    from nibabel.orientations import io_orientation, ornt_transform, apply_orientation
    if spgr_data.shape[:3] != ir_data.shape:
        transform = ornt_transform(
            io_orientation(ir_img.affine),
            io_orientation(spgr_img.affine)
        )
        ir_data = apply_orientation(ir_data, transform)
        print(f"Reoriented IR → {ir_data.shape}")

    shape = spgr_data.shape[:3]
    print(f"SPGR: {spgr_data.shape}, IR: {ir_data.shape}")

    # Protocol
    fa_deg = [2, 4, 6, 8, 10, 12, 14, 18]
    sequence = SPGRSequence(flip_angles_deg=fa_deg, TR=0.004, TE=0.0019)
    ir_fa_rad = jnp.deg2rad(5.0)
    TR_ir = 0.004
    TI = args.ti

    # Brain mask + sample voxels
    mean_spgr = spgr_data.mean(axis=-1)
    mask = mean_spgr > np.percentile(mean_spgr[mean_spgr > 0], 20)
    brain_idx = np.where(mask)
    n_brain = len(brain_idx[0])

    rng = np.random.default_rng(42)
    n_sample = min(args.n_voxels, n_brain)
    sample = rng.choice(n_brain, n_sample, replace=False)

    coords = np.stack([brain_idx[0][sample], brain_idx[1][sample],
                       brain_idx[2][sample]], axis=1).astype(np.float64)
    spgr_vox = spgr_data[brain_idx[0][sample], brain_idx[1][sample],
                          brain_idx[2][sample], :]
    ir_vox = ir_data[brain_idx[0][sample], brain_idx[1][sample],
                      brain_idx[2][sample]]

    coords_jax = jnp.array(coords)
    spgr_jax = jnp.array(spgr_vox)
    ir_jax = jnp.array(ir_vox)

    print(f"Training on {n_sample} voxels, {args.n_epochs} epochs, "
          f"batch={args.batch_size}")

    # === 1. Voxelwise DESPOT1-HIFI (baseline) ===
    print("\n--- 1. Voxelwise DESPOT1-HIFI ---")
    t0 = time.time()
    fa_rad = sequence.flip_angles_rad

    fit_hifi = jax.vmap(
        lambda s, ir: despot1hifi_fit(s, ir, fa_rad, ir_fa_rad, 0.004, TR_ir, TI,
                                       n_iters=300)
    )
    hifi_results = fit_hifi(spgr_jax, ir_jax)
    t1_hifi = np.array(hifi_results["T1"])
    b1_hifi = np.array(hifi_results["B1"])
    dt_hifi = time.time() - t0

    valid = (t1_hifi > 0.1) & (t1_hifi < 5.0) & (b1_hifi > 0.5) & (b1_hifi < 1.5)
    print(f"  Time: {dt_hifi:.1f}s")
    print(f"  T1 median: {np.median(t1_hifi[valid])*1000:.0f} ms")
    print(f"  B1 median: {np.median(b1_hifi[valid]):.3f}")
    print(f"  Valid: {valid.sum()}/{n_sample}")

    # === 2. RelaxometryPINN with B1 ===
    print(f"\n--- 2. RelaxometryPINN (n_params=3, B1 field) ---")
    key = jax.random.PRNGKey(0)
    model = RelaxometryPINN(
        n_params=3,
        hidden_size=128,
        depth=5,
        coord_scale=np.array(shape, dtype=np.float64),
        key=key,
    )

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Batch loss: mean over a batch of voxels
    @eqx.filter_value_and_grad
    def batch_loss(model, coords_batch, spgr_batch, ir_batch):
        def single_loss(coord, spgr, ir):
            return model.loss(coord, spgr, sequence,
                              ir_data=ir, ir_fa_rad=ir_fa_rad,
                              TR_ir=TR_ir, TI=TI,
                              lambda_smooth=0.001)
        losses = jax.vmap(single_loss)(coords_batch, spgr_batch, ir_batch)
        return jnp.mean(losses)

    @eqx.filter_jit
    def train_step(model, opt_state, coords_batch, spgr_batch, ir_batch):
        loss, grads = batch_loss(model, coords_batch, spgr_batch, ir_batch)
        updates, opt_state = optimizer.update(grads, opt_state,
                                               eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # Training loop
    t0 = time.time()
    bs = args.batch_size
    n_batches = max(1, n_sample // bs)
    key = jax.random.PRNGKey(1)

    for epoch in range(args.n_epochs):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n_sample)

        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * bs:(b + 1) * bs]
            model, opt_state, loss = train_step(
                model, opt_state,
                coords_jax[idx], spgr_jax[idx], ir_jax[idx]
            )
            epoch_loss += float(loss)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            avg_loss = epoch_loss / n_batches
            # Quick eval on first 10 voxels
            params_sample = jax.vmap(model.predict_params)(coords_jax[:10])
            t1_s = np.array(params_sample[:, 1])
            b1_s = np.array(params_sample[:, 2])
            print(f"  Epoch {epoch+1:4d}: loss={avg_loss:.2f}, "
                  f"T1={np.median(t1_s)*1000:.0f}ms, B1={np.median(b1_s):.3f}")

    dt_pinn = time.time() - t0

    # Final prediction on all training voxels
    pinn_params = jax.vmap(model.predict_params)(coords_jax)
    t1_pinn = np.array(pinn_params[:, 1])
    b1_pinn = np.array(pinn_params[:, 2])
    m0_pinn = np.array(pinn_params[:, 0])

    valid_pinn = (t1_pinn > 0.1) & (t1_pinn < 5.0) & (b1_pinn > 0.5) & (b1_pinn < 1.5)

    print(f"\n  Training time: {dt_pinn:.1f}s")
    print(f"  T1 median: {np.median(t1_pinn[valid_pinn])*1000:.0f} ms")
    print(f"  B1 median: {np.median(b1_pinn[valid_pinn]):.3f}")
    print(f"  B1 IQR: [{np.percentile(b1_pinn[valid_pinn], 25):.3f}, "
          f"{np.percentile(b1_pinn[valid_pinn], 75):.3f}]")
    print(f"  Valid: {valid_pinn.sum()}/{n_sample}")

    # === Comparison ===
    both_valid = valid & valid_pinn
    if both_valid.sum() > 10:
        print(f"\n=== Comparison ({both_valid.sum()} voxels) ===")
        print(f"{'':20s} {'HIFI (voxel)':>15s} {'PINN (spatial)':>15s}")
        print(f"{'T1 median':20s} "
              f"{np.median(t1_hifi[both_valid])*1000:>12.0f} ms "
              f"{np.median(t1_pinn[both_valid])*1000:>12.0f} ms")
        print(f"{'T1 IQR':20s} "
              f"[{np.percentile(t1_hifi[both_valid],25)*1000:.0f}-"
              f"{np.percentile(t1_hifi[both_valid],75)*1000:.0f}]"
              f"{'':>5s}"
              f"[{np.percentile(t1_pinn[both_valid],25)*1000:.0f}-"
              f"{np.percentile(t1_pinn[both_valid],75)*1000:.0f}]")
        print(f"{'B1 median':20s} "
              f"{np.median(b1_hifi[both_valid]):>15.3f} "
              f"{np.median(b1_pinn[both_valid]):>15.3f}")
        print(f"{'B1 IQR':20s} "
              f"[{np.percentile(b1_hifi[both_valid],25):.3f}-"
              f"{np.percentile(b1_hifi[both_valid],75):.3f}]"
              f"{'':>5s}"
              f"[{np.percentile(b1_pinn[both_valid],25):.3f}-"
              f"{np.percentile(b1_pinn[both_valid],75):.3f}]")

        # Correlation
        r_t1 = np.corrcoef(t1_hifi[both_valid], t1_pinn[both_valid])[0, 1]
        r_b1 = np.corrcoef(b1_hifi[both_valid], b1_pinn[both_valid])[0, 1]
        print(f"{'T1 correlation':20s} r = {r_t1:.3f}")
        print(f"{'B1 correlation':20s} r = {r_b1:.3f}")

        # B1 field smoothness (std of spatial differences)
        # Use coordinate-sorted neighbors
        from scipy.spatial import KDTree
        tree = KDTree(coords[both_valid])
        _, nn_idx = tree.query(coords[both_valid], k=2)  # nearest neighbor
        nn_idx = nn_idx[:, 1]  # exclude self

        b1_diff_hifi = np.std(b1_hifi[both_valid] -
                               b1_hifi[both_valid][nn_idx])
        b1_diff_pinn = np.std(b1_pinn[both_valid] -
                               b1_pinn[both_valid][nn_idx])
        print(f"\n{'B1 smoothness':20s} {'HIFI':>15s} {'PINN':>15s}")
        print(f"{'  nn-diff std':20s} {b1_diff_hifi:>15.4f} {b1_diff_pinn:>15.4f}")
        print(f"  (lower = smoother B1 field)")


if __name__ == "__main__":
    main()
