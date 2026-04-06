#!/usr/bin/env python3
"""EIT forward/inverse modeling with the SCI head model.

Demonstrates Electrical Impedance Tomography (EIT) on a realistic
head geometry using the Complete Electrode Model (CEM).

Pipeline:
    1. Build a spherical head model (or load SCI dataset)
    2. Set up an EIT measurement protocol
    3. Solve the forward problem for a reference conductivity
    4. Simulate a focal conductivity perturbation (e.g., stroke)
    5. Solve the forward problem for the perturbed conductivity
    6. Reconstruct the conductivity change via Gauss-Newton

Usage:
    # Quick test with synthetic spherical model
    python demo_eit_sci_head.py

    # With downloaded SCI head model
    python demo_eit_sci_head.py --data-dir /data/datasets/sci_head_model

    # Download and run with SCI data
    python demo_eit_sci_head.py --download --data-dir /data/datasets/sci_head_model

Reference:
    Warner A, Tate J, Burton B, Johnson CR (2019).
    A high-resolution head and brain computer model for forward and
    inverse EEG simulation. bioRxiv 552190.
"""

import argparse
import time
import numpy as np


def run_synthetic_demo():
    """Run EIT demo on a synthetic spherical head model."""
    from neurojax.geometry.sci_head_model import make_spherical_head
    from neurojax.geometry.eit import (
        adjacent_pattern, opposite_pattern,
        solve_eit_forward, compute_jacobian,
        difference_eit, simulate_perturbation,
        element_laplacian,
    )

    print("=" * 60)
    print("EIT Forward/Inverse Demo — Spherical Head Model")
    print("=" * 60)

    # ── 1. Build head model ──────────────────────────────────
    print("\n[1] Building spherical head model...")
    t0 = time.time()
    head = make_spherical_head(
        radii=(60.0, 70.0, 78.0, 85.0),
        conductivities=(0.33, 1.79, 0.006, 0.43),
        tissue_names=("Brain", "CSF", "Skull", "Scalp"),
        n_electrodes=32,
        mesh_density=12.0,
    )
    print(f"    Nodes:      {len(head.vertices):,}")
    print(f"    Elements:   {len(head.elements):,}")
    print(f"    Electrodes: {len(head.electrode_nodes)}")
    print(f"    Tissues:    {head.tissue_map}")
    print(f"    Time:       {time.time() - t0:.1f}s")

    # ── 2. Set up protocol ───────────────────────────────────
    print("\n[2] Setting up adjacent measurement protocol...")
    protocol = adjacent_pattern(len(head.electrode_nodes))
    print(f"    Injection patterns:  {len(protocol.injection)}")
    print(f"    Measurement channels: {len(protocol.measurement)}")

    # ── 3. Reference forward solve ───────────────────────────
    print("\n[3] Solving forward problem (reference)...")
    t0 = time.time()
    fwd_ref = solve_eit_forward(
        head.vertices, head.elements, head.conductivity,
        head.electrode_nodes, protocol,
    )
    print(f"    Voltage range: [{fwd_ref.measurements.min():.4e}, "
          f"{fwd_ref.measurements.max():.4e}]")
    print(f"    Time:          {time.time() - t0:.1f}s")

    # ── 4. Simulate perturbation ─────────────────────────────
    print("\n[4] Simulating focal conductivity perturbation...")
    # Simulate a conductive lesion (e.g., haemorrhagic stroke)
    # at 30 mm lateral, within the brain
    sigma_pert = simulate_perturbation(
        head.vertices, head.elements, head.conductivity,
        perturbation_centre=np.array([30.0, 0.0, 10.0]),
        perturbation_radius=15.0,
        perturbation_value=0.7,  # blood-like conductivity
    )
    n_changed = np.sum(sigma_pert != head.conductivity)
    print(f"    Perturbed elements: {n_changed} / {len(head.elements)}")
    print(f"    Perturbation:      0.33 -> 0.70 S/m (haemorrhage)")

    # ── 5. Perturbed forward solve ───────────────────────────
    print("\n[5] Solving forward problem (perturbed)...")
    t0 = time.time()
    fwd_pert = solve_eit_forward(
        head.vertices, head.elements, sigma_pert,
        head.electrode_nodes, protocol,
    )
    delta_v = fwd_pert.measurements - fwd_ref.measurements
    print(f"    Voltage change range: [{delta_v.min():.4e}, "
          f"{delta_v.max():.4e}]")
    print(f"    Max |ΔV|/|V_ref|:    "
          f"{np.max(np.abs(delta_v) / np.maximum(np.abs(fwd_ref.measurements), 1e-10)):.2%}")
    print(f"    Time:                {time.time() - t0:.1f}s")

    # ── 6. Compute Jacobian ──────────────────────────────────
    print("\n[6] Computing Jacobian (adjoint method)...")
    t0 = time.time()
    J = compute_jacobian(
        head.vertices, head.elements, head.conductivity,
        head.electrode_nodes, protocol,
        forward_result=fwd_ref,
    )
    print(f"    Jacobian shape: {J.shape}")
    print(f"    Nonzero:        {np.count_nonzero(J)} / {J.size} "
          f"({100 * np.count_nonzero(J) / J.size:.1f}%)")
    print(f"    Time:           {time.time() - t0:.1f}s")

    # ── 7. Inverse reconstruction ─────────────────────────────
    print("\n[7] Reconstructing conductivity change...")

    # Tikhonov
    t0 = time.time()
    delta_sigma_tik = difference_eit(
        fwd_ref.measurements, fwd_pert.measurements, J,
        method="tikhonov", alpha=0.01, normalise=False,
    )
    print(f"    Tikhonov: range [{delta_sigma_tik.min():.4e}, "
          f"{delta_sigma_tik.max():.4e}], time {time.time() - t0:.2f}s")

    # NOSER
    t0 = time.time()
    delta_sigma_nos = difference_eit(
        fwd_ref.measurements, fwd_pert.measurements, J,
        method="noser", alpha=0.01, normalise=False,
        sigma_ref=head.conductivity,
    )
    print(f"    NOSER:    range [{delta_sigma_nos.min():.4e}, "
          f"{delta_sigma_nos.max():.4e}], time {time.time() - t0:.2f}s")

    # TV
    t0 = time.time()
    delta_sigma_tv = difference_eit(
        fwd_ref.measurements, fwd_pert.measurements, J,
        method="tv", alpha=0.01, normalise=False, max_iter=10,
    )
    print(f"    TV:       range [{delta_sigma_tv.min():.4e}, "
          f"{delta_sigma_tv.max():.4e}], time {time.time() - t0:.2f}s")

    # ── 8. Evaluate reconstruction quality ───────────────────
    print("\n[8] Reconstruction quality...")
    true_delta = sigma_pert - head.conductivity

    for name, recon in [("Tikhonov", delta_sigma_tik),
                        ("NOSER", delta_sigma_nos),
                        ("TV", delta_sigma_tv)]:
        # Correlation with ground truth
        if np.std(recon) > 0 and np.std(true_delta) > 0:
            corr = np.corrcoef(recon, true_delta)[0, 1]
        else:
            corr = 0.0

        # Localisation: does the max reconstruction overlap the lesion?
        max_elem = np.argmax(np.abs(recon))
        centroid = np.mean(head.vertices[head.elements[max_elem]], axis=0)
        dist_to_lesion = np.linalg.norm(
            centroid - np.array([30.0, 0.0, 10.0])
        )

        print(f"    {name:10s}: corr={corr:+.3f}, "
              f"peak at [{centroid[0]:.0f},{centroid[1]:.0f},{centroid[2]:.0f}] mm, "
              f"dist to lesion={dist_to_lesion:.0f} mm")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


def run_sci_demo(data_dir: str, download: bool = False):
    """Run EIT demo with the real SCI head model dataset."""
    from neurojax.geometry.sci_head_model import (
        download_sci_head_model, load_sci_head_model,
    )
    from neurojax.geometry.eit import (
        adjacent_pattern, solve_eit_forward, compute_jacobian,
        difference_eit, simulate_perturbation,
    )

    print("=" * 60)
    print("EIT Forward/Inverse Demo — SCI Head Model")
    print("=" * 60)

    # Download if requested
    if download:
        print("\n[0] Downloading SCI head model dataset...")
        download_sci_head_model(
            data_dir, components=["mesh", "segmentation", "eeg"]
        )

    # Load
    print("\n[1] Loading SCI head model...")
    t0 = time.time()
    head = load_sci_head_model(data_dir, n_channels=128)
    print(f"    Nodes:      {len(head.vertices):,}")
    print(f"    Elements:   {len(head.elements):,}")
    print(f"    Electrodes: {len(head.electrode_nodes)}")
    print(f"    Tissues:    {head.tissue_map}")
    print(f"    Time:       {time.time() - t0:.1f}s")

    # Protocol
    n_elec = len(head.electrode_nodes)
    print(f"\n[2] Setting up adjacent protocol ({n_elec} electrodes)...")
    protocol = adjacent_pattern(n_elec)

    # Forward
    print("\n[3] Solving CEM forward (reference)...")
    t0 = time.time()
    fwd_ref = solve_eit_forward(
        head.vertices, head.elements, head.conductivity,
        head.electrode_nodes, protocol,
    )
    print(f"    Time: {time.time() - t0:.1f}s")

    # Perturbation (simulate right-hemisphere stroke)
    print("\n[4] Simulating stroke perturbation...")
    sigma_pert = simulate_perturbation(
        head.vertices, head.elements, head.conductivity,
        perturbation_centre=np.array([40.0, -20.0, 30.0]),
        perturbation_radius=20.0,
        perturbation_value=0.7,
    )

    print("\n[5] Solving CEM forward (perturbed)...")
    t0 = time.time()
    fwd_pert = solve_eit_forward(
        head.vertices, head.elements, sigma_pert,
        head.electrode_nodes, protocol,
    )
    print(f"    Time: {time.time() - t0:.1f}s")

    # Jacobian + reconstruction
    print("\n[6] Computing Jacobian...")
    t0 = time.time()
    J = compute_jacobian(
        head.vertices, head.elements, head.conductivity,
        head.electrode_nodes, protocol, forward_result=fwd_ref,
    )
    print(f"    Shape: {J.shape}, time: {time.time() - t0:.1f}s")

    print("\n[7] Reconstructing...")
    delta_sigma = difference_eit(
        fwd_ref.measurements, fwd_pert.measurements, J,
        method="tikhonov", alpha=0.01, normalise=False,
    )
    print(f"    Reconstruction range: [{delta_sigma.min():.4e}, "
          f"{delta_sigma.max():.4e}]")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="EIT demo with SCI head model"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to SCI head model dataset"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download dataset before running"
    )
    parser.add_argument(
        "--synthetic", action="store_true", default=False,
        help="Use synthetic spherical model instead of SCI data"
    )
    args = parser.parse_args()

    if args.data_dir and not args.synthetic:
        run_sci_demo(args.data_dir, download=args.download)
    else:
        run_synthetic_demo()


if __name__ == "__main__":
    main()
