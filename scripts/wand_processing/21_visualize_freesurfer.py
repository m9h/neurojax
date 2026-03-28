#!/usr/bin/env python3
"""Visualize FreeSurfer recon-all output — 6-view cortical surface figures.

Generates publication-quality figures using surfplot:
  1. Cortical thickness (6-view: L/R lateral, L/R medial, dorsal, ventral)
  2. Curvature (sulcal depth)
  3. Parcellation (Desikan-Killiany + Destrieux overlays)
  4. WM/GM contrast (pct surface contrast)
  5. T1w/T2w myelin ratio (if available)

Also generates:
  6. Subcortical volume bar chart
  7. QC metrics summary figure

Usage:
    python 21_visualize_freesurfer.py sub-08033
    python 21_visualize_freesurfer.py sub-08033 --output-dir /path/to/figures
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# FreeSurfer and surface tools
import nibabel as nib
import nibabel.freesurfer as fs

try:
    from surfplot import Plot
    from brainspace.mesh.mesh_io import read_surface as read_surf_bs
    HAS_SURFPLOT = True
except ImportError:
    HAS_SURFPLOT = False
    print("WARNING: surfplot not installed. Install with: pip install surfplot")


def get_args():
    p = argparse.ArgumentParser(description="Visualize FreeSurfer output")
    p.add_argument("subject", help="Subject ID (e.g., sub-08033)")
    p.add_argument("--subjects-dir", default=os.environ.get(
        "SUBJECTS_DIR", os.path.expanduser("~/dev/wand/derivatives/freesurfer")))
    p.add_argument("--output-dir", default=None,
                   help="Output directory for figures (default: derivatives/figures/<subject>)")
    p.add_argument("--wand-root", default=os.path.expanduser("~/dev/wand"))
    return p.parse_args()


def load_surface(subjects_dir, subject, hemi, surface="inflated"):
    """Load FreeSurfer surface as vertices + faces."""
    surf_path = os.path.join(subjects_dir, subject, "surf", f"{hemi}.{surface}")
    return fs.read_geometry(surf_path)


def load_overlay(subjects_dir, subject, hemi, overlay):
    """Load a FreeSurfer surface overlay (thickness, curv, etc.)."""
    path = os.path.join(subjects_dir, subject, "surf", f"{hemi}.{overlay}")
    if os.path.exists(path):
        return fs.read_morph_data(path)
    # Try .mgz
    path_mgz = path + ".mgz"
    if os.path.exists(path_mgz):
        return nib.load(path_mgz).get_fdata().squeeze()
    return None


def make_6view_figure(lh_surf, rh_surf, lh_data, rh_data, title, cmap,
                       vmin=None, vmax=None, output_path=None):
    """Create a 6-view surface figure using surfplot."""
    if not HAS_SURFPLOT:
        print(f"  Skipping {title} (surfplot not available)")
        return

    p = Plot(lh_surf, rh_surf,
             views=['lateral', 'medial', 'dorsal', 'ventral'],
             size=(1200, 900), zoom=1.2)

    if lh_data is not None and rh_data is not None:
        # Concatenate for both hemispheres
        data = np.concatenate([lh_data, rh_data])
        if vmin is None:
            vmin = np.percentile(data[data != 0], 2) if np.any(data != 0) else 0
        if vmax is None:
            vmax = np.percentile(data[data != 0], 98) if np.any(data != 0) else 1
        p.add_layer(data, cmap=cmap, color_range=(vmin, vmax))

    fig = p.build()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.close(fig)
    return fig


def make_subcortical_figure(subjects_dir, subject, output_path):
    """Bar chart of subcortical volumes from aseg.stats."""
    stats_file = os.path.join(subjects_dir, subject, "stats", "aseg.stats")
    if not os.path.exists(stats_file):
        print("  aseg.stats not found")
        return

    structures = {}
    with open(stats_file) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.split()
            if len(parts) >= 5:
                name = parts[4]
                volume = float(parts[3])
                # Filter to subcortical structures of interest
                if any(s in name for s in [
                    'Thalamus', 'Caudate', 'Putamen', 'Pallidum',
                    'Hippocampus', 'Amygdala', 'Accumbens'
                ]):
                    structures[name] = volume

    if not structures:
        print("  No subcortical structures found")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    names = list(structures.keys())
    volumes = list(structures.values())
    colors = ['#e74c3c' if 'Left' in n else '#3498db' for n in names]
    bars = ax.barh(range(len(names)), volumes, color=colors, edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace('Left-', 'L ').replace('Right-', 'R ') for n in names],
                        fontsize=9)
    ax.set_xlabel('Volume (mm³)', fontsize=11)
    ax.set_title(f'Subcortical Volumes — {subject}', fontsize=13)
    ax.invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch
    ax.legend([Patch(color='#e74c3c'), Patch(color='#3498db')],
              ['Left', 'Right'], loc='lower right')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def make_qc_summary_figure(subjects_dir, subject, output_path):
    """QC metrics dashboard."""
    # Collect metrics from qatools output or recompute
    metrics = {}

    # Read from qatools CSV if available
    qc_dir = os.path.join(os.path.dirname(subjects_dir), "qc", subject, "freesurfer")
    csv_files = list(Path(qc_dir).glob("*.csv")) if os.path.exists(qc_dir) else []

    # Read aseg stats for volumes
    stats_file = os.path.join(subjects_dir, subject, "stats", "aseg.stats")
    if os.path.exists(stats_file):
        with open(stats_file) as f:
            for line in f:
                if line.startswith("# Measure"):
                    parts = line.strip().split(",")
                    if len(parts) >= 4:
                        metrics[parts[1].strip()] = parts[3].strip()

    # Read Euler number
    euler_lh = euler_rh = "?"
    for hemi in ["lh", "rh"]:
        orig_path = os.path.join(subjects_dir, subject, "surf", f"{hemi}.orig.nofix")
        if os.path.exists(orig_path):
            v, f = fs.read_geometry(orig_path)
            # Euler = V - E + F = 2 - 2*holes
            n_v, n_f = len(v), len(f)
            n_e = n_f * 3 // 2  # for closed triangulated surface
            euler = n_v - n_e + n_f
            if hemi == "lh":
                euler_lh = euler
            else:
                euler_rh = euler

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Key volumes
    vol_names = ['BrainSeg', 'CortexVol', 'SubCortGrayVol', 'CerebralWhiteMatterVol']
    vol_values = [float(metrics.get(n, 0)) for n in vol_names]
    axes[0].barh(vol_names, vol_values, color='steelblue')
    axes[0].set_xlabel('Volume (mm³)')
    axes[0].set_title('Brain Volumes')

    # Panel 2: Surface metrics
    surf_metrics = {
        'Euler LH': euler_lh,
        'Euler RH': euler_rh,
        'eTIV': float(metrics.get('EstimatedTotalIntraCranialVol', 0)) / 1e6,
    }
    axes[1].text(0.5, 0.5, '\n'.join([f'{k}: {v}' for k, v in surf_metrics.items()]),
                 transform=axes[1].transAxes, ha='center', va='center', fontsize=14,
                 family='monospace')
    axes[1].set_title('Surface QC')
    axes[1].axis('off')

    # Panel 3: Thickness distribution
    for hemi, color in [('lh', '#e74c3c'), ('rh', '#3498db')]:
        thickness = load_overlay(subjects_dir, subject, hemi, 'thickness')
        if thickness is not None:
            thickness = thickness[thickness > 0]
            axes[2].hist(thickness, bins=50, alpha=0.6, color=color, label=hemi.upper())
    axes[2].set_xlabel('Thickness (mm)')
    axes[2].set_ylabel('Vertex count')
    axes[2].set_title('Cortical Thickness Distribution')
    axes[2].legend()

    plt.suptitle(f'FreeSurfer QC — {subject}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def main():
    args = get_args()
    subject = args.subject
    subjects_dir = args.subjects_dir

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(args.wand_root, "derivatives", "figures", subject)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== FreeSurfer Visualization: {subject} ===")
    print(f"  Subjects dir: {subjects_dir}")
    print(f"  Output: {out_dir}")

    # Load surfaces (inflated for visualization)
    print("\nLoading surfaces...")
    lh_inf_path = os.path.join(subjects_dir, subject, "surf", "lh.inflated")
    rh_inf_path = os.path.join(subjects_dir, subject, "surf", "rh.inflated")

    if not os.path.exists(lh_inf_path):
        print(f"ERROR: {lh_inf_path} not found. recon-all may not be complete.")
        sys.exit(1)

    # For surfplot, use brainspace mesh format
    if HAS_SURFPLOT:
        lh_surf = lh_inf_path
        rh_surf = rh_inf_path
    else:
        lh_surf = rh_surf = None

    # --- 1. Cortical Thickness ---
    print("\n1. Cortical Thickness (6-view)")
    lh_thick = load_overlay(subjects_dir, subject, "lh", "thickness")
    rh_thick = load_overlay(subjects_dir, subject, "rh", "thickness")
    make_6view_figure(lh_surf, rh_surf, lh_thick, rh_thick,
                       "Cortical Thickness", "YlOrRd",
                       vmin=1.0, vmax=4.5,
                       output_path=os.path.join(out_dir, "thickness_6view.png"))

    # --- 2. Curvature ---
    print("2. Curvature (6-view)")
    lh_curv = load_overlay(subjects_dir, subject, "lh", "curv")
    rh_curv = load_overlay(subjects_dir, subject, "rh", "curv")
    make_6view_figure(lh_surf, rh_surf, lh_curv, rh_curv,
                       "Curvature", "coolwarm",
                       vmin=-0.5, vmax=0.5,
                       output_path=os.path.join(out_dir, "curvature_6view.png"))

    # --- 3. Sulcal Depth ---
    print("3. Sulcal Depth (6-view)")
    lh_sulc = load_overlay(subjects_dir, subject, "lh", "sulc")
    rh_sulc = load_overlay(subjects_dir, subject, "rh", "sulc")
    make_6view_figure(lh_surf, rh_surf, lh_sulc, rh_sulc,
                       "Sulcal Depth", "RdBu_r",
                       output_path=os.path.join(out_dir, "sulcal_depth_6view.png"))

    # --- 4. Area ---
    print("4. Surface Area (6-view)")
    lh_area = load_overlay(subjects_dir, subject, "lh", "area")
    rh_area = load_overlay(subjects_dir, subject, "rh", "area")
    make_6view_figure(lh_surf, rh_surf, lh_area, rh_area,
                       "Surface Area", "viridis",
                       output_path=os.path.join(out_dir, "area_6view.png"))

    # --- 5. T1w/T2w Myelin (if available) ---
    myelin_dir = os.path.join(args.wand_root, "derivatives", "advanced-freesurfer",
                                subject, "myelin")
    lh_myelin_path = os.path.join(myelin_dir, "lh.T1wT2w_ratio.mgz")
    if os.path.exists(lh_myelin_path):
        print("5. T1w/T2w Myelin Ratio (6-view)")
        lh_myelin = nib.load(lh_myelin_path).get_fdata().squeeze()
        rh_myelin_path = os.path.join(myelin_dir, "rh.T1wT2w_ratio.mgz")
        rh_myelin = nib.load(rh_myelin_path).get_fdata().squeeze() if os.path.exists(rh_myelin_path) else None
        make_6view_figure(lh_surf, rh_surf, lh_myelin, rh_myelin,
                           "T1w/T2w Myelin Ratio", "magma",
                           output_path=os.path.join(out_dir, "myelin_6view.png"))
    else:
        print("5. T1w/T2w Myelin — not available (run 08_advanced_freesurfer.sh)")

    # --- 6. Subcortical Volumes ---
    print("6. Subcortical Volumes")
    make_subcortical_figure(subjects_dir, subject,
                             os.path.join(out_dir, "subcortical_volumes.png"))

    # --- 7. QC Summary ---
    print("7. QC Summary Dashboard")
    make_qc_summary_figure(subjects_dir, subject,
                            os.path.join(out_dir, "qc_summary.png"))

    print(f"\n=== Done. Figures in: {out_dir} ===")


if __name__ == "__main__":
    main()
