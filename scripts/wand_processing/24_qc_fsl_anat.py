#!/usr/bin/env python3
"""QC for fsl_anat output — generates HTML report with overlay images.

Checks:
  1. Brain extraction (BET) — brain mask overlaid on T1
  2. FAST tissue segmentation — GM/WM/CSF overlaid on T1
  3. FIRST subcortical segmentation — subcortical labels on T1
  4. FNIRT registration to MNI — T1 in MNI overlaid on template
  5. Bias correction quality — before vs after
  6. Volume statistics from T1_vols.txt

Usage:
    python 24_qc_fsl_anat.py path/to/T1w.anat
"""

import argparse
import base64
import io
import os
import sys
from datetime import datetime

import numpy as np

try:
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:
    print("Requires: pip install nibabel matplotlib")
    sys.exit(1)


def slice_overlay(bg, overlay, slices_ax=None, cmap='hot', alpha=0.5, title=""):
    """Create axial/coronal/sagittal overlay images."""
    if slices_ax is None:
        z_mid = bg.shape[2] // 2
        slices_ax = [z_mid - 15, z_mid - 5, z_mid, z_mid + 5, z_mid + 15]

    n = len(slices_ax)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]

    for i, sl in enumerate(slices_ax):
        sl = min(sl, bg.shape[2] - 1)
        axes[i].imshow(np.rot90(bg[:, :, sl]), cmap='gray', aspect='auto')
        if overlay is not None:
            ov = np.rot90(overlay[:, :, sl])
            masked = np.ma.masked_where(ov == 0, ov)
            axes[i].imshow(masked, cmap=cmap, alpha=alpha, aspect='auto')
        axes[i].axis('off')
        axes[i].set_title(f'z={sl}', fontsize=9)

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def coronal_overlay(bg, overlay, cmap='hot', alpha=0.5, title=""):
    """Coronal view overlay."""
    y_mid = bg.shape[1] // 2
    slices = [y_mid - 20, y_mid - 10, y_mid, y_mid + 10, y_mid + 20]
    n = len(slices)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))

    for i, sl in enumerate(slices):
        sl = min(sl, bg.shape[1] - 1)
        axes[i].imshow(np.rot90(bg[:, sl, :]), cmap='gray', aspect='auto')
        if overlay is not None:
            ov = np.rot90(overlay[:, sl, :])
            masked = np.ma.masked_where(ov == 0, ov)
            axes[i].imshow(masked, cmap=cmap, alpha=alpha, aspect='auto')
        axes[i].axis('off')
        axes[i].set_title(f'y={sl}', fontsize=9)

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def main():
    p = argparse.ArgumentParser(description="QC for fsl_anat output")
    p.add_argument("anat_dir", help="Path to .anat directory")
    p.add_argument("--output", default=None, help="Output HTML path")
    args = p.parse_args()

    anat_dir = args.anat_dir
    if not os.path.isdir(anat_dir):
        print(f"ERROR: {anat_dir} not found")
        sys.exit(1)

    subject = os.path.basename(anat_dir).replace('.anat', '')
    out_path = args.output or os.path.join(anat_dir, "qc_report.html")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"=== fsl_anat QC: {subject} ===")

    # Load volumes
    t1 = nib.load(os.path.join(anat_dir, "T1_biascorr.nii.gz")).get_fdata()
    t1_brain = nib.load(os.path.join(anat_dir, "T1_biascorr_brain.nii.gz")).get_fdata()
    mask = nib.load(os.path.join(anat_dir, "T1_biascorr_brain_mask.nii.gz")).get_fdata()

    # FAST segmentation
    seg = nib.load(os.path.join(anat_dir, "T1_fast_seg.nii.gz")).get_fdata()
    pve_gm = nib.load(os.path.join(anat_dir, "T1_fast_pve_1.nii.gz")).get_fdata()
    pve_wm = nib.load(os.path.join(anat_dir, "T1_fast_pve_2.nii.gz")).get_fdata()

    # Volume stats
    vols_file = os.path.join(anat_dir, "T1_vols.txt")
    vols = {}
    if os.path.exists(vols_file):
        with open(vols_file) as f:
            for line in f:
                if line.strip():
                    # Format varies; try key-value or just numbers
                    vols[f"line_{len(vols)}"] = line.strip()

    # Registration to MNI
    mni_path = os.path.join(anat_dir, "T1_to_MNI_nonlin.nii.gz")
    has_mni = os.path.exists(mni_path)

    # Generate QC images
    figs = {}
    print("  Generating overlay images...")

    # 1. Brain extraction
    figs['bet_axial'] = slice_overlay(t1, mask, cmap='winter', alpha=0.3,
                                        title="Brain Extraction (BET) — Axial")
    figs['bet_coronal'] = coronal_overlay(t1, mask, cmap='winter', alpha=0.3,
                                           title="Brain Extraction (BET) — Coronal")

    # 2. FAST tissue segmentation
    seg_cmap = ListedColormap(['black', 'blue', 'gray', 'white'])
    figs['fast_axial'] = slice_overlay(t1_brain, seg, cmap=seg_cmap, alpha=0.4,
                                        title="FAST Tissue Segmentation (CSF=blue, GM=gray, WM=white)")
    figs['fast_coronal'] = coronal_overlay(t1_brain, seg, cmap=seg_cmap, alpha=0.4,
                                            title="FAST Segmentation — Coronal")

    # 3. GM probability
    figs['gm_pve'] = slice_overlay(t1_brain, pve_gm, cmap='Reds', alpha=0.5,
                                    title="Grey Matter Probability (FAST PVE)")

    # 4. WM probability
    figs['wm_pve'] = slice_overlay(t1_brain, pve_wm, cmap='Blues', alpha=0.5,
                                    title="White Matter Probability (FAST PVE)")

    # 5. Subcortical (FIRST)
    subcort_path = os.path.join(anat_dir, "T1_subcort_seg.nii.gz")
    if os.path.exists(subcort_path):
        subcort = nib.load(subcort_path).get_fdata()
        figs['first'] = slice_overlay(t1_brain, subcort, cmap='nipy_spectral', alpha=0.5,
                                       title="FIRST Subcortical Segmentation")

    # 6. MNI registration
    if has_mni:
        mni_data = nib.load(mni_path).get_fdata()
        mni_template = os.path.join(os.environ.get('FSLDIR', '/usr/local/fsl'),
                                     'data', 'standard', 'MNI152_T1_2mm_brain.nii.gz')
        if os.path.exists(mni_template):
            template = nib.load(mni_template).get_fdata()
            # Edge overlay of T1 on template
            figs['mni_reg'] = slice_overlay(template, mni_data, cmap='hot', alpha=0.4,
                                             title="FNIRT Registration: T1 (hot) on MNI template")

    # 7. Bias correction (orig vs corrected)
    t1_orig = nib.load(os.path.join(anat_dir, "T1_orig.nii.gz")).get_fdata()
    bias = nib.load(os.path.join(anat_dir, "T1_fast_bias.nii.gz")).get_fdata()
    figs['bias'] = slice_overlay(t1_orig, bias, cmap='hot', alpha=0.4,
                                  title="Bias Field Estimate (FAST)")

    # Compute tissue volumes
    voxel_vol = np.abs(np.linalg.det(nib.load(
        os.path.join(anat_dir, "T1_biascorr.nii.gz")).affine[:3, :3]))
    gm_vol = float(pve_gm.sum() * voxel_vol) / 1000  # cm³
    wm_vol = float(pve_wm.sum() * voxel_vol) / 1000
    brain_vol = float(mask.sum() * voxel_vol) / 1000

    # Build HTML
    print("  Building HTML report...")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>fsl_anat QC — {subject}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
h2 {{ color: #34495e; margin-top: 25px; }}
.card {{ background: white; border-radius: 8px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.card img {{ max-width: 100%; }}
.m {{ display: inline-block; background: #ecf0f1; border-radius: 6px; padding: 10px 18px; margin: 4px; text-align: center; }}
.m .v {{ font-size: 20px; font-weight: bold; color: #2c3e50; }}
.m .l {{ font-size: 11px; color: #7f8c8d; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid #eee; text-align: left; }}
th {{ background: #e74c3c; color: white; }}
</style></head><body>
<h1>fsl_anat QC — {subject} <span style="color:#95a5a6;font-size:12px;float:right">{timestamp}</span></h1>

<div class="card">
<div class="m"><div class="v">{brain_vol:.0f}</div><div class="l">Brain (cm³)</div></div>
<div class="m"><div class="v">{gm_vol:.0f}</div><div class="l">GM (cm³)</div></div>
<div class="m"><div class="v">{wm_vol:.0f}</div><div class="l">WM (cm³)</div></div>
<div class="m"><div class="v">{gm_vol/(wm_vol+0.01):.2f}</div><div class="l">GM/WM Ratio</div></div>
</div>
"""

    titles = {
        'bet_axial': 'Brain Extraction — Axial',
        'bet_coronal': 'Brain Extraction — Coronal',
        'fast_axial': 'FAST Tissue Segmentation — Axial',
        'fast_coronal': 'FAST Tissue Segmentation — Coronal',
        'gm_pve': 'Grey Matter Probability',
        'wm_pve': 'White Matter Probability',
        'first': 'FIRST Subcortical Segmentation',
        'mni_reg': 'FNIRT MNI Registration',
        'bias': 'Bias Field',
    }

    for key, title in titles.items():
        if key in figs:
            html += f'<h2>{title}</h2>\n<div class="card"><img src="data:image/png;base64,{figs[key]}"></div>\n'

    html += f'<hr><p style="color:#95a5a6;font-size:11px;text-align:center;">fsl_anat QC | {timestamp}</p></body></html>'

    with open(out_path, 'w') as f:
        f.write(html)

    print(f"  Report: {out_path}")
    print(f"  Open: open {out_path}")


if __name__ == "__main__":
    main()
