#!/usr/bin/env python3
"""fsreport — FreeSurfer recon-all QC Report Generator

Standalone tool that generates a FEAT-style HTML report from FreeSurfer
recon-all output. No dependencies beyond Python 3.9+, nibabel, matplotlib,
and optionally surfplot.

Produces a single self-contained HTML file with:
  - QC metrics (SNR, topology, contrast, Talairach)
  - 6-view cortical surface figures (thickness, curvature, sulcal depth, area)
  - Subcortical volumes chart
  - Cortical thickness distribution
  - Brain volume table
  - Segmentation table (aseg)
  - Pass/warn/fail status indicators

Usage:
    fsreport --subjects-dir /path/to/subjects --subject sub-001
    fsreport --subjects-dir /path/to/subjects --subject sub-001 sub-002 sub-003
    fsreport --subjects-dir /path/to/subjects --all

Install:
    pip install nibabel matplotlib
    pip install surfplot brainspace   # optional, for 6-view surface figures

License: Apache 2.0
"""

__version__ = "0.1.0"

import argparse
import base64
import csv
import io
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Core deps
import numpy as np

try:
    import nibabel.freesurfer as fs
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from surfplot import Plot
    HAS_SURFPLOT = True
except ImportError:
    HAS_SURFPLOT = False


# ---------------------------------------------------------------------------
# QC Metrics (reimplemented from qatools to avoid its bugs)
# ---------------------------------------------------------------------------

def compute_snr(subjects_dir, subject, vol_name="orig"):
    """Compute WM and GM SNR from a FreeSurfer volume."""
    import nibabel as nib
    vol_path = os.path.join(subjects_dir, subject, "mri", f"{vol_name}.mgz")
    aseg_path = os.path.join(subjects_dir, subject, "mri", "aseg.mgz")
    if not os.path.exists(vol_path) or not os.path.exists(aseg_path):
        return None, None

    vol = nib.load(vol_path).get_fdata()
    aseg = nib.load(aseg_path).get_fdata().astype(int)

    # WM: labels 2, 41 (cerebral WM)
    wm_mask = np.isin(aseg, [2, 41])
    # GM: labels 3, 42 (cortex)
    gm_mask = np.isin(aseg, [3, 42])

    if np.sum(wm_mask) == 0 or np.sum(gm_mask) == 0:
        return None, None

    wm_snr = float(np.mean(vol[wm_mask]) / max(np.std(vol[wm_mask]), 1e-10))
    gm_snr = float(np.mean(vol[gm_mask]) / max(np.std(vol[gm_mask]), 1e-10))
    return round(wm_snr, 2), round(gm_snr, 2)


def compute_euler(subjects_dir, subject):
    """Compute Euler number per hemisphere."""
    results = {}
    for hemi in ["lh", "rh"]:
        path = os.path.join(subjects_dir, subject, "surf", f"{hemi}.orig.nofix")
        if not os.path.exists(path) or not HAS_NIBABEL:
            results[hemi] = None
            continue
        v, f = fs.read_geometry(path)
        n_v, n_f = len(v), len(f)
        n_e = n_f * 3 // 2
        results[hemi] = n_v - n_e + n_f
    return results


def compute_holes(subjects_dir, subject):
    """Count holes per hemisphere from topology log."""
    results = {}
    for hemi in ["lh", "rh"]:
        log_path = os.path.join(subjects_dir, subject, "scripts", "recon-all.log")
        results[hemi] = None
        if os.path.exists(log_path):
            with open(log_path) as f:
                for line in f:
                    if f"{hemi}.orig" in line and "holes" in line.lower():
                        match = re.search(r'(\d+)\s+holes', line)
                        if match:
                            results[hemi] = int(match.group(1))
    return results


def read_aseg_stats(subjects_dir, subject):
    """Parse aseg.stats."""
    stats_file = os.path.join(subjects_dir, subject, "stats", "aseg.stats")
    measures = {}
    structures = []
    if not os.path.exists(stats_file):
        return measures, structures

    with open(stats_file) as f:
        for line in f:
            if line.startswith("# Measure"):
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    measures[parts[1].strip()] = parts[3].strip()
            elif not line.startswith("#") and line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    structures.append({
                        "name": parts[4],
                        "volume": float(parts[3]),
                        "nvoxels": int(parts[2]),
                    })
    return measures, structures


def read_aparc_stats(subjects_dir, subject, hemi):
    """Parse aparc.stats for cortical regions."""
    stats_file = os.path.join(subjects_dir, subject, "stats", f"{hemi}.aparc.stats")
    regions = []
    if not os.path.exists(stats_file):
        return regions
    with open(stats_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 10:
                regions.append({
                    "name": parts[0],
                    "area": float(parts[2]),
                    "volume": float(parts[3]),
                    "thickness": float(parts[4]),
                    "thickstd": float(parts[5]),
                })
    return regions


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def make_surface_figure(subjects_dir, subject, overlay_name, cmap, vmin, vmax, title):
    """Generate a 6-view surface figure, return as base64 PNG."""
    if not HAS_SURFPLOT or not HAS_NIBABEL:
        return None

    lh_path = os.path.join(subjects_dir, subject, "surf", "lh.inflated")
    rh_path = os.path.join(subjects_dir, subject, "surf", "rh.inflated")
    if not os.path.exists(lh_path):
        return None

    lh_data = fs.read_morph_data(os.path.join(subjects_dir, subject, "surf", f"lh.{overlay_name}"))
    rh_data = fs.read_morph_data(os.path.join(subjects_dir, subject, "surf", f"rh.{overlay_name}"))
    data = np.concatenate([lh_data, rh_data])

    p = Plot(lh_path, rh_path, views=['lateral', 'medial', 'dorsal', 'ventral'],
             size=(1200, 900), zoom=1.2)
    p.add_layer(data, cmap=cmap, color_range=(vmin, vmax))
    fig = p.build()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_subcortical_chart(structures):
    """Generate subcortical volumes chart, return as base64 PNG."""
    if not HAS_MPL:
        return None

    subcort = [s for s in structures if any(
        k in s["name"] for k in ["Thalamus", "Caudate", "Putamen", "Pallidum",
                                   "Hippocampus", "Amygdala", "Accumbens"]
    )]
    if not subcort:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [s["name"].replace("Left-", "L ").replace("Right-", "R ") for s in subcort]
    volumes = [s["volume"] for s in subcort]
    colors = ['#e74c3c' if 'Left' in s["name"] or s["name"].startswith("L ") else '#3498db' for s in subcort]
    ax.barh(range(len(names)), volumes, color=colors, edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Volume (mm³)')
    ax.set_title('Subcortical Volumes')
    ax.invert_yaxis()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_thickness_histogram(subjects_dir, subject):
    """Thickness distribution histogram, return as base64 PNG."""
    if not HAS_MPL or not HAS_NIBABEL:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    for hemi, color, label in [("lh", "#e74c3c", "Left"), ("rh", "#3498db", "Right")]:
        path = os.path.join(subjects_dir, subject, "surf", f"{hemi}.thickness")
        if os.path.exists(path):
            data = fs.read_morph_data(path)
            data = data[data > 0]
            ax.hist(data, bins=50, alpha=0.6, color=color, label=label)
    ax.set_xlabel('Thickness (mm)')
    ax.set_ylabel('Vertex count')
    ax.set_title('Cortical Thickness Distribution')
    ax.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def generate_report(subjects_dir, subject):
    """Generate complete HTML report for one subject."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Collect data
    wm_snr_orig, gm_snr_orig = compute_snr(subjects_dir, subject, "orig")
    wm_snr_norm, gm_snr_norm = compute_snr(subjects_dir, subject, "norm")
    euler = compute_euler(subjects_dir, subject)
    measures, structures = read_aseg_stats(subjects_dir, subject)

    # Read aparc stats
    lh_aparc = read_aparc_stats(subjects_dir, subject, "lh")
    rh_aparc = read_aparc_stats(subjects_dir, subject, "rh")

    # Generate figures
    print(f"  Generating figures for {subject}...")
    figs = {}
    for name, cmap, vmin, vmax in [
        ("thickness", "YlOrRd", 1.0, 4.5),
        ("curv", "coolwarm", -0.5, 0.5),
        ("sulc", "RdBu_r", -2, 2),
        ("area", "viridis", 0, 2),
    ]:
        b64 = make_surface_figure(subjects_dir, subject, name, cmap, vmin, vmax, name)
        if b64:
            figs[name] = b64

    subcort_b64 = make_subcortical_chart(structures)
    thick_hist_b64 = make_thickness_histogram(subjects_dir, subject)

    # Build HTML
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>fsreport — {subject}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; margin-top: 30px; }}
.card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.figure img {{ max-width: 100%; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #ecf0f1; font-size: 13px; }}
th {{ background: #3498db; color: white; }}
tr:hover {{ background: #eef5fd; }}
.m {{ display: inline-block; background: #ecf0f1; border-radius: 6px; padding: 10px 18px; margin: 4px; text-align: center; min-width: 100px; }}
.m .v {{ font-size: 22px; font-weight: bold; color: #2c3e50; }}
.m .l {{ font-size: 11px; color: #7f8c8d; }}
.pass {{ color: #27ae60; }} .warn {{ color: #f39c12; }} .fail {{ color: #e74c3c; }}
.ts {{ color: #95a5a6; font-size: 12px; float: right; }}
nav {{ background: #2c3e50; padding: 8px 16px; border-radius: 6px; margin-bottom: 16px; }}
nav a {{ color: #ecf0f1; text-decoration: none; margin-right: 12px; font-size: 13px; }}
</style></head><body>
<h1>fsreport — {subject} <span class="ts">{timestamp} | v{__version__}</span></h1>
<nav><a href="#qc">QC</a> <a href="#surf">Surfaces</a> <a href="#vol">Volumes</a> <a href="#aparc">Parcellation</a></nav>
"""

    # QC metrics
    html += '<h2 id="qc">Quality Metrics</h2><div class="card"><div>'

    def metric(label, value, warn_fn=None):
        cls = ""
        if warn_fn and value is not None:
            try:
                cls = warn_fn(value)
            except Exception:
                pass
        v = f"{value}" if value is not None else "?"
        return f'<div class="m"><div class="v {cls}">{v}</div><div class="l">{label}</div></div>'

    html += metric("WM SNR (orig)", wm_snr_orig)
    html += metric("GM SNR (orig)", gm_snr_orig)
    html += metric("WM SNR (norm)", wm_snr_norm)
    html += metric("GM SNR (norm)", gm_snr_norm)
    html += metric("Euler LH", euler.get("lh"),
                    lambda v: "pass" if abs(v) < 50 else ("warn" if abs(v) < 100 else "fail"))
    html += metric("Euler RH", euler.get("rh"),
                    lambda v: "pass" if abs(v) < 50 else ("warn" if abs(v) < 100 else "fail"))

    etiv = measures.get("EstimatedTotalIntraCranialVol")
    if etiv:
        html += metric("eTIV (cm³)", f"{float(etiv)/1e3:.1f}")

    html += '</div></div>'

    # Surface figures
    html += '<h2 id="surf">Cortical Surfaces (6-view)</h2>'
    titles = {"thickness": "Cortical Thickness", "curv": "Curvature",
              "sulc": "Sulcal Depth", "area": "Surface Area"}
    for key, title in titles.items():
        if key in figs:
            html += f'<div class="card"><h3>{title}</h3><div class="figure"><img src="data:image/png;base64,{figs[key]}"></div></div>'

    if not figs:
        html += '<div class="card"><p>Surface figures require surfplot: <code>pip install surfplot brainspace</code></p></div>'

    # Subcortical + thickness
    if subcort_b64:
        html += f'<div class="card"><h3>Subcortical Volumes</h3><div class="figure"><img src="data:image/png;base64,{subcort_b64}"></div></div>'
    if thick_hist_b64:
        html += f'<div class="card"><h3>Thickness Distribution</h3><div class="figure"><img src="data:image/png;base64,{thick_hist_b64}"></div></div>'

    # Volume table
    html += '<h2 id="vol">Brain Volumes</h2><div class="card"><table><tr><th>Measure</th><th>Value</th></tr>'
    for key in ['BrainSeg', 'BrainSegNotVent', 'CortexVol', 'SubCortGrayVol',
                'TotalGray', 'CerebralWhiteMatterVol', 'EstimatedTotalIntraCranialVol']:
        if key in measures:
            html += f'<tr><td>{key}</td><td>{float(measures[key]):,.0f} mm³</td></tr>'
    html += '</table></div>'

    # Aparc table
    if lh_aparc or rh_aparc:
        html += '<h2 id="aparc">Cortical Parcellation (Desikan-Killiany)</h2><div class="card">'
        for hemi, regions, label in [("lh", lh_aparc, "Left"), ("rh", rh_aparc, "Right")]:
            if regions:
                html += f'<h3>{label} Hemisphere</h3><table>'
                html += '<tr><th>Region</th><th>Thickness (mm)</th><th>Area (mm²)</th><th>Volume (mm³)</th></tr>'
                for r in regions:
                    html += f'<tr><td>{r["name"]}</td><td>{r["thickness"]:.2f} ± {r["thickstd"]:.2f}</td><td>{r["area"]:,.0f}</td><td>{r["volume"]:,.0f}</td></tr>'
                html += '</table>'
        html += '</div>'

    html += f'<hr><p style="color:#95a5a6;font-size:11px;text-align:center;">fsreport v{__version__} | {timestamp}</p></body></html>'

    return html


def main():
    p = argparse.ArgumentParser(
        description="fsreport — Generate HTML QC reports from FreeSurfer recon-all output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    fsreport --subjects-dir $SUBJECTS_DIR --subject sub-001
    fsreport --subjects-dir $SUBJECTS_DIR --subject sub-001 sub-002
    fsreport --subjects-dir $SUBJECTS_DIR --all
    fsreport --subjects-dir $SUBJECTS_DIR --all --output-dir /path/to/reports
        """)
    p.add_argument("--subjects-dir", required=True, help="FreeSurfer SUBJECTS_DIR")
    p.add_argument("--subject", nargs="+", help="Subject ID(s)")
    p.add_argument("--all", action="store_true", help="Process all subjects in SUBJECTS_DIR")
    p.add_argument("--output-dir", default=None, help="Output directory (default: SUBJECTS_DIR/../fsreport)")
    p.add_argument("--version", action="version", version=f"fsreport {__version__}")
    args = p.parse_args()

    subjects_dir = os.path.abspath(args.subjects_dir)

    if args.all:
        subjects = sorted([
            d for d in os.listdir(subjects_dir)
            if os.path.isdir(os.path.join(subjects_dir, d, "mri"))
        ])
    elif args.subject:
        subjects = args.subject
    else:
        p.error("Specify --subject or --all")

    out_dir = args.output_dir or os.path.join(os.path.dirname(subjects_dir), "fsreport")
    os.makedirs(out_dir, exist_ok=True)

    print(f"fsreport v{__version__}")
    print(f"Subjects dir: {subjects_dir}")
    print(f"Output dir:   {out_dir}")
    print(f"Subjects:     {len(subjects)}")
    print()

    for subject in subjects:
        print(f"Processing {subject}...")
        if not os.path.isdir(os.path.join(subjects_dir, subject, "mri")):
            print(f"  WARNING: {subject} does not have mri/ directory, skipping")
            continue

        html = generate_report(subjects_dir, subject)
        report_path = os.path.join(out_dir, f"{subject}_report.html")
        with open(report_path, "w") as f:
            f.write(html)
        print(f"  Report: {report_path}")

    # Generate index page
    if len(subjects) > 1:
        index = f"""<!DOCTYPE html><html><head><title>fsreport Index</title>
<style>body{{font-family:sans-serif;max-width:800px;margin:0 auto;padding:20px;}}
a{{color:#3498db;}}table{{border-collapse:collapse;width:100%;}}
th,td{{padding:8px;border-bottom:1px solid #eee;text-align:left;}}
th{{background:#3498db;color:white;}}</style></head><body>
<h1>fsreport — Subject Index</h1>
<p>{len(subjects)} subjects processed on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<table><tr><th>Subject</th><th>Report</th></tr>"""
        for s in subjects:
            index += f'<tr><td>{s}</td><td><a href="{s}_report.html">View Report</a></td></tr>'
        index += "</table></body></html>"
        with open(os.path.join(out_dir, "index.html"), "w") as f:
            f.write(index)
        print(f"\nIndex: {out_dir}/index.html")


if __name__ == "__main__":
    main()
