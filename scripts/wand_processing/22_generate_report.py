#!/usr/bin/env python3
"""Generate an HTML QC report for WAND processing — FEAT-style.

Combines all QC images and statistics into a single browsable page:
  - FreeSurfer surface figures (6-view thickness, curvature, etc.)
  - Subcortical volumes chart
  - QC metrics table
  - qatools statistics
  - Links to FEAT/MELODIC reports when available

Usage:
    python 22_generate_report.py sub-08033
"""

import argparse
import base64
import json
import os
from pathlib import Path
from datetime import datetime


def img_to_base64(path):
    """Embed image as base64 for self-contained HTML."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = Path(path).suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "svg": "image/svg+xml"}.get(ext.strip("."), "image/png")
    return f"data:{mime};base64,{data}"


def read_qatools_csv(qc_dir):
    """Read qatools CSV output."""
    import csv
    for f in Path(qc_dir).glob("*.csv"):
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                return dict(row)
    return {}


def read_aseg_stats(subjects_dir, subject):
    """Parse aseg.stats for key volumes."""
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
                        "index": parts[0],
                        "segid": parts[1],
                        "nvoxels": parts[2],
                        "volume_mm3": parts[3],
                        "name": parts[4],
                    })
    return measures, structures


def generate_html(subject, subjects_dir, wand_root):
    figures_dir = os.path.join(wand_root, "derivatives", "figures", subject)
    qc_dir = os.path.join(wand_root, "derivatives", "qc", subject, "freesurfer")
    fmri_dir = os.path.join(wand_root, "derivatives", "fsl-fmri", subject)

    measures, structures = read_aseg_stats(subjects_dir, subject)
    qatools = read_qatools_csv(qc_dir)

    # Collect all figure paths
    figures = {}
    for name in ["thickness_6view", "curvature_6view", "sulcal_depth_6view",
                  "area_6view", "myelin_6view", "subcortical_volumes", "qc_summary"]:
        path = os.path.join(figures_dir, f"{name}.png")
        if os.path.exists(path):
            figures[name] = img_to_base64(path)

    # Find FEAT/MELODIC reports
    feat_reports = []
    if os.path.isdir(fmri_dir):
        for report in Path(fmri_dir).rglob("report.html"):
            feat_reports.append(str(report))
        for report in Path(fmri_dir).rglob("00index.html"):
            feat_reports.append(str(report))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>WAND QC Report — {subject}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }}
h3 {{ color: #7f8c8d; }}
.card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0;
         box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.figure {{ text-align: center; margin: 10px 0; }}
.figure img {{ max-width: 100%; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #ecf0f1; }}
th {{ background: #3498db; color: white; font-weight: 600; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
tr:hover {{ background: #eef5fd; }}
.metric {{ display: inline-block; background: #ecf0f1; border-radius: 6px;
           padding: 12px 20px; margin: 5px; text-align: center; min-width: 120px; }}
.metric .value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
.metric .label {{ font-size: 12px; color: #7f8c8d; }}
.pass {{ color: #27ae60; font-weight: bold; }}
.warn {{ color: #f39c12; font-weight: bold; }}
.fail {{ color: #e74c3c; font-weight: bold; }}
.nav {{ background: #2c3e50; padding: 10px 20px; border-radius: 8px; margin-bottom: 20px; }}
.nav a {{ color: #ecf0f1; text-decoration: none; margin-right: 15px; font-size: 14px; }}
.nav a:hover {{ color: #3498db; }}
.timestamp {{ color: #95a5a6; font-size: 12px; float: right; }}
</style>
</head>
<body>

<h1>🧠 WAND QC Report — {subject}
<span class="timestamp">Generated: {timestamp}</span>
</h1>

<div class="nav">
  <a href="#metrics">QC Metrics</a>
  <a href="#surfaces">Surface Views</a>
  <a href="#volumes">Volumes</a>
  <a href="#subcortical">Subcortical</a>
  <a href="#qcsummary">QC Summary</a>
  <a href="#reports">FEAT/MELODIC Reports</a>
</div>
"""

    # --- QC Metrics ---
    html += '<h2 id="metrics">QC Metrics (qatools)</h2>\n<div class="card">\n'

    if qatools:
        html += '<div style="display: flex; flex-wrap: wrap;">\n'
        metric_display = {
            "wm_snr_orig": ("WM SNR (orig)", None),
            "gm_snr_orig": ("GM SNR (orig)", None),
            "wm_snr_norm": ("WM SNR (norm)", None),
            "gm_snr_norm": ("GM SNR (norm)", None),
            "cc_size": ("CC Size", None),
            "holes_lh": ("Holes LH", lambda v: "pass" if float(v) < 30 else "warn"),
            "holes_rh": ("Holes RH", lambda v: "pass" if float(v) < 30 else "warn"),
            "defects_lh": ("Defects LH", lambda v: "pass" if float(v) < 30 else "warn"),
            "defects_rh": ("Defects RH", lambda v: "pass" if float(v) < 30 else "warn"),
            "con_snr_lh": ("Contrast SNR LH", None),
            "con_snr_rh": ("Contrast SNR RH", None),
        }
        for key, (label, check) in metric_display.items():
            val = qatools.get(key, "?")
            status_class = ""
            if check and val != "?":
                try:
                    status_class = check(val)
                except (ValueError, TypeError):
                    pass
            html += f'<div class="metric"><div class="value {status_class}">{val}</div><div class="label">{label}</div></div>\n'
        html += '</div>\n'
    else:
        html += '<p>qatools metrics not available. Run: <code>qatools.py --subjects_dir ... --output_dir ... --subjects {subject}</code></p>\n'

    html += '</div>\n'

    # --- Brain Volumes ---
    if measures:
        html += '<h2 id="volumes">Brain Volumes</h2>\n<div class="card">\n<table>\n'
        html += '<tr><th>Measure</th><th>Value</th></tr>\n'
        for key in ['BrainSeg', 'BrainSegNotVent', 'CortexVol', 'SubCortGrayVol',
                     'TotalGray', 'CerebralWhiteMatterVol', 'EstimatedTotalIntraCranialVol',
                     'lhCortexVol', 'rhCortexVol', 'MaskVol']:
            if key in measures:
                val = measures[key]
                try:
                    val_fmt = f"{float(val):,.0f} mm³"
                except ValueError:
                    val_fmt = val
                html += f'<tr><td>{key}</td><td>{val_fmt}</td></tr>\n'
        html += '</table>\n</div>\n'

    # --- Surface Figures ---
    html += '<h2 id="surfaces">Cortical Surface Views</h2>\n'

    surface_figs = [
        ("thickness_6view", "Cortical Thickness", "Darker red = thicker cortex (1-4.5 mm)"),
        ("curvature_6view", "Curvature", "Sulci (blue) and gyri (red)"),
        ("sulcal_depth_6view", "Sulcal Depth", "Depth of sulcal folds"),
        ("area_6view", "Surface Area", "Local surface area per vertex"),
        ("myelin_6view", "T1w/T2w Myelin Ratio", "Higher = more myelinated"),
    ]

    for fig_key, title, desc in surface_figs:
        if fig_key in figures:
            html += f'''<div class="card">
<h3>{title}</h3>
<p style="color: #7f8c8d; font-size: 13px;">{desc}</p>
<div class="figure"><img src="{figures[fig_key]}" alt="{title}"></div>
</div>\n'''

    # --- Subcortical ---
    if "subcortical_volumes" in figures:
        html += f'''<h2 id="subcortical">Subcortical Volumes</h2>
<div class="card">
<div class="figure"><img src="{figures['subcortical_volumes']}" alt="Subcortical Volumes"></div>
</div>\n'''

    # --- QC Summary ---
    if "qc_summary" in figures:
        html += f'''<h2 id="qcsummary">QC Summary</h2>
<div class="card">
<div class="figure"><img src="{figures['qc_summary']}" alt="QC Summary"></div>
</div>\n'''

    # --- Segmentation Table ---
    if structures:
        html += '<h2>Segmentation Details (aseg)</h2>\n<div class="card">\n<table>\n'
        html += '<tr><th>Structure</th><th>Volume (mm³)</th><th>nVoxels</th></tr>\n'
        for s in structures[:40]:  # first 40 structures
            html += f'<tr><td>{s["name"]}</td><td>{float(s["volume_mm3"]):,.1f}</td><td>{s["nvoxels"]}</td></tr>\n'
        if len(structures) > 40:
            html += f'<tr><td colspan="3"><em>... and {len(structures)-40} more</em></td></tr>\n'
        html += '</table>\n</div>\n'

    # --- FEAT/MELODIC Reports ---
    html += '<h2 id="reports">FEAT / MELODIC Reports</h2>\n<div class="card">\n'
    if feat_reports:
        html += '<ul>\n'
        for r in feat_reports:
            html += f'<li><a href="file://{r}">{os.path.relpath(r, wand_root)}</a></li>\n'
        html += '</ul>\n'
    else:
        html += '<p>No FEAT/MELODIC reports found. Run <code>19_fmri_processing.sh</code> first.</p>\n'
    html += '</div>\n'

    # --- Footer ---
    html += f'''
<hr>
<p style="color: #95a5a6; font-size: 11px; text-align: center;">
  Generated by neurojax WAND processing pipeline | FreeSurfer 8.2.0 | surfplot | {timestamp}
</p>
</body>
</html>'''

    return html


def main():
    p = argparse.ArgumentParser(description="Generate WAND QC HTML report")
    p.add_argument("subject")
    p.add_argument("--subjects-dir", default=os.environ.get(
        "SUBJECTS_DIR", os.path.expanduser("~/dev/wand/derivatives/freesurfer")))
    p.add_argument("--wand-root", default=os.path.expanduser("~/dev/wand"))
    args = p.parse_args()

    html = generate_html(args.subject, args.subjects_dir, args.wand_root)

    out_dir = os.path.join(args.wand_root, "derivatives", "figures", args.subject)
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "report.html")

    with open(report_path, "w") as f:
        f.write(html)

    print(f"Report generated: {report_path}")
    print(f"Open with: open {report_path}")


if __name__ == "__main__":
    main()
