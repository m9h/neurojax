#!/usr/bin/env python3
"""Quantitative MRI Report — Physics models, fit quality, and parameter maps.

Generates a comprehensive HTML report explaining:
  - What each qMRI acquisition measures physically
  - The signal model being fitted
  - Goodness-of-fit metrics
  - Parameter maps with physiological interpretation
  - Cross-parameter correlations
  - Comparison against literature values

Usage:
    python 25_qmri_report.py --subject sub-08033
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
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("Requires: pip install nibabel matplotlib")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Physics model descriptions (the educational/documentation component)
# ---------------------------------------------------------------------------

MODELS = {
    "T2star": {
        "title": "T2* Mapping (Multi-Echo Gradient Echo)",
        "acquisition": "7-echo MEGRE, TE = 5-35 ms (ses-06)",
        "equation": r"S(TE) = S₀ · exp(−TE / T2*)",
        "equation_html": "S(TE) = S₀ · exp(−TE / T2*)",
        "parameters": {
            "T2*": {
                "unit": "ms",
                "description": "Effective transverse relaxation time. Reflects both intrinsic T2 (spin-spin interactions) and magnetic field inhomogeneity (susceptibility effects from iron, myelin, air-tissue interfaces).",
                "physiology": "Short T2* → high iron or dense myelin (deep GM, basal ganglia). Long T2* → CSF, edema. WM shorter than GM due to myelin susceptibility.",
                "literature_3T": {"WM": "25-30 ms", "GM": "35-45 ms", "CSF": ">500 ms", "putamen": "15-20 ms"},
            },
            "S₀": {
                "unit": "a.u.",
                "description": "Signal at TE=0. Proportional to proton density × receive coil sensitivity.",
                "physiology": "Higher in tissue with more water (CSF > GM > WM).",
            },
            "R2*": {
                "unit": "Hz (s⁻¹)",
                "description": "R2* = 1/T2*. The transverse relaxation rate. Linearly related to tissue iron concentration in basal ganglia.",
                "physiology": "Used for iron quantification. Increases with age in caudate, putamen, globus pallidus. Sensitive to both paramagnetic (iron) and diamagnetic (myelin) susceptibility sources.",
            },
        },
        "fitting": "Log-linear least squares: ln(S) = ln(S₀) − TE/T2*. Simple, fast, analytically optimal for Gaussian noise in log domain.",
        "limitations": "Mono-exponential assumes single T2* per voxel. Reality: multiple compartments (myelin water has very short T2* ~10ms, free water ~40ms). Consider bi-exponential or multi-component fitting for more detail.",
    },
    "MP2RAGE": {
        "title": "MP2RAGE T1 Mapping",
        "acquisition": "Two inversion times (PSIR), TI₁ and TI₂ (ses-06)",
        "equation_html": "UNI = (INV₁ · INV₂) / (INV₁² + INV₂² + ε)",
        "parameters": {
            "T1": {
                "unit": "ms",
                "description": "Longitudinal relaxation time. Measures how quickly nuclear magnetization recovers after excitation. Determined by molecular environment (tumbling rate, macromolecular content).",
                "physiology": "Short T1 → more macromolecular content (myelin, iron). WM < GM < CSF. T1 is the gold-standard tissue contrast in structural MRI.",
                "literature_3T": {"WM": "800-900 ms", "GM": "1200-1400 ms", "CSF": "3500-4500 ms"},
            },
            "UNI": {
                "unit": "dimensionless [-0.5, 0.5]",
                "description": "The uniform (background-free) image. Ratio of the two inversion images that cancels B1+ transmit field inhomogeneity and receive coil sensitivity. Monotonically related to T1.",
                "physiology": "Self-calibrated T1 contrast without the intensity bias that plagues conventional T1w images. This is why MP2RAGE is preferred for quantitative studies.",
            },
        },
        "fitting": "Lookup table: compute UNI as function of T1 for the given sequence parameters (TI₁, TI₂, TR, α₁, α₂), then invert to find T1 from measured UNI value.",
        "limitations": "Assumes perfect inversion efficiency (typically ~96%). Sensitive to B0 inhomogeneity via off-resonance effects on inversion pulse. Two-point estimation means no goodness-of-fit metric available.",
    },
    "VFA_T1": {
        "title": "VFA T1 Mapping (DESPOT1)",
        "acquisition": "SPGR at multiple flip angles (ses-02)",
        "equation_html": "S(α) = M₀ · sin(α) · (1 − E₁) / (1 − E₁ · cos(α)), where E₁ = exp(−TR/T1)",
        "parameters": {
            "T1": {
                "unit": "ms",
                "description": "Longitudinal relaxation time from the steady-state SPGR signal. Same physical property as MP2RAGE T1 but estimated differently.",
                "physiology": "Same as MP2RAGE T1. VFA T1 is faster to acquire but more sensitive to B1+ inhomogeneity. MP2RAGE is self-calibrated.",
                "literature_3T": {"WM": "800-900 ms", "GM": "1200-1400 ms", "CSF": "3500-4500 ms"},
            },
            "M₀ (PD)": {
                "unit": "a.u.",
                "description": "Equilibrium magnetization. Proportional to proton density × T2*/TE effects × coil sensitivity.",
                "physiology": "PD reflects water content. CSF ≈ 100% water. GM > WM due to higher water content. Not absolute without calibration.",
            },
        },
        "fitting": "Linear DESPOT1: rearrange Ernst equation to y = E₁·x + M₀(1−E₁) where y = S/sin(α), x = S/tan(α). Slope gives E₁ → T1 = −TR/ln(E₁). Refined with nonlinear Gauss-Newton.",
        "limitations": "Highly sensitive to B1+ transmit field inhomogeneity. Actual flip angle ≠ nominal flip angle, especially at 3T. Requires B1+ map (AFI, DREAM, or Bloch-Siegert) for correction. Without B1 correction, T1 can be biased by 20-30%.",
    },
    "QMT": {
        "title": "Quantitative Magnetization Transfer (qMT)",
        "acquisition": "16 MT-weighted SPGR at varying offset frequencies and flip angles (ses-02)",
        "equation_html": "Two-pool model: Free water pool (observed) ⇌ Bound macromolecular pool (invisible). MT pulse saturates bound pool → reduces free pool signal via cross-relaxation.",
        "parameters": {
            "BPF (F)": {
                "unit": "fraction [0, 1]",
                "description": "Bound Pool Fraction — the fraction of protons in the macromolecular (bound) pool. This is the closest MRI measure to actual myelin content.",
                "physiology": "BPF ≈ 12-18% in WM (high myelin), 5-8% in GM, ~0% in CSF. Decreases in demyelinating diseases (MS, leukodystrophies). More specific to myelin than T1w/T2w ratio or MTR.",
                "literature_3T": {"WM": "12-18%", "GM": "5-8%", "CSF": "~0%"},
            },
            "k_f (exchange rate)": {
                "unit": "Hz (s⁻¹)",
                "description": "Forward exchange rate from free to bound pool. Reflects the rate of magnetization transfer between water and macromolecules.",
                "physiology": "Typically 2-5 Hz in WM. Reflects tissue microstructure beyond just myelin content — sensitive to membrane surface area and permeability.",
            },
            "T2_b": {
                "unit": "μs",
                "description": "T2 of the bound (macromolecular) pool. Very short (~10 μs) because macromolecules have restricted motion.",
                "physiology": "Difficult to measure precisely. Assumed or loosely constrained in most fits. Reflects the rigidity of the macromolecular environment.",
            },
        },
        "fitting": "Ramani/Sled-Pike two-pool model. Each MT-weighted volume samples a different point on the Z-spectrum (signal vs offset frequency). The 16 volumes at different offsets and flip angles overconstrain the 3-5 parameter model. Fitted with grid search initialization + Gauss-Newton refinement.",
        "limitations": "The two-pool model is a simplification — reality has multiple exchanging pools (myelin water, axonal water, extracellular water, macromolecular). The Lorentzian vs super-Lorentzian lineshape assumption affects BPF accuracy. Requires T1 map as input (from VFA or MP2RAGE).",
    },
    "mcDESPOT": {
        "title": "mcDESPOT — Myelin Water Fraction",
        "acquisition": "Combined VFA SPGR + SSFP at multiple flip angles (ses-02)",
        "equation_html": "S = f_fast · S_SPGR(T1_f, M₀·MWF) + (1−f_fast) · S_SPGR(T1_s, M₀·(1−MWF))<br>Same for SSFP with T2_fast and T2_slow pools",
        "parameters": {
            "MWF": {
                "unit": "fraction [0, 1]",
                "description": "Myelin Water Fraction — the fraction of water trapped between myelin bilayers. This water has short T1 (~300-500ms) and very short T2 (~10-20ms) due to proximity to myelin lipids.",
                "physiology": "MWF ≈ 10-20% in WM, 2-5% in GM, 0% in CSF. Directly related to myelin sheath integrity. Decreases in MS, aging, and neurodegenerative diseases.",
                "literature_3T": {"WM": "10-20%", "GM": "2-5%", "CSF": "0%"},
            },
            "T1_fast / T1_slow": {
                "unit": "ms",
                "description": "T1 of the fast (myelin water) and slow (intra/extra-cellular water) pools.",
                "physiology": "T1_fast ≈ 300-500ms (myelin water, short due to macromolecular environment). T1_slow ≈ 1000-1500ms (free water).",
            },
            "T2_fast / T2_slow": {
                "unit": "ms",
                "description": "T2 of the fast and slow pools.",
                "physiology": "T2_fast ≈ 10-20ms (myelin water). T2_slow ≈ 60-100ms (free water). The large T2 difference is what makes multi-component fitting possible.",
            },
        },
        "fitting": "Two-pool SPGR + SSFP signals summed. 6 free parameters make the landscape highly non-convex with many local minima. Traditional: stochastic region contraction. Our approach: grid search + Gauss-Newton (fast but may get trapped) or SBI amortized inference (neural posterior estimation, trained once, instant inference).",
        "limitations": "Notorious for local minima — the 6-parameter fit is ill-conditioned, especially at low SNR. Results are sensitive to B0 and B1 inhomogeneity. The accuracy of mcDESPOT MWF has been debated (Lankford & Does, 2013) — it may overestimate MWF compared to multi-echo T2 relaxometry.",
    },
}


def load_map(path):
    """Load a NIfTI map, return data + affine or None."""
    if os.path.exists(path):
        img = nib.load(path)
        return img.get_fdata(), img.affine
    return None, None


def map_figure(data, mask, title, unit, cmap, vmin, vmax, literature=None):
    """Generate multi-slice figure of a parameter map with histogram."""
    if data is None:
        return None

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, width_ratios=[2, 2, 1])

    # Axial slices
    ax1 = fig.add_subplot(gs[0])
    z_mid = data.shape[2] // 2
    slices = [z_mid - 10, z_mid, z_mid + 10]
    montage = np.concatenate([np.rot90(data[:, :, s]) for s in slices], axis=1)
    mask_montage = np.concatenate([np.rot90(mask[:, :, s]) for s in slices], axis=1) if mask is not None else montage > 0
    im = ax1.imshow(np.ma.masked_where(~mask_montage.astype(bool), montage),
                     cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title("Axial slices", fontsize=10)
    ax1.axis('off')
    plt.colorbar(im, ax=ax1, label=unit, shrink=0.7)

    # Coronal slices
    ax2 = fig.add_subplot(gs[1])
    y_mid = data.shape[1] // 2
    c_slices = [y_mid - 15, y_mid, y_mid + 15]
    c_montage = np.concatenate([np.rot90(data[:, s, :]) for s in c_slices], axis=1)
    c_mask = np.concatenate([np.rot90(mask[:, s, :]) for s in c_slices], axis=1) if mask is not None else c_montage > 0
    ax2.imshow(np.ma.masked_where(~c_mask.astype(bool), c_montage),
               cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title("Coronal slices", fontsize=10)
    ax2.axis('off')

    # Histogram with literature ranges
    ax3 = fig.add_subplot(gs[2])
    valid = data[mask > 0] if mask is not None else data[data > 0]
    valid = valid[(valid > vmin) & (valid < vmax)]
    if len(valid) > 0:
        ax3.hist(valid, bins=80, color='steelblue', alpha=0.7, density=True)
        ax3.set_xlabel(unit, fontsize=9)
        ax3.set_ylabel('Density', fontsize=9)
        ax3.set_title('Distribution', fontsize=10)

        # Add literature reference lines
        if literature:
            colors = {'WM': '#3498db', 'GM': '#e74c3c', 'CSF': '#2ecc71', 'putamen': '#9b59b6'}
            for tissue, val_str in literature.items():
                try:
                    # Parse range like "25-30 ms"
                    nums = [float(x) for x in val_str.replace('ms', '').replace('%', '').replace('~', '').split('-')]
                    mid = np.mean(nums)
                    ax3.axvline(mid, color=colors.get(tissue, 'gray'), linestyle='--',
                                label=f'{tissue}: {val_str}', linewidth=1.5)
                except (ValueError, TypeError):
                    pass
            ax3.legend(fontsize=7, loc='upper right')

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def main():
    p = argparse.ArgumentParser(description="qMRI relaxometry report")
    p.add_argument("--subject", required=True)
    p.add_argument("--wand-root", default=os.path.expanduser("~/dev/wand"))
    p.add_argument("--output", default=None)
    args = p.parse_args()

    subject = args.subject
    wand = args.wand_root
    qmri_02 = os.path.join(wand, "derivatives", "qmri", subject, "ses-02")
    qmri_06 = os.path.join(wand, "derivatives", "qmri", subject, "ses-06")
    out_path = args.output or os.path.join(wand, "derivatives", "qmri", subject, "qmri_report.html")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"=== qMRI Relaxometry Report: {subject} ===")

    # Discover available maps
    maps = {}
    map_paths = {
        "T2star": (os.path.join(qmri_06, "T2star_map.nii.gz"), "T2star"),
        "R2star": (os.path.join(qmri_06, "R2star_map.nii.gz"), "T2star"),
        "S0": (os.path.join(qmri_06, "S0_map.nii.gz"), "T2star"),
        "MP2RAGE_uni": (os.path.join(qmri_06, "MP2RAGE_uniform.nii.gz"), "MP2RAGE"),
        "T1_VFA": (os.path.join(qmri_02, "T1map.nii.gz"), "VFA_T1"),
        "QMT_bpf": (os.path.join(qmri_02, "QMT_bpf.nii.gz"), "QMT"),
        "MWF": (os.path.join(qmri_02, "MWF.nii.gz"), "mcDESPOT"),
    }

    for name, (path, model_key) in map_paths.items():
        data, affine = load_map(path)
        if data is not None:
            maps[name] = {"data": data, "model": model_key, "path": path}
            print(f"  Found: {name} ({path})")

    # Load mask
    mask_path = os.path.join(qmri_06, "megre_brain_mask.nii.gz")
    mask, _ = load_map(mask_path)

    # Generate figures
    figs = {}

    if "T2star" in maps:
        figs["T2star"] = map_figure(
            maps["T2star"]["data"] * 1000, mask,
            "T2* Map", "ms", "hot", 5, 80,
            MODELS["T2star"]["parameters"]["T2*"].get("literature_3T"))

    if "R2star" in maps:
        figs["R2star"] = map_figure(
            maps["R2star"]["data"], mask,
            "R2* Map", "Hz", "hot", 5, 100,
            MODELS["T2star"]["parameters"]["R2*"].get("literature_3T"))

    if "MP2RAGE_uni" in maps:
        figs["MP2RAGE"] = map_figure(
            maps["MP2RAGE_uni"]["data"], None,
            "MP2RAGE Uniform Image", "a.u.", "gray", -0.5, 0.5, None)

    # Build HTML
    print("  Building report...")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>qMRI Report — {subject}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; color: #2c3e50; }}
h1 {{ color: #8e44ad; border-bottom: 3px solid #8e44ad; padding-bottom: 10px; }}
h2 {{ color: #2c3e50; border-bottom: 1px solid #bdc3c7; margin-top: 30px; }}
h3 {{ color: #7f8c8d; }}
.card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.card img {{ max-width: 100%; border-radius: 4px; }}
.model-box {{ background: #f8f9fa; border-left: 4px solid #8e44ad; padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0; }}
.equation {{ font-family: 'Courier New', monospace; font-size: 16px; color: #2c3e50; background: #ecf0f1; padding: 10px; border-radius: 4px; margin: 8px 0; }}
.param {{ background: #eaf2f8; border-radius: 6px; padding: 12px; margin: 8px 0; }}
.param .name {{ font-weight: bold; color: #2980b9; }}
.param .unit {{ color: #7f8c8d; font-size: 12px; }}
.lit {{ color: #27ae60; font-size: 12px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 8px 12px; border-bottom: 1px solid #ecf0f1; text-align: left; }}
th {{ background: #8e44ad; color: white; }}
.warn {{ color: #f39c12; }} .good {{ color: #27ae60; }}
nav {{ background: #2c3e50; padding: 8px 16px; border-radius: 6px; margin-bottom: 16px; }}
nav a {{ color: #ecf0f1; text-decoration: none; margin-right: 12px; font-size: 13px; }}
</style></head><body>
<h1>Quantitative MRI Report — {subject}
<span style="color:#95a5a6;font-size:12px;float:right">{timestamp}</span></h1>
<nav>"""

    # Navigation
    for model_key in MODELS:
        html += f'<a href="#{model_key}">{MODELS[model_key]["title"].split("(")[0].strip()}</a>'
    html += '</nav>'

    # Introduction
    html += """<div class="card">
<h2>What is Quantitative MRI?</h2>
<p>Conventional MRI produces <em>weighted</em> images (T1w, T2w) where contrast depends on scanner settings,
coil sensitivity, and tissue properties mixed together. <strong>Quantitative MRI</strong> fits physics-based
signal models to extract the actual tissue properties (T1, T2*, myelin content) as numbers with units —
independent of scanner settings. These maps can be compared across subjects, sessions, and scanners.</p>
<p>Each section below explains the physics model, what was measured, how it was fitted, and what the
resulting parameter maps mean biologically.</p>
</div>"""

    # Per-model sections
    for model_key, model in MODELS.items():
        html += f'<h2 id="{model_key}">{model["title"]}</h2>'

        # Model description
        html += f'''<div class="model-box">
<p><strong>Acquisition:</strong> {model["acquisition"]}</p>
<div class="equation">{model["equation_html"]}</div>
<p><strong>Fitting method:</strong> {model["fitting"]}</p>
<p><strong>Limitations:</strong> {model["limitations"]}</p>
</div>'''

        # Parameters
        html += '<div class="card"><h3>Fitted Parameters</h3>'
        for pname, pinfo in model["parameters"].items():
            html += f'''<div class="param">
<span class="name">{pname}</span> <span class="unit">[{pinfo["unit"]}]</span>
<p>{pinfo["description"]}</p>
<p><em>{pinfo["physiology"]}</em></p>'''
            if "literature_3T" in pinfo:
                html += '<p class="lit">Literature values at 3T: '
                html += ', '.join([f'{t}: {v}' for t, v in pinfo["literature_3T"].items()])
                html += '</p>'
            html += '</div>'
        html += '</div>'

        # Parameter map figure
        fig_key = {
            "T2star": "T2star",
            "MP2RAGE": "MP2RAGE",
            "VFA_T1": None,
            "QMT": None,
            "mcDESPOT": None,
        }.get(model_key)

        if fig_key and fig_key in figs:
            html += f'<div class="card"><h3>Parameter Map</h3><img src="data:image/png;base64,{figs[fig_key]}"></div>'

        # R2* map (bonus for T2star section)
        if model_key == "T2star" and "R2star" in figs:
            html += f'<div class="card"><h3>R2* Map (= 1/T2*)</h3><img src="data:image/png;base64,{figs["R2star"]}"></div>'

    # Summary table
    html += '<h2>Available Maps Summary</h2><div class="card"><table>'
    html += '<tr><th>Parameter</th><th>Session</th><th>Status</th><th>File</th></tr>'
    for name, info in map_paths.items():
        path, model = info
        exists = os.path.exists(path)
        status = '<span class="good">✓ Available</span>' if exists else '<span class="warn">✗ Not yet fitted</span>'
        html += f'<tr><td>{name}</td><td>{os.path.basename(os.path.dirname(path))}</td><td>{status}</td><td>{os.path.basename(path)}</td></tr>'
    html += '</table></div>'

    html += f'<hr><p style="color:#95a5a6;font-size:11px;text-align:center;">qMRI Report | neurojax + sbi4dwi | {timestamp}</p></body></html>'

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(html)

    print(f"  Report: {out_path}")
    print(f"  Open: open {out_path}")


if __name__ == "__main__":
    main()
