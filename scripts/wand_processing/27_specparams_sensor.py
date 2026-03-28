#!/usr/bin/env python3
"""
Stage 27: Sensor-Level Specparams (FOOOF) from Raw MEG
======================================================
Minimal-preprocessing sensor-level spectral parameterization.
Runs directly on CTF .ds data — no source reconstruction needed.

Pipeline:
  CTF .ds → filter (1-100 Hz) → notch (50 Hz + harmonics)
  → bad channel detection → PSD (multitaper)
  → specparams per sensor → summary + HTML report

Output:
  derivatives/specparams/{subject}/ses-01/
    ├── resting_psds.npy          # (n_channels, n_freqs)
    ├── resting_freqs.npy         # (n_freqs,)
    ├── specparams_results.json   # full per-channel results
    ├── aperiodic_exponent.npy    # (n_channels,) 1/f slope
    ├── aperiodic_offset.npy      # (n_channels,)
    ├── individual_alpha_freq.npy # (n_channels,) IAF per sensor
    ├── peak_alpha_power.npy      # (n_channels,) alpha peak power
    ├── figures/
    │   ├── psd_overview.png
    │   ├── aperiodic_topo.png
    │   ├── alpha_topo.png
    │   ├── example_fits.png
    │   └── peak_frequency_hist.png
    └── specparams_report.html
"""

import os
import sys
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
WAND_ROOT = os.path.expanduser("~/dev/wand")
SUBJECT = sys.argv[1] if len(sys.argv) > 1 else "sub-08033"
TASKS = ["resting"]  # can extend to task data later

# specparams settings
FREQ_RANGE = [1, 45]
PEAK_WIDTH_LIMITS = [1, 12]
MAX_N_PEAKS = 6
MIN_PEAK_HEIGHT = 0.1
APERIODIC_MODE = "fixed"  # 'fixed' or 'knee'

# preprocessing
L_FREQ = 1.0
H_FREQ = 100.0
NOTCH_FREQS = [50, 100]  # UK mains + harmonic


def preprocess_and_compute_psd(ds_path, task_name):
    """Load CTF data, preprocess, compute multitaper PSD."""
    import mne

    print(f"\n--- Loading {task_name} MEG from {os.path.basename(ds_path)} ---")
    raw = mne.io.read_raw_ctf(ds_path, preload=True, verbose=False)
    print(f"  Channels: {len(raw.ch_names)}, Sfreq: {raw.info['sfreq']} Hz, "
          f"Duration: {raw.times[-1]:.1f}s")

    # Pick MEG channels only (gradiometers for CTF)
    raw.pick("mag")
    n_channels = len(raw.ch_names)
    print(f"  MEG channels: {n_channels}")

    # Band-pass filter
    print(f"  Filtering: {L_FREQ}-{H_FREQ} Hz")
    raw.filter(L_FREQ, H_FREQ, verbose=False)

    # Notch filter (50 Hz UK mains + harmonics)
    print(f"  Notch filter: {NOTCH_FREQS} Hz")
    raw.notch_filter(NOTCH_FREQS, verbose=False)

    # Downsample to 250 Hz (plenty for 1-45 Hz spectral analysis)
    # This reduces from 720K to 150K samples — 4.8x faster PSD
    TARGET_SFREQ = 250.0
    if raw.info["sfreq"] > TARGET_SFREQ:
        print(f"  Downsampling: {raw.info['sfreq']} → {TARGET_SFREQ} Hz")
        raw.resample(TARGET_SFREQ, verbose=False)

    # Detect bad channels via variance
    data = raw.get_data()
    ch_var = np.var(data, axis=1)
    median_var = np.median(ch_var)
    mad = np.median(np.abs(ch_var - median_var))
    threshold = median_var + 5 * 1.4826 * mad  # 5 MAD
    bad_mask = ch_var > threshold
    bad_channels = [raw.ch_names[i] for i in range(n_channels) if bad_mask[i]]
    if bad_channels:
        print(f"  Bad channels (5 MAD): {bad_channels}")
        raw.info["bads"] = bad_channels
        raw.interpolate_bads(verbose=False)
    else:
        print("  No bad channels detected")

    # Compute PSD using Welch's method (fast and standard for FOOOF)
    # 4-second windows with 50% overlap → ~0.25 Hz resolution, ~300 segments averaged
    print("  Computing Welch PSD (4s windows, 50% overlap)...")
    spectrum = raw.compute_psd(method="welch", fmin=0.5, fmax=55,
                               n_fft=1024, n_overlap=512, verbose=False)
    psds = spectrum.get_data()  # (n_channels, n_freqs) in T²/Hz
    freqs = spectrum.freqs

    # Convert to fT²/Hz for better numerical range
    psds_uv = psds * 1e30  # T² → fT² (for magnetometers)

    print(f"  PSD shape: {psds_uv.shape}, freq range: {freqs[0]:.1f}-{freqs[-1]:.1f} Hz")
    print(f"  PSD range (fT^2/Hz): min={psds_uv.min():.2e}, "
          f"median={np.median(psds_uv):.2e}, max={psds_uv.max():.2e}")
    print(f"  log10(PSD) range: {np.log10(psds_uv.min()):.1f} to {np.log10(psds_uv.max()):.1f}")

    return raw, psds_uv, freqs, bad_channels


def run_specparams(psds, freqs, ch_names):
    """Run specparams (FOOOF) on each channel's PSD.

    Compatible with specparam v2.0 API:
      - aperiodic params via fm.results.params.aperiodic.params
      - peak params via fm.results.params.periodic.params  (converted: CF, power, BW)
      - metrics via fm.results.metrics.results dict
      - modeled spectrum via fm.results.model.modeled_spectrum
    """
    from specparam import SpectralModel

    n_channels = psds.shape[0]
    results = []

    print(f"\nFitting specparams on {n_channels} channels...")

    for i in range(n_channels):
        fm = SpectralModel(
            peak_width_limits=PEAK_WIDTH_LIMITS,
            max_n_peaks=MAX_N_PEAKS,
            min_peak_height=MIN_PEAK_HEIGHT,
            aperiodic_mode=APERIODIC_MODE,
            verbose=False,
        )

        try:
            fm.fit(freqs, psds[i, :], FREQ_RANGE)

            # Check if fit produced a model (v2.0 API)
            if not fm.results.has_model:
                raise RuntimeError("Model fitting was unsuccessful")

            ap_params = fm.results.params.aperiodic.params   # [offset, exponent]
            pe_params = fm.results.params.periodic.params     # [[cf, pw, bw], ...]
            metrics = fm.results.metrics.results              # dict

            r_squared = float(metrics.get('gof_rsquared', np.nan))
            fit_error = float(metrics.get('error_mae', np.nan))

            # Extract peaks (converted params: CF, power, bandwidth)
            peaks = []
            if len(pe_params) > 0:
                for peak in pe_params:
                    peaks.append({
                        "frequency": float(peak[0]),
                        "power": float(peak[1]),
                        "bandwidth": float(peak[2]),
                    })

            # IAF: strongest peak in alpha range
            alpha_peaks = [p for p in peaks if 7 <= p["frequency"] <= 13]
            iaf = max(alpha_peaks, key=lambda x: x["power"])["frequency"] if alpha_peaks else None
            alpha_power = max(alpha_peaks, key=lambda x: x["power"])["power"] if alpha_peaks else None

            # Beta peak
            beta_peaks = [p for p in peaks if 13 < p["frequency"] <= 30]
            beta_freq = max(beta_peaks, key=lambda x: x["power"])["frequency"] if beta_peaks else None

            result = {
                "channel": ch_names[i],
                "aperiodic_offset": float(ap_params[0]),
                "aperiodic_exponent": float(ap_params[1]),
                "n_peaks": len(peaks),
                "r_squared": r_squared,
                "error": fit_error,
                "peaks": peaks,
                "iaf": iaf,
                "alpha_power": alpha_power,
                "beta_freq": beta_freq,
            }
        except Exception as e:
            result = {
                "channel": ch_names[i],
                "aperiodic_offset": np.nan,
                "aperiodic_exponent": np.nan,
                "n_peaks": 0,
                "r_squared": np.nan,
                "error": np.nan,
                "peaks": [],
                "iaf": None,
                "alpha_power": None,
                "beta_freq": None,
                "fit_error": str(e),
            }

        results.append(result)

    # Summary
    exponents = [r["aperiodic_exponent"] for r in results if not np.isnan(r["aperiodic_exponent"])]
    iafs = [r["iaf"] for r in results if r["iaf"] is not None]
    r2s = [r["r_squared"] for r in results if not np.isnan(r["r_squared"])]

    print(f"  Fit success: {len(exponents)}/{n_channels}")
    if exponents:
        print(f"  Aperiodic exponent: {np.mean(exponents):.3f} +/- {np.std(exponents):.3f}")
    if r2s:
        print(f"  Mean R^2: {np.mean(r2s):.3f}")
    print(f"  Channels with alpha peak: {len(iafs)}/{n_channels}")
    if iafs:
        print(f"  IAF: {np.mean(iafs):.1f} +/- {np.std(iafs):.1f} Hz")

    return results


def generate_figures(raw, psds, freqs, results, out_dir):
    """Generate diagnostic figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    ch_names = [r["channel"] for r in results]
    exponents = np.array([r["aperiodic_exponent"] for r in results])
    offsets = np.array([r["aperiodic_offset"] for r in results])
    iafs = np.array([r["iaf"] if r["iaf"] is not None else np.nan for r in results])
    alpha_powers = np.array([r["alpha_power"] if r["alpha_power"] is not None else np.nan for r in results])
    r2s = np.array([r["r_squared"] for r in results])

    # 1. PSD Overview — all channels overlaid
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Log-log PSD
    freq_mask = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
    for i in range(min(psds.shape[0], 275)):
        axes[0].semilogy(freqs[freq_mask], psds[i, freq_mask], alpha=0.15, color="steelblue", linewidth=0.5)
    mean_psd = np.mean(psds, axis=0)
    axes[0].semilogy(freqs[freq_mask], mean_psd[freq_mask], color="darkred", linewidth=2, label="Mean")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power (fT²/Hz)")
    axes[0].set_title("Power Spectral Density — All MEG Channels")
    axes[0].legend()
    axes[0].set_xlim(FREQ_RANGE)

    # R² distribution
    axes[1].hist(r2s[~np.isnan(r2s)], bins=30, color="steelblue", edgecolor="white")
    axes[1].axvline(np.nanmean(r2s), color="red", linestyle="--", label=f"Mean R² = {np.nanmean(r2s):.3f}")
    axes[1].set_xlabel("R²")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Specparams Fit Quality (R²)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "psd_overview.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: psd_overview.png")

    # 2. Topographic maps: aperiodic exponent + alpha power
    try:
        info = raw.info
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Aperiodic exponent topo
        im1, _ = mne.viz.plot_topomap(
            exponents, info, axes=axes[0], show=False,
            vlim=(np.nanpercentile(exponents, 5), np.nanpercentile(exponents, 95)),
            cmap="RdBu_r"
        )
        axes[0].set_title("Aperiodic Exponent (1/f slope)")
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # IAF topo (NaN where no alpha)
        valid_iaf = np.where(np.isnan(iafs), np.nanmean(iafs), iafs)
        im2, _ = mne.viz.plot_topomap(
            valid_iaf, info, axes=axes[1], show=False,
            vlim=(7, 13), cmap="plasma"
        )
        axes[1].set_title("Individual Alpha Frequency (Hz)")
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

        # Alpha power topo
        valid_alpha = np.where(np.isnan(alpha_powers), 0, alpha_powers)
        im3, _ = mne.viz.plot_topomap(
            valid_alpha, info, axes=axes[2], show=False,
            cmap="hot"
        )
        axes[2].set_title("Alpha Peak Power")
        plt.colorbar(im3, ax=axes[2], fraction=0.046)

        fig.suptitle(f"Specparams Topography — {SUBJECT} Resting MEG", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "specparams_topos.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: specparams_topos.png")
    except Exception as e:
        print(f"  Topo plot failed: {e}")

    # 3. Example fits — show 6 representative channels
    from specparam import SpectralModel

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # Pick channels with best, median, worst R² + occipital channels
    # Only consider channels with valid (non-NaN) R² for sorting
    valid_r2_mask = ~np.isnan(r2s)
    valid_indices = np.where(valid_r2_mask)[0]
    if len(valid_indices) == 0:
        print("  Skipping example fits — no successful fits")
        plt.close(fig)
    else:
        sorted_valid = valid_indices[np.argsort(r2s[valid_indices])]
        example_idx = []
        # Best 2
        example_idx.extend(sorted_valid[-2:].tolist())
        # Median 2
        mid = len(sorted_valid) // 2
        example_idx.extend(sorted_valid[mid:mid+2].tolist())
        # Find occipital-ish channels (MZO, MRO, MLO)
        for prefix in ["MZO", "MRO", "MLO"]:
            occ = [i for i, ch in enumerate(ch_names) if ch.startswith(prefix)]
            if occ:
                example_idx.append(occ[0])
                if len(example_idx) >= 6:
                    break
        # Deduplicate while preserving order
        seen = set()
        example_idx = [x for x in example_idx if not (x in seen or seen.add(x))]
        example_idx = example_idx[:6]

        for ax_idx, ch_idx in enumerate(example_idx):
            ax = axes.flat[ax_idx]
            fm = SpectralModel(
                peak_width_limits=PEAK_WIDTH_LIMITS,
                max_n_peaks=MAX_N_PEAKS,
                min_peak_height=MIN_PEAK_HEIGHT,
                aperiodic_mode=APERIODIC_MODE,
                verbose=False,
            )
            fm.fit(freqs, psds[ch_idx], FREQ_RANGE)

            # Manual plot using v2.0 API
            freq_mask = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
            f = freqs[freq_mask]
            power = np.log10(psds[ch_idx, freq_mask])

            ax.plot(f, power, "k-", linewidth=1, label="Data")

            if fm.results.has_model:
                # Aperiodic fit (v2.0 API)
                ap = fm.results.params.aperiodic.params  # [offset, exponent]
                aperiodic_fit = ap[0] - np.log10(f ** ap[1])
                ax.plot(f, aperiodic_fit, "b--", linewidth=1,
                        label=f"1/f (exp={ap[1]:.2f})")

                # Full model
                fmodel = fm.results.model.modeled_spectrum
                if fmodel is not None and len(fmodel) == len(f):
                    r2_val = fm.results.metrics.results.get('gof_rsquared', np.nan)
                    ax.plot(f, fmodel, "r-", linewidth=1.5, alpha=0.7,
                            label=f"Model (R2={r2_val:.3f})")
            else:
                ax.text(0.5, 0.5, "Fit failed", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color="red")

            ax.set_title(f"{ch_names[ch_idx]}", fontsize=10)
            ax.set_xlabel("Freq (Hz)")
            ax.set_ylabel("log10 Power")
            ax.legend(fontsize=7)

        # Clear any unused axes
        for ax_idx in range(len(example_idx), 6):
            axes.flat[ax_idx].set_visible(False)

        fig.suptitle(f"Specparams Example Fits - {SUBJECT}", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "example_fits.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: example_fits.png")

    # 4. Peak frequency histogram
    all_peak_freqs = []
    for r in results:
        for p in r["peaks"]:
            all_peak_freqs.append(p["frequency"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_peak_freqs, bins=np.arange(1, 46, 0.5), color="steelblue", edgecolor="white")
    ax.axvspan(7, 13, alpha=0.15, color="orange", label="Alpha (7-13 Hz)")
    ax.axvspan(13, 30, alpha=0.15, color="green", label="Beta (13-30 Hz)")
    ax.axvspan(4, 7, alpha=0.15, color="purple", label="Theta (4-7 Hz)")
    ax.set_xlabel("Peak Frequency (Hz)")
    ax.set_ylabel("Count (across all channels)")
    ax.set_title(f"Detected Oscillatory Peaks — {SUBJECT} Resting MEG")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "peak_frequency_hist.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: peak_frequency_hist.png")

    # 5. Aperiodic exponent vs R² scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(exponents, r2s, c=iafs, cmap="plasma",
                        alpha=0.6, edgecolor="gray", linewidth=0.3)
    ax.set_xlabel("Aperiodic Exponent")
    ax.set_ylabel("R²")
    ax.set_title("Fit Quality vs 1/f Slope (color = IAF)")
    plt.colorbar(scatter, label="IAF (Hz)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "exponent_vs_r2.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: exponent_vs_r2.png")

    return fig_dir


def generate_html_report(results, out_dir, subject, fig_dir):
    """Generate HTML report summarizing specparams results."""
    exponents = [r["aperiodic_exponent"] for r in results if not np.isnan(r["aperiodic_exponent"])]
    iafs = [r["iaf"] for r in results if r["iaf"] is not None]
    r2s = [r["r_squared"] for r in results if not np.isnan(r["r_squared"])]
    n_peaks_all = [r["n_peaks"] for r in results]

    # All detected peaks
    all_peaks = []
    for r in results:
        for p in r["peaks"]:
            all_peaks.append({**p, "channel": r["channel"]})

    # Band-specific peak counts
    theta_peaks = [p for p in all_peaks if 4 <= p["frequency"] < 7]
    alpha_peaks = [p for p in all_peaks if 7 <= p["frequency"] <= 13]
    beta_peaks = [p for p in all_peaks if 13 < p["frequency"] <= 30]
    gamma_peaks = [p for p in all_peaks if p["frequency"] > 30]

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Specparams Report — {subject}</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #34495e; margin-top: 30px; }}
.summary-box {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 15px 0; }}
.metric {{ display: inline-block; width: 200px; padding: 15px; margin: 10px; background: #ecf0f1; border-radius: 5px; text-align: center; }}
.metric .value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
.metric .label {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #3498db; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
img {{ max-width: 100%; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0; }}
.fig-container {{ background: white; padding: 15px; border-radius: 8px; margin: 15px 0; }}
.note {{ background: #fff3cd; padding: 10px 15px; border-left: 4px solid #ffc107; border-radius: 4px; margin: 15px 0; }}
.good {{ color: #27ae60; }}
.warn {{ color: #e67e22; }}
</style>
</head>
<body>
<div class="container">
<h1>Specparams (FOOOF) Report — {subject}</h1>
<p>Sensor-level spectral parameterization of resting MEG data.</p>
<p>Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<h2>Summary</h2>
<div class="summary-box">
<div class="metric">
    <div class="value">{np.mean(exponents):.2f}</div>
    <div class="label">Mean Aperiodic Exponent</div>
</div>
<div class="metric">
    <div class="value">{np.mean(iafs):.1f} Hz</div>
    <div class="label">Mean IAF ({len(iafs)}/{len(results)} ch)</div>
</div>
<div class="metric">
    <div class="value">{np.mean(r2s):.3f}</div>
    <div class="label">Mean R²</div>
</div>
<div class="metric">
    <div class="value">{len(all_peaks)}</div>
    <div class="label">Total Peaks Detected</div>
</div>
<div class="metric">
    <div class="value">{len(alpha_peaks)}</div>
    <div class="label">Alpha Peaks (7-13 Hz)</div>
</div>
<div class="metric">
    <div class="value">{len(beta_peaks)}</div>
    <div class="label">Beta Peaks (13-30 Hz)</div>
</div>
</div>

<div class="note">
<strong>Interpretation:</strong> Aperiodic exponent ~1.5-2.0 typical for resting MEG.
Higher values → steeper 1/f → more inhibition-dominated (correlates with GABA from MRS).
IAF ~8-12 Hz typical; connects to white matter conduction velocity (AxCaliber from WAND DWI).
</div>

<h2>Aperiodic Component</h2>
<div class="summary-box">
<table>
<tr><th>Statistic</th><th>Exponent</th><th>Offset</th></tr>
<tr><td>Mean ± SD</td>
    <td>{np.mean(exponents):.3f} ± {np.std(exponents):.3f}</td>
    <td>{np.mean([r['aperiodic_offset'] for r in results if not np.isnan(r['aperiodic_offset'])]):.3f} ± {np.std([r['aperiodic_offset'] for r in results if not np.isnan(r['aperiodic_offset'])]):.3f}</td>
</tr>
<tr><td>Range</td>
    <td>{np.min(exponents):.3f} – {np.max(exponents):.3f}</td>
    <td>{np.min([r['aperiodic_offset'] for r in results if not np.isnan(r['aperiodic_offset'])]):.3f} – {np.max([r['aperiodic_offset'] for r in results if not np.isnan(r['aperiodic_offset'])]):.3f}</td>
</tr>
</table>
</div>

<h2>Oscillatory Peaks by Band</h2>
<div class="summary-box">
<table>
<tr><th>Band</th><th>Range</th><th>Peaks Found</th><th>Mean Frequency</th><th>Mean Power</th></tr>
<tr><td>Theta</td><td>4-7 Hz</td><td>{len(theta_peaks)}</td>
    <td>{f'{np.mean([p["frequency"] for p in theta_peaks]):.1f} Hz' if theta_peaks else 'N/A'}</td>
    <td>{f'{np.mean([p["power"] for p in theta_peaks]):.3f}' if theta_peaks else 'N/A'}</td></tr>
<tr><td>Alpha</td><td>7-13 Hz</td><td>{len(alpha_peaks)}</td>
    <td>{f'{np.mean([p["frequency"] for p in alpha_peaks]):.1f} Hz' if alpha_peaks else 'N/A'}</td>
    <td>{f'{np.mean([p["power"] for p in alpha_peaks]):.3f}' if alpha_peaks else 'N/A'}</td></tr>
<tr><td>Beta</td><td>13-30 Hz</td><td>{len(beta_peaks)}</td>
    <td>{f'{np.mean([p["frequency"] for p in beta_peaks]):.1f} Hz' if beta_peaks else 'N/A'}</td>
    <td>{f'{np.mean([p["power"] for p in beta_peaks]):.3f}' if beta_peaks else 'N/A'}</td></tr>
<tr><td>Gamma</td><td>30-45 Hz</td><td>{len(gamma_peaks)}</td>
    <td>{f'{np.mean([p["frequency"] for p in gamma_peaks]):.1f} Hz' if gamma_peaks else 'N/A'}</td>
    <td>{f'{np.mean([p["power"] for p in gamma_peaks]):.3f}' if gamma_peaks else 'N/A'}</td></tr>
</table>
</div>

<h2>Figures</h2>
"""

    # Add figures
    for fname, title in [
        ("psd_overview.png", "PSD Overview + Fit Quality"),
        ("specparams_topos.png", "Topographic Maps"),
        ("example_fits.png", "Example Specparams Fits"),
        ("peak_frequency_hist.png", "Peak Frequency Distribution"),
        ("exponent_vs_r2.png", "Aperiodic Exponent vs R²"),
    ]:
        fpath = os.path.join(fig_dir, fname)
        if os.path.exists(fpath):
            # Use relative path
            rel_path = os.path.join("figures", fname)
            html += f"""
<div class="fig-container">
<h3>{title}</h3>
<img src="{rel_path}" alt="{title}">
</div>
"""

    # Top-10 strongest alpha channels table
    alpha_channels = [(r["channel"], r["iaf"], r["alpha_power"], r["aperiodic_exponent"])
                      for r in results if r["iaf"] is not None]
    alpha_channels.sort(key=lambda x: x[2] if x[2] else 0, reverse=True)

    html += """
<h2>Top 10 Channels by Alpha Power</h2>
<div class="summary-box">
<table>
<tr><th>Rank</th><th>Channel</th><th>IAF (Hz)</th><th>Alpha Power</th><th>1/f Exponent</th></tr>
"""
    for i, (ch, iaf, ap, exp) in enumerate(alpha_channels[:10]):
        html += f"<tr><td>{i+1}</td><td>{ch}</td><td>{iaf:.1f}</td><td>{ap:.3f}</td><td>{exp:.3f}</td></tr>\n"

    html += """</table></div>

<h2>Cross-Modal Integration Notes</h2>
<div class="note">
<p><strong>Connections to other WAND modalities:</strong></p>
<ul>
<li><strong>MRS (ses-04/05):</strong> Aperiodic exponent correlates with GABA/glutamate ratio — steeper slope ↔ more inhibition</li>
<li><strong>AxCaliber DWI (ses-02):</strong> IAF correlates with white matter conduction velocity (Hursh-Rushton law)</li>
<li><strong>QMT (ses-02):</strong> Myelination (BPF) affects conduction velocity → IAF</li>
<li><strong>T1w/T2w ratio:</strong> Myelin proxy — compare regional patterns with aperiodic exponent topography</li>
<li><strong>TMS-EEG (ses-08):</strong> Pre-stimulus alpha power predicts TMS-evoked response amplitude</li>
</ul>
</div>

<p style="color: #7f8c8d; font-size: 12px;">
Generated by neurojax specparams pipeline. specparam v2.0 (Donoghue et al. 2020, Nature Neuroscience).
</p>
</div>
</body>
</html>"""

    report_path = os.path.join(out_dir, "specparams_report.html")
    with open(report_path, "w") as f:
        f.write(html)
    print(f"  Report: {report_path}")
    return report_path


def main():
    for task in TASKS:
        ds_path = os.path.join(WAND_ROOT, SUBJECT, "ses-01", "meg",
                               f"{SUBJECT}_ses-01_task-{task}.ds")
        if not os.path.exists(ds_path):
            print(f"  Not found: {ds_path}")
            continue

        out_dir = os.path.join(WAND_ROOT, "derivatives", "specparams", SUBJECT, "ses-01")
        os.makedirs(out_dir, exist_ok=True)

        # 1. Preprocess + PSD
        raw, psds, freqs, bad_channels = preprocess_and_compute_psd(ds_path, task)

        # Save PSDs
        np.save(os.path.join(out_dir, f"{task}_psds.npy"), psds)
        np.save(os.path.join(out_dir, f"{task}_freqs.npy"), freqs)

        # 2. Run specparams
        results = run_specparams(psds, freqs, raw.ch_names)

        # Save results
        with open(os.path.join(out_dir, "specparams_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Save summary arrays
        np.save(os.path.join(out_dir, "aperiodic_exponent.npy"),
                np.array([r["aperiodic_exponent"] for r in results]))
        np.save(os.path.join(out_dir, "aperiodic_offset.npy"),
                np.array([r["aperiodic_offset"] for r in results]))
        np.save(os.path.join(out_dir, "individual_alpha_freq.npy"),
                np.array([r["iaf"] if r["iaf"] is not None else np.nan for r in results]))
        np.save(os.path.join(out_dir, "peak_alpha_power.npy"),
                np.array([r["alpha_power"] if r["alpha_power"] is not None else np.nan for r in results]))

        # 3. Figures
        fig_dir = generate_figures(raw, psds, freqs, results, out_dir)

        # 4. HTML report
        report_path = generate_html_report(results, out_dir, SUBJECT, fig_dir)

        print(f"\n=== Specparams complete for {SUBJECT} {task} ===")
        print(f"  Output: {out_dir}/")
        print(f"  Report: open {report_path}")


if __name__ == "__main__":
    main()
