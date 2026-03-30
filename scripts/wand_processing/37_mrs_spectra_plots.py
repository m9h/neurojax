#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "nibabel>=5.0",
#     "numpy>=1.24",
#     "matplotlib>=3.7",
# ]
# ///
"""Plot MRS spectra from NIfTI-MRS files for QC and paper figures.

Generates:
  - Individual spectrum per VOI
  - 4-panel comparison across VOIs
  - Water reference spectrum
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def load_nifti_mrs(path):
    """Load NIfTI-MRS and extract FID + metadata."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.complex64)

    # Try to get spectral parameters from header extensions
    hdr = img.header
    pixdim = hdr.get_zooms()
    dwell_time = pixdim[3] if len(pixdim) > 3 else None

    # Try JSON sidecar
    json_path = str(path).replace('.nii.gz', '.json').replace('.nii', '.json')
    meta = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)

    return data, dwell_time, meta


def fid_to_spectrum(fid, dwell_time=None, n_points=None):
    """Convert FID to spectrum via FFT."""
    if n_points is None:
        n_points = fid.shape[0]

    # Apply exponential line broadening (2 Hz)
    if dwell_time:
        t = np.arange(n_points) * dwell_time
        lb = np.exp(-2.0 * np.pi * t)
        fid = fid[:n_points] * lb

    spectrum = np.fft.fftshift(np.fft.fft(fid[:n_points]))
    return spectrum


def ppm_axis(n_points, dwell_time, centre_freq_mhz=297.2):
    """Compute chemical shift (ppm) axis.

    Args:
        n_points: Number of spectral points
        dwell_time: Dwell time in seconds
        centre_freq_mhz: Centre frequency in MHz (297.2 for 7T, 123.2 for 3T)
    """
    sw = 1.0 / dwell_time  # spectral width in Hz
    freq_axis = np.linspace(-sw / 2, sw / 2, n_points)
    ppm = freq_axis / centre_freq_mhz + 4.7  # reference to water at 4.7 ppm
    return ppm


def plot_spectrum(spectrum, ppm, title="", ax=None, ppm_range=(0.5, 4.5)):
    """Plot a single spectrum."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    mask = (ppm >= ppm_range[0]) & (ppm <= ppm_range[1])
    ax.plot(ppm[mask], np.real(spectrum[mask]), 'k-', linewidth=0.5)
    ax.invert_xaxis()
    ax.set_xlabel('Chemical shift (ppm)')
    ax.set_ylabel('Signal (a.u.)')
    ax.set_title(title)

    # Mark key metabolites
    metabolites = {'NAA': 2.02, 'Cr': 3.03, 'Cho': 3.22,
                   'mI': 3.56, 'Glu': 2.35, 'GABA': 3.01}
    for name, pos in metabolites.items():
        if ppm_range[0] <= pos <= ppm_range[1]:
            ax.axvline(pos, color='gray', alpha=0.3, linestyle='--')
            ax.text(pos, ax.get_ylim()[1] * 0.95, name,
                    ha='center', fontsize=7, color='gray')

    return ax


def main():
    parser = argparse.ArgumentParser(description="Plot MRS spectra")
    parser.add_argument("--subject", default="sub-08033")
    parser.add_argument("--wand-root", default="/Users/mhough/dev/wand")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    wand = Path(args.wand_root)
    mrs_dir = wand / "derivatives" / "fsl-mrs" / args.subject
    out_dir = Path(args.output) if args.output else mrs_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    vois = ["anteriorcingulate", "occipital", "rightauditory", "smleft"]
    voi_labels = ["Anterior Cingulate", "Occipital", "Right Auditory",
                  "Left Sensorimotor"]

    for ses, acq, field_mhz, field_label in [
        ("ses-04", "slaser", 297.2, "7T"),
        ("ses-05", "mega", 123.2, "3T"),
    ]:
        print(f"\n=== {ses} ({acq}, {field_label}) ===")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, (voi, label) in enumerate(zip(vois, voi_labels)):
            nifti_files = sorted(
                (mrs_dir / ses / voi).glob("svs*.nii*")
            ) if (mrs_dir / ses / voi).exists() else []

            if not nifti_files:
                print(f"  {voi}: no NIfTI-MRS found")
                axes[i].text(0.5, 0.5, f"{label}\nNo data",
                            ha='center', va='center', transform=axes[i].transAxes)
                continue

            nifti_path = nifti_files[0]
            print(f"  {voi}: {nifti_path.name}")

            data, dwell_time, meta = load_nifti_mrs(nifti_path)
            print(f"    Shape: {data.shape}, dwell: {dwell_time}")

            # Handle multi-dimensional data
            # Typical: (n_spectral, n_coils, n_averages) or (x, y, z, n_spectral, ...)
            if data.ndim >= 3:
                # Average across all dims except spectral (first or last)
                # spec2nii puts spectral in dim 3 (index 3) for NIfTI-MRS
                # but for SVS it's often (1,1,1,n_spec,...)
                shape = data.shape
                print(f"    Data shape: {shape}")

                # Squeeze spatial dims, average coils and dynamics
                d = data.squeeze()
                if d.ndim == 1:
                    fid = d
                elif d.ndim == 2:
                    fid = d.mean(axis=-1) if d.shape[0] > d.shape[1] else d.mean(axis=0)
                elif d.ndim == 3:
                    # (n_spec, n_coils, n_avg) — combine coils then average
                    fid = d.mean(axis=(1, 2))
                else:
                    fid = d.reshape(d.shape[0], -1).mean(axis=1)
            else:
                fid = data.squeeze()

            # Get dwell time from metadata
            if dwell_time is None or dwell_time == 0:
                dwell_time = meta.get("dwelltime", meta.get("SpectralDwellTime", 1.0 / 4000))

            spectrum = fid_to_spectrum(fid, dwell_time)
            ppm = ppm_axis(len(spectrum), dwell_time, field_mhz)

            title = f"{label} ({field_label} {acq})"
            plot_spectrum(spectrum, ppm, title=title, ax=axes[i])

            # Also save individual
            fig_single, ax_single = plt.subplots(1, 1, figsize=(10, 4))
            plot_spectrum(spectrum, ppm, title=title, ax=ax_single)
            fig_single.tight_layout()
            fig_single.savefig(out_dir / f"{ses}_{voi}_spectrum.png", dpi=150)
            plt.close(fig_single)

        fig.suptitle(f"WAND {args.subject} MRS Spectra — {ses} ({field_label})",
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(out_dir / f"{ses}_all_spectra.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_dir}/{ses}_all_spectra.png")


if __name__ == "__main__":
    main()
