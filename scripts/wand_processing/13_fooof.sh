#!/bin/bash
# ============================================================================
# Stage L: FOOOF / specparam — Spectral Parameterization
# ============================================================================
# Decomposes MEG power spectra into aperiodic (1/f) + periodic (peaks)
# for each parcel and each HMM/DyNeMo state.
#
# Output: IAF, aperiodic exponent, peak parameters per region per state
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"

MEG_DIR="${WAND_DERIVATIVES}/neurojax-meg/${SUBJECT}/ses-01"
FOOOF_DIR="${MEG_DIR}/fooof"
mkdir -p "${FOOOF_DIR}"

echo "=== FOOOF: ${SUBJECT} ==="

uv run python3 << PYEOF
import os, sys, json
import numpy as np
sys.path.insert(0, '${WAND_ROOT}/../neurojax/src')

fooof_dir = "${FOOOF_DIR}"

# Try specparam (new name) or fooof (old name)
try:
    from specparam import SpectralModel, SpectralGroupModel
    print("Using specparam")
except ImportError:
    try:
        from fooof import FOOOF as SpectralModel, FOOOFGroup as SpectralGroupModel
        print("Using fooof")
    except ImportError:
        print("Neither specparam nor fooof installed.")
        print("Install with: pip install specparam")
        print("Skipping FOOOF analysis.")
        sys.exit(0)

# Load static PSD
psd_file = "${MEG_DIR}/static/psd.npy"
freq_file = "${MEG_DIR}/static/frequencies.npy"

if not os.path.exists(psd_file):
    print(f"PSD not found at {psd_file}. Run 12_meg_dynamics.sh first.")
    sys.exit(1)

psd = np.load(psd_file)     # (n_freqs, n_parcels)
freqs = np.load(freq_file)  # (n_freqs,)

n_freqs, n_parcels = psd.shape
print(f"PSD: {psd.shape}, freqs: {freqs.shape}")

# --- Per-parcel FOOOF on static (resting) PSD ---
print("\nFitting FOOOF per parcel (static PSD)...")

freq_range = [1, 45]
results = []

for p in range(n_parcels):
    fm = SpectralModel(peak_width_limits=[1, 12], max_n_peaks=6, min_peak_height=0.1)
    fm.fit(freqs, psd[:, p], freq_range)

    result = {
        "parcel": p,
        "aperiodic_offset": float(fm.aperiodic_params_[0]),
        "aperiodic_exponent": float(fm.aperiodic_params_[1]),
        "n_peaks": len(fm.peak_params_),
        "r_squared": float(fm.r_squared_),
        "error": float(fm.error_),
    }

    # Extract peaks
    peaks = []
    for peak in fm.peak_params_:
        peaks.append({
            "frequency": float(peak[0]),
            "power": float(peak[1]),
            "bandwidth": float(peak[2]),
        })
    result["peaks"] = peaks

    # Individual alpha frequency (strongest peak in 7-13 Hz)
    alpha_peaks = [pk for pk in peaks if 7 <= pk["frequency"] <= 13]
    if alpha_peaks:
        iaf = max(alpha_peaks, key=lambda x: x["power"])["frequency"]
        result["iaf"] = iaf
    else:
        result["iaf"] = None

    results.append(result)

# Save
with open(os.path.join(fooof_dir, "static_fooof.json"), "w") as f:
    json.dump(results, f, indent=2)

# Summary arrays
aperiodic_exp = np.array([r["aperiodic_exponent"] for r in results])
iafs = np.array([r["iaf"] if r["iaf"] else np.nan for r in results])
np.save(os.path.join(fooof_dir, "aperiodic_exponent.npy"), aperiodic_exp)
np.save(os.path.join(fooof_dir, "individual_alpha_frequency.npy"), iafs)

print(f"  Aperiodic exponent: {np.nanmean(aperiodic_exp):.3f} ± {np.nanstd(aperiodic_exp):.3f}")
print(f"  Mean IAF: {np.nanmean(iafs):.1f} Hz (n={np.sum(~np.isnan(iafs))} parcels with alpha)")
print(f"  Mean R²: {np.mean([r['r_squared'] for r in results]):.3f}")

# --- Per-state FOOOF (HMM) ---
state_psd_file = "${MEG_DIR}/hmm/state_psd.npy"
if os.path.exists(state_psd_file):
    print("\nFitting FOOOF per state (HMM)...")
    state_psd = np.load(state_psd_file)   # (K, n_freqs, n_parcels)
    state_freqs = np.load("${MEG_DIR}/hmm/frequencies.npy")
    K = state_psd.shape[0]

    state_results = []
    for k in range(K):
        state_parcels = []
        for p in range(min(n_parcels, state_psd.shape[2])):
            fm = SpectralModel(peak_width_limits=[1, 12], max_n_peaks=6, min_peak_height=0.05)
            try:
                fm.fit(state_freqs, state_psd[k, :, p], freq_range)
                state_parcels.append({
                    "aperiodic_exponent": float(fm.aperiodic_params_[1]),
                    "n_peaks": len(fm.peak_params_),
                })
            except Exception:
                state_parcels.append({"aperiodic_exponent": np.nan, "n_peaks": 0})

        mean_exp = np.nanmean([r["aperiodic_exponent"] for r in state_parcels])
        print(f"  State {k}: aperiodic exponent = {mean_exp:.3f}")
        state_results.append(state_parcels)

    with open(os.path.join(fooof_dir, "state_fooof.json"), "w") as f:
        json.dump(state_results, f, indent=2)

print(f"\n=== FOOOF complete. Output: {fooof_dir}/ ===")
PYEOF

echo "=== FOOOF complete for ${SUBJECT} ==="
