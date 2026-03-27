#!/bin/bash
# ============================================================================
# Stage N: Feature Extraction for Prediction Model
# ============================================================================
# Collects all per-subject features from all modalities into a single
# feature vector for cross-validated TMS response prediction (Phase 4).
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"

FEAT_DIR="${WAND_DERIVATIVES}/features/${SUBJECT}"
mkdir -p "${FEAT_DIR}"

echo "=== Feature Extraction: ${SUBJECT} ==="

uv run python3 << PYEOF
import os, sys, json
import numpy as np
sys.path.insert(0, '${WAND_ROOT}/../neurojax/src')

features = {}
feat_dir = "${FEAT_DIR}"

# --- Structural features ---
conn_file = "${WAND_DERIVATIVES}/fsl-connectome/${SUBJECT}/ses-02/${SUBJECT}_atlas-desikan_desc-sift2_connectivity.csv"
if os.path.exists(conn_file):
    from neurojax.analysis.prediction import extract_connectome_features
    import jax.numpy as jnp
    Cmat = jnp.array(np.loadtxt(conn_file, delimiter=","))
    fs = extract_connectome_features(Cmat)
    for name, val in zip(fs.names, np.array(fs.features[0])):
        features[name] = float(val)
    print(f"  Structural: {len(fs.names)} features")
else:
    print("  Structural: connectome not found")

# --- Dynamics features (HMM) ---
hmm_stats = "${WAND_DERIVATIVES}/neurojax-meg/${SUBJECT}/ses-01/hmm/summary_stats.json"
if os.path.exists(hmm_stats):
    with open(hmm_stats) as f:
        stats = json.load(f)
    for k, fo in enumerate(stats["fractional_occupancy"]):
        features[f"hmm_fo_state{k}"] = fo
    for k, lt in enumerate(stats["mean_lifetime_ms"]):
        features[f"hmm_lifetime_ms_state{k}"] = lt
    features["hmm_switching_rate_hz"] = stats["switching_rate_hz"]
    print(f"  Dynamics: {len(stats['fractional_occupancy'])*2 + 1} features")
else:
    print("  Dynamics: HMM stats not found")

# --- FOOOF features ---
fooof_exp = "${WAND_DERIVATIVES}/neurojax-meg/${SUBJECT}/ses-01/fooof/aperiodic_exponent.npy"
fooof_iaf = "${WAND_DERIVATIVES}/neurojax-meg/${SUBJECT}/ses-01/fooof/individual_alpha_frequency.npy"
if os.path.exists(fooof_exp):
    exp = np.load(fooof_exp)
    features["fooof_mean_aperiodic_exponent"] = float(np.nanmean(exp))
    features["fooof_std_aperiodic_exponent"] = float(np.nanstd(exp))
    print(f"  FOOOF aperiodic: 2 features")
if os.path.exists(fooof_iaf):
    iaf = np.load(fooof_iaf)
    features["fooof_mean_iaf"] = float(np.nanmean(iaf))
    print(f"  FOOOF IAF: 1 feature")

# --- MRS features ---
mrs_summary = "${WAND_DERIVATIVES}/fsl-mrs/${SUBJECT}/summary.csv"
if os.path.exists(mrs_summary):
    # Parse MRS concentrations
    print(f"  MRS: loading from {mrs_summary}")
    # TODO: parse fsl_mrs summary CSV for GABA, Glu, NAA per VOI
else:
    # Check individual VOI results
    for voi in ["anteriorcingulate", "occipital", "rightauditory", "smleft"]:
        fit_dir = "${WAND_DERIVATIVES}/fsl-mrs/${SUBJECT}/ses-04/mrs/{}/fit".format(voi)
        if os.path.exists(fit_dir):
            features[f"mrs_{voi}_available"] = 1.0
            print(f"  MRS {voi}: fit directory found")

# --- Myelin features ---
myelin_file = "${WAND_DERIVATIVES}/advanced-freesurfer/${SUBJECT}/myelin/T1w_T2w_ratio.nii.gz"
if os.path.exists(myelin_file):
    features["myelin_t1t2_available"] = 1.0
    print(f"  Myelin: T1w/T2w ratio available")

# Save all features
with open(os.path.join(feat_dir, "features.json"), "w") as f:
    json.dump(features, f, indent=2)

feature_array = np.array(list(features.values()))
np.save(os.path.join(feat_dir, "feature_vector.npy"), feature_array)

feature_names = list(features.keys())
with open(os.path.join(feat_dir, "feature_names.json"), "w") as f:
    json.dump(feature_names, f, indent=2)

print(f"\nTotal features: {len(features)}")
print(f"Saved to: {feat_dir}/")
PYEOF

echo "=== Feature extraction complete for ${SUBJECT} ==="
