#!/bin/bash
# ============================================================================
# Stage 5: Structural Connectome Construction
# ============================================================================
# Uses FreeSurfer parcellation + FSL probtrackx2 to build:
#   - Cmat: structural connectivity matrix (streamline counts)
#   - Lmat: fiber length matrix (mean tract lengths)
#   - Both Desikan-Killiany (68 ROIs) and Destrieux (148 ROIs)
#
# This is the input for neurojax whole-brain modeling:
#   BIDSConnectomeLoader → VbjaxFitnessAdapter(weights=Cmat, delays=Lmat/v)
#
# Runtime: ~4-8 hours per subject (probtrackx2 is the bottleneck)
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"
ACQ="${2:-CHARMED}"
ATLAS="${3:-desikan}"  # desikan or destrieux

DWI_DIR="${WAND_DERIVATIVES}/fsl/${SUBJECT}/ses-02/dwi"
BPX_DIR="${DWI_DIR}/${ACQ}.bedpostX"
REG_DIR="${DWI_DIR}/reg"
CONN_DIR="${WAND_DERIVATIVES}/connectome/${SUBJECT}"
mkdir -p "${CONN_DIR}"

echo "=== Connectome Construction: ${SUBJECT} (${ATLAS}) ==="

# ---------------------------------------------------------------
# Step 1: Convert FreeSurfer parcellation to DWI space
# ---------------------------------------------------------------

if [ "${ATLAS}" = "desikan" ]; then
    PARC_FS="aparc+aseg.mgz"
    N_REGIONS=68
elif [ "${ATLAS}" = "destrieux" ]; then
    PARC_FS="aparc.a2009s+aseg.mgz"
    N_REGIONS=148
else
    echo "Unknown atlas: ${ATLAS}"
    exit 1
fi

# Convert FreeSurfer parcellation to nifti
mri_convert \
    "${SUBJECTS_DIR}/${SUBJECT}/mri/${PARC_FS}" \
    "${CONN_DIR}/${ATLAS}_parc.nii.gz"

# Transform parcellation to DWI space
flirt \
    -in "${CONN_DIR}/${ATLAS}_parc" \
    -ref "${DWI_DIR}/b0_brain" \
    -applyxfm -init "${REG_DIR}/struct2diff.mat" \
    -out "${CONN_DIR}/${ATLAS}_parc_diff" \
    -interp nearestneighbour

# ---------------------------------------------------------------
# Step 2: Create seed masks (one per ROI)
# ---------------------------------------------------------------

SEED_DIR="${CONN_DIR}/seeds_${ATLAS}"
mkdir -p "${SEED_DIR}"

echo "Extracting ROI masks..."

# Get unique non-zero labels from parcellation
python3 << 'PYEOF'
import nibabel as nib
import numpy as np
import sys, os

parc_path = os.environ['CONN_DIR'] + f"/{os.environ['ATLAS']}_parc_diff.nii.gz"
seed_dir = os.environ['SEED_DIR']

img = nib.load(parc_path)
data = img.get_fdata().astype(int)
labels = np.unique(data)
labels = labels[labels > 0]  # skip background

# FreeSurfer cortical labels for Desikan: 1001-1035 (lh), 2001-2035 (rh)
# Filter to cortical only
cortical = [l for l in labels if (1000 < l < 1100) or (2000 < l < 2100)]

print(f"Found {len(cortical)} cortical ROIs")

with open(os.path.join(seed_dir, 'roi_list.txt'), 'w') as f:
    for label in cortical:
        mask_path = os.path.join(seed_dir, f'roi_{label}.nii.gz')
        mask_data = (data == label).astype(np.float32)
        mask_img = nib.Nifti1Image(mask_data, img.affine, img.header)
        nib.save(mask_img, mask_path)
        f.write(f'{mask_path}\n')

print(f"ROI masks saved to {seed_dir}")
PYEOF

# ---------------------------------------------------------------
# Step 3: probtrackx2 — probabilistic tractography
# ---------------------------------------------------------------

TRACK_DIR="${CONN_DIR}/probtrackx_${ATLAS}"
mkdir -p "${TRACK_DIR}"

echo "Running probtrackx2 (this takes several hours)..."

# Create target list (all ROIs)
SEED_LIST="${SEED_DIR}/roi_list.txt"

# Run probtrackx2 in matrix mode
# --network: compute NxN connectivity matrix
# -l: apply distance correction (for length-based weighting)
# -c: curvature threshold (cosine of max turning angle)
# --distthresh: discard implausibly short tracks
probtrackx2 \
    -x "${SEED_LIST}" \
    -l \
    --oneout \
    --forcedir \
    --network \
    -c 0.2 \
    -S 2000 \
    --steplength=0.5 \
    -P 5000 \
    --fibthresh=0.01 \
    --distthresh=0.0 \
    --sampvox=0.0 \
    -s "${BPX_DIR}/merged" \
    -m "${BPX_DIR}/nodif_brain_mask" \
    --dir="${TRACK_DIR}" \
    --omatrix1

echo "probtrackx2 complete."

# ---------------------------------------------------------------
# Step 4: Extract connectivity matrix
# ---------------------------------------------------------------

echo "Building connectivity matrix..."

python3 << 'PYEOF'
import numpy as np
import os

track_dir = os.environ['TRACK_DIR']
conn_dir = os.environ['CONN_DIR']
atlas = os.environ['ATLAS']

# probtrackx2 --network outputs fdt_network_matrix
matrix_file = os.path.join(track_dir, 'fdt_network_matrix')

if os.path.exists(matrix_file):
    # Load the connectivity matrix
    Cmat = np.loadtxt(matrix_file)

    # Symmetrize
    Cmat = (Cmat + Cmat.T) / 2

    # Save as CSV (BIDS-compatible for neurojax BIDSConnectomeLoader)
    np.savetxt(
        os.path.join(conn_dir, f'{atlas}_desc-sift2_connectivity.csv'),
        Cmat, delimiter=','
    )

    print(f"Connectivity matrix: {Cmat.shape}")
    print(f"Non-zero connections: {np.sum(Cmat > 0)}")
    print(f"Density: {np.sum(Cmat > 0) / (Cmat.shape[0] * (Cmat.shape[0]-1)):.3f}")
    print(f"Saved to {conn_dir}/{atlas}_desc-sift2_connectivity.csv")
else:
    print(f"ERROR: {matrix_file} not found")

# Also check for length matrix
length_file = os.path.join(track_dir, 'fdt_network_matrix_lengths')
if os.path.exists(length_file):
    Lmat = np.loadtxt(length_file)
    Lmat = (Lmat + Lmat.T) / 2
    np.savetxt(
        os.path.join(conn_dir, f'{atlas}_desc-meanlength_connectivity.csv'),
        Lmat, delimiter=','
    )
    print(f"Length matrix saved ({atlas}_desc-meanlength_connectivity.csv)")

PYEOF

echo "=== Connectome construction complete for ${SUBJECT} ==="
echo "Output: ${CONN_DIR}/"
echo ""
echo "To load in neurojax:"
echo "  from neurojax.io.connectome import BIDSConnectomeLoader"
echo "  loader = BIDSConnectomeLoader('${WAND_DERIVATIVES}', '${SUBJECT#sub-}', atlas='${ATLAS}')"
echo "  data = loader.load()"
