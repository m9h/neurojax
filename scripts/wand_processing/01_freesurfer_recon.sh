#!/bin/bash
# ============================================================================
# Stage 1: FreeSurfer recon-all — Cortical Surface Reconstruction
# ============================================================================
# Input:  WAND ses-02/anat/ T1w (3T with ultra-strong gradients)
# Output: FreeSurfer subject directory with surfaces, parcellations, etc.
#
# Runtime: ~6-8 hours per subject on modern CPU
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id> [--hires]}"
HIRES="${2:-}"

# Find the T1w image
# WAND ses-02 has structural + DWI; ses-03/04 also have anat
# Use ses-02 T1w as it's co-registered with DWI session
T1_DIR="${WAND_ROOT}/${SUBJECT}/ses-02/anat"
T1W=$(find "${T1_DIR}" -name "*T1w*.nii.gz" | head -1)

if [ -z "$T1W" ]; then
    echo "ERROR: No T1w found in ${T1_DIR}"
    echo "Trying ses-03..."
    T1_DIR="${WAND_ROOT}/${SUBJECT}/ses-03/anat"
    T1W=$(find "${T1_DIR}" -name "*T1w*.nii.gz" | head -1)
fi

if [ -z "$T1W" ]; then
    echo "ERROR: No T1w found for ${SUBJECT}"
    exit 1
fi

echo "=== FreeSurfer recon-all ==="
echo "Subject: ${SUBJECT}"
echo "T1w:     ${T1W}"
echo "Output:  ${SUBJECTS_DIR}/${SUBJECT}"

# Check if already completed
if [ -f "${SUBJECTS_DIR}/${SUBJECT}/scripts/recon-all.done" ]; then
    echo "recon-all already completed for ${SUBJECT}, skipping."
    exit 0
fi

# Run recon-all
# -all: complete pipeline (motion correction, skull strip, segmentation,
#       surface extraction, parcellation, cortical thickness)
# -parallel: use multiple threads
# -openmp: OpenMP parallelism
RECON_ARGS="-all -s ${SUBJECT} -i ${T1W} -parallel -openmp ${OMP_NUM_THREADS}"

# Optional: high-resolution mode for 7T data
if [ "${HIRES}" = "--hires" ]; then
    RECON_ARGS="${RECON_ARGS} -hires"
    echo "Running in high-resolution mode (7T)"
fi

recon-all ${RECON_ARGS}

echo "=== recon-all complete for ${SUBJECT} ==="

# Also check for T2w (FLAIR) to improve pial surface
T2W=$(find "${T1_DIR}" -name "*T2w*.nii.gz" -o -name "*FLAIR*.nii.gz" | head -1)
if [ -n "$T2W" ]; then
    echo "T2w/FLAIR found, refining pial surface..."
    recon-all -s ${SUBJECT} -T2pial -T2 ${T2W}
fi
