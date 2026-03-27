#!/bin/bash
# ============================================================================
# Stage 3: bedpostx + xtract — Fiber Orientation & Tract Segmentation
# ============================================================================
# bedpostx: Bayesian estimation of diffusion parameters (crossing fibers)
# xtract:   Automated white matter tract segmentation (SOTA in FSL)
#
# bedpostx output feeds into:
#   - probtrackx2 (probabilistic tractography → connectome)
#   - TRACULA (tract-specific analysis)
#   - xtract (standardized tract extraction)
#
# Runtime: bedpostx ~12-24h on CPU, ~1h on GPU; xtract ~2-4h
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"
ACQ="${2:-CHARMED}"  # Which eddy-corrected acquisition to use

DWI_DIR="${WAND_DERIVATIVES}/fsl/${SUBJECT}/ses-02/dwi"
BPX_DIR="${WAND_DERIVATIVES}/fsl/${SUBJECT}/ses-02/dwi/${ACQ}.bedpostX"

echo "=== bedpostx + xtract: ${SUBJECT} (${ACQ}) ==="

# ---------------------------------------------------------------
# Step 1: Prepare bedpostx input directory
# ---------------------------------------------------------------

BPX_INPUT="${DWI_DIR}/${ACQ}_bedpostx_input"
mkdir -p "${BPX_INPUT}"

# bedpostx expects: data.nii.gz, bvals, bvecs, nodif_brain_mask.nii.gz
cp "${DWI_DIR}/${ACQ}_eddy.nii.gz"       "${BPX_INPUT}/data.nii.gz"
cp "${DWI_DIR}/${ACQ}_eddy.eddy_rotated_bvecs" "${BPX_INPUT}/bvecs" 2>/dev/null || \
    cp "${WAND_ROOT}/${SUBJECT}/ses-02/dwi/"*${ACQ}*"dir-AP"*".bvec" "${BPX_INPUT}/bvecs"
cp "${WAND_ROOT}/${SUBJECT}/ses-02/dwi/"*${ACQ}*"dir-AP"*".bval" "${BPX_INPUT}/bvals"
cp "${DWI_DIR}/b0_brain_mask.nii.gz"     "${BPX_INPUT}/nodif_brain_mask.nii.gz"

# Verify inputs
bedpostx_datacheck "${BPX_INPUT}"

# ---------------------------------------------------------------
# Step 2: bedpostx — Bayesian crossing fiber estimation
# ---------------------------------------------------------------

if [ -d "${BPX_DIR}" ]; then
    echo "bedpostx output exists, skipping."
else
    echo "Running bedpostx (this will take several hours on CPU)..."
    # -n 3: model up to 3 crossing fibers per voxel
    # Default: ARD (automatic relevance determination) for fiber count
    bedpostx "${BPX_INPUT}" -n 3

    # bedpostx creates output at ${BPX_INPUT}.bedpostX/
    mv "${BPX_INPUT}.bedpostX" "${BPX_DIR}" 2>/dev/null || true
fi

# ---------------------------------------------------------------
# Step 3: Register to MNI (needed for xtract)
# ---------------------------------------------------------------

REG_DIR="${DWI_DIR}/reg"
mkdir -p "${REG_DIR}"

# DWI → T1 (FreeSurfer)
if [ -f "${SUBJECTS_DIR}/${SUBJECT}/mri/brain.mgz" ]; then
    echo "Registering DWI to FreeSurfer T1..."
    # Convert FreeSurfer brain to nifti
    mri_convert "${SUBJECTS_DIR}/${SUBJECT}/mri/brain.mgz" "${REG_DIR}/T1_brain.nii.gz"

    # FLIRT: DWI b=0 → T1
    flirt \
        -in "${DWI_DIR}/b0_brain" \
        -ref "${REG_DIR}/T1_brain" \
        -omat "${REG_DIR}/diff2struct.mat" \
        -out "${REG_DIR}/b0_in_struct" \
        -dof 6

    # Invert
    convert_xfm -omat "${REG_DIR}/struct2diff.mat" -inverse "${REG_DIR}/diff2struct.mat"

    # T1 → MNI (FNIRT for nonlinear)
    flirt \
        -in "${REG_DIR}/T1_brain" \
        -ref "${FSLDIR}/data/standard/MNI152_T1_1mm_brain" \
        -omat "${REG_DIR}/struct2mni_affine.mat" \
        -out "${REG_DIR}/T1_in_MNI_affine"

    fnirt \
        --in="${REG_DIR}/T1_brain" \
        --ref="${FSLDIR}/data/standard/MNI152_T1_1mm_brain" \
        --aff="${REG_DIR}/struct2mni_affine.mat" \
        --cout="${REG_DIR}/struct2mni_warp" \
        --iout="${REG_DIR}/T1_in_MNI"

    # Invert warp
    invwarp -w "${REG_DIR}/struct2mni_warp" -o "${REG_DIR}/mni2struct_warp" -r "${REG_DIR}/T1_brain"

    echo "Registration complete."
else
    echo "WARNING: FreeSurfer not done yet. Run 01_freesurfer_recon.sh first."
fi

# ---------------------------------------------------------------
# Step 4: xtract — Automated white matter tract extraction
# ---------------------------------------------------------------

XTRACT_DIR="${WAND_DERIVATIVES}/fsl/${SUBJECT}/ses-02/xtract"

if [ -d "${XTRACT_DIR}" ]; then
    echo "xtract output exists, skipping."
else
    echo "Running xtract..."
    xtract \
        -bpx "${BPX_DIR}" \
        -out "${XTRACT_DIR}" \
        -species HUMAN \
        -str "${REG_DIR}/diff2struct.mat" \
        -stdwarp "${REG_DIR}/struct2mni_warp" "${REG_DIR}/mni2struct_warp"

    # QC
    xtract_qc -d "${XTRACT_DIR}" -species HUMAN
fi

# ---------------------------------------------------------------
# Step 5: xtract_stats — Tract-specific metrics
# ---------------------------------------------------------------

echo "Extracting tract statistics..."

# Quick DTI fit for FA/MD maps
if [ ! -f "${DWI_DIR}/${ACQ}_dti_FA.nii.gz" ]; then
    dtifit \
        -k "${BPX_INPUT}/data" \
        -o "${DWI_DIR}/${ACQ}_dti" \
        -m "${BPX_INPUT}/nodif_brain_mask" \
        -r "${BPX_INPUT}/bvecs" \
        -b "${BPX_INPUT}/bvals"
fi

xtract_stats \
    -d "${XTRACT_DIR}" \
    -xtract "${XTRACT_DIR}" \
    -w "${DWI_DIR}/${ACQ}_dti_FA.nii.gz" \
    -r "${DWI_DIR}/${ACQ}_dti_MD.nii.gz" \
    -keepfiles

echo "=== bedpostx + xtract complete for ${SUBJECT} ==="
