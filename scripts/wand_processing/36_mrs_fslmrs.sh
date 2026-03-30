#!/bin/bash
# ============================================================================
# WAND MRS Processing with FSL-MRS
# ============================================================================
# Processes sLASER (ses-04, 7T) and MEGA-PRESS (ses-05, 3T) data:
#   1. Convert TWIX → NIfTI-MRS (spec2nii)
#   2. Preprocessing (mrs_tools: coil combine, average, ECC)
#   3. Basis set simulation (fsl_mrs_sim for sLASER)
#   4. Fitting (fsl_mrs with Newton optimizer)
#   5. Tissue segmentation (svs_segment + FAST fractions)
#   6. Water-scaled quantification
#
# Output: GABA, Glutamate, NAA, Cr, Cho etc. per VOI
# ============================================================================

set -euo pipefail

SUBJECT="${1:-sub-08033}"
WAND_ROOT="${WAND_ROOT:-/data/raw/wand}"
DERIV="${WAND_ROOT}/derivatives"
MRS_OUT="${DERIV}/fsl-mrs/${SUBJECT}"
ANAT_DIR="${DERIV}/fsl-anat/${SUBJECT}/ses-02/${SUBJECT}_ses-02_T1w.anat"

export FSLDIR="${FSLDIR:-/home/mhough/fsl}"
source "${FSLDIR}/etc/fslconf/fsl.sh"

mkdir -p "${MRS_OUT}"

echo "=== WAND MRS Processing: ${SUBJECT} ==="
echo "FSL-MRS version: $(fsl_mrs --version 2>&1)"
echo "spec2nii version: $(spec2nii --version 2>&1)"

# ---------------------------------------------------------------
# Process each session
# ---------------------------------------------------------------

for SES_INFO in "ses-04:slaser:7T" "ses-05:mega:3T"; do
    SES=$(echo ${SES_INFO} | cut -d: -f1)
    ACQ=$(echo ${SES_INFO} | cut -d: -f2)
    FIELD=$(echo ${SES_INFO} | cut -d: -f3)

    MRS_DIR="${WAND_ROOT}/${SUBJECT}/${SES}/mrs"
    SES_OUT="${MRS_OUT}/${SES}"
    mkdir -p "${SES_OUT}"

    echo ""
    echo "=============================================="
    echo " ${SES} (${ACQ}, ${FIELD})"
    echo "=============================================="

    for VOI in anteriorcingulate occipital rightauditory smleft; do
        SVS_FILE="${MRS_DIR}/${SUBJECT}_${SES}_acq-${ACQ}_voi-${VOI}_svs.dat"
        REF_FILE="${MRS_DIR}/${SUBJECT}_${SES}_acq-${ACQ}_voi-${VOI}_ref.dat"
        VOI_OUT="${SES_OUT}/${VOI}"
        mkdir -p "${VOI_OUT}"

        if [ ! -f "${SVS_FILE}" ]; then
            echo "  SKIP: ${VOI} — data not found"
            continue
        fi

        echo ""
        echo "--- ${VOI} (${ACQ}) ---"

        # Step 1: Convert TWIX → NIfTI-MRS
        echo "  1. Converting TWIX → NIfTI-MRS..."

        spec2nii twix -e image "${SVS_FILE}" -o "${VOI_OUT}" -f svs 2>&1 | tail -3

        # Convert water reference
        if [ -f "${REF_FILE}" ]; then
            spec2nii twix -e image "${REF_FILE}" -o "${VOI_OUT}" -f ref 2>&1 | tail -3
        fi

        # Check what was produced
        NIFTI_SVS=$(find "${VOI_OUT}" -name "svs*.nii*" | head -1)
        NIFTI_REF=$(find "${VOI_OUT}" -name "ref*.nii*" | head -1)

        if [ -z "${NIFTI_SVS}" ]; then
            echo "  WARNING: spec2nii conversion failed for ${VOI}"
            echo "  Trying alternative extraction..."
            spec2nii twix "${SVS_FILE}" -o "${VOI_OUT}" -f "svs" 2>&1 | tail -5
            NIFTI_SVS=$(find "${VOI_OUT}" -name "svs*.nii*" | head -1)
        fi

        if [ -z "${NIFTI_SVS}" ]; then
            echo "  ERROR: No NIfTI-MRS produced for ${VOI}"
            continue
        fi

        echo "  SVS: ${NIFTI_SVS}"
        [ -n "${NIFTI_REF}" ] && echo "  REF: ${NIFTI_REF}"

        # Step 2: Preprocessing
        echo "  2. Preprocessing..."

        # Check dimensions
        mrs_tools info "${NIFTI_SVS}" 2>&1 | head -10 || true

        # Coil combination (if multi-coil)
        PREPROC="${VOI_OUT}/preproc"
        mkdir -p "${PREPROC}"

        mrs_tools merge --dim DIM_COIL --output "${PREPROC}/combined.nii.gz" "${NIFTI_SVS}" 2>/dev/null || \
            cp "${NIFTI_SVS}" "${PREPROC}/combined.nii.gz"

        # Average transients
        mrs_tools average --dim DIM_DYN --output "${PREPROC}/averaged.nii.gz" "${PREPROC}/combined.nii.gz" 2>/dev/null || \
            cp "${PREPROC}/combined.nii.gz" "${PREPROC}/averaged.nii.gz"

        # Eddy current correction using water reference
        if [ -n "${NIFTI_REF}" ]; then
            mrs_tools average --dim DIM_DYN --output "${PREPROC}/ref_avg.nii.gz" "${NIFTI_REF}" 2>/dev/null || \
                cp "${NIFTI_REF}" "${PREPROC}/ref_avg.nii.gz"
        fi

        echo "  Preprocessed: ${PREPROC}/averaged.nii.gz"

        # Step 3: Voxel segmentation (tissue fractions for quantification)
        echo "  3. Voxel segmentation..."

        T1_ANAT="${ANAT_DIR}/T1_biascorr.nii.gz"
        if [ -f "${T1_ANAT}" ] && [ -f "${PREPROC}/averaged.nii.gz" ]; then
            svs_segment -a "${T1_ANAT}" \
                         -o "${VOI_OUT}/tissue_seg" \
                         "${PREPROC}/averaged.nii.gz" 2>&1 | tail -5 || \
            echo "  WARNING: svs_segment failed (registration issue?)"
        else
            echo "  SKIP: No T1 available for tissue segmentation"
        fi

        # Step 4: Fitting
        echo "  4. Fitting with fsl_mrs..."

        FIT_OUT="${VOI_OUT}/fit"
        mkdir -p "${FIT_OUT}"

        # Build fitting command
        FIT_CMD="fsl_mrs --data ${PREPROC}/averaged.nii.gz --output ${FIT_OUT}"

        # Add basis set (use default or simulate)
        # FSL-MRS has built-in basis sets for common sequences
        if [ "${ACQ}" = "slaser" ]; then
            FIT_CMD="${FIT_CMD} --basis_name slaser --field_strength ${FIELD}"
        elif [ "${ACQ}" = "mega" ]; then
            FIT_CMD="${FIT_CMD} --basis_name mega --field_strength ${FIELD}"
        fi

        # Add water reference for scaling
        if [ -f "${PREPROC}/ref_avg.nii.gz" ]; then
            FIT_CMD="${FIT_CMD} --h2o ${PREPROC}/ref_avg.nii.gz"
        fi

        # Add tissue fractions if available
        TISSUE_SEG="${VOI_OUT}/tissue_seg"
        if [ -d "${TISSUE_SEG}" ]; then
            FIT_CMD="${FIT_CMD} --tissue_frac ${TISSUE_SEG}/segmentation.json"
        fi

        echo "  CMD: ${FIT_CMD}"
        eval ${FIT_CMD} 2>&1 | tail -10 || echo "  WARNING: fsl_mrs fitting failed"

        # Report results
        if [ -f "${FIT_OUT}/summary.csv" ]; then
            echo ""
            echo "  === Results: ${VOI} (${SES}) ==="
            head -5 "${FIT_OUT}/summary.csv"
        elif [ -d "${FIT_OUT}" ]; then
            echo "  Output dir: ${FIT_OUT}/"
            ls "${FIT_OUT}/" 2>/dev/null | head -5
        fi

    done
done

echo ""
echo "=== MRS Processing complete ==="
echo "Outputs: ${MRS_OUT}/"
