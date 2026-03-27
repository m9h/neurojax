#!/bin/bash
# ============================================================================
# Stage I: MRS Quantification — fsl_mrs
# ============================================================================
# WAND MRS: sLASER acquisitions in 4 brain regions (ses-04 + ses-05)
#   - anteriorcingulate (ACC): prefrontal E/I balance
#   - occipital: visual cortex baseline
#   - rightauditory: sensory processing
#   - smleft: left sensorimotor cortex
#
# Pipeline per VOI:
#   1. fsl_mrs_preproc: coil combine, frequency/phase align, average, water removal
#   2. svs_segment: voxel mask in T1w space → tissue fractions (GM/WM/CSF)
#   3. fsl_mrs: spectral fitting (Newton) with water-scaled quantification
#
# Output: GABA, glutamate, NAA, creatine etc. concentrations per VOI
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"

ANAT_DIR="${WAND_DERIVATIVES}/fsl-anat/${SUBJECT}/ses-03/anat/${SUBJECT}_ses-03_T1w.anat"
MRS_OUT="${WAND_DERIVATIVES}/fsl-mrs/${SUBJECT}"
mkdir -p "${MRS_OUT}"

echo "=== MRS Processing: ${SUBJECT} ==="

for SES in ses-04 ses-05; do
    MRS_DIR="${WAND_ROOT}/${SUBJECT}/${SES}/mrs"
    if [ ! -d "${MRS_DIR}" ]; then
        echo "  ${SES}: no MRS data, skipping"
        continue
    fi

    echo ""
    echo "--- ${SES} ---"
    SES_OUT="${MRS_OUT}/${SES}/mrs"
    mkdir -p "${SES_OUT}"

    # Process each VOI
    for SVS_FILE in "${MRS_DIR}"/*_svs.dat; do
        [ -f "${SVS_FILE}" ] || continue

        # Extract VOI name from filename
        VOI=$(basename "${SVS_FILE}" | sed "s/${SUBJECT}_${SES}_acq-slaser_voi-//;s/_svs.dat//")
        REF_FILE="${MRS_DIR}/${SUBJECT}_${SES}_acq-slaser_voi-${VOI}_ref.dat"

        echo "  VOI: ${VOI}"
        VOI_OUT="${SES_OUT}/${VOI}"
        mkdir -p "${VOI_OUT}"

        # Step 1: Preprocessing
        if [ -f "${REF_FILE}" ]; then
            echo "    Preprocessing with water reference..."
            fsl_mrs_preproc \
                --data "${SVS_FILE}" \
                --reference "${REF_FILE}" \
                --output "${VOI_OUT}/preproc" \
                --report \
                --overwrite 2>&1 | tail -3
        else
            echo "    WARNING: No water reference for ${VOI}, skipping preproc"
            continue
        fi

        # Step 2: SVS segmentation (voxel mask + tissue fractions)
        if [ -d "${ANAT_DIR}" ]; then
            echo "    SVS segmentation..."
            PREPROC_FILE=$(find "${VOI_OUT}/preproc" -name "*.nii*" | head -1)
            if [ -n "${PREPROC_FILE}" ]; then
                svs_segment \
                    "${PREPROC_FILE}" \
                    -a "${ANAT_DIR}" \
                    -o "${VOI_OUT}" 2>&1 | tail -2

                echo "    Tissue fractions saved to ${VOI_OUT}/"
            fi
        else
            echo "    WARNING: fsl_anat not complete, skipping tissue segmentation"
        fi

        # Step 3: Spectral fitting
        echo "    Fitting spectra..."
        # Basis set — use FSL's default or specify custom
        # For 3T sLASER, use the standard basis set
        BASIS_DIR="${FSLDIR}/data/mrs/basis"
        if [ ! -d "${BASIS_DIR}" ]; then
            echo "    WARNING: Basis set not found at ${BASIS_DIR}"
            echo "    You may need to generate one with: fsl_mrs_sim"
            continue
        fi

        TISSUE_FRAC_FILE=$(find "${VOI_OUT}" -name "*tissue_fractions*" -o -name "*segmentation*" 2>/dev/null | head -1)
        TISSUE_ARGS=""
        if [ -n "${TISSUE_FRAC_FILE}" ]; then
            TISSUE_ARGS="--tissue_frac ${TISSUE_FRAC_FILE}"
        fi

        PREPROC_FILE=$(find "${VOI_OUT}/preproc" -name "*.nii*" | head -1)
        if [ -n "${PREPROC_FILE}" ]; then
            fsl_mrs \
                --data "${PREPROC_FILE}" \
                --basis "${BASIS_DIR}" \
                --output "${VOI_OUT}/fit" \
                --algo Newton \
                --report \
                ${TISSUE_ARGS} \
                --overwrite 2>&1 | tail -3

            echo "    Fit complete: ${VOI_OUT}/fit/"
        fi
    done
done

# Summarize across VOIs
echo ""
echo "--- Summary ---"
fsl_mrs_summarise \
    --dir "${MRS_OUT}" \
    --output "${MRS_OUT}/summary.csv" 2>&1 | tail -3 || \
    echo "  (fsl_mrs_summarise may need all VOIs to be fitted first)"

echo ""
echo "=== MRS processing complete for ${SUBJECT} ==="
echo "Output: ${MRS_OUT}/"
