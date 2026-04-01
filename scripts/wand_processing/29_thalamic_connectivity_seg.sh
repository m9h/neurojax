#!/bin/bash
# ============================================================================
# Thalamic Connectivity-Based Segmentation (Johansen-Berg & Behrens)
# ============================================================================
# Runs inside freesurfer-tracula container (FS 7.4.1 + FSL)
#
# Classifies thalamic voxels by cortical connectivity using probtrackx2.
# Produces 7-region functional parcellation per hemisphere:
#   1=Prefrontal, 2=Premotor, 3=Primary Motor, 4=Somatosensory,
#   5=Posterior Parietal, 6=Temporal, 7=Occipital
#
# References:
#   Behrens et al. 2003, Nature Neuroscience 6(7):750-757
#   Johansen-Berg et al. 2005, Cerebral Cortex 15(1):31-39
#
# Usage:
#   bash /scripts/29_thalamic_connectivity_seg.sh sub-08033 [ses-02] [NSAMPLES]
#
# Requires: recon-all + TRACULA (bedpostX) complete for the subject
# ============================================================================

export FREESURFER_HOME=/usr/local/freesurfer
export FSLDIR=/home/mhough/fsl
source $FSLDIR/etc/fslconf/fsl.sh 2>/dev/null || true
source $FREESURFER_HOME/SetUpFreeSurfer.sh 2>/dev/null || true
export SUBJECTS_DIR=/subjects

set -euo pipefail

SUBJECT="${1:?Usage: $0 <subject_id> [session] [nsamples]}"
SES="${2:-ses-02}"
NSAMPLES="${3:-5000}"

FS_SUBJ="${SUBJECT}_${SES}"
FS_DIR="/subjects/${FS_SUBJ}"
TRACULA_DIR="/subjects/tracula/${FS_SUBJ}"
BPX_DIR="${TRACULA_DIR}/dmri.bedpostX"
XFM_DIR="${TRACULA_DIR}/dmri/xfms"
DIFF_REF="${TRACULA_DIR}/dmri/nodif_brain_mask.nii.gz"

OUTDIR="/subjects/thalamic_segmentation/${FS_SUBJ}"
mkdir -p ${OUTDIR}/{masks,targets,probtrackx/{lh,rh},results,xfms,qc}

echo "=== Thalamic Connectivity Segmentation ==="
echo "Subject:  ${FS_SUBJ}"
echo "Samples:  ${NSAMPLES}"
echo "Output:   ${OUTDIR}"
echo ""

# Verify inputs
for F in "${FS_DIR}/mri/aseg.mgz" "${FS_DIR}/mri/aparc+aseg.mgz" \
         "${BPX_DIR}/merged_th1samples.nii.gz" "${DIFF_REF}" \
         "${XFM_DIR}/anatorig2diff.bbr.lta"; do
    [ -f "$F" ] || { echo "ERROR: Missing $F"; exit 1; }
done

# ---------------------------------------------------------------
# Step 1: Extract thalamus seed masks
# ---------------------------------------------------------------
echo ">>> Step 1: Thalamus seed masks"

for HEMI_INFO in "lh:10" "rh:49"; do
    HEMI="${HEMI_INFO%%:*}"
    LABEL="${HEMI_INFO##*:}"

    if [ ! -f "${OUTDIR}/masks/${HEMI}_thalamus_diff.nii.gz" ]; then
        mri_binarize --i ${FS_DIR}/mri/aseg.mgz \
            --match ${LABEL} \
            --o ${OUTDIR}/masks/${HEMI}_thalamus_anat.nii.gz

        mri_vol2vol \
            --mov ${OUTDIR}/masks/${HEMI}_thalamus_anat.nii.gz \
            --targ ${DIFF_REF} \
            --lta ${XFM_DIR}/anatorig2diff.bbr.lta \
            --o ${OUTDIR}/masks/${HEMI}_thalamus_diff.nii.gz \
            --nearest

        fslmaths ${OUTDIR}/masks/${HEMI}_thalamus_diff.nii.gz \
            -thr 0.5 -bin ${OUTDIR}/masks/${HEMI}_thalamus_diff.nii.gz
    fi

    NVOX=$(fslstats ${OUTDIR}/masks/${HEMI}_thalamus_diff.nii.gz -V | awk '{print $1}')
    echo "  ${HEMI} thalamus: ${NVOX} voxels in diff space"
done

# ---------------------------------------------------------------
# Step 2: Create 7 cortical targets per hemisphere
# ---------------------------------------------------------------
echo ""
echo ">>> Step 2: Cortical target masks"

# DK label IDs (lh; rh = lh + 1000)
declare -A DK_TARGETS
DK_TARGETS[prefrontal]="1028 1027 1003 1018 1020 1019 1012 1014 1032 1026 1002"
DK_TARGETS[somatosensory]="1022"
DK_TARGETS[posterior_parietal]="1029 1008 1025 1031"
DK_TARGETS[temporal]="1030 1015 1009 1001 1007 1034 1006 1033 1016"
DK_TARGETS[occipital]="1011 1013 1005 1021"

for HEMI in lh rh; do
    echo "  ${HEMI}:"

    # Offset for right hemisphere
    if [ "$HEMI" = "rh" ]; then OFFSET=1000; else OFFSET=0; fi

    # --- DK-based targets ---
    for TARGET in prefrontal somatosensory posterior_parietal temporal occipital; do
        if [ ! -f "${OUTDIR}/targets/${HEMI}_${TARGET}_diff.nii.gz" ]; then
            # Build match list with hemisphere offset
            MATCH_IDS=""
            for ID in ${DK_TARGETS[$TARGET]}; do
                MATCH_IDS="${MATCH_IDS} --match $((ID + OFFSET))"
            done

            mri_binarize --i ${FS_DIR}/mri/aparc+aseg.mgz \
                ${MATCH_IDS} \
                --o ${OUTDIR}/targets/${HEMI}_${TARGET}_anat.nii.gz

            mri_vol2vol \
                --mov ${OUTDIR}/targets/${HEMI}_${TARGET}_anat.nii.gz \
                --targ ${DIFF_REF} \
                --lta ${XFM_DIR}/anatorig2diff.bbr.lta \
                --o ${OUTDIR}/targets/${HEMI}_${TARGET}_diff.nii.gz \
                --nearest

            fslmaths ${OUTDIR}/targets/${HEMI}_${TARGET}_diff.nii.gz \
                -thr 0.5 -bin ${OUTDIR}/targets/${HEMI}_${TARGET}_diff.nii.gz
        fi
        NVOX=$(fslstats ${OUTDIR}/targets/${HEMI}_${TARGET}_diff.nii.gz -V | awk '{print $1}')
        echo "    ${TARGET}: ${NVOX} voxels"
    done

    # --- BA-based targets (motor/premotor) ---
    for BA_TARGET in "premotor:BA6" "primary_motor:BA4a BA4p"; do
        TARGET="${BA_TARGET%%:*}"
        BA_LABELS="${BA_TARGET##*:}"

        if [ ! -f "${OUTDIR}/targets/${HEMI}_${TARGET}_diff.nii.gz" ]; then
            # Initialize empty volume
            mri_binarize --i ${FS_DIR}/mri/aseg.mgz --match 0 \
                --o ${OUTDIR}/targets/${HEMI}_${TARGET}_anat.nii.gz

            for BA in ${BA_LABELS}; do
                LABEL_FILE="${FS_DIR}/label/${HEMI}.${BA}_exvivo.label"
                if [ -f "$LABEL_FILE" ]; then
                    mri_label2vol \
                        --label ${LABEL_FILE} \
                        --temp ${FS_DIR}/mri/aseg.mgz \
                        --subject ${FS_SUBJ} \
                        --hemi ${HEMI} \
                        --proj frac 0 1 0.01 \
                        --fillthresh 0.3 \
                        --o ${OUTDIR}/targets/${HEMI}_${BA}_vol.nii.gz \
                        --identity

                    fslmaths ${OUTDIR}/targets/${HEMI}_${TARGET}_anat.nii.gz \
                        -add ${OUTDIR}/targets/${HEMI}_${BA}_vol.nii.gz \
                        -bin ${OUTDIR}/targets/${HEMI}_${TARGET}_anat.nii.gz

                    rm -f ${OUTDIR}/targets/${HEMI}_${BA}_vol.nii.gz
                else
                    echo "    WARNING: ${LABEL_FILE} not found"
                fi
            done

            mri_vol2vol \
                --mov ${OUTDIR}/targets/${HEMI}_${TARGET}_anat.nii.gz \
                --targ ${DIFF_REF} \
                --lta ${XFM_DIR}/anatorig2diff.bbr.lta \
                --o ${OUTDIR}/targets/${HEMI}_${TARGET}_diff.nii.gz \
                --nearest

            fslmaths ${OUTDIR}/targets/${HEMI}_${TARGET}_diff.nii.gz \
                -thr 0.5 -bin ${OUTDIR}/targets/${HEMI}_${TARGET}_diff.nii.gz
        fi
        NVOX=$(fslstats ${OUTDIR}/targets/${HEMI}_${TARGET}_diff.nii.gz -V | awk '{print $1}')
        echo "    ${TARGET}: ${NVOX} voxels"
    done

    # Remove prefrontal voxels that overlap with premotor (premotor takes priority)
    fslmaths ${OUTDIR}/targets/${HEMI}_prefrontal_diff.nii.gz \
        -sub ${OUTDIR}/targets/${HEMI}_premotor_diff.nii.gz \
        -thr 0.5 -bin ${OUTDIR}/targets/${HEMI}_prefrontal_diff.nii.gz

    # Write target list (order determines label values in find_the_biggest)
    cat > ${OUTDIR}/targets/${HEMI}_target_list.txt << TARGETS
${OUTDIR}/targets/${HEMI}_prefrontal_diff.nii.gz
${OUTDIR}/targets/${HEMI}_premotor_diff.nii.gz
${OUTDIR}/targets/${HEMI}_primary_motor_diff.nii.gz
${OUTDIR}/targets/${HEMI}_somatosensory_diff.nii.gz
${OUTDIR}/targets/${HEMI}_posterior_parietal_diff.nii.gz
${OUTDIR}/targets/${HEMI}_temporal_diff.nii.gz
${OUTDIR}/targets/${HEMI}_occipital_diff.nii.gz
TARGETS
done

# ---------------------------------------------------------------
# Step 3: Run probtrackx2
# ---------------------------------------------------------------
echo ""
echo ">>> Step 3: probtrackx2"

for HEMI in lh rh; do
    if [ ! -f "${OUTDIR}/probtrackx/${HEMI}/fdt_paths.nii.gz" ]; then
        echo "  Running ${HEMI}..."
        probtrackx2_gpu11.0 \
            -x ${OUTDIR}/masks/${HEMI}_thalamus_diff.nii.gz \
            -s ${BPX_DIR}/merged \
            -m ${BPX_DIR}/nodif_brain_mask.nii.gz \
            --dir=${OUTDIR}/probtrackx/${HEMI} \
            --targetmasks=${OUTDIR}/targets/${HEMI}_target_list.txt \
            --os2t --s2tastext \
            -l --onewaycondition --forcedir \
            -c 0.2 -S 2000 --steplength=0.5 \
            -P ${NSAMPLES} --fibthresh=0.01 \
            --opd
    else
        echo "  ${HEMI}: already complete"
    fi
done

# ---------------------------------------------------------------
# Step 4: Winner-takes-all classification
# ---------------------------------------------------------------
echo ""
echo ">>> Step 4: Classification (find_the_biggest)"

for HEMI in lh rh; do
    find_the_biggest \
        ${OUTDIR}/probtrackx/${HEMI}/seeds_to_* \
        ${OUTDIR}/results/${HEMI}_thalamus_seg.nii.gz

    # Mask to thalamus only
    fslmaths ${OUTDIR}/results/${HEMI}_thalamus_seg.nii.gz \
        -mas ${OUTDIR}/masks/${HEMI}_thalamus_diff.nii.gz \
        ${OUTDIR}/results/${HEMI}_thalamus_seg.nii.gz

    # Report classification
    TOTAL=$(fslstats ${OUTDIR}/masks/${HEMI}_thalamus_diff.nii.gz -V | awk '{print $1}')
    CLASSIFIED=$(fslstats ${OUTDIR}/results/${HEMI}_thalamus_seg.nii.gz -V | awk '{print $1}')
    echo "  ${HEMI}: ${CLASSIFIED}/${TOTAL} voxels classified"
done

# ---------------------------------------------------------------
# Step 5: Warp results to anatomical space
# ---------------------------------------------------------------
echo ""
echo ">>> Step 5: Warp to anatomical space"

for HEMI in lh rh; do
    mri_vol2vol \
        --mov ${OUTDIR}/results/${HEMI}_thalamus_seg.nii.gz \
        --targ ${FS_DIR}/mri/orig.mgz \
        --lta ${XFM_DIR}/diff2anatorig.bbr.lta \
        --o ${OUTDIR}/results/${HEMI}_thalamus_seg_anat.nii.gz \
        --nearest
done

echo ""
echo "=== Thalamic connectivity segmentation complete ==="
echo "Labels: 1=Prefrontal 2=Premotor 3=Motor 4=Somatosensory"
echo "        5=Post.Parietal 6=Temporal 7=Occipital"
echo "Results: ${OUTDIR}/results/"
