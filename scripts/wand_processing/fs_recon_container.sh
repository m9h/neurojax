#!/bin/bash
# ============================================================================
# FreeSurfer recon-all for WAND — runs inside podman freesurfer container
# ============================================================================
# Two modes per subject:
#   ses-02: Connectom T1w only (for DWI/tractography integration)
#   ses-03: Prisma T1w + T2w with -T2pial -cw256 (for cortical morphometry)
#
# Usage:
#   bash /scripts/fs_recon_container.sh sub-08033 ses-02 [THREADS]
#   bash /scripts/fs_recon_container.sh sub-08033 ses-03 [THREADS]
#
# Output subject IDs: sub-08033_ses-02, sub-08033_ses-03
# ============================================================================

# Source FreeSurfer env BEFORE strict mode (has unbound variable bugs)
export FREESURFER_HOME=/usr/local/freesurfer/8.2.0-1
export SUBJECTS_DIR=/subjects
FS_FREESURFERENV_NO_OUTPUT=1 source "$FREESURFER_HOME/FreeSurferEnv.sh" 2>/dev/null || true

set -euo pipefail

SUBJECT="${1:?Usage: $0 <subject_id> <ses-02|ses-03> [threads]}"
SESSION="${2:?Usage: $0 <subject_id> <ses-02|ses-03> [threads]}"
THREADS="${3:-8}"

WAND_ROOT=/data/raw/wand
SUBJ_ID="${SUBJECT}_${SESSION}"

# ---------------------------------------------------------------
# Skip if already done
# ---------------------------------------------------------------
if [ -f "${SUBJECTS_DIR}/${SUBJ_ID}/scripts/recon-all.done" ]; then
    echo "recon-all already completed for ${SUBJ_ID}, skipping."
else
    # ---------------------------------------------------------------
    # Build recon-all arguments based on session
    # ---------------------------------------------------------------
    T1W="${WAND_ROOT}/${SUBJECT}/${SESSION}/anat/${SUBJECT}_${SESSION}_T1w.nii.gz"
    if [ ! -f "$T1W" ]; then
        echo "ERROR: No T1w found at ${T1W}"
        exit 1
    fi

    RECON_ARGS="-all -s ${SUBJ_ID} -i ${T1W} -parallel -openmp ${THREADS}"

    T2W=""
    if [ "$SESSION" = "ses-03" ]; then
        T2W="${WAND_ROOT}/${SUBJECT}/ses-03/anat/${SUBJECT}_ses-03_rec-nlgradcorr_T2w.nii.gz"
        if [ -f "$T2W" ]; then
            RECON_ARGS="${RECON_ARGS} -T2 ${T2W} -T2pial"
        fi
        RECON_ARGS="${RECON_ARGS} -cw256"
    elif [ "$SESSION" = "ses-04" ]; then
        RECON_ARGS="${RECON_ARGS} -hires"
    fi

    echo "=== FreeSurfer recon-all ==="
    echo "Subject:  ${SUBJ_ID}"
    echo "T1w:      ${T1W}"
    [ -n "${T2W}" ] && [ -f "${T2W}" ] && echo "T2w:      ${T2W}"
    echo "Threads:  ${THREADS}"
    echo "Output:   ${SUBJECTS_DIR}/${SUBJ_ID}"
    echo ""

    recon-all ${RECON_ARGS}
    echo "=== recon-all complete for ${SUBJ_ID} ==="
fi

# ---------------------------------------------------------------
# Advanced segmentations (run after recon-all)
# ---------------------------------------------------------------
echo ""
echo "=== Advanced segmentations: ${SUBJ_ID} ==="

# Hippocampal subfields + amygdala nuclei
if [ ! -f "${SUBJECTS_DIR}/${SUBJ_ID}/mri/lh.hippoSfVolumes-T1.v22.txt" ]; then
    echo ">>> Hippocampal subfields + amygdala nuclei"
    segmentHA_T1.sh ${SUBJ_ID}
fi

# Thalamic nuclei
if [ ! -f "${SUBJECTS_DIR}/${SUBJ_ID}/mri/ThalamicNuclei.v13.T1.FSvoxelSpace.mgz" ]; then
    echo ">>> Thalamic nuclei"
    segmentThalamicNuclei.sh ${SUBJ_ID}
fi

# Hypothalamic subunits
if [ ! -f "${SUBJECTS_DIR}/${SUBJ_ID}/mri/hypothalamic_subunits_seg.v1.mgz" ]; then
    echo ">>> Hypothalamic subunits"
    mri_segment_hypothalamic_subunits --s ${SUBJ_ID} --sd ${SUBJECTS_DIR}
fi

# Brainstem substructures
if [ ! -f "${SUBJECTS_DIR}/${SUBJ_ID}/mri/brainstemSsVolumes.v2.txt" ]; then
    echo ">>> Brainstem substructures"
    segmentBS.sh ${SUBJ_ID}
fi

echo "=== Done: ${SUBJ_ID} ==="
