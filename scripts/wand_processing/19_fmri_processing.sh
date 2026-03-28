#!/bin/bash
# ============================================================================
# Stage O: fMRI Processing — MELODIC + FEAT (FSL)
# ============================================================================
# FSL-native processing for WAND fMRI data:
#
# ses-03 (3T, TR=2s):
#   task-rest          → MELODIC ICA (resting-state networks)
#   task-categorylocaliser (2 runs) → FEAT GLM + MELODIC dimest
#   task-reversallearning → FEAT GLM + MELODIC dimest
#
# ses-06 (7T, TR=1.5s):
#   task-rest          → MELODIC ICA (test-retest with ses-03)
#
# Pipeline per run:
#   1. FEAT preprocessing (motion correction, spatial smoothing, HP filter)
#   2. MELODIC ICA (dimensionality estimation + structured noise identification)
#   3. FIX noise cleanup (optional, if FIX training data available)
#   4. FEAT GLM (task data only: design matrix → contrasts → stats)
#   5. Dual regression (project group ICA to individual)
#
# All runs produce HTML reports.
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"

FMRI_DIR="${WAND_DERIVATIVES}/fsl-fmri/${SUBJECT}"
mkdir -p "${FMRI_DIR}"

echo "=== fMRI Processing: ${SUBJECT} ==="

# ---------------------------------------------------------------
# Helper: create FEAT design for preprocessing + MELODIC
# ---------------------------------------------------------------

create_preproc_design() {
    local INPUT=$1
    local OUTPUT=$2
    local TR=$3
    local NVOLS=$4
    local HIGHRES=$5
    local DESIGN=$6

    cat > "${DESIGN}" << FSFDEOF
# FEAT design for preprocessing
set fmri(version) 6.00
set fmri(level) 1
set fmri(analysis) 7
set fmri(relative_yn) 0
set fmri(help_yn) 1
set fmri(featwatcher_yn) 1
set fmri(sscleanup_yn) 0
set fmri(outputdir) "${OUTPUT}"
set fmri(tr) ${TR}
set fmri(npts) ${NVOLS}
set fmri(ndelete) 0
set fmri(tagfirst) 1
set fmri(multiple) 1
set fmri(inputtype) 2
set fmri(filtering_yn) 1
set fmri(brain_thresh) 10
set fmri(critical_z) 5.3
set fmri(noise) 0.66
set fmri(noisear) 0.34
set fmri(newdir_yn) 0
set fmri(mc) 1
set fmri(sh_yn) 0
set fmri(regunwarp_yn) 0
set fmri(dwell) 0.00056
set fmri(te) 0.03
set fmri(signallossthresh) 10
set fmri(unwarp_dir) y-
set fmri(st) 0
set fmri(bet_yn) 1
set fmri(smooth) 5
set fmri(norm_yn) 0
set fmri(perfsub_yn) 0
set fmri(temphp_yn) 1
set fmri(templp_yn) 0
set fmri(melodic_yn) 1
set fmri(stats_yn) 0
set fmri(prewhiten_yn) 1
set fmri(motionevs) 0
set fmri(robust_yn) 0
set fmri(mixed_yn) 2
set fmri(evs_orig) 0
set fmri(evs_real) 0
set fmri(ncon_orig) 0
set fmri(ncon_real) 0
set fmri(paradigm_hp) 100
set fmri(totalVoxels) 0
set fmri(regstandard) "${FSLDIR}/data/standard/MNI152_T1_2mm_brain"
set fmri(regstandard_search) 90
set fmri(regstandard_dof) 12
set fmri(regstandard_nonlinear_yn) 1
set fmri(regstandard_nonlinear_warpres) 10
set feat_files(1) "${INPUT}"
set highres_files(1) "${HIGHRES}"
FSFDEOF
}

# ---------------------------------------------------------------
# Find T1 brain for registration
# ---------------------------------------------------------------

HIGHRES="${WAND_DERIVATIVES}/fsl-anat/${SUBJECT}/ses-03/anat/${SUBJECT}_ses-03_T1w.anat/T1_biascorr_brain"
if [ ! -f "${HIGHRES}.nii.gz" ]; then
    # Fallback to FreeSurfer
    HIGHRES="${SUBJECTS_DIR}/${SUBJECT}/mri/brain"
    if [ -f "${HIGHRES}.mgz" ]; then
        mri_convert "${HIGHRES}.mgz" "${FMRI_DIR}/T1_brain.nii.gz"
        HIGHRES="${FMRI_DIR}/T1_brain"
    else
        echo "WARNING: No T1 brain found. Registration will fail."
        HIGHRES=""
    fi
fi

# ---------------------------------------------------------------
# Process each fMRI run
# ---------------------------------------------------------------

for SES in ses-03 ses-06; do
    FUNC_DIR="${WAND_ROOT}/${SUBJECT}/${SES}/func"
    [ -d "${FUNC_DIR}" ] || continue

    for BOLD_FILE in "${FUNC_DIR}"/*bold.nii.gz; do
        [ -f "${BOLD_FILE}" ] || continue

        # Extract task name and run
        BASENAME=$(basename "${BOLD_FILE}" .nii.gz)
        TASK=$(echo "${BASENAME}" | grep -oP 'task-\K[^_]+')
        RUN=$(echo "${BASENAME}" | grep -oP 'run-\K[0-9]+' || echo "01")

        echo ""
        echo "--- ${SES} / task-${TASK} run-${RUN} ---"

        # Get TR and number of volumes
        TR=$(python3 -c "import json; d=json.load(open('${BOLD_FILE%.nii.gz}.json')); print(d.get('RepetitionTime', 2.0))" 2>/dev/null || echo "2.0")
        NVOLS=$(fslnvols "${BOLD_FILE}")
        echo "  TR=${TR}s, ${NVOLS} volumes"

        RUN_DIR="${FMRI_DIR}/${SES}/task-${TASK}_run-${RUN}"
        mkdir -p "${RUN_DIR}"

        # ---- Step 1: FEAT preprocessing + MELODIC ----
        FEAT_DIR="${RUN_DIR}.feat"
        if [ -d "${FEAT_DIR}" ]; then
            echo "  FEAT already complete, skipping"
        else
            echo "  Running FEAT preprocessing + MELODIC..."
            DESIGN="${RUN_DIR}_design.fsf"
            create_preproc_design "${BOLD_FILE}" "${RUN_DIR}" "${TR}" "${NVOLS}" "${HIGHRES}" "${DESIGN}"
            feat "${DESIGN}"
            echo "  FEAT report: ${FEAT_DIR}/report.html"
        fi

        # ---- Step 2: Standalone MELODIC for dimensionality ----
        MELODIC_DIR="${RUN_DIR}_melodic"
        if [ -d "${MELODIC_DIR}" ]; then
            echo "  MELODIC already complete"
        else
            echo "  Running standalone MELODIC (all dimest methods)..."
            # Run MELODIC with automatic dimensionality
            melodic \
                -i "${BOLD_FILE}" \
                -o "${MELODIC_DIR}" \
                --tr="${TR}" \
                --nobet \
                -a concat \
                --report \
                -v 2>&1 | tail -3

            # Also estimate dimensionality with each method
            for METHOD in lap bic mdl aic mean; do
                DIM=$(melodic \
                    -i "${BOLD_FILE}" \
                    --tr="${TR}" \
                    --nobet \
                    --dimest="${METHOD}" \
                    -o "${RUN_DIR}_dimest_${METHOD}" 2>&1 | grep -oP 'Estimated.*?(\d+)' | grep -oP '\d+$' || echo "?")
                echo "    ${METHOD}: ${DIM} components"
            done

            echo "  MELODIC report: ${MELODIC_DIR}/report/00index.html"
        fi

        # ---- Step 3: FEAT GLM for task data ----
        if [ "${TASK}" != "rest" ]; then
            EVENTS="${FUNC_DIR}/${BASENAME%_bold}_events.tsv"
            if [ -f "${EVENTS}" ]; then
                echo "  Task events found: ${EVENTS}"
                echo "  (Full FEAT GLM design requires task-specific EV setup)"
                echo "  TODO: Create task-specific .fsf design for ${TASK}"
            fi
        fi

    done
done

# ---------------------------------------------------------------
# Resting-state: dual regression if group ICA available
# ---------------------------------------------------------------

echo ""
echo "--- Dual Regression (placeholder) ---"
echo "  Requires group ICA maps from all subjects."
echo "  Run after processing all 170 subjects."

echo ""
echo "=== fMRI processing complete for ${SUBJECT} ==="
echo ""
echo "Reports:"
for REPORT in $(find "${FMRI_DIR}" -name "report.html" -o -name "00index.html" 2>/dev/null); do
    echo "  ${REPORT}"
done
