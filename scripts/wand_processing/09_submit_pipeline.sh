#!/bin/bash
# ============================================================================
# WAND Pipeline — fsl_sub Job Submission with Dependencies
# ============================================================================
# Submits the full processing pipeline as managed jobs via fsl_sub.
# Each stage waits for its dependencies before starting.
#
# On local machine: runs sequentially via shell plugin
# On DGX Spark: switch ~/.fsl_sub.yml method to 'slurm' for parallel execution
#
# Usage: ./09_submit_pipeline.sh sub-08033
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${WAND_DERIVATIVES}/logs/${SUBJECT}"
mkdir -p "${LOG_DIR}"

echo "=============================================="
echo " WAND Pipeline — fsl_sub Job Submission"
echo " Subject: ${SUBJECT}"
echo " Log dir: ${LOG_DIR}"
echo "=============================================="

# ---------------------------------------------------------------
# Stage A: fsl_anat (T1w preprocessing)
# ---------------------------------------------------------------

T1W="${WAND_ROOT}/${SUBJECT}/ses-03/anat/${SUBJECT}_ses-03_T1w.nii.gz"
ANAT_DIR="${WAND_DERIVATIVES}/fsl-anat/${SUBJECT}/ses-03/anat"
mkdir -p "${ANAT_DIR}"

JOB_ANAT=$(fsl_sub \
    -l "${LOG_DIR}" \
    -N "anat_${SUBJECT}" \
    -R 8 \
    -T 120 \
    fsl_anat -i "${T1W}" -o "${ANAT_DIR}/${SUBJECT}_ses-03_T1w" --strongbias)
echo "Stage A (fsl_anat): job ${JOB_ANAT}"

# ---------------------------------------------------------------
# Stage B: FreeSurfer recon-all (if not already running)
# ---------------------------------------------------------------

FS_DONE="${SUBJECTS_DIR}/${SUBJECT}/scripts/recon-all.done"
if [ -f "${FS_DONE}" ]; then
    echo "Stage B (recon-all): already complete, skipping"
    JOB_FS=""
else
    JOB_FS=$(fsl_sub \
        -l "${LOG_DIR}" \
        -N "recon_${SUBJECT}" \
        -R 8 \
        -T 480 \
        -s shmem,4 \
        bash "${SCRIPT_DIR}/01_freesurfer_recon.sh" "${SUBJECT}")
    echo "Stage B (recon-all): job ${JOB_FS}"
fi

# ---------------------------------------------------------------
# Stage C: DWI preprocessing (topup + eddy)
# ---------------------------------------------------------------

JOB_DWI=$(fsl_sub \
    -l "${LOG_DIR}" \
    -N "dwi_${SUBJECT}" \
    -R 8 \
    -T 60 \
    bash "${SCRIPT_DIR}/02_dwi_preproc.sh" "${SUBJECT}")
echo "Stage C (DWI preproc): job ${JOB_DWI}"

# ---------------------------------------------------------------
# Stage D: bedpostx (depends on DWI)
# ---------------------------------------------------------------

JOB_BPX=$(fsl_sub \
    -l "${LOG_DIR}" \
    -N "bedpostx_${SUBJECT}" \
    -R 12 \
    -T 1440 \
    -j "${JOB_DWI}" \
    bash "${SCRIPT_DIR}/03_bedpostx_xtract.sh" "${SUBJECT}")
echo "Stage D (bedpostx+xtract): job ${JOB_BPX} (after ${JOB_DWI})"

# ---------------------------------------------------------------
# Stage E: TRACULA (depends on FreeSurfer + bedpostx)
# ---------------------------------------------------------------

TRACULA_DEPS="${JOB_BPX}"
[ -n "${JOB_FS}" ] && TRACULA_DEPS="${TRACULA_DEPS},${JOB_FS}"

JOB_TRAC=$(fsl_sub \
    -l "${LOG_DIR}" \
    -N "tracula_${SUBJECT}" \
    -R 8 \
    -T 240 \
    -j "${TRACULA_DEPS}" \
    bash "${SCRIPT_DIR}/04_tracula.sh" "${SUBJECT}")
echo "Stage E (TRACULA): job ${JOB_TRAC} (after bedpostx+recon-all)"

# ---------------------------------------------------------------
# Stage F: Connectome (depends on bedpostx + FreeSurfer)
# ---------------------------------------------------------------

CONN_DEPS="${JOB_BPX}"
[ -n "${JOB_FS}" ] && CONN_DEPS="${CONN_DEPS},${JOB_FS}"

JOB_CONN=$(fsl_sub \
    -l "${LOG_DIR}" \
    -N "connectome_${SUBJECT}" \
    -R 8 \
    -T 480 \
    -j "${CONN_DEPS}" \
    bash "${SCRIPT_DIR}/05_connectome.sh" "${SUBJECT}")
echo "Stage F (connectome): job ${JOB_CONN} (after bedpostx+recon-all)"

# ---------------------------------------------------------------
# Stage G: Advanced FreeSurfer (depends on recon-all)
# ---------------------------------------------------------------

ADV_DEPS=""
[ -n "${JOB_FS}" ] && ADV_DEPS="-j ${JOB_FS}"

JOB_ADV=$(fsl_sub \
    -l "${LOG_DIR}" \
    -N "advanced_fs_${SUBJECT}" \
    -R 8 \
    -T 240 \
    ${ADV_DEPS} \
    bash "${SCRIPT_DIR}/08_advanced_freesurfer.sh" "${SUBJECT}")
echo "Stage G (advanced FS): job ${JOB_ADV}"

# ---------------------------------------------------------------
# Stage H: Microstructure (depends on DWI)
# ---------------------------------------------------------------

JOB_MICRO=$(fsl_sub \
    -l "${LOG_DIR}" \
    -N "microstructure_${SUBJECT}" \
    -R 8 \
    -T 120 \
    -j "${JOB_DWI}" \
    bash "${SCRIPT_DIR}/07_microstructure_comparison.sh" "${SUBJECT}")
echo "Stage H (microstructure): job ${JOB_MICRO} (after DWI preproc)"

# ---------------------------------------------------------------
# Stage I: MRS quantification (depends on fsl_anat for tissue seg)
# ---------------------------------------------------------------

echo ""
echo "--- MRS Processing ---"

MRS_DIR="${WAND_DERIVATIVES}/fsl-mrs/${SUBJECT}"
mkdir -p "${MRS_DIR}"

# MRS needs tissue segmentation from fsl_anat for water-scaled quantification
JOB_MRS=$(fsl_sub \
    -l "${LOG_DIR}" \
    -N "mrs_${SUBJECT}" \
    -R 4 \
    -T 60 \
    -j "${JOB_ANAT}" \
    bash "${SCRIPT_DIR}/10_mrs_processing.sh" "${SUBJECT}")
echo "Stage I (MRS): job ${JOB_MRS} (after fsl_anat)"

# ---------------------------------------------------------------
# Stage J: MEG source reconstruction (depends on FreeSurfer)
# ---------------------------------------------------------------

echo ""
echo "--- MEG Dynamics Pipeline ---"

MEG_DEPS=""
[ -n "${JOB_FS}" ] && MEG_DEPS="-j ${JOB_FS}"

JOB_MEG_SRC=$(fsl_sub \
    -l "${LOG_DIR}" \
    -N "meg_source_${SUBJECT}" \
    -R 8 \
    -T 60 \
    ${MEG_DEPS} \
    bash "${SCRIPT_DIR}/11_meg_source_recon.sh" "${SUBJECT}")
echo "Stage J (MEG source recon): job ${JOB_MEG_SRC}"

# ---------------------------------------------------------------
# Stage K: HMM + DyNeMo brain states (depends on MEG source recon)
# ---------------------------------------------------------------

JOB_DYNAMICS=$(fsl_sub \
    -l "${LOG_DIR}" \
    -N "dynamics_${SUBJECT}" \
    -R 8 \
    -T 120 \
    -j "${JOB_MEG_SRC}" \
    bash "${SCRIPT_DIR}/12_meg_dynamics.sh" "${SUBJECT}")
echo "Stage K (HMM/DyNeMo): job ${JOB_DYNAMICS} (after MEG source)"

# ---------------------------------------------------------------
# Stage L: FOOOF spectral parameterization (depends on MEG source)
# ---------------------------------------------------------------

JOB_FOOOF=$(fsl_sub \
    -l "${LOG_DIR}" \
    -N "fooof_${SUBJECT}" \
    -R 4 \
    -T 30 \
    -j "${JOB_MEG_SRC}" \
    bash "${SCRIPT_DIR}/13_fooof.sh" "${SUBJECT}")
echo "Stage L (FOOOF): job ${JOB_FOOOF} (after MEG source)"

# ---------------------------------------------------------------
# Stage M: TMS-EEG fitting (depends on connectome + MEG dynamics)
# ---------------------------------------------------------------

TMS_DEPS="${JOB_CONN}"
[ -n "${JOB_DYNAMICS}" ] && TMS_DEPS="${TMS_DEPS},${JOB_DYNAMICS}"

JOB_TMS=$(fsl_sub \
    -l "${LOG_DIR}" \
    -N "tms_fit_${SUBJECT}" \
    -R 8 \
    -T 240 \
    -j "${TMS_DEPS}" \
    bash "${SCRIPT_DIR}/14_tms_fitting.sh" "${SUBJECT}")
echo "Stage M (TMS fitting): job ${JOB_TMS} (after connectome + dynamics)"

# ---------------------------------------------------------------
# Stage N: Prediction features (depends on everything)
# ---------------------------------------------------------------

ALL_DEPS="${JOB_TMS},${JOB_MRS},${JOB_FOOOF},${JOB_ADV},${JOB_MICRO}"

JOB_FEATURES=$(fsl_sub \
    -l "${LOG_DIR}" \
    -N "features_${SUBJECT}" \
    -R 4 \
    -T 30 \
    -j "${ALL_DEPS}" \
    bash "${SCRIPT_DIR}/15_extract_features.sh" "${SUBJECT}")
echo "Stage N (feature extraction): job ${JOB_FEATURES} (after all)"

echo ""
echo "=============================================="
echo " All jobs submitted"
echo " Monitor: ls ${LOG_DIR}/"
echo "=============================================="
echo ""
echo " Dependency graph:"
echo ""
echo "   A (fsl_anat) ──────────── I (MRS: fsl_mrs)"
echo "   B (recon-all) ──┬── E (TRACULA)"
echo "                   ├── F (connectome) ──────── M (TMS fitting)"
echo "                   ├── G (advanced FS)              │"
echo "                   └── J (MEG source) ──┬── K (HMM/DyNeMo) ──┘"
echo "                                        └── L (FOOOF)"
echo "   C (DWI preproc) ┬── D (bedpostx) ──┬── E"
echo "                   │                   └── F"
echo "                   └── H (microstructure)"
echo ""
echo "   N (features) ← waits for ALL above"
