#!/bin/bash
# ============================================================================
# TRACULA for WAND — runs inside podman freesurfer container
# ============================================================================
# Uses CUBRIC-preprocessed CHARMED data (eddy-corrected) + ses-02 FreeSurfer
#
# Usage:
#   bash /scripts/tracula_container.sh sub-08033 [THREADS]
#
# Requires: sub-XXXXX_ses-02 recon-all complete
# ============================================================================

# Source FreeSurfer env before strict mode
# Runs in freesurfer-tracula container (FS 7.4.1 — has full TRACULA)
export FREESURFER_HOME=/usr/local/freesurfer
export SUBJECTS_DIR=/subjects
FS_FREESURFERENV_NO_OUTPUT=1 source "$FREESURFER_HOME/FreeSurferEnv.sh" 2>/dev/null || true

set -euo pipefail

SUBJECT="${1:?Usage: $0 <subject_id> [threads]}"
THREADS="${2:-8}"

WAND_ROOT=/data/raw/wand
EDDY_DIR="${WAND_ROOT}/derivatives/eddy_qc/preprocessed/${SUBJECT}"
FS_SUBJ="${SUBJECT}_ses-02"
TRACULA_DIR="/subjects/tracula"
DMRIRC="/subjects/tracula/${SUBJECT}_dmrirc"

# ---------------------------------------------------------------
# Verify inputs
# ---------------------------------------------------------------
if [ ! -f "${SUBJECTS_DIR}/${FS_SUBJ}/scripts/recon-all.done" ]; then
    echo "ERROR: recon-all not complete for ${FS_SUBJ}"
    exit 1
fi

DWI="${EDDY_DIR}/${SUBJECT}_eddy_corrected_data.nii.gz"
BVEC="${EDDY_DIR}/${SUBJECT}_eddy_corrected_data.eddy_rotated_bvecs"
BVAL="${WAND_ROOT}/${SUBJECT}/ses-02/dwi/${SUBJECT}_ses-02_acq-CHARMED_dir-AP_part-mag_dwi.bval"
MASK="${EDDY_DIR}/${SUBJECT}_b0_brain_mask.nii.gz"

for F in "$DWI" "$BVEC" "$BVAL" "$MASK"; do
    if [ ! -f "$F" ]; then
        echo "ERROR: Missing input: $F"
        exit 1
    fi
done

echo "=== TRACULA ==="
echo "Subject:    ${SUBJECT}"
echo "FS recon:   ${FS_SUBJ}"
echo "DWI:        ${DWI}"
echo "Threads:    ${THREADS}"
echo ""

mkdir -p "${TRACULA_DIR}"

# ---------------------------------------------------------------
# Skip if already done
# ---------------------------------------------------------------
if [ -f "${TRACULA_DIR}/${SUBJECT}/dpath/merged_avg33_mni_bbr.done" ]; then
    echo "TRACULA already completed for ${SUBJECT}, skipping."
    exit 0
fi

# ---------------------------------------------------------------
# Write dmrirc config
# ---------------------------------------------------------------
cat > "${DMRIRC}" << EOF
# TRACULA config for WAND ${SUBJECT}
# Using CUBRIC-preprocessed CHARMED eddy-corrected data

setenv SUBJECTS_DIR /subjects

set subjlist = ( ${FS_SUBJ} )
set dtroot = ${TRACULA_DIR}

# Pre-processed DWI (skip TRACULA preprocessing)
set dcmlist = ( ${DWI} )
set bvecfile = ${BVEC}
set bvalfile = ${BVAL}

# Already eddy-corrected by CUBRIC (topup + eddy_cuda + s2v)
set doeddy = 0
set dorotbvecs = 0

# Brain mask from CUBRIC preprocessing
set usemaskanat = 0

# Registration
set doregbbr = 1
set doregmni = 1
set doregcvs = 0

# bedpostx
set doBedpost = 1
set nburnin = 200

# Pathway reconstruction
# Use default pathways and control points (TRACULA handles this automatically)

# Number of threads
set nthreads = ${THREADS}
EOF

echo "Config written to ${DMRIRC}"

# ---------------------------------------------------------------
# Run TRACULA steps: prep → bedpost → paths
# ---------------------------------------------------------------

echo ">>> Step 1: Preprocessing (registration only, DWI already corrected)"
trac-all -prep -c "${DMRIRC}"

echo ">>> Step 2: bedpostx (GPU)"
# Use FSL's GPU bedpostx instead of TRACULA's built-in
# FSL mounted at /fsl in the container
export FSLDIR=/home/mhough/fsl
export PATH=$FSLDIR/share/fsl/bin:$FSLDIR/bin:$PATH
source $FSLDIR/etc/fslconf/fsl.sh 2>/dev/null || true
bedpostx_gpu "${TRACULA_DIR}/${FS_SUBJ}/dmri" -n 3 -model 2

echo ">>> Step 3: Pathway reconstruction"
trac-all -path -c "${DMRIRC}"

echo "=== TRACULA complete: ${SUBJECT} ==="
