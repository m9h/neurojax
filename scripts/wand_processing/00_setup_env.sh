#!/bin/bash
# ============================================================================
# WAND Processing Pipeline — Environment Setup
# ============================================================================
# Sources: FSL 6.0.7, FreeSurfer 8.2.0, TRACULA
#
# Usage: source scripts/wand_processing/00_setup_env.sh
# ============================================================================

# FSL
export FSLDIR="${FSLDIR:-/Users/mhough/fsl}"
source "${FSLDIR}/etc/fslconf/fsl.sh"
export PATH="${FSLDIR}/bin:${PATH}"

# FreeSurfer
export FREESURFER_HOME="${FREESURFER_HOME:-/Applications/freesurfer/8.2.0}"
# Note: 8.1.0 also available at /Applications/freesurfer/8.1.0
# FS 8.2.0 SetUpFreeSurfer.sh has unbound variable bug — temporarily disable strict mode
set +u
source "${FREESURFER_HOME}/SetUpFreeSurfer.sh" 2>/dev/null
set -u

# Subjects directory (FreeSurfer output)
export SUBJECTS_DIR="${SUBJECTS_DIR:-/Users/mhough/dev/wand/derivatives/freesurfer}"
mkdir -p "$SUBJECTS_DIR"

# WAND paths
export WAND_ROOT="${WAND_ROOT:-/Users/mhough/dev/wand}"
export WAND_DERIVATIVES="${WAND_ROOT}/derivatives"
mkdir -p "${WAND_DERIVATIVES}/fsl"
mkdir -p "${WAND_DERIVATIVES}/freesurfer"
mkdir -p "${WAND_DERIVATIVES}/tracula"
mkdir -p "${WAND_DERIVATIVES}/connectome"

# Number of parallel threads (adjust for your machine)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NSLOTS="${NSLOTS:-4}"
# FSL eddy uses --nthr, FreeSurfer uses -openmp, both respect OMP_NUM_THREADS
export FSL_EDDY_THREADS="${OMP_NUM_THREADS}"

echo "=== WAND Processing Environment ==="
echo "FSL:         $(cat ${FSLDIR}/etc/fslversion)"
echo "FreeSurfer:  $(freesurfer --version 2>&1 | head -1)"
echo "WAND root:   ${WAND_ROOT}"
echo "Derivatives:  ${WAND_DERIVATIVES}"
echo "SUBJECTS_DIR: ${SUBJECTS_DIR}"
echo "Threads:      ${OMP_NUM_THREADS}"
echo "===================================="
