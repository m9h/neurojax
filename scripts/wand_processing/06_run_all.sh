#!/bin/bash
# ============================================================================
# WAND Processing Pipeline — Run All Stages for One Subject
# ============================================================================
# SOTA pipeline: FreeSurfer 8.2.0 + FSL 6.0.7 + TRACULA
#
# Usage: ./06_run_all.sh sub-08033
#
# Stages:
#   01: FreeSurfer recon-all      (~6-8h)
#   02: DWI preprocessing         (~30-60min)
#   03: bedpostx + xtract         (~12-24h bedpostx, ~2-4h xtract)
#   04: TRACULA                   (~2-4h)
#   05: Connectome construction   (~4-8h)
#
# Total: ~24-48 hours for one subject (CPU)
# With GPU bedpostx: ~12-20 hours
# ============================================================================

set -euo pipefail

SUBJECT="${1:?Usage: $0 <subject_id>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo " WAND SOTA Processing Pipeline"
echo " Subject: ${SUBJECT}"
echo " Started: $(date)"
echo "========================================"

# Stage 1: FreeSurfer
echo ""
echo ">>> Stage 1: FreeSurfer recon-all"
bash "${SCRIPT_DIR}/01_freesurfer_recon.sh" "${SUBJECT}"

# Stage 2: DWI preprocessing
echo ""
echo ">>> Stage 2: DWI preprocessing (topup + eddy)"
bash "${SCRIPT_DIR}/02_dwi_preproc.sh" "${SUBJECT}"

# Stage 3: bedpostx + xtract
echo ""
echo ">>> Stage 3: bedpostx + xtract"
bash "${SCRIPT_DIR}/03_bedpostx_xtract.sh" "${SUBJECT}"

# Stage 4: TRACULA
echo ""
echo ">>> Stage 4: TRACULA"
bash "${SCRIPT_DIR}/04_tracula.sh" "${SUBJECT}"

# Stage 5: Connectome
echo ""
echo ">>> Stage 5: Connectome construction"
bash "${SCRIPT_DIR}/05_connectome.sh" "${SUBJECT}"

echo ""
echo "========================================"
echo " WAND Processing Complete"
echo " Subject: ${SUBJECT}"
echo " Finished: $(date)"
echo "========================================"
echo ""
echo "Outputs:"
echo "  FreeSurfer: ${SUBJECTS_DIR}/${SUBJECT}/"
echo "  DWI/FSL:    ${WAND_DERIVATIVES}/fsl/${SUBJECT}/"
echo "  TRACULA:    ${WAND_DERIVATIVES}/tracula/${SUBJECT}/"
echo "  xtract:     ${WAND_DERIVATIVES}/fsl/${SUBJECT}/ses-02/xtract/"
echo "  Connectome: ${WAND_DERIVATIVES}/connectome/${SUBJECT}/"
echo ""
echo "Next steps:"
echo "  1. Run AxCaliber/CHARMED microstructure estimation (sbi4dwi)"
echo "  2. Load connectome in neurojax:"
echo "     from neurojax.io.connectome import BIDSConnectomeLoader"
echo "     loader = BIDSConnectomeLoader('derivatives/connectome', '${SUBJECT#sub-}')"
echo "  3. Run MEG analysis: WANDMEGLoader → HMM/DyNeMo"
echo "  4. Run TMS-EEG fitting: VbjaxFitnessAdapter + TMSProtocol"
