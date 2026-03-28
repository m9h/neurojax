#!/bin/bash
# ============================================================================
# Stage 28: Post-Eddy Diffusion Chain
# ============================================================================
# Run after CHARMED eddy completes. Chains:
#   1. dtifit (FA, MD, V1)
#   2. bedpostx (crossing fibers)
#   3. AxCaliber topup + eddy
#   4. dtifit on AxCaliber (for QC)
#
# Usage: ./28_post_eddy_chain.sh sub-08033
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"

WAND="${WAND_ROOT}"
DWI_DIR="${WAND}/${SUBJECT}/ses-02/dwi"
DWI_OUT="${WAND_DERIVATIVES}/fsl-dwi/${SUBJECT}/ses-02/dwi"
LOGS="${WAND_DERIVATIVES}/logs/${SUBJECT}"
mkdir -p "${DWI_OUT}" "${LOGS}"

# ---------------------------------------------------------------
# Verify CHARMED eddy completed
# ---------------------------------------------------------------
CHARMED_EDDY="${DWI_OUT}/CHARMED_eddy.nii.gz"
if [ ! -f "${CHARMED_EDDY}" ]; then
    echo "ERROR: CHARMED eddy output not found: ${CHARMED_EDDY}"
    echo "Wait for eddy to complete before running this script."
    exit 1
fi
echo "=== Post-Eddy Chain: ${SUBJECT} ==="
echo "  CHARMED eddy: ${CHARMED_EDDY}"

# ---------------------------------------------------------------
# Step 1: dtifit on CHARMED
# ---------------------------------------------------------------
echo ""
echo "--- Step 1: dtifit (CHARMED) ---"

DTIFIT_OUT="${DWI_OUT}/dtifit"
mkdir -p "${DTIFIT_OUT}"

JOB_DTIFIT=$(fsl_sub -l "${LOGS}" -N dtifit_charmed -T 15 \
    dtifit \
    --data="${CHARMED_EDDY}" \
    --mask="${DWI_OUT}/b0_brain_mask.nii.gz" \
    --bvecs="${DWI_OUT}/CHARMED_eddy.eddy_rotated_bvecs" \
    --bvals="${DWI_DIR}/${SUBJECT}_ses-02_acq-CHARMED_dir-AP_part-mag_dwi.bval" \
    --out="${DTIFIT_OUT}/${SUBJECT}_CHARMED" \
    --save_tensor \
    --sse)

echo "  dtifit job: ${JOB_DTIFIT}"
echo "  Outputs: FA, MD, V1, L1-3, tensor, SSE"

# ---------------------------------------------------------------
# Step 2: bedpostx (crossing fibers) — depends on dtifit
# ---------------------------------------------------------------
echo ""
echo "--- Step 2: bedpostx (CHARMED) ---"

# bedpostx expects a specific directory structure
BEDPOSTX_DIR="${DWI_OUT}/bedpostx_input"
mkdir -p "${BEDPOSTX_DIR}"

# Create bedpostx input directory with symlinks
ln -sf "${CHARMED_EDDY}" "${BEDPOSTX_DIR}/data.nii.gz"
ln -sf "${DWI_OUT}/b0_brain_mask.nii.gz" "${BEDPOSTX_DIR}/nodif_brain_mask.nii.gz"
ln -sf "${DWI_OUT}/CHARMED_eddy.eddy_rotated_bvecs" "${BEDPOSTX_DIR}/bvecs"
ln -sf "${DWI_DIR}/${SUBJECT}_ses-02_acq-CHARMED_dir-AP_part-mag_dwi.bval" "${BEDPOSTX_DIR}/bvals"

# Run bedpostx with 3 fibers (multi-shell data supports this)
JOB_BEDPOSTX=$(fsl_sub -l "${LOGS}" -N bedpostx_charmed -T 720 -j "${JOB_DTIFIT}" \
    bedpostx "${BEDPOSTX_DIR}" -n 3 --model=2)

echo "  bedpostx job: ${JOB_BEDPOSTX} (depends on ${JOB_DTIFIT})"
echo "  Model 2 (multi-shell), 3 fibers"
echo "  WARNING: bedpostx may take 12-24 hours on CPU"

# ---------------------------------------------------------------
# Step 3: AxCaliber topup
# ---------------------------------------------------------------
echo ""
echo "--- Step 3: AxCaliber topup ---"

AXC_OUT="${WAND_DERIVATIVES}/fsl-dwi/${SUBJECT}/ses-02/dwi/axcaliber"
mkdir -p "${AXC_OUT}"

# Concatenate all AxCaliber b0s for topup
# Extract b0 from each AxCaliber shell + PA reference
echo "  Extracting b0 volumes from AxCaliber acquisitions..."

B0_LIST=""
for i in 1 2 3 4; do
    AXC_FILE="${DWI_DIR}/${SUBJECT}_ses-02_acq-AxCaliber${i}_dir-AP_part-mag_dwi"
    if [ -f "${AXC_FILE}.nii.gz" ]; then
        # Find first b0 index
        B0_IDX=$(python3 -c "
import numpy as np
bvals = np.loadtxt('${AXC_FILE}.bval')
b0_idx = np.where(bvals < 50)[0]
print(b0_idx[0])
")
        fslroi "${AXC_FILE}" "${AXC_OUT}/b0_axc${i}" "${B0_IDX}" 1
        B0_LIST="${B0_LIST} ${AXC_OUT}/b0_axc${i}"
        echo "    AxCaliber${i}: b0 at index ${B0_IDX}"
    fi
done

# PA reference
PA_FILE="${DWI_DIR}/${SUBJECT}_ses-02_acq-AxCaliberRef_dir-PA_part-mag_dwi"
if [ -f "${PA_FILE}.nii.gz" ]; then
    fslroi "${PA_FILE}" "${AXC_OUT}/b0_pa" 0 1
    B0_LIST="${B0_LIST} ${AXC_OUT}/b0_pa"
    echo "    PA reference: extracted"
fi

# Merge AP + PA b0s
fslmerge -t "${AXC_OUT}/b0_all" ${B0_LIST}
echo "  Merged b0s: $(fslnvols ${AXC_OUT}/b0_all) volumes"

# Create acqparams for topup
# AP: 0 -1 0 readout_time, PA: 0 1 0 readout_time
# Get readout time from JSON
READOUT=$(python3 -c "
import json
d = json.load(open('${DWI_DIR}/${SUBJECT}_ses-02_acq-AxCaliber1_dir-AP_part-mag_dwi.json'))
print(d.get('TotalReadoutTime', 0.05))
")
echo "  Readout time: ${READOUT}s"

N_AP=$(echo "${B0_LIST}" | wc -w)
N_AP=$((N_AP - 1))  # subtract PA
{
    for i in $(seq 1 ${N_AP}); do
        echo "0 -1 0 ${READOUT}"
    done
    echo "0 1 0 ${READOUT}"
} > "${AXC_OUT}/acqparams.txt"

# Run topup
JOB_TOPUP_AXC=$(fsl_sub -l "${LOGS}" -N topup_axcaliber -T 30 \
    topup \
    --imain="${AXC_OUT}/b0_all" \
    --datain="${AXC_OUT}/acqparams.txt" \
    --config=b02b0.cnf \
    --out="${AXC_OUT}/topup_results" \
    --iout="${AXC_OUT}/b0_corrected")

echo "  topup job: ${JOB_TOPUP_AXC}"

# ---------------------------------------------------------------
# Step 4: AxCaliber concatenation + eddy
# ---------------------------------------------------------------
echo ""
echo "--- Step 4: AxCaliber concatenation + eddy ---"

# Concatenate all 4 AxCaliber acquisitions
echo "  Concatenating AxCaliber 1-4..."
AXC_FILES=""
AXC_BVALS=""
AXC_BVECS=""
for i in 1 2 3 4; do
    AXC_FILE="${DWI_DIR}/${SUBJECT}_ses-02_acq-AxCaliber${i}_dir-AP_part-mag_dwi"
    if [ -f "${AXC_FILE}.nii.gz" ]; then
        AXC_FILES="${AXC_FILES} ${AXC_FILE}.nii.gz"
        AXC_BVALS="${AXC_BVALS} ${AXC_FILE}.bval"
        AXC_BVECS="${AXC_BVECS} ${AXC_FILE}.bvec"
    fi
done

fslmerge -t "${AXC_OUT}/axcaliber_all" ${AXC_FILES}
echo "  Merged volumes: $(fslnvols ${AXC_OUT}/axcaliber_all)"

# Concatenate bvals and bvecs
paste -d' ' ${AXC_BVALS} > "${AXC_OUT}/bvals_all.txt"
paste -d' ' ${AXC_BVECS} > "${AXC_OUT}/bvecs_all.txt"

# Create eddy index file
N_VOLS=$(fslnvols "${AXC_OUT}/axcaliber_all")
python3 -c "print(' '.join(['1']*${N_VOLS}))" > "${AXC_OUT}/index.txt"

# BET on corrected b0
JOB_BET_AXC=$(fsl_sub -l "${LOGS}" -N bet_axcaliber -T 5 -j "${JOB_TOPUP_AXC}" \
    bash -c "fslmaths ${AXC_OUT}/b0_corrected -Tmean ${AXC_OUT}/b0_mean && \
             bet ${AXC_OUT}/b0_mean ${AXC_OUT}/b0_brain -m -f 0.3")

# Eddy on concatenated AxCaliber
JOB_EDDY_AXC=$(fsl_sub -l "${LOGS}" -N eddy_axcaliber -T 120 -j "${JOB_BET_AXC}" \
    eddy_cpu \
    --imain="${AXC_OUT}/axcaliber_all.nii.gz" \
    --mask="${AXC_OUT}/b0_brain_mask.nii.gz" \
    --acqp="${AXC_OUT}/acqparams.txt" \
    --index="${AXC_OUT}/index.txt" \
    --bvecs="${AXC_OUT}/bvecs_all.txt" \
    --bvals="${AXC_OUT}/bvals_all.txt" \
    --topup="${AXC_OUT}/topup_results" \
    --out="${AXC_OUT}/axcaliber_eddy" \
    --nthr=4 \
    --data_is_shelled \
    --verbose)

echo "  eddy job: ${JOB_EDDY_AXC} (depends on ${JOB_BET_AXC})"
echo "  264 volumes, b up to 15500 s/mm² — this will take many hours"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo ""
echo "=== Post-Eddy Chain Submitted ==="
echo ""
echo "Job Chain:"
echo "  1. dtifit (CHARMED):     ${JOB_DTIFIT}"
echo "  2. bedpostx (CHARMED):   ${JOB_BEDPOSTX} → after dtifit"
echo "  3. topup (AxCaliber):    ${JOB_TOPUP_AXC}"
echo "  4. bet (AxCaliber):      ${JOB_BET_AXC} → after topup"
echo "  5. eddy (AxCaliber):     ${JOB_EDDY_AXC} → after bet"
echo ""
echo "After bedpostx completes:"
echo "  → Run probtrackx2 for connectome (script 04_tracula.sh)"
echo "  → Run sbi4dwi for microstructure fitting"
echo ""
echo "Expected outputs:"
echo "  CHARMED dtifit:  ${DTIFIT_OUT}/${SUBJECT}_CHARMED_FA.nii.gz"
echo "  CHARMED bedpostx: ${BEDPOSTX_DIR}.bedpostX/"
echo "  AxCaliber eddy:  ${AXC_OUT}/axcaliber_eddy.nii.gz"
