#!/bin/bash
# ============================================================================
# Stage 2: DWI Preprocessing — FSL topup + eddy
# ============================================================================
# WAND DWI acquisitions (ses-02):
#   - AxCaliber1-4 (dir-AP): multi-shell for axon diameter estimation
#   - AxCaliberRef (dir-PA): reverse phase-encode for topup
#   - CHARMED (dir-AP + dir-PA): composite hindered and restricted model
#
# Pipeline:
#   1. Merge AP/PA volumes for topup
#   2. topup: estimate susceptibility distortion field
#   3. eddy: motion + eddy current + susceptibility correction
#   4. dtifit: quick DTI for QC (FA, MD maps)
#
# Runtime: ~30-60 min per subject
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"

DWI_DIR="${WAND_ROOT}/${SUBJECT}/ses-02/dwi"
OUT_DIR="${WAND_DERIVATIVES}/fsl/${SUBJECT}/ses-02/dwi"
mkdir -p "${OUT_DIR}"

echo "=== DWI Preprocessing: ${SUBJECT} ==="

# ---------------------------------------------------------------
# Step 1: Identify AP and PA volumes for topup
# ---------------------------------------------------------------

# Use CHARMED AP/PA pair for topup (both have matched b=0 volumes)
AP_DWI=$(find "${DWI_DIR}" -name "*CHARMED*dir-AP*dwi.nii.gz" | head -1)
PA_DWI=$(find "${DWI_DIR}" -name "*CHARMED*dir-PA*dwi.nii.gz" | head -1)

if [ -z "$AP_DWI" ] || [ -z "$PA_DWI" ]; then
    # Fallback: AxCaliber AP + AxCaliberRef PA
    AP_DWI=$(find "${DWI_DIR}" -name "*AxCaliber1*dir-AP*dwi.nii.gz" | head -1)
    PA_DWI=$(find "${DWI_DIR}" -name "*AxCaliberRef*dir-PA*dwi.nii.gz" | head -1)
fi

echo "AP: ${AP_DWI}"
echo "PA: ${PA_DWI}"

# Extract b=0 volumes for topup
AP_BVAL="${AP_DWI%.nii.gz}.bval"
PA_BVAL="${PA_DWI%.nii.gz}.bval"

# Find indices of b=0 volumes
python3 -c "
import numpy as np
bvals = np.loadtxt('${AP_BVAL}')
b0_idx = np.where(bvals < 50)[0]
print(','.join(map(str, b0_idx[:3])))  # first 3 b=0s
" > "${OUT_DIR}/ap_b0_indices.txt"

python3 -c "
import numpy as np
bvals = np.loadtxt('${PA_BVAL}')
b0_idx = np.where(bvals < 50)[0]
print(','.join(map(str, b0_idx[:3])))
" > "${OUT_DIR}/pa_b0_indices.txt"

# Extract b=0 volumes
AP_B0_IDX=$(cat "${OUT_DIR}/ap_b0_indices.txt" | tr ',' ' ' | awk '{print $1}')
PA_B0_IDX=$(cat "${OUT_DIR}/pa_b0_indices.txt" | tr ',' ' ' | awk '{print $1}')

fslroi "${AP_DWI}" "${OUT_DIR}/AP_b0" ${AP_B0_IDX} 1
fslroi "${PA_DWI}" "${OUT_DIR}/PA_b0" ${PA_B0_IDX} 1

# Merge AP/PA b=0s for topup
fslmerge -t "${OUT_DIR}/AP_PA_b0" "${OUT_DIR}/AP_b0" "${OUT_DIR}/PA_b0"

# ---------------------------------------------------------------
# Step 2: topup — susceptibility distortion correction
# ---------------------------------------------------------------

# Create acquisition parameters file
# AP: 0 -1 0 readout_time
# PA: 0  1 0 readout_time
# Readout time from JSON (approximate if not available)
READOUT=$(python3 -c "
import json
with open('${AP_DWI%.nii.gz}.json') as f:
    j = json.load(f)
print(j.get('TotalReadoutTime', 0.05))
")

cat > "${OUT_DIR}/acqparams.txt" << EOF
0 -1 0 ${READOUT}
0  1 0 ${READOUT}
EOF

echo "Running topup..."
topup \
    --imain="${OUT_DIR}/AP_PA_b0" \
    --datain="${OUT_DIR}/acqparams.txt" \
    --config=b02b0.cnf \
    --out="${OUT_DIR}/topup_results" \
    --fout="${OUT_DIR}/topup_field" \
    --iout="${OUT_DIR}/topup_b0_corrected"

# ---------------------------------------------------------------
# Step 3: Create brain mask from corrected b=0
# ---------------------------------------------------------------

fslmaths "${OUT_DIR}/topup_b0_corrected" -Tmean "${OUT_DIR}/topup_b0_mean"
bet "${OUT_DIR}/topup_b0_mean" "${OUT_DIR}/b0_brain" -m -f 0.3

# ---------------------------------------------------------------
# Step 4: eddy — motion + eddy current correction
# ---------------------------------------------------------------

# Process each DWI acquisition separately, then combine
for ACQ in AxCaliber1 AxCaliber2 AxCaliber3 AxCaliber4 CHARMED; do
    DWI_FILE=$(find "${DWI_DIR}" -name "*${ACQ}*dir-AP*dwi.nii.gz" | head -1)
    if [ -z "$DWI_FILE" ]; then continue; fi

    BVAL="${DWI_FILE%.nii.gz}.bval"
    BVEC="${DWI_FILE%.nii.gz}.bvec"
    NVOLS=$(fslnvols "${DWI_FILE}")

    echo "Processing ${ACQ}: ${NVOLS} volumes"

    # Create index file (all volumes use AP acquisition = line 1 of acqparams)
    python3 -c "print('\n'.join(['1'] * ${NVOLS}))" > "${OUT_DIR}/${ACQ}_index.txt"

    # Run eddy
    eddy_cpu \
        --imain="${DWI_FILE}" \
        --mask="${OUT_DIR}/b0_brain_mask" \
        --acqp="${OUT_DIR}/acqparams.txt" \
        --index="${OUT_DIR}/${ACQ}_index.txt" \
        --bvecs="${BVEC}" \
        --bvals="${BVAL}" \
        --topup="${OUT_DIR}/topup_results" \
        --out="${OUT_DIR}/${ACQ}_eddy" \
        --data_is_shelled \
        --verbose

    echo "${ACQ} eddy complete."
done

echo "=== DWI preprocessing complete for ${SUBJECT} ==="
