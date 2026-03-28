#!/bin/bash
# ============================================================================
# Stage 28: Post-Eddy Diffusion Chain — DGX Spark Version
# ============================================================================
# GPU-accelerated diffusion pipeline for DGX Spark (A100/H100).
# Uses eddy_cuda + bedpostx_gpu for massive speedup.
#
# Prerequisites on DGX:
#   - FSL 6.0.7+ with CUDA support (eddy_cuda, bedpostx_gpu)
#   - WAND data accessible at $WAND_ROOT
#   - neurojax/sbi4dwi repos for microstructure fitting
#
# Usage: sbatch 28_post_eddy_chain_dgx.sh sub-08033
#   or:  ./28_post_eddy_chain_dgx.sh sub-08033
# ============================================================================

#SBATCH --job-name=wand-dwi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

# --- Environment ---
export FSLDIR="${FSLDIR:-/usr/local/fsl}"
source "${FSLDIR}/etc/fslconf/fsl.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"
WAND_ROOT="${WAND_ROOT:-/data/wand}"
WAND_DERIVATIVES="${WAND_ROOT}/derivatives"
DWI_DIR="${WAND_ROOT}/${SUBJECT}/ses-02/dwi"
DWI_OUT="${WAND_DERIVATIVES}/fsl-dwi/${SUBJECT}/ses-02/dwi"
LOGS="${WAND_DERIVATIVES}/logs/${SUBJECT}"
mkdir -p "${DWI_OUT}" "${LOGS}"

echo "=== DGX Spark DWI Pipeline: ${SUBJECT} ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# ---------------------------------------------------------------
# Phase 1: CHARMED (standard multi-shell DWI)
# ---------------------------------------------------------------
echo ""
echo "=== Phase 1: CHARMED Processing ==="

# 1a. Topup (CPU — fast enough)
if [ ! -f "${DWI_OUT}/topup_results_fieldcoef.nii.gz" ]; then
    echo "--- topup ---"
    # Extract AP + PA b0s, merge, run topup
    fslroi "${DWI_DIR}/${SUBJECT}_ses-02_acq-CHARMED_dir-AP_part-mag_dwi" \
           "${DWI_OUT}/b0_AP" 0 1
    fslroi "${DWI_DIR}/${SUBJECT}_ses-02_acq-CHARMED_dir-PA_part-mag_dwi" \
           "${DWI_OUT}/b0_PA" 0 1
    fslmerge -t "${DWI_OUT}/b0_AP_PA" "${DWI_OUT}/b0_AP" "${DWI_OUT}/b0_PA"

    READOUT=$(python3 -c "
import json
d = json.load(open('${DWI_DIR}/${SUBJECT}_ses-02_acq-CHARMED_dir-AP_part-mag_dwi.json'))
print(d.get('TotalReadoutTime', 0.05))
")
    echo "0 -1 0 ${READOUT}" > "${DWI_OUT}/acqparams.txt"
    echo "0 1 0 ${READOUT}" >> "${DWI_OUT}/acqparams.txt"

    topup --imain="${DWI_OUT}/b0_AP_PA" \
          --datain="${DWI_OUT}/acqparams.txt" \
          --config=b02b0.cnf \
          --out="${DWI_OUT}/topup_results" \
          --iout="${DWI_OUT}/b0_corrected"
    echo "  topup complete"
else
    echo "  topup: already done"
fi

# Brain mask
if [ ! -f "${DWI_OUT}/b0_brain_mask.nii.gz" ]; then
    fslmaths "${DWI_OUT}/b0_corrected" -Tmean "${DWI_OUT}/b0_mean"
    bet "${DWI_OUT}/b0_mean" "${DWI_OUT}/b0_brain" -m -f 0.3
fi

# 1b. Eddy (GPU — 30 min vs 40+ hours on CPU)
if [ ! -f "${DWI_OUT}/CHARMED_eddy.nii.gz" ]; then
    echo "--- eddy_cuda (CHARMED, 266 volumes) ---"
    N_VOLS=$(fslnvols "${DWI_DIR}/${SUBJECT}_ses-02_acq-CHARMED_dir-AP_part-mag_dwi")
    python3 -c "print(' '.join(['1']*${N_VOLS}))" > "${DWI_OUT}/CHARMED_index.txt"

    eddy_cuda \
        --imain="${DWI_DIR}/${SUBJECT}_ses-02_acq-CHARMED_dir-AP_part-mag_dwi.nii.gz" \
        --mask="${DWI_OUT}/b0_brain_mask.nii.gz" \
        --acqp="${DWI_OUT}/acqparams.txt" \
        --index="${DWI_OUT}/CHARMED_index.txt" \
        --bvecs="${DWI_DIR}/${SUBJECT}_ses-02_acq-CHARMED_dir-AP_part-mag_dwi.bvec" \
        --bvals="${DWI_DIR}/${SUBJECT}_ses-02_acq-CHARMED_dir-AP_part-mag_dwi.bval" \
        --topup="${DWI_OUT}/topup_results" \
        --out="${DWI_OUT}/CHARMED_eddy" \
        --data_is_shelled \
        --repol \
        --verbose
    echo "  eddy_cuda complete: $(fslnvols ${DWI_OUT}/CHARMED_eddy) volumes"
else
    echo "  eddy CHARMED: already done"
fi

# 1c. dtifit (CPU — fast)
echo "--- dtifit (CHARMED) ---"
DTIFIT="${DWI_OUT}/dtifit"
mkdir -p "${DTIFIT}"
dtifit \
    --data="${DWI_OUT}/CHARMED_eddy" \
    --mask="${DWI_OUT}/b0_brain_mask.nii.gz" \
    --bvecs="${DWI_OUT}/CHARMED_eddy.eddy_rotated_bvecs" \
    --bvals="${DWI_DIR}/${SUBJECT}_ses-02_acq-CHARMED_dir-AP_part-mag_dwi.bval" \
    --out="${DTIFIT}/${SUBJECT}_CHARMED" \
    --save_tensor --sse
echo "  FA range: $(fslstats ${DTIFIT}/${SUBJECT}_CHARMED_FA -R)"
echo "  MD range: $(fslstats ${DTIFIT}/${SUBJECT}_CHARMED_MD -R)"

# 1d. bedpostx (GPU — 2-3 hours vs 24 on CPU)
echo "--- bedpostx_gpu (CHARMED, 3 fibers, model 2) ---"
BPXDIR="${DWI_OUT}/bedpostx_input"
mkdir -p "${BPXDIR}"
ln -sf "${DWI_OUT}/CHARMED_eddy.nii.gz" "${BPXDIR}/data.nii.gz"
ln -sf "${DWI_OUT}/b0_brain_mask.nii.gz" "${BPXDIR}/nodif_brain_mask.nii.gz"
ln -sf "${DWI_OUT}/CHARMED_eddy.eddy_rotated_bvecs" "${BPXDIR}/bvecs"
ln -sf "${DWI_DIR}/${SUBJECT}_ses-02_acq-CHARMED_dir-AP_part-mag_dwi.bval" "${BPXDIR}/bvals"

bedpostx_gpu "${BPXDIR}" -n 3 --model=2
echo "  bedpostx_gpu complete"

# ---------------------------------------------------------------
# Phase 2: AxCaliber (ultra-high b-value for axon diameter)
# ---------------------------------------------------------------
echo ""
echo "=== Phase 2: AxCaliber Processing ==="

AXC_OUT="${DWI_OUT}/axcaliber"
mkdir -p "${AXC_OUT}"

# 2a. Concatenate all 4 AxCaliber acquisitions
echo "--- Concatenating AxCaliber 1-4 ---"
AXC_FILES=""
for i in 1 2 3 4; do
    F="${DWI_DIR}/${SUBJECT}_ses-02_acq-AxCaliber${i}_dir-AP_part-mag_dwi.nii.gz"
    if [ -f "$F" ]; then AXC_FILES="${AXC_FILES} $F"; fi
done
fslmerge -t "${AXC_OUT}/axcaliber_all" ${AXC_FILES}
echo "  Volumes: $(fslnvols ${AXC_OUT}/axcaliber_all)"

# Concatenate bvals/bvecs
paste -d' ' ${DWI_DIR}/${SUBJECT}_ses-02_acq-AxCaliber{1,2,3,4}_dir-AP_part-mag_dwi.bval > "${AXC_OUT}/bvals_all.txt"
paste -d' ' ${DWI_DIR}/${SUBJECT}_ses-02_acq-AxCaliber{1,2,3,4}_dir-AP_part-mag_dwi.bvec > "${AXC_OUT}/bvecs_all.txt"

# 2b. Topup for AxCaliber
echo "--- topup (AxCaliber) ---"
# Extract b0 from first acquisition + PA reference
fslroi "${DWI_DIR}/${SUBJECT}_ses-02_acq-AxCaliber1_dir-AP_part-mag_dwi" "${AXC_OUT}/b0_AP" 0 1
fslroi "${DWI_DIR}/${SUBJECT}_ses-02_acq-AxCaliberRef_dir-PA_part-mag_dwi" "${AXC_OUT}/b0_PA" 0 1
fslmerge -t "${AXC_OUT}/b0_AP_PA" "${AXC_OUT}/b0_AP" "${AXC_OUT}/b0_PA"

READOUT_AXC=$(python3 -c "
import json
d = json.load(open('${DWI_DIR}/${SUBJECT}_ses-02_acq-AxCaliber1_dir-AP_part-mag_dwi.json'))
print(d.get('TotalReadoutTime', 0.05))
")
echo "0 -1 0 ${READOUT_AXC}" > "${AXC_OUT}/acqparams.txt"
echo "0 1 0 ${READOUT_AXC}" >> "${AXC_OUT}/acqparams.txt"

topup --imain="${AXC_OUT}/b0_AP_PA" \
      --datain="${AXC_OUT}/acqparams.txt" \
      --config=b02b0.cnf \
      --out="${AXC_OUT}/topup_results" \
      --iout="${AXC_OUT}/b0_corrected"

fslmaths "${AXC_OUT}/b0_corrected" -Tmean "${AXC_OUT}/b0_mean"
bet "${AXC_OUT}/b0_mean" "${AXC_OUT}/b0_brain" -m -f 0.3

# 2c. Eddy for AxCaliber (GPU)
echo "--- eddy_cuda (AxCaliber, 264 volumes, b≤15500) ---"
N_AXC=$(fslnvols "${AXC_OUT}/axcaliber_all")
python3 -c "print(' '.join(['1']*${N_AXC}))" > "${AXC_OUT}/index.txt"

eddy_cuda \
    --imain="${AXC_OUT}/axcaliber_all.nii.gz" \
    --mask="${AXC_OUT}/b0_brain_mask.nii.gz" \
    --acqp="${AXC_OUT}/acqparams.txt" \
    --index="${AXC_OUT}/index.txt" \
    --bvecs="${AXC_OUT}/bvecs_all.txt" \
    --bvals="${AXC_OUT}/bvals_all.txt" \
    --topup="${AXC_OUT}/topup_results" \
    --out="${AXC_OUT}/axcaliber_eddy" \
    --data_is_shelled \
    --repol \
    --verbose
echo "  eddy_cuda (AxCaliber) complete"

# ---------------------------------------------------------------
# Phase 3: Microstructure fitting (sbi4dwi on GPU)
# ---------------------------------------------------------------
echo ""
echo "=== Phase 3: sbi4dwi Microstructure Fitting ==="

MICRO_OUT="${WAND_DERIVATIVES}/sbi4dwi/${SUBJECT}/ses-02"
mkdir -p "${MICRO_OUT}"

# NODDI from CHARMED (multi-shell)
echo "--- NODDI (CHARMED) ---"
python3 -c "
import sys
sys.path.insert(0, 'PATH_TO_SBI4DWI')
# TODO: Import sbi4dwi NODDI fitter
# Fit NODDI (neurite density, orientation dispersion, CSF fraction)
# Uses b=0, 1200, 2400 shells from CHARMED
print('NODDI fitting: TODO — need sbi4dwi path configured on DGX')
"

# AxCaliber diameter fitting
echo "--- AxCaliber diameter (sbi4dwi) ---"
python3 -c "
# TODO: sbi4dwi AxCaliber/CHARMED axon diameter estimation
# Uses ultra-high b-values (b≤15500) with known gradient waveforms
print('AxCaliber fitting: TODO — need sbi4dwi configured on DGX')
"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo ""
echo "=== DGX Pipeline Complete: ${SUBJECT} ==="
echo ""
echo "CHARMED outputs:"
echo "  Eddy-corrected:  ${DWI_OUT}/CHARMED_eddy.nii.gz"
echo "  FA map:          ${DTIFIT}/${SUBJECT}_CHARMED_FA.nii.gz"
echo "  MD map:          ${DTIFIT}/${SUBJECT}_CHARMED_MD.nii.gz"
echo "  bedpostx:        ${BPXDIR}.bedpostX/"
echo ""
echo "AxCaliber outputs:"
echo "  Eddy-corrected:  ${AXC_OUT}/axcaliber_eddy.nii.gz"
echo ""
echo "Next steps:"
echo "  - probtrackx2 → Cmat + Lmat"
echo "  - ConnectomeMapper → Dmat (conduction delays)"
echo "  - Register to ses-02 T1w → to FreeSurfer parcellation"
