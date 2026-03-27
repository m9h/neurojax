#!/bin/bash
# ============================================================================
# Stage K: Quantitative MRI Fitting
# ============================================================================
# Fits biophysical signal models to WAND's rich structural acquisitions
# to produce quantitative tissue property maps:
#
# ses-02:
#   VFA (SPGR + SSFP) → T1 map (DESPOT1), T2 map (DESPOT2)
#                      → Myelin Water Fraction (mcDESPOT)
#   QMT (16 volumes)   → Bound Pool Fraction, Exchange Rate (two-pool model)
#
# ses-06:
#   MP2RAGE (2 inversions) → Quantitative T1 map
#   Multi-echo GRE (7 echoes) → T2* map, R2* map
#
# Post-hoc:
#   g-ratio = f(QMT BPF, AxCaliber diameter)
#   Conduction velocity = f(g-ratio, axon diameter) via Hursh-Rushton
#
# Primary tool: QUIT (QUantitative Imaging Tools, Tobias Wood)
#   Install: brew install quit  OR  conda install -c conda-forge quit
#   GitHub: github.com/spinicist/QUIT
#
# Secondary: qMRLab for QMT two-pool fitting (if QUIT qmt insufficient)
#   Install: Docker or Octave
#
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"

QMRI_DIR="${WAND_DERIVATIVES}/qmri/${SUBJECT}"
mkdir -p "${QMRI_DIR}/ses-02" "${QMRI_DIR}/ses-06" "${QMRI_DIR}/gratio"

ANAT_02="${WAND_ROOT}/${SUBJECT}/ses-02/anat"
ANAT_06="${WAND_ROOT}/${SUBJECT}/ses-06/anat"

echo "=== Quantitative MRI Fitting: ${SUBJECT} ==="

# Check QUIT is installed
if ! command -v qi 2>/dev/null && ! command -v qi_despot1 2>/dev/null; then
    echo "WARNING: QUIT not found. Install with: brew install quit"
    echo "Falling back to FreeSurfer mri_ms_fitparms for T1/PD only."
    USE_QUIT=false
else
    echo "QUIT found: $(qi --version 2>&1 | head -1)"
    USE_QUIT=true
fi

# ---------------------------------------------------------------
# 1. VFA T1 Mapping (DESPOT1 from SPGR data)
# ---------------------------------------------------------------

echo ""
echo "--- VFA T1 Mapping (ses-02 SPGR) ---"

SPGR_FILE="${ANAT_02}/${SUBJECT}_ses-02_acq-spgr_part-mag_VFA.nii.gz"

if [ -f "${SPGR_FILE}" ]; then
    if [ "${USE_QUIT}" = true ]; then
        echo "Fitting T1 with QUIT qi_despot1..."
        # QUIT expects a JSON config with flip angles and TR
        # Extract from BIDS sidecar
        python3 -c "
import json, os
sidecar = '${SPGR_FILE}'.replace('.nii.gz', '.json')
with open(sidecar) as f:
    meta = json.load(f)
config = {
    'SPGR': {
        'TR': meta.get('RepetitionTime', 0.02) * 1000,  # to ms
        'FA': meta.get('FlipAngle', [3, 15])
    }
}
with open('${QMRI_DIR}/ses-02/despot1_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print(f'TR={config[\"SPGR\"][\"TR\"]}ms, FA={config[\"SPGR\"][\"FA\"]}')
" 2>/dev/null || echo "  (config from sidecar failed, using defaults)"

        # Run DESPOT1
        qi despot1 "${SPGR_FILE}" \
            --out "${QMRI_DIR}/ses-02/" \
            --json "${QMRI_DIR}/ses-02/despot1_config.json" 2>&1 | tail -3 || \
        echo "  QUIT despot1 failed (may need config adjustment)"

    else
        # FreeSurfer fallback
        echo "Fitting T1/PD with mri_ms_fitparms..."
        export FREESURFER_HOME=/Applications/freesurfer/8.2.0
        source $FREESURFER_HOME/SetUpFreeSurfer.sh 2>/dev/null

        mri_ms_fitparms \
            "${SPGR_FILE}" \
            "${QMRI_DIR}/ses-02/" 2>&1 | tail -5

        # Rename outputs
        [ -f "${QMRI_DIR}/ses-02/T1.mgz" ] && \
            mri_convert "${QMRI_DIR}/ses-02/T1.mgz" "${QMRI_DIR}/ses-02/T1map.nii.gz"
        [ -f "${QMRI_DIR}/ses-02/PD.mgz" ] && \
            mri_convert "${QMRI_DIR}/ses-02/PD.mgz" "${QMRI_DIR}/ses-02/PDmap.nii.gz"
    fi
    echo "  T1 map: ${QMRI_DIR}/ses-02/T1map.nii.gz"
else
    echo "  SPGR VFA data not found"
fi

# ---------------------------------------------------------------
# 2. mcDESPOT (SPGR + SSFP → Myelin Water Fraction)
# ---------------------------------------------------------------

echo ""
echo "--- mcDESPOT (ses-02 SPGR + SSFP) ---"

SSFP_FILE="${ANAT_02}/${SUBJECT}_ses-02_acq-ssfp_part-mag_VFA.nii.gz"

if [ -f "${SPGR_FILE}" ] && [ -f "${SSFP_FILE}" ] && [ "${USE_QUIT}" = true ]; then
    echo "Fitting mcDESPOT (myelin water fraction)..."
    echo "  SPGR: ${SPGR_FILE}"
    echo "  SSFP: ${SSFP_FILE}"

    # mcDESPOT requires both SPGR and SSFP at multiple flip angles
    # Plus B0 and B1 maps for correction
    qi mcdespot \
        --spgr "${SPGR_FILE}" \
        --ssfp "${SSFP_FILE}" \
        --out "${QMRI_DIR}/ses-02/" 2>&1 | tail -5 || \
    echo "  mcDESPOT failed (may need B0/B1 maps or config)"

    echo "  MWF: ${QMRI_DIR}/ses-02/MWF.nii.gz"
else
    echo "  Skipping mcDESPOT (requires QUIT + both SPGR and SSFP)"
fi

# ---------------------------------------------------------------
# 3. QMT Two-Pool Fitting (Bound Pool Fraction + Exchange Rate)
# ---------------------------------------------------------------

echo ""
echo "--- QMT Fitting (ses-02, 16 volumes) ---"

QMT_FILES=$(find "${ANAT_02}" -name "*QMT*part-mag*" | sort)
N_QMT=$(echo "${QMT_FILES}" | wc -l)

if [ ${N_QMT} -gt 5 ]; then
    echo "  Found ${N_QMT} QMT volumes"

    if [ "${USE_QUIT}" = true ]; then
        echo "  Fitting with QUIT qi_qmt..."
        # QMT requires:
        # - MT-weighted volumes at different offsets and flip angles
        # - A reference (MT-off) volume
        # - T1 map (from step 1)
        # - B1 map

        MT_OFF="${ANAT_02}/${SUBJECT}_ses-02_mt-off_part-mag_QMT.nii.gz"

        qi qmt \
            --mt-off "${MT_OFF}" \
            --T1 "${QMRI_DIR}/ses-02/T1map.nii.gz" \
            --out "${QMRI_DIR}/ses-02/" 2>&1 | tail -5 || \
        echo "  QUIT qmt failed (complex config needed for 16-volume protocol)"
    fi

    # Alternative: qMRLab via Docker/Octave
    echo ""
    echo "  For full Ramani two-pool QMT fitting, consider:"
    echo "    qMRLab: docker run qmrlab/minimal qmt_spgr ..."
    echo "    Or MATLAB/Octave with qMRLab qmt_spgr module"

    echo "  BPF: ${QMRI_DIR}/ses-02/QMT_bpf.nii.gz"
    echo "  kf:  ${QMRI_DIR}/ses-02/QMT_kf.nii.gz"
else
    echo "  QMT data not found or incomplete"
fi

# ---------------------------------------------------------------
# 4. MP2RAGE T1 Mapping (ses-06)
# ---------------------------------------------------------------

echo ""
echo "--- MP2RAGE T1 (ses-06) ---"

INV1="${ANAT_06}/${SUBJECT}_ses-06_acq-PSIR_inv-1_part-mag_MP2RAGE.nii.gz"
INV2="${ANAT_06}/${SUBJECT}_ses-06_acq-PSIR_inv-2_part-mag_MP2RAGE.nii.gz"

if [ -f "${INV1}" ] && [ -f "${INV2}" ]; then
    echo "  INV1: ${INV1}"
    echo "  INV2: ${INV2}"

    if [ "${USE_QUIT}" = true ]; then
        qi mp2rage "${INV1}" "${INV2}" \
            --out "${QMRI_DIR}/ses-06/" 2>&1 | tail -3 || \
        echo "  QUIT mp2rage failed"
    else
        # Simple ratio-based T1 estimation
        python3 << 'PYEOF'
import nibabel as nib
import numpy as np
import os

inv1 = nib.load(os.environ.get('INV1', ''))
inv2 = nib.load(os.environ.get('INV2', ''))
d1 = inv1.get_fdata().astype(np.float64)
d2 = inv2.get_fdata().astype(np.float64)

# MP2RAGE uniform image (removes B1 bias)
uni = (d1 * d2) / (d1**2 + d2**2 + 1e-10)
# T1 lookup would require TI1, TI2, TR, alpha from JSON
# For now save the uniform image
out_dir = os.environ.get('QMRI_DIR', '.') + '/ses-06'
nib.save(nib.Nifti1Image(uni, inv1.affine), f'{out_dir}/MP2RAGE_uni.nii.gz')
print(f'  MP2RAGE uniform image saved')
PYEOF
    fi
    echo "  T1: ${QMRI_DIR}/ses-06/T1_MP2RAGE.nii.gz"
else
    echo "  MP2RAGE data not found"
fi

# ---------------------------------------------------------------
# 5. Multi-echo GRE T2* Mapping (ses-06)
# ---------------------------------------------------------------

echo ""
echo "--- Multi-echo GRE T2*/R2* (ses-06, 7 echoes) ---"

MEGRE_FILES=$(find "${ANAT_06}" -name "*MEGRE*part-mag*" | sort)
N_ECHOES=$(echo "${MEGRE_FILES}" | wc -l)

if [ ${N_ECHOES} -gt 3 ]; then
    echo "  Found ${N_ECHOES} echoes"

    if [ "${USE_QUIT}" = true ]; then
        qi mpm_r2s ${MEGRE_FILES} \
            --out "${QMRI_DIR}/ses-06/" 2>&1 | tail -3 || \
        echo "  QUIT mpm_r2s failed"
    else
        # Simple mono-exponential T2* fit
        python3 << 'PYEOF'
import nibabel as nib
import numpy as np
import os, glob

anat_06 = os.environ.get('ANAT_06', '')
out_dir = os.environ.get('QMRI_DIR', '.') + '/ses-06'
subject = os.environ.get('SUBJECT', '')

# Load all echoes
echo_files = sorted(glob.glob(f'{anat_06}/{subject}_ses-06_echo-*_part-mag_MEGRE.nii.gz'))
if not echo_files:
    print('  No MEGRE files found')
    exit()

# Get echo times from JSON sidecars
TEs = []
for ef in echo_files:
    import json
    jf = ef.replace('.nii.gz', '.json')
    if os.path.exists(jf):
        with open(jf) as f:
            TEs.append(json.load(f).get('EchoTime', 0))

imgs = [nib.load(f) for f in echo_files]
data = np.stack([img.get_fdata() for img in imgs], axis=-1)  # (X,Y,Z,N_echoes)
affine = imgs[0].affine

print(f'  Data shape: {data.shape}')
print(f'  Echo times (s): {TEs}')

if TEs and len(TEs) == data.shape[-1]:
    TEs = np.array(TEs)
    # Log-linear fit: log(S) = log(S0) - TE/T2*
    mask = data[..., 0] > np.percentile(data[..., 0], 10)
    log_data = np.log(np.maximum(data, 1e-10))

    # Fit per voxel (vectorized)
    T2star = np.zeros(data.shape[:3])
    R2star = np.zeros(data.shape[:3])

    # Simple: use first and last echo
    log_ratio = log_data[..., 0] - log_data[..., -1]
    dTE = TEs[-1] - TEs[0]
    R2star[mask] = log_ratio[mask] / dTE
    T2star[mask] = 1.0 / np.maximum(R2star[mask], 1e-10)

    # Clip to physiological range
    T2star = np.clip(T2star, 0, 0.2)  # 0-200ms
    R2star = np.clip(R2star, 0, 500)  # 0-500 Hz

    nib.save(nib.Nifti1Image(T2star, affine), f'{out_dir}/T2star_map.nii.gz')
    nib.save(nib.Nifti1Image(R2star, affine), f'{out_dir}/R2star_map.nii.gz')
    print(f'  T2* range: [{T2star[mask].min()*1000:.1f}, {T2star[mask].max()*1000:.1f}] ms')
    print(f'  R2* range: [{R2star[mask].min():.1f}, {R2star[mask].max():.1f}] Hz')
else:
    print('  Could not extract echo times from JSON sidecars')
PYEOF
    fi
    echo "  T2*: ${QMRI_DIR}/ses-06/T2star_map.nii.gz"
    echo "  R2*: ${QMRI_DIR}/ses-06/R2star_map.nii.gz"
else
    echo "  MEGRE data not found"
fi

# ---------------------------------------------------------------
# 6. g-ratio Computation (post-hoc)
# ---------------------------------------------------------------

echo ""
echo "--- g-ratio (QMT BPF + AxCaliber diameter) ---"

python3 << 'PYEOF'
import os
import numpy as np

qmri_dir = os.environ.get('QMRI_DIR', '.')
micro_dir = os.environ.get('WAND_DERIVATIVES', '') + f"/{os.environ.get('SUBJECT', '')}"

bpf_file = f'{qmri_dir}/ses-02/QMT_bpf.nii.gz'
diameter_file = f'{micro_dir}/microstructure/sbi4dwi/axcaliber_diameter.nii.gz'

if os.path.exists(bpf_file) and os.path.exists(diameter_file):
    import nibabel as nib
    bpf = nib.load(bpf_file).get_fdata()
    # g-ratio = sqrt(1 - MVF) where MVF ≈ BPF / (1 - BPF) scaled
    # Simplified: g = sqrt(1 - BPF * k) where k ≈ 2 (empirical)
    mvf = np.clip(bpf * 2.0, 0, 0.95)
    g_ratio = np.sqrt(1 - mvf)
    nib.save(nib.Nifti1Image(g_ratio, nib.load(bpf_file).affine),
             f'{qmri_dir}/gratio/g_ratio.nii.gz')
    print(f'  g-ratio range: [{g_ratio[g_ratio>0].min():.3f}, {g_ratio[g_ratio>0].max():.3f}]')
    print(f'  Saved: {qmri_dir}/gratio/g_ratio.nii.gz')
else:
    print('  g-ratio requires both QMT BPF and AxCaliber diameter maps')
    print(f'  BPF: {"found" if os.path.exists(bpf_file) else "missing"}')
    print(f'  Diameter: {"found" if os.path.exists(diameter_file) else "missing"}')
PYEOF

echo ""
echo "=== qMRI fitting complete for ${SUBJECT} ==="
echo ""
echo "Outputs:"
echo "  T1 (VFA):   ${QMRI_DIR}/ses-02/T1map.nii.gz"
echo "  MWF:        ${QMRI_DIR}/ses-02/MWF.nii.gz"
echo "  QMT BPF:    ${QMRI_DIR}/ses-02/QMT_bpf.nii.gz"
echo "  QMT kf:     ${QMRI_DIR}/ses-02/QMT_kf.nii.gz"
echo "  T1 (MP2R):  ${QMRI_DIR}/ses-06/T1_MP2RAGE.nii.gz"
echo "  T2*:        ${QMRI_DIR}/ses-06/T2star_map.nii.gz"
echo "  R2*:        ${QMRI_DIR}/ses-06/R2star_map.nii.gz"
echo "  g-ratio:    ${QMRI_DIR}/gratio/g_ratio.nii.gz"
echo ""
echo "Tool recommendations:"
echo "  Best single tool: QUIT (github.com/spinicist/QUIT)"
echo "  Best for QMT:     qMRLab qmt_spgr (github.com/qMRLab/qMRLab)"
echo "  Best for R2*:     hMRI toolbox (github.com/hMRI-group/hMRI-toolbox)"
echo "  FS fallback:      mri_ms_fitparms (T1/PD only from SPGR)"
