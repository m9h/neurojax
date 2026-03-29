#!/bin/bash
# ============================================================================
# WAND ses-02/06 qMRI fitting with QUIT (full biophysical models)
# ============================================================================
# Replaces the simplified Python fitting with QUIT's validated solvers:
#   1. DESPOT1 (NLLS) → T1 map from SPGR VFA
#   2. qi qmt (Ramani, super-Lorentzian) → BPF, kf from QMT
#   3. qi multiecho → T2*/R2* from 7-echo GRE (ses-06)
#   4. qi mp2rage → quantitative T1 from MP2RAGE (ses-06)
# ============================================================================

set -euo pipefail

SUBJECT="${1:-sub-08033}"
WAND_ROOT="${WAND_ROOT:-/Users/mhough/dev/wand}"
DERIV="${WAND_ROOT}/derivatives"
QMRI_OUT="${DERIV}/qmri/${SUBJECT}"
ANAT_02="${WAND_ROOT}/${SUBJECT}/ses-02/anat"
ANAT_06="${WAND_ROOT}/${SUBJECT}/ses-06/anat"

mkdir -p "${QMRI_OUT}/ses-02/quit" "${QMRI_OUT}/ses-06/quit"

echo "=== QUIT qMRI Fitting: ${SUBJECT} ==="
echo "QUIT version: $(qi --version 2>&1)"

# ---------------------------------------------------------------
# 1. DESPOT1 T1 from SPGR VFA
# ---------------------------------------------------------------
echo ""
echo "--- 1. DESPOT1 (NLLS) from SPGR VFA ---"

SPGR="${ANAT_02}/${SUBJECT}_ses-02_acq-spgr_part-mag_VFA.nii.gz"

if [ -f "${SPGR}" ]; then
    # CUBRIC mcDESPOT SPGR: 8 flip angles, TR_exc = 4ms
    # FlipAngle in JSON is 18 (last FA); actual FAs: 2,4,6,8,10,12,14,18
    cat > "${QMRI_OUT}/ses-02/quit/despot1.json" << 'EOF'
{
    "SPGR": {
        "TR": 0.004,
        "FA": [2, 4, 6, 8, 10, 12, 14, 18]
    }
}
EOF

    echo "  SPGR: ${SPGR}"
    echo "  Config: TR=4ms, FA=[2,4,6,8,10,12,14,18]"

    qi despot1 "${SPGR}" \
        --json="${QMRI_OUT}/ses-02/quit/despot1.json" \
        -o "${QMRI_OUT}/ses-02/quit/D1_" \
        -a n \
        -T 4 \
        -m "${QMRI_OUT}/ses-02/brain_mask.nii.gz" 2>&1 | tail -5

    echo "  T1: ${QMRI_OUT}/ses-02/quit/D1_D1_T1.nii.gz"
    echo "  PD: ${QMRI_OUT}/ses-02/quit/D1_D1_PD.nii.gz"
    ls "${QMRI_OUT}/ses-02/quit/D1_"* 2>/dev/null
else
    echo "  SPGR not found"
fi

# ---------------------------------------------------------------
# 2. QMT Ramani model (super-Lorentzian)
# ---------------------------------------------------------------
echo ""
echo "--- 2. QMT (Ramani + super-Lorentzian) ---"

# QUIT qi_qmt expects a single 4D MT-saturation file
# and a JSON describing the MT pulse parameters.
# We need to: (a) merge the individual QMT volumes, (b) create the JSON config

# First check if T1 map is available from step 1
T1_MAP="${QMRI_OUT}/ses-02/quit/D1_D1_T1.nii.gz"
if [ ! -f "${T1_MAP}" ]; then
    T1_MAP="${QMRI_OUT}/ses-02/T1map.nii.gz"
fi

# Merge QMT volumes into 4D (mt-off first, then mt-on in order)
MTOFF="${ANAT_02}/${SUBJECT}_ses-02_mt-off_part-mag_QMT.nii.gz"

if [ -f "${MTOFF}" ] && [ -f "${T1_MAP}" ]; then
    echo "  Merging QMT volumes..."

    # Build the merge list in the order QUIT expects
    # QUIT qmt wants: MT-saturation data = (S_mt / S_ref - 1) or just MT-weighted volumes
    # Actually QUIT wants the MT-sat data as input, computed as:
    # MTsat = (M0*alpha/T1 - signal) / (M0*alpha/T1 + signal/alpha) - see QUIT docs

    # Simpler: use the MTR approach and let QUIT handle the Ramani linearization
    # QUIT qmt expects a JSON with the MT pulse details

    QMT_VOLS=""
    QMT_JSON_ENTRIES=""
    for TAG_INFO in \
        "flip-1_mt-1:332:56360" \
        "flip-1_mt-2:332:1000" \
        "flip-2_mt-1:628:47180" \
        "flip-2_mt-2:628:12060" \
        "flip-2_mt-3:628:2750" \
        "flip-2_mt-4:628:2770" \
        "flip-2_mt-5:628:2790" \
        "flip-2_mt-6:628:2890" \
        "flip-3_mt-1:333:1000"; do

        TAG=$(echo "${TAG_INFO}" | cut -d: -f1)
        FA_MT=$(echo "${TAG_INFO}" | cut -d: -f2)
        OFFSET=$(echo "${TAG_INFO}" | cut -d: -f3)

        VOL="${ANAT_02}/${SUBJECT}_ses-02_${TAG}_part-mag_QMT.nii.gz"
        if [ -f "${VOL}" ]; then
            QMT_VOLS="${QMT_VOLS} ${VOL}"
        fi
    done

    # Merge into 4D
    fslmerge -t "${QMRI_OUT}/ses-02/quit/qmt_merged.nii.gz" ${QMT_VOLS}
    N_VOLS=$(fslnvols "${QMRI_OUT}/ses-02/quit/qmt_merged.nii.gz")
    echo "  Merged ${N_VOLS} QMT volumes"

    # Create QUIT qmt JSON config
    # The Ramani model needs: sat_f0 (offset freqs), sat_angle (MT FA in rad),
    # TR, FA (readout), pulse duration, pulse shape
    cat > "${QMRI_OUT}/ses-02/quit/qmt.json" << 'EOF'
{
    "MTSat": {
        "TR": 0.055,
        "Trf": 0.015,
        "FA": 5,
        "sat_f0": [56360, 1000, 47180, 12060, 2750, 2770, 2790, 2890, 1000],
        "sat_angle": [332, 332, 628, 628, 628, 628, 628, 628, 333],
        "pulse": {"name": "Gauss", "bandwidth": 100, "p1": 0.416, "p2": 0.295}
    }
}
EOF

    echo "  Running QUIT qmt (Ramani + super-Lorentzian)..."
    qi qmt "${QMRI_OUT}/ses-02/quit/qmt_merged.nii.gz" \
        --json="${QMRI_OUT}/ses-02/quit/qmt.json" \
        --T1="${T1_MAP}" \
        -l Superlorentzian \
        -o "${QMRI_OUT}/ses-02/quit/QMT_" \
        -m "${QMRI_OUT}/ses-02/brain_mask.nii.gz" \
        -T 4 2>&1 | tail -10

    echo "  Outputs:"
    ls "${QMRI_OUT}/ses-02/quit/QMT_"* 2>/dev/null
else
    echo "  QMT or T1 map not available"
fi

# ---------------------------------------------------------------
# 3. Multi-echo T2* (ses-06)
# ---------------------------------------------------------------
echo ""
echo "--- 3. Multi-echo T2*/R2* (ses-06, QUIT) ---"

MEGRE_MERGED="${QMRI_OUT}/ses-06/megre_merged_7echo.nii.gz"

if [ -f "${MEGRE_MERGED}" ]; then
    # Extract echo times from JSON sidecars
    TEs=$(python3 -c "
import json, glob, os
anat = '${ANAT_06}'
subject = '${SUBJECT}'
files = sorted(glob.glob(f'{anat}/{subject}_ses-06_echo-*_part-mag_MEGRE.json'))
tes = [json.load(open(f))['EchoTime'] for f in files]
print(','.join(f'{t}' for t in tes))
")
    echo "  Echo times: ${TEs}"

    cat > "${QMRI_OUT}/ses-06/quit/multiecho.json" << EOFJ
{
    "MultiEcho": {
        "TE": [${TEs}]
    }
}
EOFJ

    qi multiecho "${MEGRE_MERGED}" \
        --json="${QMRI_OUT}/ses-06/quit/multiecho.json" \
        -o "${QMRI_OUT}/ses-06/quit/ME_" \
        -m "${QMRI_OUT}/ses-06/megre_brain_mask.nii.gz" \
        -T 4 2>&1 | tail -5

    echo "  Outputs:"
    ls "${QMRI_OUT}/ses-06/quit/ME_"* 2>/dev/null
else
    echo "  MEGRE merged file not found at ${MEGRE_MERGED}"
fi

# ---------------------------------------------------------------
# 4. MP2RAGE T1 (ses-06)
# ---------------------------------------------------------------
echo ""
echo "--- 4. MP2RAGE quantitative T1 (ses-06, QUIT) ---"

INV1="${ANAT_06}/${SUBJECT}_ses-06_acq-PSIR_inv-1_part-mag_MP2RAGE.nii.gz"
INV2="${ANAT_06}/${SUBJECT}_ses-06_acq-PSIR_inv-2_part-mag_MP2RAGE.nii.gz"

if [ -f "${INV1}" ] && [ -f "${INV2}" ]; then
    # Merge inv1 and inv2 into complex/4D for QUIT
    fslmerge -t "${QMRI_OUT}/ses-06/quit/mp2rage_merged.nii.gz" "${INV1}" "${INV2}"

    # MP2RAGE parameters from CUBRIC protocol
    # Need TI1, TI2, TR, alpha1, alpha2 from JSON sidecars
    python3 -c "
import json
m1 = json.load(open('${INV1}'.replace('.nii.gz', '.json')))
m2 = json.load(open('${INV2}'.replace('.nii.gz', '.json')))
ti1 = m1.get('InversionTime', 0.7)
ti2 = m2.get('InversionTime', 2.5)
tr = m1.get('RepetitionTime', 5.0)
fa1 = m1.get('FlipAngle', 4)
fa2 = m2.get('FlipAngle', 5)
config = {
    'MP2RAGE': {
        'TR': tr,
        'TI': [ti1, ti2],
        'FA': [fa1, fa2],
        'ETL': m1.get('EchoTrainLength', 176),
        'k0': 0
    }
}
json.dump(config, open('${QMRI_OUT}/ses-06/quit/mp2rage.json', 'w'), indent=2)
print(f'TI=[{ti1}, {ti2}], TR={tr}, FA=[{fa1}, {fa2}]')
"

    qi mp2rage "${QMRI_OUT}/ses-06/quit/mp2rage_merged.nii.gz" \
        --json="${QMRI_OUT}/ses-06/quit/mp2rage.json" \
        -o "${QMRI_OUT}/ses-06/quit/MP2_" \
        -T 4 2>&1 | tail -5

    echo "  Outputs:"
    ls "${QMRI_OUT}/ses-06/quit/MP2_"* 2>/dev/null
else
    echo "  MP2RAGE inversions not found"
fi

echo ""
echo "=== QUIT fitting complete ==="
echo "Outputs in: ${QMRI_OUT}/ses-02/quit/ and ${QMRI_OUT}/ses-06/quit/"
