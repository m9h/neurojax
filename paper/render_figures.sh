#!/bin/bash
# Generate publication-quality figures using fsleyes render
# All figures go to paper/figures/ as PNG
set -euo pipefail

WAND="/Users/mhough/dev/wand"
SUB="sub-08033"
DERIV="${WAND}/derivatives"
FS="${DERIV}/freesurfer/${SUB}"
QMRI="${DERIV}/qmri/${SUB}"
PERF="${DERIV}/perfusion/${SUB}"
ADV="${DERIV}/advanced-freesurfer/${SUB}"
DTI="${DERIV}/fsl-dwi/${SUB}/ses-02/dtifit"
FIGS="$(dirname "$0")/figures"
mkdir -p "${FIGS}"

export FSLDIR=/Users/mhough/fsl
source "$FSLDIR/etc/fslconf/fsl.sh"

FSLEYES="fsleyes render"
ORTHO="--scene ortho --xzoom 1200 --yzoom 1200 --zzoom 1200 --hideCursor --hidex --hidey --hidez --hideLabels"
SZ="--size 1800 600"

echo "=== Rendering paper figures with fsleyes ==="

# --- Fig 1: T1w ---
echo "Fig 1: T1w"
${FSLEYES} ${SZ} --scene ortho --hideCursor --hideLabels \
    -of "${FIGS}/fig_t1w.png" \
    "${WAND}/${SUB}/ses-03/anat/${SUB}_ses-03_T1w.nii.gz" \
    --cmap greyscale --displayRange 0 800

# --- Fig 1b: T1w with brain mask overlay ---
echo "Fig 1b: T1w + brain mask"
ANAT_DIR="${DERIV}/fsl-anat/${SUB}/ses-03/anat/${SUB}_ses-03_T1w.anat"
${FSLEYES} ${SZ} --scene ortho --hideCursor --hideLabels \
    -of "${FIGS}/fig_t1w_bet.png" \
    "${ANAT_DIR}/T1_biascorr.nii.gz" --cmap greyscale --displayRange 0 800 \
    "${ANAT_DIR}/T1_biascorr_brain_mask.nii.gz" --cmap red-yellow --alpha 30

# --- Fig 1c: T1w with FreeSurfer surfaces ---
echo "Fig 1c: T1w + surfaces"
if [ -f "${FS}/surf/lh.pial" ]; then
    ${FSLEYES} ${SZ} --scene ortho --hideCursor --hideLabels \
        -of "${FIGS}/fig_t1w_surfaces.png" \
        "${FS}/mri/T1.mgz" --cmap greyscale --displayRange 0 300 \
        "${FS}/surf/lh.white" --colour 1 1 0 --alpha 40 \
        "${FS}/surf/rh.white" --colour 1 1 0 --alpha 40 \
        "${FS}/surf/lh.pial" --colour 1 0 0 --alpha 30 \
        "${FS}/surf/rh.pial" --colour 1 0 0 --alpha 30
fi

# --- Fig 2: T1w/T2w myelin ratio ---
echo "Fig 2: Myelin ratio"
${FSLEYES} ${SZ} --scene ortho --hideCursor --hideLabels \
    -of "${FIGS}/fig_myelin.png" \
    "${ADV}/myelin/T1w_T2w_ratio.nii.gz" --cmap hot --displayRange 0.5 3.0

# --- Fig 3: VFA T1 map ---
echo "Fig 3: VFA T1"
${FSLEYES} ${SZ} --scene ortho --hideCursor --hideLabels \
    -of "${FIGS}/fig_t1map.png" \
    "${QMRI}/ses-02/quit/D1_D1_T1.nii.gz" --cmap hot --displayRange 0.2 2.5

# --- Fig 4: QMT BPF (masked) ---
echo "Fig 4: QMT BPF"
# Mask BPF to brain for cleaner display
fslmaths "${QMRI}/ses-02/quit/QMT_QMT_f_b.nii.gz" -mas "${QMRI}/ses-02/brain_mask.nii.gz" "${FIGS}/bpf_masked.nii.gz" 2>/dev/null || true
BPF_FILE="${FIGS}/bpf_masked.nii.gz"
[ ! -f "${BPF_FILE}" ] && BPF_FILE="${QMRI}/ses-02/quit/QMT_QMT_f_b.nii.gz"
${FSLEYES} ${SZ} --scene ortho --hideCursor --hideLabels \
    -of "${FIGS}/fig_bpf.png" \
    "${BPF_FILE}" --cmap hot --displayRange 0.02 0.20

# --- Fig 5: MEGRE echo 1 ---
echo "Fig 5: MEGRE"
${FSLEYES} ${SZ} --scene ortho --hideCursor --hideLabels \
    -of "${FIGS}/fig_megre.png" \
    "${WAND}/${SUB}/ses-06/anat/${SUB}_ses-06_echo-01_part-mag_MEGRE.nii.gz" \
    --cmap greyscale

# --- Fig 6: MP2RAGE ---
echo "Fig 6: MP2RAGE"
${FSLEYES} ${SZ} --scene ortho --hideCursor --hideLabels \
    -of "${FIGS}/fig_mp2rage.png" \
    "${QMRI}/ses-06/MP2RAGE_uniform.nii.gz" --cmap greyscale

# --- Fig 7: CBF ---
echo "Fig 7: CBF"
CBF="${PERF}/oxford_asl/native_space/perfusion_calib.nii.gz"
if [ -f "${CBF}" ]; then
    ${FSLEYES} ${SZ} --scene ortho --hideCursor --hideLabels \
        -of "${FIGS}/fig_cbf.png" \
        "${CBF}" --cmap hot --displayRange 5 100
fi

# --- Fig 8: CMRO2 ---
echo "Fig 8: CMRO2"
if [ -f "${PERF}/CMRO2_map.nii.gz" ]; then
    ${FSLEYES} ${SZ} --scene ortho --hideCursor --hideLabels \
        -of "${FIGS}/fig_cmro2.png" \
        "${PERF}/CMRO2_map.nii.gz" --cmap hot --displayRange 10 250
fi

# --- Fig 9: FA ---
echo "Fig 9: FA"
${FSLEYES} ${SZ} --scene ortho --hideCursor --hideLabels \
    -of "${FIGS}/fig_fa.png" \
    "${DTI}/dtifit_FA.nii.gz" --cmap greyscale --displayRange 0 0.8

# --- Fig 10: FA + V1 colour ---
echo "Fig 10: FA + V1 colour"
${FSLEYES} ${SZ} --scene ortho --hideCursor --hideLabels \
    -of "${FIGS}/fig_fa_colour.png" \
    "${DTI}/dtifit_FA.nii.gz" --cmap greyscale --displayRange 0 0.8 \
    "${DTI}/dtifit_V1.nii.gz" --overlayType rgbvector --modulateImage "${DTI}/dtifit_FA.nii.gz"

# --- Fig 11: g-ratio ---
echo "Fig 11: g-ratio"
${FSLEYES} ${SZ} --scene ortho --hideCursor --hideLabels \
    -of "${FIGS}/fig_gratio.png" \
    "${QMRI}/gratio/g_ratio_proxy.nii.gz" --cmap cool --displayRange 0.5 1.0

echo ""
echo "=== All figures rendered to ${FIGS}/ ==="
ls -lh "${FIGS}"/*.png
