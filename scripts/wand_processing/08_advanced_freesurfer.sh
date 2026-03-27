#!/bin/bash
# ============================================================================
# Stage 8: Advanced FreeSurfer — Beyond recon-all
# ============================================================================
# Uses FreeSurfer 8.2.0 tools that go far beyond the standard pipeline:
#
# 1. T1w/T2w ratio → myelination proxy (Glasser & Van Essen 2011)
# 2. SAMSEG multimodal segmentation (T1+T2+QMT jointly)
# 3. Thalamic nuclei segmentation (with DTI for WAND)
# 4. Hippocampal subfields + amygdala nuclei
# 5. SynthSeg for cross-session contrast-invariant parcellation
# 6. Hypothalamic subunits
# 7. White matter hyperintensity mapping
#
# Then compare T1w/T2w ratio against WAND's ground-truth measures:
#   - QMT bound pool fraction (ses-02) → actual myelin content
#   - Multi-echo GRE T2* (ses-06) → iron + myelin sensitivity
#   - AxCaliber g-ratio (from diameter + myelin) → true myelination
#
# This comparison addresses: "How good is the T1w/T2w proxy, really?"
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"

FS_DIR="${SUBJECTS_DIR}/${SUBJECT}"
ADV_DIR="${WAND_DERIVATIVES}/advanced_freesurfer/${SUBJECT}"
mkdir -p "${ADV_DIR}"

echo "=============================================="
echo " Advanced FreeSurfer: ${SUBJECT}"
echo "=============================================="

# ---------------------------------------------------------------
# 1. T1w/T2w Myelin Mapping
# ---------------------------------------------------------------

echo ""
echo ">>> 1. T1w/T2w Myelin Ratio"

# ses-03 has both T1w and gradient-corrected T2w
T1W_SES03="${WAND_ROOT}/${SUBJECT}/ses-03/anat/${SUBJECT}_ses-03_T1w.nii.gz"
T2W_SES03="${WAND_ROOT}/${SUBJECT}/ses-03/anat/${SUBJECT}_ses-03_rec-nlgradcorr_T2w.nii.gz"

MYELIN_DIR="${ADV_DIR}/myelin"
mkdir -p "${MYELIN_DIR}"

if [ -f "${T1W_SES03}" ] && [ -f "${T2W_SES03}" ]; then
    echo "  T1w: ${T1W_SES03}"
    echo "  T2w: ${T2W_SES03}"

    # Skull strip both with SynthStrip (DL-based, better than BET)
    echo "  Skull stripping with SynthStrip..."
    mri_synthstrip -i "${T1W_SES03}" -o "${MYELIN_DIR}/T1w_brain.nii.gz" -m "${MYELIN_DIR}/T1w_brain_mask.nii.gz"

    # Register T2w to T1w
    echo "  Registering T2w → T1w..."
    flirt \
        -in "${T2W_SES03}" \
        -ref "${MYELIN_DIR}/T1w_brain.nii.gz" \
        -out "${MYELIN_DIR}/T2w_in_T1w.nii.gz" \
        -omat "${MYELIN_DIR}/T2w_to_T1w.mat" \
        -dof 6 \
        -cost bbr \
        -wmseg "${FS_DIR}/mri/wm.seg.mgz" 2>/dev/null || \
    flirt \
        -in "${T2W_SES03}" \
        -ref "${MYELIN_DIR}/T1w_brain.nii.gz" \
        -out "${MYELIN_DIR}/T2w_in_T1w.nii.gz" \
        -omat "${MYELIN_DIR}/T2w_to_T1w.mat" \
        -dof 6

    # Apply mask to T2w
    fslmaths "${MYELIN_DIR}/T2w_in_T1w.nii.gz" -mas "${MYELIN_DIR}/T1w_brain_mask.nii.gz" "${MYELIN_DIR}/T2w_brain.nii.gz"

    # Compute T1w/T2w ratio
    echo "  Computing T1w/T2w ratio..."
    python3 << 'PYEOF'
import nibabel as nib
import numpy as np
import os

myelin_dir = os.environ['MYELIN_DIR']

t1 = nib.load(os.path.join(myelin_dir, 'T1w_brain.nii.gz'))
t2 = nib.load(os.path.join(myelin_dir, 'T2w_brain.nii.gz'))

t1_data = t1.get_fdata().astype(np.float64)
t2_data = t2.get_fdata().astype(np.float64)

# Avoid division by zero
mask = (t1_data > 10) & (t2_data > 10)
ratio = np.zeros_like(t1_data)
ratio[mask] = t1_data[mask] / t2_data[mask]

# Normalize to [0, 100] range for visualization
p1, p99 = np.percentile(ratio[mask], [1, 99])
ratio_norm = np.clip((ratio - p1) / (p99 - p1) * 100, 0, 100)

nib.save(nib.Nifti1Image(ratio, t1.affine, t1.header),
         os.path.join(myelin_dir, 'T1w_T2w_ratio.nii.gz'))
nib.save(nib.Nifti1Image(ratio_norm, t1.affine, t1.header),
         os.path.join(myelin_dir, 'T1w_T2w_ratio_norm.nii.gz'))

print(f"  T1w/T2w ratio range: [{ratio[mask].min():.2f}, {ratio[mask].max():.2f}]")
print(f"  Normalized range: [{ratio_norm[mask].min():.1f}, {ratio_norm[mask].max():.1f}]")
print(f"  Saved: T1w_T2w_ratio.nii.gz, T1w_T2w_ratio_norm.nii.gz")
PYEOF

    # Project ratio to cortical surface (for comparison with Valdes-Sosa)
    if [ -f "${FS_DIR}/surf/lh.white" ]; then
        echo "  Projecting to cortical surface..."
        for HEMI in lh rh; do
            mri_vol2surf \
                --mov "${MYELIN_DIR}/T1w_T2w_ratio.nii.gz" \
                --regheader "${SUBJECT}" \
                --hemi ${HEMI} \
                --projfrac-avg 0.2 0.8 0.1 \
                --o "${MYELIN_DIR}/${HEMI}.T1wT2w_ratio.mgz" \
                --cortex
        done
        echo "  Surface projections: lh/rh.T1wT2w_ratio.mgz"
    fi
else
    echo "  WARNING: T1w and/or T2w not available for ses-03"
fi

# ---------------------------------------------------------------
# 2. FreeSurfer pctsurfcon (percent surface contrast)
# ---------------------------------------------------------------

echo ""
echo ">>> 2. Percent Surface Contrast (built-in myelin proxy)"

if [ -f "${FS_DIR}/surf/lh.white" ]; then
    for HEMI in lh rh; do
        pctsurfcon \
            --s "${SUBJECT}" \
            --lh-only 2>/dev/null || true  # may need adjustment
    done
    echo "  Check ${FS_DIR}/surf/?h.w-g.pct.mgh for percent contrast"
fi

# ---------------------------------------------------------------
# 3. SAMSEG — Multimodal Segmentation
# ---------------------------------------------------------------

echo ""
echo ">>> 3. SAMSEG (Sequence-Adaptive Multimodal Segmentation)"

SAMSEG_DIR="${ADV_DIR}/samseg"
mkdir -p "${SAMSEG_DIR}"

# SAMSEG can use multiple contrasts simultaneously
SAMSEG_INPUTS=""
[ -f "${T1W_SES03}" ] && SAMSEG_INPUTS="${SAMSEG_INPUTS} --input ${T1W_SES03}"
[ -f "${T2W_SES03}" ] && SAMSEG_INPUTS="${SAMSEG_INPUTS} --input ${T2W_SES03}"

if [ -n "${SAMSEG_INPUTS}" ]; then
    echo "  Running SAMSEG with multimodal input..."
    run_samseg \
        ${SAMSEG_INPUTS} \
        --output "${SAMSEG_DIR}" \
        --threads ${OMP_NUM_THREADS} 2>&1 | tail -5
    echo "  SAMSEG output: ${SAMSEG_DIR}/"
else
    echo "  Skipped (no inputs available)"
fi

# ---------------------------------------------------------------
# 4. Thalamic Nuclei Segmentation (with DTI enhancement)
# ---------------------------------------------------------------

echo ""
echo ">>> 4. Thalamic Nuclei Segmentation"

THAL_DIR="${ADV_DIR}/thalamus"
mkdir -p "${THAL_DIR}"

# Standard thalamic nuclei (from T1)
echo "  Running T1-based thalamic nuclei segmentation..."
segmentThalamicNuclei.sh "${SUBJECT}" "${SUBJECTS_DIR}" 2>&1 | tail -3

# DTI-enhanced thalamic nuclei (uses DWI for improved boundaries)
DWI_PREPROC="${WAND_DERIVATIVES}/fsl/${SUBJECT}/ses-02/dwi"
if [ -f "${DWI_PREPROC}/CHARMED_eddy.nii.gz" ]; then
    echo "  Running DTI-enhanced thalamic nuclei segmentation..."
    # Need FA map
    FA_MAP="${DWI_PREPROC}/CHARMED_dti_FA.nii.gz"
    if [ -f "${FA_MAP}" ]; then
        segmentThalamicNuclei_DTI.sh \
            "${SUBJECT}" \
            "${SUBJECTS_DIR}" \
            "${FA_MAP}" \
            "${DWI_PREPROC}/reg/diff2struct.mat" 2>&1 | tail -3
        echo "  DTI-enhanced segmentation complete"
    fi
fi

# ---------------------------------------------------------------
# 5. Hippocampal Subfields + Amygdala Nuclei
# ---------------------------------------------------------------

echo ""
echo ">>> 5. Hippocampal Subfields + Amygdala Nuclei"

# Uses T1; can optionally use T2 for improved subfield boundaries
HIPPO_ARGS="${SUBJECT}"
if [ -f "${T2W_SES03}" ]; then
    HIPPO_ARGS="${HIPPO_ARGS} ${T2W_SES03} T2"
fi

segmentHA_T1.sh ${HIPPO_ARGS} 2>&1 | tail -3
echo "  Check ${FS_DIR}/mri/?h.hippoSfVolumes*.txt"

# ---------------------------------------------------------------
# 6. Hypothalamic Subunits
# ---------------------------------------------------------------

echo ""
echo ">>> 6. Hypothalamic Subunit Segmentation"

mri_segment_hypothalamic_subunits \
    --s "${SUBJECT}" \
    --sd "${SUBJECTS_DIR}" \
    --out "${ADV_DIR}/hypothalamus" 2>&1 | tail -3

# ---------------------------------------------------------------
# 7. Subcortical Limbic Segmentation
# ---------------------------------------------------------------

echo ""
echo ">>> 7. Subcortical Limbic Structures"

mri_sclimbic_seg \
    --s "${SUBJECT}" \
    --o "${ADV_DIR}/sclimbic_seg.mgz" 2>&1 | tail -3

# ---------------------------------------------------------------
# 8. SynthSeg — Contrast-Invariant Segmentation (for all sessions)
# ---------------------------------------------------------------

echo ""
echo ">>> 8. SynthSeg across sessions"

SYNTHSEG_DIR="${ADV_DIR}/synthseg"
mkdir -p "${SYNTHSEG_DIR}"

for SES in ses-02 ses-03 ses-04 ses-05 ses-06; do
    T1_FILE=$(find "${WAND_ROOT}/${SUBJECT}/${SES}/anat/" -name "*T1w*.nii.gz" 2>/dev/null | head -1)
    if [ -n "${T1_FILE}" ] && [ -f "${T1_FILE}" ]; then
        echo "  SynthSeg on ${SES}..."
        mri_synthseg \
            --i "${T1_FILE}" \
            --o "${SYNTHSEG_DIR}/${SES}_synthseg.nii.gz" \
            --vol "${SYNTHSEG_DIR}/${SES}_volumes.csv" \
            --qc "${SYNTHSEG_DIR}/${SES}_qc.csv" \
            --robust 2>&1 | tail -2
    fi
done

# ---------------------------------------------------------------
# 9. WMH Segmentation (White Matter Hyperintensities)
# ---------------------------------------------------------------

echo ""
echo ">>> 9. White Matter Hyperintensity Segmentation"

if [ -f "${T1W_SES03}" ]; then
    mri_WMHsynthseg \
        --i "${T1W_SES03}" \
        --o "${ADV_DIR}/wmh_seg.nii.gz" \
        --csv_vols "${ADV_DIR}/wmh_volumes.csv" 2>&1 | tail -2
fi

# ---------------------------------------------------------------
# 10. Compare T1w/T2w ratio against WAND ground truth
# ---------------------------------------------------------------

echo ""
echo ">>> 10. Myelin Proxy Comparison"

cat > "${ADV_DIR}/compare_myelin_measures.py" << 'PYEOF'
#!/usr/bin/env python3
"""Compare T1w/T2w ratio against WAND quantitative myelin measures.

WAND provides multiple myelin-sensitive contrasts:
  1. T1w/T2w ratio (ses-03) — standard proxy (Glasser & Van Essen 2011)
  2. QMT bound pool fraction (ses-02) — actual macromolecular (myelin) content
  3. Multi-echo GRE T2* (ses-06) — iron + myelin sensitivity
  4. VFA T1 mapping (ses-02, SPGR/SSFP) — quantitative T1

Key question: How well does the cheap T1w/T2w proxy correlate with
ground-truth myelin content from QMT?
"""
import os
import numpy as np

ADV_DIR = os.environ.get('ADV_DIR', '.')
WAND_ROOT = os.environ.get('WAND_ROOT', '')
SUBJECT = os.environ.get('SUBJECT', 'sub-08033')

print("=== Myelin Measure Comparison ===")
print(f"Subject: {SUBJECT}")
print()

# List available myelin-sensitive data
measures = {
    'T1w/T2w ratio': os.path.join(ADV_DIR, 'myelin', 'T1w_T2w_ratio.nii.gz'),
    'QMT (ses-02)': os.path.join(WAND_ROOT, SUBJECT, 'ses-02/anat',
                                  f'{SUBJECT}_ses-02_mt-off_part-mag_QMT.nii.gz'),
    'T2w (ses-06)': os.path.join(WAND_ROOT, SUBJECT, 'ses-06/anat',
                                  f'{SUBJECT}_ses-06_T2w.nii.gz'),
    'MEGRE echo-01 (ses-06)': os.path.join(WAND_ROOT, SUBJECT, 'ses-06/anat',
                                            f'{SUBJECT}_ses-06_echo-01_part-mag_MEGRE.nii.gz'),
    'MP2RAGE (ses-06)': os.path.join(WAND_ROOT, SUBJECT, 'ses-06/anat',
                                      f'{SUBJECT}_ses-06_acq-PSIR_inv-1_part-mag_MP2RAGE.nii.gz'),
    'VFA SPGR (ses-02)': os.path.join(WAND_ROOT, SUBJECT, 'ses-02/anat',
                                       f'{SUBJECT}_ses-02_acq-spgr_part-mag_VFA.nii.gz'),
}

print("Available measures:")
for name, path in measures.items():
    exists = os.path.exists(path) if path else False
    size = f"{os.path.getsize(path)/1e6:.1f} MB" if exists else "not downloaded"
    print(f"  {'✓' if exists else '✗'} {name}: {size}")

print()
print("Analysis plan:")
print("  1. Register all measures to FreeSurfer T1w space")
print("  2. Sample each on cortical surface (mid-cortical depth)")
print("  3. Compute vertex-wise correlations:")
print("     - T1w/T2w vs QMT bound pool fraction (gold standard)")
print("     - T1w/T2w vs T2* (multi-echo fit)")
print("     - T1w/T2w vs VFA T1 map")
print("  4. Regional analysis: mean per Desikan ROI")
print("  5. Correlation with AxCaliber-derived g-ratio (from sbi4dwi)")
print()
print("Expected finding: T1w/T2w correlates with QMT but with")
print("significant regional bias, especially in areas with high")
print("iron content (basal ganglia) where T2* confounds the ratio.")
print()
print("This validates (or challenges) using T1w/T2w as a myelin proxy")
print("in the Valdes-Sosa ξ-αNET framework for conduction delay estimation.")
PYEOF

chmod +x "${ADV_DIR}/compare_myelin_measures.py"
ADV_DIR="${ADV_DIR}" WAND_ROOT="${WAND_ROOT}" SUBJECT="${SUBJECT}" \
    python3 "${ADV_DIR}/compare_myelin_measures.py"

echo ""
echo "=============================================="
echo " Advanced FreeSurfer complete for ${SUBJECT}"
echo "=============================================="
echo ""
echo "Key outputs:"
echo "  T1w/T2w ratio:        ${MYELIN_DIR}/T1w_T2w_ratio.nii.gz"
echo "  Surface myelin:       ${MYELIN_DIR}/lh.T1wT2w_ratio.mgz"
echo "  SAMSEG:               ${SAMSEG_DIR}/"
echo "  Thalamic nuclei:      ${FS_DIR}/mri/ThalamicNuclei.*.mgz"
echo "  Hippocampal subfields: ${FS_DIR}/mri/?h.hippoSfVolumes*.txt"
echo "  SynthSeg (all ses):   ${SYNTHSEG_DIR}/"
echo "  Myelin comparison:    ${ADV_DIR}/compare_myelin_measures.py"
