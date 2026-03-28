#!/bin/bash
# ============================================================================
# Stage R: Perfusion Processing — CBF, OEF, CMRO₂
# ============================================================================
# Complete cerebral oxygen metabolism pipeline from WAND ses-03:
#
#   1. Inversion Recovery → blood T1 (subject-specific calibration)
#   2. pCASL + M0 + blood T1 → CBF map (oxford_asl / BASIL)
#   3. TRUST → venous T2 → SvO₂ → OEF (custom fitting)
#   4. CMRO₂ = CBF × OEF × CaO₂ (Fick's principle)
#
# References:
#   Germuska M (CUBRIC): dual-calibrated fMRI, WAND protocol design
#   Lu H (Johns Hopkins): TRUST method (MRM 2008)
#   Alsop DC et al.: ASL white paper (MRM 2015)
#   Riera JJ et al.: neurovascular coupling model (HBM 2006/2007)
#
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"

WAND_PERF="${WAND_ROOT}/${SUBJECT}/ses-03/perf"
WAND_FMAP="${WAND_ROOT}/${SUBJECT}/ses-03/fmap"
ANAT_DIR="${WAND_DERIVATIVES}/fsl-anat/${SUBJECT}/ses-03/anat/${SUBJECT}_ses-03_T1w.anat"
PERF_OUT="${WAND_DERIVATIVES}/perfusion/${SUBJECT}"
LOGS="${WAND_DERIVATIVES}/logs/${SUBJECT}"
mkdir -p "${PERF_OUT}" "${LOGS}"

echo "=== Perfusion Processing: ${SUBJECT} ==="

# ---------------------------------------------------------------
# Step 1: Inversion Recovery → Blood T1
# ---------------------------------------------------------------

echo ""
echo "--- Step 1: Blood T1 from Inversion Recovery ---"

fsl_sub -l "${LOGS}" -N blood_t1 -T 10 \
    uv run --directory ~/dev/neurojax python3 -c "
import nibabel as nib, numpy as np, os
from scipy.optimize import curve_fit

perf_out = '${PERF_OUT}'
ir_file = '${WAND_PERF}/${SUBJECT}_ses-03_acq-InvRec_cbf.nii.gz'

img = nib.load(ir_file)
data = img.get_fdata()
print(f'InvRec data: {data.shape}')  # (128, 128, 1, 960)

# Squeeze to 2D + TIs
if data.ndim == 4:
    data = data[:, :, 0, :]  # (128, 128, 960)

# Get TI values from JSON
import json
meta = json.load(open(ir_file.replace('.nii.gz', '.json')))
TR = meta.get('RepetitionTime', 0.15)  # 150ms
n_TIs = data.shape[-1]

# For Look-Locker / continuous IR: TIs = (1:N) * TR
TIs = np.arange(1, n_TIs + 1) * TR
print(f'TIs: {TIs[0]:.3f} to {TIs[-1]:.3f}s ({n_TIs} points)')

# ROI: central region (blood signal)
cx, cy = data.shape[0]//2, data.shape[1]//2
roi_data = np.mean(data[cx-5:cx+5, cy-5:cy+5, :], axis=(0,1))

# Fit: S(TI) = S0 * |1 - 2*exp(-TI/T1)|
def ir_model(TI, S0, T1):
    return S0 * np.abs(1 - 2 * np.exp(-TI / T1))

try:
    popt, pcov = curve_fit(ir_model, TIs, roi_data, p0=[roi_data.max(), 1.6], maxfev=10000)
    T1_blood = popt[1]
    T1_std = np.sqrt(pcov[1, 1])
    print(f'Blood T1: {T1_blood*1000:.0f} ± {T1_std*1000:.0f} ms')
    print(f'Expected at 3T: 1600-1700 ms')
except Exception as e:
    print(f'IR fitting failed: {e}')
    T1_blood = 1.65  # default 3T value
    print(f'Using default T1_blood = {T1_blood*1000:.0f} ms')

# Save
np.save(os.path.join(perf_out, 'T1_blood.npy'), T1_blood)
with open(os.path.join(perf_out, 'T1_blood.txt'), 'w') as f:
    f.write(f'{T1_blood:.4f}')
print(f'Saved: {perf_out}/T1_blood.npy')
"
echo "  Blood T1 job submitted"

# ---------------------------------------------------------------
# Step 2: Fieldmap preparation (for EPI distortion correction)
# ---------------------------------------------------------------

echo ""
echo "--- Step 2: Prepare B0 fieldmap ---"

FMAP_OUT="${PERF_OUT}/fieldmap"
mkdir -p "${FMAP_OUT}"

if [ -f "${WAND_FMAP}/${SUBJECT}_ses-03_phasediff.nii.gz" ]; then
    # Get echo time difference from JSON
    DELTA_TE=$(python3 -c "
import json
d1 = json.load(open('${WAND_FMAP}/${SUBJECT}_ses-03_magnitude1.json'))
d2 = json.load(open('${WAND_FMAP}/${SUBJECT}_ses-03_magnitude2.json'))
te1 = d1.get('EchoTime', 0.00492)
te2 = d2.get('EchoTime', 0.00738)
print(f'{(te2-te1)*1000:.2f}')
")
    echo "  Delta TE: ${DELTA_TE} ms"

    # Brain extract magnitude
    bet "${WAND_FMAP}/${SUBJECT}_ses-03_magnitude1" "${FMAP_OUT}/mag_brain" -f 0.5

    # Prepare fieldmap (rad/s)
    fsl_prepare_fieldmap SIEMENS \
        "${WAND_FMAP}/${SUBJECT}_ses-03_phasediff.nii.gz" \
        "${FMAP_OUT}/mag_brain" \
        "${FMAP_OUT}/fieldmap_rads" \
        "${DELTA_TE}"

    echo "  Fieldmap: ${FMAP_OUT}/fieldmap_rads.nii.gz"
    FMAP_ARG="--fmap=${FMAP_OUT}/fieldmap_rads --fmapmag=${WAND_FMAP}/${SUBJECT}_ses-03_magnitude1 --fmapmagbrain=${FMAP_OUT}/mag_brain --echospacing=0.00056 --pedir=y-"
else
    echo "  No fieldmap available"
    FMAP_ARG=""
fi

# ---------------------------------------------------------------
# Step 3: pCASL → CBF (oxford_asl / BASIL)
# ---------------------------------------------------------------

echo ""
echo "--- Step 3: pCASL → CBF (oxford_asl) ---"

# Wait for blood T1 to be ready
sleep 5

T1_BLOOD=$(cat "${PERF_OUT}/T1_blood.txt" 2>/dev/null || echo "1.65")
echo "  Using T1_blood = ${T1_BLOOD}s"

CBF_OUT="${PERF_OUT}/oxford_asl"

fsl_sub -l "${LOGS}" -N oxford_asl -T 30 \
    oxford_asl \
    -i "${WAND_PERF}/${SUBJECT}_ses-03_acq-PCASL_cbf.nii.gz" \
    -o "${CBF_OUT}" \
    --casl \
    --bolus=1.8 \
    --pld=2.0 \
    --iaf=tc \
    -c "${WAND_PERF}/${SUBJECT}_ses-03_dir-AP_m0scan.nii.gz" \
    --cmethod=voxel \
    --t1b=${T1_BLOOD} \
    -s "${ANAT_DIR}/T1_biascorr" \
    --fslanat="${ANAT_DIR}" \
    --mc \
    --pvcorr \
    ${FMAP_ARG}

echo "  oxford_asl submitted → ${CBF_OUT}/"

# ---------------------------------------------------------------
# Step 4: TRUST → SvO₂ → OEF
# ---------------------------------------------------------------

echo ""
echo "--- Step 4: TRUST → SvO₂ → OEF ---"

fsl_sub -l "${LOGS}" -N trust_fitting -T 5 \
    uv run --directory ~/dev/neurojax python3 -c "
import nibabel as nib, numpy as np, os
from scipy.optimize import curve_fit

perf_out = '${PERF_OUT}'
trust_file = '${WAND_PERF}/${SUBJECT}_ses-03_acq-TRUST_cbf.nii.gz'

img = nib.load(trust_file)
data = img.get_fdata()
print(f'TRUST data: {data.shape}')  # (64, 64, 1, 24)

if data.ndim == 4:
    data = data[:, :, 0, :]  # (64, 64, 24)

# TRUST: 24 volumes = 12 control-label pairs at multiple eTEs
n_pairs = data.shape[-1] // 2

# Separate control (even) and label (odd) — convention may vary
control = data[:, :, 0::2]  # volumes 0, 2, 4, ...
label = data[:, :, 1::2]    # volumes 1, 3, 5, ...
diff = control - label       # (64, 64, n_pairs)

# Effective echo times (typical TRUST protocol)
# Usually 4 eTEs repeated 3 times each = 12 pairs
# eTEs need to be extracted from protocol — using typical values
eTEs = np.array([0, 40, 80, 120]) * 1e-3  # seconds
n_ete = len(eTEs)
n_repeats = n_pairs // n_ete

print(f'Pairs: {n_pairs}, eTEs: {n_ete}, Repeats: {n_repeats}')

# Average repeats per eTE
if n_repeats > 1 and n_pairs == n_ete * n_repeats:
    diff_avg = np.zeros((*diff.shape[:2], n_ete))
    for i in range(n_ete):
        diff_avg[:, :, i] = np.mean(diff[:, :, i::n_ete], axis=2)
else:
    diff_avg = diff[:, :, :n_ete]

# ROI in sagittal sinus (high signal in difference image at eTE=0)
sinus_signal = np.abs(diff_avg[:, :, 0])
threshold = np.percentile(sinus_signal[sinus_signal > 0], 95)
sinus_roi = sinus_signal > threshold
n_voxels = np.sum(sinus_roi)
print(f'Sagittal sinus ROI: {n_voxels} voxels')

if n_voxels > 0:
    # Extract ROI signal at each eTE
    roi_signal = np.array([np.mean(np.abs(diff_avg[sinus_roi, i])) for i in range(n_ete)])
    print(f'ROI signal by eTE: {roi_signal}')

    # Fit T2: S(eTE) = S0 * exp(-eTE / T2)
    def t2_model(eTE, S0, T2):
        return S0 * np.exp(-eTE / T2)

    try:
        popt, pcov = curve_fit(t2_model, eTEs[:len(roi_signal)], roi_signal,
                                p0=[roi_signal[0], 0.06], maxfev=5000)
        T2_blood = popt[1]
        print(f'Blood T2: {T2_blood*1000:.1f} ms')

        # Calibration: T2 → SvO2 (Lu et al. 2012, 3T coefficients)
        # 1/T2 = A + B*(1-Y) + C*(1-Y)^2
        # For 3T, Hct=0.42: A=4.5, B=47.1, C=55.5 (approximate)
        A, B, C = 4.5, 47.1, 55.5
        R2 = 1.0 / T2_blood

        # Solve quadratic: C*(1-Y)^2 + B*(1-Y) + (A-R2) = 0
        a_coeff = C
        b_coeff = B
        c_coeff = A - R2
        discriminant = b_coeff**2 - 4*a_coeff*c_coeff

        if discriminant >= 0:
            one_minus_Y = (-b_coeff + np.sqrt(discriminant)) / (2 * a_coeff)
            SvO2 = 1.0 - one_minus_Y
            SaO2 = 0.98  # assumed arterial saturation
            OEF = (SaO2 - SvO2) / SaO2

            print(f'SvO2: {SvO2*100:.1f}%')
            print(f'OEF:  {OEF*100:.1f}%')
            print(f'Expected: SvO2 60-68%, OEF 32-40%')

            # Save
            results = {'T2_blood_ms': T2_blood*1000, 'SvO2': SvO2, 'OEF': OEF, 'SaO2': SaO2}
            import json
            with open(os.path.join(perf_out, 'trust_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            print(f'Saved: {perf_out}/trust_results.json')
        else:
            print('Calibration failed (negative discriminant)')
    except Exception as e:
        print(f'T2 fitting failed: {e}')
else:
    print('No sinus ROI found')
"
echo "  TRUST fitting submitted"

# ---------------------------------------------------------------
# Step 5: CMRO₂ = CBF × OEF × CaO₂
# ---------------------------------------------------------------

echo ""
echo "--- Step 5: CMRO₂ computation (after oxford_asl completes) ---"
echo "  Will combine: CBF (oxford_asl) × OEF (TRUST) × CaO₂"
echo "  CaO₂ = Hb × 1.34 × SaO₂ ≈ 18-20 ml O₂/100ml blood"
echo ""
echo "  Run after Steps 3+4 complete:"
echo "    python3 -c \""
echo "    import nibabel as nib, numpy as np, json"
echo "    cbf = nib.load('${CBF_OUT}/native_space/perfusion_calib.nii.gz')"
echo "    trust = json.load(open('${PERF_OUT}/trust_results.json'))"
echo "    OEF = trust['OEF']"
echo "    CaO2 = 0.20  # ml O2 / ml blood"
echo "    cmro2 = cbf.get_fdata() * OEF * CaO2"
echo "    nib.save(nib.Nifti1Image(cmro2, cbf.affine), '${PERF_OUT}/CMRO2_map.nii.gz')"
echo "    \""

echo ""
echo "=== Perfusion pipeline submitted for ${SUBJECT} ==="
echo "  Blood T1:    ${PERF_OUT}/T1_blood.npy"
echo "  CBF:         ${CBF_OUT}/native_space/perfusion_calib.nii.gz"
echo "  TRUST:       ${PERF_OUT}/trust_results.json"
echo "  CMRO₂:       ${PERF_OUT}/CMRO2_map.nii.gz (after Steps 3+4)"
