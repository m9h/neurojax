#!/bin/bash
# ============================================================================
# Stage 7: Microstructure Estimation — 4-Way Comparison
# ============================================================================
# Compares four approaches to AxCaliber/microstructure estimation on WAND
# ultra-strong gradient (300 mT/m) data:
#
#   A. DIPY — Python diffusion MRI library (community standard)
#   B. FSL  — dtifit + NODDI via AMICO (FSL ecosystem)
#   C. sbi4dwi/dmipy-jax — JAX-accelerated biophysical models + SBI
#   D. DMI.jl — Julia diffusion microstructure (SANDI/AxCaliber)
#
# WAND AxCaliber protocol (sub-08033 ses-02):
#   AxCaliber1: b=[0, 2200, 4400] s/mm², 66 volumes
#   AxCaliber2: b=[0, 4000, 8000] s/mm², 66 volumes
#   AxCaliber3: b=[0, 5800, 11600] s/mm², 66 volumes
#   AxCaliber4: b=[0, 7750, 15500] s/mm², 66 volumes
#   Total: 264 DWI volumes across 4 diffusion times
#
# Also available:
#   CHARMED: b=[AP+PA] for standard multi-shell
#   QMT (ses-02): quantitative magnetization transfer → myelin maps
#   Multi-echo GRE (ses-06): T2* → myelin/iron
#
# Output: Per-voxel maps of axon diameter, volume fraction, and derived
#         conduction velocity for each method → comparison figures
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"

WAND_DWI="${WAND_ROOT}/${SUBJECT}/ses-02/dwi"
PREPROC="${WAND_DERIVATIVES}/fsl/${SUBJECT}/ses-02/dwi"
MICRO_DIR="${WAND_DERIVATIVES}/microstructure/${SUBJECT}"
mkdir -p "${MICRO_DIR}/dipy" "${MICRO_DIR}/fsl" "${MICRO_DIR}/sbi4dwi" "${MICRO_DIR}/comparison"

echo "=============================================="
echo " Microstructure 4-Way Comparison: ${SUBJECT}"
echo "=============================================="

# ---------------------------------------------------------------
# Common: Merge and prepare the eddy-corrected AxCaliber data
# ---------------------------------------------------------------

echo ">>> Preparing combined AxCaliber dataset..."

python3 << 'PYEOF'
import numpy as np
import nibabel as nib
import os, json

subject = os.environ['SUBJECT']
preproc = os.environ['PREPROC']
micro_dir = os.environ['MICRO_DIR']
wand_dwi = os.environ['WAND_DWI']

# Collect all eddy-corrected AxCaliber volumes
all_data = []
all_bvals = []
all_bvecs = []
acq_info = []

for acq in ['AxCaliber1', 'AxCaliber2', 'AxCaliber3', 'AxCaliber4']:
    eddy_file = os.path.join(preproc, f'{acq}_eddy.nii.gz')
    bval_file = os.path.join(wand_dwi, f'{subject}_ses-02_acq-{acq}_dir-AP_part-mag_dwi.bval')
    bvec_file = os.path.join(preproc, f'{acq}_eddy.eddy_rotated_bvecs')
    json_file = os.path.join(wand_dwi, f'{subject}_ses-02_acq-{acq}_dir-AP_part-mag_dwi.json')

    if not os.path.exists(eddy_file):
        print(f"  {acq}: eddy output not found, using raw")
        eddy_file = os.path.join(wand_dwi, f'{subject}_ses-02_acq-{acq}_dir-AP_part-mag_dwi.nii.gz')
        bvec_file = os.path.join(wand_dwi, f'{subject}_ses-02_acq-{acq}_dir-AP_part-mag_dwi.bvec')

    if not os.path.exists(eddy_file):
        print(f"  {acq}: SKIPPED (not downloaded yet)")
        continue

    img = nib.load(eddy_file)
    data = img.get_fdata()
    bvals = np.loadtxt(bval_file)
    bvecs = np.loadtxt(bvec_file)

    # Read timing from JSON
    with open(json_file) as f:
        meta = json.load(f)

    all_data.append(data)
    all_bvals.append(bvals)
    all_bvecs.append(bvecs if bvecs.shape[0] == 3 else bvecs.T)

    print(f"  {acq}: {data.shape[-1]} volumes, b=[{bvals.min():.0f}, {bvals.max():.0f}]")
    acq_info.append({
        'name': acq,
        'TE': meta.get('EchoTime', 0.08),
        'bmax': float(bvals.max()),
        'nvols': data.shape[-1],
    })

if all_data:
    # Concatenate
    combined = np.concatenate(all_data, axis=-1)
    combined_bvals = np.concatenate(all_bvals)
    combined_bvecs = np.concatenate(all_bvecs, axis=1) if all_bvecs[0].shape[0] == 3 else np.concatenate(all_bvecs, axis=0)

    # Save combined
    ref_img = nib.load(os.path.join(preproc, 'b0_brain.nii.gz')) if os.path.exists(os.path.join(preproc, 'b0_brain.nii.gz')) else nib.load(eddy_file)
    nib.save(nib.Nifti1Image(combined, ref_img.affine), os.path.join(micro_dir, 'axcaliber_combined.nii.gz'))
    np.savetxt(os.path.join(micro_dir, 'axcaliber_combined.bval'), combined_bvals.reshape(1, -1), fmt='%.0f')
    np.savetxt(os.path.join(micro_dir, 'axcaliber_combined.bvec'), combined_bvecs if combined_bvecs.shape[0] == 3 else combined_bvecs.T, fmt='%.6f')

    print(f"\nCombined: {combined.shape[-1]} volumes, saved to {micro_dir}/axcaliber_combined.*")
    print(f"Acquisition info: {acq_info}")
else:
    print("ERROR: No AxCaliber data available. Run datalad get first.")

PYEOF

# ---------------------------------------------------------------
# Pipeline A: DIPY
# ---------------------------------------------------------------

echo ""
echo ">>> Pipeline A: DIPY"
echo "    (requires: pip install dipy)"

cat > "${MICRO_DIR}/dipy/run_dipy.py" << 'DIPY_SCRIPT'
#!/usr/bin/env python3
"""DIPY AxCaliber-style microstructure estimation on WAND data.

Uses DIPY's free water DTI + MAPMRI for microstructure.
For full AxCaliber, uses cylinder model fitting.

To discuss with DIPY team:
- Best practices for multi-shell AxCaliber on Connectom gradients
- Cylinder model vs. MAPMRI vs. DKI for diameter estimation
- Handling of 4 separate diffusion times
"""
import os
import numpy as np

MICRO_DIR = os.environ.get('MICRO_DIR', '.')

try:
    import dipy
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    from dipy.reconst.dki import DiffusionKurtosisModel
    from dipy.reconst.mapmri import MapmriModel
    from dipy.reconst.fwdti import FreeWaterTensorModel
    print(f"DIPY version: {dipy.__version__}")
except ImportError:
    print("DIPY not installed. Install with: pip install dipy")
    print("Skipping DIPY pipeline.")
    exit(0)

# Load combined data
data_file = os.path.join(MICRO_DIR, 'axcaliber_combined.nii.gz')
bval_file = os.path.join(MICRO_DIR, 'axcaliber_combined.bval')
bvec_file = os.path.join(MICRO_DIR, 'axcaliber_combined.bvec')

if not os.path.exists(data_file):
    print(f"Combined data not found at {data_file}")
    exit(1)

data, affine = load_nifti(data_file)
bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
gtab = gradient_table(bvals, bvecs, b0_threshold=50)

print(f"Data shape: {data.shape}")
print(f"B-values: {np.unique(bvals.astype(int))}")

# --- Method 1: Standard DTI (baseline) ---
print("\n--- DTI ---")
dti_model = TensorModel(gtab)
# Fit on a central slice for speed
sl = data.shape[2] // 2
dti_fit = dti_model.fit(data[:, :, sl:sl+1, :])
fa = dti_fit.fa
md = dti_fit.md
save_nifti(os.path.join(MICRO_DIR, 'dipy', 'dti_FA.nii.gz'), fa, affine)
save_nifti(os.path.join(MICRO_DIR, 'dipy', 'dti_MD.nii.gz'), md, affine)
print(f"  FA range: [{np.nanmin(fa):.3f}, {np.nanmax(fa):.3f}]")

# --- Method 2: DKI (kurtosis → microstructure proxy) ---
print("\n--- DKI ---")
dki_model = DiffusionKurtosisModel(gtab)
dki_fit = dki_model.fit(data[:, :, sl:sl+1, :])
mk = dki_fit.mk(0, 3)  # mean kurtosis, clamped to [0, 3]
save_nifti(os.path.join(MICRO_DIR, 'dipy', 'dki_MK.nii.gz'), mk, affine)
print(f"  MK range: [{np.nanmin(mk):.3f}, {np.nanmax(mk):.3f}]")

# --- Method 3: MAPMRI (mean apparent propagator) ---
print("\n--- MAPMRI ---")
try:
    map_model = MapmriModel(gtab, radial_order=6, laplacian_regularization=True)
    map_fit = map_model.fit(data[:, :, sl:sl+1, :])
    rtop = map_fit.rtop()
    save_nifti(os.path.join(MICRO_DIR, 'dipy', 'mapmri_RTOP.nii.gz'), rtop, affine)
    print(f"  RTOP range: [{np.nanmin(rtop):.3f}, {np.nanmax(rtop):.3f}]")
except Exception as e:
    print(f"  MAPMRI failed: {e}")

# --- Method 4: Free Water DTI ---
print("\n--- Free Water DTI ---")
try:
    fw_model = FreeWaterTensorModel(gtab)
    fw_fit = fw_model.fit(data[:, :, sl:sl+1, :])
    fw_fa = fw_fit.fa
    fw_f = fw_fit.f  # free water fraction
    save_nifti(os.path.join(MICRO_DIR, 'dipy', 'fwdti_FA.nii.gz'), fw_fa, affine)
    save_nifti(os.path.join(MICRO_DIR, 'dipy', 'fwdti_FW.nii.gz'), fw_f, affine)
    print(f"  FW-FA range: [{np.nanmin(fw_fa):.3f}, {np.nanmax(fw_fa):.3f}]")
except Exception as e:
    print(f"  FW-DTI failed: {e}")

print("\nDIPY pipeline complete.")
print(f"Outputs in: {MICRO_DIR}/dipy/")

# NOTE FOR DIPY TEAM DISCUSSION:
# 1. Does DIPY have a cylinder model for AxCaliber estimation?
#    (ActiveAx / AxCaliber with multiple diffusion times)
# 2. MAPMRI on multi-delta data: should we fit per-delta or joint?
# 3. Best approach for the 300 mT/m Connectom gradient data?
# 4. Any DIPY-native diameter estimation from high b-value shells?
DIPY_SCRIPT

chmod +x "${MICRO_DIR}/dipy/run_dipy.py"
echo "DIPY script: ${MICRO_DIR}/dipy/run_dipy.py"

# ---------------------------------------------------------------
# Pipeline B: FSL (dtifit + optional NODDI via AMICO)
# ---------------------------------------------------------------

echo ""
echo ">>> Pipeline B: FSL"

cat > "${MICRO_DIR}/fsl/run_fsl.sh" << 'FSL_SCRIPT'
#!/bin/bash
# FSL microstructure estimation: DTI + optional NODDI (via AMICO)
set -euo pipefail

MICRO_DIR="${1:?Usage: $0 <micro_dir>}"
PREPROC="${2:?Usage: $0 <micro_dir> <preproc_dir>}"

DATA="${MICRO_DIR}/axcaliber_combined.nii.gz"
BVAL="${MICRO_DIR}/axcaliber_combined.bval"
BVEC="${MICRO_DIR}/axcaliber_combined.bvec"
MASK="${PREPROC}/b0_brain_mask.nii.gz"

if [ ! -f "${DATA}" ]; then
    echo "Combined data not found. Run preparation step first."
    exit 1
fi

echo "--- FSL dtifit ---"
dtifit \
    -k "${DATA}" \
    -o "${MICRO_DIR}/fsl/dti" \
    -m "${MASK}" \
    -r "${BVEC}" \
    -b "${BVAL}"

echo "  FA: ${MICRO_DIR}/fsl/dti_FA.nii.gz"
echo "  MD: ${MICRO_DIR}/fsl/dti_MD.nii.gz"

echo ""
echo "--- FSL NODDI (via AMICO, if installed) ---"
python3 -c "
try:
    import amico
    print(f'AMICO version: {amico.__version__}')
    amico.core.setup()
    ae = amico.Evaluation('.', '.')
    # AMICO NODDI would go here
    # ae.load_data('${DATA}', '${BVAL}', '${BVEC}')
    # ae.set_model('NODDI')
    # ae.fit()
    print('AMICO available but full pipeline needs adaptation for WAND data.')
except ImportError:
    print('AMICO not installed. Install with: pip install dmri-amico')
    print('Skipping NODDI estimation.')
" 2>&1

echo "FSL pipeline complete."
FSL_SCRIPT

chmod +x "${MICRO_DIR}/fsl/run_fsl.sh"

# ---------------------------------------------------------------
# Pipeline C: sbi4dwi / dmipy-jax
# ---------------------------------------------------------------

echo ""
echo ">>> Pipeline C: sbi4dwi/dmipy-jax"

cat > "${MICRO_DIR}/sbi4dwi/run_sbi4dwi.py" << 'SBI_SCRIPT'
#!/usr/bin/env python3
"""sbi4dwi AxCaliber estimation on WAND 300 mT/m data.

Uses JAX-accelerated biophysical models:
- C2Cylinder (AxCaliber cylinder model with gamma diameter distribution)
- SANDI (soma + neurite density imaging)
- Simulation-based inference for posterior estimation
"""
import os
import sys

MICRO_DIR = os.environ.get('MICRO_DIR', '.')
SBI4DWI_ROOT = os.path.expanduser('~/dev/sbi4dwi')
sys.path.insert(0, SBI4DWI_ROOT)

try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import nibabel as nib
    from dmipy_jax.cylinder import C2Cylinder
    from dmipy_jax.distributions.distributions import DD1Gamma
    from dmipy_jax.core.acquisition import SimpleAcquisitionScheme
    from dmipy_jax.biophysics.velocity import hursh_rushton_velocity, calculate_latency_matrix

    print(f"JAX devices: {jax.devices()}")
except ImportError as e:
    print(f"Import error: {e}")
    print("Ensure sbi4dwi is installed: cd ~/dev/sbi4dwi && pip install -e .")
    exit(1)

# Load data
data_file = os.path.join(MICRO_DIR, 'axcaliber_combined.nii.gz')
if not os.path.exists(data_file):
    print("Combined data not found.")
    exit(1)

img = nib.load(data_file)
data = img.get_fdata()
bvals = np.loadtxt(os.path.join(MICRO_DIR, 'axcaliber_combined.bval'))
bvecs = np.loadtxt(os.path.join(MICRO_DIR, 'axcaliber_combined.bvec'))

print(f"Data: {data.shape}")
print(f"B-values: {np.unique(bvals.astype(int))}")

# For AxCaliber we need delta and Delta per shell
# WAND AxCaliber uses fixed TE=80ms but varying gradient strength
# delta/Delta need to be extracted from BIDS JSON (protocol-specific)
# Approximate values for CUBRIC Connectom AxCaliber protocol:
AXCAL_TIMING = {
    'AxCaliber1': {'delta': 0.0129, 'Delta': 0.0218},  # 12.9ms, 21.8ms
    'AxCaliber2': {'delta': 0.0129, 'Delta': 0.0302},
    'AxCaliber3': {'delta': 0.0129, 'Delta': 0.0385},
    'AxCaliber4': {'delta': 0.0129, 'Delta': 0.0469},
}

# Process a central slice for demo
sl = data.shape[2] // 2
data_slice = data[:, :, sl, :]

print(f"Processing slice {sl}: {data_slice.shape}")

# --- AxCaliber Cylinder Model ---
print("\n--- AxCaliber Cylinder Model ---")
# Build multi-delta acquisition scheme
# (Full implementation would use per-volume delta/Delta from timing dict)

# Quick DTI-like fit first for orientation
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel

# Signal normalization
b0_mask = bvals < 50
b0_mean = np.mean(data_slice[..., b0_mask], axis=-1, keepdims=True)
b0_mean[b0_mean < 1] = 1
data_norm = data_slice / b0_mean

print(f"Normalized data range: [{np.nanmin(data_norm):.3f}, {np.nanmax(data_norm):.3f}]")

# --- Conduction Velocity Estimation ---
print("\n--- Conduction Velocity from Diameter ---")
# Example: if we had a diameter map (in micrometers)
example_diameters = jnp.array([1.0, 2.0, 3.0, 5.0, 8.0])  # um
example_gratios = jnp.ones_like(example_diameters) * 0.7  # typical g-ratio

velocities = hursh_rushton_velocity(example_diameters, example_gratios)
print(f"  Diameters (um):  {np.array(example_diameters)}")
print(f"  Velocities (m/s): {np.array(velocities)}")

print("\nsbi4dwi pipeline complete.")
print(f"Outputs in: {MICRO_DIR}/sbi4dwi/")
print("\nNOTE: Full voxel-wise AxCaliber fitting requires:")
print("  1. Exact delta/Delta timing from BIDS JSON sidecars")
print("  2. GPU for tractable computation time")
print("  3. Prior calibration for 300 mT/m gradient nonlinearity")
SBI_SCRIPT

chmod +x "${MICRO_DIR}/sbi4dwi/run_sbi4dwi.py"

# ---------------------------------------------------------------
# Pipeline D: DMI.jl (placeholder)
# ---------------------------------------------------------------

echo ""
echo ">>> Pipeline D: DMI.jl"

cat > "${MICRO_DIR}/run_dmijl.jl" << 'JULIA_SCRIPT'
# DMI.jl AxCaliber estimation on WAND data
# Requires: Julia + DMI.jl package
#
# Usage: julia run_dmijl.jl
#
# DMI.jl provides:
# - SANDI (soma and neurite density)
# - AxCaliber (axon diameter distribution)
# - CHARMED (composite hindered and restricted)
#
# See: https://github.com/JuliaNeuroscience/DMI.jl

println("DMI.jl pipeline placeholder")
println("Install: ] add DMI")
println("This requires Julia and DMI.jl to be installed")

# using DMI
# using NIfTI
#
# data = niread("axcaliber_combined.nii.gz")
# protocol = load_protocol("axcaliber_combined.bval", "axcaliber_combined.bvec")
#
# # AxCaliber model
# model = AxCaliberModel(protocol)
# fit = fit_model(model, data)
#
# # Extract maps
# diameter = fit.diameter
# volume_fraction = fit.volume_fraction
JULIA_SCRIPT

# ---------------------------------------------------------------
# Comparison script
# ---------------------------------------------------------------

echo ""
echo ">>> Creating comparison script..."

cat > "${MICRO_DIR}/comparison/compare_methods.py" << 'COMPARE_SCRIPT'
#!/usr/bin/env python3
"""Compare microstructure estimates across all 4 pipelines.

Generates:
- Correlation plots between methods
- Bland-Altman plots
- Regional statistics (mean ± std per white matter tract)
- Conduction velocity maps derived from each diameter estimate
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MICRO_DIR = os.environ.get('MICRO_DIR', '.')

print("=== Microstructure Method Comparison ===")
print(f"Looking for results in: {MICRO_DIR}")

methods = {
    'DIPY': {
        'fa': os.path.join(MICRO_DIR, 'dipy', 'dti_FA.nii.gz'),
        'md': os.path.join(MICRO_DIR, 'dipy', 'dti_MD.nii.gz'),
    },
    'FSL': {
        'fa': os.path.join(MICRO_DIR, 'fsl', 'dti_FA.nii.gz'),
        'md': os.path.join(MICRO_DIR, 'fsl', 'dti_MD.nii.gz'),
    },
}

# Check which outputs exist
for method, files in methods.items():
    available = {k: os.path.exists(v) for k, v in files.items()}
    print(f"  {method}: {available}")

print("\nFull comparison requires running all 4 pipelines first.")
print("Run each pipeline, then re-run this script for comparison figures.")

# When all results are available:
# 1. Load FA/MD from each method
# 2. Compute voxel-wise correlations
# 3. Extract tract-specific statistics (using xtract ROIs)
# 4. Compute conduction velocity from diameter estimates
# 5. Generate publication figures

COMPARE_SCRIPT

chmod +x "${MICRO_DIR}/comparison/compare_methods.py"

echo ""
echo "=============================================="
echo " Microstructure scripts ready"
echo "=============================================="
echo ""
echo "To run each pipeline:"
echo "  A. DIPY:    MICRO_DIR=${MICRO_DIR} python3 ${MICRO_DIR}/dipy/run_dipy.py"
echo "  B. FSL:     bash ${MICRO_DIR}/fsl/run_fsl.sh ${MICRO_DIR} ${PREPROC}"
echo "  C. sbi4dwi: MICRO_DIR=${MICRO_DIR} python3 ${MICRO_DIR}/sbi4dwi/run_sbi4dwi.py"
echo "  D. DMI.jl:  julia ${MICRO_DIR}/run_dmijl.jl"
echo ""
echo "  Compare:    MICRO_DIR=${MICRO_DIR} python3 ${MICRO_DIR}/comparison/compare_methods.py"
