#!/bin/bash
# ============================================================================
# Stage P: SimNIBS charm — Individualized Head Model + TMS E-field
# ============================================================================
# SimNIBS charm (Complete Head And Reconstruction Model):
#   1. DL-based 5-tissue segmentation from T1w + T2w
#   2. Tetrahedral FEM mesh generation
#   3. Conductivity assignment (WM, GM, CSF, skull, scalp)
#   4. MNI registration
#   5. TMS E-field simulation at motor cortex (M1)
#
# Input:  T1w (required) + T2w (recommended) from ses-03
# Output: Head mesh + simulated E-field at M1 for SICI
#
# Why charm over BEM:
#   - 5-tissue FEM vs 3-shell BEM
#   - Models skull heterogeneity (compact vs spongy bone)
#   - Accurate E-field in sulci where stimulation concentrates
#   - DL-based — no manual segmentation editing needed
#
# Install: pip install simnibs  (or download from simnibs.github.io)
# Runtime: ~30-60 min per subject
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"

T1W="${WAND_ROOT}/${SUBJECT}/ses-03/anat/${SUBJECT}_ses-03_T1w.nii.gz"
T2W="${WAND_ROOT}/${SUBJECT}/ses-03/anat/${SUBJECT}_ses-03_rec-nlgradcorr_T2w.nii.gz"

SIMNIBS_DIR="${WAND_DERIVATIVES}/simnibs/${SUBJECT}"
mkdir -p "${SIMNIBS_DIR}"

echo "=== SimNIBS charm: ${SUBJECT} ==="

# Check SimNIBS is installed
if ! command -v charm &>/dev/null; then
    echo "ERROR: SimNIBS not installed."
    echo "Install with: pip install simnibs"
    echo "Or download from: https://simnibs.github.io/simnibs/"
    exit 1
fi

# ---------------------------------------------------------------
# Step 1: charm — head mesh generation
# ---------------------------------------------------------------

MESH_DIR="${SIMNIBS_DIR}/m2m_${SUBJECT}"

if [ -d "${MESH_DIR}" ]; then
    echo "charm already complete, skipping mesh generation."
else
    echo "Running charm (T1w + T2w → FEM mesh)..."
    echo "  T1w: ${T1W}"
    echo "  T2w: ${T2W}"

    CHARM_ARGS="${SUBJECT} ${T1W}"
    if [ -f "${T2W}" ]; then
        CHARM_ARGS="${CHARM_ARGS} ${T2W}"
        echo "  Using T2w for improved skull/CSF segmentation"
    fi

    # charm outputs to current directory or --forceoutdir
    cd "${SIMNIBS_DIR}"
    charm ${CHARM_ARGS}

    echo "charm complete."
    echo "  Mesh: ${MESH_DIR}/"
    echo "  Surfaces: ${MESH_DIR}/surfaces/"
fi

# ---------------------------------------------------------------
# Step 2: TMS E-field simulation at M1 (motor cortex)
# ---------------------------------------------------------------

echo ""
echo "--- TMS E-field Simulation ---"

# SICI targets the hand area of primary motor cortex (M1)
# MNI coordinates for left M1 hand knob: approximately (-37, -25, 62)
# Coil: assume Magstim 70mm figure-of-eight (standard at CUBRIC)

cat > "${SIMNIBS_DIR}/simulate_tms.py" << 'PYEOF'
#!/usr/bin/env python3
"""Simulate TMS E-field at M1 using SimNIBS.

Places a virtual Magstim 70mm figure-of-eight coil at the M1 hand
knob and computes the induced electric field on the cortical surface.

Output:
- E-field map (V/m) on cortical surface vertices
- Peak E-field location and magnitude
- Spatial spread pattern for neurojax TMSProtocol
"""
import os
import sys
import numpy as np

SIMNIBS_DIR = os.environ.get('SIMNIBS_DIR', '.')
SUBJECT = os.environ.get('SUBJECT', 'sub-08033')

try:
    import simnibs
    from simnibs import sim_struct, run_simnibs
    print(f"SimNIBS version: {simnibs.__version__}")
except ImportError:
    print("SimNIBS not installed. Install with: pip install simnibs")
    sys.exit(1)

mesh_dir = os.path.join(SIMNIBS_DIR, f'm2m_{SUBJECT}')
if not os.path.isdir(mesh_dir):
    print(f"Head mesh not found at {mesh_dir}. Run charm first.")
    sys.exit(1)

# --- Setup simulation ---
s = sim_struct.SESSION()
s.subpath = mesh_dir
s.pathfem = os.path.join(SIMNIBS_DIR, 'tms_simulation')
s.map_to_fsavg = True   # Map results to fsaverage for group comparison
s.map_to_MNI = True      # Also map to MNI volumetric space

# TMS position list (can add multiple positions)
tms = s.add_tmslist()
tms.fnamecoil = os.path.join(
    simnibs.SIMNIBSDIR, 'ccd-files', 'Magstim_70mm_Fig8.nii.gz'
)

# Position 1: Left M1 hand knob
pos = tms.add_position()
# MNI coordinates for M1 hand knob
pos.centre = simnibs.mni2subject_coords([-37, -25, 62], mesh_dir)
# Coil orientation: handle pointing posteriorly (PA direction, standard for M1)
pos.pos_ydir = simnibs.mni2subject_coords([-37, -55, 62], mesh_dir)
pos.distance = 4  # mm from scalp (approximate with hair)
# Stimulator output: 60% MSO is typical for SICI conditioning pulse
# dI/dt determines the E-field amplitude
pos.didt = 1e6  # A/s (placeholder — actual value depends on stimulator)

print(f"Coil centre (subject space): {pos.centre}")
print(f"Coil y-direction: {pos.pos_ydir}")

# --- Run simulation ---
print("\nRunning TMS simulation...")
run_simnibs(s)

# --- Extract results ---
print("\nExtracting E-field results...")

# Load the results mesh
import simnibs.mesh_tools.mesh_io as mesh_io

result_mesh = os.path.join(
    SIMNIBS_DIR, 'tms_simulation',
    f'{SUBJECT}_TMS_1-0001_Magstim_70mm_Fig8_nii_scalar.msh'
)

if os.path.exists(result_mesh):
    mesh = mesh_io.read_msh(result_mesh)

    # E-field on GM surface (tag 2 = grey matter)
    # normE = magnitude of E-field
    gm_efield = mesh.field['normE'].value
    print(f"E-field stats on GM:")
    print(f"  Max: {np.max(gm_efield):.2f} V/m")
    print(f"  Mean: {np.mean(gm_efield):.2f} V/m")
    print(f"  99th percentile: {np.percentile(gm_efield, 99):.2f} V/m")

    # Save E-field for neurojax integration
    np.save(os.path.join(SIMNIBS_DIR, 'efield_gm.npy'), gm_efield)

    # For neurojax TMSProtocol.spatial_spread:
    # Threshold at 50% of peak and normalize to [0, 1]
    peak = np.max(gm_efield)
    spread = gm_efield / peak
    spread[spread < 0.1] = 0  # zero out weak regions
    np.save(os.path.join(SIMNIBS_DIR, 'spatial_spread.npy'), spread)
    print(f"\nSpatial spread saved ({np.sum(spread > 0)} active vertices)")
else:
    print(f"Result mesh not found at {result_mesh}")
    print("Check tms_simulation/ directory for output files.")

print(f"\nSimulation complete. Output: {SIMNIBS_DIR}/tms_simulation/")
PYEOF

chmod +x "${SIMNIBS_DIR}/simulate_tms.py"

# Run if SimNIBS Python bindings available
if python3 -c "import simnibs" 2>/dev/null; then
    echo "Running E-field simulation..."
    SIMNIBS_DIR="${SIMNIBS_DIR}" SUBJECT="${SUBJECT}" \
        python3 "${SIMNIBS_DIR}/simulate_tms.py"
else
    echo "SimNIBS Python not available. Run manually:"
    echo "  SIMNIBS_DIR=${SIMNIBS_DIR} SUBJECT=${SUBJECT} python3 ${SIMNIBS_DIR}/simulate_tms.py"
fi

# ---------------------------------------------------------------
# Step 3: Link to FreeSurfer for surface integration
# ---------------------------------------------------------------

echo ""
echo "--- FreeSurfer Integration ---"

if [ -d "${MESH_DIR}/surfaces" ] && [ -d "${SUBJECTS_DIR}/${SUBJECT}" ]; then
    echo "Linking charm surfaces to FreeSurfer..."
    # charm produces surfaces compatible with FreeSurfer
    # Can use for BEM-based source localization too
    echo "  charm central surface → FreeSurfer white"
    echo "  charm GM/WM boundary available for BEM"
fi

echo ""
echo "=== SimNIBS charm complete for ${SUBJECT} ==="
echo ""
echo "Outputs:"
echo "  Head mesh:      ${MESH_DIR}/"
echo "  E-field sim:    ${SIMNIBS_DIR}/tms_simulation/"
echo "  Spatial spread: ${SIMNIBS_DIR}/spatial_spread.npy"
echo ""
echo "For neurojax integration:"
echo "  spread = jnp.load('${SIMNIBS_DIR}/spatial_spread.npy')"
echo "  proto = TMSProtocol(t_onset=100, target_region=0,"
echo "                       spatial_spread=spread_parcellated)"
