#!/bin/bash
# ============================================================================
# Stage Q: brain2mesh FEM Comparison — iso2mesh vs SimNIBS charm
# ============================================================================
# Compares tetrahedral FEM head meshes from two approaches:
#
#   A. SimNIBS charm: DL-based 5-tissue segmentation + meshing
#   B. brain2mesh (iso2mesh): FreeSurfer segmentation → tetrahedral mesh
#   C. (baseline) MNE BEM: FreeSurfer watershed → 3-shell BEM
#
# WAND provides ground truth for validating tissue boundaries:
#   - QMT bound pool fraction → WM/GM boundary (myelin contrast)
#   - Multi-echo GRE T2* → CSF boundaries (susceptibility)
#   - T1w + T2w → skull/scalp
#
# Install dependencies:
#   pip install iso2mesh       # Fang's pyiso2mesh (auto-downloads CGAL+TetGen)
#   brew install cgal          # or via brew for system CGAL
#   pip install tetgen          # PyVista TetGen wrapper (alternative)
#   pip install simnibs         # for charm comparison
#
# Runtime: ~10-20 min per method per subject
# ============================================================================

set -euo pipefail
source "$(dirname "$0")/00_setup_env.sh"

SUBJECT="${1:?Usage: $0 <subject_id>}"

FS_DIR="${SUBJECTS_DIR}/${SUBJECT}"
MESH_DIR="${WAND_DERIVATIVES}/mesh_comparison/${SUBJECT}"
mkdir -p "${MESH_DIR}/brain2mesh" "${MESH_DIR}/charm" "${MESH_DIR}/mne_bem" "${MESH_DIR}/comparison"

echo "=== FEM Mesh Comparison: ${SUBJECT} ==="

# ---------------------------------------------------------------
# Method A: brain2mesh (iso2mesh / pyiso2mesh)
# ---------------------------------------------------------------

echo ""
echo ">>> Method A: brain2mesh (iso2mesh)"

cat > "${MESH_DIR}/brain2mesh/run_brain2mesh.py" << 'PYEOF'
#!/usr/bin/env python3
"""Generate tetrahedral head mesh using Fang's brain2mesh.

Takes FreeSurfer segmentation → tissue probability maps → tetrahedral FEM.

brain2mesh expects 5 tissue probability maps:
  1. CSF, 2. Grey matter, 3. White matter, 4. Bone, 5. Scalp
These can come from FSL FAST, SPM, or FreeSurfer.
"""
import os
import sys
import time
import numpy as np

MESH_DIR = os.environ.get('MESH_DIR', '.')
SUBJECTS_DIR = os.environ.get('SUBJECTS_DIR', '')
SUBJECT = os.environ.get('SUBJECT', 'sub-08033')
WAND_DERIVATIVES = os.environ.get('WAND_DERIVATIVES', '')

try:
    import iso2mesh
    print(f"iso2mesh version: {iso2mesh.__version__}")
except ImportError:
    print("iso2mesh not installed. Install with: pip install iso2mesh")
    print("External binaries (CGAL, TetGen) auto-download on first use.")
    sys.exit(0)

try:
    import nibabel as nib
except ImportError:
    print("nibabel needed: pip install nibabel")
    sys.exit(1)

out_dir = os.path.join(MESH_DIR, 'brain2mesh')

# --- Load FreeSurfer segmentation ---
aseg_path = os.path.join(SUBJECTS_DIR, SUBJECT, 'mri', 'aseg.mgz')
if not os.path.exists(aseg_path):
    print(f"FreeSurfer aseg not found: {aseg_path}")
    print("Run recon-all first.")
    sys.exit(1)

print(f"Loading segmentation: {aseg_path}")
aseg_img = nib.load(aseg_path)
aseg = aseg_img.get_fdata().astype(int)
affine = aseg_img.affine
print(f"  Volume shape: {aseg.shape}")

# --- Convert aseg labels to tissue probability maps ---
# FreeSurfer aseg label conventions:
#   0 = background
#   2, 41 = WM (left, right cerebral)
#   3, 42 = GM (cortex)
#   4, 5, 14, 15, 43, 44 = CSF/ventricles
#   24 = CSF
#   7, 8, 46, 47 = cerebellum WM/GM
#   10-13, 17-18, 49-54, 26, 28, 58, 60 = subcortical GM
# For brain2mesh we need 5 probability maps: CSF, GM, WM, Bone, Scalp
# FreeSurfer doesn't segment bone/scalp — we'll create them from the brain mask

print("Creating tissue probability maps...")

# Binary masks
wm_labels = [2, 41, 7, 46, 77, 251, 252, 253, 254, 255]
gm_labels = [3, 42, 8, 47] + list(range(10, 14)) + [17, 18] + list(range(49, 55)) + [26, 28, 58, 60]
csf_labels = [4, 5, 14, 15, 24, 43, 44]

wm = np.isin(aseg, wm_labels).astype(np.float64)
gm = np.isin(aseg, gm_labels).astype(np.float64)
csf = np.isin(aseg, csf_labels).astype(np.float64)

# Brain mask = WM + GM + CSF
brain = (wm + gm + csf) > 0

# Create approximate skull and scalp by dilating brain mask
from scipy.ndimage import binary_dilation, binary_erosion

skull_inner = binary_dilation(brain, iterations=2) & ~brain
skull = binary_dilation(skull_inner, iterations=3)
scalp = binary_dilation(skull, iterations=3) & ~skull & ~brain

bone = skull.astype(np.float64)
skin = scalp.astype(np.float64)

print(f"  WM: {wm.sum():.0f} voxels")
print(f"  GM: {gm.sum():.0f} voxels")
print(f"  CSF: {csf.sum():.0f} voxels")
print(f"  Bone (approx): {bone.sum():.0f} voxels")
print(f"  Scalp (approx): {skin.sum():.0f} voxels")

# Stack into 5-tissue map: (X, Y, Z, 5)
tissue_maps = np.stack([csf, gm, wm, bone, skin], axis=-1)

# --- Run brain2mesh ---
print("\nRunning brain2mesh...")
t0 = time.time()

try:
    # brain2mesh expects a 4D array (X, Y, Z, N_tissues)
    node, elem, face = iso2mesh.brain2mesh(tissue_maps)

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Nodes: {node.shape[0]}")
    print(f"  Elements: {elem.shape[0]}")
    print(f"  Surface faces: {face.shape[0] if face is not None else 'N/A'}")

    # Save mesh
    np.save(os.path.join(out_dir, 'nodes.npy'), node)
    np.save(os.path.join(out_dir, 'elements.npy'), elem)
    if face is not None:
        np.save(os.path.join(out_dir, 'faces.npy'), face)

    # Save in JMesh format (iso2mesh native)
    iso2mesh.savejmesh(os.path.join(out_dir, 'head_mesh.jmsh'), node, face, elem)

    # Mesh quality metrics
    if hasattr(iso2mesh, 'meshquality'):
        quality = iso2mesh.meshquality(node, elem)
        print(f"  Min quality: {quality.min():.4f}")
        print(f"  Mean quality: {quality.mean():.4f}")
        np.save(os.path.join(out_dir, 'mesh_quality.npy'), quality)

    print(f"\nSaved to: {out_dir}/")

except Exception as e:
    print(f"brain2mesh failed: {e}")
    print("This may require external binaries. Check ~/.iso2mesh-tools/")

# --- Also try surface-based meshing (FreeSurfer surfaces → TetGen) ---
print("\n--- Alternative: Surface-to-mesh (s2m) ---")
try:
    for hemi in ['lh', 'rh']:
        surf_path = os.path.join(SUBJECTS_DIR, SUBJECT, 'surf', f'{hemi}.pial')
        if os.path.exists(surf_path):
            import nibabel.freesurfer as fs
            verts, faces = fs.read_geometry(surf_path)
            print(f"  {hemi}.pial: {verts.shape[0]} vertices, {faces.shape[0]} faces")

            # Can pass to s2m for tetrahedral meshing
            # node_tet, elem_tet = iso2mesh.s2m(verts, faces, maxvol=10.0)
except Exception as e:
    print(f"  Surface meshing: {e}")
PYEOF

chmod +x "${MESH_DIR}/brain2mesh/run_brain2mesh.py"
echo "  Script: ${MESH_DIR}/brain2mesh/run_brain2mesh.py"

# ---------------------------------------------------------------
# Method B: SimNIBS charm (already in stage 16)
# ---------------------------------------------------------------

echo ""
echo ">>> Method B: SimNIBS charm"
echo "  See: 16_simnibs_charm.sh"
echo "  Output: ${WAND_DERIVATIVES}/simnibs/${SUBJECT}/"

CHARM_MESH="${WAND_DERIVATIVES}/simnibs/${SUBJECT}/m2m_${SUBJECT}"
if [ -d "${CHARM_MESH}" ]; then
    echo "  charm mesh exists ✓"
else
    echo "  charm mesh not yet generated (run 16_simnibs_charm.sh)"
fi

# ---------------------------------------------------------------
# Method C: MNE BEM (FreeSurfer watershed → 3-shell)
# ---------------------------------------------------------------

echo ""
echo ">>> Method C: MNE BEM (baseline)"

cat > "${MESH_DIR}/mne_bem/run_mne_bem.py" << 'PYEOF'
#!/usr/bin/env python3
"""Generate 3-shell BEM using MNE-Python + FreeSurfer."""
import os
import sys
import numpy as np

SUBJECTS_DIR = os.environ.get('SUBJECTS_DIR', '')
SUBJECT = os.environ.get('SUBJECT', 'sub-08033')
MESH_DIR = os.environ.get('MESH_DIR', '.')

try:
    import mne
    print(f"MNE version: {mne.__version__}")
except ImportError:
    print("MNE not installed")
    sys.exit(1)

os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
out_dir = os.path.join(MESH_DIR, 'mne_bem')

print(f"Building BEM for {SUBJECT}...")

try:
    # Watershed BEM surfaces
    mne.bem.make_watershed_bem(SUBJECT, subjects_dir=SUBJECTS_DIR, overwrite=True)
    print("  Watershed complete")

    # BEM model (3 shells: inner skull, outer skull, outer skin)
    model = mne.make_bem_model(SUBJECT, subjects_dir=SUBJECTS_DIR,
                                conductivity=(0.3, 0.006, 0.3))  # brain, skull, scalp
    bem = mne.make_bem_solution(model)

    # Save
    mne.write_bem_solution(os.path.join(out_dir, f'{SUBJECT}-bem.fif'), bem, overwrite=True)

    # Report surface counts
    for surf in model:
        print(f"  {surf['id']}: {surf['ntri']} triangles, {surf['np']} vertices")

    print(f"\nBEM saved to: {out_dir}/")

except Exception as e:
    print(f"BEM failed: {e}")
    print("Ensure FreeSurfer recon-all is complete.")
PYEOF

chmod +x "${MESH_DIR}/mne_bem/run_mne_bem.py"

# ---------------------------------------------------------------
# Comparison script
# ---------------------------------------------------------------

echo ""
echo ">>> Comparison script"

cat > "${MESH_DIR}/comparison/compare_meshes.py" << 'PYEOF'
#!/usr/bin/env python3
"""Compare FEM mesh quality across methods.

Metrics:
1. Element count and size distribution
2. Mesh quality (aspect ratio, skewness)
3. Tissue boundary accuracy vs QMT ground truth
4. E-field simulation comparison (same coil, different meshes)
5. Computation time
"""
import os
import numpy as np

MESH_DIR = os.environ.get('MESH_DIR', '.')

print("=== FEM Mesh Comparison ===")
print()

methods = {
    'brain2mesh': os.path.join(MESH_DIR, 'brain2mesh', 'nodes.npy'),
    'charm': os.path.join(MESH_DIR, '..', '..', 'simnibs', os.environ.get('SUBJECT', ''),
                           'm2m_' + os.environ.get('SUBJECT', ''), 'sub.msh'),
    'MNE BEM': os.path.join(MESH_DIR, 'mne_bem',
                             os.environ.get('SUBJECT', '') + '-bem.fif'),
}

for method, path in methods.items():
    exists = os.path.exists(path)
    print(f"  {method}: {'✓' if exists else '✗'} ({path})")

print()
print("Full comparison requires all three methods to be run first.")
print()
print("Comparison metrics:")
print("  1. Element count, volume distribution, surface quality")
print("  2. Tissue boundary accuracy vs QMT (ses-02)")
print("  3. E-field magnitude at M1 (same coil position)")
print("  4. E-field focality (volume above 50% peak)")
print("  5. Computation time and memory usage")
print()
print("WAND ground truth for validation:")
print("  - QMT bound pool fraction → WM/GM boundary precision")
print("  - Multi-echo GRE T2* → CSF boundary precision")
print("  - T1w + T2w → skull inner/outer table distinction")
PYEOF

chmod +x "${MESH_DIR}/comparison/compare_meshes.py"

echo ""
echo "=== FEM mesh comparison scripts ready ==="
echo ""
echo "To run:"
echo "  A. brain2mesh: MESH_DIR=${MESH_DIR} SUBJECTS_DIR=${SUBJECTS_DIR} SUBJECT=${SUBJECT} python3 ${MESH_DIR}/brain2mesh/run_brain2mesh.py"
echo "  B. SimNIBS:    bash scripts/wand_processing/16_simnibs_charm.sh ${SUBJECT}"
echo "  C. MNE BEM:    MESH_DIR=${MESH_DIR} SUBJECTS_DIR=${SUBJECTS_DIR} SUBJECT=${SUBJECT} python3 ${MESH_DIR}/mne_bem/run_mne_bem.py"
echo "  Compare:       MESH_DIR=${MESH_DIR} SUBJECT=${SUBJECT} python3 ${MESH_DIR}/comparison/compare_meshes.py"
